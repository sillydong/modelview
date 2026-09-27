package decode

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/sillydong/modelview/internal/model"
)

// groupMaxEps 与 llama.cpp 的 GROUP_MAX_EPS 一致：低于它视为全零块。
//
// 不设这个分支的话，全零块会走到 1/scale，scale 为 0 时得到 Inf，
// 后面 round(Inf) 再转整数是未定义行为。
const groupMaxEps = 1e-15

// Quantize 把 vals 编码成该格式的块。
//
// **按各格式真实的编码算法**，移植自 llama.cpp 的
// ggml/src/ggml-quants.c（函数名写在各自实现的注释里）。
// 不是"块内最大绝对值除以级数"那种统一公式 —— 那套东西的结果取决于
// "级数怎么定、块多大"这两个任意选择，实测同一批真实 F16 权重上
// 光是级数从 31 改成 63 就摆动 6.26 dB，比它自身的误差还大，
// 拿它做量化决策没有意义。真实编码算法没有这个自由度。
//
// dst 的长度必须是 (len(vals)/BlockElems)*BlockBytes。
func Quantize(d model.Dtype, vals []float32, dst []byte) error {
	perBlock, ok := d.BlockBytes()
	if !ok || !d.IsQuantized() {
		return ErrUnsupported{Dtype: d}
	}
	elems := d.BlockElems()
	if elems <= 0 {
		return ErrUnsupported{Dtype: d}
	}
	if int64(len(vals))%elems != 0 {
		return fmt.Errorf("元素数 %d 不是块大小 %d 的整数倍", len(vals), elems)
	}
	if want := int64(len(vals)) / elems * perBlock; int64(len(dst)) != want {
		return fmt.Errorf("目标缓冲 %d 字节，应为 %d", len(dst), want)
	}

	switch d {
	case model.DtypeQ8_0:
		quantizeQ8_0(vals, dst)
	case model.DtypeQ6K:
		quantizeQ6K(vals, dst)
	case model.DtypeQ4K:
		quantizeQ4K(vals, dst)
	default:
		// 其余类型本工具不做。分两类：
		//   - Q4_0/Q4_1/Q5_0/Q5_1：gguf.quants 里有编码器，**可以**验证，
		//     只是没有需求驱动 —— 三档并排只需要 Q8_0/Q6_K/Q4_K
		//   - Q2_K/Q3_K/Q5_K/Q8_1/Q8_K：参考实现里是 NotImplementedError，
		//     做了没有独立来源可对，按"未验证的不给数字"不做
		return ErrUnsupported{Dtype: d}
	}
	return nil
}

// quantizeQ8_0 移植自 quantize_row_q8_0_ref。
//
// 舍入用 roundf（平局**远离零**）—— Go 里是 math.Round。
// 注意这与 K 系列的 nearest_int（平局**取偶**）不同：
// 两条路径的舍入模式确实不一样，用错不会崩，只会让一部分块对不上。
//
// 另一个容易写错的点：码用的是**未舍入**的 1/d，而写进块头的 d 是 fp16 舍入过的。
// K 系列反过来 —— 它们把 d 读回来再算码。Q8_0 这里照上游，不能统一。
func quantizeQ8_0(vals []float32, dst []byte) {
	const perBlock, elems = 34, 32
	for b := range len(vals) / elems {
		blk := dst[b*perBlock : (b+1)*perBlock]
		x := vals[b*elems : (b+1)*elems]

		var amax float32
		for _, v := range x {
			if a := abs32(v); a > amax {
				amax = a
			}
		}
		d := amax / 127
		var id float32
		// d == 0 只在整块全零时出现（amax 就是块内绝对值的最大值）。
		// 此时 v*id = 0*Inf = NaN，而 int8(NaN) 在 amd64/arm64 上都截断成 0，
		// 与上游 roundf(0.0) 的结果一致 —— 实测把这一句改成 `if true`
		// 编出来的字节完全相同，所以它是**防御性**的，不是承重的。
		if d != 0 {
			id = 1 / d
		}
		binary.LittleEndian.PutUint16(blk, F32ToF16(d))
		for j, v := range x {
			blk[2+j] = byte(int8(math.Round(float64(v * id))))
		}
	}
}

// abs32 是 float32 的绝对值，避免 math.Abs 的 float64 往返。
func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// nearestEven 是 K 系列用的取整：四舍五入、**平局取偶**。
//
// llama.cpp 的 nearest_int 是魔数法（fval + 12582912.0f 再取位），
// 等价于 IEEE 的 round-to-nearest-even。这与 Q8_0 那条路径的 roundf
// （平局远离零）不同 —— 实测用错会让一部分块的字节对不上。
func nearestEven(v float32) int {
	return int(math.RoundToEven(float64(v)))
}

func clampInt(v, lo, hi int) int {
	return min(max(v, lo), hi)
}

// makeQXQuants 移植自 make_qx_quants（rmse_type == 1、qw == nil 的路径）。
//
// 它不是"最大绝对值除以 nmax"就完事：先按 nmax 定一个初值，
// 再在 nmax±0.9 上以 0.1 为步长试 18 个候选（is ∈ [-9,9] 去掉 0），
// 用加权最小二乘挑最优。权重是 x²（rmse_type == 1）。
//
// 返回子块 scale 与 16 个**已加 nmax 偏移**的码（码域 [0, 2*nmax-1]）。
func makeQXQuants(x []float32, nmax int) (float32, [16]int) {
	var L [16]int
	var amax, mx float32
	for _, v := range x {
		if a := abs32(v); a > amax {
			amax, mx = a, v // 记的是**带符号**的那个极值，不是绝对值
		}
	}
	if amax < groupMaxEps {
		return 0, L
	}

	// accumulate 按给定的 iscale 算一遍码与两个加权和。
	accumulate := func(isc float32) (float32, float32, [16]int) {
		var out [16]int
		var sumlx, suml2 float32
		for i, v := range x {
			l := clampInt(nearestEven(isc*v), -nmax, nmax-1)
			out[i] = l + nmax
			w := v * v // rmse_type == 1 的权重
			sumlx += w * v * float32(l)
			suml2 += w * float32(l) * float32(l)
		}
		return sumlx, suml2, out
	}

	sumlx, suml2, cand := accumulate(float32(-nmax) / mx)
	var scale float32
	if suml2 != 0 {
		scale = sumlx / suml2
	}
	L, best := cand, scale*sumlx

	for is := -9; is <= 9; is++ {
		if is == 0 {
			continue
		}
		isc := -(float32(nmax) + 0.1*float32(is)) / mx
		sumlx, suml2, cand := accumulate(isc)
		if suml2 > 0 && sumlx*sumlx > best*suml2 {
			scale = sumlx / suml2
			L, best = cand, scale*sumlx
		}
	}
	return scale, L
}

// quantizeQ6K 移植自 quantize_row_q6_K_ref。
//
// 256 个元素分 16 个子块（每块 16 个），子块**线性排列** ——
// 交织只发生在位打包那一步，不在 scale 的计算里。
func quantizeQ6K(vals []float32, dst []byte) {
	const perBlock, elems = 210, 256
	for b := range len(vals) / elems {
		blk := dst[b*perBlock : (b+1)*perBlock]
		x := vals[b*elems : (b+1)*elems]
		clear(blk)

		var L [256]int
		var scales [16]float32
		var maxScale, maxAbsScale float32
		for ib := range 16 {
			s, l := makeQXQuants(x[ib*16:(ib+1)*16], 32)
			scales[ib] = s
			copy(L[ib*16:(ib+1)*16], l[:])
			// 记的是**带符号**的那个极值 scale —— 实测真实文件里
			// 子 scale 确实有负的，与它无关的写法会在别处露馅
			if a := abs32(s); a > maxAbsScale {
				maxAbsScale, maxScale = a, s
			}
		}
		if maxAbsScale < groupMaxEps {
			// 全零块：d 记 0，其余字节已经 clear 成全 0
			binary.LittleEndian.PutUint16(blk[208:], F32ToF16(0))
			continue
		}

		iscale := float32(-128) / maxScale
		binary.LittleEndian.PutUint16(blk[208:], F32ToF16(1/iscale))
		for ib := range 16 {
			blk[192+ib] = byte(int8(min(127, nearestEven(iscale*scales[ib]))))
		}
		q6kCodes(x, &L, blk)
	}
}

// q6kCodes 在 d 与 16 个子 scale **已经写进 blk** 的前提下，算码并打包。
//
// 从 quantizeQ6K 里单独抽出来，是为了能被真实块验证：真实文件的块里
// 已经带着它自己的 d 与子 scale，把解码出来的值喂回来重算，字节必须完全相同。
// 这条路绕开了 scale 搜索（那一步存在下面说的**符号规范自由度**，
// 重编码未必选中同一个符号），单独钉住「码与打包」这两段。
//
// 符号规范自由度：子块内 `d * sc * (q-32)` 的乘积决定数值，
// 所以 `(d, sc) → (-d, -sc)` 并把码映射成 `64-q` 得到的是同一批数值。
// 重构出来的子块两端绝对值相等（±31×p）时，make_qx_quants 的
// "绝对值更大就换"（严格大于）比较会留下先遇到的那个 —— 原始 float32
// 权重不会正好平局，重编码就可能选到相反的符号。实测 40 个真实 Q6_K 块里
// 有 7 个是这种情况，**解码值完全相同**，只是字节不同。
//
// firstPass 是 makeQXQuants 给出的那批码。它只在 `d * 子scale == 0` 的子块上
// 会被保留（上游就是 `if (!d) continue;`）—— 那种子块的解码结果恒为 0，
// 码取什么都不影响数值，但会影响字节。要与 llama.cpp 逐字节一致就得照做。
// 传 nil 表示零初始化，调用方需自行保证没有这种子块（真实块验证就是这么用的）。
func q6kCodes(x []float32, firstPass *[256]int, blk []byte) {
	var L [256]int
	if firstPass != nil {
		L = *firstPass
	}
	// 从**写进去的 fp16** 读回来用，不是用原来的 1/iscale ——
	// 真实格式存的就是 fp16，这一步的舍入会影响后面的码
	d := F16ToF32(binary.LittleEndian.Uint16(blk[208:]))
	for j := range 16 {
		dd := d * float32(int8(blk[192+j]))
		if dd == 0 {
			continue
		}
		for ii := range 16 {
			L[16*j+ii] = clampInt(nearestEven(x[16*j+ii]/dd), -32, 31) + 32
		}
	}
	packQ6K(&L, blk)
}

// packQ6K 把 256 个 6 位码打进 ql(128) + qh(64)。
// 是 quantize_row_q6_K_ref 末尾那段双重循环的直译，
// 与解码时的解包互为逆运算（解码那份已在 ③a 与 llama.cpp 逐值比对过）。
func packQ6K(L *[256]int, blk []byte) {
	ql, qh := blk[0:128], blk[128:192]
	for j := 0; j < 256; j += 128 {
		for l := range 32 {
			q1 := byte(L[j+l] & 0xF)
			q2 := byte(L[j+l+32] & 0xF)
			q3 := byte(L[j+l+64] & 0xF)
			q4 := byte(L[j+l+96] & 0xF)
			off := (j / 128) * 64
			ql[off+l] = q1 | (q3 << 4)
			ql[off+l+32] = q2 | (q4 << 4)
			qh[(j/128)*32+l] = byte(L[j+l]>>4) | byte(L[j+l+32]>>4)<<2 |
				byte(L[j+l+64]>>4)<<4 | byte(L[j+l+96]>>4)<<6
		}
	}
}

// makeQKX2Quants 移植自 make_qkx2_quants（use_mad == false 的路径）。
//
// 比 Q6_K 的 scale 搜索复杂一倍：scale 与 min 要**同时**求。
// 它在 [rmin, rmin+rdelta*nstep] 上扫 21 步（Q4_K 调用时是 -1 到 1、步长 0.1），
// 每步解一个加权最小二乘，取误差最小的那组。
//
// 注意 mn 在候选胜出时会被更新，**后续迭代用的就是更新后的 mn** ——
// 这不是 bug，上游就是这么写的，照着来才能逐字节一致。
//
// 返回子块 scale、min（已取负，与上游的 *the_min 一致）、32 个码。
func makeQKX2Quants(x, weights []float32, nmax int,
	rmin, rdelta float32, nstep int) (float32, float32, [32]int) {

	var L [32]int
	mn, mx := x[0], x[0]
	sumW, sumX := weights[0], weights[0]*x[0]
	for i := 1; i < len(x); i++ {
		if x[i] < mn {
			mn = x[i]
		}
		if x[i] > mx {
			mx = x[i]
		}
		w := weights[i]
		sumW += w
		sumX += w * x[i]
	}
	if mn > 0 {
		mn = 0
	}
	if mx == mn {
		// 常量块：全部码记 0，min 取 -x[0]
		return 0, -mn, L
	}

	iscale := float32(nmax) / (mx - mn)
	scale := 1 / iscale
	var bestErr float32
	var cand [32]int
	for i := range x {
		l := clampInt(nearestEven(iscale*(x[i]-mn)), 0, nmax)
		cand[i], L[i] = l, l
		diff := scale*float32(l) + mn - x[i]
		// 与上游同序：w*(d*d)，不是 (w*d)*d。实测两者结果相同，
		// 但既然声称"直译"，就不该留这种无谓的差异
		bestErr += weights[i] * (diff * diff)
	}
	if nstep < 1 {
		return scale, -mn, L
	}

	for is := 0; is <= nstep; is++ {
		isc := (rmin + rdelta*float32(is) + float32(nmax)) / (mx - mn)
		var sumL, sumL2, sumXL float32
		for i := range x {
			l := clampInt(nearestEven(isc*(x[i]-mn)), 0, nmax)
			cand[i] = l
			w := weights[i]
			sumL += w * float32(l)
			sumL2 += w * float32(l) * float32(l)
			sumXL += w * float32(l) * x[i]
		}
		D := sumW*sumL2 - sumL*sumL
		if D <= 0 {
			continue
		}
		thisScale := (sumW*sumXL - sumX*sumL) / D
		thisMin := (sumL2*sumX - sumL*sumXL) / D
		if thisMin > 0 {
			thisMin = 0
			thisScale = sumXL / sumL2
		}
		var curErr float32
		for i := range x {
			diff := thisScale*float32(cand[i]) + thisMin - x[i]
			curErr += weights[i] * (diff * diff)
		}
		if curErr < bestErr {
			L = cand
			bestErr = curErr
			scale = thisScale
			mn = thisMin // 上游改的是 min，最后才取负
		}
	}
	return scale, -mn, L
}

// quantizeQ4K 移植自 quantize_row_q4_K_ref。
//
// 与 Q6_K 的三处结构差异：
//   - 子块是 32 个元素（不是 16），每块 8 个
//   - 每个子块是**非对称**的：既要有 scale 也要有 min
//   - 权重不是 x²，而是 av_x + |x|（av_x 是子块内 x² 均值的平方根）
func quantizeQ4K(vals []float32, dst []byte) {
	const perBlock, elems = 144, 256
	for b := range len(vals) / elems {
		blk := dst[b*perBlock : (b+1)*perBlock]
		x := vals[b*elems : (b+1)*elems]
		clear(blk)

		var mins, scales [8]float32
		var weights [32]float32
		// 扣掉 min 之后 scale 恒为正，两个上界都从 0 起
		var maxScale, maxMin float32

		for j := range 8 {
			seg := x[32*j : 32*j+32]
			var sumX2 float32
			for _, v := range seg {
				sumX2 += v * v
			}
			avX := float32(math.Sqrt(float64(sumX2 / 32)))
			for l, v := range seg {
				weights[l] = avX + abs32(v)
			}
			s, m, _ := makeQKX2Quants(seg, weights[:], 15, -1, 0.1, 20)
			scales[j], mins[j] = s, m
			// 第一遍的码这里用不上：q4kCodes 会用**存进去的 scale/min**
			// 重算一遍（上游也是这么做的，所以传 nil）
			if s > maxScale {
				maxScale = s
			}
			if m > maxMin {
				maxMin = m
			}
		}

		var invScale, invMin float32
		if maxScale > 0 {
			invScale = 63 / maxScale
		}
		if maxMin > 0 {
			invMin = 63 / maxMin
		}
		for j := range 8 {
			ls := byte(min(63, nearestEven(invScale*scales[j])))
			lm := byte(min(63, nearestEven(invMin*mins[j])))
			if j < 4 {
				blk[4+j] = ls
				blk[4+j+4] = lm
			} else {
				blk[4+j+4] = (ls & 0xF) | ((lm & 0xF) << 4)
				blk[4+j-4] |= (ls >> 4) << 6
				blk[4+j] |= (lm >> 4) << 6
			}
		}
		binary.LittleEndian.PutUint16(blk, F32ToF16(maxScale/63))
		binary.LittleEndian.PutUint16(blk[2:], F32ToF16(maxMin/63))

		q4kCodes(x, nil, blk)
	}
}

// q4kCodes 在 d/dmin 与 8 组子 scale/min **已经写进 blk** 的前提下，算码并打包。
//
// 与 q6kCodes 同样的结构与理由：真实格式存的是量化后的 scale，
// 必须从 blk 里读回来用，而不是拿算出来的原值。
//
// firstPass 的说明同 q6kCodes：只在 `d * 子scale == 0` 的子块上保留。
func q4kCodes(x []float32, firstPass *[256]int, blk []byte) {
	var L [256]int
	if firstPass != nil {
		L = *firstPass
	}
	d := F16ToF32(binary.LittleEndian.Uint16(blk))
	dmin := F16ToF32(binary.LittleEndian.Uint16(blk[2:]))
	for j := range 8 {
		sc, m := getScaleMinK4(j, blk[4:16])
		dd := d * float32(sc)
		if dd == 0 {
			continue
		}
		dm := dmin * float32(m)
		for ii := range 32 {
			L[32*j+ii] = clampInt(nearestEven((x[32*j+ii]+dm)/dd), 0, 15)
		}
	}
	packQ4K(&L, blk)
}

// packQ4K 把 256 个 4 位码打进 qs(128)。
func packQ4K(L *[256]int, blk []byte) {
	q := blk[16:144]
	for j := 0; j < 256; j += 64 {
		for l := range 32 {
			q[l] = byte(L[j+l] | (L[j+l+32] << 4))
		}
		q = q[32:]
	}
}
