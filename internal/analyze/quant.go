package analyze

import (
	"math"
	"math/rand"
	"sort"

	"github.com/sillydong/modelview/internal/decode"
	"github.com/sillydong/modelview/internal/model"
)

// quantTargets 是模拟的目标档位，按压缩程度从轻到重。
//
// 三档并排是为了让用户直接看出「压到哪一档才划算」——
// 只有一档的话没法判断多压两级的代价有多大。
// 顺序是契约：测试与输出都按这个顺序。
// 只存 Dtype：档位名恒等于 string(dtype)，写两遍迟早漂移
// （加一档要改四处，漏改不会有编译错误）。
var quantTargets = []model.Dtype{
	model.DtypeQ8_0,
	model.DtypeQ6K,
	model.DtypeQ4K,
}

// simulateAll 对一批浮点值做三档量化模拟。
//
// **走各格式真实的编码器**：编码 → 解码 → 与原值比对。
// 不是「块内最大绝对值除以级数」那种统一公式 —— 那套东西的结果取决于
// 「级数怎么定、块多大」这两个任意选择，实测同一批真实 F16 权重上
// 光是级数从 31 改成 63 就摆动 6.26 dB，比它自身的误差还大。
// 复现：go run tools/compare_uniform_formula.go
//
// srcBits 是源类型的每权重位宽，用来算压缩比（F16 是 16、F32 是 32）。
//
// 元素数不是块大小的整数倍时，末尾补零凑满一个块再编码。
//
// **补零的方向不固定，不要假设它**。第一版注释写的是"零会把 scale 拉低、
// 所以误差偏小"，那是错的：scale 是块内极值除以级数，**加零不可能降低极值**。
// 补零块里参与极值的真实值更少，可能偏小也可能偏大 ——
// 实测同样一批真实值、只改尾部长度与填充内容，21 组对照里
// 偏小 9 次、偏大 5 次、完全相同 7 次。
//
// 补进去的零不计入误差统计。权重张量的元素数在实际模型里
// 基本都是块大小的整数倍，这条路径很少走到。
func simulateAll(vals []float32, srcBits float64) []model.QuantSim {
	out := make([]model.QuantSim, 0, len(quantTargets))
	for _, d := range quantTargets {
		out = append(out, simulateTarget(vals, d, srcBits))
	}
	return out
}

// simulateTarget 模拟压到一种格式，返回误差统计。
func simulateTarget(vals []float32, d model.Dtype, srcBits float64) model.QuantSim {
	s := model.QuantSim{Target: string(d), BitsPerWeight: d.BitsPerWeight()}
	if bw := d.BitsPerWeight(); bw > 0 && srcBits > 0 {
		s.Compression = srcBits / bw
	}
	if len(vals) == 0 {
		return s
	}

	elems := d.BlockElems()
	perBlock, ok := d.BlockBytes()
	if !ok || elems <= 0 {
		return s
	}

	// 补零凑整块
	padded := vals
	if rem := int64(len(vals)) % elems; rem != 0 {
		padded = make([]float32, int64(len(vals))+elems-rem)
		copy(padded, vals)
	}
	nBlocks := int64(len(padded)) / elems

	buf := make([]byte, nBlocks*perBlock)
	if err := decode.Quantize(d, padded, buf); err != nil {
		return s
	}
	back := make([]float32, len(padded))
	if err := decode.Decode(d, buf, back); err != nil {
		return s
	}

	var sumAbs, sumSq, sumErrSq float64
	for i, v := range vals { // 只统计原值那部分，不含补的零
		x := float64(v)
		e := math.Abs(x - float64(back[i]))
		sumAbs += e
		sumSq += x * x
		sumErrSq += e * e
		s.MaxAbsErr = math.Max(s.MaxAbsErr, e)
		// 相对误差的分母接近 0 时会变成 Inf —— 权重里有大量接近 0 的值，
		// 只在原值够大时才计入
		if denom := math.Abs(x); denom > 1e-20 {
			s.MaxRelErr = math.Max(s.MaxRelErr, e/denom)
		}
	}
	s.MeanAbsErr = sumAbs / float64(len(vals))

	// 信噪比：信号功率 / 噪声功率。全零输入时两者都是 0，
	// 记 0 而不是 NaN 或 -Inf ——「没有信号也没有噪声」与「算不出来」是两回事。
	if sumErrSq > 0 && sumSq > 0 {
		s.SNRDB = 10 * math.Log10(sumSq/sumErrSq)
	}
	return s
}

// quantMedianBudget 是算中位数时最多保留的 |scale| 样本数（4 字节/个 = 8 MB）。
//
// 精确中位数必须看到全部值，而 3.11 亿权重的 Q6_K 有 1944 万个子块 ——
// 全留下是 78 MB，且随张量大小无上限增长。分布摘要用抽样足够，
// 超预算时置 ScaleMedianSampled。
//
// 其余统计量（min/max/均值/零计数/最扁）**都是全量精确的**，
// 它们才是"被压得最狠"这个结论的依据，不能抽样。
const quantMedianBudget = 1 << 21

// quantSampleSeed 让蓄水池抽样可复现：同一个文件跑两次得到同一个中位数。
//
// **不能用等距抽样代替**：步长会和数据的周期混叠。实测构造 90% 小值 +
// 每 10 个一个大值的序列，步长 10 的等距抽样正好**只抽到大值**，
// 中位数从 1 变成 4。块内子块位置、每层的张量切分都可能带这种周期。
const quantSampleSeed = 0x6d6f64656c766965

// quantAgg 累积一个已量化张量的块级信息。
//
// **边读边算，不保留全部子块**。早先的实现先把全部 SubScale 收进一个切片
// 再统计，实测 3.11 亿权重的 Q6_K 要 311 MB，且 --sample-limit 这条阀门
// 对它完全无效（--sample-limit 1 时峰值 806 MB，改动前只有 41 MB）。
type quantAgg struct {
	d          model.Dtype
	perBlock   int64 // 一个块含多少个子块
	blockElems int

	rng  *rand.Rand
	seen int64

	nBlocks, subBlocks int64
	minS, maxS, sum    float64
	zeroCount          int64

	// 当前块已读到的 |scale|。块内归一要求先看完整块才知道分母，
	// 所以缓冲一个块（最多 16 个 float32）
	curBlock []float32

	flatRatio float64
	flatIndex int64
	flatScale float64

	medians       []float32
	medianSampled bool
}

func newQuantAgg(d model.Dtype, totalSubs int64) *quantAgg {
	a := &quantAgg{
		d:         d,
		rng:       rand.New(rand.NewSource(quantSampleSeed)),
		flatRatio: math.Inf(1),
		// minS/maxS 必须从 ±Inf 起 —— 用零值的话 0 会被当成一个真实观测到的
		// 最小值，于是每个张量的 ScaleMin 都是 0
		minS: math.Inf(1),
		maxS: math.Inf(-1),
		// blockElems 是**子块**覆盖的权重数（Q6_K 是 16、Q4_K 是 32），
		// 不是块元素数（256）。它只能从 Scales 的返回值学到，
		// 不能拿 d.BlockElems() 顶替 —— 那会让 perBlock 恒为 1
	}
	if totalSubs > quantMedianBudget {
		a.medianSampled = true
	}
	return a
}

// add 处理一段连续子块。调用方保证这段正好覆盖若干个完整的块。
func (a *quantAgg) add(subs []decode.SubScale) {
	if len(subs) == 0 {
		return
	}
	if a.blockElems == 0 {
		a.blockElems = subs[0].Elems
	}
	if a.perBlock == 0 && a.blockElems > 0 {
		a.perBlock = max(a.d.BlockElems()/int64(a.blockElems), 1)
		a.curBlock = make([]float32, 0, a.perBlock)
	}

	for _, s := range subs {
		v := math.Abs(float64(s.Scale))
		a.subBlocks++
		a.sum += v
		a.minS, a.maxS = math.Min(a.minS, v), math.Max(a.maxS, v)
		if v == 0 {
			a.zeroCount++
		}
		a.reservoir(float32(v))

		a.curBlock = append(a.curBlock, float32(v))
		if len(a.curBlock) == int(a.perBlock) {
			a.flushBlock()
		}
	}
}

// reservoir 用蓄水池抽样维持一份不超过预算的中位数样本。
//
// Algorithm R：前 budget 个全收，之后第 i 个以 budget/(i+1) 的概率
// 替换池里随机一个位置。样本对**顺序**无偏，因此不会像等距抽样那样
// 与周期性数据混叠。种子固定，同一个文件每次跑得到同一个中位数。
func (a *quantAgg) reservoir(v float32) {
	i := a.seen
	a.seen++
	if len(a.medians) < quantMedianBudget {
		a.medians = append(a.medians, v)
		return
	}
	if j := a.rng.Int63n(i + 1); j < int64(len(a.medians)) {
		a.medians[j] = v
	}
}

// flushBlock 处理一个攒满的块：块内归一后更新"最扁"的候选。
func (a *quantAgg) flushBlock() {
	a.nBlocks++
	blkMax := 0.0
	for _, v := range a.curBlock {
		blkMax = math.Max(blkMax, float64(v))
	}
	for i, v := range a.curBlock {
		// 整块 scale 全 0 时比值记 0（完全压平），而不是 1 ——
		// 记 1 的话最该被点名的块反而不会入选
		ratio := 0.0
		if blkMax > 0 {
			ratio = float64(v) / blkMax
		}
		// 严格小于：平局时保留先遇到的，结果稳定
		if ratio < a.flatRatio {
			a.flatRatio = ratio
			a.flatIndex = a.subBlocks - int64(len(a.curBlock)) + int64(i)
			a.flatScale = float64(v)
		}
	}
	a.curBlock = a.curBlock[:0]
}

// finish 产出诊断结果。允许最后留一个没攒满的块 ——
// 张量的元素数理论上一定是块大小的整数倍，但真出现不满的块时
// 不能把已经读到的子块丢掉。
func (a *quantAgg) finish() model.QuantInfo {
	if len(a.curBlock) > 0 {
		a.flushBlock()
	}
	q := model.QuantInfo{
		Scheme:          string(a.d),
		BitsPerWeight:   a.d.BitsPerWeight(),
		Blocks:          a.nBlocks,
		SubBlocks:       a.subBlocks,
		ScaleMin:        a.minS,
		ScaleMax:        a.maxS,
		ZeroScaleBlocks: a.zeroCount,
		FlattestRatio:   a.flatRatio,
		FlattestScale:   a.flatScale,
		FlattestIndex:   a.flatIndex,
	}
	if a.subBlocks == 0 {
		// 空输入：min/max 还是 ±Inf，中位数没有
		q.ScaleMin, q.ScaleMax = 0, 0
		q.FlattestRatio = 0
		return q
	}
	q.BlockElems = a.blockElems
	q.ScaleMean = a.sum / float64(a.subBlocks)
	q.ScaleMedian, q.ScaleMedianSampled = medianOf(a.medians), a.medianSampled
	return q
}

// medianOf 返回已排序样本的中位数（偶数个时取中间两个的平均）。
func medianOf(vals []float32) float64 {
	if len(vals) == 0 {
		return 0
	}
	sort.Slice(vals, func(i, j int) bool { return vals[i] < vals[j] })
	n := len(vals)
	if n%2 == 1 {
		return float64(vals[n/2])
	}
	return (float64(vals[n/2-1]) + float64(vals[n/2])) / 2
}

// quantDiag 汇总一批已经拿到的子块。流式路径用 quantAgg 直接累积，
// 这个入口给"手上已经有整段子块"的调用方（测试、以及已有数据的复用）。
func quantDiag(d model.Dtype, subs []decode.SubScale) model.QuantInfo {
	a := newQuantAgg(d, int64(len(subs)))
	a.add(subs)
	return a.finish()
}
