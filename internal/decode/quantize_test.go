package decode

import (
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

// 编码再解码，差值必须落在该格式的量化步长之内。
//
// 这条只验证"大致对"，真正的门禁是后面的逐字节比对。
func TestQuantize_往返误差有界(t *testing.T) {
	vals := make([]float32, 256)
	for i := range vals {
		vals[i] = float32(math.Sin(float64(i)*0.1)) * 0.1
	}
	tests := []struct {
		d      model.Dtype
		maxErr float64
	}{
		{model.DtypeQ8_0, 0.1 / 127},
		{model.DtypeQ6K, 0.1 / 30}, // 6 位但子 scale 会被量化，留余量
		{model.DtypeQ4K, 0.1 / 10}, // 4 位 + 两级 scale
	}
	for _, tt := range tests {
		t.Run(string(tt.d), func(t *testing.T) {
			perBlock, _ := tt.d.BlockBytes()
			n := int64(len(vals)) / tt.d.BlockElems()
			dst := make([]byte, n*perBlock)
			if err := Quantize(tt.d, vals, dst); err != nil {
				t.Fatalf("Quantize 失败: %v", err)
			}
			back := make([]float32, len(vals))
			if err := Decode(tt.d, dst, back); err != nil {
				t.Fatalf("Decode 失败: %v", err)
			}
			for i := range vals {
				if e := math.Abs(float64(vals[i] - back[i])); e > tt.maxErr {
					t.Fatalf("[%d] 误差 %g 超过上界 %g", i, e, tt.maxErr)
				}
			}
		})
	}
}

// 全零块：不能产生 Inf / NaN，且**字节必须全 0**。
//
// 字节这一层不能省：d 记 0 时码取什么值解出来都是 0，
// 只断言"解出的值是 0"的话，把码写坏也发现不了 ——
// 变异验证时去掉 `if d != 0` 的分母保护，这条测试照样绿，才发现断言太弱。
//
// （注意上游对全零块的处理：Q8_0 是 d=0 且 id=0 让码自然归零；
// Q6_K/Q4_K 是先 memset 再写 d=0。两者结果都是整块 0 字节。）
func TestQuantize_全零块(t *testing.T) {
	for _, d := range []model.Dtype{model.DtypeQ8_0, model.DtypeQ6K, model.DtypeQ4K} {
		t.Run(string(d), func(t *testing.T) {
			perBlock, _ := d.BlockBytes()
			n := int64(256) / d.BlockElems()
			dst := make([]byte, n*perBlock)
			if err := Quantize(d, make([]float32, 256), dst); err != nil {
				t.Fatalf("Quantize 失败: %v", err)
			}
			for i, b := range dst {
				if b != 0 {
					t.Fatalf("全零块编出来第 %d 字节是 0x%02X，应为 0 —— "+
						"上游对全零块写的是整块 0", i, b)
				}
			}
			back := make([]float32, 256)
			if err := Decode(d, dst, back); err != nil {
				t.Fatalf("Decode 失败: %v", err)
			}
			for i, v := range back {
				if v != 0 {
					t.Fatalf("[%d] = %v，全零块应解出全零", i, v)
				}
			}
		})
	}
}

// 未收录的类型必须报错。
func TestQuantize_未收录类型报错(t *testing.T) {
	for _, d := range []model.Dtype{model.DtypeF32, model.DtypeQ4_0, model.DtypeIQ2XXS} {
		if err := Quantize(d, make([]float32, 256), make([]byte, 4096)); err == nil {
			t.Errorf("%s 没有编码器，应报错", d)
		}
	}
}

// 元素数与缓冲大小必须匹配，否则报错而不是越界写。
func TestQuantize_尺寸不符报错(t *testing.T) {
	if err := Quantize(model.DtypeQ8_0, make([]float32, 64), make([]byte, 10)); err == nil {
		t.Error("缓冲太小应报错")
	}
	if err := Quantize(model.DtypeQ8_0, make([]float32, 33), make([]byte, 34)); err == nil {
		t.Error("元素数不是块大小整数倍应报错")
	}
}

// K 系列的取整是「平局取偶」，不是「平局远离零」。
//
// 真实块验证**覆盖不到**这一点：真实块的数值是 d*sc*(q-32)，
// 除以 dd 之后恰好是整数，永远落不到平局上。随机数据也几乎不可能
// 恰好落在 .5 —— 变异验证时把 nearestEven 换成 math.Round，
// 真实块那条测试照样全绿，才发现是这个缺口。必须专门构造 ±k.5 的输入。
func TestQuantize_K系列取整_平局取偶(t *testing.T) {
	t.Run("Q6_K", func(t *testing.T) {
		// d = fp16(1.0)，16 个子 scale 全填 1 → 每个子块 dd = 1.0
		blk := make([]byte, 210)
		for i := range 16 {
			blk[192+i] = 1
		}
		binary.LittleEndian.PutUint16(blk[208:], 0x3C00)

		vals := make([]float32, 256)
		for i := range vals {
			vals[i] = float32(i%16) - 8 + 0.5 // -7.5 .. 7.5
		}
		q6kCodes(vals, nil, blk)
		checkTies(t, model.DtypeQ6K, blk, vals)
	})

	t.Run("Q4_K", func(t *testing.T) {
		// d = fp16(1.0)、dmin = fp16(0)，12 个 scale 字节全 1 →
		// 每个子块 dd = 1、dm = 0（见 TestScales_单块/Q4_K 的推导）
		blk := make([]byte, 144)
		binary.LittleEndian.PutUint16(blk, 0x3C00)
		for i := range 12 {
			blk[4+i] = 0x01
		}

		vals := make([]float32, 256)
		for i := range vals {
			// 0.5 .. 14.5。上界取 14.5 而不是 15.5：码域是 [0,15]，
			// 15.5 会被 clamp 掉，那条用例就分不清取整模式了
			vals[i] = float32(i%15) + 0.5
		}
		q4kCodes(vals, nil, blk)
		checkTies(t, model.DtypeQ4K, blk, vals)
	})
}

// checkTies 断言解码结果等于"平局取偶"的舍入值。
//
// 用 math.Round（平局远离零）实现的话，负半数会解成奇数那一侧，这里立刻红。
func checkTies(t *testing.T, d model.Dtype, blk []byte, vals []float32) {
	t.Helper()
	back := make([]float32, len(vals))
	if err := Decode(d, blk, back); err != nil {
		t.Fatalf("Decode 失败: %v", err)
	}
	ties := 0
	for i := range vals {
		want := float32(math.RoundToEven(float64(vals[i])))
		if vals[i] != want {
			ties++
		}
		if back[i] != want {
			t.Fatalf("[%d] 输入 %v 解出 %v，want %v —— 取整用的是平局远离零",
				i, vals[i], back[i], want)
		}
	}
	if ties == 0 {
		t.Fatal("构造的输入里没有平局 —— 这条测试会变成空转")
	}
	t.Logf("%d 个元素中有 %d 个落在舍入平局上", len(vals), ties)
}

// Q8_0 的编码器必须与 llama.cpp 参考实现**逐字节**一致。
//
// 比对的是字节而不是反量化的值：比对值只能验证"编解码这一对自洽"，
// 比对字节才能验证编码器本身与参考实现相同。
func TestQuantize_Q8_0与参考实现逐字节一致(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "quant_sim.json"))
	if err != nil {
		t.Fatalf("读 quant_sim.json 失败（用 uv run --with gguf --with numpy "+
			"python3 tools/verify_gguf_quant_sim.py 生成）: %v", err)
	}
	var v struct {
		Block int `json:"block"`
		Cases []struct {
			Input []float64 `json:"input"`
			Bytes string    `json:"bytes"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("解析: %v", err)
	}
	if len(v.Cases) == 0 {
		t.Fatal("cases 是空的 —— 比对会变成空转")
	}
	if v.Block != int(model.DtypeQ8_0.BlockElems()) {
		t.Fatalf("真值的块元素数 %d，块表 %d", v.Block, model.DtypeQ8_0.BlockElems())
	}

	for ci, c := range v.Cases {
		vals := make([]float32, len(c.Input))
		for i, f := range c.Input {
			vals[i] = float32(f)
		}
		want, err := hex.DecodeString(c.Bytes)
		if err != nil {
			t.Fatalf("第 %d 组真值不是合法 hex: %v", ci, err)
		}
		got := make([]byte, len(want))
		if err := Quantize(model.DtypeQ8_0, vals, got); err != nil {
			t.Fatalf("第 %d 组 Quantize 失败: %v", ci, err)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("第 %d 组第 %d 字节: got 0x%02X want 0x%02X（输入 %d 个值）",
					ci, i, got[i], want[i], len(vals))
			}
		}
	}
}

// realBlocks 读真实块样本。缺文件判失败而不是跳过 ——
// 没有样本，码与打包这两段就完全没有外部守卫。
func realBlocks(t *testing.T, dtype string) []string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "real_blocks.json"))
	if err != nil {
		t.Fatalf("读 testdata/real_blocks.json 失败（用 uv run --with gguf --with numpy "+
			"python3 tools/extract_real_blocks.py 生成）: %v", err)
	}
	var v map[string]json.RawMessage
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("解析 real_blocks.json: %v", err)
	}
	seg, ok := v[dtype]
	if !ok {
		t.Fatalf("样本里没有 %s", dtype)
	}
	var one struct {
		BlockElems int      `json:"block_elems"`
		BlockBytes int      `json:"block_bytes"`
		Blocks     []string `json:"blocks"`
	}
	if err := json.Unmarshal(seg, &one); err != nil {
		t.Fatalf("解析 %s 段: %v", dtype, err)
	}
	if len(one.Blocks) == 0 {
		t.Fatalf("样本里 %s 的块数是 0 —— 测试会变成空转", dtype)
	}
	// 样本自身的结构必须与块表吻合，否则比对基准就是错的
	d := model.Dtype(dtype)
	if got := int(d.BlockElems()); got != one.BlockElems {
		t.Fatalf("%s: 样本块元素数 %d，块表 %d", dtype, one.BlockElems, got)
	}
	if bb, _ := d.BlockBytes(); int(bb) != one.BlockBytes {
		t.Fatalf("%s: 样本块字节数 %d，块表 %d", dtype, one.BlockBytes, bb)
	}
	return one.Blocks
}

// 真实块解码后重新编码，字节必须与原块完全相同。
//
// 这是**码与打包**两段的硬门禁：真实块里已经带着它自己的 d/dmin/子 scale，
// q6kCodes/q4kCodes 直接读回来用，不碰 scale 搜索 ——
// 所以这条断言与"用了什么重要性矩阵"无关，必须 100% 通过。
//
// 它同时也在验解码器：编码器算出的码只有与解码器互逆，才能还原原字节。
func TestQuantize_真实块_码与打包逐字节一致(t *testing.T) {
	for _, dtype := range []string{"Q6_K", "Q4_K"} {
		t.Run(dtype, func(t *testing.T) {
			d := model.Dtype(dtype)
			blocks := realBlocks(t, dtype)
			elems := int(d.BlockElems())

			for bi, h := range blocks {
				blk, err := hex.DecodeString(h)
				if err != nil {
					t.Fatalf("第 %d 块不是合法 hex: %v", bi, err)
				}
				orig := append([]byte(nil), blk...)

				// 真实块里不该有"d×子scale == 0"的子块 —— 有的话
				// q6kCodes/q4kCodes 传 nil（零初始化第一遍的码）就不忠实了
				if hasZeroSubScale(d, blk) {
					t.Fatalf("第 %d 块含 scale 为 0 的子块，本用例的假设不成立", bi)
				}

				// 解码成数值，再喂回编码器
				vals := make([]float32, elems)
				if err := Decode(d, blk, vals); err != nil {
					t.Fatalf("第 %d 块解码失败: %v", bi, err)
				}

				switch d {
				case model.DtypeQ6K:
					q6kCodes(vals, nil, blk)
				case model.DtypeQ4K:
					q4kCodes(vals, nil, blk)
				}

				if n := diffCount(blk, orig); n != 0 {
					t.Fatalf("第 %d 块重编码后有 %d/%d 个字节不同",
						bi, n, len(orig))
				}
			}
			t.Logf("%s: %d 个真实块全部逐字节一致", dtype, len(blocks))
		})
	}
}

// hasZeroSubScale 判断块里有没有"解码贡献恒为 0"的子块。
func hasZeroSubScale(d model.Dtype, blk []byte) bool {
	switch d {
	case model.DtypeQ6K:
		if F16ToF32(binary.LittleEndian.Uint16(blk[208:])) == 0 {
			return true
		}
		for ib := range 16 {
			if int8(blk[192+ib]) == 0 {
				return true
			}
		}
	case model.DtypeQ4K:
		if F16ToF32(binary.LittleEndian.Uint16(blk)) == 0 {
			return true
		}
		for j := range 8 {
			if sc, _ := getScaleMinK4(j, blk[4:16]); sc == 0 {
				return true
			}
		}
	}
	return false
}

// diffCount 数两段字节里不同的位置数（报错时定位用）。
func diffCount(a, b []byte) int {
	n := 0
	for i := range a {
		if a[i] != b[i] {
			n++
		}
	}
	return n
}

// 完整编码器（含 scale 搜索）对真实块的效果。
//
// **不比字节，比解码值。** 原因是 Q6_K/Q4_K 的子块存在一个**符号规范自由度**：
// 子块内 `d * sc * (q-32)` 的乘积决定数值，`(d, sc) → (-d, -sc)` 并同步把码
// 映射成 `64-q` 就得到同一批数值。当子块的两端绝对值相等（±31×p）时，
// make_qx_quants 用"绝对值更大就换"的**严格大于**比较挑极值，
// 平局时留下先遇到的那个 —— 而"重构出来的值"天然是平局，
// 原始 float32 权重则不是。于是重编码可能选中相反的符号。
//
// 实测：字节一致的块 Q6_K 33/40、Q4_K 39/40，但**解码值**完全相同的是
// Q6_K 40/40、Q4_K 39/40。这个差别不是编码器的缺陷，也不是重要性矩阵 ——
// 计划阶段曾把 33/40 归因于"真实文件用了重要性矩阵"，实测证明是错的：
// Q6_K 的真实块 40/40 都能被等权编码器精确复现。
//
// 所以门禁放在解码值上，字节一致率只作观测值打出来。
//
// Q4_K 那 1/40 是真的选出了不同的 scale（不是符号规范），来源无法确定 ——
// 可能是重要性矩阵，也可能上游改过参数，所以门禁留一个块的余量。
func TestQuantize_完整编码器对真实块(t *testing.T) {
	// 门禁卡在实测值上，不留宽松的百分比 —— 早先写 90% 时，
	// 把 makeQKX2Quants 里"候选胜出时更新 min"那一步删掉，
	// 只有 1 个块受影响，测试照样绿。留一个块的余量给 Q4_K 那个已知的真差异。
	// （为什么 Q4_K 能留余量而 Q6_K 不能：Q6_K 实测 40/40，没有已知的真差异。）
	tests := []struct {
		dtype   string
		minSame int
	}{
		{"Q6_K", 40},
		{"Q4_K", 39},
	}

	for _, tt := range tests {
		t.Run(tt.dtype, func(t *testing.T) {
			d := model.Dtype(tt.dtype)
			blocks := realBlocks(t, tt.dtype)
			elems := int(d.BlockElems())
			perBlock, _ := d.BlockBytes()

			byteSame, valSame := 0, 0
			for bi, h := range blocks {
				blk, err := hex.DecodeString(h)
				if err != nil {
					t.Fatalf("不是合法 hex: %v", err)
				}
				vals := make([]float32, elems)
				if err := Decode(d, blk, vals); err != nil {
					t.Fatalf("解码失败: %v", err)
				}
				out := make([]byte, perBlock)
				if err := Quantize(d, vals, out); err != nil {
					t.Fatalf("Quantize 失败: %v", err)
				}
				if diffCount(out, blk) == 0 {
					byteSame++
				}
				back := make([]float32, elems)
				if err := Decode(d, out, back); err != nil {
					t.Fatalf("重解码失败: %v", err)
				}
				same := true
				for i := range vals {
					if vals[i] != back[i] {
						same = false
						break
					}
				}
				if same {
					valSame++
				} else {
					t.Logf("第 %d 块解码值不同", bi)
				}
			}
			byteRate := float64(byteSame) / float64(len(blocks))
			valRate := float64(valSame) / float64(len(blocks))
			t.Logf("%s: 字节一致 %d/%d（%.0f%%），解码值相同 %d/%d（%.0f%%）",
				tt.dtype, byteSame, len(blocks), byteRate*100,
				valSame, len(blocks), valRate*100)
			if valSame < tt.minSame {
				t.Errorf("%s 只有 %d/%d 个真实块的解码值能被复现（下限 %d）—— "+
					"scale 搜索那一段可能改坏了",
					tt.dtype, valSame, len(blocks), tt.minSame)
			}
		})
	}
}
