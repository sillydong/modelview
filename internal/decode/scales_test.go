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

// le16 是 fp16 位模式的小端字节。
func le16(bits uint16) []byte {
	return []byte{byte(bits), byte(bits >> 8)}
}

// times 是 n 个重复字节。
func times(b byte, n int) []byte {
	out := make([]byte, n)
	for i := range out {
		out[i] = b
	}
	return out
}

// cat 把若干字节段拼起来。
func cat(parts ...[]byte) []byte {
	var out []byte
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// 每种类型的一个块，字段填成**互不相同的非零值**，检查提取结果。
//
// 夹具必须让"字段位置"可分辨：早先这组用例把非目标字段全填 0，
// 结果把 Q6_K 的 d 从偏移 208 挪到 0 也照样通过（两边都是 0）——
// 变异验证试出来的。所以现在每个块的非语义区域填 0x40 之类的花样，
// 读错位置必然得到一个不同的值。
func TestScales_单块(t *testing.T) {
	// fp16：0.0625 = 0x2C00，0.5 = 0x3800
	const dHalf, mHalf = 0x2C00, 0x3800
	const filler = 0x40

	tests := []struct {
		name string
		d    model.Dtype
		blk  []byte
		want []SubScale
	}{
		{
			// d(2) + qs(32)
			name: "Q8_0 单子块",
			d:    model.DtypeQ8_0,
			blk:  cat(le16(dHalf), times(filler, 32)),
			want: []SubScale{{Scale: 0.0625, Elems: 32}},
		},
		{
			// d(2) + qs(16)，对称量化
			name: "Q4_0 单子块",
			d:    model.DtypeQ4_0,
			blk:  cat(le16(dHalf), times(filler, 16)),
			want: []SubScale{{Scale: 0.0625, Elems: 32}},
		},
		{
			// d(2) + qh(4) + qs(16)，对称量化
			name: "Q5_0 单子块",
			d:    model.DtypeQ5_0,
			blk:  cat(le16(dHalf), times(filler, 4), times(filler, 16)),
			want: []SubScale{{Scale: 0.0625, Elems: 32}},
		},
		{
			// d(2) + m(2) + qs(16)
			name: "Q4_1 带 min",
			d:    model.DtypeQ4_1,
			blk:  cat(le16(dHalf), le16(mHalf), times(filler, 16)),
			want: []SubScale{{Scale: 0.0625, Min: 0.5, Elems: 32}},
		},
		{
			// d(2) + m(2) + qh(4) + qs(16)
			name: "Q5_1 带 min",
			d:    model.DtypeQ5_1,
			blk:  cat(le16(dHalf), le16(mHalf), times(filler, 4), times(filler, 16)),
			want: []SubScale{{Scale: 0.0625, Min: 0.5, Elems: 32}},
		},
		{
			// d(2) + dmin(2) + scales(12) + qs(128)。
			// 12 个 scale 字节全填 0x01：j<4 时 sc 取低 6 位、m 取后一字节的低 6 位，
			// 都是 1；j>=4 时低位从 q[j+4] 取（=1）、高 2 位从 q[j-4] 取（0x01>>6=0），
			// 而 m 的高 2 位来自 q[j]（同样为 0）—— 所以后 4 个子块 Min 为 0。
			name: "Q4_K 八子块",
			d:    model.DtypeQ4K,
			blk:  cat(le16(dHalf), le16(mHalf), times(0x01, 12), times(filler, 128)),
			want: []SubScale{
				{Scale: 0.0625, Min: 0.5, Elems: 32}, {Scale: 0.0625, Min: 0.5, Elems: 32},
				{Scale: 0.0625, Min: 0.5, Elems: 32}, {Scale: 0.0625, Min: 0.5, Elems: 32},
				{Scale: 0.0625, Min: 0, Elems: 32}, {Scale: 0.0625, Min: 0, Elems: 32},
				{Scale: 0.0625, Min: 0, Elems: 32}, {Scale: 0.0625, Min: 0, Elems: 32},
			},
		},
		{
			// d(2) + dmin(2) + scales(12) + qh(32) + qs(128) —— scale 布局与 Q4_K 相同
			name: "Q5_K 八子块",
			d:    model.DtypeQ5K,
			blk:  cat(le16(dHalf), le16(mHalf), times(0x01, 12), times(filler, 32), times(filler, 128)),
			want: []SubScale{
				{Scale: 0.0625, Min: 0.5, Elems: 32}, {Scale: 0.0625, Min: 0.5, Elems: 32},
				{Scale: 0.0625, Min: 0.5, Elems: 32}, {Scale: 0.0625, Min: 0.5, Elems: 32},
				{Scale: 0.0625, Min: 0, Elems: 32}, {Scale: 0.0625, Min: 0, Elems: 32},
				{Scale: 0.0625, Min: 0, Elems: 32}, {Scale: 0.0625, Min: 0, Elems: 32},
			},
		},
		{
			// ql(128) + qh(64) + scales(16 个 int8) + d(2)。
			// **d 在块尾（208）** —— 前面全填 0x40，所以读错到块头会得到别的值。
			// 子 scale 填 1（int8），Scale = 0.0625 × 1。
			name: "Q6_K 十六子块",
			d:    model.DtypeQ6K,
			blk:  cat(times(filler, 192), times(0x01, 16), le16(dHalf)),
			want: func() []SubScale {
				out := make([]SubScale, 16)
				for i := range out {
					out[i] = SubScale{Scale: 0.0625, Elems: 16}
				}
				return out
			}(),
		},
		{
			// scales(16) + qs(64) + d(2) + dmin(2)。
			// scale 字节 0x21：低 4 位是 sc=1，高 4 位是 m=2。
			name: "Q2_K 十六子块",
			d:    model.DtypeQ2K,
			blk:  cat(times(0x21, 16), times(filler, 64), le16(dHalf), le16(mHalf)),
			want: func() []SubScale {
				out := make([]SubScale, 16)
				for i := range out {
					out[i] = SubScale{Scale: 0.0625, Min: 1, Elems: 16}
				}
				return out
			}(),
		},
		{
			// hmask(32) + qs(64) + scales(12) + d(2)。
			// 12 个 scale 字节全 0 —— 解包后每个都是 0-32 = -32，所以 Scale 是**负数**。
			// 这个用例同时钉住了 d 在块尾（108）这件事。
			name: "Q3_K 十六子块",
			d:    model.DtypeQ3K,
			blk:  cat(times(0, 96), times(0, 12), le16(dHalf)),
			want: func() []SubScale {
				out := make([]SubScale, 16)
				for i := range out {
					out[i] = SubScale{Scale: 0.0625 * -32, Elems: 16}
				}
				return out
			}(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 夹具长度必须与块表一致，否则测的是别的东西
			if bb, _ := tt.d.BlockBytes(); int64(len(tt.blk)) != bb {
				t.Fatalf("夹具长度 %d，块表 %d —— 夹具本身写错了", len(tt.blk), bb)
			}
			got, err := Scales(tt.d, tt.blk)
			if err != nil {
				t.Fatalf("Scales 失败: %v", err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("子块数 = %d, want %d", len(got), len(tt.want))
			}
			// 反向门禁：期望值不能全是"空"的，否则读错位置也能过
			nonzero := 0
			for _, w := range tt.want {
				if w.Scale != 0 || w.Min != 0 {
					nonzero++
				}
			}
			if nonzero == 0 {
				t.Fatal("期望值全为 0 —— 这个用例分辨不出任何偏移错误")
			}

			for i := range got {
				// float32 比较：scale 是 fp16 提升上来的，值本身就是 float32
				if math.Float32bits(got[i].Scale) != math.Float32bits(tt.want[i].Scale) ||
					got[i].Min != tt.want[i].Min || got[i].Elems != tt.want[i].Elems {
					t.Errorf("子块[%d] = %+v, want %+v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

// Q4_K 的 6 位 scale 必须被正确解包 —— 这是最容易写错的地方。
func TestScales_Q4K_六位解包(t *testing.T) {
	blk := make([]byte, 144)
	// d = 1.0, dmin = 1.0
	binary.LittleEndian.PutUint16(blk[0:], 0x3C00)
	binary.LittleEndian.PutUint16(blk[2:], 0x3C00)
	// scales 前 8 字节填 0x3F（低 4 位 15，高 4 位 3），后 4 字节填 0xC0
	// 这会让前 4 个子块的 sc = 15 | (高2位<<4)，m = 3 | (高2位<<4)
	for i := range 8 {
		blk[4+i] = 0x3F
	}
	for i := range 4 {
		blk[12+i] = 0xC0
	}

	got, err := Scales(model.DtypeQ4K, blk)
	if err != nil {
		t.Fatalf("Scales 失败: %v", err)
	}
	if len(got) != 8 {
		t.Fatalf("子块数 = %d, want 8", len(got))
	}
	// 每个子块覆盖 32 个权重
	for i, s := range got {
		if s.Elems != 32 {
			t.Errorf("子块[%d].Elems = %d, want 32", i, s.Elems)
		}
		// scale 与 min 都应落在 6 位能表示的范围里（0..63），乘上 d 后不超过 63
		if s.Scale < 0 || s.Scale > 63 {
			t.Errorf("子块[%d].Scale = %v，超出 6 位范围", i, s.Scale)
		}
		if s.Min < 0 || s.Min > 63 {
			t.Errorf("子块[%d].Min = %v，超出 6 位范围", i, s.Min)
		}
	}
	// 高 2 位必须真的被拼进低 4 位：后 4 个子块的 sc 低位来自 q[j+4]=0xC0 的低 4 位（0），
	// 高 2 位来自 q[j-4]=0x3F 的高 2 位（0）—— 若实现漏掉高 2 位，sc 仍是 0，
	// 所以这条用例必须让高 2 位非零才分辨得出。
	blk2 := make([]byte, 144)
	binary.LittleEndian.PutUint16(blk2[0:], 0x3C00) // d    = 1.0
	binary.LittleEndian.PutUint16(blk2[2:], 0x3C00) // dmin = 1.0（不设的话 Min 恒为 0，m 就观察不到）
	for i := range 4 {
		blk2[4+i] = 0x03 // q[0..3] 低 4 位 = 3
	}
	for i := range 4 {
		blk2[8+i] = 0x01 // q[4..7] 低 4 位 = 1
	}
	// q[8..11]：高 2 位放 2（0b10 << 6 = 0x80），低 4 位放 5
	for i := range 4 {
		blk2[12+i] = 0x85
	}
	got2, err := Scales(model.DtypeQ4K, blk2)
	if err != nil {
		t.Fatalf("Scales 失败: %v", err)
	}
	// j=4：sc = (q[8] & 0x0F) | ((q[0] >> 6) << 4) = 5 | (0 << 4) = 5
	//      m  = (q[8] >> 4)   | ((q[4] >> 6) << 4) = 8 | 0         = 8
	if got2[4].Scale != 5 {
		t.Errorf("子块[4].Scale = %v, want 5（高 2 位为 0 时低位应原样保留）", got2[4].Scale)
	}
	if got2[4].Min != 8 {
		t.Errorf("子块[4].Min = %v, want 8", got2[4].Min)
	}
	// 再让高 2 位非零，验证它被拼到 bit4-5：
	// q[0] = 0xC3 → 低 4 位 3，高 2 位 3；(q[0] >> 6) << 4 = 3 << 4 = 48
	blk2[4] = 0xC3
	got2, err = Scales(model.DtypeQ4K, blk2)
	if err != nil {
		t.Fatalf("Scales 失败: %v", err)
	}
	// j=4：sc = 5 | 48 = 53
	if got2[4].Scale != 53 {
		t.Errorf("子块[4].Scale = %v, want 53 —— 高 2 位没被拼进第 4、5 位", got2[4].Scale)
	}
}

// 未收录的类型必须报错，不能返回一堆 0。
func TestScales_未收录类型报错(t *testing.T) {
	for _, d := range []model.Dtype{model.DtypeF32, model.DtypeIQ2XXS, model.DtypeUnknown} {
		if _, err := Scales(d, make([]byte, 4096)); err == nil {
			t.Errorf("%s 不是已量化的块类型，应报错", d)
		}
	}
}

// 字节不足必须报错。
func TestScales_字节不足报错(t *testing.T) {
	if _, err := Scales(model.DtypeQ4K, make([]byte, 10)); err == nil {
		t.Fatal("字节不足应报错")
	}
}

// 多块时每个块都要提取到，且顺序与文件里一致。
//
// 只有一块的用例发现不了"第二块起偏移没跟上"这类错误。
func TestScales_多块(t *testing.T) {
	// 三块 Q8_0，d 分别是 0.0625 / 0.5 / 0.03125
	blk := func(bits uint16) []byte { return cat(le16(bits), times(0x40, 32)) }
	blkA := blk(0x2C00) // 0.0625
	blkB := blk(0x3800) // 0.5
	blkC := blk(0x2800) // 0.03125 = 2^-5
	raw := cat(blkA, blkB, blkC)

	got, err := Scales(model.DtypeQ8_0, raw)
	if err != nil {
		t.Fatalf("Scales 失败: %v", err)
	}
	want := []float32{0.0625, 0.5, 0.03125}
	if len(got) != len(want) {
		t.Fatalf("子块数 = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Scale != want[i] {
			t.Errorf("子块[%d].Scale = %v, want %v", i, got[i].Scale, want[i])
		}
	}
}

// ScalesSupported 与 Scales 必须永远一致。
//
// 两者读的是同一张表（scalesExtractors），但调用方（缓存的跳过判断）
// 拿 ScalesSupported 决定"要不要再算一次"：若是它说支持而 Scales 报错，
// 那个张量会每次运行都重扫；反过来则会永远得不到诊断且不报错。
// 这条用例逐个类型把两个函数对一遍，让"分成两处"这件事不可能漂移。
func TestScalesSupported_与Scales一致(t *testing.T) {
	// 覆盖已经收录的、块结构收录但布局没收录的、以及非量化类型
	all := []model.Dtype{
		model.DtypeQ4_0, model.DtypeQ4_1, model.DtypeQ5_0, model.DtypeQ5_1,
		model.DtypeQ8_0, model.DtypeQ8_1,
		model.DtypeQ2K, model.DtypeQ3K, model.DtypeQ4K, model.DtypeQ5K,
		model.DtypeQ6K, model.DtypeQ8K,
		model.DtypeIQ1S, model.DtypeIQ2XXS, model.DtypeIQ4NL,
		model.DtypeF32, model.DtypeF16, model.DtypeI32, model.DtypeBool,
		model.DtypeUnknown,
	}
	supported := 0
	for _, d := range all {
		t.Run(string(d), func(t *testing.T) {
			want := ScalesSupported(d)
			// 造一个足够长的缓冲：能过块大小检查的那种长度
			n := int64(4096)
			if bb, ok := d.BlockBytes(); ok && bb > 0 && n%bb != 0 {
				n = bb * 64
			}
			_, err := Scales(d, make([]byte, n))
			got := err == nil
			if got != want {
				t.Errorf("ScalesSupported(%s) = %v，但 Scales 的结果是 err=%v —— 两者必须一致",
					d, want, err)
			}
			if want {
				supported++
				if !d.IsQuantized() {
					t.Errorf("%s 不是量化类型，不该说支持", d)
				}
			}
		})
	}
	// 反向门禁：一个都没支持说明表结构变了，测试会静默变成空转
	if supported == 0 {
		t.Fatal("没有任何类型被判定为支持 —— 这条用例失去了意义")
	}
	t.Logf("%d/%d 个类型支持块头 scale 提取", supported, len(all))
}

// 真实块上，Scales 提取的 |scale| 必须能整除「解码值的相邻最小间隔」。
//
// **这是独立来源**：间隔由 gguf.quants（llama.cpp 官方 Python 绑定）
// 解码后算得，与 Scales 读块头的方式没有共同代码。
// 已有的两类测试都不够：合成夹具是同一份理解写两遍；
// 真实文件那条只查"值落在 (scale, min) 区间内"，而 Scales 与 Decode
// 共用 getScaleMinK4 —— 它错了两者一起错，区间约束照样成立。
//
// 子块内的值是等间隔格点（对称量化 value = scale×q，
// 带 min 的 value = scale×q − min），所以相邻不同值的最小间隔
// 必然是 |scale| 的正整数倍。偏移取错、解包写错、子块顺序错乱
// 都会让这个整除关系不成立。
func TestScales_真实块_与独立解码反推的步长一致(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "real_scales.json"))
	if err != nil {
		t.Fatalf("读 testdata/real_scales.json 失败（用 uv run --with gguf --with numpy "+
			"python3 tools/verify_gguf_scales.py 生成）: %v", err)
	}
	var all map[string]json.RawMessage
	if err := json.Unmarshal(raw, &all); err != nil {
		t.Fatalf("解析: %v", err)
	}

	total, exact, divisors := 0, 0, 0
	for _, name := range []string{"Q6_K", "Q4_K"} {
		seg, ok := all[name]
		if !ok {
			t.Fatalf("参照里没有 %s", name)
		}
		var ref struct {
			Tensor     string       `json:"tensor"`
			StartBlock int          `json:"start_block"`
			SubElems   int          `json:"sub_elems"`
			Gaps       [][]*float64 `json:"gaps"`
		}
		if err := json.Unmarshal(seg, &ref); err != nil {
			t.Fatalf("解析 %s 段: %v", name, err)
		}
		if len(ref.Gaps) == 0 {
			t.Fatalf("%s 的参照没有块 —— 会变成空转", name)
		}

		// 两个脚本从同一个张量的同一位置取块，必须对得上 ——
		// 对不上的话比的是两批不同的数据，结论无意义
		blocks := realBlocks(t, name)
		var blkMeta struct {
			Tensor     string `json:"tensor"`
			StartBlock int    `json:"start_block"`
		}
		{
			var v map[string]json.RawMessage
			braw, err := os.ReadFile(filepath.Join("testdata", "real_blocks.json"))
			if err != nil {
				t.Fatalf("读 real_blocks.json: %v", err)
			}
			if err := json.Unmarshal(braw, &v); err != nil {
				t.Fatalf("解析: %v", err)
			}
			if err := json.Unmarshal(v[name], &blkMeta); err != nil {
				t.Fatalf("解析 %s: %v", name, err)
			}
		}
		if blkMeta.Tensor != ref.Tensor || blkMeta.StartBlock != ref.StartBlock {
			t.Fatalf("%s 的样本来源对不上：块样本取自 %s 第 %d 块，步长参照取自 %s 第 %d 块",
				name, blkMeta.Tensor, blkMeta.StartBlock, ref.Tensor, ref.StartBlock)
		}

		d := model.Dtype(name)
		if int64(ref.SubElems) != d.BlockElems()/int64(len(ref.Gaps[0])) {
			t.Fatalf("%s: 参照说每个子块 %d 个权重，由 %d 个子块推得每块 %d 个 —— 对不上",
				name, ref.SubElems, len(ref.Gaps[0]), d.BlockElems())
		}

		for bi, gaps := range ref.Gaps {
			if bi >= len(blocks) {
				break
			}
			blk, err := hex.DecodeString(blocks[bi])
			if err != nil {
				t.Fatalf("第 %d 块不是合法 hex: %v", bi, err)
			}
			subs, err := Scales(d, blk)
			if err != nil {
				t.Fatalf("第 %d 块 Scales 失败: %v", bi, err)
			}
			if len(subs) != len(gaps) {
				t.Fatalf("%s 第 %d 块：Scales 给出 %d 个子块，参照有 %d 个",
					name, bi, len(subs), len(gaps))
			}
			for j, gap := range gaps {
				if gap == nil {
					continue // 该子块只有一个不同值（常量），间隔无意义
				}
				total++
				s := math.Abs(float64(subs[j].Scale))
				if s == 0 {
					t.Errorf("%s 第 %d 块子块 %d：Scales 给出 0，但解码值有间隔 %g",
						name, bi, j, *gap)
					continue
				}
				ratio := *gap / s
				// 间隔必须是 |scale| 的整数倍
				if math.Abs(ratio-math.Round(ratio)) > 1e-3*math.Max(ratio, 1) {
					t.Errorf("%s 第 %d 块子块 %d：最小间隔 %g 不是 |scale| %g 的整数倍（比值 %g）",
						name, bi, j, *gap, s, ratio)
					continue
				}
				divisors++
				if math.Abs(ratio-1) < 1e-3 {
					exact++
				}
			}
		}
	}

	if total == 0 || divisors == 0 {
		t.Fatal("一个子块都没比到 —— 这条检查会变成空转")
	}
	// 绝大多数子块里相邻两个码都存在，所以间隔应当**正好**是 |scale|。
	// 门槛取自实测值（下面打出来），不是拍脑袋
	const minExact = 0.9
	if rate := float64(exact) / float64(divisors); rate < minExact {
		t.Errorf("只有 %.0f%% 的子块间隔正好等于 |scale|（下限 %.0f%%）—— "+
			"提取多半系统性偏大或偏小", rate*100, minExact*100)
	}
	t.Logf("%d/%d 个子块的间隔是 |scale| 的整数倍，其中 %d 个正好等于（%.0f%%）",
		divisors, total, exact, float64(exact)/float64(divisors)*100)
}
