package model

import "testing"

func TestByteSize(t *testing.T) {
	tests := []struct {
		in   Dtype
		want int64
		ok   bool
	}{
		{DtypeF64, 8, true},
		{DtypeF32, 4, true},
		{DtypeF16, 2, true},
		{DtypeBF16, 2, true},
		{DtypeF8E4M3, 1, true},
		{DtypeI64, 8, true},
		{DtypeU64, 8, true},
		{DtypeU16, 2, true},
		{DtypeI8, 1, true},
		{DtypeBool, 1, true},
		// 量化类型按块存储，没有"每元素字节数"
		{DtypeQ4K, 0, false},
		{DtypeQ8_0, 0, false},
		{DtypeQ6K, 0, false},
		// IQ 系列连位宽都没收录
		{DtypeIQ2XXS, 0, false},
		// 未知类型
		{DtypeUnknown, 0, false},
		{Dtype("不存在"), 0, false},
	}
	for _, tt := range tests {
		got, ok := tt.in.ByteSize()
		if ok != tt.ok || got != tt.want {
			t.Errorf("ByteSize(%s) = (%d, %v), want (%d, %v)", tt.in, got, ok, tt.want, tt.ok)
		}
	}
}

// 位宽表里每个非量化类型都必须能取到字节数 ——
// 这是反向门禁：新加一个非量化 dtype 却漏配 BlockSize/BitsPerWeight 时，
// ByteSize 会静默返回 false，让调用方的完整性校验凭空消失。
func TestByteSize_非量化类型全覆盖(t *testing.T) {
	for d, p := range dtypeTable {
		if p.IsQuantized {
			continue
		}
		if _, ok := d.ByteSize(); !ok {
			t.Errorf("%s 是非量化类型（BitsPerWeight=%v, BlockSize=%d）却取不到字节数",
				d, p.BitsPerWeight, p.BlockSize)
		}
	}
}

// 每个 GGML 类型码都必须映射到一个**已在 dtypeTable 里收录**的 Dtype，
// 且映射是双射（一个码对一个类型，一个类型对一个码）。
//
// 这张表原先在 parser/gguf 里，是速查表要列类型码时收敛过来的 ——
// 分成两份时它们可以各自漂移，而"码 12 是 Q4_K 还是 Q6_K"这种错
// 不会让任何东西编译失败。
func TestGGMLCode_与类型表双射(t *testing.T) {
	byCode := map[uint32]Dtype{}
	covered := 0
	for d, p := range dtypeTable {
		if p.GGMLCode < 0 {
			continue // 没有 GGML 类型码（safetensors 专有的 dtype）
		}
		covered++
		code := uint32(p.GGMLCode)
		if prev, ok := byCode[code]; ok {
			t.Errorf("类型码 %d 同时映射到 %s 与 %s", code, prev, d)
		}
		byCode[code] = d
		if got, ok := d.GGMLCode(); !ok || got != code {
			t.Errorf("%s.GGMLCode() = (%d, %v), want (%d, true)", d, got, ok, code)
		}
	}
	// 反向门禁：覆盖数掉了说明表结构变了，测试会静默变成空转。
	//
	// 31 是实测值：0..30 去掉 4/5（上游已废弃的 Q4_2/Q4_3）得 29，
	// 加上 MXFP4(39) 与 NVFP4(40) = 31。
	//
	// 这个数被改过两次，两次都是对着 gguf 包（= ggml 的常量表）数出来的，
	// 不是估的：写计划时这里写 31，是把"0 到 30 共 31 个数"当成了
	// "31 个码"；后来加 MXFP4/NVFP4 时又变成 31 —— 数字一样但含义不同，
	// 所以对照的是 gguf 包的定义而不是记忆。
	const wantCovered = 31
	if covered != wantCovered {
		t.Errorf("收录了 %d 个类型码，预期 %d 个 —— 表结构变了，重新核对后再改这个数",
			covered, wantCovered)
	}
	if len(byCode) != covered {
		t.Errorf("码的个数 %d 与类型个数 %d 不等 —— 不是双射", len(byCode), covered)
	}
}

// 没有 GGML 类型码的类型必须明确返回 false，不能给一个 0
// （0 是合法的类型码，会给到 F32）。
func TestGGMLCode_未收录返回false(t *testing.T) {
	for _, d := range []Dtype{DtypeUnknown, Dtype("不存在"),
		DtypeBool, DtypeU8, DtypeF8E4M3} {
		if code, ok := d.GGMLCode(); ok {
			t.Errorf("%s.GGMLCode() = (%d, true)，应为 false —— "+
				"0 是合法类型码，不能拿它当哨兵", d, code)
		}
	}
}

// GGML 类型码必须与上游一致 —— **独立抄录的清单**，不是从表里读的。
//
// 为什么单靠双射测试不够：它只验"码与类型内部一致"（不重不漏、来回可逆），
// 一个**唯一但错误**的码照样通过。实测：把 MXFP4 的 39 改成 38，
// 双射测试全绿 —— 而 38 根本不是 ggml 的类型码（它是 llama.cpp 的
// file_type 编号，两者是不同的枚举，很容易混）。
//
// 下面的清单逐条抄自 gguf 包的 GGMLQuantizationType（= ggml 的
// ggml_type 枚举）。抄错本身没人管，但它与表**两处独立**，
// 改表时这里会红 —— 这正是要的。
//
// 可重跑的对照：tools/verify_ggml_types.py（直接读 gguf 包比一遍）。
func TestGGMLCode_与上游常量一致(t *testing.T) {
	upstream := []struct {
		code uint32
		name string
	}{
		{0, "F32"},
		{1, "F16"},
		{2, "Q4_0"},
		{3, "Q4_1"},
		{6, "Q5_0"},
		{7, "Q5_1"},
		{8, "Q8_0"},
		{9, "Q8_1"},
		{10, "Q2_K"},
		{11, "Q3_K"},
		{12, "Q4_K"},
		{13, "Q5_K"},
		{14, "Q6_K"},
		{15, "Q8_K"},
		{16, "IQ2_XXS"},
		{17, "IQ2_XS"},
		{18, "IQ3_XXS"},
		{19, "IQ1_S"},
		{20, "IQ4_NL"},
		{21, "IQ3_S"},
		{22, "IQ2_S"},
		{23, "IQ4_XS"},
		{24, "I8"},
		{25, "I16"},
		{26, "I32"},
		{27, "I64"},
		{28, "F64"},
		{29, "IQ1_M"},
		{30, "BF16"},
		{39, "MXFP4"},
		{40, "NVFP4"},
	}
	byCode := map[uint32]string{}
	for d := range AllDtypes() {
		if c, ok := d.GGMLCode(); ok {
			byCode[c] = string(d)
		}
	}
	for _, u := range upstream {
		got, ok := byCode[u.code]
		if !ok {
			t.Errorf("ggml 类型码 %d（%s）我们没收录", u.code, u.name)
			continue
		}
		if got != u.name {
			t.Errorf("ggml 类型码 %d 我们叫 %q，上游叫 %q", u.code, got, u.name)
		}
		delete(byCode, u.code)
	}
	for c, name := range byCode {
		t.Errorf("我们收了类型码 %d（%s），但上面那份上游清单里没有 —— "+
			"要么抄漏了，要么这个码根本不存在", c, name)
	}
	// 反向门禁：清单被截断时上面两条会"零缺失"地通过
	if len(upstream) != 31 {
		t.Fatalf("上游清单 %d 条，预期 31 条 —— 被截断了？", len(upstream))
	}
}
