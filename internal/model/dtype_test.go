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
	// 29 是实测值：GGML 类型码 0..30 去掉 4/5（上游已废弃的 Q4_2/Q4_3）
	// 正好 29 个。写计划时这里写的是 31，是把"0 到 30 共 31 个数"
	// 当成了"31 个码"—— 对着代码一数就露馅了，这正是反向门禁的意义。
	const wantCovered = 29
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
