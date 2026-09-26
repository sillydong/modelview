package model

import "testing"

// BlockBytes 是「一个块占多少字节」的唯一出处。
//
// 数值原先在 parser/gguf 里按 GGML 类型码另存一份，本测试逐项钉住迁移结果 ——
// 抄错一个数字，这里立刻报出来。
func TestBlockBytes(t *testing.T) {
	tests := []struct {
		d    Dtype
		want int64
	}{
		{DtypeF64, 8}, {DtypeF32, 4}, {DtypeF16, 2}, {DtypeBF16, 2},
		{DtypeF8E4M3, 1}, {DtypeF8E5M2, 1},
		{DtypeI64, 8}, {DtypeI32, 4}, {DtypeI16, 2}, {DtypeI8, 1},
		{DtypeU64, 8}, {DtypeU32, 4}, {DtypeU16, 2}, {DtypeU8, 1},
		{DtypeBool, 1},
		{DtypeQ4_0, 18}, {DtypeQ4_1, 20}, {DtypeQ5_0, 22}, {DtypeQ5_1, 24},
		{DtypeQ8_0, 34}, {DtypeQ8_1, 36},
		{DtypeQ2K, 84}, {DtypeQ3K, 110}, {DtypeQ4K, 144},
		{DtypeQ5K, 176}, {DtypeQ6K, 210}, {DtypeQ8K, 292},
	}
	for _, tt := range tests {
		got, ok := tt.d.BlockBytes()
		if !ok || got != tt.want {
			t.Errorf("BlockBytes(%s) = (%d, %v), want (%d, true)", tt.d, got, ok, tt.want)
		}
	}
}

// IQ 系列没有收录块结构，必须返回 false 而不是猜一个数字。
func TestBlockBytes_IQ系列未收录(t *testing.T) {
	for _, d := range []Dtype{DtypeIQ1S, DtypeIQ2XXS, DtypeIQ4NL, DtypeUnknown, Dtype("不存在")} {
		if _, ok := d.BlockBytes(); ok {
			t.Errorf("BlockBytes(%s) 应为 false —— IQ 系列块结构未验证，不能给数字", d)
		}
	}
}

// 块的元素数：非量化类型是 1，量化类型是 32 或 256。
func TestBlockElems(t *testing.T) {
	tests := []struct {
		d    Dtype
		want int64
	}{
		{DtypeF32, 1}, {DtypeF16, 1}, {DtypeBool, 1},
		{DtypeQ4_0, 32}, {DtypeQ8_0, 32},
		{DtypeQ2K, 256}, {DtypeQ4K, 256}, {DtypeQ6K, 256},
		{DtypeUnknown, 0}, {Dtype("不存在"), 0},
	}
	for _, tt := range tests {
		if got := tt.d.BlockElems(); got != tt.want {
			t.Errorf("BlockElems(%s) = %d, want %d", tt.d, got, tt.want)
		}
	}
}

// 非量化类型的块字节数必须等于每元素字节数 —— 两者是同一件事的两种说法。
func TestBlockBytes_与ByteSize一致(t *testing.T) {
	for d, p := range dtypeTable {
		if p.IsQuantized {
			continue
		}
		bs, ok := d.ByteSize()
		if !ok {
			continue
		}
		bb, ok := d.BlockBytes()
		if !ok || bb != bs {
			t.Errorf("%s: BlockBytes=%d ByteSize=%d —— 非量化类型两者必须相同", d, bb, bs)
		}
	}
}

// 量化类型的位宽必须能由块字节数推导出来。
// 这是位宽表与块表之间的交叉约束：改了一处不改另一处，这里会红。
func TestBlockBytes_量化位宽自洽(t *testing.T) {
	covered := 0
	for d, p := range dtypeTable {
		if !p.IsQuantized || p.BitsPerWeight == 0 {
			continue
		}
		covered++
		bb, ok := d.BlockBytes()
		if !ok {
			t.Errorf("%s 声明了位宽 %v 却没有块字节数", d, p.BitsPerWeight)
			continue
		}
		got := float64(bb) * 8 / float64(p.BlockSize)
		if diff := got - p.BitsPerWeight; diff > 1e-9 || diff < -1e-9 {
			t.Errorf("%s: 由块字节数推得 %v bit/权重，表里写的是 %v", d, got, p.BitsPerWeight)
		}
	}
	// 反向门禁：覆盖数掉了说明 table 结构变了，测试会静默变成空转。
	//
	// 12 是实测值：Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1 Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K。
	// 写成 18 会让测试失败 —— 这条门禁第一次运行就抓出了我自己写错的预期值。
	const wantCovered = 12
	if covered != wantCovered {
		t.Errorf("覆盖了 %d 个量化类型，预期 %d 个 —— 表结构变了，重新核对后再改这个数",
			covered, wantCovered)
	}
}

// IQ 系列不能有块字节数，但必须有块元素数（界面上要显示"每块 256 个权重"）。
func TestBlockBytes_IQ系列有位宽表但无块字节(t *testing.T) {
	for d, p := range dtypeTable {
		if !p.IsQuantized || p.BitsPerWeight != 0 {
			continue
		}
		if _, ok := d.BlockBytes(); ok {
			t.Errorf("%s 是 IQ 系列（无位宽），不该有块字节数", d)
		}
		if d.BlockElems() <= 0 {
			t.Errorf("%s 应有块元素数", d)
		}
	}
}
