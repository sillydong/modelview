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
