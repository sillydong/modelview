package gguf

import (
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

func TestGGMLDtype(t *testing.T) {
	tests := []struct {
		code uint32
		want model.Dtype
	}{
		{0, model.DtypeF32},
		{1, model.DtypeF16},
		{6, model.DtypeQ5_0},
		{8, model.DtypeQ8_0},
		{12, model.DtypeQ4K},
		{14, model.DtypeQ6K},
		{30, model.DtypeBF16},
	}
	for _, tt := range tests {
		got, ok := ggmlDtype(tt.code)
		if !ok {
			t.Errorf("ggmlDtype(%d) 未识别", tt.code)
			continue
		}
		if got != tt.want {
			t.Errorf("ggmlDtype(%d) = %q, want %q", tt.code, got, tt.want)
		}
	}

	if _, ok := ggmlDtype(999); ok {
		t.Error("未知类型码应返回 ok=false")
	}
}

// 每个已识别的类型码都必须在 model 的位宽表里有对应项，
// 否则用于展示的位宽会是 0。
func TestGGMLDtype_全部在model表内(t *testing.T) {
	for code := uint32(0); code <= 40; code++ {
		dt, ok := ggmlDtype(code)
		if !ok {
			continue
		}
		if !dt.Known() {
			t.Errorf("类型码 %d → %q 在 model.dtypeTable 中缺失", code, dt)
		}
	}
}

// 块表与位宽表必须自洽：由块字节数推出的位宽应等于 model 表里声明的位宽。
// 两张表分处两个包，这条测试是它们之间唯一的防漂移闸门。
func TestBlockTable_与位宽表自洽(t *testing.T) {
	for code, size := range blockBytes {
		dt, ok := ggmlDtype(code)
		if !ok {
			t.Errorf("类型码 %d 在块表里但不在类型码表里", code)
			continue
		}
		elems := blockElemCount(code)
		derived := float64(size) * 8 / float64(elems)
		declared := dt.BitsPerWeight()
		if declared == 0 {
			// IQ 系列故意不声明位宽，跳过。
			continue
		}
		if diff := derived - declared; diff > 1e-9 || diff < -1e-9 {
			t.Errorf("%s（码 %d）：块表推出 %.4f bits/weight，位宽表声明 %.4f",
				dt, code, derived, declared)
		}
	}
}

// 未声明位宽的类型（IQ 系列）必须也不在块表里 —— 否则算出的占用大小会与展示的位宽矛盾。
func TestBlockTable_未声明位宽的类型不在块表(t *testing.T) {
	for code := uint32(0); code <= 40; code++ {
		dt, ok := ggmlDtype(code)
		if !ok {
			continue
		}
		if dt.BitsPerWeight() == 0 {
			if _, inBlock := blockBytes[code]; inBlock {
				t.Errorf("%s（码 %d）未声明位宽，却在块表里", dt, code)
			}
		}
	}
}

func TestFormatValue(t *testing.T) {
	tests := []struct {
		name string
		in   any
		want string
	}{
		{"uint32", uint32(36), "36"},
		{"int64", int64(-5), "-5"},
		{"float32 整数", float32(2048), "2048"},
		{"bool 真", true, "true"},
		{"bool 假", false, "false"},
		{"字符串", "qwen2", "qwen2"},
		{"空字符串", "", ""},
		{"短数组", arrayValue{ElemType: typeUint32, Len: 3,
			NumElems: []any{uint32(1), uint32(2), uint32(3)}}, "[1, 2, 3]"},
		{"长数组截断显示", arrayValue{ElemType: typeUint32, Len: 100,
			NumElems: mkAny(100)}, "[0, 1, 2, 3, 4, 5, 6, 7, … 共 100 项]"},
		{"字符串数组", arrayValue{ElemType: typeString, Len: 2,
			StrElems: []string{"a", "b"}}, `["a", "b"]`},
		{"空数组", arrayValue{ElemType: typeUint32, Len: 0,
			NumElems: []any{}}, "[]"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := formatValue(tt.in)
			if got != tt.want {
				t.Errorf("formatValue = %q, want %q", got, tt.want)
			}
		})
	}
}

func mkAny(n int) []any {
	out := make([]any, n)
	for i := range out {
		out[i] = uint32(i)
	}
	return out
}
