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
//
// 遍历 map 而不是遍历 0..N —— 固定上界会在新增更大的类型码时静默失效。
func TestGGMLDtype_全部在model表内(t *testing.T) {
	for code, dt := range ggmlTypeCode {
		if !dt.Known() {
			t.Errorf("类型码 %d → %q 在 model.dtypeTable 中缺失", code, dt)
		}
	}
}

// 反向闸门：**每个声明了位宽的类型码，都必须在块表里有一项**。
//
// 这是防漂移的关键一环。位宽表与块表是同一事实的两处表达，
// 只检查"块表里的码在位宽表里有对应项"是不够的 ——
// 那会让"位宽表有、块表没有"的新增类型静默通过，
// 结果是该类型所有张量的 byte_size 恒为 0，而没有任何测试失败。
func TestBlockTable_声明位宽的类型必须在块表(t *testing.T) {
	for code, dt := range ggmlTypeCode {
		if dt.BitsPerWeight() == 0 {
			continue // IQ 系列故意不声明位宽
		}
		if _, ok := dt.BlockBytes(); !ok {
			t.Errorf("类型码 %d（%s）声明了位宽 %.4f，却取不到块字节数 —— "+
				"该类型的张量将无法计算占用大小",
				code, dt, dt.BitsPerWeight())
		}
	}
}

// 反向闸门之二：未声明位宽的类型（IQ 系列）必须也取不到块字节数 ——
// 否则算出的占用大小会与展示的位宽矛盾。
func TestBlockTable_未声明位宽的类型取不到块字节(t *testing.T) {
	for code, dt := range ggmlTypeCode {
		if dt.BitsPerWeight() != 0 {
			continue
		}
		if _, ok := dt.BlockBytes(); ok {
			t.Errorf("%s（码 %d）未声明位宽，却取到了块字节数", dt, code)
		}
	}
}

// 原先这里还有两条门禁，块表迁到 model.Dtype 之后它们成了同义反复
// （块表与位宽表本就是同一张表），已删：
//
//   - 「块表里的每个码都必须是已知类型码」：现在 BlockBytes 由 Dtype 自己给出，
//     不存在"码认识但块表不认识"的组合了。
//   - 「块表与位宽表自洽」：同一张表，恒等成立。等价的检查搬到了
//     model 的 TestBlockBytes_量化位宽自洽 —— 那里仍然能抓到
//     "改了字节数忘了改位宽"（或反过来）。

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
		// 截断阈值是 maxArrayPrint = 8。边界两侧都要有用例 ——
		// 只测 3 项和 100 项的话，把 > 改成 >= 不会有任何测试失败。
		{"恰好 8 项不截断", arrayValue{ElemType: typeUint32, Len: 8,
			NumElems: mkAny(8)}, "[0, 1, 2, 3, 4, 5, 6, 7]"},
		{"9 项开始截断", arrayValue{ElemType: typeUint32, Len: 9,
			NumElems: mkAny(9)}, "[0, 1, 2, 3, 4, 5, 6, 7, … 共 9 项]"},
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
