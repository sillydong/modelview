package gguf

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/sillydong/modelview/internal/model"
)

// ggmlTypeCode 把 GGML 类型码映射到 model.Dtype。
// 未知码返回 ok=false，调用方应保留原始码并在界面上标注为未知。
var ggmlTypeCode = map[uint32]model.Dtype{
	0:  model.DtypeF32,
	1:  model.DtypeF16,
	2:  model.DtypeQ4_0,
	3:  model.DtypeQ4_1,
	6:  model.DtypeQ5_0,
	7:  model.DtypeQ5_1,
	8:  model.DtypeQ8_0,
	9:  model.DtypeQ8_1,
	10: model.DtypeQ2K,
	11: model.DtypeQ3K,
	12: model.DtypeQ4K,
	13: model.DtypeQ5K,
	14: model.DtypeQ6K,
	15: model.DtypeQ8K,
	16: model.DtypeIQ2XXS,
	17: model.DtypeIQ2XS,
	18: model.DtypeIQ3XXS,
	19: model.DtypeIQ1S,
	20: model.DtypeIQ4NL,
	21: model.DtypeIQ3S,
	22: model.DtypeIQ2S,
	23: model.DtypeIQ4XS,
	24: model.DtypeI8,
	25: model.DtypeI16,
	26: model.DtypeI32,
	27: model.DtypeI64,
	28: model.DtypeF64,
	29: model.DtypeIQ1M,
	30: model.DtypeBF16,
}

func ggmlDtype(code uint32) (model.Dtype, bool) {
	d, ok := ggmlTypeCode[code]
	return d, ok
}

// maxArrayPrint 是数组在字符串形式里最多打印多少个元素。
const maxArrayPrint = 8

// formatValue 把元数据值转成可读字符串。
//
// 长数组只打印前若干项并标注总长度 —— 这是**显示层**的截断，
// 原始值仍完整保留在 MetaKV.Raw 里，JSON 输出不受影响。
func formatValue(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case bool:
		return strconv.FormatBool(x)
	case float32:
		return strconv.FormatFloat(float64(x), 'g', -1, 32)
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	case arrayValue:
		return formatArray(x)
	default:
		return fmt.Sprint(x)
	}
}

func formatArray(a arrayValue) string {
	var elem []string
	switch {
	case a.StrElems != nil:
		for _, s := range a.StrElems {
			elem = append(elem, strconv.Quote(s))
		}
	case a.NumElems != nil:
		for _, n := range a.NumElems {
			elem = append(elem, formatValue(n))
		}
	case a.Nested != nil:
		for _, sub := range a.Nested {
			elem = append(elem, formatArray(sub))
		}
	}

	if len(elem) > maxArrayPrint {
		head := strings.Join(elem[:maxArrayPrint], ", ")
		return fmt.Sprintf("[%s, … 共 %d 项]", head, a.Len)
	}
	return "[" + strings.Join(elem, ", ") + "]"
}
