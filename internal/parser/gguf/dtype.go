package gguf

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/sillydong/modelview/internal/model"
)

// ggmlDtype 把 GGML 类型码映射到 model.Dtype。
//
// 表本身在 model 里（类型码是 Dtype 自己的属性，见表里的 GGMLCode），
// 这里只是按码反查 —— 两处各存一份的话，"码 12 是 Q4_K 还是 Q6_K"
// 这种错不会有任何东西编译失败。
//
// 线性扫 30 来项、只在解析头部时调用，不值得为它建索引。
func ggmlDtype(code uint32) (model.Dtype, bool) {
	for d := range model.AllDtypes() {
		if c, ok := d.GGMLCode(); ok && c == code {
			return d, true
		}
	}
	return model.DtypeUnknown, false
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
