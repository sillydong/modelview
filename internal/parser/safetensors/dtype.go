package safetensors

import (
	"fmt"

	"github.com/sillydong/modelview/internal/model"
)

// dtypeByCode 把 safetensors 头部里的 dtype 字符串映射到 model.Dtype。
//
// 取值来自 safetensors 规范。**本地 37 个文件只覆盖了 F32**，
// 其余取值按规范实现但没有本地文件验证。
var dtypeByCode = map[string]model.Dtype{
	"F64":     model.DtypeF64,
	"F32":     model.DtypeF32,
	"F16":     model.DtypeF16,
	"BF16":    model.DtypeBF16,
	"F8_E4M3": model.DtypeF8E4M3,
	"F8_E5M2": model.DtypeF8E5M2,
	"I64":     model.DtypeI64,
	"I32":     model.DtypeI32,
	"I16":     model.DtypeI16,
	"I8":      model.DtypeI8,
	"U64":     model.DtypeU64,
	"U32":     model.DtypeU32,
	"U16":     model.DtypeU16,
	"U8":      model.DtypeU8,
	"BOOL":    model.DtypeBool,
}

// ErrUnknownDtype 表示头部里出现了未知的 dtype 字符串。
type ErrUnknownDtype struct{ Code string }

func (e ErrUnknownDtype) Error() string {
	return fmt.Sprintf("未知的 safetensors dtype %q", e.Code)
}

func parseDtype(code string) (model.Dtype, error) {
	d, ok := dtypeByCode[code]
	if !ok {
		return model.DtypeUnknown, ErrUnknownDtype{Code: code}
	}
	return d, nil
}
