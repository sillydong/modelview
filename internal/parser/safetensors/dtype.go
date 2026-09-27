package safetensors

import (
	"fmt"

	"github.com/sillydong/modelview/internal/model"
)

// ErrUnknownDtype 表示头部里出现了未知的 dtype 字符串。
type ErrUnknownDtype struct{ Code string }

func (e ErrUnknownDtype) Error() string {
	return fmt.Sprintf("未知的 safetensors dtype %q", e.Code)
}

// parseDtype 把 safetensors 头部里的 dtype 字符串映射到 model.Dtype。
//
// 取值清单在 model 里（model.SafetensorsDtypes）—— 速查表要展示同一份清单，
// 这里再写一份的话，将来 safetensors 加一个新类型时必然只改一处。
//
// **本地 37 个文件只覆盖了 F32**，其余取值按规范实现但没有本地文件验证。
func parseDtype(code string) (model.Dtype, error) {
	d, ok := model.DtypeFromSafetensors(code)
	if !ok {
		return model.DtypeUnknown, ErrUnknownDtype{Code: code}
	}
	return d, nil
}
