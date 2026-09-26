package pytorch

import (
	"fmt"

	"github.com/sillydong/modelview/internal/model"
)

// storageDtype 把 PyTorch 的存储类名映射到 model.Dtype。
//
// **实测本地 22 个 .pt 只用到 FloatStorage 与 HalfStorage**，
// 其余取值按 PyTorch 的命名约定列出，但没有本地文件验证。
var storageDtype = map[string]model.Dtype{
	"DoubleStorage":   model.DtypeF64,
	"FloatStorage":    model.DtypeF32,
	"HalfStorage":     model.DtypeF16,
	"BFloat16Storage": model.DtypeBF16,
	"ByteStorage":     model.DtypeU8,
	"CharStorage":     model.DtypeI8,
	"ShortStorage":    model.DtypeI16,
	"IntStorage":      model.DtypeI32,
	"LongStorage":     model.DtypeI64,
	"BoolStorage":     model.DtypeBool,
}

// ErrUnknownStorage 表示遇到了未收录的存储类名。
type ErrUnknownStorage struct{ Class string }

func (e ErrUnknownStorage) Error() string {
	return fmt.Sprintf("未收录的 PyTorch 存储类 %q", e.Class)
}

// parseStorageClass 从 GLOBAL 引用里取出类名并映射。
//
// 分隔符**两种都要处理**：pickletools 显示为 "torch FloatStorage"（空格），
// 而本包的 GLOBAL 解析用点号连接模块名与类名，得到 "torch.FloatStorage"。
// 只按空格切分会取到完整串，导致所有真实文件都查表失败。
func parseStorageClass(full string) (model.Dtype, error) {
	name := full
	if i := lastSeparator(full); i >= 0 {
		name = full[i+1:]
	}
	d, ok := storageDtype[name]
	if !ok {
		return model.DtypeUnknown, ErrUnknownStorage{Class: name}
	}
	return d, nil
}

// lastSeparator 返回最后一个 '.' 或 ' ' 的位置；都没有则返回 -1。
func lastSeparator(s string) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '.' || s[i] == ' ' {
			return i
		}
	}
	return -1
}
