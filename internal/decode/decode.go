// Package decode 把张量的原始字节解码成 float32 数值。
//
// 本包不认识文件格式，只认识 model.Dtype 与一段字节。
//
// 量化块的内部布局（哪个字节是哪一位、scale 怎么打包）**只能靠独立参照确定**。
// 本包的实现逐值比对过 llama.cpp 官方的 gguf.quants（见
// tools/verify_ggml_dequant.py），几处反直觉的地方在各自函数上有注释说明。
package decode

import (
	"fmt"

	"github.com/sillydong/modelview/internal/model"
)

// ErrUnsupported 表示该类型的解码未收录。
//
// 单独一个类型而不是 fmt.Errorf：调用方要能用 errors.As 区分
// 「这个类型我们解不了」和「这段字节坏了」。
type ErrUnsupported struct{ Dtype model.Dtype }

func (e ErrUnsupported) Error() string {
	return fmt.Sprintf("类型 %s 的解码未实现", e.Dtype)
}

// Decode 把 src 里的原始字节解码成 len(dst) 个 float32 数值。
//
// src 必须至少包含 len(dst) 个元素所需的字节（量化类型按整块计）。
// 解码的语义是「原样还原权重」，不做任何归一化或裁剪。
func Decode(d model.Dtype, src []byte, dst []float32) error {
	if len(dst) == 0 {
		return nil
	}
	if !d.IsQuantized() {
		return decodePlain(d, src, dst)
	}
	return decodeQuant(d, src, dst)
}
