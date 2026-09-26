package decode

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/sillydong/modelview/internal/model"
)

// decodePlain 处理非量化类型：每个元素独立解码，没有块结构。
//
// 所有格式（GGUF / safetensors / PyTorch）的非量化数据都是小端紧密排列，
// 这一点由各解析器在读取阶段已经确认（PyTorch 侧大端会被拒绝）。
func decodePlain(d model.Dtype, src []byte, dst []float32) error {
	eb, ok := d.ByteSize()
	if !ok {
		return ErrUnsupported{Dtype: d}
	}
	if int64(len(src)) < int64(len(dst))*eb {
		return fmt.Errorf("字节不足：%d 个 %s 元素需要 %d 字节，只有 %d",
			len(dst), d, int64(len(dst))*eb, len(src))
	}

	switch d {
	case model.DtypeF32:
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
		}
	case model.DtypeF64:
		for i := range dst {
			dst[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(src[i*8:])))
		}
	case model.DtypeF16:
		for i := range dst {
			dst[i] = f16ToF32(binary.LittleEndian.Uint16(src[i*2:]))
		}
	case model.DtypeBF16:
		// bf16 就是 float32 砍掉低 16 位尾数，补零即可
		for i := range dst {
			dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(src[i*2:])) << 16)
		}
	case model.DtypeI8:
		for i := range dst {
			dst[i] = float32(int8(src[i]))
		}
	case model.DtypeU8:
		for i := range dst {
			dst[i] = float32(src[i])
		}
	case model.DtypeI16:
		for i := range dst {
			dst[i] = float32(int16(binary.LittleEndian.Uint16(src[i*2:])))
		}
	case model.DtypeU16:
		for i := range dst {
			dst[i] = float32(binary.LittleEndian.Uint16(src[i*2:]))
		}
	case model.DtypeI32:
		for i := range dst {
			dst[i] = float32(int32(binary.LittleEndian.Uint32(src[i*4:])))
		}
	case model.DtypeU32:
		for i := range dst {
			dst[i] = float32(binary.LittleEndian.Uint32(src[i*4:]))
		}
	case model.DtypeI64:
		for i := range dst {
			dst[i] = float32(int64(binary.LittleEndian.Uint64(src[i*8:])))
		}
	case model.DtypeU64:
		for i := range dst {
			dst[i] = float32(binary.LittleEndian.Uint64(src[i*8:]))
		}
	case model.DtypeBool:
		for i := range dst {
			// **必须无条件写**，不能只在非零时赋值。
			//
			// dst 是调用方给的：analyze 跨分块复用它（internal/analyze/analyze.go），
			// 只在非零时赋值会让 false 元素保留**上一轮**的值 ——
			// 8,388,609 个元素、只有一个 0 的张量会算出 zero_ratio=0、min=1，
			// 那个 0 完全不可见。而单块文件恰好正确（新分配的 dst 本来就是 0），
			// 所以小文件测试发现不了。
			//
			// 其余 13 种类型都是写满 dst 的，这里不能例外。
			if src[i] != 0 {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
	default:
		// F8_E4M3 / F8_E5M2 落到这里：两种 FP8 的指数位与偏置不同，
		// 本项目没有可用语料验证，不给可能错误的数字。
		return ErrUnsupported{Dtype: d}
	}
	return nil
}
