package decode

import (
	"encoding/binary"
	"fmt"

	"github.com/sillydong/modelview/internal/model"
)

// SubScale 是一个（子）块的缩放参数。
//
// K 系列的一个 256 元素块里有多个子块，每个子块有自己的 scale 与 min；
// Q8_0 / Q4_0 这类 32 元素块只有一个，Min 恒为 0（对称量化）。
type SubScale struct {
	Scale float32
	Min   float32
	// Elems 是这个 (Scale, Min) 覆盖的权重个数。
	Elems int
}

// Scales 提取一个张量里每个（子）块的 scale 与 min。
//
// **只读块头，不反量化权重** —— 这比 Decode 便宜得多，
// 所以块级诊断可以覆盖全部块而不受采样影响。
//
// 只支持已量化的块类型；非量化类型与未收录类型都返回 ErrUnsupported。
func Scales(d model.Dtype, src []byte) ([]SubScale, error) {
	perBlock, ok := d.BlockBytes()
	if !ok || !d.IsQuantized() {
		return nil, ErrUnsupported{Dtype: d}
	}
	if int64(len(src))%perBlock != 0 {
		return nil, fmt.Errorf("字节数 %d 不是块大小 %d 的整数倍", len(src), perBlock)
	}
	nBlocks := int64(len(src)) / perBlock

	fn, ok := scalesExtractors[d]
	if !ok {
		// Q8_1 与 Q8_K：块字节数收录了，但布局没有独立参照验证过。
		// IQ 系列连块字节数都没有。
		return nil, ErrUnsupported{Dtype: d}
	}
	return fn(src, nBlocks, perBlock), nil
}

// scalesExtractors 是「块头里怎么取 scale/min」的唯一出处。
//
// 做成表而不是 Scales 里的 switch，是为了让 ScalesSupported 与 Scales
// 读同一份事实 —— 分成两处表达的话，迟早出现
// 「ScalesSupported 说支持、Scales 却报错」的漂移，
// 而调用方（缓存的跳过判断）会据此认为这个张量永远没算完、每次重扫。
var scalesExtractors = map[model.Dtype]func([]byte, int64, int64) []SubScale{
	model.DtypeQ4_0: func(src []byte, n, pb int64) []SubScale {
		return scalesSingle(src, n, pb, 0, noMin, 32)
	},
	model.DtypeQ4_1: func(src []byte, n, pb int64) []SubScale {
		return scalesSingle(src, n, pb, 0, 2, 32)
	},
	model.DtypeQ5_0: func(src []byte, n, pb int64) []SubScale {
		return scalesSingle(src, n, pb, 0, noMin, 32)
	},
	model.DtypeQ5_1: func(src []byte, n, pb int64) []SubScale {
		return scalesSingle(src, n, pb, 0, 2, 32)
	},
	model.DtypeQ8_0: func(src []byte, n, pb int64) []SubScale {
		return scalesSingle(src, n, pb, 0, noMin, 32)
	},
	model.DtypeQ4K: scalesK4,
	model.DtypeQ5K: scalesK4,
	model.DtypeQ2K: scalesQ2K,
	model.DtypeQ3K: scalesQ3K,
	model.DtypeQ6K: scalesQ6K,
}

// ScalesSupported 表示这个类型的块头 scale 布局已收录。
//
// 调用方拿它区分「还没有诊断」与「这个类型永远算不出诊断」——
// 后者不该让缓存每次都重扫。
func ScalesSupported(d model.Dtype) bool {
	if !d.IsQuantized() {
		return false
	}
	if _, ok := d.BlockBytes(); !ok {
		return false
	}
	_, ok := scalesExtractors[d]
	return ok
}

// noMin 表示该类型是对称量化，块里没有 min 字段。
const noMin = -1

// scalesSingle 处理 32 元素块：d 在块头的第 dOff 字节，min 在第 mOff 字节
// （mOff == noMin 表示没有 min，此时 Min 恒为 0）。
func scalesSingle(src []byte, nBlocks, perBlock int64, dOff, mOff, elems int) []SubScale {
	out := make([]SubScale, 0, nBlocks)
	for b := int64(0); b < nBlocks; b++ {
		blk := src[b*perBlock:]
		s := SubScale{
			Scale: f16ToF32(binary.LittleEndian.Uint16(blk[dOff:])),
			Elems: elems,
		}
		if mOff >= 0 {
			s.Min = f16ToF32(binary.LittleEndian.Uint16(blk[mOff:]))
		}
		out = append(out, s)
	}
	return out
}

// scalesK4 处理 Q4_K / Q5_K：d(2) + dmin(2) + scales(12) + …，每块 8 个子块。
//
// 子块的 sc/m 是 6 位字段，解包方式与解码时完全一致（复用 getScaleMinK4）——
// 那份布局已在 ③a 与 llama.cpp 逐值比对过。
func scalesK4(src []byte, nBlocks, perBlock int64) []SubScale {
	out := make([]SubScale, 0, nBlocks*8)
	for b := int64(0); b < nBlocks; b++ {
		blk := src[b*perBlock:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		scales := blk[4:16]
		for j := range 8 {
			sc, m := getScaleMinK4(j, scales)
			out = append(out, SubScale{
				Scale: d * float32(sc),
				Min:   dmin * float32(m),
				Elems: 32,
			})
		}
	}
	return out
}

// scalesQ2K：scales(16) + qs(64) + d(2) + dmin(2)，每块 16 个子块、每个 16 个权重。
// 每个 scale 字节的低 4 位是 sc、高 4 位是 m。
func scalesQ2K(src []byte, nBlocks, perBlock int64) []SubScale {
	out := make([]SubScale, 0, nBlocks*16)
	for b := int64(0); b < nBlocks; b++ {
		blk := src[b*perBlock:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[80:]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[82:]))
		for i := range 16 {
			sc := blk[i]
			out = append(out, SubScale{
				Scale: d * float32(sc&0x0F),
				Min:   dmin * float32(sc>>4),
				Elems: 16,
			})
		}
	}
	return out
}

// scalesQ3K：hmask(32) + qs(64) + scales(12) + d(2)，每块 16 个子块、每个 16 个权重。
//
// 6 位 scale 的解包**复用解码时那份**（unpackQ3KScales）——
// 这套打包方式极其反直觉，凡是同一份布局只用一份实现。
func scalesQ3K(src []byte, nBlocks, perBlock int64) []SubScale {
	out := make([]SubScale, 0, nBlocks*16)
	for b := int64(0); b < nBlocks; b++ {
		blk := src[b*perBlock:]
		dAll := f16ToF32(binary.LittleEndian.Uint16(blk[108:]))
		sc := unpackQ3KScales(blk[96:108])
		for j := range 16 {
			out = append(out, SubScale{
				Scale: dAll * float32(sc[j]),
				Elems: 16,
			})
		}
	}
	return out
}

// scalesQ6K：ql(128) + qh(64) + scales(16 个 int8) + d(2)。
// 注意 **d 在块尾**（偏移 208）。16 个子块、每个 16 个权重。
func scalesQ6K(src []byte, nBlocks, perBlock int64) []SubScale {
	out := make([]SubScale, 0, nBlocks*16)
	for b := int64(0); b < nBlocks; b++ {
		blk := src[b*perBlock:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk[208:]))
		for i := range 16 {
			out = append(out, SubScale{
				Scale: d * float32(int8(blk[192+i])),
				Elems: 16,
			})
		}
	}
	return out
}
