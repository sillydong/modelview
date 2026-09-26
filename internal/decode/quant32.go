package decode

import "encoding/binary"

// Q4_0：每块 32 个权重，18 字节 = d(2) + qs(16)。
//
// 顺序容易写错：qs 的第 j 个字节的低半字节是元素 j，高半字节是元素 j+16，
// **不是**元素 2j 和 2j+1。写成后者的活，字节数、张量数、参数总量全都对，
// 只有数值是错的 —— 所以必须靠逐值比对才发现。
func dqQ4_0(src []byte, dst []float32) error {
	const perBlock, elems = 18, 32
	for b := 0; b*elems < len(dst); b++ {
		blk := src[b*perBlock:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		out := dst[b*elems : (b+1)*elems]
		for j := range 16 {
			out[j] = float32(int8(blk[2+j]&0x0F)-8) * d
			out[j+16] = float32(int8(blk[2+j]>>4)-8) * d
		}
	}
	return nil
}

// Q4_1：20 字节 = d(2) + m(2) + qs(16)。与 Q4_0 的区别是没有 -8 偏移，改为 +m。
func dqQ4_1(src []byte, dst []float32) error {
	const perBlock, elems = 20, 32
	for b := 0; b*elems < len(dst); b++ {
		blk := src[b*perBlock:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		m := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		out := dst[b*elems : (b+1)*elems]
		for j := range 16 {
			out[j] = float32(blk[4+j]&0x0F)*d + m
			out[j+16] = float32(blk[4+j]>>4)*d + m
		}
	}
	return nil
}

// Q5_0：22 字节 = d(2) + qh(4) + qs(16)。第 5 位来自 qh 这个 u32 的各个比特。
//
// 两处易错：
//   - 元素 i 用的是 qh 的**第 i 位**（i 取 0..31），不是"低半字节用低 16 位"之类的分段
//   - 取出比特后要 `<< 4` 移到第 5 位的位置再做或运算；
//     漏掉 `<< 4` 会让**一半的值**错掉，而另一半是对的
func dqQ5_0(src []byte, dst []float32) error {
	const perBlock, elems = 22, 32
	for b := 0; b*elems < len(dst); b++ {
		blk := src[b*perBlock:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		qh := binary.LittleEndian.Uint32(blk[2:])
		out := dst[b*elems : (b+1)*elems]
		for j := range 16 {
			xh0 := ((qh >> uint(j)) << 4) & 0x10
			xh1 := ((qh >> uint(j+16)) << 4) & 0x10
			out[j] = float32(int8((blk[6+j]&0x0F)|uint8(xh0))-16) * d
			out[j+16] = float32(int8((blk[6+j]>>4)|uint8(xh1))-16) * d
		}
	}
	return nil
}

// Q5_1：24 字节 = d(2) + m(2) + qh(4) + qs(16)。与 Q5_0 同构，去掉 -16 改为 +m。
func dqQ5_1(src []byte, dst []float32) error {
	const perBlock, elems = 24, 32
	for b := 0; b*elems < len(dst); b++ {
		blk := src[b*perBlock:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		m := f16ToF32(binary.LittleEndian.Uint16(blk[2:]))
		qh := binary.LittleEndian.Uint32(blk[4:])
		out := dst[b*elems : (b+1)*elems]
		for j := range 16 {
			xh0 := ((qh >> uint(j)) << 4) & 0x10
			xh1 := ((qh >> uint(j+16)) << 4) & 0x10
			out[j] = float32(uint8(blk[8+j]&0x0F)|uint8(xh0))*d + m
			out[j+16] = float32(uint8(blk[8+j]>>4)|uint8(xh1))*d + m
		}
	}
	return nil
}

// Q8_0：34 字节 = d(2) + qs(32 个 int8)。唯一一个元素顺序线性的类型。
func dqQ8_0(src []byte, dst []float32) error {
	const perBlock, elems = 34, 32
	for b := 0; b*elems < len(dst); b++ {
		blk := src[b*perBlock:]
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		out := dst[b*elems : (b+1)*elems]
		for j := range 32 {
			out[j] = float32(int8(blk[2+j])) * d
		}
	}
	return nil
}
