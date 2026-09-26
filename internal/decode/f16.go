package decode

import "math"

// f16ToF32 把 IEEE 半精度位模式转成 float32。
//
// 纯位运算，不依赖任何库。三条分支分别是零/次正规、正规、无穷与 NaN。
func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	switch {
	case exp == 0 && mant == 0:
		// ±0
		return math.Float32frombits(sign << 31)
	case exp == 0:
		// 次正规数：把它规格化成 float32 的正规数。
		// 指数从 1 - 15 起算，尾数每左移一位指数减一。
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3FF
		return math.Float32frombits(sign<<31 | e<<23 | mant<<13)
	case exp == 0x1F:
		// 无穷或 NaN：指数全 1，尾数原样搬过去（保留 NaN 的有效载荷）
		return math.Float32frombits(sign<<31 | 0xFF<<23 | mant<<13)
	default:
		return math.Float32frombits(sign<<31 | (exp+127-15)<<23 | mant<<13)
	}
}
