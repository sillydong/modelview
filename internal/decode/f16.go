package decode

import "math"

// F16ToF32 把 IEEE 半精度位模式转成 float32。
//
// 导出是为了让 internal/analyze 的量化模拟用同一份实现 —— 写两份会漂移。
func F16ToF32(h uint16) float32 { return f16ToF32(h) }

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

// F32ToF16 把 float32 舍入成最接近的 IEEE 半精度位模式（平局取偶）。
//
// 量化编码要用它：块头的 scale 是 fp16，写入时的舍入与从 fp16 读回来的
// 值必须是同一个 —— 用未舍入的原值继续算码，编出来的字节就对不上真实文件。
//
// 浮点转换写错时症状极其隐蔽：绝大多数值都能对上，只有进位、次正规、
// 溢出这几类边界会错，而它们恰好是 scale 最容易出现的量级。
func F32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32(bits>>23&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF

	switch {
	case bits>>23&0xFF == 0xFF: // Inf / NaN
		if mant != 0 {
			// NaN：保留有效载荷的高位，并置上 quiet 位
			return sign | 0x7E00 | uint16(mant>>13)
		}
		return sign | 0x7C00
	case exp >= 0x1F: // 上溢到无穷
		return sign | 0x7C00
	case exp <= 0: // 次正规数或下溢
		if exp < -10 {
			return sign // 小到连次正规数都表示不了
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := uint32(1) << (shift - 1)
		m := mant >> shift
		// 平局取偶：恰好一半时看向最低位的奇偶
		if mant&half != 0 && (mant&(half-1) != 0 || m&1 != 0) {
			m++
		}
		return sign | uint16(m)
	default:
		m := mant >> 13
		rem := mant & 0x1FFF
		if rem > 0x1000 || (rem == 0x1000 && m&1 != 0) {
			m++
			if m == 0x400 { // 尾数进位溢出到指数
				m = 0
				exp++
				if exp >= 0x1F {
					return sign | 0x7C00
				}
			}
		}
		return sign | uint16(exp)<<10 | uint16(m)
	}
}
