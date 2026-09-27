package decode

import (
	"math"
	"testing"
)

func TestF16ToF32(t *testing.T) {
	tests := []struct {
		name string
		bits uint16
		want float32
	}{
		{"零", 0x0000, 0},
		{"一", 0x3C00, 1},
		{"负一", 0xBC00, -1},
		{"0.0625", 0x2C00, 0.0625}, // 交叉验证脚本里所有 scale 用的就是这个值
		{"65504 最大值", 0x7BFF, 65504},
		{"正无穷", 0x7C00, float32(math.Inf(1))},
		{"负无穷", 0xFC00, float32(math.Inf(-1))},
		{"次正规数 2^-24", 0x0001, float32(math.Ldexp(1, -24))},
		{"次正规数最大值", 0x03FF, float32(1023.0 / 16777216.0)}, // 1023 × 2^-24
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := f16ToF32(tt.bits); got != tt.want {
				t.Errorf("f16ToF32(0x%04X) = %v, want %v", tt.bits, got, tt.want)
			}
		})
	}
}

// 负零的符号位必须保留 —— 丢掉它会让 copysign 之类的判断失效。
func TestF16ToF32_负零(t *testing.T) {
	got := f16ToF32(0x8000)
	if got != 0 {
		t.Errorf("f16ToF32(0x8000) = %v, want 0", got)
	}
	if !math.Signbit(float64(got)) {
		t.Error("负零丢了符号位")
	}
}

// NaN 必须仍然是 NaN（位模式可以不同，但 IsNaN 必须成立）。
func TestF16ToF32_NaN(t *testing.T) {
	for _, bits := range []uint16{0x7E00, 0x7C01, 0xFE00} {
		if got := f16ToF32(bits); !math.IsNaN(float64(got)) {
			t.Errorf("f16ToF32(0x%04X) = %v, 应为 NaN", bits, got)
		}
	}
}

// F32ToF16 的往返：能从 fp16 精确表示的值，来回转必须回到原位。
func TestF32ToF16_往返(t *testing.T) {
	for _, h := range []uint16{0x0000, 0x3C00, 0xBC00, 0x2C00, 0x7BFF, 0x0001, 0x03FF} {
		f := f16ToF32(h)
		if got := F32ToF16(f); got != h {
			t.Errorf("f16ToF32(0x%04X)=%v 再转回 = 0x%04X, want 0x%04X", h, f, got, h)
		}
	}
}

// F32ToF16 的边界：进位、次正规、溢出、零。
//
// 这几类正是写错时唯一会出问题的位置，其余绝大多数值随便怎么写都对。
func TestF32ToF16_边界(t *testing.T) {
	tests := []struct {
		name string
		in   float32
		want uint16
	}{
		{"零", 0, 0x0000},
		{"负零", float32(math.Copysign(0, -1)), 0x8000},
		{"一", 1, 0x3C00},
		{"负一", -1, 0xBC00},
		{"0.0625", 0.0625, 0x2C00},
		{"fp16 最大值", 65504, 0x7BFF},
		{"上溢", 1e10, 0x7C00},
		{"负上溢", -1e10, 0xFC00},
		{"正无穷", float32(math.Inf(1)), 0x7C00},
		{"负无穷", float32(math.Inf(-1)), 0xFC00},
		{"最小次正规数", float32(math.Ldexp(1, -24)), 0x0001},
		{"半个最小次正规数 → 0", float32(math.Ldexp(1, -25)), 0x0000},
		{"次正规数最大值", float32(1023.0 / 16777216.0), 0x03FF},
		// 尾数进位溢出到指数：0x7BFF 是 fp16 最大有限值，
		// 比它大一点点的 float32 应当进位到无穷而不是回绕到 0x0000
		{"刚好超过最大值", float32(65520), 0x7C00},
		// 平局取偶。0x3C00 = 1.0，0x3C01 = 1 + 2^-10，
		// 两者的中点是 1 + 2^-11（float32 可精确表示）—— 尾数最低位为偶，保留 0x3C00。
		{"平局取偶（保留偶数）", float32(1.0 + 1.0/2048.0), 0x3C00},
		// 0x3C01 = 1 + 2^-10（尾数 1，奇），0x3C02 = 1 + 2^-9（尾数 2，偶），
		// 中点是 1 + 1.5*2^-10 —— 进位到偶数 0x3C02。
		{"平局取偶（进位到偶数）", float32(1.0 + 1.5/1024.0), 0x3C02},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := F32ToF16(tt.in); got != tt.want {
				t.Errorf("F32ToF16(%v) = 0x%04X, want 0x%04X", tt.in, got, tt.want)
			}
		})
	}
	// NaN：必须仍是 NaN，不能变成无穷（丢掉有效载荷就变成无穷了）
	for _, in := range []float32{float32(math.NaN()), -float32(math.NaN())} {
		got := F32ToF16(in)
		if got&0x7C00 != 0x7C00 || got&0x3FF == 0 {
			t.Errorf("F32ToF16(NaN) = 0x%04X，应转成 fp16 的 NaN 而不是无穷", got)
		}
	}
}
