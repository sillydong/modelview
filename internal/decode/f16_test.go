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
