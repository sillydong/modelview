package decode

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

func TestDecode_非量化类型(t *testing.T) {
	tests := []struct {
		name string
		d    model.Dtype
		src  []byte
		want []float32
	}{
		{"F32", model.DtypeF32, le32(1.5, -2.25), []float32{1.5, -2.25}},
		{"F16", model.DtypeF16, []byte{0x00, 0x3C, 0x00, 0xBC}, []float32{1, -1}},
		// bf16 就是 float32 的高 16 位
		{"BF16", model.DtypeBF16, []byte{0x80, 0x3F, 0x00, 0xC0}, []float32{1, -2}},
		{"I8", model.DtypeI8, []byte{0x7F, 0x80, 0xFF}, []float32{127, -128, -1}},
		{"U8", model.DtypeU8, []byte{0, 255}, []float32{0, 255}},
		{"I16", model.DtypeI16, []byte{0xFF, 0x7F, 0x00, 0x80}, []float32{32767, -32768}},
		{"BOOL", model.DtypeBool, []byte{1, 0, 2}, []float32{1, 0, 1}}, // 非零即真
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			if err := Decode(tt.d, tt.src, dst); err != nil {
				t.Fatalf("Decode 失败: %v", err)
			}
			for i := range dst {
				if dst[i] != tt.want[i] {
					t.Errorf("[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestDecode_F64与I32(t *testing.T) {
	src := make([]byte, 0, 12)
	src = binary.LittleEndian.AppendUint64(src, math.Float64bits(-3.5))
	// 必须经变量转换：常量表达式里的 uint32(int32(-7)) 会被 Go 判为溢出
	var neg int32 = -7
	src = binary.LittleEndian.AppendUint32(src, uint32(neg))
	got := make([]float32, 2)

	if err := Decode(model.DtypeF64, src[:8], got[:1]); err != nil {
		t.Fatal(err)
	}
	if got[0] != -3.5 {
		t.Errorf("F64 = %v, want -3.5", got[0])
	}
	if err := Decode(model.DtypeI32, src[8:], got[1:]); err != nil {
		t.Fatal(err)
	}
	if got[1] != -7 {
		t.Errorf("I32 = %v, want -7", got[1])
	}
}

// 字节数不足时必须报错，不能读越界也不能静默给 0。
func TestDecode_字节不足报错(t *testing.T) {
	dst := make([]float32, 4)
	if err := Decode(model.DtypeF32, make([]byte, 8), dst); err == nil {
		t.Fatal("字节不足应报错")
	}
}

// 未收录的类型必须报错，而不是返回一堆 0。
//
// FP8 与 IQ/Q8_K 都走这条路：它们的布局本项目没有可用语料或独立参照验证，
// 给数字不如明确说"不知道"。
func TestDecode_未收录类型报错(t *testing.T) {
	dst := make([]float32, 256)
	for _, d := range []model.Dtype{
		model.DtypeIQ2XXS, model.DtypeQ8K, model.DtypeQ8_1,
		model.DtypeF8E4M3, model.DtypeF8E5M2, model.DtypeUnknown,
	} {
		if err := Decode(d, make([]byte, 4096), dst); err == nil {
			t.Errorf("%s 未收录块结构，应报错", d)
		}
	}
}

// 零长度目标不该报错（空张量是合法的）。
//
// 也要覆盖**未收录类型**配空目标：没有元素要解，就不该报"类型不支持"。
// 少了这条断言，Decode 开头那句 len(dst) == 0 提前返回是可以删掉而不被发现的。
func TestDecode_零长度(t *testing.T) {
	for _, d := range []model.Dtype{
		model.DtypeF32, model.DtypeF16, model.DtypeQ4K,
		model.DtypeIQ2XXS, model.DtypeF8E4M3, model.DtypeUnknown,
	} {
		if err := Decode(d, nil, nil); err != nil {
			t.Errorf("空解码（%s）不该报错: %v", d, err)
		}
	}
}

func le32(vs ...float32) []byte {
	out := make([]byte, 0, 4*len(vs))
	for _, v := range vs {
		out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
	}
	return out
}

// Decode 必须写满 dst —— 调用方可能复用它。
//
// 这条不是假想的：internal/analyze 的 decoded 缓冲就是跨分块复用的，
// 只在"条件成立时赋值"的实现在跨块时会让元素保留上一轮的值。
// 实测 BOOL 分支曾漏写 0，8,388,609 个元素、只有一个 0 的张量算出了
// zero_ratio=0、min=1 —— 那个 0 完全不可见。
//
// 测试传一个**预先填满非零值**的 dst：不写满的实现会把这些残留值
// 当成结果返回。
func TestDecode_必须写满目标缓冲(t *testing.T) {
	tests := []struct {
		name string
		d    model.Dtype
		src  []byte
		want []float32
	}{
		{"BOOL 的 false 必须写成 0", model.DtypeBool,
			[]byte{1, 0, 1, 0}, []float32{1, 0, 1, 0}},
		{"F32", model.DtypeF32, le32(1, 0), []float32{1, 0}},
		{"F16", model.DtypeF16, []byte{0x00, 0x3C, 0x00, 0x00}, []float32{1, 0}},
		{"I8", model.DtypeI8, []byte{1, 0}, []float32{1, 0}},
		{"U8", model.DtypeU8, []byte{1, 0}, []float32{1, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			for i := range dst {
				dst[i] = 999 // 残留值：写满的实现在结果里看不到它
			}
			if err := Decode(tt.d, tt.src, dst); err != nil {
				t.Fatalf("Decode 失败: %v", err)
			}
			for i := range dst {
				if dst[i] != tt.want[i] {
					t.Errorf("[%d] = %v, want %v —— 该位置没被写入", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

// 无符号类型必须按无符号解，64 位类型必须做符号扩展。
//
// 这四种类型本地语料里一个都没有（GGUF 里只用到 I64；U16/U32/U64 只在
// safetensors 里可能出现），所以真实文件回归盖不住 —— 必须靠构造用例。
// 覆盖之前，把这四个分支的赋值改成 dst[i] = -1 也不会有任何测试变红。
//
// 样本值都刻意选在"有符号/无符号解释不同"的位置：
// 0xFFFF 无符号是 65535、有符号是 -1；差一个符号就看得出。
func TestDecode_无符号与64位符号扩展(t *testing.T) {
	tests := []struct {
		name string
		d    model.Dtype
		src  []byte
		want []float32
	}{
		{"U16 不能被当成有符号", model.DtypeU16,
			[]byte{0xFF, 0xFF, 0x00, 0x00}, []float32{65535, 0}},
		{"U16 最大值", model.DtypeU16,
			[]byte{0xFF, 0xFF}, []float32{65535}},
		{"U32 不能被当成有符号", model.DtypeU32,
			[]byte{0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00}, []float32{4294967295, 1}},
		{"U64 不能被当成有符号", model.DtypeU64,
			binary.LittleEndian.AppendUint64(nil, ^uint64(0)), []float32{18446744073709551615}},
		{"I16 最小值", model.DtypeI16, []byte{0x00, 0x80}, []float32{-32768}},
		{"I64 负一要符号扩展", model.DtypeI64,
			binary.LittleEndian.AppendUint64(nil, uint64(0xFFFFFFFFFFFFFFFF)), []float32{-1}},
		{"I64 不能被当成无符号", model.DtypeI64,
			binary.LittleEndian.AppendUint64(nil, uint64(0x8000000000000000)),
			[]float32{float32(-9223372036854775808)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			if err := Decode(tt.d, tt.src, dst); err != nil {
				t.Fatalf("Decode 失败: %v", err)
			}
			for i := range dst {
				if dst[i] != tt.want[i] {
					t.Errorf("[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}
