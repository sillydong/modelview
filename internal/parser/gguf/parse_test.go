package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

// buildMinimalGGUF 拼一个完整的最小 GGUF：
// 3 条元数据（含 alignment）+ 1 个 F32 张量 + 数据区。
func buildMinimalGGUF() []byte {
	b := newBuilder()
	b.header(3, 1, 3)
	b.kv("general.architecture", typeString).str("test-arch")
	b.kv("general.alignment", typeUint32).u32(32)
	b.kv("t.vec", typeArray).u32(typeUint32).u64(3).u32(1).u32(2).u32(3)

	// 张量描述符：形状 [4]，F32，偏移 0
	b.str("w")
	b.u32(1)
	b.u64(4)
	b.u32(0) // F32
	b.u64(0)

	// 对齐填充
	hdrLen := int64(len(b.bytes()))
	pad := alignUp(hdrLen, 32) - hdrLen
	b.raw(make([]byte, pad))

	// 数据区：4 个 float32 = 1.0 2.0 3.0 4.0
	b.f32(1).f32(2).f32(3).f32(4)
	return b.bytes()
}

func writeFile(t *testing.T, name string, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatalf("写文件失败: %v", err)
	}
	return p
}

func TestParse_最小文件(t *testing.T) {
	p := writeFile(t, "min.gguf", buildMinimalGGUF())

	m, err := Parse(p)
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}

	if m.Format != model.FormatGGUF {
		t.Errorf("Format = %q, want %q", m.Format, model.FormatGGUF)
	}
	if m.Version != "v3" {
		t.Errorf("Version = %q, want v3", m.Version)
	}
	if m.Arch != "test-arch" {
		t.Errorf("Arch = %q, want test-arch", m.Arch)
	}
	if len(m.Metadata) != 3 {
		t.Fatalf("元数据条数 = %d, want 3", len(m.Metadata))
	}
	if m.Metadata[0].Key != "general.architecture" || m.Metadata[0].Value != "test-arch" {
		t.Errorf("metadata[0] = %+v", m.Metadata[0])
	}
	if m.Metadata[2].Value != "[1, 2, 3]" {
		t.Errorf("metadata[2].Value = %q, want [1, 2, 3]", m.Metadata[2].Value)
	}

	if len(m.Tensors) != 1 {
		t.Fatalf("张量个数 = %d, want 1", len(m.Tensors))
	}
	tn := m.Tensors[0]
	if tn.Name != "w" {
		t.Errorf("名称 = %q", tn.Name)
	}
	if tn.ParamCount != 4 {
		t.Errorf("元素数 = %d, want 4", tn.ParamCount)
	}
	if tn.ByteSize != 16 {
		t.Errorf("字节数 = %d, want 16", tn.ByteSize)
	}
	if tn.Dtype != model.DtypeF32 {
		t.Errorf("类型 = %q, want F32", tn.Dtype)
	}
	if got := m.TotalParams(); got != 4 {
		t.Errorf("TotalParams = %d, want 4", got)
	}
}

// 数据区偏移必须正确 —— 张量描述符结束位置向上对齐后才是数据起点。
func TestParse_数据区偏移正确(t *testing.T) {
	p := writeFile(t, "min.gguf", buildMinimalGGUF())

	m, err := Parse(p)
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}
	if m.Tensors[0].Offset%32 != 0 {
		t.Errorf("偏移 %d 未按 32 对齐", m.Tensors[0].Offset)
	}

	f, err := os.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	//nolint:errcheck // 只读文件，Close 失败不影响测试
	defer f.Close()

	buf := make([]byte, 16)
	if _, err := f.ReadAt(buf, m.Tensors[0].Offset); err != nil {
		t.Fatalf("按偏移读取失败: %v", err)
	}
	want := []float32{1, 2, 3, 4}
	for i := range want {
		got := math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		if got != want[i] {
			t.Errorf("第 %d 个值 = %v, want %v", i, got, want[i])
		}
	}
}

// 非默认 alignment 必须被采纳。
func TestParse_自定义对齐(t *testing.T) {
	b := newBuilder()
	b.header(3, 1, 1)
	b.kv("general.alignment", typeUint32).u32(64)
	b.str("w")
	b.u32(1)
	b.u64(4)
	b.u32(0)
	b.u64(0)
	hdrLen := int64(len(b.bytes()))
	b.raw(make([]byte, alignUp(hdrLen, 64)-hdrLen))
	b.f32(1).f32(2).f32(3).f32(4)

	p := writeFile(t, "align64.gguf", b.bytes())
	m, err := Parse(p)
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}
	if got := m.Alignment; got != 64 {
		t.Errorf("Alignment = %d, want 64", got)
	}
	if m.Tensors[0].Offset%64 != 0 {
		t.Errorf("偏移 %d 未按 64 对齐", m.Tensors[0].Offset)
	}
}

func TestParse_非GGUF报错(t *testing.T) {
	p := writeFile(t, "not.gguf", []byte("NOPE12345678"))
	if _, err := Parse(p); err == nil {
		t.Fatal("非 GGUF 文件应报错")
	}
}

func TestParse_文件不存在报错(t *testing.T) {
	if _, err := Parse(filepath.Join(t.TempDir(), "missing.gguf")); err == nil {
		t.Fatal("不存在的文件应报错")
	}
}

func TestParse_不支持的版本报错(t *testing.T) {
	b := newBuilder()
	b.header(1, 0, 0) // v1
	p := writeFile(t, "v1.gguf", b.bytes())
	if _, err := Parse(p); err == nil {
		t.Fatal("GGUF v1 应报错")
	}
}

// 算不出大小的张量必须产生告警，且带 SizeUnknown 标记。
// 静默给 0 会让用户以为"这个张量真的是 0 字节"。
func TestParse_未收录类型产生告警(t *testing.T) {
	b := newBuilder()
	b.header(3, 1, 0)
	b.str("iq_tensor")
	b.u32(1)
	b.u64(256)
	b.u32(16) // IQ2_XXS，块表未收录
	b.u64(0)
	b.raw(make([]byte, alignUp(int64(len(b.bytes())), 32)-int64(len(b.bytes()))))

	m, err := Parse(writeFile(t, "iq.gguf", b.bytes()))
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}
	if len(m.Warnings) == 0 {
		t.Fatal("未收录类型必须产生告警，实际没有")
	}
	if !strings.Contains(m.Warnings[0], "iq_tensor") {
		t.Errorf("告警应指明是哪个张量，实际: %q", m.Warnings[0])
	}
	if !m.Tensors[0].SizeUnknown {
		t.Error("SizeUnknown 应为 true —— 否则 ByteSize==0 会被读成真实值")
	}
}

// alignment 取值异常必须告警，不能静默退回默认值 ——
// 否则所有张量 offset 整片偏移而界面看不出异常。
func TestParse_alignment异常告警(t *testing.T) {
	tests := []struct {
		name  string
		value func(b *builder)
	}{
		{"alignment 为 0", func(b *builder) { b.kv("general.alignment", typeUint32).u32(0) }},
		{"alignment 类型错误（字符串）", func(b *builder) { b.kv("general.alignment", typeString).str("32") }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := newBuilder()
			b.header(3, 0, 1)
			tt.value(b)

			m, err := Parse(writeFile(t, "x.gguf", b.bytes()))
			if err != nil {
				t.Fatalf("Parse 失败: %v", err)
			}
			found := false
			for _, w := range m.Warnings {
				if strings.Contains(w, "alignment") {
					found = true
				}
			}
			if !found {
				t.Errorf("应产生 alignment 告警，实际告警: %v", m.Warnings)
			}
		})
	}
}

// 正常文件不应产生任何告警 —— 否则告警就失去信号价值。
func TestParse_正常文件无告警(t *testing.T) {
	m, err := Parse(writeFile(t, "min.gguf", buildMinimalGGUF()))
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}
	if len(m.Warnings) != 0 {
		t.Errorf("正常文件不该有告警，实际: %v", m.Warnings)
	}
}

// 未收录类型（IQ 系列）不应导致整体解析失败，只是字节数为 0。
func TestParse_未收录类型不致命(t *testing.T) {
	b := newBuilder()
	b.header(3, 1, 0)
	b.str("iq_tensor")
	b.u32(1)
	b.u64(256)
	b.u32(16) // IQ2_XXS，块表未收录
	b.u64(0)
	hdrLen := int64(len(b.bytes()))
	b.raw(make([]byte, alignUp(hdrLen, 32)-hdrLen))

	p := writeFile(t, "iq.gguf", b.bytes())
	m, err := Parse(p)
	if err != nil {
		t.Fatalf("未收录类型不应导致失败: %v", err)
	}
	if len(m.Tensors) != 1 {
		t.Fatalf("张量个数 = %d, want 1", len(m.Tensors))
	}
	if m.Tensors[0].ByteSize != 0 {
		t.Errorf("未收录类型的字节数应为 0，实际 %d", m.Tensors[0].ByteSize)
	}
	if m.Tensors[0].Dtype != model.DtypeIQ2XXS {
		t.Errorf("类型 = %q, want IQ2_XXS", m.Tensors[0].Dtype)
	}
	if m.Tensors[0].ParamCount != 256 {
		t.Errorf("元素数 = %d, want 256", m.Tensors[0].ParamCount)
	}
}
