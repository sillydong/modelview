package detect

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

// writeTemp 写一个临时文件并返回其路径。
func writeTemp(t *testing.T, name string, b []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatalf("写临时文件失败: %v", err)
	}
	return p
}

func TestProbe(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want model.Format
	}{
		{
			name: "GGUF magic",
			data: []byte{'G', 'G', 'U', 'F', 3, 0, 0, 0},
			want: model.FormatGGUF,
		},
		{
			name: "safetensors: 小头部长度后跟 JSON 左括号",
			// 头部长度 2（小端 u64），随后是 "{"
			data: []byte{2, 0, 0, 0, 0, 0, 0, 0, '{', '}'},
			want: model.FormatSafeTensors,
		},
		{
			name: "PyTorch: ZIP magic",
			data: []byte{'P', 'K', 3, 4, 0, 0, 0, 0},
			want: model.FormatPyTorch,
		},
		{
			name: "未知: 随机字节",
			data: []byte{0xde, 0xad, 0xbe, 0xef, 0x00, 0x11, 0x22, 0x33},
			want: model.FormatUnknown,
		},
		{
			name: "未知: 头部长度过大（不是 safetensors）",
			data: []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, '{', '}'},
			want: model.FormatUnknown,
		},
		{
			name: "空文件",
			data: nil,
			want: model.FormatUnknown,
		},
		{
			name: "不足 8 字节",
			data: []byte{'G', 'G'},
			want: model.FormatUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := writeTemp(t, "f.bin", tt.data)
			got, err := Probe(p)
			if err != nil {
				t.Fatalf("Probe 返回错误: %v", err)
			}
			if got != tt.want {
				t.Errorf("Probe = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestProbe_不存在的文件返回错误(t *testing.T) {
	_, err := Probe(filepath.Join(t.TempDir(), "不存在"))
	if err == nil {
		t.Fatal("期望报错，实际为 nil")
	}
}

// 扩展名不影响判断：一个 .bin 文件内容是 ZIP，应判为 PyTorch。
func TestProbe_不看扩展名(t *testing.T) {
	p := writeTemp(t, "looks-like-model.bin", []byte{'P', 'K', 3, 4, 0, 0, 0, 0})
	got, err := Probe(p)
	if err != nil {
		t.Fatalf("Probe 返回错误: %v", err)
	}
	if got != model.FormatPyTorch {
		t.Errorf("Probe = %q, want %q", got, model.FormatPyTorch)
	}
}
