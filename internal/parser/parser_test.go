package parser

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func writeTemp(t *testing.T, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "f")
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestParse_未实现格式返回ErrUnsupported(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"safetensors 头", []byte{2, 0, 0, 0, 0, 0, 0, 0, '{', '}'}},
		{"pytorch zip", []byte{'P', 'K', 3, 4, 0, 0, 0, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse(writeTemp(t, tt.data))
			if !errors.Is(err, ErrUnsupported) {
				t.Fatalf("err = %v, 期望能 errors.Is(err, ErrUnsupported)", err)
			}
		})
	}
}

func TestParse_未知格式返回ErrUnsupported(t *testing.T) {
	_, err := Parse(writeTemp(t, []byte{0xde, 0xad, 0xbe, 0xef, 1, 2, 3, 4}))
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("err = %v, 期望 ErrUnsupported", err)
	}
}

// GGUF 分支必须真的走到 gguf.Parse —— 传一个 magic 正确但内容截断的文件，
// 错误信息应来自 gguf 包而不是 "无法识别"。
func TestParse_GGUF分支被走到(t *testing.T) {
	_, err := Parse(writeTemp(t, []byte{'G', 'G', 'U', 'F', 3, 0, 0, 0}))
	if err == nil {
		t.Fatal("截断的 GGUF 应报错")
	}
	if errors.Is(err, ErrUnsupported) {
		t.Fatalf("应走 gguf.Parse 分支，实际返回了 ErrUnsupported: %v", err)
	}
}

func TestParse_文件不存在(t *testing.T) {
	_, err := Parse(filepath.Join(t.TempDir(), "missing"))
	if err == nil {
		t.Fatal("不存在的文件应报错")
	}
}
