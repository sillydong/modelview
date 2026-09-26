package parser

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
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

func TestParse_未知格式返回ErrUnsupported(t *testing.T) {
	_, err := Parse(writeTemp(t, []byte{0xde, 0xad, 0xbe, 0xef, 1, 2, 3, 4}))
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("err = %v, 期望 ErrUnsupported", err)
	}
}

// 三种格式的分支都必须真的被走到，而且必须走到**对的那个**。
//
// 只判"有错误"是空转的：格式没接线会通过，把两个分支对调也会通过。
// 所以每条都钉住一个只可能由该格式包产出的错误片段。
func TestParse_三种格式分支都被走到(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		// want 是该格式独有的出错环节，对调分支后必然不匹配
		want string
	}{
		{
			"GGUF",
			[]byte{'G', 'G', 'U', 'F', 3, 0, 0, 0},
			"读取张量个数",
		},
		{
			// 前 8 字节是头部长度 8，第 9 字节是 '{' 所以能通过 detect，
			// 但头部本身是非法 JSON（尾随逗号）——错必须来自 safetensors.Parse
			"SafeTensors",
			append([]byte{8, 0, 0, 0, 0, 0, 0, 0}, []byte(`{"a":1,}`)...),
			"解析头部 JSON",
		},
		{
			// ZIP magic 正确但后面不是合法 ZIP
			"PyTorch",
			[]byte{'P', 'K', 3, 4, 0, 0, 0, 0},
			"打开 ZIP",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse(writeTemp(t, tt.data))
			if err == nil {
				t.Fatal("截断/伪造的内容应报错")
			}
			if errors.Is(err, ErrUnsupported) {
				t.Fatalf("%s 已实现，不该返回 ErrUnsupported: %v", tt.name, err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("错误应来自 %s 解析器的 %q 环节，实际: %v", tt.name, tt.want, err)
			}
		})
	}
}

func TestParse_文件不存在(t *testing.T) {
	_, err := Parse(filepath.Join(t.TempDir(), "missing"))
	if err == nil {
		t.Fatal("不存在的文件应报错")
	}
}
