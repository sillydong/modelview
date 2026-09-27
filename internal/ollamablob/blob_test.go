package ollamablob

import (
	"strings"
	"testing"
)

func TestIsInProgressBlob_边界(t *testing.T) {
	hex64 := strings.Repeat("a1", 32)
	tests := []struct {
		name string
		want bool
	}{
		{"sha256-" + hex64, false},                                // 下完了
		{"sha256-" + hex64 + "-partial", true},                    // 预分配的目标文件
		{"sha256-" + hex64 + "-partial-0", true},                  // 分片
		{"sha256-" + hex64 + "-partial-12", true},                 // 分片
		{"sha256-abc", false},                                     // 短名字，不像 ollama 的 blob
		{"sha256-" + hex64 + "-partialx", true},                   // 前缀匹配 -partial，仍是下载中
		{"sha256-" + hex64 + "x-partial", false},                  // 64 位之后不是十六进制 → 不是我们的形状
		{"sha256-partial", false},                                 // 长度不够
		{"other-" + hex64 + "-partial", false},                    // 不是 sha256- 前缀
		{"sha256-" + strings.Repeat("z", 64) + "-partial", false}, // 非十六进制
	}
	for _, tt := range tests {
		if got := IsInProgress(tt.name); got != tt.want {
			t.Errorf("IsInProgress(%q) = %v, want %v", tt.name, got, tt.want)
		}
	}
}
