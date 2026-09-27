// Package ollamablob 认识 ollama 在磁盘上的 blob 命名。
//
// 单独成一个包，是因为"这个文件是不是还没下完"这件事**有两处以上要用**：
//   - `internal/discover` 判定孤儿 blob 时要排除它们
//   - `internal/parser/gguf` 的测试在语料目录里找真实文件时要跳过它们
//   - 测试的语料定位、tools/ 下的脚本同理
//
// 而 `discover` 依赖 `parser`（`discover → parser → parser/gguf`），
// 所以那些地方**不能** import `discover` —— 当初的应对是在每处抄一份，
// 于是同一个判据有了 5 份，其中一份的注释还声称"判据相同"而实际更宽
// （`Contains("-partial")` 会把 `sha256-abc-partial` 这种短名字也算成下载中，
// 而正本要求恰好 64 位十六进制）。
//
// 这个包**不 import 任何内部包**（叶子包），所以谁都能用，抄本也就不需要了。
package ollamablob

import "strings"

// digestHexLen 是 sha256 十六进制的长度。
const digestHexLen = 64

// IsInProgress 判断一个 blob 文件名是不是"还没下完"。
//
// ollama 下载时先建 `<name>-partial`（预分配到最终大小，所以**看着是满的**），
// 另有一批 `<name>-partial-<序号>` 分片。它们不是 blob，是下载的中间状态。
//
// **报成孤儿是危险的**：实测拉 gpt-oss:20b 到一半时，扫描输出说
// "另有 17 个孤儿 blob 可回收 12.85 GiB"，而那 12.85 GiB 正是那个
// 下到一半的模型 —— 用户照着"可回收"去删，就把自己的下载毁了。
// 反过来，判据放宽成 strings.Contains(name, "partial") 同样有害：
// 一个名字里恰好含 partial 的正常 blob 会被永远排除在孤儿之外，
// 变成一处的静默漏报。
//
// 所以判据用**精确形状**：`sha256-` + 恰好 64 位十六进制 + `-partial` 前缀。
func IsInProgress(name string) bool {
	rest, ok := strings.CutPrefix(name, "sha256-")
	if !ok || len(rest) <= digestHexLen {
		return false
	}
	for _, c := range rest[:digestHexLen] {
		if !isHexDigit(c) {
			return false
		}
	}
	return strings.HasPrefix(rest[digestHexLen:], "-partial")
}

// isHexDigit 判断一个字符是不是小写十六进制位。
//
// ollama 写的 digest 是小写的，所以只认 a-f —— 大写 A-F 会让判据
// 在"某个工具改成了大写"时静默失效，那种失效表现为半成品被当成孤儿。
func isHexDigit(c rune) bool {
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')
}
