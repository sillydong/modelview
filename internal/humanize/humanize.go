// Package humanize 把数字变成人看的字符串。
//
// 单独一个包而不是放在 cmd 里：**CLI 与 TUI 必须显示同样的数字**。
// 各写一份的后果不是报错，是"命令行列出的模型是 1.80 GiB、
// 界面里点进去是 1.8 GiB"—— 用户会以为是两个不同的数，
// 而没有任何东西会红。计划 ④b 的 TUI 要用这同一份。
package humanize

import (
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Bytes 用 1024 进制（KiB/MiB/GiB/TiB）缩写字节数，与文件管理器一致。
func Bytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	units := []string{"KiB", "MiB", "GiB", "TiB"}
	v := float64(n)
	for _, u := range units {
		v /= unit
		if v < unit {
			return fmt.Sprintf("%.2f %s", v, u)
		}
	}
	return fmt.Sprintf("%.2f PiB", v/unit)
}

// Count 用 SI 前缀（K/M/G/T，1000 进制）缩写参数量。
//
// **刻意不使用 "B"**：摘要里同一屏的 Bytes 用 "B" 表示字节，
// 两个 B 并排出现会让 "3.086 B" 被误读成三个字节。
func Count(n int64) string {
	if n < 1000 {
		return strconv.FormatInt(n, 10)
	}
	units := []string{"K", "M", "G", "T"}
	v := float64(n)
	for _, u := range units {
		v /= 1000
		if v < 1000 {
			return fmt.Sprintf("%.3f %s", v, u)
		}
	}
	return fmt.Sprintf("%.3f P", v/1000)
}

// Comma 给整数加千位分隔符。
func Comma(n int64) string {
	s := strconv.FormatInt(n, 10)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	var sb strings.Builder
	if neg {
		sb.WriteByte('-')
	}
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			sb.WriteByte(',')
		}
		sb.WriteRune(c)
	}
	return sb.String()
}

// Dims 把形状写成 "[2048, 151936]"。
func Dims(dims []int64) string {
	var sb strings.Builder
	sb.WriteByte('[')
	for i, d := range dims {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(strconv.FormatInt(d, 10))
	}
	sb.WriteByte(']')
	return sb.String()
}

// Truncate 按**字节**截断到 n 以内，超出时以 "..." 结尾。
//
// 按字节而不是按 rune：用途是给列宽封顶，而一个汉字 3 字节却只占
// 2 列 —— 精确对齐要算显示宽度，不值得为它引入依赖。
// 要说清楚的是：**封的是字节数这个上限，不是"对齐"** ——
// 28 字节的中文名在屏幕上占 56 列，比同长度的 ASCII 宽一倍，
// 列宽仍然对不齐。这是刻意的取舍：不引依赖，接受中文名那列偏宽。
//
// **切点必须回退到 rune 边界**：否则 `qwen2.5-3b-中文微调.gguf`
// 这种名字会切出半个汉字，输出里出现替换字节 —— 那不只是难看，
// 是**非法 UTF-8**。回退后可能短一两个字节，
// 那没关系 —— 列宽是上限，不是配额。
//
// 具体的坏样子：**按字节硬切**（不回退）时
// `Truncate("中文模型名很长的名字abc", 8)` 会切出 "中\xe6\x96..."。
// 本函数不会 —— 上面那个回退循环就是为它写的；
// 这句话描述的是"没有这个循环"的写法，不是本函数的行为。
//
// 截断**不是"只影响观感"**：完整路径在最后一列不假，
// 但第一列的名字被切掉的部分，用户得去最后一列里自己找回来。
// 所以宁可短一点也不切出坏字节。
func Truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	// 给 "..." 留位置；n 太小时连省略号都放不下，那就只截 n 字节
	cut := n
	if n > 3 {
		cut = n - 3
	}
	// 回退到 rune 边界：cut 落在多字节字符中间时往前挪
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	if n <= 3 {
		return s[:cut]
	}
	return s[:cut] + "..."
}
