// Package humanize 把数字变成人看的字符串。
//
// 单独一个包而不是放在 cmd 里：**CLI 与 TUI 必须显示同样的数字**。
// 各写一份的后果不是报错，是"命令行列出的模型是 1.80 GiB、
// 界面里点进去是 1.8 GiB"—— 用户会以为是两个不同的数，
// 而没有任何东西会红。计划 ④b 的 TUI 要用这同一份。
//
// 判据是**这个量在别的语境里还有没有意义**，不是入参类型：
// Bytes/Count/Percent/Float 在任何程序里都成立，所以在这里；
// 位宽那种只在量化语境里才有意义的量留在 render（BitsPerWeight）——
// 它收的也是裸 float64，按入参类型判会得出相反的结论。
package humanize

import (
	"fmt"
	"math"
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

// Float 打印统计量：常用区间内 4 位小数，区间外退回 4 位有效数字。
//
// 权重的动态范围常常横跨好几个数量级（1e-5 到 1e-1），
// 定点格式会让小值全变成 0.0000 —— 所以 [1e-3, 1e5) 之外要走 'g'。
// 实测：`Float(6.5625)` = "6.5625"、`Float(123456.5)` = "1.235e+05"。
//
// **别拿它打位宽**，但理由不是"会打短" —— 实测它打得准。
// 真正的理由有两条：
//   - 它**不是为精确性设计的**：区间外会掉进科学计数法，
//     而位宽是文件里的精确值，显示成 `1.235e+05` 毫无意义
//   - 它会**补尾零**：`Float(4.5)` = "4.5000"，
//     而位宽显示成 4.5000 是错的（那多出来的四位看着像精度）
//
// 位宽用 render.BitsPerWeight：按构造精确，且去尾零。
func Float(v float64) string {
	switch {
	case v == 0:
		return "0"
	case math.IsNaN(v):
		return "NaN"
	case math.IsInf(v, 1):
		return "+Inf"
	case math.IsInf(v, -1):
		return "-Inf"
	}
	if a := math.Abs(v); a >= 1e-3 && a < 1e5 {
		return strconv.FormatFloat(v, 'f', 4, 64)
	}
	return strconv.FormatFloat(v, 'g', 4, 64)
}

// Percent 把 0..1 的比例打成百分数。
//
// 叫 Percent 而不是 Ratio：它**返回的就是百分数**（"50.00%"），
// 与 Bytes/Count/Comma 一样"输出是什么就叫什么"—— 叫 Ratio 会让人
// 以为打的是 0.5。
func Percent(v float64) string {
	if v == 0 {
		return "0"
	}
	if v < 0.0001 {
		return fmt.Sprintf("%.2g%%", v*100)
	}
	return fmt.Sprintf("%.2f%%", v*100)
}
