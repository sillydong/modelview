// Package render 把模型数据渲染成给人看的文本。
//
// CLI 与 TUI 必须显示**同样的数字**：各写一份的后果不是报错，
// 是"命令行说 σ=0.0268、界面说 σ=0.027"—— 用户会以为是两个不同的数，
// 而没有任何东西会红。与 humanize 的分工：
//   - humanize：纯数字（字节、参数量、形状、截断、浮点、百分数）
//   - render：带模型语义的（统计量、量化诊断、模拟结果）
package render

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/sillydong/modelview/internal/humanize"
	"github.com/sillydong/modelview/internal/model"
)

// TensorQuant 决定张量行末尾挂哪一段量化信息。
//
// 两者不该同时出现：浮点张量有模拟（QuantSims）、量化张量有诊断（Quant），
// 接入层保证了互斥。同时有值时优先显示模拟 —— 那说明数据来源不寻常，
// 但宁可显示一个也不要显示两个互相矛盾的数字。
func TensorQuant(t *model.Tensor) string {
	if s := QuantSim(t.QuantSims); s != "" {
		return s
	}
	return QuantExisting(t.Quant)
}

// QuantSim 把三档模拟压成一行。
//
// 这是按各格式**真实编码器**算出来的误差，不是统一公式的估算。
//
// 仍然标出「模拟」二字，因为还有一处差别必须让用户知道：
// 模拟是等权编码，真实文件可能是用重要性矩阵加权编的 ——
// 本工具没有那份标定数据。实测在 qwen2.5:3b 上这没造成数值差异
// （真实块 40/40 都能被等权编码器精确复现），但换一个文件就未必。
//
// 采样出来的数字前面加 ≈，与 Stats 同一个约定：
// 把样本的数字当成精确值是误导，而一行里没地方写"这是采样值"。
func QuantSim(sims []model.QuantSim) string {
	if len(sims) == 0 {
		return ""
	}
	mark := ""
	if sims[0].Sampled {
		mark = "≈"
	}
	parts := make([]string, 0, len(sims))
	for _, s := range sims {
		parts = append(parts, quantSimLine(s.Target, BitsPerWeight(s.BitsPerWeight), s))
	}
	return mark + "模拟[" + strings.Join(parts, " | ") + "]"
}

// quantSimLine 是单档的格式串本体。
//
// 单独抽出来是为了让补齐版与不补齐版共用同一个格式：
// 各写一份的后果不是报错，是两边的数字慢慢分叉。
func quantSimLine(target, bits string, s model.QuantSim) string {
	// 用 "bit/权重" 而不是 "B/权重"：B 在这个输出里已经是**字节**
	// （同一行就有 243.43 MiB），且"类型分布"段用的就是 bit/权重。
	// 位宽交给 BitsPerWeight —— 它不能走 %.4g（6.5625 会被抹成 6.562）
	return fmt.Sprintf("%s %s bit/权重 %.1fdB ×%.2f", target, bits, s.SNRDB, s.Compression)
}

// QuantSimLines 把若干档模拟排成列对齐的多行（TUI 详情页用）。
//
// 与 QuantSim 的分工：那个是 CLI 的单行版（各档用 " | " 接起来，
// 不能补齐，补齐会多出空格）；这个是多行版，按最宽的档名与位宽补齐。
// 两者共用 quantSimLine 的格式，所以数字不会分叉。
//
// 采样时**每一行**都加 ≈，口径与 QuantSim 相同（只看 sims[0]）：
// 采样是常态（默认上限 1e7 个元素，一个 4096×4096 的 f16 就有 1677 万），
// 少一行标记就是那一行在把采样值当精确值显示。只看第一档是为了让
// 三行要么都有、要么都没有 —— 逐档判断会做出参差不齐的列，
// 也与单行版的口径不一致（真实数据里三档出自同一次分析，必然一致）。
//
// 返回的每一行都**不含** "模拟[...]" 包装：那是"整套结果"的属性，
// 按档拆开之后没有地方安放 —— 详情页自己在这一节的开头写一次"模拟值"。
// 但 ≈ **要**带上：它是逐档数字自己的属性。
//
// 列宽按**字节**算（len），不是显示宽度：Target 全是 ASCII
// （Q8_0/IQ2_XXS），两者一致；换成含中文的档名就会错位。
// 与 humanize.Truncate 同一个取舍 —— 不引显示宽度依赖，接受这个上限。
func QuantSimLines(sims []model.QuantSim) []string {
	if len(sims) == 0 {
		return nil
	}
	// 补齐宽度要先把所有档扫一遍：档名左对齐（从同一列起），
	// 位宽右对齐（数字右对齐才看得清量级）
	bits := make([]string, len(sims))
	wTarget, wBits := 0, 0
	for i, s := range sims {
		bits[i] = BitsPerWeight(s.BitsPerWeight)
		wTarget = max(wTarget, len(s.Target))
		wBits = max(wBits, len(bits[i]))
	}
	// ≈ 与单行版同口径：只看第一档，三行要么都有、要么都没有
	mark := ""
	if sims[0].Sampled {
		mark = "≈"
	}
	lines := make([]string, 0, len(sims))
	for i, s := range sims {
		lines = append(lines, mark+quantSimLine(
			fmt.Sprintf("%-*s", wTarget, s.Target),
			fmt.Sprintf("%*s", wBits, bits[i]), s))
	}
	return lines
}

// QuantExisting 把块级诊断压成一行。
//
// 位宽与压缩比不在这里重复：类型分布那一节已经按类型给过 bit/权重。
// TUI 要分行摆时用 QuantExistingLines —— 内容一致，只是摆法不同。
func QuantExisting(q *model.QuantInfo) string {
	return strings.Join(QuantExistingLines(q), " ")
}

// QuantExistingLines 把块级诊断拆成若干行。
//
// 拆行是给 TUI 详情页用的：单行版 100+ 列，80 列终端下会被静默截断。
// 实测最坏 54 显示列（含 TUI 的 2 列缩进），80 列终端下有余量。
// 行内容与单行版**逐字节相同**（QuantExisting 就是把它 join 回去），
// 所以多一个显示入口不会让数字分叉 —— 唯一的例外是抽样中位数：
// 两类输出都会带上标记，那是有意的，见下。
//
// **调用方保证 q 不是零值**：analyze 那边有 `q.SubBlocks > 0` 的守护。
// 零值会打出一条看着像事实的诊断（"0 bit/权重 0 子块"、"最扁 #0（比值 0）"）
// —— TUI 手写 fixture 时漏填字段就是这个下场。
//
// 最多 3 行：方案/位宽/子块数、scale 范围与中位、两个警告。
func QuantExistingLines(q *model.QuantInfo) []string {
	if q == nil {
		return nil
	}
	// 中位数是这里**唯一**可能来自抽样的量：其余统计量都是流式精确的，
	// 只有中位数必须看到全部值，子块数上千万时超预算就改等距抽样
	//（见 model.QuantInfo 的说明）。
	//
	// 抽样时 ≈ 与"（抽样）"两个都要：只加 ≈ 会被读成排版装饰，
	// 只写"（抽样）"又不够显眼 —— 而用户会拿这个数去比另一个文件。
	median := humanize.Float(q.ScaleMedian)
	if q.ScaleMedianSampled {
		median = "≈" + median + "（抽样）"
	}
	lines := []string{
		fmt.Sprintf("%s %s bit/权重 %s 子块",
			q.Scheme, BitsPerWeight(q.BitsPerWeight), humanize.Count(q.SubBlocks)),
		fmt.Sprintf("scale[%s, %s] 中位 %s",
			humanize.Float(q.ScaleMin), humanize.Float(q.ScaleMax), median),
	}
	// 被压平的**子块**数是最直接的证据，必须显示。
	// 单位必须写"子块"而不是"块"：同一行前面刚写过"N 子块"，
	// 而一个块含 8 或 16 个子块，写成"块"会差一个数量级
	var warn []string
	if q.ZeroScaleBlocks > 0 {
		warn = append(warn, fmt.Sprintf("⚠压平 %s 子块", humanize.Count(q.ZeroScaleBlocks)))
	}
	// 被压得最狠的那个子块。model.QuantInfo 的注释承诺了界面上要显示它
	//（"第 N 个子块（张量内第 N×BlockElems 个权重）"），
	// 只进 JSON 不显示的话那个承诺就是假的
	if q.FlattestRatio < 1 && q.FlattestRatio >= 0 {
		warn = append(warn, fmt.Sprintf("最扁 #%d（比值 %s）",
			q.FlattestIndex, humanize.Float(q.FlattestRatio)))
	}
	// 两个警告都可能缺席；都不在时这一行**不返回** ——
	// 空行会让 TUI 白占一行（每行前面还要加 2 个空格前缀）。
	//
	// 两者之间只用一个空格：QuantExisting 是 strings.Join(lines, " ")，
	// 改成两个空格会让 CLI 的单行输出多出一个空格 —— 实测量化模型上
	// 有 4 行同时命中两个警告，那就是用户会看到的漂移。
	if len(warn) > 0 {
		lines = append(lines, strings.Join(warn, " "))
	}
	return lines
}

// Stats 把统计压成一行。
//
// 采样过的数字前面加 ≈ —— 把样本统计量当成全量是误导，
// 而一行里没地方写"这是采样值"。
func Stats(s *model.Stats) string {
	mark := ""
	if s.Sampled {
		mark = "≈"
	}
	out := fmt.Sprintf("%s[%s, %s] μ=%s σ=%s 零=%s 离群=%s",
		mark, humanize.Float(s.Min), humanize.Float(s.Max),
		humanize.Float(s.Mean), humanize.Float(s.Std),
		humanize.Percent(s.ZeroRatio), humanize.Percent(s.OutlierRatio))

	// 非有限值必须显示出来。
	//
	// 它们只被计数、不参与统计，所以一个**全是 NaN** 的张量
	// Min/Max/Mean/Std 全是零值，看起来跟真正的零张量一模一样。
	// 正是 model.Stats 注释里点名要避免的那种误导，只是换了个形式。
	if s.NaN > 0 || s.Inf > 0 {
		out += fmt.Sprintf(" ⚠NaN=%d Inf=%d", s.NaN, s.Inf)
	}
	return out
}

// BitsPerWeight 把位宽打成精确值。
//
// **不能用 %.4g**：它把 6.5625 打成 6.562、3.4375 打成 3.438、
// 2.0625 打成 2.062（实测）—— 位宽是文件里的精确值，少一位就是错的。
// 也不能用 humanize.Float（那是给统计量用的：常用区间内 4 位小数，
// 且会补尾零 —— Float(4.5) = "4.5000"，位宽显示成那样是错的）。
//
// 输入前提：**正的、有限的、1/16 的整数倍**（GGUF 的位宽都是这个形状）。
// 超出前提不报错，但结果无意义：NaN/±Inf 原样打出、负号会留着、
// 次正规数（5e-324）被 4 位小数抹成 "0" —— 非零打成零。
// 这几种都有测试钉着，改实现时会看到。
func BitsPerWeight(v float64) string {
	// 定点 4 位小数：位宽都是 x.0625 / x.5 这种有限小数，4 位足够，
	// 再多就只是噪音。然后去掉尾部的 0 与孤零零的小数点 ——
	// 但**必须**在去掉之后保住整数部分（"0.0000" 不能变成 ""）
	s := strconv.FormatFloat(v, 'f', 4, 64)
	if strings.Contains(s, ".") {
		s = strings.TrimRight(s, "0")
		s = strings.TrimSuffix(s, ".")
	}
	return s
}
