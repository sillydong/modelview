// Package render 把模型数据渲染成给人看的文本。
//
// CLI 与 TUI 必须显示**同样的数字**：各写一份的后果不是报错，
// 是"命令行说 σ=0.0268、界面说 σ=0.027"—— 用户会以为是两个不同的数，
// 而没有任何东西会红。与 humanize 的分工：
//   - humanize：纯数字（字节、参数量、形状、截断）
//   - render：带模型语义的（统计量、量化诊断、模拟结果）
package render

import (
	"fmt"
	"math"
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
		// 用 "bit/权重" 而不是 "B/权重"：B 在这个输出里已经是**字节**
		// （同一行就有 243.43 MiB），且"类型分布"段用的就是 bit/权重。
		// %.4g 也与那一段一致，8.5 不会打成 8.5000
		parts = append(parts, fmt.Sprintf("%s %.4g bit/权重 %.1fdB ×%.2f",
			s.Target, s.BitsPerWeight, s.SNRDB, s.Compression))
	}
	return mark + "模拟[" + strings.Join(parts, " | ") + "]"
}

// QuantExisting 把块级诊断压成一行。
//
// 位宽与压缩比不在这里重复：类型分布那一节已经按类型给过 bit/权重。
func QuantExisting(q *model.QuantInfo) string {
	if q == nil {
		return ""
	}
	out := fmt.Sprintf("%s %.4g bit/权重 %s 子块 scale[%s, %s] 中位 %s",
		q.Scheme, q.BitsPerWeight, humanize.Count(q.SubBlocks),
		Float(q.ScaleMin), Float(q.ScaleMax), Float(q.ScaleMedian))
	// 被压平的**子块**数是最直接的证据，必须显示。
	// 单位必须写"子块"而不是"块"：同一行前面刚写过"N 子块"，
	// 而一个块含 8 或 16 个子块，写成"块"会差一个数量级
	if q.ZeroScaleBlocks > 0 {
		out += fmt.Sprintf(" ⚠压平 %s 子块", humanize.Count(q.ZeroScaleBlocks))
	}
	// 被压得最狠的那个子块。model.QuantInfo 的注释承诺了界面上要显示它
	//（"第 N 个子块（张量内第 N×BlockElems 个权重）"），
	// 只进 JSON 不显示的话那个承诺就是假的
	if q.FlattestRatio < 1 && q.FlattestRatio >= 0 {
		out += fmt.Sprintf(" 最扁 #%d（比值 %s）",
			q.FlattestIndex, Float(q.FlattestRatio))
	}
	return out
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
		mark, Float(s.Min), Float(s.Max),
		Float(s.Mean), Float(s.Std),
		Ratio(s.ZeroRatio), Ratio(s.OutlierRatio))

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

// Float 用 4 位有效数字打印统计量。
//
// 权重的动态范围常常横跨好几个数量级（1e-5 到 1e-1），
// 定点格式会让小值全变成 0.0000。
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

// Ratio 把 0..1 的比例打成百分数。
func Ratio(v float64) string {
	if v == 0 {
		return "0"
	}
	if v < 0.0001 {
		return fmt.Sprintf("%.2g%%", v*100)
	}
	return fmt.Sprintf("%.2f%%", v*100)
}
