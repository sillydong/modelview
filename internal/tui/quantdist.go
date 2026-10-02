package tui

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/sillydong/modelview/internal/humanize"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/ref"
	"github.com/sillydong/modelview/internal/render"
)

// quantDistText 是"量化分布"那一栏的内容。
//
// 与 ModelView 分开成函数是为了能直接测：走 ModelView.View 的话
// 测出来的字符串里全是左右栏拼接的痕迹（前导空格、分隔列），
// 断言会被迫写成"包含"而不是"这一栏的内容是什么"。
func quantDistText(m *model.Model) string {
	if len(m.Tensors) == 0 {
		return styleHint.Render("这个模型没有张量")
	}

	rows := dtypeRows(m)
	total := m.TensorBytes()
	table := dtypeTable(rows, total)

	var sb strings.Builder
	sb.WriteString(styleSection.Render("量化分布（按类型）") + "\n\n")
	// 表头与数据行**共用** dtypeTable 算出的那一组列宽：两处各算一遍的话，
	// 改了一处另一处不会红，表现是"表头与数据列对不上"。
	sb.WriteString(styleDim.Render(table[0]) + "\n")
	for _, line := range table[1:] {
		sb.WriteString(line + "\n")
	}

	// **整体 bit/权重** = 总字节 × 8 / 总参数。
	//
	// 它与任何单个类型的位宽都可能不同：混合档位的模型落在中间。
	// 这正是用户想知道的那一个数 —— "这个文件到底多密"。
	if params := m.TotalParams(); params > 0 && total > 0 {
		// 同一个格式器：它是"位宽"这个量的**主**出口，
		// 上一行按类型的那一列也走它（见 dtypeRows 的调用点）。
		//
		// **不说"唯一"**：internal/ref 的速查表位宽走的是它自己的
		// `humanFloat`（`%g` 10 位有效数字），那是另一条出口。
		// 两者的逐条一致性现在由 `ref.TestQuants_位宽的两条出口一致`
		// 钉着（23 个量化位宽逐字节比），所以它是**契约**了；
		// 但契约是"两边打出同一个字符串"，不是"只有这一个格式器" ——
		// 新写显示位宽的地方仍然走 render.BitsPerWeight。
		fmt.Fprintf(&sb, "\n整体       %s bit/权重（含块头开销）\n",
			render.BitsPerWeight(float64(total)*8/float64(params)))
	}

	sb.WriteString(declaredFileType(m))
	return sb.String()
}

type dtypeRow struct {
	dtype  model.Dtype
	count  int
	params int64
	bytes  int64
	bpw    float64
}

// dtypeRows 按类型汇总，**按占用字节降序**。
//
// 必须排序：来源是 map，遍历顺序随机 —— 不排的话任何一个按键
// 触发的重绘都会让行的先后变一次，看起来就是字在抖。
// 按字节而不是按个数：用户想知道的是"钱花在哪"，
// 一个 243 MiB 的 Q6_K 比 100 个小 F32 重要得多。
// 字节相同时按类型名升序，只为让结果唯一。
func dtypeRows(m *model.Model) []dtypeRow {
	byDtype := make(map[model.Dtype]*dtypeRow, 8)
	for _, tn := range m.Tensors {
		r, ok := byDtype[tn.Dtype]
		if !ok {
			r = &dtypeRow{dtype: tn.Dtype}
			byDtype[tn.Dtype] = r
		}
		r.count++
		r.params += tn.ParamCount
		r.bytes += tn.ByteSize
	}
	rows := make([]dtypeRow, 0, len(byDtype))
	for _, r := range byDtype {
		// **分母为 0 时留 0 而不是除**：ParamCount 为 0 的张量是存在的
		//（未收录的类型算不出元素数），除下去就是 +Inf，
		// 而 +Inf 打到界面上是 " +Inf"，看着像个数值。
		if r.params > 0 {
			r.bpw = float64(r.bytes) * 8 / float64(r.params)
		}
		rows = append(rows, *r)
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].bytes != rows[j].bytes {
			return rows[i].bytes > rows[j].bytes
		}
		return rows[i].dtype < rows[j].dtype
	})
	return rows
}

// dtypeTable 把"按类型"那张表拼成对齐的多行：第一行是表头，后面是数据。
//
// **补齐必须按显示宽度算，不能按字节**：表头是中文标签（3 字节只占 2 列），
// 数据行的类型名与数字是 ASCII（1 字节 1 列）—— 按字节补齐时中文标签被
// 算得比实际宽，表头因此比数据行宽出一大截，列标题与数据对不上。
// 实测（按字节的旧版，`%-8s %6s …`）：表头 72 列、数据行 60 列；
// 80 列终端下根视图的 truncateLines 还会切掉表头尾部 —— 切掉的恰好是
// "bit/权重" 这个列标题的尾巴，而这一栏存在的理由就是给人看位宽。
//
// 列宽取"表头标签"与"该列全部数据"的**较大者**，两者共用同一组列宽：
// 只看数据的话，比数据宽的标签会把自己那一列顶出去，表头与数据行
// 又变成不一样宽 —— 对齐是这两者的共同约束，不是谁单方面决定的。
func dtypeTable(rows []dtypeRow, total int64) []string {
	head := []string{"类型", "张量", "参数", "占用", "占比", "bit/权重"}
	body := make([][]string, 0, len(rows))
	for _, r := range rows {
		pct := 0.0
		if total > 0 {
			pct = float64(r.bytes) / float64(total) * 100
		}
		body = append(body, []string{
			string(r.dtype),
			strconv.Itoa(r.count),
			humanize.Count(r.params),
			humanize.Bytes(r.bytes),
			fmt.Sprintf("%.1f%%", pct),
			render.BitsPerWeight(r.bpw),
		})
	}

	w := make([]int, len(head))
	for i, h := range head {
		w[i] = lipgloss.Width(h)
	}
	for _, row := range body {
		for i, c := range row {
			w[i] = max(w[i], lipgloss.Width(c))
		}
	}

	// **类型左对齐，其余右对齐**：数字右对齐才看得出量级（"4.5" 与 "32"
	// 对齐之后一眼能比大小），而类型名是标识符，左对齐才好扫。
	right := []bool{false, true, true, true, true, true}
	pad := func(s string, width int, right bool) string {
		gap := width - lipgloss.Width(s)
		if gap <= 0 {
			return s
		}
		if right {
			return strings.Repeat(" ", gap) + s
		}
		return s + strings.Repeat(" ", gap)
	}
	line := func(cells []string) string {
		var sb strings.Builder
		for i, c := range cells {
			if i > 0 {
				sb.WriteByte(' ')
			}
			sb.WriteString(pad(c, w[i], right[i]))
		}
		return sb.String()
	}

	out := make([]string, 0, len(body)+1)
	out = append(out, line(head))
	for _, row := range body {
		out = append(out, line(row))
	}
	return out
}

// declaredFileType 显示文件自己声明的档位。
//
// **只显示，不判定它和实际分布是否一致**（理由见计划里那段）：
// 档位名到 model.Dtype 没有可靠的映射，靠字符串规则推会推错，
// 而推错的结论比没有结论更糟。
func declaredFileType(m *model.Model) string {
	kv, ok := findMeta(m, "general.file_type")
	if !ok {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("\n" + styleSection.Render("文件声明") + "\n")

	// **不重复印 `key = value`**：e.Title 自带 `key = value（ENUM）`，
	// 再在前面印一遍就是同一句话出现两次（实测那一行因此占到 67 列，
	// 而 80 列终端下这一栏恰好只有 67 列可用 —— 零余量）。
	// 元数据栏也有同样的冗余，但那边一行只放一条、没有宽度压力。
	line, guessed := "", false
	if raw, ok := fileTypeCode(kv); ok {
		if e, ok := ref.FileTypeByCode(raw); ok {
			line = "[" + e.Title + "]"
			guessed = ref.FileTypeGuessed(raw)
		}
	}
	if line == "" {
		// 码翻译不出来（值不是整数、或这个码不在表里）时退回 `key = value`：
		// 什么都不印就是"用户看到一个值，工具一句话都不说"
		line = kv.Key + " = " + kv.Value
	}
	// 三行分别拼好再 join，因为**警告必须另起一行**：与条目名拼在一起时
	// 废弃档位那条（`general.file_type = 4（已废弃：MOSTLY_Q4_1_SOME_F16）`
	// 自身 57 列）加上警告就是 93 列（实测）—— 而这一栏最坏可用宽度只有 64
	//（80 − nav 最宽 15 − 1），真终端下尾巴会被 truncateLines 切掉。
	// 同一段文案，只是换个位置；行首加 ⚠ 补回原来靠括号带出的"这是警告"。
	lines := []string{"  " + line}
	if guessed {
		lines = append(lines, styleWarn.Render(
			"  ⚠ 上游猜的，未必与实际张量类型相符"))
	}
	// 提示语**比原来短**：原句 62 列，是当时这一栏最宽的一行，而最坏可用
	// 宽度只有 64 —— 只剩 2 列余量，nav 再长一点就被切。只把"上面那张表"
	// 缩成"上表"，三个意思一个不少：这是文件写的、不是实际分布、以那张表为准。
	lines = append(lines, styleHint.Render(
		"  这是文件自己写的声明，不代表实际分布 —— 上表才是实际的"))
	sb.WriteString(strings.Join(lines, "\n") + "\n")
	return sb.String()
}

// findMeta 按 key 找一条元数据。
func findMeta(m *model.Model, key string) (model.MetaKV, bool) {
	for _, kv := range m.Metadata {
		if kv.Key == key {
			return kv, true
		}
	}
	return model.MetaKV{}, false
}

// **这里原本有一个 trimFloat，现在删掉了** —— 它被提到了
// internal/render 里叫 BitsPerWeight（Task 1b 做的）。
//
// 提上去的理由不是"两处代码重复"，而是**两处代码会分叉**：
// 位宽这个数在 CLI 的 --stats、TUI 的类型分布表、TUI 的张量详情页
// 三处出现，各写一份格式化的话，有一天会变成
// "这一页说 6.5625、那一页说 6.562"，而没有任何东西会红。
// 这正是 internal/render 这个包存在的全部理由。
