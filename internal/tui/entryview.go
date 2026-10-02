package tui

import (
	"fmt"
	"strings"
	"unicode/utf8"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/sillydong/modelview/internal/humanize"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/ref"
)

// entryTarget 是条目页里一个可跳转的目标。
//
// **SeeAlso 与"在本模型中"（Task 10 加）合并成一条可导航的列表**，
// 而不是两段各带一个光标：两段加起来通常不超过 10 行，
// 为此引入 Tab 切焦点、外加"当前哪一段有焦点"的显示，
// 成本比让用户多按两下 ↓ 高得多。
//
// 带 ok 而不是把查不到的过滤掉：ref.Entry 的注释明说 SeeAlso
// **不校验存在性**、指向未来的条目是允许的。过滤掉的话，
// 作者写了 `quant:Q4_KX` 这种笔误、或者指向一条还没写的条目时，
// 界面上一点痕迹都没有 —— 而"那条关联去哪了"是没人查得出来的。
type entryTarget struct {
	group string // 小节标题，用来在切换分组时打一行标题
	label string
	hint  string // 行尾的灰色补充说明，没有就空
	ok    bool   // false 表示这条跳不过去
	cmd   tea.Cmd
}

// EntryView 是单条速查表条目的详情。
type EntryView struct {
	e ref.Entry

	// m 是当前模型，**occurrences() 是它的消费者**（"在本模型中"那一节）。
	// 为 nil 时那一节整段不显示：没有模型上下文时的事实是"不知道"。
	m *model.Model

	targets []entryTarget
	cursor  int
}

// **注意：这里对 SeeAlso 是急切递归构造**（构造 target 的视图时会再构造
// 它的 target…）。今天安全 —— 实测 249 条里最长链 1、环 0 ——
// 但环会让"打开那一页"栈溢出。守卫在 internal/ref 的 TestSeeAlso_无环。
//
// （那个"最长链 1"是**记忆化量法**的结果：旧版把它挂在 DFS 深度上，
// 涂黑过的节点不再下探，于是同一个图被量成 0 —— 链长不致命，但一个
// 会随枚举顺序变的数字不能当事实抄进注释。）
func NewEntryView(m *model.Model, e ref.Entry) EntryView {
	v := EntryView{m: m, e: e}
	// **"在本模型中"排在前面**：用户查一条规范，最想知道的是
	// "它在我这个模型里是哪一行"，而 SeeAlso 是"还想知道什么"
	v.targets = append(v.targets, occurrences(m, e)...)
	for _, id := range e.SeeAlso {
		t := entryTarget{group: "相关条目", label: id}
		if target, ok := ref.ByID(id); ok {
			t.label, t.ok = target.Title, true
			t.cmd = pushCmd(NewEntryView(m, target))
		} else {
			t.hint = "（速查表里没有这条）"
		}
		v.targets = append(v.targets, t)
	}
	return v
}

func (v EntryView) Title() string { return "速查表 · " + v.e.Title }

func (v EntryView) Init() tea.Cmd { return nil }

// Modal 恒为 false：速查表条目是只读的，q 与 Esc 照旧归根视图。
func (v EntryView) Modal() bool { return false }

func (v EntryView) Update(msg tea.Msg) (View, tea.Cmd) {
	key, ok := msg.(tea.KeyMsg)
	if !ok {
		return v, nil
	}
	switch key.String() {
	case "up", "k":
		if v.cursor > 0 {
			v.cursor--
		}
	case "down", "j":
		if v.cursor < len(v.targets)-1 {
			v.cursor++
		}
	case "enter":
		// 跳不过去的目标**什么都不做**：推一张空白卡片进去，
		// 用户只会以为是自己按错了
		if v.canJump() {
			return v, v.targets[v.cursor].cmd
		}
	}
	return v, nil
}

// canJump 表示 Enter 现在真的有得跳 —— **与 Update 里那条判断同一句**。
//
// 只判 `len(targets) > 0` 不够：SeeAlso 不校验存在性（ref.Entry 的注释），
// 一条都查不到时整页一个能跳的都没有，帮助栏那时列"Enter 跳转"
// 就是在骗用户按。光标也要算进来 —— 跳不跳是**当前这一行**的属性。
func (v EntryView) canJump() bool {
	return v.cursor < len(v.targets) && v.targets[v.cursor].ok
}

func (v EntryView) Help() []string {
	if len(v.targets) == 0 {
		return []string{keyHelp + " 速查表", keyEsc + " 返回", keyQuit + " 退出"}
	}
	bindings := []string{keyUp + " " + keyDown + " 选择"}
	if v.canJump() {
		bindings = append(bindings, keyEnter+" 跳转")
	}
	return append(bindings, keyHelp+" 速查表", keyEsc+" 返回", keyQuit+" 退出")
}

func (v EntryView) View(width, _ int) string {
	var sb strings.Builder
	sb.WriteString(styleSection.Render(humanize.Truncate(v.e.Title, width)) + "\n")
	sb.WriteString(styleDim.Render(v.e.ID) + "\n\n")
	for _, f := range v.e.Fields {
		// 值要折行：量化条目的说明里有整段中文，
		// 一行放不下时超宽会被终端自动折行，整个界面往下错位
		fmt.Fprintf(&sb, "%s  %s\n", styleField.Render(f.Key),
			renderEmphasis(f.Value, max(width-lipgloss.Width(f.Key)-2, 10), emph))
	}
	if v.e.Notes != "" {
		sb.WriteString("\n" + renderEmphasis(v.e.Notes, width, emph) + "\n")
	}
	// 跳转目标按 group 分段渲染，标题只在切换分组时打一次。
	//
	// 光标是**跨段连续**的（↑↓ 一路走到底）：写死成"每段一个光标"
	// 的话，第一段的最后一项按 ↓ 该去哪就没有答案 ——
	// 要么卡住，要么跳到第二段的开头却不告诉用户，
	// 而两种在屏幕上都看不出来。
	lastGroup := ""
	for i, t := range v.targets {
		if t.group != lastGroup {
			sb.WriteString("\n" + styleSection.Render(t.group) + "\n")
			lastGroup = t.group
		}
		if i == v.cursor {
			sb.WriteString(styleSelected.Render("▸ "+t.label) +
				styleDim.Render(t.hint) + "\n")
			continue
		}
		if !t.ok {
			// 跳不过去的用灰色，并且把原始 ID 显示出来 ——
			// 用户才好知道该去补哪一条
			sb.WriteString(styleDim.Render("  "+t.label+t.hint) + "\n")
			continue
		}
		sb.WriteString("  " + t.label + styleDim.Render(t.hint) + "\n")
	}
	// **末尾绝不留换行** —— 上面每一行都以 "\n" 结尾，所以最后会多出一个。
	//
	// 多出来那一行会被根视图的 padTo 当成"内容多了一行"，于是它砍掉**真实内容**
	// 并打上一句"…还有 N 行没显示"。`joinHorizontal` 的注释里记过这条教训，
	// 而 `Library` 正是踩了它（实测：屏幕上最后一个模型不见了，
	// 范围提示被换成 padTo 那句）。
	//
	// `TrimRight` 而不是 `TrimSuffix`：Notes 缺省、且 fields 与 targets 都为空时，
	// 末尾可能连着两个换行（实测踩过）—— `TrimSuffix` 只去一个，守卫仍会红。
	return strings.TrimRight(sb.String(), "\n")
}

// boldMark 是条目文本里的粗体标记。
//
// 速查表的数据（`ref/` 里的 Notes 与字段值）是当作 Markdown 写的，
// 里面有几十处 `**…**`。在这之前没有任何东西负责渲染它们 ——
// 实测截图：条目详情页上原样显示 `**最常见的 4 位档**`。
const boldMark = "**"

// renderEmphasis 把 `**…**` 换成粗体，同时按显示宽度折行。
//
// ## 为什么解析与折行必须在一趟里做完
//
// 先给整串上样式再交给 wrapText 是错的：wrapText 按 rune 累加
// lipgloss.Width，而转义序列会被算成 0 宽（或者按字符数算进去），
// 于是断行点落在转义序列中间，屏幕上出现半个转义码 ——
// 剩下的部分被终端当成普通文字吃掉一大段。
// 这里只有裸文本参与算宽度，样式在**已经确定断点之后**才贴上去。
//
// ## bold 为什么是参数
//
// 测试里不能直接调 styleEmph.Render：`go test` 的 stdout 不是终端，
// lipgloss 会退化成 ASCII profile 并**直接把原文返回**，于是
// "样式有没有加上"在断言里看不出来 —— 那样的测试是空转的
// （它只证明标记被吃掉了，证明不了粗体生效）。
// 传一个确定的函数进来，测试才断言得出样式作用在哪一段上。
//
// ## 标记不成对时整个当普通文本
//
// 不是"从第一个标记开始一直粗到底"：半应用会让数据里少写一个 `**`
// 变成不可见的事故（星号被吃掉、后面整段悄悄变粗），
// 而宁可露出两个星号，让人看见这里写错了。
func renderEmphasis(s string, width int, bold func(string) string) string {
	if strings.Count(s, boldMark)%2 != 0 {
		return wrapText(s, width)
	}

	var (
		out  strings.Builder
		run  strings.Builder // 当前这一段同样式的文本
		isB  bool            // 当前段是不是粗体
		line int             // 当前行已占的显示宽度
	)
	flush := func() {
		if run.Len() == 0 {
			return
		}
		if isB {
			out.WriteString(bold(run.String()))
		} else {
			out.WriteString(run.String())
		}
		run.Reset()
	}

	for i := 0; i < len(s); {
		if strings.HasPrefix(s[i:], boldMark) {
			flush() // 换样式之前先结算上一段，免得粗体区间串到标记之后
			isB = !isB
			i += len(boldMark)
			continue
		}
		r, size := utf8.DecodeRuneInString(s[i:])
		i += size
		rw := lipgloss.Width(string(r))
		if width > 0 && line+rw > width {
			flush()
			out.WriteByte('\n')
			line = 0
		}
		run.WriteRune(r)
		line += rw
	}
	flush()
	return out.String()
}
