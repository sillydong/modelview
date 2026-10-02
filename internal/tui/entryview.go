package tui

import (
	"fmt"
	"strings"

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
			wrapText(f.Value, max(width-lipgloss.Width(f.Key)-2, 10)))
	}
	if v.e.Notes != "" {
		sb.WriteString("\n" + wrapText(v.e.Notes, width) + "\n")
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
