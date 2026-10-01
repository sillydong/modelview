package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/humanize"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/ref"
)

// EntryView 是单条速查表条目的详情。
//
// Task 7 只做骨架：标题、ID、字段、Notes。
// SeeAlso 互跳在 Task 8 补（它要能推入新的 EntryView，那是另一套状态）。
type EntryView struct {
	e ref.Entry

	// m 是当前模型，用来算"在本模型中出现的位置"（那份列表在 Task 10 才加）。
	//
	// Task 7 的骨架只是把它存下来：**构造 EntryView 的地方才知道
	// 当前看的是哪个模型**（RefView / ModelView），事后再也拿不到 ——
	// 所以构造参数现在就是两参，与 reftable.go 的调用点一致。
	// 可以为 nil —— 为 nil 时"在本模型中"那一节整个不显示。
	m *model.Model
}

func NewEntryView(m *model.Model, e ref.Entry) EntryView { return EntryView{m: m, e: e} }

func (v EntryView) Title() string { return "速查表 · " + v.e.Title }

func (v EntryView) Init() tea.Cmd { return nil }

func (v EntryView) Update(tea.Msg) (View, tea.Cmd) { return v, nil }

func (v EntryView) Help() []string {
	return []string{keyEsc + " 返回", keyQuit + " 退出"}
}

func (v EntryView) View(width, _ int) string {
	var sb strings.Builder
	sb.WriteString(styleSection.Render(humanize.Truncate(v.e.Title, width)) + "\n")
	sb.WriteString(styleDim.Render(v.e.ID) + "\n\n")
	for _, f := range v.e.Fields {
		fmt.Fprintf(&sb, "%s  %s\n", styleField.Render(f.Key), f.Value)
	}
	if v.e.Notes != "" {
		sb.WriteString("\n" + wrapText(v.e.Notes, width) + "\n")
	}
	// **末尾不留换行**：padTo 按 "\n" 切行，多出来的空元素让它多算一行 ——
	// "…还有 N 行没显示"里的 N 会比实际丢掉的多一（joinHorizontal 里
	// 记过同一条）。Task 8 会把这一页补全，补的时候别把换行写回来。
	//
	// 用 TrimRight 而不是 TrimSuffix：ID 那一行写的是 "\n\n"（它后面
	// 本该跟一个空行），没有字段也没有 Notes 时，末尾就挂着两个换行 ——
	// 只去一个的话还剩一个（TestViews_原始输出不留末尾换行 抓到的正是
	// 这一种，审计探针里那条造了字段与 Notes，绕开了它）。
	return strings.TrimRight(sb.String(), "\n")
}
