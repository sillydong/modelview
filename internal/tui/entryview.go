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
	return sb.String()
}
