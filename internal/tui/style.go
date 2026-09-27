package tui

import "github.com/charmbracelet/lipgloss"

// 样式集中一处。**不要在 View 里手写 ANSI 转义** ——
// 那样没法在测试里断言"颜色对不对"，而样式散在各处时，
// 换一个主题要翻遍所有文件。
//
// **用到才加**：提前把后面任务要用的调色板写进来，会让这一版的
// lint 报一堆 unused（实测 17 条）—— 而每个提交都该是干净的。
// 后续的视图往这里加，位置固定在这一处。
var (
	// 标题栏：模型名那一行
	styleTitle = lipgloss.NewStyle().Bold(true)

	// 底部帮助栏
	styleHelp = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
)
