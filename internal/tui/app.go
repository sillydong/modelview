// Package tui 是 modelview 的交互界面。
//
// # 可测性是设计出来的，不是补出来的
//
// 所有 Update / View 都是**纯函数**：它们不读文件、不扫目录、
// 不碰终端。任何 I/O 都包成 tea.Cmd 异步返回消息。
//
// 这样测试可以直接构造消息喂给 Update、断言 View() 的字符串，
// 不需要 pty、不需要等真实 I/O、也不会因为机器上没有模型而跳过。
// 一个需要终端才能测的 TUI，实际上就是没测。
package tui

import (
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// View 是一层界面。**每一层自己处理按键与渲染**，
// 根 Model 只负责栈与分发 —— 把分发写成一个巨大的 switch
// 会让每加一个视图就改一次那个 switch。
type View interface {
	// Init 返回这一层**首次显示时要跑的异步任务**，没有就返回 nil。
	//
	// **忘了把它接上，界面会永远停在初始状态而没有任何报错** ——
	// 实测：Library.Init 里发起扫描的那条命令从来没被执行过，
	// 屏幕一直停在"正在扫描模型目录…"，而所有单元测试都是绿的
	//（它们直接调 Update，不经过 bubbletea 的运行时）。
	// 是 pty 里跑真终端才看出来的。
	Init() tea.Cmd
	Update(msg tea.Msg) (View, tea.Cmd)
	// View 按给定的内容区尺寸渲染。**尺寸由参数传入而不是从 Model 读**：
	// 这样测试可以拿 80×24 直接调，不必模拟一个终端。
	View(width, height int) string
	// Help 返回这一层支持的按键提示，给底部帮助栏用。
	Help() []string
}

// pushMsg 让任意视图请求"进入下一层"。
//
// 用消息而不是从 Update 直接返回新 Model：视图拿不到栈，
// 它只该说"我想深入这一项"，栈怎么变是根 Model 的事。
type pushMsg struct{ v View }

// popMsg 让视图请求返回上一层。
type popMsg struct{}

// Model 是根模型。
type Model struct {
	stack  []View
	width  int
	height int
}

// 尺寸未知时的默认值。**不能是 0** —— 按 0 宽渲染会得到空串，
// 界面第一帧闪一下白屏。80×24 是 spec 定的最小终端。
const (
	defaultWidth  = 80
	defaultHeight = 24
)

// 布局：标题栏 1 行 + 内容 + 帮助栏 1 行。
const (
	headerLines = 1
	helpLines   = 1
)

// New 造一个根 Model，root 是栈底视图。
//
// heightAdj 是留给终端自身的高度余量（一般传 0）。
func New(root View, heightAdj int) Model {
	return Model{
		stack:  []View{root},
		width:  defaultWidth,
		height: defaultHeight - heightAdj,
	}
}

// Init 跑栈底视图的初始化任务（模型库的扫描就是从这里发起的）。
//
// 只跑栈底的：上层视图是用户"点进去"的，它们的 Init 在 pushMsg 时跑。
func (m Model) Init() tea.Cmd { return m.stack[0].Init() }

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		// 尺寸也要转给栈顶 —— 有些视图要按宽度决定列宽
		return m.forward(msg)

	case tea.KeyMsg:
		// 全局按键优先于视图自己的处理：q 在任何一层都该能退出，
		// 否则进了张量详情就没法一键退出，只能一层层 Esc 出来
		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		case "esc":
			if len(m.stack) > 1 {
				return m.pop()
			}
			// 根视图的 Esc 不退出程序（用户想的是"取消"，不是"关掉"）
			return m, nil
		}

	case pushMsg:
		m.stack = append(m.stack, msg.v)
		// **新压进来的视图也要跑它的 Init** ——
		// 不跑的话"进模型视图"会永远停在"正在解析…"
		return m, msg.v.Init()

	case popMsg:
		if len(m.stack) > 1 {
			return m.pop()
		}
		return m, nil
	}
	return m.forward(msg)
}

// forward 把消息交给栈顶视图，并把它的返回值接回栈里。
//
// **栈顶可能是被替换过的**（视图处理消息后返回一个新视图），
// 所以要写回 stack[len-1]。
func (m Model) forward(msg tea.Msg) (tea.Model, tea.Cmd) {
	top := len(m.stack) - 1
	next, cmd := m.stack[top].Update(msg)
	m.stack[top] = next
	return m, cmd
}

func (m Model) pop() (tea.Model, tea.Cmd) {
	m.stack = m.stack[:len(m.stack)-1]
	return m, nil
}

func (m Model) View() string {
	if len(m.stack) == 0 {
		return ""
	}
	top := m.stack[len(m.stack)-1]
	body := padTo(top.View(m.width, m.contentHeight()), m.contentHeight(), m.width)

	// 用 lipgloss 拼而不是手写 "\n"：它按**显示宽度**算，
	// 中文不会被当成两个字符宽（用 len() 算会把中文行算短，导致错位）
	return lipgloss.JoinVertical(lipgloss.Left,
		truncateLines(m.header(), m.width),
		body,
		truncateLines(helpLine(top.Help()), m.width))
}

func (m Model) contentHeight() int {
	h := m.height - headerLines - helpLines
	if h < 1 {
		return 1
	}
	return h
}

// header 是顶部那一行：显示栈顶视图的标题。
func (m Model) header() string {
	title := "modelview"
	if t, ok := m.stack[len(m.stack)-1].(interface{ Title() string }); ok {
		title = t.Title()
	}
	return styleTitle.Render(title)
}

// padTo 把 body 补到恰好 h 行，每行不超过 w 显示列（截断交给 truncateLines）。
//
// **必须补**：不补的话底部帮助栏会浮在内容下方而不是贴着屏幕底，
// 而且内容短的时候整个界面会随视图切换上下跳。
// 补的是空行（而不是空格），终端渲染没有区别，但测试里好读。
func padTo(body string, h, w int) string {
	lines := strings.Split(body, "\n")
	if len(lines) > h {
		lines = lines[:h]
	}
	for len(lines) < h {
		lines = append(lines, "")
	}
	// 截断交给 truncateLines（头尾两行也用它）——
	// 两处各写一遍的话，改了一处另一处不会红
	return truncateLines(strings.Join(lines, "\n"), w)
}

// truncateLines 把一串文本按宽度逐行截断。
//
// 超宽会让终端自动折行，整个界面往下错位 —— 这是 TUI 最常见的破相方式。
// 用 lipgloss 的 MaxWidth 而不是按字节切：它算的是**显示宽度**，
// 中文不会被当成两个字符宽（按 len() 算会把中文行算短，截了等于没截）。
func truncateLines(s string, width int) string {
	if width <= 0 {
		return s
	}
	var out []string
	for _, line := range strings.Split(s, "\n") {
		if lipgloss.Width(line) > width {
			line = lipgloss.NewStyle().MaxWidth(width).Render(line)
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}
