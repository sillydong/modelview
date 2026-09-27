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
	"fmt"
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
	//
	// height 是**内容区**的高度（已去掉标题栏与帮助栏）。各层应当用它
	// 决定显示多少行；实在用不上的（如概览那种定长内容）可以忽略，
	// 根视图的 padTo 会兜底 —— 但兜底会打上"…还有 N 行没显示"的记号，
	// 所以**要展示长列表的视图必须自己接 height**。
	View(width, height int) string
	// Help 返回这一层支持的按键提示，给底部帮助栏用。
	Help() []string
}

// pushMsg 让任意视图请求"进入下一层"。
//
// 用消息而不是从 Update 直接返回新 Model：视图拿不到栈，
// 它只该说"我想深入这一项"，栈怎么变是根 Model 的事。
type pushMsg struct{ v View }

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
// **没有 heightAdj 参数**：它曾经存在，但全部 13 个调用点（1 个生产 + 12 个测试）
// 传的都是 0 —— 留着一个永远不变、又没人知道该传什么的参数，
// 只会让每个调用点都要想一下"这里该传几"。
func New(root View) Model {
	return Model{
		stack:  []View{root},
		width:  defaultWidth,
		height: defaultHeight,
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

// window 算出要显示 [start, end) 这一段，保证 cursor 落在里面。
//
// **列表必须跟着光标滚动**：不滚的话，用户按 ↓ 越过一屏之后
// 看不到自己在选什么，此时按 Enter 打开的是一个看不见的模型。
//
// 光标放在窗口偏上的位置而不是正中间：翻页时视觉更稳，
// 一行一行往下按的时候也不会每按一次整屏都跳。
func window(total, cursor, capacity int) (start, end int) {
	if capacity <= 0 || total <= capacity {
		return 0, total
	}
	start = cursor - capacity/3
	if start < 0 {
		start = 0
	}
	if start+capacity > total {
		start = total - capacity
	}
	return start, start + capacity
}

// padTo 把 body 补到恰好 h 行，每行不超过 w 显示列（截断交给 truncateLines）。
//
// **必须补**：不补的话底部帮助栏会浮在内容下方而不是贴着屏幕底，
// 而且内容短的时候整个界面会随视图切换上下跳。
// 补的是空行（而不是空格），终端渲染没有区别，但测试里好读。
func padTo(body string, h, w int) string {
	lines := strings.Split(body, "\n")
	if len(lines) > h {
		// **截断要说出来**：静默截断会让用户以为这就是全部内容。
		// 各视图应当自己按 height 裁剪（Library 就是这么做的），
		// 这里是兜底，兜底也要留个记号。
		hidden := len(lines) - h + 1
		lines = append(lines[:h-1], styleHint.Render(
			fmt.Sprintf("…还有 %d 行没显示", hidden)))
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
