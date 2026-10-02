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
	// Title 是标题栏显示的名字。**合进接口是为了防静默降级**：
	// 以前靠类型断言拿，忘实现就静默退回默认标题。
	Title() string
	// Modal 表示这个视图现在要独占按键：绝大多数视图返回 false，
	// 会输入文字的视图在输入态返回 true。
	//
	// **合进接口而不是留作外部断言，理由是把三种静默失效变成编译错误**：
	// 接收者写成指针 → 不再实现 View，pushCmd 那行直接编不过；
	// Update 某条分支返回一个不实现 View 的类型 → 返回语句编不过；
	// 新视图忘实现 → 编不过。
	// 这三种都不是假想的：本分支做模态时，前两种各被临时探针实测复现过
	//（指针接收者那次的后果是过滤框里敲 q 直接退出程序，而全部单测绿着）。
	//
	// **它堵不住的是"返回了另一个也实现 View 的类型"**（如 EntryView{}）：
	// 那种写法编得过，而模态性照丢不误 —— 实测过。这一种只能靠运行时守卫
	//（TestApp_模态性经过Update仍在）盯，编译器不管语义。
	// 别把接口这条保证读得比它实际更强。
	//
	// 为什么界面上需要一个"能独占按键"的概念：没有它就没法有一个能输入
	// 文字的框 —— 全局按键处理排在视图之前，输入 "qwen" 的第一下就退出
	// 了程序。而这个问题**单元测试抓不到** —— 直接调 view.Update 绕过了
	// 根 Model（本文件开头那段注释讲的就是同一种漏法）。
	//
	// 用"整屏独占"而不是"按键级白名单"（如 ClaimsKey(key) bool）：
	// **不是**因为白名单要列举的键多 —— 非 q/esc 的键本来就会经根视图末尾的
	// 兜底 forward 转给视图，白名单只要声明 q 与 esc 两个键。真正的理由是
	// **所有权**：白名单把"q 该不该退出"这个决定从根视图漏给了每一个视图，
	// 漏一个就是用户要么退不出去、要么在输入框里打着字就退出了；
	// 而且根视图每新增一个全局键（?、tab…），每个视图都得跟着改一遍。
	//
	// 这段说明原先挂在一个未导出的 `modalView` 上，那个办法还有个更隐蔽的
	// 洞：**未导出只封住了接口名，封不住方法名** —— Go 的接口满足是结构性
	// 的，任何包里只要有个类型碰巧有 `Modal() bool` 就自动满足它。
	// 合进 View 之后这个洞没有了（谁要当视图，谁就得显式声明），
	// 但方法名本身仍然只是个约定 —— 类型断言那种走法只是整个删掉了。
	Modal() bool
}

// top 返回栈顶视图。
//
// **栈永不为空，所以这里没有兜底**：`New` 保证至少一层，`pop` 与
// `popToMsg` 都在只剩一层时停手 —— 空栈意味着内部不变量被破坏，
// 不是用户能到达的状态（零值的 Model 同理：它本来就不该被构造）。
//
// 原先 `View()` 里有一条 `len(m.stack) == 0 { return "" }`，而
// `topModal` / `forward` / `header` / `Init` 四处都没有 —— 同一个前提
// 五个读者读出两种结论。而且兜底那一处会把"栈被清空"咽成一片空白：
// 界面全白、没有任何东西会红，比当场崩掉难查得多。
// 取舍统一成这一处显式 panic，理由在这里说一次。
func (m Model) top() View {
	if len(m.stack) == 0 {
		panic("tui: 视图栈为空 —— New() 必须给一个栈底，pop 不该弹掉最后一层")
	}
	return m.stack[len(m.stack)-1]
}

// topModal 判断栈顶视图是否要独占按键。**只看栈顶**：
// 下层视图即使处在模态状态也不该拦按键 —— 用户看到的是栈顶那一屏，
// 按键就该归它。
//
// 从类型断言改成直接调（Modal 已经是 View 的一部分）之后，这里不再有
// "断言失败就当非模态"的静默分支；栈空的前提与兜底理由见 top()。
func (m Model) topModal() bool { return m.top().Modal() }

// pushCmd 造一条"进入下一层"的命令，省得每个视图都写一遍闭包。
//
// 返回 tea.Cmd 而不是 pushMsg：视图 Update 的签名要的是 Cmd，
// 直接返回消息的话每个调用点都得包一层 `func() tea.Msg { … }`。
//
// 名字跟 fillCmd 一样带 Cmd：它**只造命令、不动栈**（真正改栈的是根 Model
// 收到 pushMsg 之后），而 `pop` 是直接改栈的 —— 叫成 push 会跟 pop
// 看着对称、行为却不对称。
func pushCmd(v View) tea.Cmd {
	return func() tea.Msg { return pushMsg{v: v} }
}

// pushMsg 让任意视图请求"进入下一层"。
//
// 用消息而不是从 Update 直接返回新 Model：视图拿不到栈，
// 它只该说"我想深入这一项"，栈怎么变是根 Model 的事。
type pushMsg struct{ v View }

// popToMsg 请求根视图一直弹栈，直到栈顶是一个 ModelView，
// 然后把 msg 交给它。
//
// **"弹到 ModelView"而不是"弹 N 层"**：EntryView 可能来自
// ModelView → EntryView（从元数据跳进来，1 层），
// 也可能来自 ModelView → RefView → EntryView → EntryView
// （SeeAlso 里连跳几次，3 层）。让调用方写层数的话，
// 它必须自己数清楚，而数错的表现是弹到不相干的地方或者压根没弹 ——
// 两种都不报错，只是"跳转没反应"。
//
// 栈里没有 ModelView 时（不该发生，但状态机的东西不假设）
// 弹到只剩栈底，把消息交过去 —— 不认识的视图会忽略它。
type popToMsg struct{ msg tea.Msg }

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
		// **Ctrl+C 永远不被拦**：模态视图可能吃掉 q（那正是它的作用），
		// 忘了处理退出时用户还能有一条确定的出路。
		if msg.String() == "ctrl+c" {
			return m, tea.Quit
		}
		// 模态视图独占按键：能输入文字的框必须拿到 q 与 Esc，
		// 否则过滤 "qwen" 的第一下就退出程序、第一下 Esc 就跳回上一层。
		if m.topModal() {
			return m.forward(msg)
		}
		switch msg.String() {
		case "q":
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

	case popToMsg:
		for len(m.stack) > 1 {
			if _, ok := m.stack[len(m.stack)-1].(ModelView); ok {
				break
			}
			m.stack = m.stack[:len(m.stack)-1]
		}
		return m.forward(msg.msg)

	}
	return m.forward(msg)
}

// forward 把消息交给栈顶视图，并把它的返回值接回栈里。
//
// **栈顶可能是被替换过的**（视图处理消息后返回一个新视图），
// 所以要写回 stack[len-1]。
func (m Model) forward(msg tea.Msg) (tea.Model, tea.Cmd) {
	i := len(m.stack) - 1
	next, cmd := m.top().Update(msg)
	m.stack[i] = next
	return m, cmd
}

func (m Model) pop() (tea.Model, tea.Cmd) {
	m.stack = m.stack[:len(m.stack)-1]
	return m, nil
}

func (m Model) View() string {
	top := m.top()
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
//
// 原先拿不到 Title() 就退回默认的 "modelview"：忘实现的视图不会报错、
// 也不会崩，只是标题栏显示程序名而不是这一屏的名字 —— 看着像对的。
// Title 合进 View 之后没有这个退路了。
func (m Model) header() string {
	return styleTitle.Render(m.top().Title())
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

// listWindow 算出要渲染的区间 [start,end)，以及还有没有没显示完的
// （该不该打一行"显示第 N–M 个"的范围提示）。
//
// **提示行必须先扣**：反过来（先按满容量取窗口、再砍一行放提示）砍掉的是
// 窗口的**最后一行** —— 光标停在末尾时那正是用户选着的那一条，而提示里的
// start/end 来自截断前的窗口，屏幕上印的是一句对不上的话
// （实测：显示 101–118，提示写"显示第 102–120 个"）。
//
// showHint 用 `end-start < capacity` 表达"这一行放得下吗"，
// **不再依赖调用方把自己拼的行数与 capacity 保持同步** ——
// 那正是 Library 抄错的地方：它有两个可选尾行（范围提示 + 在途进度），
// 却只按一个扣减，容量只剩 1 行时提示与进度一起被 padTo 顶掉
// （这个模式在四处各写一遍，错了三次；所以判定收在这一个函数里）。
//
// capacity 是**扣掉标题、提示行等之后留给列表的总行数**，包含范围提示行：
// 有内容没显示完时这里会自己让出一行给它。
func listWindow(total, cursor, capacity int) (start, end int, showHint bool) {
	rows := capacity
	if total > rows && rows > 1 {
		rows--
	}
	start, end = window(total, cursor, rows)
	showHint = (start > 0 || end < total) && end-start < capacity
	return start, end, showHint
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
