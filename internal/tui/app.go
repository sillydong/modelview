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

	"github.com/sillydong/modelview/internal/model"
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

// modelProvider 让根视图沿栈找到"当前上下文里的模型"。
//
// 用可选接口而不是让根 Model 存一个"最近看过的模型"字段：
// 存字段就有"什么时候该清"的问题（用户退回模型库之后那个字段还活着吗），
// 而沿栈找是**派生**的 —— 栈里没有 ModelView 时它自然是 nil，
// 不可能变陈旧。这与 topModal() 用"看栈顶"而不是存标志位是同一个理由。
//
// 返回值允许是 nil：ModelView 还在异步解析时 m 就是 nil，
// 而"还没解析出来"与"没有上下文"对速查表是同一种事实 —— 不知道。
type modelProvider interface{ ContextModel() *model.Model }

// contextModel 沿栈自上而下找第一个提供上下文的视图，都没有就返回 nil。
//
// **找到第一个就停**：那一个是这一屏的答案，它说 nil 就是"不知道"。
//
// 这条规则今天与"跳过返回 nil 的那层、继续往下找"**不可区分**（栈里
// 最多只有一个 ModelView），也就是说后者是一个等价变异、任何测试都
// 打不死。写成"找到就停"是因为它在**任何**栈形下都答得对：将来多一个
// provider 时，"还没解析出来"不会被下层某个模型的上下文顶替掉 ——
// 那时屏幕上显示的是"正在解析…"，而条目页却会显示另一个模型的
// "在本模型中"。
func (m Model) contextModel() *model.Model {
	for i := len(m.stack) - 1; i >= 0; i-- {
		if p, ok := m.stack[i].(modelProvider); ok {
			return p.ContextModel()
		}
	}
	return nil
}

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
		case "backspace":
			// spec §3：Esc / Backspace 都返回上一层。
			//
			// **输入态走不到这里** —— 能输入文字的视图在 topModal() 那条
			// 分支就被整屏接管了，所以过滤框/搜索框里的退格仍然是删字。
			// 这正是想要的分工：正在打字时退格是"删一个字符"，
			// 不在打字时退格是"退回去"。
			//
			// 这条分支必须在模态判定**之后**（它就在这个 switch 里）：
			// 提到前面的话，过滤 "abc" 的第一下退格会变成弹栈。
			// TestApp_输入态退格仍归输入框 钉的就是这个顺序。
			if len(m.stack) > 1 {
				return m.pop()
			}
			// 根视图的退格与根视图的 Esc 一样不退出程序
			return m, nil
		case keyHelp:
			// **这条分支必须在上面那个模态判定之后**（它就在这个 switch 里）：
			// 过滤框里的 `?` 是用户要输入的一个字符，不是快捷键 ——
			// 判据挪到模态判定之前的话，在过滤框里永远打不出问号。
			// 这个顺序**只有经根视图才测得到**：直接调 view.Update 的测试
			// 绕过了全局按键，那正是本文件开头那段注释讲的漏法。
			//
			// 不带上下文的模型时（模型库那一屏）传 nil 是对的：
			// 条目页的"在本模型中"那时整节不显示 —— 事实是"不知道"。
			return m, pushCmd(NewRefView(m.contextModel()))
		}

	case pushMsg:
		m.stack = append(m.stack, msg.v)
		// **新压进来的视图也要跑它的 Init** ——
		// 不跑的话"进模型视图"会永远停在"正在解析…"
		return m, msg.v.Init()

	case batchScannedMsg:
		// **扫描消息不归栈顶**：用户可能正开着速查表或张量列表在看，
		// 它们不处理这条消息，链会断在那里。理由与栈里没有 ModelView
		// 时的取舍写在 routeToModelView 上。
		return m.routeToModelView(msg)

	case tensorScannedMsg, tickMsg:
		// 与 batchScannedMsg 同一个理由。`?` 就列在详情页的帮助栏里，
		// 任何时刻都能按 —— 交给栈顶的话扫描结果被咽掉，回来时
		// scanning 还是 true、tn.Stats 还是 nil，spinner 永远转下去。
		// tickMsg 是它有节奏的心跳，断了的话动画也会停住。
		return m.routeToTensorView(msg)

	case itemFilledMsg:
		// 模型库首屏「先列文件、再逐个读头部」那条链的结果。
		// 被栈顶吃掉的话，回来时进度行永远停在 0/N，
		// 那些行的格式与参数量永远是空的。
		return m.routeToLibrary(msg)

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

// routeToModelView 把**只属于 ModelView** 的消息送给栈里那个 ModelView，
// 而不是栈顶。
//
// 栈顶可能是用户中途推上来的别的视图（按 `?` 开速查表、Enter 进张量列表），
// 它不处理这条消息 —— 消息被咽掉，ModelView 那条自续的链就断在那里，
// 而它的 `scanning` 还是 true、进度行继续写着"正在扫描…"，**冻在一个
// 数字上不动**（实测：扫描中按 `?`，回来时停在第 7 张）。
// 扫描是**后台**的（spec §8），后台就不该因为用户翻到别的屏而停。
//
// 这与 popToMsg 是同一个理由：**消息该归谁，由"谁是那个视图"决定，
// 不是由"谁恰好在栈顶"决定**。区别是 popToMsg 会弹栈，这里只转发。
//
// 栈里没有 ModelView 时**丢弃**：那条链的宿主已经不在栈上了（用户按 Esc
// 退回了模型库），它的状态随视图一起没了 —— 屏幕上没有进度行、没有东西
// 可续，也没有东西需要清。交给栈顶是同一个结果（没有别的视图处理
// 这个消息类型），但那样会把"这条消息只归 ModelView"这条规则藏起来。
func (m Model) routeToModelView(msg tea.Msg) (tea.Model, tea.Cmd) {
	for i := len(m.stack) - 1; i >= 0; i-- {
		mv, ok := m.stack[i].(ModelView)
		if !ok {
			continue
		}
		next, cmd := mv.Update(msg)
		m.stack[i] = next
		return m, cmd
	}
	return m, nil
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

// routeToTensorView 把只属于 TensorView 的消息送给栈里那个 TensorView。
//
// 与 routeToModelView 是同一个理由（见那里的说明）：栈顶可能是用户
// 中途推上来的速查表或条目页。交给栈顶的话扫描结果被咽掉，
// 回来时 scanning 还是 true、tn.Stats 还是 nil，spinner 永远转下去。
//
// 归**栈里最靠上的**那个：同一时刻只有一个详情页在扫。
// 栈里没有时丢弃 —— 与 routeToModelView 同一个取舍，那条链的宿主
// 已经不在栈上了，它的状态随视图一起没了。
func (m Model) routeToTensorView(msg tea.Msg) (tea.Model, tea.Cmd) {
	for i := len(m.stack) - 1; i >= 0; i-- {
		tv, ok := m.stack[i].(TensorView)
		if !ok {
			continue
		}
		next, cmd := tv.Update(msg)
		m.stack[i] = next
		return m, cmd
	}
	return m, nil
}

// routeToLibrary 把只属于 Library 的消息送给栈里那个 Library。
//
// itemFilledMsg 是模型库首屏「先列文件、再逐个读头部」那条链的结果。
// 用户在这期间按 `?`，顶层是速查表 —— 被咽掉的话回来时进度行
// 永远停在 0/N，那些行的格式与参数量永远是空的。
func (m Model) routeToLibrary(msg tea.Msg) (tea.Model, tea.Cmd) {
	for i := len(m.stack) - 1; i >= 0; i-- {
		lib, ok := m.stack[i].(Library)
		if !ok {
			continue
		}
		next, cmd := lib.Update(msg)
		m.stack[i] = next
		return m, cmd
	}
	return m, nil
}
