package tui

import (
	"context"
	"errors"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/sillydong/modelview/internal/discover"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/ref"
)

// fakeView 是最小的 View 实现，只回显自己的标题。
//
// 放在测试文件里而不是生产代码里：它没有生产用途，
// 而放在生产代码里会让"这个类型还有别的实现吗"变成一个要翻代码的问题。
type fakeView struct{ title string }

func (f fakeView) Init() tea.Cmd                  { return nil }
func (f fakeView) Update(tea.Msg) (View, tea.Cmd) { return f, nil }
func (f fakeView) View(w, h int) string           { return f.title }
func (f fakeView) Help() []string                 { return nil }
func (f fakeView) Title() string                  { return f.title }

// Modal 恒为 false —— 要模态行为的测试用 modalFakeView，它才带开关。
func (f fakeView) Modal() bool { return false }

// key 造一个按键消息，省得每个测试都写一遍。
func key(s string) tea.KeyMsg {
	switch s {
	case "enter":
		return tea.KeyMsg{Type: tea.KeyEnter}
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEsc}
	case "up":
		return tea.KeyMsg{Type: tea.KeyUp}
	case "down":
		return tea.KeyMsg{Type: tea.KeyDown}
	case "tab":
		// **必须显式映射**：不映射的话会落到 default 分支，造出一个
		// Type=KeyRunes、内容是字面量 "tab" 的按键。它对 `msg.String()`
		// 恰好等价于真的 Tab（key.go 的 keyNames 里 keyHT 就是 "tab"），
		// 所以测试一直绿着 —— 但"按了 Tab"与"粘贴了 t/a/b 三个字符"
		// 是两回事，靠巧合一致的东西会在实现改判 `msg.Type` 时静默失效。
		return tea.KeyMsg{Type: tea.KeyTab}
	case "ctrl+c":
		return tea.KeyMsg{Type: tea.KeyCtrlC}
	case "backspace":
		// **必须显式映射**：不映射的话会落到 default 分支，造出一个
		// Type=KeyRunes、内容是字面量 "backspace" 的按键 ——
		// 测试看着在验退格，实际在往过滤词里打那九个字母，
		// 而且因为没有断言"只删了一个字"以外的差别，它会一直绿。
		return tea.KeyMsg{Type: tea.KeyBackspace}
	default:
		return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
	}
}

// 尺寸消息必须让界面记住 —— 所有 View 都按它算换行。
func TestApp_记住终端尺寸(t *testing.T) {
	m := New(fakeView{title: "根"})
	m2, _ := m.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	got := m2.(Model)
	if got.width != 100 || got.height != 30 {
		t.Errorf("尺寸 = %dx%d, want 100x30", got.width, got.height)
	}
}

// q 与 Ctrl+C 都要能退出 —— 只支持一个的话，
// 习惯了另一个的用户会以为程序卡死。
func TestApp_q与CtrlC都退出(t *testing.T) {
	for _, k := range []tea.KeyMsg{key("q"), key("ctrl+c")} {
		m := New(fakeView{title: "根"})
		_, cmd := m.Update(k)
		if cmd == nil {
			t.Errorf("按 %v 没有返回退出命令", k)
			continue
		}
		if _, ok := cmd().(tea.QuitMsg); !ok {
			t.Errorf("按 %v 返回的不是退出命令", k)
		}
	}
}

// 栈里只剩根视图时，Esc **不该退出程序** ——
// 用户在根视图按 Esc 想的是"取消当前操作"，不是"关掉程序"。
func TestApp_根视图按Esc不退出(t *testing.T) {
	m := New(fakeView{title: "根"})
	m2, cmd := m.Update(key("esc"))
	if cmd != nil {
		if _, isQuit := cmd().(tea.QuitMsg); isQuit {
			t.Error("在根视图按 Esc 退出了程序")
		}
	}
	if len(m2.(Model).stack) != 1 {
		t.Error("在根视图按 Esc 把根视图也弹掉了")
	}
}

// 视图栈：push 之后 Esc 回到上一层。
func TestApp_视图栈(t *testing.T) {
	m := New(fakeView{title: "根"})

	m2, _ := m.Update(pushMsg{v: fakeView{title: "第二层"}})
	if n := len(m2.(Model).stack); n != 2 {
		t.Fatalf("push 之后栈深 %d, want 2", n)
	}
	if !strings.Contains(m2.(Model).View(), "第二层") {
		t.Errorf("View 没显示栈顶视图的内容:\n%s", m2.(Model).View())
	}

	m3, _ := m2.(Model).Update(key("esc"))
	if n := len(m3.(Model).stack); n != 1 {
		t.Fatalf("Esc 之后栈深 %d, want 1", n)
	}
	if !strings.Contains(m3.(Model).View(), "根") {
		t.Errorf("Esc 之后没回到根视图:\n%s", m3.(Model).View())
	}
}

// 渲染必须**恰好占满给定的高度**，且不超宽。
//
// 高度不补的话，底部帮助栏会浮在内容下面而不是贴着屏幕底，
// 内容短的时候整个界面还会随视图切换上下跳。
// 超宽会让终端折行，界面错位 —— 这是 TUI 最常见的破相方式。
func TestApp_渲染不超宽且占满高度(t *testing.T) {
	m := New(fakeView{title: "根"})
	m2, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	m = m2.(Model)

	out := m.View()
	lines := strings.Split(out, "\n")
	if len(lines) != 24 {
		t.Errorf("渲染了 %d 行，终端高 24 —— 超出的行会被截掉、不足会留白", len(lines))
	}
	for i, l := range lines {
		if w := lipgloss.Width(l); w > 80 {
			t.Errorf("第 %d 行宽 %d 超过 80: %q", i, w, l)
		}
	}
}

// 内容比屏幕高的视图要被截到屏幕高度，不能把帮助栏顶出屏幕。
func TestApp_内容过长被截断(t *testing.T) {
	long := strings.Repeat("一行内容\n", 100)
	m := New(viewWithBody{body: long})
	m2, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 10})
	m = m2.(Model)

	out := m.View()
	if n := len(strings.Split(out, "\n")); n != 10 {
		t.Errorf("内容 100 行、屏幕 10 行，渲染出 %d 行 —— 应当截到 10 行", n)
	}
}

// 还没收到尺寸消息时也要能渲染（初始帧）—— 返回空串会让界面闪一下。
func TestApp_没有尺寸时也能渲染(t *testing.T) {
	m := New(fakeView{title: "根"})
	if out := m.View(); out == "" {
		t.Error("没有尺寸消息时 View() 返回空串 —— 首帧会闪")
	}
}

// viewWithBody 是内容可定制的测试视图（fakeView 只有一行）。
type viewWithBody struct{ body string }

func (v viewWithBody) Init() tea.Cmd                  { return nil }
func (v viewWithBody) Update(tea.Msg) (View, tea.Cmd) { return v, nil }
func (v viewWithBody) View(w, h int) string           { return v.body }
func (v viewWithBody) Help() []string                 { return nil }
func (v viewWithBody) Title() string                  { return "长内容" }
func (v viewWithBody) Modal() bool                    { return false }

// **内容超宽必须被截掉，不能留给终端折行。**
//
// 这条是变异验证逼出来的：上面那条"不超宽"断言用的内容都短于 80 列，
// 于是截断那段代码从来没被执行过 —— 把它整段删掉，测试照样绿。
// 超宽不截的后果是终端自动折行，整个界面往下错位、帮助栏跑出屏幕。
func TestApp_内容超宽被截断(t *testing.T) {
	wide := strings.Repeat("很宽的一行", 40) // 5×3×40 = 600 显示列
	m := New(viewWithBody{body: wide})
	m2, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 10})
	m = m2.(Model)

	for i, l := range strings.Split(m.View(), "\n") {
		if w := lipgloss.Width(l); w > 80 {
			t.Errorf("第 %d 行宽 %d 超过 80 —— 截断没生效:\n%q", i, w, l)
		}
	}
}

// 标题与帮助栏超宽同样要截 —— 它们是另外两处调用点。
func TestApp_标题与帮助栏超宽被截断(t *testing.T) {
	// 用有超长标题与超长帮助栏的视图：标题那一格必须真的超长，
	// 否则测不到标题那一处截断
	m, _ := New(longTitleView{}).Update(tea.WindowSizeMsg{Width: 20, Height: 6})
	for i, l := range strings.Split(m.(Model).View(), "\n") {
		if w := lipgloss.Width(l); w > 20 {
			t.Errorf("第 %d 行宽 %d 超过 20: %q", i, w, l)
		}
	}
}

type longTitleView struct{}

func (longTitleView) Init() tea.Cmd                  { return nil }
func (longTitleView) Update(tea.Msg) (View, tea.Cmd) { return longTitleView{}, nil }
func (longTitleView) View(w, h int) string           { return "内容" }
func (longTitleView) Help() []string {
	return []string{strings.Repeat("很长的帮助文本", 10)}
}
func (longTitleView) Title() string { return strings.Repeat("超长标题", 20) }
func (longTitleView) Modal() bool   { return false }

// **Init 必须真的被调用** —— 忘了接上的表现是"界面永远停在初始状态"，
// 而所有直接调 Update 的单元测试都是绿的。
//
// 实测：Library.Init 里发起扫描的那条命令从来没被执行过，
// 屏幕停在"正在扫描模型目录…"，是 pty 里跑真终端才看出来的。
func TestApp_Init会被调用(t *testing.T) {
	ran := false
	m := New(initTrackingView{onInit: func() { ran = true }})

	cmd := m.Init()
	if cmd == nil {
		t.Fatal("Model.Init() 返回 nil —— 根视图的初始化任务没被接上")
	}
	cmd()
	if !ran {
		t.Error("根视图的 Init() 没有被调用")
	}
}

// **压栈（pushMsg）**进来的视图也要跑它的 Init —— 不跑的话
// "进模型视图"会停在"正在解析…"。
//
// 名字里写"压栈"而不是"push"：`pushCmd` 是造命令的辅助函数、
// `pushMsg` 才是"压栈"这件事，两个都叫 push 的话读名字分不出这条测的是哪个。
func TestApp_压栈的视图也会跑Init(t *testing.T) {
	ran := false
	m := New(fakeView{title: "根"})
	_, cmd := m.Update(pushMsg{v: initTrackingView{onInit: func() { ran = true }}})
	if cmd == nil {
		t.Fatal("pushMsg 没有返回新视图的 Init 命令")
	}
	cmd()
	if !ran {
		t.Error("push 进来的视图的 Init() 没有被调用")
	}
}

// initTrackingView 记录自己的 Init 被调用过。
type initTrackingView struct{ onInit func() }

func (v initTrackingView) Init() tea.Cmd {
	return func() tea.Msg {
		v.onInit()
		return nil
	}
}
func (v initTrackingView) Update(tea.Msg) (View, tea.Cmd) { return v, nil }
func (v initTrackingView) View(w, h int) string           { return "跟踪" }
func (v initTrackingView) Help() []string                 { return nil }
func (v initTrackingView) Title() string                  { return "跟踪" }
func (v initTrackingView) Modal() bool                    { return false }

// modalFakeView 是一个会声明自己"现在要独占按键"的视图。
//
// 它记录收到的按键 —— 这是断言的关键：只看根视图返回的命令
// 分不清"视图拿到了 q"和"谁都没拿到 q"（两种情况下 cmd 都是 nil）。
type modalFakeView struct {
	fakeView
	modal bool
	seen  []string
}

func (v modalFakeView) Update(msg tea.Msg) (View, tea.Cmd) {
	if k, ok := msg.(tea.KeyMsg); ok {
		v.seen = append(v.seen, k.String())
	}
	return v, nil
}

// Modal 让这个视图可以选择性地声明独占。
func (v modalFakeView) Modal() bool { return v.modal }

// 模态视图必须拿到 q —— 否则用户在过滤框里敲 "qwen" 的第一下就退出了程序。
//
// **这条测试必须经过根 Model**：直接调 view.Update 的话，
// "根 Model 把按键吞了"这件事根本不在被测范围里，测试会假绿。
func TestApp_模态视图能拿到q(t *testing.T) {
	root := New(fakeView{title: "根"})
	pushed, _ := root.Update(pushMsg{v: modalFakeView{modal: true}})
	m := pushed.(Model)

	next, cmd := m.Update(key("q"))
	if cmd != nil {
		if _, isQuit := cmd().(tea.QuitMsg); isQuit {
			t.Fatal("模态视图下按 q 退出了程序 —— 用户没法筛 qwen")
		}
	}
	top := next.(Model).stack[len(next.(Model).stack)-1].(modalFakeView)
	if len(top.seen) != 1 || top.seen[0] != "q" {
		t.Errorf("视图收到的按键 = %v, want [q]", top.seen)
	}
}

// 模态视图下 Esc 也不弹栈 —— 它是"取消输入"，不是"返回上一层"。
func TestApp_模态视图下Esc不弹栈(t *testing.T) {
	root := New(fakeView{title: "根"})
	pushed, _ := root.Update(pushMsg{v: modalFakeView{modal: true}})
	m := pushed.(Model)

	next, _ := m.Update(key("esc"))
	m2 := next.(Model)
	if len(m2.stack) != 2 {
		t.Fatalf("栈深 = %d, want 2 —— Esc 被根视图抢先弹栈了", len(m2.stack))
	}
	if top := m2.stack[1].(modalFakeView); len(top.seen) != 1 || top.seen[0] != "esc" {
		t.Errorf("视图收到的按键 = %v, want [esc]", top.seen)
	}
}

// Ctrl+C 在模态下仍然退出，而且**不转给视图**。
//
// 把这条保证放在根视图，而不是"要求每个模态视图自己记得处理退出"：
// 后者漏一个就是用户被困住，且没有任何东西会红。
// 按 q 是不能指望的 —— 它可能被当成输入吃掉，那正是模态的意义。
func TestApp_模态视图下CtrlC仍退出(t *testing.T) {
	root := New(fakeView{title: "根"})
	pushed, _ := root.Update(pushMsg{v: modalFakeView{modal: true}})
	m := pushed.(Model)

	next, cmd := m.Update(key("ctrl+c"))
	if cmd == nil {
		t.Fatal("模态视图下 Ctrl+C 没有退出 —— 用户被困住了")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Error("Ctrl+C 返回的不是退出命令")
	}
	// 断言 Update **返回的**模型，而不是 Update 之前的 `m`：读 `m.stack[1]`
	// 只是靠 slice 共享底层数组侥幸成立（forward 里是 `m.stack[top] = next`
	// 原地写）—— 哪天有人给 Update 加一句 `slices.Clone(m.stack)`，
	// 那条断言会静默变成空转：永远绿，什么也不验。
	if top := next.(Model).stack[1].(modalFakeView); len(top.seen) != 0 {
		t.Errorf("Ctrl+C 被转给了视图: %v", top.seen)
	}
}

// 非模态视图的行为一个字都不能变：q 退出、Esc 弹栈。
// 这条是回归保护 —— 改全局按键最容易伤到的就是这里。
func TestApp_非模态视图q与Esc照旧(t *testing.T) {
	root := New(fakeView{title: "根"})
	pushed, _ := root.Update(pushMsg{v: modalFakeView{modal: false}})
	m := pushed.(Model)

	_, cmd := m.Update(key("q"))
	if cmd == nil {
		t.Fatal("非模态视图下 q 没退出")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Error("q 返回的不是退出命令")
	}

	next, _ := m.Update(key("esc"))
	if n := len(next.(Model).stack); n != 1 {
		t.Errorf("Esc 之后栈深 = %d, want 1", n)
	}
}

// **只有栈顶的模态性算数**：栈底模态、栈顶非模态时，按键不该被拦。
//
// 这条守的是 topModal 注释里那句"只看栈顶"。不测的话，
// 把实现改成"栈里任何一层模态就独占"也能过 —— 而那种实现下，
// 用户在栈顶（非模态）按 q 会发现退不出去，因为底下某一层还开着输入框。
func TestApp_只看栈顶的模态性(t *testing.T) {
	root := New(modalFakeView{modal: true})                    // 栈底：模态
	pushed, _ := root.Update(pushMsg{v: fakeView{title: "顶"}}) // 栈顶：非模态
	m := pushed.(Model)

	_, cmd := m.Update(key("q"))
	if cmd == nil {
		t.Fatal("栈顶非模态时 q 被拦住了 —— 用户退不出去")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Error("q 返回的不是退出命令")
	}
}

// **连续两次按键之后仍是模态的** —— 第二下才暴露的那类丢失。
//
// 根视图把 Update 的返回值写回栈顶（app.go 的 forward），所以任何一条
// 分支返回了别的具体类型（或返回一个输入态已经为假的副本），模态性都会
// 在第一下按键之后静默消失：第一下进输入态正常，第二下敲 q 就退出程序。
// 只测一次按键看不出这个。
//
// **两个模态视图各走一格**：`Modal()` 是各视图自己实现的判据
// （TensorsView.filtering / RefView.searching），合进 View 只堵住了
// "忘了实现"与"接收者写成指针"，堵不住"Update 把状态改回去"。
// 只守一个的话，另一个丢模态的表现同样是"在输入框里敲 q 退出程序"，
// 而没有任何测试会红。
func TestApp_模态性经过Update仍在(t *testing.T) {
	cases := []struct {
		name string
		view View
		// 两个视图把输入的词记在各自字段上，断言只能是闭包 ——
		// 硬抽一层公共抽象的话，抽出来的东西比它替掉的两行还长
		check func(t *testing.T, top View)
	}{
		{
			name: "TensorsView",
			view: newTensors(fakeModelWithTensors(5)),
			check: func(t *testing.T, top View) {
				tv, ok := top.(TensorsView)
				if !ok {
					t.Fatalf("栈顶是 %T，want TensorsView —— Update 返回了别的具体类型", top)
				}
				if tv.filter != "q" {
					t.Errorf("过滤词 = %q, want q —— q 没被视图收下", tv.filter)
				}
			},
		},
		{
			name: "RefView",
			view: NewRefView(nil),
			check: func(t *testing.T, top View) {
				rv, ok := top.(RefView)
				if !ok {
					t.Fatalf("栈顶是 %T，want RefView —— Update 返回了别的具体类型", top)
				}
				if rv.search != "q" {
					t.Errorf("搜索词 = %q, want q —— q 没被视图收下", rv.search)
				}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := New(fakeView{title: "根"})
			pushed, _ := root.Update(pushMsg{v: tc.view})
			m := pushed.(Model)

			// 第一下 `/` 走的是根视图末尾的兜底 forward（此时还不是模态的），
			// 第二下 `q` 只能走模态分支 —— Update 某条分支返回了别的具体类型的话，
			// 这一下就会被根视图当成全局的"退出"。
			for i, k := range []string{"/", "q"} {
				next, cmd := m.Update(key(k))
				if cmd != nil {
					if _, isQuit := cmd().(tea.QuitMsg); isQuit {
						t.Fatalf("第 %d 下按键（%q）触发了退出 —— 模态性在第一下之后丢了", i+1, k)
					}
				}
				m = next.(Model)
			}

			top := m.stack[len(m.stack)-1]
			tc.check(t, top)
			if !top.Modal() {
				t.Error("栈顶不再声明模态 —— 此时敲 q 会退出程序")
			}
		})
	}
}

// **根视图不能吞按键** —— 这条以前没人钉住。
//
// internal/tui 的既有测试全部直接调 view.Update（library_test /
// modelview_test 无一例外），所以"根视图不转发按键"这类回归
// 一条测试都抓不到。实测：在 `case tea.KeyMsg` 的内层 switch 之后
// 插一句 `return m, nil`，**已提交的全部测试仍然全绿** ——
// 而真终端里模型库的 ↓/j/↑/k/r/enter 全部失效，整个界面不可导航。
//
// 这正是 app.go 开头那段注释讲的漏法（单测绿、真终端坏），
// 只不过方向相反：模态那次是根视图**多**处理了按键，
// 这条守的是它**少**转了按键。
//
// 三格各自守哪条路径，是实测出来的分工、不是设计出来的：
//   - 非模态两格走根视图末尾的兜底 forward，守的是那句转发；
//   - 模态那格走的是模态分支的 early return，**到不了兜底** ——
//     把模态分支改成 `return m, nil` 时只有它会红，而把兜底掐掉时
//     它仍然绿。别把它当成兜底的守卫。
func TestApp_根视图不吞按键(t *testing.T) {
	cases := []struct {
		name  string
		modal bool
		key   string
	}{
		{"非模态+j", false, "j"},
		{"非模态+up", false, "up"},
		{"模态+j", true, "j"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := New(fakeView{title: "根"})
			pushed, _ := root.Update(pushMsg{v: modalFakeView{modal: tc.modal}})
			m := pushed.(Model)

			next, _ := m.Update(key(tc.key))
			top := next.(Model).stack[len(next.(Model).stack)-1].(modalFakeView)
			if len(top.seen) != 1 || top.seen[0] != tc.key {
				t.Errorf("栈顶视图收到的按键 = %v, want [%s] —— "+
					"根视图把普通按键吞了，真终端里整个界面会不可导航",
					top.seen, tc.key)
			}
		})
	}
}

// namedView 给跨视图的约定测试用：一个视图 + 它在报告里的名字。
type namedView struct {
	name string
	v    View
}

// allViews 把当前**每一个** View 实现各造一个实例。
//
// 跨视图的约定（末尾换行、行数与高度）只有把每个实现都列进来才守得住 ——
// 漏一个就是一条没人看的路径。加了新视图就往这里加一行。
func allViews() []namedView {
	lib, _ := fakeLibrary()
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	// Library 的另外两条早退分支各来一档，尤其是**空库**：
	// 它是首次运行就会撞上的第一屏（本机一个模型都没有），
	// 而原先这一格固定用非空的 fakeLibrary() —— 空库那一支谁都没看，
	// 实测它的 emptyView 以换行结尾（padTo 会因此多砍一行、
	// 把"还有 N 行"说大一）。"正在扫描"那一档不跑 Init()，不碰真实磁盘。
	emptyLib := NewLibrary()
	emptyLib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{}
	}
	emptyLib2, _ := emptyLib.Update(emptyLib.Init()())
	emptyLib = emptyLib2.(Library)

	m := fakeModel()
	mv := loadModelView(m)

	// TensorView 要**两条分支各来一个**：扫描中 / 已出结果。
	// 只造"扫描中"的那个，`results()` 那一支的末尾换行就没人看 ——
	// 变异验证确认过：把它的 TrimSuffix 去掉，测试照样绿。
	// 整数类型 + 有 Stats 是 NeedsWork 为假的捷径（既非量化也非浮点）。
	done := &model.Tensor{Name: "done.weight", Dims: []int64{4, 4},
		Dtype: model.DtypeI32, ByteSize: 64, ParamCount: 16,
		Stats: &model.Stats{Count: 16}}
	m2 := fakeModel()
	m2.Tensors = append(m2.Tensors, done)

	// "空库 + 安全提示"是**空库那一段与 notices 拼起来**的分支：两条
	// 都在，且顺序是提示在前、说明在后。`res.Errs` 本身**仍不是独立分支**
	//（它永远不是末行：后面要么跟列表、要么跟空库说明），但这一档的
	// fixture 里带了 Errs，顺带把 wrapText 那条折行路径也纳进跨视图守卫。
	warnLib := emptyLibraryWithWarnings()

	// 扫描中 / 扫描完成两态：进度行是**另一条渲染分支**（View 里多拼一行、
	// 并且把内容区高度扣掉一行），不摆出来的话"末尾不留换行"这条守卫
	// 看不到它。走的是真实的链（这里没有 t 也能跑：只有两张张量）。
	scanning := mv
	scanning2, _ := scanning.Update(key(keyScanAll))

	// **下面几档是为"帮助栏的分支"补的**（都走真实的按键路径）：
	// 每类视图原先只有一个 fixture，于是另一些 Help 分支上的键掉了
	// 没有任何测试会红（实测：把下面三支里的 `?` 删掉，全绿）。
	metaFocus := gotoSection(mv, sectionMetadata)
	mf, _ := metaFocus.Update(key("tab"))
	metaFocus = mf.(ModelView)

	searched := NewRefView(m)
	sv, _ := searched.Update(key("/"))
	searched = sv.(RefView)
	sv, _ = searched.Update(key("Q4"))
	searched = sv.(RefView)
	sv, _ = searched.Update(key("enter")) // 确认搜索：非模态、Help 是"改搜索词"那一支
	searched = sv.(RefView)

	// 模态两档：`?` 在输入态**不许列**（那是用户要打的字符），而"跳过模态"
	// 那一支此前没有任何 fixture 走到过 —— 给搜索态加上 `?` 也没人红。
	searchingRef := NewRefView(m)
	sr, _ := searchingRef.Update(key("/"))
	searchingRef = sr.(RefView)

	filtering := NewTensorsView(m)
	fv, _ := filtering.Update(key("/"))
	filtering = fv.(TensorsView)

	return []namedView{
		{"Library", lib},
		{"Library/空库", emptyLib},
		{"Library/空库+安全提示", warnLib},
		// **"正在扫描"那一档不跑 Init 是有意的**：跑它就会真的去 stat
		// 本机的模型目录，这个守卫的结论会随开发机上装了什么而变
		//（Library 的字段注释里记过同一条）。View 本身不碰 I/O。
		{"Library/载入中", NewLibrary()},
		{"ModelView/概览", mv},
		{"ModelView/元数据", gotoSection(mv, sectionMetadata)},
		{"ModelView/张量", gotoSection(mv, sectionTensors)},
		{"ModelView/量化分布", gotoSection(mv, sectionQuantDist)},
		{"ModelView/速查表", gotoSection(mv, sectionRef)},
		{"ModelView/扫描中", scanning2.(ModelView)},
		{"ModelView/扫描完成", scanAllQuiet(loadModelView(fakeModel()))},
		{"TensorsView", NewTensorsView(m)},
		{"TensorView/扫描中", NewTensorView(m, m.Tensors[0])},
		{"TensorView/已出结果", NewTensorView(m2, done)},
		// 名字里一个可解释的段都没有时的分支（Help 与 View 各少一段）。
		// 上面两个 fixture 的名字都含 `weight`，**碰不到这一支** ——
		// 少了它，"两个分支都要列 `?`"这条约定有一半没人看。
		{"TensorView/名字无段", NewTensorView(m, &model.Tensor{
			Name: "xyzzy", Dtype: model.DtypeF32, ByteSize: 4, ParamCount: 1})},
		{"ModelView/元数据焦点在右栏", metaFocus},
		{"RefView", NewRefView(nil)},
		{"RefView/已确认搜索", searched},
		{"RefView/搜索输入中", searchingRef},
		{"TensorsView/过滤输入中", filtering},
		{"EntryView", NewEntryView(nil, ref.Entry{ID: "quant:Q4_K", Title: "Q4_K"})},
		// 有跳得动的 SeeAlso 那一支（Help 里多一个"Enter 跳转"）
		{"EntryView/有跳转目标", NewEntryView(nil, ref.Entry{
			ID: "quant:Q4_K", Title: "Q4_K", SeeAlso: []string{"float:F16"}})},
	}
}

// **每个视图交给根视图的原始输出都不许以换行结尾。**
//
// padTo 是按 "\n" 切行数出来的：末尾那个空元素会让它**多算一行** ——
// 内容放不下时"…还有 N 行没显示"里的 N 比实际丢掉的多一（屏幕上印一个
// 错的数字），内容恰好放得下时还会被白白砍掉一行。
// 实测踩过三次：Library、TensorView、EntryView（joinHorizontal 的注释里
// 已经记过同一条，那三处都没走那条约定）。
//
// 这条从**各层自己的 View** 断言，不经根视图：经根视图是测不出来的 ——
// padTo 永远把结果整理成恰好 h 行（④b-1 的教训）。
func TestViews_原始输出不留末尾换行(t *testing.T) {
	for _, nv := range allViews() {
		for _, h := range []int{6, 20, 40} {
			out := nv.v.View(120, h)
			if strings.HasSuffix(out, "\n") {
				t.Errorf("%s（h=%d）：交给根视图的原始输出以换行结尾 —— "+
					"padTo 会因此多算一行、多砍一行，并把「还有 N 行」说大一",
					nv.name, h)
			}
		}
	}
}

// ===== `?` 全局速查表（spec §8.1"随时可开"）=====

// scanAllQuiet 把「扫描全部」跑到结束（假扫描，不碰磁盘）。
//
// 只有两张张量，所以循环有界；上界也写出来，免得哪天实现坏了变成
// 一条跑到测试超时的死循环（那种失败比断言失败难查得多）。
func scanAllQuiet(v ModelView) ModelView {
	v.scan = fakeScan(nil)
	next, cmd := v.Update(key(keyScanAll))
	v = next.(ModelView)
	for range len(v.m.Tensors) + 1 {
		if cmd == nil {
			break
		}
		next, cmd = v.Update(cmd())
		v = next.(ModelView)
	}
	return v
}

// **任何非模态的地方按 `?` 都能开速查表**，Esc 逐层退回。
func TestApp_问号随处可开(t *testing.T) {
	root := New(fakeView{title: "根"})
	next, cmd := root.Update(key(keyHelp))
	if cmd == nil {
		t.Fatal("按 `?` 没有返回命令 —— 速查表打不开")
	}
	m := runCmd(t, next.(Model), cmd)
	if n := len(m.stack); n != 2 {
		t.Fatalf("栈深 = %d, want 2", n)
	}
	if _, ok := m.stack[1].(RefView); !ok {
		t.Fatalf("栈顶是 %T, want RefView —— `?` 没推入速查表", m.stack[1])
	}

	// Esc 逐层退回（速查表 → 原来那一屏）
	back, _ := m.Update(key("esc"))
	if n := len(back.(Model).stack); n != 1 {
		t.Errorf("Esc 之后栈深 = %d, want 1", n)
	}
}

// openEntryByTitle 在速查表里搜一个标题、打开那条条目，返回新的根 Model。
//
// 走的是真实路径（`/` 进输入态 → 整段词一次进 → Enter 确认 → Enter 打开），
// 不直接构造 EntryView —— 直接构造的话，"上下文模型有没有从根视图
// 经 RefView 传到条目页"这件事根本不在被测范围里。
func openEntryByTitle(t *testing.T, m Model, title string) Model {
	t.Helper()
	for _, k := range []string{"/", title, "enter"} {
		next, _ := m.Update(key(k))
		m = next.(Model)
	}
	next, cmd := m.Update(key("enter"))
	if cmd == nil {
		t.Fatalf("速查表里搜 %q 之后按 Enter 打不开条目", title)
	}
	return runCmd(t, next.(Model), cmd)
}

// **上下文关联**：栈里有模型时按 `?`，速查表要带上它 ——
// 条目页的"在本模型中"那一节靠它算出来。
func TestApp_问号带上上下文模型(t *testing.T) {
	m := fakeModel()
	root := New(fakeView{title: "根"})
	pushed, _ := root.Update(pushMsg{v: loadModelView(m)})
	board := pushed.(Model)

	next, cmd := board.Update(key(keyHelp))
	board = runCmd(t, next.(Model), cmd)

	rv, ok := board.stack[len(board.stack)-1].(RefView)
	if !ok {
		t.Fatalf("栈顶是 %T, want RefView", board.stack[len(board.stack)-1])
	}
	if rv.m != m {
		t.Fatalf("速查表拿到的模型 = %v, want 模型视图里那一个", rv.m)
	}

	// 一路开到条目页：有了上下文，"在本模型中"那一段要算得出来
	board = openEntryByTitle(t, board, "MOSTLY_Q4_K_M")
	ev, ok := board.stack[len(board.stack)-1].(EntryView)
	if !ok {
		t.Fatalf("栈顶是 %T, want EntryView", board.stack[len(board.stack)-1])
	}
	if out := ev.View(120, 40); !strings.Contains(out, "在本模型中") {
		t.Errorf("条目页没显示「在本模型中」—— 上下文模型没传下去:\n%s", out)
	}
}

// **模型库那一屏没有上下文模型**：速查表拿到 nil，条目页不显示
// "在本模型中" —— 没有上下文时的事实是"不知道"，不是"没有"。
//
// 与上一条**用同一个条目**：两条路径的差别只有上下文，所以
// "条目页显示不显示"这件事只可能是上下文带来的。
func TestApp_模型库按问号没有上下文(t *testing.T) {
	lib, _ := fakeLibrary()
	lib2, _ := lib.Update(lib.Init()())
	board := New(lib2.(Library))

	next, cmd := board.Update(key(keyHelp))
	board = runCmd(t, next.(Model), cmd)

	rv, ok := board.stack[len(board.stack)-1].(RefView)
	if !ok {
		t.Fatalf("栈顶是 %T, want RefView", board.stack[len(board.stack)-1])
	}
	if rv.m != nil {
		t.Fatalf("模型库那一屏带上了模型 %v —— 那一屏没有上下文", rv.m)
	}

	board = openEntryByTitle(t, board, "MOSTLY_Q4_K_M")
	ev, ok := board.stack[len(board.stack)-1].(EntryView)
	if !ok {
		t.Fatalf("栈顶是 %T, want EntryView", board.stack[len(board.stack)-1])
	}
	out := ev.View(120, 40)
	// **先确认这一页真的渲染出来了**：否则"没有在本模型中"可能只是
	// 因为整页是空的（空页也满足 Contains == false）
	if !strings.Contains(out, "MOSTLY_Q4_K_M") || !strings.Contains(out, "相关条目") {
		t.Fatalf("条目页没渲染出内容，下面的否定断言会变成空转:\n%s", out)
	}
	if strings.Contains(out, "在本模型中") {
		t.Errorf("没有上下文模型却显示了「在本模型中」—— "+
			"用户会读成「这个模型里没有」:\n%s", out)
	}
}

// **模态视图必须优先拿到 `?` 与 `a`**：用户正在过滤框里想输入一个问号
// （或者想筛名字里带 a 的张量），不能因为根视图抢走它们而弹出速查表、
// 或者开始扫描。
//
// 这条**必须经过根视图**：直接调 view.Update 绕过了全局按键那一层，
// 而"根视图把按键吞了"正是那一层的事（app.go 开头那段注释讲的漏法）。
//
// **两格今天证明的东西不一样，别当成一样强**：
//   - `?`：根视图里真的有那条全局分支，判据挪到模态判定之前这格就会红
//     （变异验证过）；
//   - `a`：根视图里**没有**全局 `a`（它只归 ModelView），所以这格今天
//     走的是兜底转发那条路 —— 它守的是"将来有人把 `a` 加进全局 switch 时
//     别加在模态判定之前"，而不是现存的某个分支。
func TestApp_模态下可打印键不被全局抢走(t *testing.T) {
	for _, k := range []string{keyHelp, keyScanAll} {
		t.Run(k, func(t *testing.T) {
			root := New(fakeView{title: "根"})
			pushed, _ := root.Update(pushMsg{v: newTensors(fakeModelWithTensors(3))})
			m := pushed.(Model)

			// 先进过滤态（`/` 这时还不是模态的，走根视图末尾的兜底转发）
			opened, _ := m.Update(key("/"))
			m = opened.(Model)

			next, cmd := m.Update(key(k))
			m = next.(Model)
			if cmd != nil {
				t.Errorf("模态下按 %q 返回了命令 —— 根视图把它当全局键处理了", k)
			}
			if n := len(m.stack); n != 2 {
				t.Fatalf("栈深 = %d, want 2 —— 按 %q 推入了新视图", n, k)
			}
			tv, ok := m.stack[1].(TensorsView)
			if !ok {
				t.Fatalf("栈顶是 %T, want TensorsView", m.stack[1])
			}
			if tv.filter != k {
				t.Errorf("过滤词 = %q, want %q —— 这个字符没进输入框", tv.filter, k)
			}
		})
	}
}

// 帮助栏里 `?` 这一格的**两个方向**都要守住：
//   - 非模态的每一屏按 `?` 都真的能开速查表 → 必须列（漏了用户不知道
//     有这条路，而它是新加的全局键）；
//   - 输入态（模态）里 `?` 是用户要打的一个字符 → **不许列**（列了就是
//     骗用户按，按下去只会往输入框里多一个问号）。
//
// 两个方向都走 allViews()，所以每类视图的每个 Help 分支都要在这里有
// 一个 fixture —— 这一版之前每类视图只有一个 fixture，实测把三处
// 分支上的 `?` 删掉、把输入态的 `?` 加上，全都没有测试红。
func TestViews_帮助栏问号的两个方向(t *testing.T) {
	for _, nv := range allViews() {
		help := strings.Join(nv.v.Help(), " ")
		has := strings.Contains(help, keyHelp+" 速查表")
		if nv.v.Modal() {
			if has {
				t.Errorf("%s（输入态）的帮助栏列了 %q —— 那一态下它是用户要打的字符: %q",
					nv.name, keyHelp, help)
			}
			continue
		}
		if !has {
			t.Errorf("%s（非模态）的帮助栏没列 %q —— 按 `?` 真的能开速查表: %q",
				nv.name, keyHelp+" 速查表", help)
		}
	}
}

// **扫描是后台的**：中途推入别的视图（速查表、张量列表）不能把链弄断。
//
// 这条以前是坏的：消息只交给栈顶，栈顶换人之后那条结果落进了不处理它的
// 视图 —— 链停在原地，而 ModelView 那边 `scanning` 还是 true、进度行继续
// 写着"正在扫描…"，冻在一个数字上（真终端实测：扫描中按 `?`，回来时停在
// 7/434）。现在根视图按"谁是 ModelView"路由（routeToModelView），与
// popToMsg 同一个理由：消息该归谁，由"谁是那个视图"决定，不是由"谁恰好
// 在栈顶"决定。
//
// 三件事都要验：链没断（还返回下一条命令）、**不在栈顶的那一层也在推进**、
// 回到那一屏时看到的是推进后的终态而不是冻住的数字。
func TestApp_扫描在后台继续推进(t *testing.T) {
	model := fakeModel()
	v := loadModelView(model)
	v.scan = fakeScan(nil)
	root := New(fakeView{title: "根"})
	pushed, _ := root.Update(pushMsg{v: v})
	board := pushed.(Model)

	// 按 a 起步：命令先拿在手里不执行，模拟"已经派发出去、还没回来"
	next, scanCmd := board.Update(key(keyScanAll))
	board = next.(Model)
	if scanCmd == nil {
		t.Fatal("按 a 没有起步")
	}

	// 用户这时候按 `?` 打开速查表（链正在等第 0 张的结果）
	next, pushRef := board.Update(key(keyHelp))
	board = runCmd(t, next.(Model), pushRef)
	if len(board.stack) != 3 {
		t.Fatalf("栈深 = %d, want 3（根 + ModelView + 速查表）", len(board.stack))
	}

	// 第 0 张的结果回来：**要送到栈里那个 ModelView**，链继续
	next, nextCmd := board.Update(scanCmd())
	board = next.(Model)
	if nextCmd == nil {
		t.Fatal("结果落在速查表上、链断了 —— 这条消息没有送到 ModelView")
	}
	if inner := board.stack[1].(ModelView); inner.scanDone != 1 {
		t.Errorf("不在栈顶的那一层进度 = %d, want 1", inner.scanDone)
	}

	// 第 1 张（最后一张）：链应当自己停
	next, nextCmd = board.Update(nextCmd())
	board = next.(Model)
	if nextCmd != nil {
		t.Error("扫完最后一张还在调度")
	}

	// 回到模型视图：看到的是**推进之后**的终态，不是冻住的 0/2
	next, _ = board.Update(key("esc"))
	board = next.(Model)
	mv, ok := board.stack[len(board.stack)-1].(ModelView)
	if !ok {
		t.Fatalf("栈顶是 %T, want ModelView", board.stack[len(board.stack)-1])
	}
	if mv.scanDone != len(model.Tensors) {
		t.Errorf("回到这一屏时进度 = %d, want %d —— 后台没有推进",
			mv.scanDone, len(model.Tensors))
	}
	if out := mv.View(120, 20); !strings.Contains(out, "扫描完成 2/2") {
		t.Errorf("回到这一屏看到的不是终态:\n%s", out)
	}
}

// 栈里没有 ModelView 时（用户已经按 Esc 退回模型库）扫描消息**丢弃**：
// 那条链的宿主已经不在了，屏幕上也没有进度行 —— 没有东西可续。
// 丢弃不是静默吞错：这里要的是"不崩、不返回命令、也不弹栈"。
func TestApp_栈里没有ModelView时扫描消息被丢弃(t *testing.T) {
	v := loadModelView(fakeModel())
	v.scan = fakeScan(nil)
	root := New(fakeView{title: "根"})
	pushed, _ := root.Update(pushMsg{v: v})
	board := pushed.(Model)
	next, scanCmd := board.Update(key(keyScanAll))
	board = next.(Model)

	// 退回模型库：ModelView 连同它的扫描状态一起没了
	next, _ = board.Update(key("esc"))
	board = next.(Model)
	if len(board.stack) != 1 {
		t.Fatalf("栈深 = %d, want 1（退回模型库）", len(board.stack))
	}

	next, cmd := board.Update(scanCmd())
	board = next.(Model)
	if cmd != nil {
		t.Error("没有 ModelView 了却返回了命令 —— 链会飘在栈外继续跑")
	}
	if len(board.stack) != 1 {
		t.Errorf("栈深 = %d, want 1 —— 丢弃不该动栈", len(board.stack))
	}
}

// **路由不放松模型认领**：路由是按"谁是 ModelView"送消息的，它不看这条
// 消息属于哪个模型 —— 上一轮那条链的残影（用户退回模型库、换了模型再进来）
// 照样会送到这一层的 ModelView 上。所以 `msg.m != v.m` 那一条判据是这条
// 路由的**安全前提**，不许因为"反正送到了正确的图层"而删掉。
func TestApp_路由不放松模型认领(t *testing.T) {
	// 变量名不能叫 model：那会遮住 model 包（这个测试要构造 model.Stats）
	mm := fakeModel()
	v := loadModelView(mm)
	v.scan = fakeScan(nil)
	root := New(fakeView{title: "根"})
	pushed, _ := root.Update(pushMsg{v: v})
	board := pushed.(Model)

	next, scanCmd := board.Update(key(keyScanAll))
	board = next.(Model)
	if scanCmd == nil {
		t.Fatal("按 a 没有起步")
	}

	// 另一个模型的链上飞回来的消息（里面装着一份看着很正常的结果）
	foreign := fakeModelWithTensors(3)
	next, cmd := board.Update(batchScannedMsg{
		m: foreign, idx: 0,
		stats: &model.Stats{Count: 42},
		quant: &model.QuantInfo{Scheme: "Q4_K"},
	})
	board = next.(Model)

	if cmd != nil {
		t.Error("认不出的消息却让链继续了 —— 那会走出来两条链")
	}
	mv := board.stack[len(board.stack)-1].(ModelView)
	if mv.scanDone != 0 {
		t.Errorf("进度 = %d, want 0 —— 别的模型的结果被算进来了", mv.scanDone)
	}
	if mm.Tensors[0].Stats != nil {
		t.Error("别的模型的结果合并进了这个模型的张量")
	}
}

// topOf 返回栈顶视图（测试用）。
func topOf(m tea.Model) View { return m.(Model).top() }

// findTensorView 返回栈里最靠上的 TensorView。
//
// 返回的是**值拷贝**：这些测试只读字段，不写回栈里。
func findTensorView(m tea.Model) *TensorView {
	mm := m.(Model)
	for i := len(mm.stack) - 1; i >= 0; i-- {
		if tv, ok := mm.stack[i].(TensorView); ok {
			return &tv
		}
	}
	return nil
}

// findLibrary 返回栈里最靠上的 Library（同样是值拷贝）。
func findLibrary(m tea.Model) *Library {
	mm := m.(Model)
	for i := len(mm.stack) - 1; i >= 0; i-- {
		if lib, ok := mm.stack[i].(Library); ok {
			return &lib
		}
	}
	return nil
}

// 结果消息必须送到**发起它**的那个视图，而不是恰好压在栈顶的那个。
//
// 与 routeToModelView 是同一个理由：消息该归谁由"谁是那个视图"决定，
// 不是由"谁恰好在栈顶"决定。此处是同一个 bug 的另外两个实例 ——
// 帮助栏里就列着 `?`，用户在任何时刻都能按。
func TestApp_张量扫描结果不被栈顶吃掉(t *testing.T) {
	tn := &model.Tensor{Name: "t", Dtype: model.DtypeF32, ParamCount: 8, ByteSize: 32}
	mdl := &model.Model{Path: "x.gguf", Tensors: []*model.Tensor{tn}}
	tv := NewTensorView(mdl, tn)
	if !tv.scanning {
		t.Fatal("Stats 为 nil 时 NewTensorView 应当进入扫描中")
	}

	var m tea.Model = New(tv)

	// 用户按 ? 把速查表压上来（此时扫描结果还没回来）。
	// **? 返回的是一个 Cmd，要执行它才会产生 pushMsg** —— 丢掉的话
	// 速查表根本没压上来，测试会在一个假的场景里通过。
	m, cmd := m.Update(key("?"))
	if cmd == nil {
		t.Fatal("按 ? 应当返回一个压栈命令")
	}
	m, _ = m.Update(cmd())
	if _, ok := topOf(m).(RefView); !ok {
		t.Fatalf("栈顶应当是 RefView，得到 %T", topOf(m))
	}

	// **字段名是 name/stats/quant/sims/err，不是 tn**
	m, _ = m.Update(tensorScannedMsg{name: "t", stats: &model.Stats{Count: 8}})

	m, _ = m.Update(key("esc"))
	got := findTensorView(m)
	if got == nil {
		t.Fatal("栈里找不到 TensorView")
	}
	if got.tn.Stats == nil {
		t.Error("扫描结果被栈顶视图吃掉了（tn.Stats 仍为 nil）")
	}
	if got.scanning {
		t.Error("scanning 仍为 true —— spinner 会永远转下去")
	}
}

// 模型库首屏的 Fill 结果同理：被栈顶吃掉的话，回来时进度行
// 永远停在 0/N，那些行的格式与参数量永远是空的。
func TestApp_库填充结果不被栈顶吃掉(t *testing.T) {
	lib := NewLibrary()
	lib.items = []discover.Item{{Name: "m", Path: "/x"}}
	lib.gen = 1

	var m tea.Model = New(lib)
	m, cmd := m.Update(key("?"))
	if cmd == nil {
		t.Fatal("按 ? 应当返回一个压栈命令")
	}
	m, _ = m.Update(cmd())
	if _, ok := topOf(m).(RefView); !ok {
		t.Fatalf("栈顶应当是 RefView，得到 %T", topOf(m))
	}

	filled := discover.Item{Name: "m", Path: "/x", Format: "GGUF"}
	m, _ = m.Update(itemFilledMsg{index: 0, item: filled, gen: 1})

	m, _ = m.Update(key("esc"))
	got := findLibrary(m)
	if got == nil {
		t.Fatal("栈里找不到 Library")
	}
	if got.filled != 1 {
		t.Errorf("filled = %d, want 1 —— Fill 结果被栈顶吃掉了", got.filled)
	}
	if got.items[0].Format != "GGUF" {
		t.Errorf("条目没被填充：%+v", got.items[0])
	}
}

// 非输入态下 Backspace 与 Esc 同义（spec §3 的按键表）。
//
// 原来 7 种状态下 Backspace 都不返回：库/概览/无过滤的列表/详情页
// 完全无反应；有过滤词时删掉过滤词的最后一个字符 —— 用户想退回去
// 却改了过滤条件。
func TestApp_非输入态退格等于返回(t *testing.T) {
	m := New(fakeView{title: "根"})
	m2, _ := m.Update(pushMsg{v: fakeView{title: "第二层"}})
	if len(m2.(Model).stack) != 2 {
		t.Fatal("压栈失败")
	}

	m3, _ := m2.(Model).Update(tea.KeyMsg{Type: tea.KeyBackspace})
	if n := len(m3.(Model).stack); n != 1 {
		t.Errorf("退格没有返回上一层，栈深 = %d, want 1", n)
	}
}

// 根视图的退格与根视图的 Esc 一样，不退出程序也不崩。
func TestApp_根视图退格无动作(t *testing.T) {
	m := New(fakeView{title: "根"})
	m2, cmd := m.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	if cmd != nil {
		if _, quit := cmd().(tea.QuitMsg); quit {
			t.Error("根视图按退格不该退出程序")
		}
	}
	if n := len(m2.(Model).stack); n != 1 {
		t.Errorf("栈深 = %d, want 1", n)
	}
}

// **输入态下退格仍然是删字**，不能被全局那条抢走。
//
// 这是这条改动最容易做错的地方：把 backspace 提到模态判定之前的话，
// 过滤框里的退格会变成"返回上一层" —— 而用户正在打字。
func TestApp_输入态退格仍归输入框(t *testing.T) {
	m := New(fakeView{title: "根"})
	m2, _ := m.Update(pushMsg{v: NewTensorsView(&model.Model{
		Path:    "x",
		Tensors: []*model.Tensor{{Name: "abc", Dtype: model.DtypeF32}},
	})})
	tv := m2.(Model).stack[1].(TensorsView)
	tv2, _ := tv.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("/")})
	tv2, _ = tv2.(TensorsView).Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("ab")})
	if tv2.(TensorsView).filter != "ab" {
		t.Fatalf("filter = %q, want ab", tv2.(TensorsView).filter)
	}
	// 经根视图发退格：视图在输入态（Modal），应当被它自己吃掉
	m3 := m2.(Model)
	m3.stack[1] = tv2
	m4, _ := m3.Update(tea.KeyMsg{Type: tea.KeyBackspace})

	if n := len(m4.(Model).stack); n != 2 {
		t.Fatalf("输入态下退格把视图弹掉了，栈深 = %d, want 2", n)
	}
	got := m4.(Model).stack[1].(TensorsView)
	if got.filter != "a" {
		t.Errorf("filter = %q, want %q —— 输入态的退格应当删字", got.filter, "a")
	}
}

// 用**真的 ModelView**（不是 fakeView）走一遍：退格要能从模型页回到库。
//
// 上一条用的是 fakeView，走不到 ModelView 的按键分支 —— 如果 ModelView
// 自己吃掉了 backspace，那条测试照样绿。
func TestApp_退格从真的模型页返回库(t *testing.T) {
	lib := NewLibrary()
	var m tea.Model = New(lib)

	m, _ = m.Update(pushMsg{v: NewModelView(fakeModel())})
	if n := len(m.(Model).stack); n != 2 {
		t.Fatalf("栈深 = %d, want 2", n)
	}

	m, _ = m.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	if n := len(m.(Model).stack); n != 1 {
		t.Errorf("退格没有从模型页返回库，栈深 = %d —— ModelView 自己吃掉了它？", n)
	}
}

// Cmd 里 panic 不该掀掉整个 TUI —— 应当变成一条可显示的消息。
//
// bubbletea **不 recover 用户的 Cmd**：解析、解码、量化模拟都跑在 Cmd
// 的 goroutine 里，任何越界/nil 解引用都会掀掉整个界面，终端还留在
// alt-screen 里（用户只能 Ctrl+C）。spec §4.0 明文要求"在边界 recover"。
//
// 这是**纵深防御**：700 次随机变异（GGUF 300 + safetensors 200 +
// pytorch 200）一次 panic 都没打出来。它的价值是把"未知边界 → 整个
// 界面消失"降级成"这一项报错、其余照常"。
func TestApp_Cmd里的panic被兜住(t *testing.T) {
	cmd := safeCmd(func() tea.Msg { panic("boom") })
	msg := cmd()
	if _, ok := msg.(panicMsg); !ok {
		t.Fatalf("panic 没有被兜住，得到 %T", msg)
	}
}

// 正常返回的 Cmd 不受影响（包装不能改变语义）。
func TestApp_safeCmd透传正常结果(t *testing.T) {
	// 用可比较的消息类型：tea.KeyMsg 里有 []rune，不能用 != 比
	want := "hello"
	if got := safeCmd(func() tea.Msg { return want })(); got != tea.Msg(want) {
		t.Errorf("safeCmd 改变了正常返回值: %v", got)
	}
}

// panic 消息要**看得见** —— 只吞掉的话用户不知道有东西崩了。
func TestApp_panic消息在界面上可见(t *testing.T) {
	m := New(fakeView{title: "根"})
	m2, _ := m.Update(panicMsg{err: errors.New("内部错误: boom")})
	out := m2.(Model).View()
	if !strings.Contains(out, "boom") {
		t.Errorf("panic 消息没显示出来:\n%s", out)
	}
}

// **接线要有测试**：safeCmd 本身测过了，但"真的包在那些 Cmd 上吗"
// 是另一件事 —— 注释声称包了而实际没包，就是声称漂移。
//
// 用会 panic 的注入函数走真实路径：解析入口、单张量扫描。
func TestApp_真实的Cmd确实被包住(t *testing.T) {
	// ① ModelView 的解析 Cmd
	mv := NewModelViewFromPath("/x/m.gguf", "m")
	mv.parse = func(string) (*model.Model, error) { panic("解析爆了") }
	msg := mv.Init()()
	if _, ok := msg.(panicMsg); !ok {
		t.Errorf("解析 Cmd 没被包住，得到 %T", msg)
	}

	// ② ModelView 的单张量扫描 Cmd
	mdl := fakeModel()
	mv2 := NewModelView(mdl)
	mv2.scan = func(context.Context, *model.Model, *model.Tensor) error { panic("扫描爆了") }
	msg = mv2.scanOneCmd(0)()
	if _, ok := msg.(panicMsg); !ok {
		t.Errorf("扫描 Cmd 没被包住，得到 %T", msg)
	}

	// ③ TensorView 的扫描 Cmd
	tv := NewTensorView(mdl, mdl.Tensors[0])
	tv.scan = func(context.Context, *model.Model, *model.Tensor) error { panic("详情页爆了") }
	msg = tv.scanCmd()()
	if _, ok := msg.(panicMsg); !ok {
		t.Errorf("详情页的扫描 Cmd 没被包住，得到 %T", msg)
	}
}
