package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
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
	case "ctrl+c":
		return tea.KeyMsg{Type: tea.KeyCtrlC}
	default:
		return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
	}
}

// 尺寸消息必须让界面记住 —— 所有 View 都按它算换行。
func TestApp_记住终端尺寸(t *testing.T) {
	m := New(fakeView{title: "根"}, 0)
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
		m := New(fakeView{title: "根"}, 0)
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
	m := New(fakeView{title: "根"}, 0)
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
	m := New(fakeView{title: "根"}, 0)

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
	m := New(fakeView{title: "根"}, 0)
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
	m := New(viewWithBody{body: long}, 0)
	m2, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 10})
	m = m2.(Model)

	out := m.View()
	if n := len(strings.Split(out, "\n")); n != 10 {
		t.Errorf("内容 100 行、屏幕 10 行，渲染出 %d 行 —— 应当截到 10 行", n)
	}
}

// 还没收到尺寸消息时也要能渲染（初始帧）—— 返回空串会让界面闪一下。
func TestApp_没有尺寸时也能渲染(t *testing.T) {
	m := New(fakeView{title: "根"}, 0)
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

// **内容超宽必须被截掉，不能留给终端折行。**
//
// 这条是变异验证逼出来的：上面那条"不超宽"断言用的内容都短于 80 列，
// 于是截断那段代码从来没被执行过 —— 把它整段删掉，测试照样绿。
// 超宽不截的后果是终端自动折行，整个界面往下错位、帮助栏跑出屏幕。
func TestApp_内容超宽被截断(t *testing.T) {
	wide := strings.Repeat("很宽的一行", 40) // 5×3×40 = 600 显示列
	m := New(viewWithBody{body: wide}, 0)
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
	m := New(viewWithBody{body: "x"}, 0)
	m2, _ := m.Update(tea.WindowSizeMsg{Width: 20, Height: 6})
	m = m2.(Model)

	// 栈顶没有 Title() 时用默认标题；这里用有超长标题的视图
	long := longTitleView{}
	m3, _ := New(long, 0).Update(tea.WindowSizeMsg{Width: 20, Height: 6})
	for i, l := range strings.Split(m3.(Model).View(), "\n") {
		if w := lipgloss.Width(l); w > 20 {
			t.Errorf("第 %d 行宽 %d 超过 20: %q", i, w, l)
		}
	}
	_ = m
}

type longTitleView struct{}

func (longTitleView) Init() tea.Cmd                  { return nil }
func (longTitleView) Update(tea.Msg) (View, tea.Cmd) { return longTitleView{}, nil }
func (longTitleView) View(w, h int) string           { return "内容" }
func (longTitleView) Help() []string {
	return []string{strings.Repeat("很长的帮助文本", 10)}
}
func (longTitleView) Title() string { return strings.Repeat("超长标题", 20) }

// **Init 必须真的被调用** —— 忘了接上的表现是"界面永远停在初始状态"，
// 而所有直接调 Update 的单元测试都是绿的。
//
// 实测：Library.Init 里发起扫描的那条命令从来没被执行过，
// 屏幕停在"正在扫描模型目录…"，是 pty 里跑真终端才看出来的。
func TestApp_Init会被调用(t *testing.T) {
	ran := false
	m := New(initTrackingView{onInit: func() { ran = true }}, 0)

	cmd := m.Init()
	if cmd == nil {
		t.Fatal("Model.Init() 返回 nil —— 根视图的初始化任务没被接上")
	}
	cmd()
	if !ran {
		t.Error("根视图的 Init() 没有被调用")
	}
}

// push 进来的视图也要跑它的 Init —— 不跑的话"进模型视图"会停在"正在解析…"
func TestApp_push的视图也会跑Init(t *testing.T) {
	ran := false
	m := New(fakeView{title: "根"}, 0)
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
