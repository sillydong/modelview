package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

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
	m := New(viewWithBody{body: "x"})
	m2, _ := m.Update(tea.WindowSizeMsg{Width: 20, Height: 6})
	m = m2.(Model)

	// 栈顶没有 Title() 时用默认标题；这里用有超长标题的视图
	long := longTitleView{}
	m3, _ := New(long).Update(tea.WindowSizeMsg{Width: 20, Height: 6})
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

// push 进来的视图也要跑它的 Init —— 不跑的话"进模型视图"会停在"正在解析…"
func TestApp_push的视图也会跑Init(t *testing.T) {
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

	return []namedView{
		{"Library", lib},
		{"ModelView/概览", mv},
		{"ModelView/元数据", gotoSection(mv, sectionMetadata)},
		{"ModelView/张量", gotoSection(mv, sectionTensors)},
		{"ModelView/量化分布", gotoSection(mv, sectionQuantDist)},
		{"ModelView/速查表", gotoSection(mv, sectionRef)},
		{"TensorsView", NewTensorsView(m)},
		{"TensorView/扫描中", NewTensorView(m, m.Tensors[0])},
		{"TensorView/已出结果", NewTensorView(m2, done)},
		{"RefView", NewRefView(nil)},
		{"EntryView", NewEntryView(nil, ref.Entry{ID: "quant:Q4_K", Title: "Q4_K"})},
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
