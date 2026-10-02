package tui

import (
	"strings"
	"testing"
	"unicode/utf8"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/ref"
)

// 表要全部列出，右边是选中表的条目。
func TestRefView_列出全部表(t *testing.T) {
	v := NewRefView(nil)
	out := v.View(120, 40)
	for _, tb := range ref.Tables() {
		if !strings.Contains(out, tb.Title) {
			t.Errorf("没列出表 %q:\n%s", tb.Title, out)
		}
	}
	// 第一张表的第一条要在右栏里
	if first := ref.Tables()[0].Entries[0]; !strings.Contains(out, first.Title) {
		t.Errorf("右栏没显示第一条 %q:\n%s", first.Title, out)
	}
}

// Tab 之后 ↑↓ 走的是右栏的光标，左栏不动。
//
// 这是"两个光标"这件事唯一的行为证据：只断言"Tab 改了个字段"
// 的话，把 ↑↓ 全部接到左栏上也能过。
func TestRefView_Tab切换焦点(t *testing.T) {
	v := NewRefView(nil)

	// 左栏：↓ 只动表光标
	v2, _ := v.Update(key("down"))
	v = v2.(RefView)
	if v.tableCursor != 1 || v.entryCursor != 0 {
		t.Fatalf("左栏焦点下 ↓ 之后 tableCursor=%d entryCursor=%d, want 1/0",
			v.tableCursor, v.entryCursor)
	}

	// 切到右栏：↓ 只动条目光标、表不动
	v2, _ = v.Update(key("tab"))
	v = v2.(RefView)
	if v.focus != focusEntries {
		t.Fatal("Tab 之后焦点不在右栏")
	}
	v2, _ = v.Update(key("down"))
	v = v2.(RefView)
	if v.tableCursor != 1 || v.entryCursor != 1 {
		t.Errorf("右栏焦点下 ↓ 之后 tableCursor=%d entryCursor=%d, want 1/1",
			v.tableCursor, v.entryCursor)
	}
}

// 换表时条目光标要**归零**，否则从 118 条的 keys 切到 21 条的 floats，
// 光标停在 100 会直接越界（或者更糟：静默指到别的条目上）。
func TestRefView_换表时光标归零(t *testing.T) {
	v := NewRefView(nil)
	v2, _ := v.Update(key("tab"))
	v = v2.(RefView)
	for range 5 {
		v2, _ = v.Update(key("down"))
		v = v2.(RefView)
	}
	if v.entryCursor != 5 {
		t.Fatalf("entryCursor = %d, want 5", v.entryCursor)
	}
	// 回左栏换表
	v2, _ = v.Update(key("tab"))
	v = v2.(RefView)
	v2, _ = v.Update(key("down"))
	v = v2.(RefView)
	if v.entryCursor != 0 {
		t.Errorf("换表之后 entryCursor = %d, want 0", v.entryCursor)
	}
	if v.tableCursor != 1 {
		t.Errorf("tableCursor = %d, want 1", v.tableCursor)
	}
}

// 选中项在两个焦点下都必须可见 —— 与模型库、张量列表同一条要求。
func TestRefView_选中项始终可见(t *testing.T) {
	v := NewRefView(nil)
	// 右栏：keys 表有 118 条，翻到底
	v2, _ := v.Update(key("tab"))
	v = v2.(RefView)
	for range 60 {
		v2, _ = v.Update(key("down"))
		v = v2.(RefView)
	}
	out := v.View(120, 20)
	want := v.entries()[v.entryCursor].Title
	if !strings.Contains(out, want) {
		t.Fatalf("右栏光标在第 %d 条，屏幕上看不到 %q:\n%s", v.entryCursor, want, out)
	}

	// 左栏：表只有 4 张，但终端只有 5 行高时也该看见选中的那张
	v3, _ := v.Update(key("tab"))
	v = v3.(RefView)
	for range 3 {
		v4, _ := v.Update(key("down"))
		v = v4.(RefView)
	}
	small := v.View(120, 5)
	if !strings.Contains(small, ref.Tables()[v.tableCursor].Title) {
		t.Errorf("高度 5 时看不到选中的表:\n%s", small)
	}
}

// 一屏放不下时要**明说显示的是哪一段** —— 不说的话用户会以为
// 看到的就是全部（118 条的 keys 表在 24 行的终端里只放得下 20 条）。
//
// 这条是计划里没写的：把整个 footer 分支删掉，计划里的十条测试全绿 ——
// 而那行提示正是"选中的那条为什么不见了"之外唯一能说明范围的东西。
func TestRefView_超屏要说明显示范围(t *testing.T) {
	v := NewRefView(nil)
	// 换到 118 条的 GGUF 元数据键
	for range 2 {
		v2, _ := v.Update(key("down"))
		v = v2.(RefView)
	}
	out := v.View(120, 20)
	if !strings.Contains(out, "…显示第") || !strings.Contains(out, "共 118 条") {
		t.Errorf("118 条只显示了一屏，却没说明显示的是哪一段:\n%s", out)
	}

	// 反面对照：放得下时不该出现这句话（否则它就是个装饰）
	small := NewRefView(nil).View(120, 40)
	if strings.Contains(small, "…显示第") {
		t.Errorf("21 条、40 行放得下，不该说「显示第 N–M 条」:\n%s", small)
	}
}

// 视图交给根视图的原始输出不能超过高度。
func TestRefView_不超高(t *testing.T) {
	v := NewRefView(nil)
	for _, h := range []int{6, 12, 24} {
		raw := v.View(120, h)
		if n := len(strings.Split(raw, "\n")); n > h {
			t.Errorf("高度 %d：给了 %d 行", h, n)
		}
	}
}

// `/` 搜索跨全部表 —— 搜索是**全局**的，不是只搜当前这张。
func TestRefView_搜索跨表(t *testing.T) {
	v := NewRefView(nil)
	// Q4_K 这条在 quants 表里，不是第一张表
	v2, _ := v.Update(key("/"))
	v = v2.(RefView)
	for _, r := range "Q4_K" {
		v3, _ := v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		v = v3.(RefView)
	}
	out := v.View(120, 30)
	if !strings.Contains(out, "Q4_K") {
		t.Errorf("搜 Q4_K 没有结果:\n%s", out)
	}
	if !strings.Contains(out, "搜索") {
		t.Errorf("没标出这是搜索结果:\n%s", out)
	}

	// **上面两条都证明不了"跨表"**：输入行里回显着刚敲的词
	//（"搜索：Q4_K▏"），所以 `Contains(out, "Q4_K")` 在"只搜当前这张表、
	// 一条都没命中"时同样成立 —— 变异验证确认过：把 entries() 的搜索分支
	// 改成永不生效（`v.search != "" && false`），上面两条全绿。
	// 所以要钉住**结果本身来自别的表**：Q4_K 在"量化方案"表里，
	// 只搜第一张表（数值格式）的话它不可能出现在结果里。
	var crossed bool
	for _, e := range v.entries() {
		if e.ID == "quant:Q4_K" {
			crossed = true
		}
	}
	if !crossed {
		t.Errorf("结果里没有别的表的条目（搜索只搜了当前这张表）：%d 条",
			len(v.entries()))
	}
}

// 无匹配要明说 —— 给一个空白列表，用户会以为界面坏了。
func TestRefView_无匹配有提示(t *testing.T) {
	v := NewRefView(nil)
	v2, _ := v.Update(key("/"))
	v = v2.(RefView)
	for _, r := range "zzzzzz" {
		v3, _ := v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		v = v3.(RefView)
	}
	out := v.View(120, 30)
	if !strings.Contains(out, "没有匹配") {
		t.Errorf("无匹配时没提示:\n%s", out)
	}
}

// 搜索态是模态的（否则敲 q 会退出程序）。
func TestRefView_搜索态是模态的(t *testing.T) {
	v := NewRefView(nil)
	if v.Modal() {
		t.Error("非搜索态不该是模态的")
	}
	v2, _ := v.Update(key("/"))
	if !v2.(RefView).Modal() {
		t.Error("搜索态必须是模态的，否则根视图会把 q 吃掉直接退出程序")
	}
}

// Enter 推入条目详情，带的是**当前选中**的那一条。
//
// **模型也要跟着进条目页**（`NewEntryView(v.m, …)` 而不是 `nil`）：
// 条目页的"在本模型中"读的就是 `EntryView.m`，传 nil 的那一支是
// "不知道"而不是"没有"—— 表现只是那一节整段不见，界面上不报错。
//
// 这里原先用的是 `NewRefView(nil)`，**两边都是 nil，测不出差别**：
// 实测把 `v.m` 改成 `nil`、整包跑测试一条都不红。所以这个测试要用
// 一个真模型，并且断言带过去的正是它。
func TestRefView_Enter进入条目(t *testing.T) {
	v := NewRefView(fakeModel())
	v2, _ := v.Update(key("tab"))
	v = v2.(RefView)
	v2, _ = v.Update(key("down"))
	v = v2.(RefView)

	want := v.entries()[v.entryCursor]
	_, cmd := v.Update(key("enter"))
	if cmd == nil {
		t.Fatal("Enter 没有返回命令")
	}
	msg, ok := cmd().(pushMsg)
	if !ok {
		t.Fatalf("返回的不是 pushMsg: %T", cmd())
	}
	ev, ok := msg.v.(EntryView)
	if !ok {
		t.Fatalf("推入的不是 EntryView: %T", msg.v)
	}
	if ev.e.ID != want.ID {
		t.Errorf("推入的是 %q, want %q", ev.e.ID, want.ID)
	}
	if ev.m == nil {
		t.Fatal("推入的条目页没有带模型 —— " +
			"「在本模型中」那一节会是空的（跳得进去、跳不出来）")
	}
	if ev.m != v.m {
		t.Errorf("带过去的不是当前这个模型：%p, want %p", ev.m, v.m)
	}
}

// 空表不能崩 —— 表是内置数据，但"内置的一定非空"是个假设，
// 而假设被打破时的表现是索引越界 panic。
func TestRefView_表为空不崩(t *testing.T) {
	v := NewRefView(nil)
	v.tables = []ref.Table{{ID: "empty", Title: "空表"}}
	out := v.View(120, 20)
	if !strings.Contains(out, "空表") {
		t.Errorf("空表没列出:\n%s", out)
	}
	// 不该 panic
	v2, _ := v.Update(key("tab"))
	v2.(RefView).Update(key("down"))
	v2.(RefView).Update(key("enter"))
}

// 表列表整个为空时 `Title()` 也不能崩。
//
// 它与 `entries()` 必须用**同一处**越界判据：原先 entries() 挡了、
// Title() 直接下标 —— 同一个前提被两个读者读出两个结论，
// 而真为空时只有一个是对的（另一个 panic）。
// `Title()` 由根视图的标题栏调用，panic 就是整屏崩掉。
func TestRefView_表列表为空不崩(t *testing.T) {
	v := NewRefView(nil)
	v.tables = nil
	if out := v.View(120, 20); out == "" {
		t.Error("表列表为空时什么都没渲染")
	}
	if title := v.Title(); title == "" {
		t.Error("表列表为空时标题是空的")
	}
}

// **终端比左栏还窄时不能 panic。**
//
// 右栏宽度 = width − 左栏宽 − 1，窄到负数时搜索态那行会把它交给
// humanize.Truncate —— 那个函数对负数是 `s[:cut]` 切片越界 panic。
// 实测（修之前）：宽 0/1/10 三档全 panic，宽 17 起才正常。
// 这在真终端里可达：窗口拖到 17 列以下，或 pty 报 0×0
// （drive_tui 的文件头就写着 pty 默认 winsize 是 0×0）。
func TestRefView_窄终端不崩(t *testing.T) {
	v := NewRefView(nil)
	v2, _ := v.Update(key("/"))
	v = v2.(RefView)
	for _, w := range []int{0, 1, 10, 17} {
		out := v.View(w, 24)
		// 左栏还在（搜索态下它塌成一行"搜索结果"；
		// 窄到 0 列时它自己也超宽，那是根视图截断的事）
		if !strings.Contains(out, "搜索结果") {
			t.Errorf("宽度 %d：左栏没渲染出来:\n%s", w, out)
		}
	}
}

// **空格必须能进搜索词，判据是 `len(msg.Runes) > 0` 而不是 `msg.Type == tea.KeyRunes`。**
//
// bubbletea v1.3.10 把单独一个空格设成 `Type: KeySpace`，但 `Runes` 仍然是
// `[' ']`（key.go:698-701）—— 按 Type 判的话它会被静默丢掉。
// 这条是计划里没写的：照计划给的 `msg.Type == tea.KeyRunes` 写，
// 空格会丢，而上面那些用 `tea.KeyMsg{Type: tea.KeyRunes}` 造的按键
// 全都察觉不到（tensors.go 里有一段同样的说明）。
func TestRefView_搜索词能输入空格(t *testing.T) {
	v := NewRefView(nil)
	v2, _ := v.Update(key("/"))
	v = v2.(RefView)
	// 真实空格按键：Type 是 KeySpace，不是 KeyRunes
	v3, _ := v.Update(tea.KeyMsg{Type: tea.KeySpace, Runes: []rune{' '}})
	v = v3.(RefView)
	if v.search != " " {
		t.Errorf("空格没进搜索词：search = %q, want %q", v.search, " ")
	}
}

// confirmSearch 造一个"输入并确认了搜索词"的视图。
//
// 确认（Enter）之后**词还留着**、但不再是模态的 —— 面朝上层的按键行为
// 与输入态完全不同，所以要单独造这一态。
func confirmSearch(kw string) RefView {
	v := NewRefView(nil)
	v2, _ := v.Update(key("/"))
	v = v2.(RefView)
	for _, r := range kw {
		v3, _ := v.Update(key(string(r)))
		v = v3.(RefView)
	}
	v4, _ := v.Update(key("enter"))
	return v4.(RefView)
}

// **搜索无匹配时帮助栏不能说"Enter 详情"** —— 一条都没有，按下去什么也不做。
//
// 无匹配是搜索的正常结局（不是错误状态）；判据与 Update 里那个是同一个
// （RefView.canOpen），所以不存在"帮助栏说能开、按下去没反应"。
func TestRefView_无匹配帮助栏不列Enter(t *testing.T) {
	v := confirmSearch("qqzzxx")
	if n := len(v.entries()); n != 0 {
		t.Fatalf("前提不成立：搜 %q 匹配到 %d 条", "qqzzxx", n)
	}
	if help := strings.Join(v.Help(), " "); strings.Contains(help, keyEnter) {
		t.Errorf("无匹配时帮助栏列了 %s —— 按下去没反应: %q", keyEnter, help)
	}
	if _, cmd := v.Update(key("enter")); cmd != nil {
		t.Error("无匹配时按 Enter 居然有动作 —— 帮助栏与 Update 读的不是同一个判据")
	}

	// 正对照：有匹配时两个方向都要成立
	full := NewRefView(nil)
	if help := strings.Join(full.Help(), " "); !strings.Contains(help, keyEnter) {
		t.Errorf("有匹配时帮助栏少了 %s: %q", keyEnter, help)
	}
	if _, cmd := full.Update(key("enter")); cmd == nil {
		t.Error("有匹配时按 Enter 没动作")
	}
}

// **已确认的搜索词也能用 Backspace 删字** —— 与 `TensorsView` 同一条路。
//
// 两个都是"列表 + 一个过滤词"的视图，同一个键在一处能删、在另一处
// 静默失效的话，用户在一边学会的动作到另一边只会以为是自己按错了。
func TestRefView_已确认搜索态Backspace删字(t *testing.T) {
	v := confirmSearch("qwen")
	// 已确认态**不是**模态的：不是的话按键会走根视图别的分支，
	// 这条测试验的就不是"到得了这里"了
	if v.Modal() {
		t.Fatal("已确认搜索态不该是模态的")
	}
	// 先把条目光标移开 —— 不移的话"删字之后归零"看不出任何变化
	for range 3 {
		v2, _ := v.Update(key("down"))
		v = v2.(RefView)
	}
	if v.entryCursor != 3 {
		t.Fatalf("前提不成立：↓ 三下之后 entryCursor = %d（搜 %q 匹配 %d 条）",
			v.entryCursor, "qwen", len(v.entries()))
	}

	v3, _ := v.Update(key("backspace"))
	v = v3.(RefView)
	if v.search != "qwe" {
		t.Errorf("删一个字符之后 search = %q, want %q", v.search, "qwe")
	}
	if v.entryCursor != 0 {
		t.Errorf("词变了之后条目光标 = %d, want 0 —— 右栏内容整个换了一份",
			v.entryCursor)
	}

	// 删到底：回到"当前表"那一态，右栏是表里的条目而不是空搜索结果
	for range 3 {
		v4, _ := v.Update(key("backspace"))
		v = v4.(RefView)
	}
	if v.search != "" {
		t.Fatalf("删到底之后 search = %q, want 空", v.search)
	}
	if v.tableCursor != 0 || v.entryCursor != 0 {
		t.Errorf("回到当前表之后光标没归零：table=%d entry=%d",
			v.tableCursor, v.entryCursor)
	}
	if n, want := len(v.entries()), len(ref.Tables()[0].Entries); n != want {
		t.Errorf("删空之后右栏 %d 条, want 第一张表的 %d 条", n, want)
	}
	// 空词上继续按不该出事（也不该把 search 变成别的）
	if v6, _ := v.Update(key("backspace")); v6.(RefView).search != "" {
		t.Error("已经是空词了，Backspace 却改动了它")
	}
}

// 速查表的搜索框与张量过滤框是同一条约定：退格删一个**字符**。
func TestRefView_退格删一个字符(t *testing.T) {
	v := NewRefView(nil)
	v2, _ := v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("/")})
	v = v2.(RefView)
	v2, _ = v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("中")})
	v = v2.(RefView)
	if v.search != "中" {
		t.Fatalf("search = %q, want %q", v.search, "中")
	}
	v2, _ = v.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	v = v2.(RefView)
	if v.search != "" {
		t.Errorf("退格后 search = %q（% x），want 空串", v.search, v.search)
	}
	if !utf8.ValidString(v.search) {
		t.Errorf("退格切出了非法 UTF-8: % x", v.search)
	}
}
