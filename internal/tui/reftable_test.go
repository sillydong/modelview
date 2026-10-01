package tui

import (
	"strings"
	"testing"

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
func TestRefView_Enter进入条目(t *testing.T) {
	v := NewRefView(nil)
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
