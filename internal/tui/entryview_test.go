package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"

	"github.com/sillydong/modelview/internal/ref"
)

func TestEntryView_渲染字段与说明(t *testing.T) {
	e := ref.Entry{
		ID: "quant:TEST", Title: "测试条目",
		Fields: []ref.Field{{Key: "位宽", Value: "4.5"}},
		Notes:  "这是一段说明",
	}
	out := NewEntryView(nil, e).View(100, 30)
	for _, want := range []string{"测试条目", "quant:TEST", "位宽", "4.5", "这是一段说明"} {
		if !strings.Contains(out, want) {
			t.Errorf("没渲染 %q:\n%s", want, out)
		}
	}
}

// SeeAlso 里能查到的，显示它的标题（而不是只显示 ID）。
func TestEntryView_SeeAlso显示标题(t *testing.T) {
	e := ref.Entry{ID: "a", Title: "A", SeeAlso: []string{"quant:Q4_K"}}
	target, ok := ref.ByID("quant:Q4_K")
	if !ok {
		t.Fatal("速查表里没有 quant:Q4_K —— 测试的前提不成立")
	}
	out := NewEntryView(nil, e).View(100, 30)
	if !strings.Contains(out, target.Title) {
		t.Errorf("没显示相关条目的标题 %q:\n%s", target.Title, out)
	}
}

// **指向不存在 ID 的 SeeAlso 要明说**，而不是渲染成一条正常的可跳项。
//
// ref.Entry 的注释明说 SeeAlso **不校验存在性**、指向未来的条目是允许的，
// 所以这条路径迟早会被走到。按 Enter 没反应时用户会以为界面坏了。
func TestEntryView_SeeAlso指向不存在(t *testing.T) {
	e := ref.Entry{ID: "a", Title: "A", SeeAlso: []string{"quant:NOT_YET"}}
	v := NewEntryView(nil, e)
	out := v.View(100, 30)
	if !strings.Contains(out, "quant:NOT_YET") {
		t.Errorf("没显示那个 ID —— 用户无从知道该补什么:\n%s", out)
	}
	if !strings.Contains(out, "没有这条") {
		t.Errorf("没标出这条还不存在:\n%s", out)
	}

	// **按 Enter 不能推入任何东西** —— 推一个空条目进去，
	// 用户看到的是一张白卡片
	_, cmd := v.Update(key("enter"))
	if cmd != nil {
		if msg, ok := cmd().(pushMsg); ok {
			t.Errorf("跳到了不存在的条目: %+v", msg.v)
		}
	}
}

// ↑↓ 走 SeeAlso 的光标，Enter 推入对应的条目。
func TestEntryView_SeeAlso可跳转(t *testing.T) {
	e := ref.Entry{ID: "a", Title: "A", SeeAlso: []string{"quant:Q4_K", "quant:Q6_K"}}
	v := NewEntryView(nil, e)

	if len(v.targets) != 2 {
		t.Fatalf("解析出 %d 条关联, want 2", len(v.targets))
	}
	v2, _ := v.Update(key("down"))
	v = v2.(EntryView)
	if v.cursor != 1 {
		t.Fatalf("cursor = %d, want 1", v.cursor)
	}

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
	if ev.e.ID != "quant:Q6_K" {
		t.Errorf("推入的是 %q, want quant:Q6_K", ev.e.ID)
	}
}

// **光标停在不可跳的那条上时，↑↓ 还要能走掉** ——
// 否则用户被卡在一个按 Enter 没反应的项上，只能 Esc 出去。
func TestEntryView_光标不会卡在不可跳的项上(t *testing.T) {
	e := ref.Entry{ID: "a", Title: "A",
		SeeAlso: []string{"quant:NOT_YET", "quant:Q4_K"}}
	v := NewEntryView(nil, e)
	if v.targets[0].ok {
		t.Fatal("测试前提不成立：第一条应当是查不到的")
	}
	v2, _ := v.Update(key("down"))
	if v2.(EntryView).cursor != 1 {
		t.Error("光标没能从不可跳的项上移走")
	}
}

// 没有 SeeAlso 时不该显示那一节（也不该有一行空标题）。
func TestEntryView_无SeeAlso不显示那一节(t *testing.T) {
	out := NewEntryView(nil, ref.Entry{ID: "a", Title: "A"}).View(100, 30)
	if strings.Contains(out, "相关条目") {
		t.Errorf("没有 SeeAlso 却显示了「相关条目」那一节:\n%s", out)
	}
}

// 窄终端下 Notes 要折行，不能超宽 —— 超宽会被终端自动折行，整个界面错位。
// 测试用的粗体函数：用真的 ANSI 序列，不是 "<b>" 之类的代用标记。
//
// 理由是宽度断言：lipgloss.Width 把 ANSI 转义算作 0 宽，所以带样式的串
// 折行之后每行仍然应当不超过 width。换成可见的替代标记，宽度断言量的
// 就成了那些标记，与生产路径不是一回事。
func ansiBold(s string) string { return "\x1b[1m" + s + "\x1b[0m" }

// **条目里的 `**粗体**` 要变成真粗体，星号不能原样打到屏幕上。**
//
// 这条测试是被真实截图逼出来的：速查表条目 detail 页上显示的是
// `**最常见的 4 位档**`，星号一个不少。数据写的是 Markdown，
// 而 EntryView 直接 wrapText 输出，中间没有任何东西负责渲染。
//
// 断言用全等而不是 Contains：Contains 在"标记被吃掉了但样式没加"时
// 也会通过，那正是这个 bug 的另一半。
func TestRenderEmphasis_粗体标记换成样式(t *testing.T) {
	got := renderEmphasis("前面**重点**后面", 100, ansiBold)
	want := "前面" + ansiBold("重点") + "后面"
	if got != want {
		t.Errorf("渲染结果不对\n got %q\nwant %q", got, want)
	}
}

// 折行必须在**去掉标记之后**按显示宽度算，而且不能把 ANSI 序列切开。
//
// 反例是"先给整串上样式再交给 wrapText"：那样宽度里混进了转义序列的
// 字符（或者被算成 0 宽导致行超长），断行点还可能落在转义序列中间。
func TestRenderEmphasis_折行不超宽也不切断样式(t *testing.T) {
	long := strings.Repeat("很长的说明文字**重点**再长一点。", 6)
	out := renderEmphasis(long, 30, ansiBold)
	for _, line := range strings.Split(out, "\n") {
		if n := lipgloss.Width(line); n > 30 {
			t.Errorf("有一行宽 %d 列，超过 30：%q", n, line)
		}
		// 每一行的转义序列都必须是完整的：奇数个 \x1b 说明被切在半路
		if n := strings.Count(line, "\x1b"); n%2 != 0 {
			t.Errorf("转义序列被切断（%d 个 ESC）:%q", n, line)
		}
	}
	if !strings.Contains(out, ansiBold("重点")) {
		t.Errorf("粗体跨行后没有保住样式：\n%q", out)
	}
}

// **标记不成对时整个当普通文本**，而不是"从第一个标记开始一直粗到底"。
//
// 半应用会让"数据里少写了一个 `**`"变成不可见：星号被吃掉、后面整段
// 悄悄变粗，而屏幕上没有任何东西提示这里出过错。宁可露出两个星号。
func TestRenderEmphasis_奇数标记不半应用(t *testing.T) {
	got := renderEmphasis("正常**从这里开始没闭合", 100, ansiBold)
	want := wrapText("正常**从这里开始没闭合", 100)
	if got != want {
		t.Errorf("不成对时应当原样输出\n got %q\nwant %q", got, want)
	}
}

// 没有标记的文本走的是原路径：渲染结果必须与 wrapText 逐字相同，
// 否则这次改动会顺手改掉所有条目的折行行为。
func TestRenderEmphasis_无标记时与wrapText一致(t *testing.T) {
	long := strings.Repeat("没有标记的普通说明文字。", 8)
	if got, want := renderEmphasis(long, 30, ansiBold), wrapText(long, 30); got != want {
		t.Errorf("无标记时的输出与 wrapText 不一致\n got %q\nwant %q", got, want)
	}
}

// 端到端：条目详情页上不能出现字面星号。
func TestEntryView_粗体标记不上屏(t *testing.T) {
	e := ref.Entry{ID: "a", Title: "A", Notes: "**最常见的 4 位档**。_M 档会把关键张量升到 Q6_K"}
	v := NewEntryView(nil, ref.Entry{
		ID: e.ID, Title: e.Title, Notes: e.Notes,
		Fields: []ref.Field{{Key: "枚举名", Value: "**MOSTLY_Q4_K_M**"}},
	})
	out := v.View(100, 30)
	if strings.Contains(out, "**") {
		t.Errorf("星号原样漏到屏幕上了：\n%s", out)
	}
	for _, want := range []string{"最常见的 4 位档", "MOSTLY_Q4_K_M"} {
		if !strings.Contains(out, want) {
			t.Errorf("文字被吃掉了，找不到 %q：\n%s", want, out)
		}
	}
}

func TestEntryView_窄终端折行(t *testing.T) {
	long := strings.Repeat("这是一段很长的中文说明，用来验证折行。", 5)
	v := NewEntryView(nil, ref.Entry{ID: "a", Title: "A", Notes: long})
	out := v.View(40, 30)
	for _, line := range strings.Split(out, "\n") {
		if n := lipgloss.Width(line); n > 40 {
			t.Errorf("有一行宽 %d 列，超过 40:\n%s", n, line)
		}
	}
}

// **一个能跳的目标都没有时，帮助栏不能说"Enter 跳转"** —— 按下去什么也不做。
//
// SeeAlso 不校验存在性（ref.Entry 的注释），所以"整页都跳不过去"是可达的：
// 作者写了笔误、或者指向一条还没写的条目。判据与 Update 里那个是同一个
// （EntryView.canJump），所以不存在"帮助栏说能跳、按下去没反应"。
func TestEntryView_全跳不过去时帮助栏不列Enter(t *testing.T) {
	e := ref.Entry{ID: "a", Title: "A", SeeAlso: []string{"quant:NOT_YET"}}
	v := NewEntryView(nil, e)
	if len(v.targets) != 1 || v.targets[0].ok {
		t.Fatalf("前提不成立：targets = %+v", v.targets)
	}
	if help := strings.Join(v.Help(), " "); strings.Contains(help, keyEnter) {
		t.Errorf("一条都跳不过去，帮助栏却列了 %s: %q", keyEnter, help)
	}
	if _, cmd := v.Update(key("enter")); cmd != nil {
		t.Error("跳不过去却按出了动作 —— 推一张空白卡片进去，用户只会以为是自己按错了")
	}
}

// **跳不跳是"当前这一行"的属性**，不是"这一页有没有目标"的属性。
//
// 一条查不到、一条查得到的页面上：光标压着第一条时帮助栏不该说 Enter 能跳
// （按下去确实什么都不做），↓ 移过去才该出现 —— 只有把光标算进判据里，
// 帮助栏与 Update 才是同一个结论。
func TestEntryView_光标压着跳不过去的行时帮助栏不列Enter(t *testing.T) {
	e := ref.Entry{ID: "a", Title: "A",
		SeeAlso: []string{"quant:NOT_YET", "quant:Q4_K"}}
	v := NewEntryView(nil, e)
	if len(v.targets) != 2 || v.targets[0].ok || !v.targets[1].ok {
		t.Fatalf("前提不成立：targets = %+v", v.targets)
	}

	if help := strings.Join(v.Help(), " "); strings.Contains(help, keyEnter) {
		t.Errorf("光标压着跳不过去的那一行，帮助栏却列了 %s: %q", keyEnter, help)
	}
	if _, cmd := v.Update(key("enter")); cmd != nil {
		t.Error("光标压着跳不过去的那一行，按 Enter 却出了动作")
	}

	// 移到跳得动的那一条：帮助栏与行为都要跟着变
	v2, _ := v.Update(key("down"))
	v = v2.(EntryView)
	if help := strings.Join(v.Help(), " "); !strings.Contains(help, keyEnter) {
		t.Errorf("光标移到跳得动的行之后帮助栏少了 %s: %q", keyEnter, help)
	}
	if _, cmd := v.Update(key("enter")); cmd == nil {
		t.Error("跳得动的行按 Enter 没动作")
	}

	// 再移回去：Enter 又该从帮助栏消失
	v3, _ := v.Update(key("up"))
	v = v3.(EntryView)
	if help := strings.Join(v.Help(), " "); strings.Contains(help, keyEnter) {
		t.Errorf("光标移回跳不过去的行，帮助栏还列着 %s: %q", keyEnter, help)
	}
}
