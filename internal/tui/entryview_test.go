package tui

import (
	"strings"
	"testing"

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
func TestEntryView_窄终端折行(t *testing.T) {
	long := strings.Repeat("这是一段很长的中文说明，用来验证折行。", 5)
	v := NewEntryView(nil, ref.Entry{ID: "a", Title: "A", Notes: long})
	out := v.View(40, 30)
	for _, line := range strings.Split(out, "\n") {
		if n := displayWidth(line); n > 40 {
			t.Errorf("有一行宽 %d 列，超过 40:\n%s", n, line)
		}
	}
}
