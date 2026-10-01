package tui

import (
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/model"
)

// tourModel 是一个小而全的模型：有元数据、有多种类型的张量。
func tourModel() *model.Model {
	m := &model.Model{Path: "/x/m.gguf", Format: model.FormatGGUF, Version: "v3",
		FileSize: 1 << 20}
	m.Metadata = []model.MetaKV{
		{Key: "general.architecture", Value: "qwen2"},
		{Key: "general.file_type", Value: "15", Raw: uint32(15)},
	}
	m.Tensors = []*model.Tensor{
		{Name: "blk.0.attn_q.weight", Dtype: model.DtypeQ4K,
			Dims: []int64{64}, ByteSize: 36, ParamCount: 64},
		{Name: "token_embd.weight", Dtype: model.DtypeQ6K,
			Dims: []int64{64}, ByteSize: 52, ParamCount: 64},
	}
	return m
}

// **从模型库一路点进张量详情再退回来** —— 每一步都经过根 Model。
//
// 分视图的单元测试各自都是绿的，但它们证明不了"按键真的能走到那一步"：
// 根视图的全局按键、栈的推入弹出、Init 的执行时机都不在各视图的测试范围里。
// 实测踩过只在这条路径上才会暴露的 bug（Library.Init 从没被执行过）。
func TestTour_走完整条视图链(t *testing.T) {
	// **栈底用 fakeView 而不是模型库**：模型库的 Init 会真的扫盘，
	// 而这条测试要验的是"按键能不能走通整条链"，
	// 不需要一个真实（且因机器而异）的模型库。
	root := New(fakeView{title: "根"})
	m := root

	// 直接推进模型视图（模型库的 Enter 已经在别处测过）
	next, _ := m.Update(pushMsg{v: NewModelView(tourModel())})
	m = next.(Model)

	// 元数据 → Tab → 选中 file_type 那一行
	next, _ = m.Update(key("down"))
	m = next.(Model)
	next, _ = m.Update(key("tab"))
	m = next.(Model)
	next, _ = m.Update(key("down"))
	m = next.(Model)

	// Enter 进速查表条目
	next, cmd := m.Update(key("enter"))
	m = next.(Model)
	m = runCmd(t, m, cmd)

	// 条目页里应能跳回模型（"在本模型中"）
	top := m.stack[len(m.stack)-1]
	ev, ok := top.(EntryView)
	if !ok {
		t.Fatalf("栈顶是 %T, want EntryView", top)
	}
	if len(ev.targets) == 0 {
		t.Fatal("条目页里一个跳转目标都没有 —— 「在本模型中」那一段没接上")
	}

	// Enter 跳回去：栈应当弹到 ModelView，且光标停在元数据那一行
	next, cmd = m.Update(key("enter"))
	m = next.(Model)
	m = runCmd(t, m, cmd)

	if len(m.stack) != 2 {
		t.Errorf("栈深 = %d, want 2（栈底 + ModelView）—— 没有弹回模型页", len(m.stack))
	}
	mv, ok := m.stack[len(m.stack)-1].(ModelView)
	if !ok {
		t.Fatalf("栈顶是 %T, want ModelView", m.stack[len(m.stack)-1])
	}
	if mv.focus != focusBody {
		t.Error("跳回来之后焦点不在内容栏 —— 用户还得再按一次 Tab 才能继续看")
	}
	if mv.metaCursor != 1 {
		t.Errorf("选中的是第 %d 条, want 1（general.file_type）", mv.metaCursor)
	}

	// Esc 一路退回模型库
	for range 3 {
		next, _ = m.Update(key("esc"))
		m = next.(Model)
	}
	if len(m.stack) != 1 {
		t.Errorf("连按 Esc 之后栈深 = %d, want 1", len(m.stack))
	}
}

// runCmd 执行一条命令直到拿到消息（nil 命令返回 nil）。
//
// **不能只判 nil 就跳过**：有些路径上命令是 tea.Batch 包起来的，
// 那种情况下 cmd() 返回的是 BatchMsg，需要展开 —— 这里直接断言
// "拿到的不是 BatchMsg"，逼着实现别在测试路径上用它。
func runCmd(t *testing.T, m Model, cmd tea.Cmd) Model {
	t.Helper()
	if cmd == nil {
		t.Fatal("这一步没有返回命令 —— 链子断了")
	}
	msg := cmd()
	if _, isBatch := msg.(tea.BatchMsg); isBatch {
		t.Fatal("返回了 BatchMsg，这条测试不处理批处理命令")
	}
	next, _ := m.Update(msg)
	return next.(Model)
}
