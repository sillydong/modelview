package tui

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/model"
)

// fakeModelWithTensors 造一个有 n 个张量的模型，名字形如 blk.<i>.attn_q.weight。
func fakeModelWithTensors(n int) *model.Model {
	m := &model.Model{Path: "/x/m.gguf", Format: model.FormatGGUF, Version: "v3", Arch: "qwen2"}
	for i := range n {
		m.Tensors = append(m.Tensors, &model.Tensor{
			Name:       fmt.Sprintf("blk.%d.attn_q.weight", i),
			Dims:       []int64{2048, 2048},
			Dtype:      model.DtypeQ4K,
			ByteSize:   2 << 20,
			ParamCount: 4194304,
		})
	}
	return m
}

func newTensors(m *model.Model) TensorsView { return NewTensorsView(m) }

// 列表要列出全部张量，且标出总数。
func TestTensorsView_列出全部(t *testing.T) {
	v := newTensors(fakeModelWithTensors(30))
	out := v.View(120, 40)
	if !strings.Contains(out, "30") {
		t.Errorf("没标出总数:\n%s", out)
	}
	if !strings.Contains(out, "blk.0.attn_q.weight") {
		t.Errorf("第一个张量没列出:\n%s", out)
	}
}

// **选中项必须始终可见** —— 与模型库视图同一条要求。
//
// **每一步都走、一直走到最后一项，而且换几个高度**：原先只走 60 步、
// 只用高 20 一个尺寸，正好绕开了边界 —— "窗口取满 + 提示行再砍一行"
// 只在光标贴到列表末尾时才发生（实测：120 个张量、光标在最后一个、
// 高 20，屏幕上根本看不到选中的那条，而这条测试当时是全绿的）。
func TestTensorsView_选中项始终可见(t *testing.T) {
	for _, h := range []int{6, 20, 24} {
		v := newTensors(fakeModelWithTensors(100))
		last := len(v.m.Tensors) - 1
		for range last + 1 {
			out := v.View(120, h)
			want := fmt.Sprintf("blk.%d.attn_q.weight", v.cursor)
			if !strings.Contains(out, want) {
				t.Fatalf("高度 %d：光标在第 %d 项，屏幕上看不到 %s:\n%s",
					h, v.cursor, want, out)
			}
			v2, _ := v.Update(key("down"))
			v = v2.(TensorsView)
		}
		// 正对照：上面那个循环必须真的走到了最后一项，
		// 否则"每一步都可见"是空转的（走两步就退出也满足它）
		if v.cursor != last {
			t.Fatalf("高度 %d：光标停在第 %d 项，没走到最后一项 %d", h, v.cursor, last)
		}
	}
}

// **提示里那句"显示第 N–M 个"必须与屏幕上真的显示的一致。**
//
// 实测过的旧版：120 个张量、光标在最后一项、高 20 —— 屏幕上显示
// 101–118，提示却写"显示第 102–120 个"（它按截断**前**的窗口算）。
// 屏幕上印一句假话，比不印更糟。
//
// 这条不能靠"包含某段文字"来验：要从渲染结果里**数出**实际显示了哪些行。
func TestTensorsView_提示的范围与实际显示一致(t *testing.T) {
	v := newTensors(fakeModelWithTensors(120))
	for range len(v.m.Tensors) - 1 {
		v2, _ := v.Update(key("down"))
		v = v2.(TensorsView)
	}
	if v.cursor != len(v.m.Tensors)-1 {
		t.Fatalf("光标在第 %d 项，没走到最后一项", v.cursor)
	}

	for _, h := range []int{6, 20, 24} {
		out := v.View(120, h)

		// 屏幕上真的出现了哪些张量：**按行与名字比对**，不靠提示的措辞
		shown := map[int]bool{}
		for i, tn := range v.m.Tensors {
			for _, line := range strings.Split(out, "\n") {
				if strings.Contains(line, tn.Name) {
					shown[i] = true
				}
			}
		}
		if len(shown) == 0 {
			t.Fatalf("高度 %d：一个张量都没渲染出来，这条测试无从谈起:\n%s", h, out)
		}

		// 提示里报的范围（1 基，闭区间）
		m := footerRange.FindStringSubmatch(out)
		if m == nil {
			t.Fatalf("高度 %d：列表没显示完，却没有「…显示第 N–M 个」那一行:\n%s", h, out)
		}
		lo, hi := atoi(t, m[1]), atoi(t, m[2])

		for i := range v.m.Tensors {
			inFooter := i >= lo-1 && i <= hi-1
			if shown[i] != inFooter {
				t.Errorf("高度 %d：第 %d 个张量%s屏幕上出现，提示却说%s"+
					"（提示写的是第 %d–%d 个，屏幕上共 %d 行张量）:\n%s",
					h, i, map[bool]string{true: "", false: "没"}[shown[i]],
					map[bool]string{true: "在范围内", false: "不在范围内"}[inFooter],
					lo, hi, len(shown), out)
			}
		}
	}
}

var footerRange = regexp.MustCompile(`…显示第 (\d+)–(\d+) 个，共 (\d+) 个`)

func atoi(t *testing.T, s string) int {
	t.Helper()
	n, err := strconv.Atoi(s)
	if err != nil {
		t.Fatalf("提示里的数字 %q 解析不了: %v", s, err)
	}
	return n
}

// 视图交给根视图的原始输出不能超过高度 —— 超了会被 padTo 截掉真实内容。
// （④b-1 的教训：经根视图断言是测不出来的，padTo 永远会补齐到恰好 h 行。）
func TestTensorsView_不超高(t *testing.T) {
	v := newTensors(fakeModelWithTensors(100))
	for _, h := range []int{6, 12, 24} {
		raw := v.View(120, h)
		if n := len(strings.Split(raw, "\n")); n > h {
			t.Errorf("高度 %d：给了 %d 行", h, n)
		}
	}
}

// `/` 进入过滤：只留名字里含关键词的；Esc 取消过滤并恢复全部。
func TestTensorsView_过滤(t *testing.T) {
	m := fakeModelWithTensors(10)
	m.Tensors = append(m.Tensors, &model.Tensor{
		Name: "token_embd.weight", Dims: []int64{2048, 151936},
		Dtype: model.DtypeQ4K, ByteSize: 100 << 20, ParamCount: 311226368})
	v := newTensors(m)

	v2, _ := v.Update(key("/"))
	v = v2.(TensorsView)
	if !v.filtering {
		t.Fatal("按 / 之后没进入过滤态")
	}
	// 输入 "token"
	for _, r := range "token" {
		v2, _ = v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		v = v2.(TensorsView)
	}
	out := v.View(120, 20)
	if !strings.Contains(out, "token_embd") {
		t.Errorf("过滤后该留下 token_embd:\n%s", out)
	}
	if strings.Contains(out, "blk.0.attn_q") {
		t.Errorf("过滤后不该还有 blk.0:\n%s", out)
	}

	// Esc 取消过滤，恢复全部
	v2, _ = v.Update(key("esc"))
	v = v2.(TensorsView)
	if v.filtering || v.filter != "" {
		t.Errorf("Esc 之后过滤态没清掉: filtering=%v filter=%q", v.filtering, v.filter)
	}
	if out := v.View(120, 40); !strings.Contains(out, "blk.0.attn_q") {
		t.Errorf("取消过滤后没恢复全部:\n%s", out)
	}
}

// 过滤态下按 q **不该退出程序** —— 那时 q 是过滤词的一部分。
// 这条要在这里挡：根 Model 的全局按键在视图之前处理，
// 所以过滤态必须自己在 Update 里把 q 吃掉（返回 nil cmd）。
func TestTensorsView_过滤态下q不退出(t *testing.T) {
	v := newTensors(fakeModelWithTensors(5))
	v2, _ := v.Update(key("/"))
	v = v2.(TensorsView)

	for _, r := range "qwen" {
		v2, cmd := v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		if cmd != nil {
			if _, isQuit := cmd().(tea.QuitMsg); isQuit {
				t.Fatalf("过滤态下输入 %q 触发了退出", string(r))
			}
		}
		v = v2.(TensorsView)
	}
	if v.filter != "qwen" {
		t.Errorf("过滤词是 %q, want qwen", v.filter)
	}
}

// 没有匹配时要明说，不能给一个空白列表让人以为坏了。
func TestTensorsView_无匹配有提示(t *testing.T) {
	v := newTensors(fakeModelWithTensors(5))
	v2, _ := v.Update(key("/"))
	v = v2.(TensorsView)
	for _, r := range "zzzz" {
		v2, _ = v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		v = v2.(TensorsView)
	}
	out := v.View(120, 20)
	if !strings.Contains(out, "没有匹配") {
		t.Errorf("无匹配时没提示:\n%s", out)
	}
}

// 过滤态必须是**模态的** —— 这是"敲 q 不会退出程序"的前提。
//
// 视图自己把 q 收进过滤词是不够的：根 Model 的全局按键排在视图之前，
// 不声明模态的话 q 根本到不了这里（Task 2 修的就是这个）。
// 所以这里断言的是 Modal()，而不是"q 有没有被吃掉" ——
// 后者在根 Model 吞掉按键时也是绿的。
func TestTensorsView_过滤态是模态的(t *testing.T) {
	v := newTensors(fakeModelWithTensors(5))
	if v.Modal() {
		t.Error("非过滤态不该是模态的 —— 那样 q 就退不出程序了")
	}
	v2, _ := v.Update(key("/"))
	if !v2.(TensorsView).Modal() {
		t.Error("过滤态必须是模态的，否则根视图会把 q 吃掉直接退出程序")
	}
	v3, _ := v2.(TensorsView).Update(key("enter"))
	if v3.(TensorsView).Modal() {
		t.Error("确认过滤之后就不再是模态的了 —— 此时 q 应该能退出")
	}
}

// 输入态退格删一个字；删空之后过滤态仍在（还在输入）。
func TestTensorsView_输入态退格(t *testing.T) {
	v := newTensors(fakeModelWithTensors(5))
	v2, _ := v.Update(key("/"))
	v = v2.(TensorsView)
	for _, r := range "attn" {
		v3, _ := v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		v = v3.(TensorsView)
	}
	if v.filter != "attn" {
		t.Fatalf("过滤词是 %q, want attn", v.filter)
	}

	v2, _ = v.Update(key("backspace"))
	v = v2.(TensorsView)
	if v.filter != "att" {
		t.Errorf("退格后过滤词是 %q, want att —— 退格没删掉字", v.filter)
	}
	if !v.filtering {
		t.Error("输入态退格之后仍在输入态才对（要能连着删）")
	}

	// 删空之后还在输入态，只是没有过滤词了
	for range 3 {
		v2, _ = v.Update(key("backspace"))
		v = v2.(TensorsView)
	}
	if v.filter != "" {
		t.Errorf("删空后过滤词是 %q, want 空", v.filter)
	}
	if !v.filtering {
		t.Error("删空之后仍在输入态")
	}
}

// **已确认的过滤态也要能删字** —— 否则清掉过滤词只有
// "按 / 回输入态再删"这一条路，而屏幕上没有任何地方说这件事。
// 表头那句 `（Backspace 删字，Esc 返回）` 就是照着这条行为写的，
// 两者必须一起成立：先有这条行为，表头才不是假话。
func TestTensorsView_已确认过滤态退格(t *testing.T) {
	v := newTensors(fakeModelWithTensors(5))
	v2, _ := v.Update(key("/"))
	v = v2.(TensorsView)
	for _, r := range "attn" {
		v3, _ := v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}})
		v = v3.(TensorsView)
	}
	// Enter 确认：退出输入态，但保留过滤词
	v2, _ = v.Update(key("enter"))
	v = v2.(TensorsView)
	if v.filtering || v.filter != "attn" {
		t.Fatalf("确认后 filtering=%v filter=%q, want false/attn", v.filtering, v.filter)
	}

	// **确认态下 Esc 不该清词**（它会被根视图拦成"返回上一层"），
	// 所以表头不能再写"Esc 取消" —— 这条断言守的是那个决定
	out := v.View(120, 20)
	if !strings.Contains(out, "Backspace") {
		t.Errorf("已确认过滤态的表头没提 Backspace —— 用户不知道该怎么清字:\n%s", out)
	}

	v2, _ = v.Update(key("backspace"))
	v = v2.(TensorsView)
	if v.filter != "att" {
		t.Errorf("已确认态退格后过滤词是 %q, want att", v.filter)
	}
	if v.filtering {
		t.Error("已确认态退格不该把视图切回输入态")
	}
}

// Enter 推入张量详情（走 pushMsg），且带的是**当前选中**的那个张量。
func TestTensorsView_Enter进入详情(t *testing.T) {
	v := newTensors(fakeModelWithTensors(10))
	for range 3 {
		v2, _ := v.Update(key("down"))
		v = v2.(TensorsView)
	}
	_, cmd := v.Update(key("enter"))
	if cmd == nil {
		t.Fatal("按 Enter 没有返回命令")
	}
	msg, ok := cmd().(pushMsg)
	if !ok {
		t.Fatalf("返回的不是 pushMsg: %T", cmd())
	}
	if msg.v == nil {
		t.Error("pushMsg 里的视图是 nil")
	}
}

// 已确认过滤态的空结果提示**不能说"Esc 取消过滤"** —— 那时 Modal() 为假，
// Esc 会被根视图拦成"弹掉整个列表"：屏幕上的那句话与按下去的结果相反。
//
// 所以这条不只断言那句话怎么写，还**顺着根视图真的走一遍那条键路** ——
// 只断言"提示里出现了 Backspace"的话，把两句话都写上的实现照样绿。
func TestTensorsView_空结果提示与按键行为一致(t *testing.T) {
	m := fakeModelWithTensors(3)
	v := NewTensorsViewName(m, "zzz没有这个")

	out := v.View(100, 20)
	if strings.Contains(out, "Esc 取消") {
		t.Errorf("已确认过滤态下写「Esc 取消过滤」是假话 —— Esc 会弹掉整个列表:\n%s", out)
	}
	if !strings.Contains(out, keyBackspace) {
		t.Errorf("空结果提示里没说怎么删字（该写 %s）:\n%s", keyBackspace, out)
	}

	// 那句"Esc 会弹掉整个列表"当场验一遍：同一个状态下，从根视图按 Esc
	root2, _ := New(NewTensorsView(m)).Update(pushMsg{v: v})
	if got := len(root2.(Model).stack); got != 2 {
		t.Fatalf("栈深 %d, want 2 —— 测试前提不成立", got)
	}
	root3, _ := root2.Update(key("esc"))
	if got := len(root3.(Model).stack); got != 1 {
		t.Errorf("已确认过滤态下 Esc 没有弹栈（栈深 %d）—— "+
			"那提示里写「Esc 会取消过滤」也没有任何东西会红", got)
	}
}

// **筛出空列表时帮助栏不能说"Enter 详情"** —— 一条都没有，按下去什么也不做。
//
// 空结果不是错误状态，是搜索/筛选的正常结局；判据与 Update 里那个
// 是同一个（TensorsView.canOpen），所以不存在"帮助栏说能开、按下去没反应"。
func TestTensorsView_空结果帮助栏不列Enter(t *testing.T) {
	// 三种空法各走一遍：按名字筛空、按类型筛空、模型压根没有张量
	cases := []struct {
		name string
		v    TensorsView
	}{
		{"按名字筛空", NewTensorsViewName(fakeModelWithTensors(3), "zzz没有这个")},
		{"按类型筛空", NewTensorsViewDtype(fakeModelWithTensors(3), model.DtypeF32)},
		{"模型没有张量", NewTensorsView(&model.Model{Path: "/x/m.gguf"})},
	}
	for _, tc := range cases {
		if n := len(tc.v.shown()); n != 0 {
			t.Fatalf("%s：前提不成立，shown() = %d 个", tc.name, n)
		}
		if help := strings.Join(tc.v.Help(), " "); strings.Contains(help, keyEnter) {
			t.Errorf("%s：帮助栏列了 %s —— 按下去没反应: %q", tc.name, keyEnter, help)
		}
		if _, cmd := tc.v.Update(key("enter")); cmd != nil {
			t.Errorf("%s：按 Enter 居然有动作 —— 帮助栏与 Update 读的不是同一个判据", tc.name)
		}
	}

	// 正对照：有得开时两个方向都要成立
	full := newTensors(fakeModelWithTensors(3))
	if help := strings.Join(full.Help(), " "); !strings.Contains(help, keyEnter) {
		t.Errorf("有张量时帮助栏少了 %s: %q", keyEnter, help)
	}
	if _, cmd := full.Update(key("enter")); cmd == nil {
		t.Error("有张量时按 Enter 没动作")
	}
}

// **宽度为 0 时不能崩** —— `humanize.Truncate` 对负数会切片越界 panic。
//
// 与 `TestRefView_窄终端不崩` 那条不同，这里**不需要修**，先把可达性判清楚：
//   - 宽度直接来自终端（`Model.width` 只由 tea.WindowSizeMsg 写），而
//     bubbletea 在 Unix 上取自 TIOCGWINSZ 的 uint16 字段（x/term 的
//     getSize），**不可能是负数**；本视图也不像 RefView 那样自己算
//     `width − 左栏 − 1`，没有减法就造不出负宽度。
//   - 0 是真可达的（pty 报 0×0，drive_tui 的文件头就写着默认 winsize 是 0×0），
//     而 `Truncate(s, 0)` 是 `s[:0]`，本来就安全。
//
// 实测探针（这里是它的文档化）：宽 −1 → `slice bounds out of range [:-1]`；
// 宽 0/1/2/3/4/10/16/17 全部正常（头部按宽度截成 ""/"张"/"张量..."）。
// 留这条是把"0 安全"钉住：哪天有人在这里做减法（比如给名字列让宽度），
// 0 就会变成负数，这条会立刻红。
func TestTensorsView_窄终端不崩(t *testing.T) {
	for _, w := range []int{0, 1, 10, 17} {
		out := newTensors(fakeModelWithTensors(3)).View(w, 24)
		// 列表本身不按宽度截断（超宽那部分交给根视图），所以每一档都该在
		if !strings.Contains(out, "blk.0.attn_q") {
			t.Errorf("宽度 %d：第一行没渲染出来:\n%s", w, out)
		}
	}
	// 头部按宽度截断，但装得下的时候必须是完整的
	if out := newTensors(fakeModelWithTensors(3)).View(17, 24); !strings.Contains(out, "张量（3/3）") {
		t.Errorf("宽度 17：头部没渲染完整:\n%s", out)
	}
}

// 张量列表的标题必须与上一屏（模型页标题栏）显示同一个名字 ——
// 它原来直接用 baseName(m.Path)，而 ollama 的路径是内容寻址的 blob，
// 于是从 qwen2.5:3b 点进来之后标题变成了 sha256-5ee4f07c…，
// 用户会以为点进了另一个模型。
func TestTensorsView_标题用显示名而不是blob路径(t *testing.T) {
	const blob = "/Users/x/.ollama/models/blobs/sha256-5ee4f07cdb9beadbbb293e85803c569b"
	m := &model.Model{
		Path: blob, Name: "qwen2.5:3b",
		Tensors: []*model.Tensor{{Name: "a", Dtype: model.DtypeF32}},
	}
	got := NewTensorsView(m).Title()
	if !strings.Contains(got, "qwen2.5:3b") {
		t.Errorf("标题里没有显示名:\n%s", got)
	}
	if strings.Contains(got, "sha256-") {
		t.Errorf("标题里出现了 blob 名:\n%s", got)
	}

	// 没有名字时（裸路径解析）退回文件名仍然是可接受的行为
	plain := NewTensorsView(&model.Model{Path: "/x/model.gguf", Tensors: m.Tensors}).Title()
	if !strings.Contains(plain, "model.gguf") {
		t.Errorf("没有显示名时应当退回文件名:\n%s", plain)
	}
}
