package tui

import (
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/ref"
)

// metaModel 造一个有 n 条元数据的模型。
//
// **第一条是可跳的**（general.file_type 在速查表里有条目），
// 其余是普通人造键 —— 这样"带关联的行能跳、不带关联的不能跳"
// 两条断言都能落在同一个模型上。
func metaModel(n int) *model.Model {
	m := &model.Model{Path: "/x/m.gguf", Format: model.FormatGGUF, Version: "v3",
		FileSize: 1 << 20}
	m.Metadata = append(m.Metadata, model.MetaKV{
		Key: "general.file_type", Value: "15", Raw: uint32(15)})
	for i := 1; i < n; i++ {
		m.Metadata = append(m.Metadata, model.MetaKV{
			Key: fmt.Sprintf("custom.key_%03d", i), Value: fmt.Sprintf("值 %d", i)})
	}
	m.Tensors = []*model.Tensor{{Name: "w", Dtype: model.DtypeF32, ByteSize: 4, ParamCount: 1}}
	return m
}

// newMetaView 造一个停在"元数据"栏的 ModelView。
func newMetaView(m *model.Model) ModelView {
	v := NewModelView(m)
	v.cursor = int(sectionMetadata)
	return v
}

// Tab 之后 ↑↓ 走右栏的内容光标，左栏的栏目选择不动。
//
// 这是"两个光标"唯一的行为证据：只断言"Tab 改了个字段"的话，
// 把 ↑↓ 全接到左栏上也能过。
func TestModelView_元数据Tab切换焦点(t *testing.T) {
	v := newMetaView(metaModel(30))

	v2, _ := v.Update(key("tab"))
	v = v2.(ModelView)
	if v.focus != focusBody {
		t.Fatal("Tab 之后焦点不在右栏")
	}
	v2, _ = v.Update(key("down"))
	v = v2.(ModelView)
	if v.cursor != int(sectionMetadata) {
		t.Errorf("右栏焦点下左栏被动了：cursor = %d", v.cursor)
	}
	if v.metaCursor != 1 {
		t.Errorf("metaCursor = %d, want 1", v.metaCursor)
	}
}

// **元数据栏的选中项必须始终可见** —— 52 条元数据一屏放不下（实测 gemma4:26b），
// 不滚的话用户按 ↓ 越过一屏之后看不到自己在选什么，
// 此时按 Enter 打开的是一个看不见的条目。
func TestModelView_元数据选中项始终可见(t *testing.T) {
	v := newMetaView(metaModel(60))
	v2, _ := v.Update(key("tab"))
	v = v2.(ModelView)

	for range 55 {
		v3, _ := v.Update(key("down"))
		v = v3.(ModelView)
	}
	if v.metaCursor != 55 {
		t.Fatalf("metaCursor = %d, want 55", v.metaCursor)
	}
	out := v.View(110, 20)
	want := v.m.Metadata[v.metaCursor].Key
	if !strings.Contains(out, want) {
		t.Fatalf("光标在第 %d 条 (%s)，屏幕上看不到:\n%s", v.metaCursor, want, out)
	}
}

// 一屏放不下时仍然要说"还有多少条" —— 而且**不能再说"滚动在 ④b-2 里做"**：
// 滚动已经做了，那句话现在是假的。
func TestModelView_元数据提示已更新(t *testing.T) {
	v := newMetaView(metaModel(60))
	out := v.View(110, 20)
	if strings.Contains(out, "④b-2") {
		t.Errorf("界面里还留着「滚动在 ④b-2 里做」—— 那句话现在是假的:\n%s", out)
	}
	if !strings.Contains(out, "还有") && !strings.Contains(out, "显示第") {
		t.Errorf("放不下时没说漏了多少:\n%s", out)
	}
}

// Enter 在**带关联的行**上推入条目详情，带的是那一行对应的条目。
func TestModelView_元数据跳速查表(t *testing.T) {
	v := newMetaView(metaModel(10))
	want, ok := ref.FileTypeByCode(15)
	if !ok {
		t.Fatal("速查表里没有 file_type 15 —— 测试前提不成立")
	}

	// 第一条就是 general.file_type
	_, cmd := v.Update(key("enter"))
	if cmd == nil {
		t.Fatal("在带关联的行上按 Enter 没有返回命令")
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

// **跳过去时要把模型带进条目页**（`NewEntryView(v.m, e)` 而不是 `nil`）。
//
// Task 10 的"在本模型中"那一节读的就是 `EntryView.m`；传 nil 的话那一节
// 整段不显示（`occurrences()` 对 nil 的契约就是返回 nil），于是元数据栏的
// 跳转变成**单向**的：跳得过去、回不来。
//
// 这条是变异验证补出来的：把 `v.m` 改回 `nil`，**整包测试一条都不红**
// （不带 --run 跑全部也试过）—— 在 Task 10 之前，这条路径上没有东西看着它。
// 现在能观察到的最小事实是"带过去的就是当前这个模型"：`occurrences()`
// 还没实现，没有更靠外的后果可断言。
func TestModelView_跳条目时把模型带过去(t *testing.T) {
	v := newMetaView(metaModel(10))
	_, cmd := v.Update(key("enter"))
	if cmd == nil {
		t.Fatal("在带关联的行上按 Enter 没有返回命令")
	}
	msg, ok := cmd().(pushMsg)
	if !ok {
		t.Fatalf("返回的不是 pushMsg: %T", cmd())
	}
	ev, ok := msg.v.(EntryView)
	if !ok {
		t.Fatalf("推入的不是 EntryView: %T", msg.v)
	}
	if ev.m == nil {
		t.Fatal("推入的条目页没有带模型 —— Task 10 的「在本模型中」在元数据这条路" +
			"上会是空的（单向跳转）")
	}
	if ev.m != v.m {
		t.Errorf("带过去的不是当前这个模型：%p, want %p", ev.m, v.m)
	}
}

// Enter 在**不带关联的行**上什么都不做 ——
// 推一个空白进去比不推更糟：用户会以为是自己按错了。
func TestModelView_无关联行Enter无效(t *testing.T) {
	v := newMetaView(metaModel(10))
	v2, _ := v.Update(key("tab"))
	v = v2.(ModelView)
	v3, _ := v.Update(key("down")) // 移到 custom.key_001
	v = v3.(ModelView)

	_, cmd := v.Update(key("enter"))
	if cmd != nil {
		if _, isPush := cmd().(pushMsg); isPush {
			t.Error("在没有人造关联的行上按 Enter 推入了视图")
		}
	}
}

// **非元数据栏按 Tab 不该改变焦点** —— 概览/量化分布的内容是定长文字，
// 没有可选项；切过去会让 ↑↓ 突然不能切栏目，而屏幕上没有任何提示说明为什么。
func TestModelView_非元数据栏Tab无效(t *testing.T) {
	v := NewModelView(metaModel(10)) // cursor = 0 = 概览
	v2, _ := v.Update(key("tab"))
	v = v2.(ModelView)
	if v.focus != focusNav {
		t.Error("概览栏按 Tab 之后焦点被切走了 —— ↑↓ 会突然不能切栏目")
	}
	v3, _ := v.Update(key("down"))
	if v3.(ModelView).cursor != int(sectionMetadata) {
		t.Error("概览栏按 Tab 之后 ↓ 没能切到下一栏")
	}
}

// 换栏目时元数据光标归零 —— 从 60 条的元数据切走再切回来，
// 留着 55 的位置会让"还有多少条"那行算错。
func TestModelView_换栏目时元数据光标归零(t *testing.T) {
	v := newMetaView(metaModel(60))
	v2, _ := v.Update(key("tab"))
	v = v2.(ModelView)
	for range 10 {
		v3, _ := v.Update(key("down"))
		v = v3.(ModelView)
	}
	// 回左栏，切到"张量"，再切回"元数据"
	v4, _ := v.Update(key("tab"))
	v = v4.(ModelView)
	v5, _ := v.Update(key("up"))
	v = v5.(ModelView)
	if v.cursor != int(sectionMetadata)-1 {
		t.Fatalf("cursor = %d", v.cursor)
	}
	v6, _ := v.Update(key("down"))
	v = v6.(ModelView)
	if v.metaCursor != 0 {
		t.Errorf("切回元数据后 metaCursor = %d, want 0", v.metaCursor)
	}
}

// **提示里那句"显示第 N–M 条"必须与屏幕上真的显示的一致。**
//
// 照 `TestTensorsView_提示的范围与实际显示一致` 的写法：从渲染结果里
// **按行数出**实际显示了哪些条目，而不是断言"包含某段文字" ——
// 后者在"范围算错但措辞没变"时照样绿。
//
// 光标走到最后一条（窗口被下边界夹住的位置）最容易露馅：
// 本仓实测过 120 个张量时屏幕上显示 101–118、提示却写"显示第 102–120 个"。
func TestModelView_元数据提示的范围与实际显示一致(t *testing.T) {
	for _, h := range []int{6, 20, 24} {
		v := newMetaView(metaModel(60))
		v2, _ := v.Update(key("tab")) // 焦点给右栏，光标一路走到底
		v = v2.(ModelView)
		for range 59 {
			v3, _ := v.Update(key("down"))
			v = v3.(ModelView)
		}
		if v.metaCursor != 59 {
			t.Fatalf("高度 %d：光标在第 %d 条，没走到最后一条", h, v.metaCursor)
		}
		out := v.View(120, h)

		// 屏幕上真的出现了哪些条目：**按行与键比对**，不靠提示的措辞
		shown := map[int]bool{}
		for i, kv := range v.m.Metadata {
			for _, line := range strings.Split(out, "\n") {
				if strings.Contains(line, kv.Key) {
					shown[i] = true
				}
			}
		}
		if len(shown) == 0 {
			t.Fatalf("高度 %d：一条元数据都没渲染出来，这条测试无从谈起:\n%s", h, out)
		}

		// 提示里报的范围（1 基，闭区间）
		m := metaFooterRange.FindStringSubmatch(out)
		if m == nil {
			t.Fatalf("高度 %d：列表没显示完，却没有「…显示第 N–M 条」那一行:\n%s", h, out)
		}
		lo, hi := atoi(t, m[1]), atoi(t, m[2])
		if hi-lo+1 != len(shown) {
			t.Errorf("高度 %d：提示说显示了 %d 条，屏幕上却有 %d 条:\n%s",
				h, hi-lo+1, len(shown), out)
		}

		for i := range v.m.Metadata {
			inFooter := i >= lo-1 && i <= hi-1
			if shown[i] != inFooter {
				t.Errorf("高度 %d：第 %d 条（%s）%s屏幕上出现，提示却说%s"+
					"（提示写的是第 %d–%d 条）:\n%s",
					h, i, v.m.Metadata[i].Key,
					map[bool]string{true: "", false: "没"}[shown[i]],
					map[bool]string{true: "在范围内", false: "不在范围内"}[inFooter],
					lo, hi, out)
			}
		}
	}
}

var metaFooterRange = regexp.MustCompile(`…显示第 (\d+)–(\d+) 条，共 (\d+) 条`)

// **极矮终端下不能硬塞提示行。** 内容区只有 3 行时（终端 5 行），
// 标题就占了 2 行 —— 再塞一句"显示第 N–M 条"就会顶掉唯一那条数据，
// 而且被顶掉之后 padTo 会补上一句"…还有 N 行没显示"：
// 屏幕上剩下的提示反而更差（信息从"这一段是第几条"退化成"有东西没显示"）。
//
// 高度从 3 起：listCap = height-3 → 只有 h ≤ 3 时提示行才扣不出来。
//
// **直接问右栏**（`body`）：左栏固定 5 行，比 h 高时整个视图的行数由左栏
// 决定 —— 经 `View()` 断言只会测到左栏那 5 行，测不到右栏有没有硬塞。
func TestModelView_矮终端元数据不超高(t *testing.T) {
	v := newMetaView(metaModel(60))
	for _, h := range []int{3, 4, 5, 6} {
		body := v.body(80, h)
		if n := len(strings.Split(strings.TrimRight(body, "\n"), "\n")); n > h {
			t.Errorf("高度 %d：右栏给了 %d 行，超出会被根视图截掉真实内容:\n%s",
				h, n, body)
			continue
		}
		if h == 3 {
			// 唯一那条数据不能被提示行顶掉（顶掉之后 padTo 会换上
			// 它自己那句"…还有 N 行没显示"，信息更差）
			if !strings.Contains(body, "general.file_type") {
				t.Errorf("高度 3：唯一一条数据被顶掉了:\n%s", body)
			}
			if strings.Contains(body, "显示第") {
				t.Errorf("高度 3 放不下提示行，却塞进去了:\n%s", body)
			}
		}
	}
}

// **帮助栏必须与实际行为一致**（`keys.go`：列了不支持的等于骗用户按）。
//
// 元数据栏是唯一同时有 Tab 与 Enter 的栏目：Tab 切焦点、Enter 跳条目 ——
// 两条都要列，而且 ↑↓ 那句要跟着焦点换（右栏是"选择"、左栏是"切换栏目"）。
// 反面对照在概览栏：它的 Enter 什么也不做，帮助栏就不该列 Enter。
func TestModelView_元数据帮助栏与实际一致(t *testing.T) {
	nav := strings.Join(newMetaView(metaModel(10)).Help(), " ")
	for _, want := range []string{keyUp, keyDown, keyTab, keyEnter, "切换栏目"} {
		if !strings.Contains(nav, want) {
			t.Errorf("元数据栏（左栏焦点）帮助栏少了 %q: %q", want, nav)
		}
	}

	v2, _ := newMetaView(metaModel(10)).Update(key("tab"))
	body := strings.Join(v2.(ModelView).Help(), " ")
	if !strings.Contains(body, "选择") || strings.Contains(body, "切换栏目") {
		t.Errorf("焦点在右栏时帮助栏还是左栏那份: %q", body)
	}
	if !strings.Contains(body, keyTab) || !strings.Contains(body, keyEnter) {
		t.Errorf("焦点在右栏时帮助栏少了回左栏或查条目的键: %q", body)
	}

	// 反面对照：概览栏的 Enter 没有动作 —— 列了就是骗用户按
	ov := strings.Join(NewModelView(metaModel(10)).Help(), " ")
	if strings.Contains(ov, keyEnter) {
		t.Errorf("概览栏的 Enter 什么也不做，帮助栏却列了它: %q", ov)
	}
	if strings.Contains(ov, keyTab) {
		t.Errorf("概览栏没有可选项，帮助栏却列了 %s: %q", keyTab, ov)
	}
}

// **记号与跳转必须一致**：屏幕上挂着 `◂` 的行按 Enter 一定跳得动，
// 没挂的一定跳不动。
//
// 渲染时挂不挂记号、Enter 跳不跳，是两条独立的代码路径 ——
// 判定写岔了的表现是"有记号却跳不动"（或者更糟："没记号却能跳"），
// 而屏幕上看不出异常，用户只会以为是自己按错了。
func TestModelView_记号与跳转一致(t *testing.T) {
	m := metaModel(20)
	// 三类行都要有：①值释义（file_type）②键释义（按后缀命中的 block_count）
	// ③查不到的 —— 只放人造键的话，键释义那条分支一次都不会被走到
	//（变异验证确认过：把渲染的键记号整条删掉，测试照样绿）
	m.Metadata = append(m.Metadata,
		model.MetaKV{Key: "qwen2.block_count", Value: "30", Raw: uint32(30)},
		model.MetaKV{Key: "vendor.no_such_key_anywhere", Value: "1", Raw: uint32(1)})
	v := newMetaView(m)
	v2, _ := v.Update(key("tab"))
	v = v2.(ModelView)

	keyMarks, keyJumps := 0, 0
	for i, kv := range v.m.Metadata {
		if v.metaCursor != i {
			t.Fatalf("第 %d 条：光标停在第 %d 条", i, v.metaCursor)
		}
		var line string
		for _, l := range strings.Split(v.View(120, 40), "\n") {
			if strings.Contains(l, kv.Key) {
				line = l
			}
		}
		if line == "" {
			t.Fatalf("第 %d 条（%s）没渲染出来:\n%s", i, kv.Key, v.View(120, 40))
		}
		hasMark := strings.Contains(line, "◂")

		_, cmd := v.Update(key("enter"))
		jumped := false
		if cmd != nil {
			_, jumped = cmd().(pushMsg)
		}
		if hasMark != jumped {
			t.Errorf("第 %d 条（%s）：记号=%v 但跳得动=%v —— %q",
				i, kv.Key, hasMark, jumped, line)
		}
		if _, isValueRef := fileTypeEntry(kv); !isValueRef {
			if hasMark {
				keyMarks++
			}
			if jumped {
				keyJumps++
			}
		}
		if i < len(v.m.Metadata)-1 {
			v3, _ := v.Update(key("down"))
			v = v3.(ModelView)
		}
	}
	// 正对照：键释义那一条分支必须真的被走到过
	if keyMarks == 0 || keyJumps == 0 {
		t.Fatalf("没有一行是靠键释义命中的（记号 %d 行、跳得动 %d 行）—— "+
			"这条测试测不到那条分支", keyMarks, keyJumps)
	}
}

// **元数据为空时不能崩**：safetensors 可以没有 __metadata__ 段
// （parser 里是 `if len(hdr.Metadata) > 0`），而这一栏照样能走到 ——
// 光标压在 0 上、一条都没有，取下标就是越界 panic。
// 而且它不只是"空文件"才有：只要够运气不好，进这一栏按一下 Enter 就撞上。
func TestModelView_元数据为空不崩(t *testing.T) {
	m := &model.Model{Path: "/x/m.safetensors", Format: model.FormatSafeTensors, FileSize: 1 << 10}
	m.Tensors = []*model.Tensor{{Name: "w", Dtype: model.DtypeF32, ByteSize: 4, ParamCount: 1}}
	v := gotoSection(NewModelView(m), sectionMetadata)

	if out := v.View(100, 20); !strings.Contains(out, "元数据（0 条）") {
		t.Errorf("空元数据栏没渲染出标题:\n%s", out)
	}
	// 空的一栏没有可选项 —— Tab 不该把焦点切走：
	// 切过去之后 ↑↓ 再也不能切栏目，而屏幕上只是个 0 条的空列表
	if v2, _ := v.Update(key("tab")); v2.(ModelView).focus != focusNav {
		t.Error("空元数据栏按 Tab 切走了焦点 —— ↑↓ 会突然不能切栏目")
	}
	// 空列表上按 Enter 不该崩（下标是越界的），也不该推视图
	if _, cmd := v.Update(key("enter")); cmd != nil {
		if _, isPush := cmd().(pushMsg); isPush {
			t.Error("空元数据上按 Enter 推入了视图")
		}
	}
	// 左栏照常能用：↓ 到张量栏、↑ 回元数据，全程不该崩
	v3, _ := v.Update(key("down"))
	v = v3.(ModelView)
	if v.cursor != int(sectionTensors) {
		t.Fatalf("空元数据栏之后 ↓ 停在第 %d 栏，want 张量栏", v.cursor)
	}
	v4, _ := v.Update(key("up"))
	if got := v4.(ModelView); got.metaCursor != 0 {
		t.Errorf("换栏目后 metaCursor = %d, want 0", got.metaCursor)
	}
}

// **帮助栏与实际行为必须逐栏、逐行一致** —— `keys.go` 那条"列了不支持的
// 等于骗用户按、支持的不能漏"的机械化版本。
//
// 不是重复劳动：`opensOnEnter`（帮助栏列不列 Enter）与 `Update` 的
// Enter 分支是同一个决定的两处写法，漂移了没有别的测试会红
// （症状只是"帮助栏写了一个按了没反应的键"）。Tab 同理。
//
// **必须走遍每一行**：第一版只在 cursor=0 上比对，而 `metaModel` 的第 0 条
// （general.file_type）恰好跳得动 —— 判据退回成栏目级（"这一栏有条目就列
// Enter"）时它照样绿，1..9 那些跳不动的行上印的假话一条都没被看见
// （实测旧实现：第 1 条 custom.key_001 帮助栏列 Enter、按下去没动作）。
// 所以 fixture 要同时有可跳与不可跳的行，且每一行都要真的把光标移过去。
//
// 空元数据的模型也走一遍：它是唯一"栏目在、但没有可选项"的情况。
func TestModelView_帮助与行为逐行一致(t *testing.T) {
	// 三类行都要有：①值释义（file_type，可跳）②键释义（按后缀命中的
	// block_count，可跳）③查不到的（不可跳）—— 只放一类的话，
	// "逐行判定"与"栏目级判定"在 fixture 上恰好等价，测不出区别。
	withRefs := metaModel(10)
	withRefs.Metadata = append(withRefs.Metadata,
		model.MetaKV{Key: "qwen2.block_count", Value: "30", Raw: uint32(30)})

	empty := &model.Model{Path: "/x/m.safetensors", Format: model.FormatSafeTensors, FileSize: 1 << 10}
	empty.Tensors = []*model.Tensor{{Name: "w", Dtype: model.DtypeF32, ByteSize: 4, ParamCount: 1}}

	for _, tc := range []struct {
		name string
		m    *model.Model
	}{{"有元数据", withRefs}, {"空元数据", empty}} {
		for sec := section(0); int(sec) < len(sections); sec++ {
			// 元数据栏逐行走（0..n-1）；其余栏目没有"行"这个概念，走一次
			rows := 1
			if sec == sectionMetadata {
				rows = max(len(tc.m.Metadata), 1)
			}
			v := gotoSection(NewModelView(tc.m), sec)
			if sec == sectionMetadata && len(tc.m.Metadata) > 0 {
				// 先 Tab 进右栏，↓ 才走"行"；否则 ↓ 切的是栏目
				v2, _ := v.Update(key("tab"))
				v = v2.(ModelView)
			}
			for row := 0; row < rows; row++ {
				if v.metaCursor != row {
					t.Fatalf("%s·%s：想验第 %d 行，光标却停在 %d —— 这条测试没走到那一行",
						tc.name, sections[sec], row, v.metaCursor)
				}
				name := fmt.Sprintf("%s·%s·第 %d 行", tc.name, sections[sec], row)
				assertHelpMatchesAction(t, name, v)
				if row == rows-1 {
					break
				}
				v2, _ := v.Update(key("down"))
				v = v2.(ModelView)
			}
		}
	}
}

// assertHelpMatchesAction 断言一个 ModelView 的帮助栏与按键的实际效果一致。
//
// Enter 与 Tab 两处都是"帮助栏列不列"与"按下去动不动"的比对 ——
// 抽出来是因为逐行那条测试要对每一行调一次，两个测试再各抄一遍的话，
// 将来加一个键就得在两处同步改。
func assertHelpMatchesAction(t *testing.T, name string, v ModelView) {
	t.Helper()
	help := strings.Join(v.Help(), " ")

	_, cmd := v.Update(key("enter"))
	acts := false
	if cmd != nil {
		_, acts = cmd().(pushMsg)
	}
	if got := strings.Contains(help, keyEnter); got != acts {
		t.Errorf("%s：帮助栏列 Enter=%v，按下去有动作=%v —— help=%q",
			name, got, acts, help)
	}

	v2, _ := v.Update(key("tab"))
	moved := v2.(ModelView).focus != v.focus
	if got := strings.Contains(help, keyTab); got != moved {
		t.Errorf("%s：帮助栏列 %s=%v，按下去焦点动了=%v —— help=%q",
			name, keyTab, got, moved, help)
	}
}

// **载入中（m == nil）时帮助栏与实际行为也要一致。**
//
// 根视图每一帧都调 Help()，而 Update 开头那句 `v.m == nil` 早退让所有
// Enter 都不做任何事 —— 这时帮助栏列着 Enter 就是同一类假话。
// "张量"/"速查表"两栏尤其明显：它们的判据与元数据不同，条目数根本
// 还没得判，所以 `opensOnEnter` 里那个 `v.m != nil` 是单独的一格。
//
// 这条与"逐行一致"合起来覆盖 ModelView 的全部状态：
// 后者管"有模型、光标停在哪一行"，这条管"模型还没回来"。
func TestModelView_载入中帮助与行为一致(t *testing.T) {
	for sec := section(0); int(sec) < len(sections); sec++ {
		v := gotoSection(NewModelViewFromPath("/x/m.gguf", "m"), sec)
		assertHelpMatchesAction(t, "载入中·"+sections[sec], v)

		// 反面对照：解析完成后，"张量"/"速查表"两栏要真的列出 Enter ——
		// 少了的话上面那条断言只是"两边都空"，什么也没验到。
		// （元数据栏的行级比对在"逐行一致"那条里，概览与量化分布本来就没有）
		if sec != sectionTensors && sec != sectionRef {
			continue
		}
		loaded := gotoSection(loadModelView(metaModel(10)), sec)
		if help := strings.Join(loaded.Help(), " "); !strings.Contains(help, keyEnter) {
			t.Errorf("解析完成后·%s：帮助栏少了 %s: %q", sections[sec], keyEnter, help)
		}
	}
}

// **解析还没回来时（m == nil）按到元数据栏，Help() 不能崩。**
//
// `hasBodyCursor`/`opensOnEnter` 要看元数据条数，而根视图**每一帧**都调
// Help()（`View()` 有自己的 m == nil 早退，Help() 没有）。所以这不是理论
// 路径：在"正在解析…"上按一下 ↓ 就停在元数据栏了。
func TestModelView_载入中帮助栏不崩(t *testing.T) {
	v := gotoSection(NewModelViewFromPath("/x/m.gguf", "m"), sectionMetadata)
	help := strings.Join(v.Help(), " ")
	if strings.Contains(help, keyTab) || strings.Contains(help, keyEnter) {
		t.Errorf("模型还没解析出来，帮助栏却说这一栏有可选项: %q", help)
	}
	// 正对照：解析完成后同一栏两样都要出现
	loaded := strings.Join(gotoSection(loadModelView(metaModel(10)), sectionMetadata).Help(), " ")
	if !strings.Contains(loaded, keyTab) || !strings.Contains(loaded, keyEnter) {
		t.Errorf("解析完成后帮助栏少了 %s 或 %s: %q", keyTab, keyEnter, loaded)
	}
}
