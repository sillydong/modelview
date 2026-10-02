package tui

import (
	"context"
	"errors"

	"fmt"
	"strconv"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/model"
)

// fakeScan 造一个不碰磁盘的单张量扫描：把"扫过了"写进 tn
// （NeedsWork 与详情页都看这一份），fail 里的名字返回错误。
//
// 它写的是**传进来的那个 tn**（生产代码传的是副本），所以
// "结果什么时候合并回 m"这件事在测试里是可观察的。
func fakeScan(fail map[string]bool) func(context.Context, *model.Model, *model.Tensor) error {
	return func(_ context.Context, _ *model.Model, tn *model.Tensor) error {
		if fail[tn.Name] {
			return errors.New("假的读取失败：" + tn.Name)
		}
		tn.Stats = &model.Stats{Count: tn.ParamCount}
		// **三个字段都要造**：合并那一行是 `Stats, Quant, QuantSims` 一起
		// 赋的，少造一个就有一条"漏合并也不会有测试红"的路 ——
		// 变异验证实测：把 Quant 那格改成 nil，只造 Stats/QuantSims 的
		// 那版测试全绿（详情页的「量化诊断」整节消失，不报错）。
		tn.Quant = &model.QuantInfo{Scheme: "Q4_K", Blocks: tn.ParamCount}
		tn.QuantSims = []model.QuantSim{{Target: "Q8_0", BitsPerWeight: 8.5}}
		return nil
	}
}

// fakeModel 造一个与真实文件同形状的最小模型（qwen2.5:3b 的骨架）。
func fakeModel() *model.Model {
	return &model.Model{
		Path:     "/x/qwen2.5-3b.gguf",
		Format:   model.FormatGGUF,
		Version:  "v3",
		FileSize: 1929903008,
		Arch:     "qwen2",
		Metadata: []model.MetaKV{
			{Key: "general.architecture", Value: "qwen2", Raw: "qwen2"},
			{Key: "general.file_type", Value: "15", Raw: uint32(15)},
			{Key: "qwen2.block_count", Value: "36", Raw: uint32(36)},
		},
		Tensors: []*model.Tensor{
			{Name: "token_embd.weight", Dims: []int64{2048, 151936},
				Dtype: model.DtypeQ4K, ByteSize: 100 << 20, ParamCount: 311226368},
			{Name: "blk.0.attn_q.weight", Dims: []int64{2048, 2048},
				Dtype: model.DtypeQ4K, ByteSize: 2 << 20, ParamCount: 4194304},
		},
	}
}

// loadModelView 造一个已经加载好的视图，跳过异步那一步。
func loadModelView(m *model.Model) ModelView {
	v, _ := NewModelView(m).Update(modelLoadedMsg{m: m})
	return v.(ModelView)
}

// gotoSection 按 ↓ 走到目标栏目。
//
// **不直接写 section(v.cursor)**：那个字段已经删了（它是 cursor 的派生值），
// 而且直接写字段会绕过生产路径 —— 测试里"按 ↓ 导航"这条路径
// 就只在 TestModelView_导航 里被走过一次。
func gotoSection(v ModelView, want section) ModelView {
	for int(v.cursor) < int(want) {
		v2, _ := v.Update(key("down"))
		v = v2.(ModelView)
	}
	return v
}

// 概览要显示格式、大小、张量数、参数量 —— 用户最先要看的四项。
func TestModelView_概览(t *testing.T) {
	v := loadModelView(fakeModel())
	out := v.View(110, 30)
	for _, want := range []string{"GGUF", "v3", "1.80 GiB", "2"} {
		if !strings.Contains(out, want) {
			t.Errorf("概览里缺少 %q:\n%s", want, out)
		}
	}
}

// **file_type 旁边要把码翻译出来**，而且要能看出可跳转 —— spec §8.2 的核心。
func TestModelView_元数据里file_type有关联(t *testing.T) {
	v := loadModelView(fakeModel())
	v = gotoSection(v, sectionMetadata)
	out := v.View(140, 40)

	if !strings.Contains(out, "general.file_type") {
		t.Fatalf("元数据列表里没有 file_type:\n%s", out)
	}
	if !strings.Contains(out, "MOSTLY_Q4_K_M") {
		t.Errorf("file_type 旁边没标出档位名（应当是 MOSTLY_Q4_K_M）:\n%s", out)
	}
	if !strings.Contains(out, "◂") {
		t.Errorf("可跳转的标注没渲染出来:\n%s", out)
	}
}

// 已废弃的 file_type 要标出来 —— 真实文件里会出现（实测 gpt-oss:20b 写的就是 4）。
func TestModelView_废弃的file_type被标出(t *testing.T) {
	m := fakeModel()
	m.Metadata[1] = model.MetaKV{Key: "general.file_type", Value: "4", Raw: uint32(4)}
	v := loadModelView(m)
	v = gotoSection(v, sectionMetadata)
	out := v.View(140, 40)
	if !strings.Contains(out, "已废弃") {
		t.Errorf("file_type=4 是废弃编号，界面没标出来:\n%s", out)
	}
}

// 未收录的键要显示原文 —— 用户看到文件里有这个键，界面上却找不到，
// 会以为工具漏读了。
func TestModelView_未收录的键也显示(t *testing.T) {
	m := fakeModel()
	m.Metadata = append(m.Metadata, model.MetaKV{
		Key: "some.vendor.custom_key", Value: "42", Raw: uint32(42)})
	v := loadModelView(m)
	v = gotoSection(v, sectionMetadata)
	out := v.View(140, 40)
	if !strings.Contains(out, "some.vendor.custom_key") {
		t.Errorf("未收录的键没显示:\n%s", out)
	}
}

// 左栏导航：↓ 换栏目，右栏内容跟着变；到末尾不越界。
func TestModelView_导航(t *testing.T) {
	v := loadModelView(fakeModel())
	first := section(v.cursor)

	v2, _ := v.Update(key("down"))
	v = v2.(ModelView)
	if section(v.cursor) == first {
		t.Error("按 ↓ 之后栏目没变")
	}

	for range 10 {
		v2, _ = v.Update(key("down"))
		v = v2.(ModelView)
	}
	if int(section(v.cursor)) >= len(sections) {
		t.Errorf("导航越界: section=%d, 共 %d 项", section(v.cursor), len(sections))
	}
	if int(section(v.cursor)) != len(sections)-1 {
		t.Errorf("一直按 ↓ 应当停在最后一项，实际 %d", section(v.cursor))
	}
}

// 读失败要显示原因，不能装作没事 —— 也不能显示一个空的界面。
func TestModelView_读失败要显示原因(t *testing.T) {
	v, _ := NewModelView(nil).Update(modelLoadedMsg{err: errors.New("不是已识别的模型格式")})
	out := v.(ModelView).View(80, 20)
	if !strings.Contains(out, "不是已识别的模型格式") {
		t.Errorf("读失败时没显示原因:\n%s", out)
	}
}

// 还没解析完时要给提示，不能是空白 —— 空白会让人以为卡住了。
func TestModelView_载入中(t *testing.T) {
	v := NewModelViewFromPath("/x/whatever.gguf", "")
	out := v.View(80, 20)
	if !strings.Contains(out, "解析") {
		t.Errorf("载入中没有提示:\n%s", out)
	}
}

// 标题要能看出是哪个模型 —— 用户深处详情时得知道自己在看哪个。
func TestModelView_标题(t *testing.T) {
	v := loadModelView(fakeModel())
	title := v.Title()
	for _, want := range []string{"qwen2.5-3b.gguf", "GGUF"} {
		if !strings.Contains(title, want) {
			t.Errorf("标题里缺少 %q: %q", want, title)
		}
	}
}

// 帮助栏只列**真的支持**的键 —— 这条规矩是两个方向。
//
// 反例原先写的是 `?`（那时根视图里没有那条全局分支，速查表只能从
// "速查表"栏目按 Enter 进）。`?` 与 `a` 接上之后它们从"不许列"
// 变成"必须列" —— 漏了就是用户不知道有这条路，而 `a` 没有任何
// 别的入口。
//
// 反方向（列了按不动）由两处守着：`TestModelView_元数据帮助栏与实际一致`
// （概览栏不许列 Enter / Tab）与 `TestModelView_空模型帮助栏不列扫描键`。
func TestModelView_帮助栏只列支持的键(t *testing.T) {
	v := loadModelView(fakeModel())
	help := strings.Join(v.Help(), " ")
	for _, want := range []string{keyUp, keyDown, keyEsc, keyQuit,
		keyHelp + " 速查表", keyScanAll + " 扫描全部"} {
		if !strings.Contains(help, want) {
			t.Errorf("帮助栏少了 %q: %q", want, help)
		}
	}
}

// 告警要显示出来（未知类型、解析降级），而且要有上限 ——
// 144 条告警会把界面顶穿（实测 gpt-oss:20b 修 MXFP4 之前就是这样）。
func TestModelView_告警有显示且有上限(t *testing.T) {
	m := fakeModel()
	for range 50 {
		m.Warnings = append(m.Warnings, "张量 x 使用未知的 GGML 类型码 39")
	}
	v := loadModelView(m)
	out := v.View(110, 40)
	if !strings.Contains(out, "告警") {
		t.Errorf("有告警却没显示:\n%s", out)
	}
	if n := strings.Count(out, "使用未知的 GGML 类型码"); n > 10 {
		t.Errorf("告警显示了 %d 条 —— 会把界面顶穿，应当有个上限", n)
	}
	if !strings.Contains(out, "还有") {
		t.Errorf("告警被截断时要说「还有 N 条」:\n%s", out)
	}
}

// 标题栏要显示**模型库给的名字**，不是 blobs 下那一长串 sha256。
//
// 实测：不传名字时标题是
// "sha256-7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df · GGUF v3 · …"
// —— 71 个字符的哈希，用户看不出这是哪个模型。
func TestModelView_标题用模型名而不是哈希(t *testing.T) {
	hash := "sha256-" + strings.Repeat("ab", 32)
	v := NewModelViewFromPath("/blobs/"+hash, "gemma4:26b")
	title := v.Title()
	if !strings.Contains(title, "gemma4:26b") {
		t.Errorf("标题里没有模型名: %q", title)
	}
	if strings.Contains(title, hash) {
		t.Errorf("标题里显示了哈希: %q", title)
	}
	// 载入中时也要显示名字 —— 用户刚点进来就知道自己在等哪个
	if !strings.Contains(NewModelViewFromPath("/x", "qwen2.5:3b").Title(), "qwen2.5:3b") {
		t.Error("载入中的标题没有模型名")
	}

	// **解析完成之后**也要用名字。这条不能省：上面两条走的是
	// "m 还是 nil"的早退分支，碰不到 displayName() ——
	// 变异验证时确认过，把 displayName 改成永远退回文件名，上面两条照样绿。
	m := fakeModel()
	m.Path = "/blobs/" + hash
	loaded := ModelView{m: m, name: "gemma4:26b"}
	tl := loaded.Title()
	if !strings.Contains(tl, "gemma4:26b") {
		t.Errorf("解析完成后标题没用模型名: %q", tl)
	}
	if strings.Contains(tl, hash) {
		t.Errorf("解析完成后标题显示了哈希: %q", tl)
	}
}

// 没有名字时退回文件名（通用目录的条目就有名字，但手工构造的可能没有）。
func TestModelView_没有名字时退回文件名(t *testing.T) {
	m := fakeModel()
	v := loadModelView(m)
	if !strings.Contains(v.Title(), "qwen2.5-3b.gguf") {
		t.Errorf("没有名字时该退回文件名: %q", v.Title())
	}
}

// 元数据的**通用可跳转标记**（不是 file_type 那条）必须自己被测到。
//
// 原先唯一的断言是 Contains(out, "◂")，而那个 ◂ 由 file_type 那条分支
// 自己就产出了 —— **断言的字符串来自另一个被测对象**。
// 变异验证确认过：只删通用标记那条分支，测试全绿。
func TestModelView_通用关联标记(t *testing.T) {
	v := loadModelView(fakeModel())
	v = gotoSection(v, sectionMetadata)
	out := v.View(140, 40)

	// qwen2.block_count 在速查表里是按后缀命中的条目，它该有 ◂
	var line string
	for _, l := range strings.Split(out, "\n") {
		if strings.Contains(l, "qwen2.block_count") {
			line = l
		}
	}
	if line == "" {
		t.Fatalf("元数据里没有 block_count 这一行:\n%s", out)
	}
	if !strings.Contains(line, "◂") {
		t.Errorf("未收录键的通用关联标记没渲染出来: %q", line)
	}

	// 反面对照：**查不到的键不该有 ◂** —— 否则记号就成了装饰
	m := fakeModel()
	m.Metadata = append(m.Metadata, model.MetaKV{
		Key: "vendor.no_such_key_anywhere", Value: "1", Raw: uint32(1)})
	v2 := gotoSection(loadModelView(m), sectionMetadata)
	for _, l := range strings.Split(v2.View(140, 40), "\n") {
		if strings.Contains(l, "vendor.no_such_key_anywhere") && strings.Contains(l, "◂") {
			t.Errorf("查不到的键也挂了可跳转记号: %q", l)
		}
	}
}

// **一个键最多一个记号**：两个一模一样的 ◂ 指向两个不同的条目，
// 按 Enter 跳转时"跳哪一个"就没有答案。
func TestModelView_一个键只有一个记号(t *testing.T) {
	v := gotoSection(loadModelView(fakeModel()), sectionMetadata)
	for _, l := range strings.Split(v.View(140, 40), "\n") {
		if strings.Contains(l, "general.file_type") {
			if n := strings.Count(l, "◂"); n > 1 {
				t.Errorf("file_type 那一行有 %d 个跳转记号: %q", n, l)
			}
		}
	}
}

// **上游猜出来的档位要标出来。**
//
// general.file_type = 1024|15 表示"这个值是上游猜的，不是文件里写死的" ——
// 它可能与该文件实际的张量类型不符，而用户会拿它当事实。
// 实测本机 5 个模型都没有这个标志位，但 ref 包专门为此留了
// FileTypeGuessed，TUI 一开始漏掉了它。
func TestModelView_上游猜的档位要标出(t *testing.T) {
	m := fakeModel()
	m.Metadata[1] = model.MetaKV{
		Key: "general.file_type", Value: "1039", Raw: uint32(1024 | 15)}
	v := gotoSection(loadModelView(m), sectionMetadata)
	out := v.View(160, 40)

	var line string
	for _, l := range strings.Split(out, "\n") {
		if strings.Contains(l, "general.file_type") {
			line = l
		}
	}
	if !strings.Contains(line, "猜") {
		t.Errorf("上游猜出来的档位没有标出来 —— 用户会当成文件里写的事实: %q", line)
	}
	// 反面对照：文件里写死的档位不该出现这句话
	v2 := gotoSection(loadModelView(fakeModel()), sectionMetadata)
	for _, l := range strings.Split(v2.View(160, 40), "\n") {
		if strings.Contains(l, "general.file_type") && strings.Contains(l, "猜") {
			t.Errorf("正常档位被标成了「猜的」: %q", l)
		}
	}
}

// **左栏要补成列，右栏各行从同一列开始。**
//
// 原先 nav 不补齐，各栏目名字长短不一（"概览" vs "元数据 (52)"），
// 真终端里右栏每行的起始列都在跳（实测 6/13/11/9/7 列）。
func TestModelView_右栏对齐(t *testing.T) {
	v := loadModelView(fakeModel())
	out := v.View(120, 30)

	// 概览那一栏的右栏从固定列开始；换一栏之后仍是同一列
	colOf := func(s string, marker string) int {
		for _, l := range strings.Split(s, "\n") {
			if i := strings.Index(l, marker); i >= 0 {
				return displayWidth(l[:i])
			}
		}
		return -1
	}
	overviewCol := colOf(out, "文件")
	if overviewCol < 0 {
		t.Fatalf("概览栏没有「文件」那一行:\n%s", out)
	}

	meta := gotoSection(v, sectionMetadata)
	if got := colOf(meta.View(120, 30), "general.file_type"); got != overviewCol {
		t.Errorf("换栏之后右栏起始列变了：概览 %d 列、元数据 %d 列", overviewCol, got)
	}
}

// 元数据一屏放不下时要**明说显示了哪一段**，不能静默只显示前 N 条。
//
// Task 9 把这句话从"还有 N 条没显示（滚动在 ④b-2 里做）"改成了范围 ——
// ④b-1 那一版没有内容光标，滚动也还没做；现在两样都有了，
// 报条数就不再够用（用户要知道的是"现在屏幕上是第几条到第几条"）。
// 措辞的精确性由 `TestModelView_元数据提示的范围与实际显示一致` 盯着。
func TestModelView_元数据超屏要说明(t *testing.T) {
	m := fakeModel()
	for i := range 60 {
		m.Metadata = append(m.Metadata, model.MetaKV{
			Key: fmt.Sprintf("vendor.key_%02d", i), Value: "v", Raw: uint32(i)})
	}
	v := gotoSection(loadModelView(m), sectionMetadata)
	out := v.View(120, 24)

	if !strings.Contains(out, "显示第") || !strings.Contains(out, "共 63 条") {
		t.Errorf("元数据 %d 条、屏幕 24 行，没说明显示了哪一段:\n%s", len(m.Metadata), out)
	}
	// 反面对照：放得下时不该出现这句话
	small := gotoSection(loadModelView(fakeModel()), sectionMetadata)
	if strings.Contains(small.View(120, 40), "显示第") {
		t.Error("3 条元数据、40 行的屏幕，不该说「显示第 N–M 条」")
	}
}

// **左栏用完之后的行，右栏仍要从同一列开始。**
//
// 实测：概览有 8 行内容、左栏只有 5 行，超出那几行（"总参数""张量占用"）
// 从第 0 列开始 —— 因为拼的时候左边什么都没写。
func TestModelView_左栏用完后右栏仍对齐(t *testing.T) {
	v := loadModelView(fakeModel())
	out := v.View(120, 30)

	col := func(marker string) int {
		for _, l := range strings.Split(out, "\n") {
			if i := strings.Index(l, marker); i >= 0 {
				return displayWidth(l[:i])
			}
		}
		return -1
	}
	first := col("文件")
	last := col("张量占用")
	if first < 0 || last < 0 {
		t.Fatalf("概览里缺少行（文件=%d 张量占用=%d）:\n%s", first, last, out)
	}
	if first != last {
		t.Errorf("右栏第 1 行从 %d 列开始、最后一行从 %d 列开始 —— 超出左栏的行没对齐",
			first, last)
	}
}

// **整个视图渲染出来必须恰好占满给定的高度**，且不超宽。
//
// 这条从根 Model 走一遍：各层自己算好的行数要是与根视图的理解不一致，
// 用户看到的就是被多截一刀、或者底部浮空。
// 实测踩过一次：joinHorizontal 只 TrimRight 了左栏，右栏末尾那个换行
// 让结果多出一行，于是元数据自己算好的"还有 54 条"被根视图的
// "还有 3 行"顶掉了 —— 两个提示都说得通，但用户看到的是错的那个。
func TestModelView_渲染恰好占满高度(t *testing.T) {
	m := fakeModel()
	for range 60 {
		m.Metadata = append(m.Metadata, model.MetaKV{
			Key: "gemma4.attention.sliding_window_pattern_x", Value: "v", Raw: true})
	}
	for _, sec := range []section{sectionOverview, sectionMetadata, sectionTensors} {
		v := gotoSection(loadModelView(m), sec)
		for _, h := range []int{8, 12, 20, 40} {
			// **直接问视图要输出**，不经根视图。
			//
			// 经根视图是测不出问题的：padTo 永远把结果补齐或截到恰好 h 行，
			// 所以"渲染出 h 行"这条断言恒真（变异验证确认过 ——
			// 把右栏末尾的换行留回去、把元数据的裁剪关掉，它照样绿）。
			//
			// 真正的不变式是：**各层交给根视图的原始输出，
			// 按 padTo 的计数方式（简单按 "\n" 切）不能超过高度** ——
			// 超了就会被截，而截掉的是真实内容、补上的是"还有 N 行没显示"。
			raw := v.View(100, h)
			if n := len(strings.Split(raw, "\n")); n > h {
				t.Errorf("栏目 %d、高度 %d：视图给了 %d 行，超出会被根视图截掉真实内容",
					sec, h, n)
			}
			for i, l := range strings.Split(raw, "\n") {
				if w := displayWidth(l); w > 100 {
					t.Errorf("栏目 %d、高度 %d：第 %d 行宽 %d: %q", sec, h, i, w, l)
				}
			}

			// 经根视图仍然要恰好占满（这是 padTo 的性质，顺带验一下）
			root, _ := New(v).Update(tea.WindowSizeMsg{Width: 100, Height: h})
			if out := root.(Model).View(); len(strings.Split(out, "\n")) != h {
				t.Errorf("栏目 %d、高度 %d：根视图渲染出 %d 行",
					sec, h, len(strings.Split(out, "\n")))
			}
		}
	}
}

// **解析还没回来时 Enter 什么也不做** —— 不能推入子视图。
//
// 推入的子视图拿到的是 nil 的 *model.Model，而它们全都无保护地取
// m.Tensors / m.Metadata —— 那是一条 panic 路径（实测复现过）。
// 它只在"进模型后立刻按 Enter"这个几秒的窗口里可达，手测几乎撞不到，
// 所以只有这条测试看着它。
func TestModelView_解析未完成时Enter不推子视图(t *testing.T) {
	v := gotoSection(NewModelViewFromPath("/x/m.gguf", "m"), sectionTensors)
	if _, cmd := v.Update(key("enter")); cmd != nil {
		if msg, ok := cmd().(pushMsg); ok {
			t.Fatalf("解析未完成时 Enter 推入了 %T —— 它拿到的是 nil 的 *model.Model，"+
				"View 里取 m.Tensors 就是 panic", msg.v)
		}
		t.Fatalf("解析未完成时 Enter 返回了命令 %T，want 什么也不做", cmd())
	}

	// **正对照**：解析完成后同一个键要能推入 —— 少了它，这条测试在
	// "Enter 整个被删掉"时也是绿的（那同样是 bug，只是另一种）。
	loaded := gotoSection(loadModelView(fakeModel()), sectionTensors)
	_, cmd := loaded.Update(key("enter"))
	if cmd == nil {
		t.Fatal("解析完成后 Enter 没有推入子视图")
	}
	if _, ok := cmd().(pushMsg); !ok {
		t.Fatalf("解析完成后 Enter 返回的不是 pushMsg: %T", cmd())
	}
}

// **进速查表时也要把模型带过去**（`NewRefView(v.m)` 而不是 `nil`）。
//
// 速查表是"模型 → 条目"这条链的上半段：它带的 m 会一直传到条目页，
// 而条目页的"在本模型中"读的就是那个 m。传 nil 的话从那以后的每一条
// 都少一节，且不报错（`occurrences()` 对 nil 的契约就是返回 nil）——
// 实测：把 `v.m` 改成 `nil`，原有测试一条都不红。
//
// 与 `TestRefView_Enter进入条目` 是同一条链上的两段，各看着一个传参点。
func TestModelView_进速查表时把模型带过去(t *testing.T) {
	v := gotoSection(loadModelView(fakeModel()), sectionRef)
	_, cmd := v.Update(key("enter"))
	if cmd == nil {
		t.Fatal("速查表栏 Enter 没有推入子视图")
	}
	msg, ok := cmd().(pushMsg)
	if !ok {
		t.Fatalf("返回的不是 pushMsg: %T", cmd())
	}
	rv, ok := msg.v.(RefView)
	if !ok {
		t.Fatalf("推入的不是 RefView: %T", msg.v)
	}
	if rv.m == nil {
		t.Fatal("推入的速查表没有带模型 —— 从它进去的条目页" +
			"「在本模型中」会是空的")
	}
	if rv.m != v.m {
		t.Errorf("带过去的不是当前这个模型：%p, want %p", rv.m, v.m)
	}
}

// **含换行的值必须被压成一行**：一条元数据撑成几十行的话，这一栏的
// 行数核账（listCap / window / "显示第 N–M 条"）全乱 —— 实测 qwen2.5 的
// tokenizer.chat_template 有几十个换行，80×24 下屏幕上出现的是根视图
// padTo 那句"…还有 54 行没显示"，范围提示整句被顶掉。
//
// **不能静默丢内容**：所以断言的是"换行被换成了看得见的记号"，
// 而不是"某一行里没有换行"（后者把值砍到第一行也能过）。
func TestModelView_含换行的值只占一行(t *testing.T) {
	// 三个换行的最小形态：那个条目只占一行
	three := fakeModel()
	three.Metadata = append(three.Metadata, model.MetaKV{
		Key: "tokenizer.chat_template", Value: "第一行\n第二行\n第三行\n第四行"})
	out := gotoSection(loadModelView(three), sectionMetadata).View(120, 24)
	if !strings.Contains(out, "第一行␊第二行␊第三行␊第四行") {
		t.Errorf("值里的换行没换成可见记号 —— 要么被静默丢掉，"+
			"要么把一条撑成了四行:\n%s", out)
	}

	// 几十个换行（真实 chat_template 的形态）：这一栏自己算好的行数
	// 不能被撑爆 —— 撑爆时 padTo 会把它那句范围提示换成"…还有 N 行没显示"
	many := fakeModel()
	many.Metadata = append(many.Metadata, model.MetaKV{
		Key: "tokenizer.chat_template", Value: strings.Repeat("行\n", 30) + "尾"})
	out = gotoSection(loadModelView(many), sectionMetadata).View(120, 24)
	if !strings.Contains(out, strings.Repeat("行␊", 30)+"尾") {
		t.Errorf("几十个换行没有被压成一行:\n%s", out)
	}
	if n := len(strings.Split(out, "\n")); n > 24 {
		t.Errorf("渲染出 %d 行，超出 24 行的高度 —— "+
			"padTo 会把这一栏自己的提示顶掉", n)
	}
}

// ===== 「扫描全部」（`a`）=====

// 按 `a` 要进入扫描态、返回"扫第 0 张"的命令，屏幕上进度行真的出现。
func TestModelView_a开始扫描(t *testing.T) {
	v := loadModelView(fakeModel())
	v.scan = fakeScan(nil)

	v2, cmd := v.Update(key(keyScanAll))
	v = v2.(ModelView)
	if cmd == nil {
		t.Fatal("按 a 没有返回扫描命令 —— 一张都不会扫")
	}
	if !v.scanning {
		t.Fatal("按 a 之后不在扫描态")
	}

	out := v.View(120, 20)
	if !strings.Contains(out, "正在扫描张量… 0/2") {
		t.Errorf("进度行没出现（或数字不对）:\n%s", out)
	}
	// **"（a 取消）"这半句也要断言**：它是"再按一次能取消"唯一的提示，
	// 而删掉它不会有任何测试红（变异验证实测过）—— 用户敢不敢按下去
	// 就靠这一句。帮助栏里那半句由下一个断言管，两处各自断言。
	if !strings.Contains(out, "（"+keyScanAll+" 取消）") {
		t.Errorf("进度行没告诉用户怎么取消:\n%s", out)
	}
	if help := strings.Join(v.Help(), " "); !strings.Contains(help, keyScanAll+" 取消") {
		t.Errorf("扫描中帮助栏该说 %q: %q", keyScanAll+" 取消", help)
	}
}

// **进度行在每个栏目都要看得见**：扫描是后台跑的，换个栏目就看不见的话
// 用户不知道它还在不在跑。它占的是内容区第一行（跨左右两栏）。
//
// 顺带钉住"交给根视图的原始输出不超过给定高度"（超出的行会被 padTo
// 砍掉，砍的是各栏目自己算好的最后一行）。**"高度里扣掉了进度行那一行"
// 这一条它守不住** —— 元数据那一栏自己会按给到的高度收着渲染，
// 扣不扣都不超行数（变异验证实测：把 -1 去掉它照样绿）。
// 那一条由 TestModelView_进度行从内容区里扣 专门守。
func TestModelView_a各栏目都看得见进度行(t *testing.T) {
	v := loadModelView(fakeModel())
	v.scan = fakeScan(nil)
	v2, _ := v.Update(key(keyScanAll))
	v = v2.(ModelView)

	for _, s := range []section{sectionOverview, sectionMetadata, sectionTensors, sectionQuantDist, sectionRef} {
		out := gotoSection(v, s).View(120, 20)
		if !strings.Contains(out, "正在扫描张量…") {
			t.Errorf("第 %d 栏看不到进度行:\n%s", s, out)
		}
		if n := len(strings.Split(out, "\n")); n > 20 {
			t.Errorf("第 %d 栏渲染出 %d 行，超出 20 —— "+
				"进度行那一行没有从各栏目的行数里扣掉", s, n)
		}
	}
}

// 收到一条结果：进度 +1、结果合并进**那一张**张量、再调度下一条。
//
// 三件事分开断言：只测"进度变了"的话，把结果合并到别的张量上、
// 或者压根不合并（只在进度上加一），测试都会绿。
func TestModelView_a逐张推进(t *testing.T) {
	v := loadModelView(fakeModel())
	v.scan = fakeScan(nil)
	v2, cmd := v.Update(key(keyScanAll))
	v = v2.(ModelView)

	msg := cmd()
	if v.m.Tensors[0].Stats != nil {
		t.Fatal("命令还没回来，共享的张量上已经有统计了 —— 合并没在主线程做")
	}

	v3, next := v.Update(msg)
	v = v3.(ModelView)
	if v.scanDone != 1 {
		t.Errorf("进度 = %d, want 1", v.scanDone)
	}
	// **三个字段逐个断言**：合并是一行三个赋值，只查其中一个的话，
	// 漏掉另两个（比如 Quant）不会有任何东西红 —— 而 Quant 丢掉的表现是
	// 详情页「量化诊断」整节消失、且每次点开都要重扫块头。
	if v.m.Tensors[0].Stats == nil {
		t.Error("统计没合并进那一张张量")
	}
	if v.m.Tensors[0].Quant == nil {
		t.Error("量化诊断没合并进那一张张量")
	}
	if v.m.Tensors[0].QuantSims == nil {
		t.Error("量化模拟没合并进那一张张量")
	}
	if v.m.Tensors[1].Stats != nil || v.m.Tensors[1].Quant != nil {
		t.Error("结果合并到了另一张张量上")
	}
	if next == nil {
		t.Fatal("扫完一张之后没有调度下一条 —— 链断了")
	}
}

// **扫到最后一张必须停**：再返回命令就是一条永远走不完的空转链
// （那时 scanDone 已经等于总数，下一条命令的位次是越界的）。
//
// 这里只断言"没有返回命令"，**不去执行它**：执行了就是越界 panic，
// 而这条测试要区分的是"停"与"不停"，不是"崩不崩"。
func TestModelView_a扫完就停(t *testing.T) {
	v := loadModelView(fakeModel())
	v.scan = fakeScan(nil)
	// **前提要显式断言**：模型里一张张量都没有时，"扫完就停"是**空集通过**
	//（第一次按 a 就该返回 nil，循环一次都不进）—— 夹具哪天变成空的，
	// 这条测试会绿着什么都不验。
	if len(v.m.Tensors) == 0 {
		t.Fatal("前提不成立：夹具里没有张量")
	}
	v2, cmd := v.Update(key(keyScanAll))
	v = v2.(ModelView)

	for i := range len(v.m.Tensors) {
		if cmd == nil {
			t.Fatalf("第 %d 张还没扫，链就断了", i)
		}
		next, c := v.Update(cmd())
		v, cmd = next.(ModelView), c
	}
	if cmd != nil {
		t.Error("扫完最后一张仍然返回了命令 —— 链会一直空转下去")
	}
	if v.scanning {
		t.Error("扫完之后仍在扫描态")
	}
	if v.scanDone != len(v.m.Tensors) {
		t.Errorf("收尾时进度 = %d, want %d", v.scanDone, len(v.m.Tensors))
	}

	out := v.View(120, 20)
	if !strings.Contains(out, "扫描完成 2/2") {
		t.Errorf("扫完之后屏幕上没有结果（分不清扫完了还是链断了）:\n%s", out)
	}
	if strings.Contains(out, "正在扫描") {
		t.Errorf("扫完之后还挂着扫描中那一行:\n%s", out)
	}
}

// **取消之后不再调度下一条**，在途那一条回来也不合并；
// 但**已经扫出来的结果要留着** —— 回滚等于把用户等的时间丢掉。
func TestModelView_a取消(t *testing.T) {
	v := loadModelView(fakeModel())
	v.scan = fakeScan(nil)
	v2, cmd0 := v.Update(key(keyScanAll))
	v = v2.(ModelView)

	// 先收下第 0 张：链此时在等第 1 张
	v3, cmd1 := v.Update(cmd0())
	v = v3.(ModelView)
	if cmd1 == nil {
		t.Fatal("第 1 张没有被调度 —— 前提不成立")
	}

	// 再按 a：取消
	v4, cancelCmd := v.Update(key(keyScanAll))
	v = v4.(ModelView)
	if cancelCmd != nil {
		t.Error("取消还返回了命令 —— 链会继续扫下去")
	}
	if v.scanning {
		t.Error("取消之后还在扫描态")
	}

	// 在途那一条回来了：不该合并、不该再调度
	done := v.scanDone
	v5, after := v.Update(cmd1())
	v = v5.(ModelView)
	if after != nil {
		t.Error("取消之后又调度了下一条")
	}
	if v.scanDone != done {
		t.Errorf("取消之后仍合并了在途的结果：进度 %d → %d", done, v.scanDone)
	}
	if out := v.View(120, 20); strings.Contains(out, "正在扫描") {
		t.Errorf("取消之后进度行还挂着:\n%s", out)
	}
	if v.m.Tensors[0].Stats == nil {
		t.Error("取消把已经扫出来的结果回滚了")
	}
	if v.m.Tensors[1].Stats != nil {
		t.Error("取消之后在途那一条的结果还是被合并了")
	}
}

// **单张失败不中断整体**（与 analyze.Analyze 的同名契约一致）：
// 记一个失败数、继续扫下一张；失败数要能看出来。
func TestModelView_a单张失败不中断(t *testing.T) {
	m := fakeModel()
	v := loadModelView(m)
	v.scan = fakeScan(map[string]bool{m.Tensors[0].Name: true})

	v2, cmd := v.Update(key(keyScanAll))
	v = v2.(ModelView)

	v3, cmd := v.Update(cmd())
	v = v3.(ModelView)
	if cmd == nil {
		t.Fatal("第一张失败就断了链 —— 一张坏张量不该让整个模型扫不下去")
	}
	if v.scanDone != 1 || len(v.failed) != 1 {
		t.Errorf("进度/失败 = %d/%d, want 1/1", v.scanDone, len(v.failed))
	}
	// **失败的那一张什么都不合并**：副本里那半份结果不落地 ——
	// 落地的话详情页会显示"只有统计、没有量化诊断"的半份数据
	if v.m.Tensors[0].Stats != nil || v.m.Tensors[0].Quant != nil {
		t.Error("失败的那一张还是把副本里的半份结果合并进去了")
	}
	if out := v.View(120, 20); !strings.Contains(out, "失败 1") {
		t.Errorf("失败数没显示在进度行上:\n%s", out)
	}

	// 失败的那一张没有结果，剩下的照常
	v4, cmd := v.Update(cmd())
	v = v4.(ModelView)
	if cmd != nil {
		t.Fatal("第二张扫完还在调度")
	}
	if len(v.failed) != 1 {
		t.Errorf("失败计数 = %d, want 1", len(v.failed))
	}
	if v.m.Tensors[1].Stats == nil {
		t.Error("前一张失败之后，后面那张没被扫")
	}
	if out := v.View(120, 20); !strings.Contains(out, "扫描完成 2/2") ||
		!strings.Contains(out, "失败 1") {
		t.Errorf("扫完之后看不出有失败:\n%s", out)
	}
}

// **副本必须在构造命令时（主线程）就拷好**，不能挪进闭包。
//
// 闭包由 bubbletea 在另一个 goroutine 里执行，而这条链跑着的时候
// 用户完全可能按 Enter 进列表、点开同一张量的详情页 —— 那边的
// TensorView 会把结果就地写进**同一个** *model.Tensor。在 goroutine 里
// 拷就是在读一个正在被写的字段（-race 抓得到，真终端里的表现是
// 偶发的半截统计）。
//
// 判据是"命令执行时看到的值是**造命令那一刻**的"：造好命令之后
// 主线程改共享张量，扫描函数不该看见那次改动。
func TestModelView_a副本在造命令时就拷好(t *testing.T) {
	v := loadModelView(fakeModel())
	var sawStats *model.Stats
	v.scan = func(_ context.Context, _ *model.Model, tn *model.Tensor) error {
		sawStats = tn.Stats
		return nil
	}

	_, cmd := v.Update(key(keyScanAll))
	// 模拟详情页在这一刻合并了结果（同一个 *model.Tensor）
	v.m.Tensors[0].Stats = &model.Stats{Count: 7}
	cmd()

	if sawStats != nil {
		t.Error("扫描读到了造命令之后才写进去的统计 —— " +
			"副本是在 goroutine 里拷的，与详情页的扫描构成数据竞争")
	}
}

// **别的模型的结果不许合并进来**：用户退回模型库、换一个模型再进来时，
// 上一轮在途的消息会落在新的 ModelView 上 —— 两张张量的位次可能相同。
//
// 丢弃之后**新的链照常走**：那条消息是多余的（这一轮自己的还在途），
// 不是这一轮在等的那一条 —— 但也不能让它把进度推着走。
func TestModelView_a拒绝别的模型的结果(t *testing.T) {
	old := loadModelView(fakeModel())
	old.scan = fakeScan(nil)
	_, oldCmd := old.Update(key(keyScanAll))

	fresh := loadModelView(fakeModelWithTensors(2))
	fresh.scan = fakeScan(nil)
	f2, freshCmd := fresh.Update(key(keyScanAll))
	fresh = f2.(ModelView)
	if freshCmd == nil {
		t.Fatal("新模型的扫描没有起步")
	}

	f3, cmd := fresh.Update(oldCmd())
	fresh = f3.(ModelView)
	if fresh.scanDone != 0 || fresh.m.Tensors[0].Stats != nil {
		t.Error("把上一个模型的结果合并进了新模型")
	}
	if cmd != nil {
		t.Error("丢弃残影时返回了命令 —— 这一轮的链会走出来两条")
	}

	// 这一轮自己的那条回来：链照常前进
	f4, cmd := fresh.Update(freshCmd())
	fresh = f4.(ModelView)
	if fresh.scanDone != 1 || cmd == nil {
		t.Errorf("残影把链打断了：进度 %d，有没有下一条 = %v", fresh.scanDone, cmd != nil)
	}
}

// 取消后重启：上一轮在途的那条回来时**位次对不上**（新一轮又从 0 开始），
// 必须整个丢弃 —— 不丢的话它的结果会被算到新一轮头上。
func TestModelView_a重启后丢弃上一轮的残影(t *testing.T) {
	v := loadModelView(fakeModel())
	v.scan = fakeScan(nil)
	v2, cmd0 := v.Update(key(keyScanAll))
	v = v2.(ModelView)

	v3, staleCmd := v.Update(cmd0()) // 第 0 张收下，第 1 张在途
	v = v3.(ModelView)

	v4, _ := v.Update(key(keyScanAll)) // 取消
	v = v4.(ModelView)
	v5, restartCmd := v.Update(key(keyScanAll)) // 重启
	v = v5.(ModelView)
	if !v.scanning || v.scanDone != 0 || restartCmd == nil {
		t.Fatalf("重启没有从 0 开始：scanning=%v scanDone=%d", v.scanning, v.scanDone)
	}

	v6, cmd := v.Update(staleCmd()) // 上一轮第 1 张的残影
	v = v6.(ModelView)
	if v.scanDone != 0 || cmd != nil {
		t.Errorf("上一轮的残影被当成了这一轮的结果：进度 = %d", v.scanDone)
	}
	if v.m.Tensors[1].Stats != nil {
		t.Error("残影的结果被合并了")
	}
}

// 模型里一张张量都没有：`a` 什么也不做，帮助栏也就不列它 ——
// 列了就是骗用户按（keys.go 的规矩，两个方向都要管）。
func TestModelView_空模型帮助栏不列扫描键(t *testing.T) {
	m := fakeModel()
	m.Tensors = nil
	v := loadModelView(m)

	if help := strings.Join(v.Help(), " "); strings.Contains(help, "扫描") {
		t.Errorf("没有张量可扫，帮助栏却列了扫描键: %q", help)
	}
	v2, cmd := v.Update(key(keyScanAll))
	v = v2.(ModelView)
	if cmd != nil || v.scanning {
		t.Error("空模型按 a 动了 —— 帮助栏与 Update 读的不是同一个判据")
	}
}

// 进度行占的那一行**是从内容区里扣掉的**，不是白送的：同一个高度下，
// 扫描中的元数据列表比不扫描时少显示一条。
//
// 不扣的话这一层交给根视图的原始输出会比给定高度多出一行，而 padTo
// 兜底时砍掉的是各栏目自己算好的最后一行 —— 元数据那一栏正是它的
// 范围提示（用户据此知道自己在第几条、后面还有多少），换上的是一句
// 措辞更差、信息更少的"…还有 N 行没显示"。
//
// 断言用**屏幕上的范围**（"显示第 1–N 条"）而不是行数：行数那一条
// 对"少扣一行"不敏感 —— 元数据栏会按给到的高度收着渲染，两种写法
// 都不超行数，差别全在"最后显示到第几条"。
func TestModelView_进度行从内容区里扣(t *testing.T) {
	m := metaModel(60) // 够长：两种渲染都会打"显示第 1–N 条"
	plain := gotoSection(loadModelView(m), sectionMetadata)
	scan := plain
	scan.scan = fakeScan(nil)
	s2, _ := scan.Update(key(keyScanAll))
	scan = s2.(ModelView)

	outPlain := plain.View(120, 20)
	outScan := scan.View(120, 20)
	if !strings.Contains(outScan, "正在扫描张量…") {
		t.Fatalf("前提不成立：扫描态没渲染出进度行:\n%s", outScan)
	}
	plainRows, scanRows := shownLast(t, outPlain), shownLast(t, outScan)
	if scanRows != plainRows-1 {
		t.Errorf("进度行占了屏幕一行，列表却仍显示到第 %d 条（不扫描时是第 %d 条）—— "+
			"那一行没有从内容区里扣掉", scanRows, plainRows)
	}
}

// shownLast 从"…显示第 1–N 条，共 M 条"里取出 N。
func shownLast(t *testing.T, out string) int {
	t.Helper()
	const prefix = "显示第 1–"
	i := strings.Index(out, prefix)
	if i < 0 {
		t.Fatalf("渲染里没有范围提示 %q:\n%s", prefix, out)
	}
	rest := out[i+len(prefix):]
	j := strings.Index(rest, " 条")
	if j < 0 {
		t.Fatalf("范围提示的格式不对:\n%s", out)
	}
	n, err := strconv.Atoi(rest[:j])
	if err != nil {
		t.Fatalf("范围提示里的条数 %q 不是数字: %v", rest[:j], err)
	}
	return n
}

// **失败的结果不许把已经算好的结果清掉** —— 两个写者（这条链与详情页）
// 之间没有版本号，判不出新旧，所以旧副本只在**成功**时才写回。
//
// 序列（都是真实路径）：批量派发第 0 张（副本是空的）→ 用户点开详情页，
// TensorView 把好结果合并进同一张 tn → 批量那条旧副本带着"读取失败"
// 回来。无条件合并的话，这里会把 tn.Stats 写回 nil —— 详情页上刚显示
// 出来的数字当场消失，而且没有任何地方会报错。
func TestModelView_a失败的结果不清掉详情页的结果(t *testing.T) {
	v := loadModelView(fakeModel())
	v.scan = fakeScan(map[string]bool{v.m.Tensors[0].Name: true})
	v2, cmd := v.Update(key(keyScanAll))
	v = v2.(ModelView)

	// 详情页在这一刻算出了好结果（同一个 *model.Tensor）
	v.m.Tensors[0].Stats = &model.Stats{Count: 42}
	v.m.Tensors[0].Quant = &model.QuantInfo{Scheme: "Q4_K"}

	v3, _ := v.Update(cmd())
	v = v3.(ModelView)
	if v.m.Tensors[0].Stats == nil || v.m.Tensors[0].Stats.Count != 42 {
		t.Error("带失败的旧副本把详情页已经算好的统计清掉了")
	}
	if v.m.Tensors[0].Quant == nil {
		t.Error("带失败的旧副本把详情页已经算好的量化诊断清掉了")
	}
	if len(v.failed) != 1 {
		t.Errorf("失败计数 = %d, want 1", len(v.failed))
	}
}

// 存储判据必须用 TiedGroups，不能拿"两个数字不相等"当判据。
//
// safetensors 的 data_offsets 允许不从 0 开始，此时 StorageBytes
// （数据区跨度）会**大于**张量字节和 —— 打出来就是"去重后仅占 32 B
// （差值 -16 B 是共享的存储）"，自相矛盾。CLI 的 printSummary 早就
// 改用 TiedGroups 并写明了理由，TUI 没跟上。
func TestModelView_跨度为负差时不显示共享存储(t *testing.T) {
	m := &model.Model{
		Path: "x.safetensors", Format: model.FormatSafeTensors,
		Tensors: []*model.Tensor{
			{Name: "a", Dtype: model.DtypeF32, ByteSize: 16, ParamCount: 4},
		},
		StorageBytes: 32, // 跨度 32 > 字节和 16，且没有任何绑定
	}
	out := NewModelView(m).overview()
	if strings.Contains(out, "共享") {
		t.Errorf("没有绑定却说了共享存储:\n%s", out)
	}
	if strings.Contains(out, "-16 B") {
		t.Errorf("打出了负数差值:\n%s", out)
	}
}

// 反面：**真的**有权重绑定时必须说出来 —— 否则用户看到
// "张量占用"与"去重后"两个不一样的数会以为哪边算错了。
func TestModelView_有权重绑定时要说明(t *testing.T) {
	m := &model.Model{
		Path: "x.pt", Format: model.FormatPyTorch,
		Tensors: []*model.Tensor{
			{Name: "embed", Dtype: model.DtypeF32, ByteSize: 16, ParamCount: 4},
			{Name: "head", Dtype: model.DtypeF32, ByteSize: 16, ParamCount: 4},
		},
		StorageBytes: 16,
		TiedGroups:   [][]string{{"embed", "head"}},
	}
	out := NewModelView(m).overview()
	if !strings.Contains(out, "共享") {
		t.Errorf("有权重绑定却没说:\n%s", out)
	}
	// 差值必须是正的：32 - 16 = 16
	if !strings.Contains(out, "16 B") {
		t.Errorf("没打出正确的差值:\n%s", out)
	}
	if strings.Contains(out, "-") && strings.Contains(out, "差值 -") {
		t.Errorf("差值是负的:\n%s", out)
	}
}

// 没注入解析入口时要给一条可显示的错误，**不能 nil 解引用**。
//
// 那只可能是接线漏了（cmd 忘了 WithParse），而一次 nil 解引用 panic
// 会掀掉整个 TUI、终端还留在 alt-screen 里。这条守卫没有测试的话
// 就是一段没人看着的代码。
func TestModelView_未注入解析入口不崩(t *testing.T) {
	v := NewModelViewFromPath("/x/m.gguf", "m") // parse 为 nil
	cmd := v.Init()
	if cmd == nil {
		t.Fatal("有待解析路径时 Init 必须返回命令")
	}
	msg, ok := cmd().(modelLoadedMsg)
	if !ok {
		t.Fatalf("Init 的命令没有产出 modelLoadedMsg")
	}
	if msg.err == nil {
		t.Error("没有注入解析入口却没有报错 —— 会一直停在载入中")
	}
	if !strings.Contains(msg.err.Error(), "解析入口") {
		t.Errorf("错误信息应说清是接线漏了，得到: %v", msg.err)
	}
}

// spec §3 承诺 `r` = 强制重扫（忽略缓存）。模型页按 r 原来毫无反应。
//
// TUI 里没有"命中缓存"这回事（analyze.One 按契约不读写缓存），所以
// 这里的语义是"把已有结果清掉再走一遍扫描链"—— 让用户能强制刷新。
func TestModelView_r键清空结果并重扫(t *testing.T) {
	m := fakeModel()
	// 先造出"已经算过"的状态
	for _, tn := range m.Tensors {
		tn.Stats = &model.Stats{Count: tn.ParamCount}
	}
	v := NewModelView(m)
	v.scan = func(context.Context, *model.Model, *model.Tensor) error { return nil }

	next, _ := v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("r")})
	v = next.(ModelView)

	if !v.scanning {
		t.Error("按 r 后没有进入扫描中")
	}
	for _, tn := range m.Tensors {
		if tn.Stats != nil {
			t.Errorf("%s 的结果没被清掉 —— 那就不是重扫，用户分不出「重算了"+
				"结果相同」与「根本没重算」", tn.Name)
		}
	}
}

// 没有模型（还在解析）时按 r 不该崩，也不该进入扫描中。
func TestModelView_r键在没模型时无动作(t *testing.T) {
	v := NewModelViewFromPath("/x/m.gguf", "m")
	next, _ := v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("r")})
	v = next.(ModelView)
	if v.scanning {
		t.Error("模型还没解析出来就进入了扫描中")
	}
}

var errFake = errors.New("模拟失败")

// 失败名单必须传到张量列表 —— 不传的话标红那一栏永远是空的，
// 而 ModelView 是**唯一**知道哪张扫失败了的视图。
func TestModelView_失败名单传给张量列表(t *testing.T) {
	m := fakeModel()
	v := NewModelView(m)
	v.scan = func(context.Context, *model.Model, *model.Tensor) error { return errFake }

	// 扫第一张，让它失败
	next, _ := v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("a")})
	v = next.(ModelView)
	next, _ = v.Update(batchScannedMsg{m: m, idx: 0, err: errFake})
	v = next.(ModelView)
	if len(v.failed) != 1 {
		t.Fatalf("失败名单 = %v, want 1 个", v.failed)
	}

	// 推到张量列表那一栏并按 Enter
	for range int(sectionTensors) {
		next, _ = v.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("j")})
		v = next.(ModelView)
	}
	next, cmd := v.Update(tea.KeyMsg{Type: tea.KeyEnter})
	v = next.(ModelView)
	if cmd == nil {
		t.Fatal("Enter 应当推出张量列表")
	}
	msg := cmd()
	push, ok := msg.(pushMsg)
	if !ok {
		t.Fatalf("期望 pushMsg，得到 %T", msg)
	}
	tv, ok := push.v.(TensorsView)
	if !ok {
		t.Fatalf("推的不是 TensorsView，是 %T", push.v)
	}
	if len(tv.failed) != len(v.failed) {
		t.Errorf("列表拿到的失败名单 = %v，ModelView 手里的是 %v —— 没传过去",
			tv.failed, v.failed)
	}
}
