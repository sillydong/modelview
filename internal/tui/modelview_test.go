package tui

import (
	"errors"

	"fmt"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/model"
)

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

// 帮助栏不能列还没做的键（张量详情/速查表在 ④b-2）。
func TestModelView_帮助栏只列支持的键(t *testing.T) {
	v := loadModelView(fakeModel())
	help := strings.Join(v.Help(), " ")
	for _, want := range []string{keyUp, keyDown, keyEsc, keyQuit} {
		if !strings.Contains(help, want) {
			t.Errorf("帮助栏少了 %s: %q", want, help)
		}
	}
	// 速查表视图在 ④b-2；现在按 ? 应该给出"还没做"的说明而不是没反应
	if strings.Contains(help, keyHelp) {
		t.Errorf("速查表还没做，帮助栏不该列 %s: %q", keyHelp, help)
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
// ④b-2 接 Enter 跳转时"跳哪一个"没有答案。
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
