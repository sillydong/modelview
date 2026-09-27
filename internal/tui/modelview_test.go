package tui

import (
	"errors"
	"strings"
	"testing"

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
	v.section = sectionMetadata
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
	v.section = sectionMetadata
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
	v.section = sectionMetadata
	out := v.View(140, 40)
	if !strings.Contains(out, "some.vendor.custom_key") {
		t.Errorf("未收录的键没显示:\n%s", out)
	}
}

// 左栏导航：↓ 换栏目，右栏内容跟着变；到末尾不越界。
func TestModelView_导航(t *testing.T) {
	v := loadModelView(fakeModel())
	first := v.section

	v2, _ := v.Update(key("down"))
	v = v2.(ModelView)
	if v.section == first {
		t.Error("按 ↓ 之后栏目没变")
	}

	for range 10 {
		v2, _ = v.Update(key("down"))
		v = v2.(ModelView)
	}
	if int(v.section) >= len(sections) {
		t.Errorf("导航越界: section=%d, 共 %d 项", v.section, len(sections))
	}
	if int(v.section) != len(sections)-1 {
		t.Errorf("一直按 ↓ 应当停在最后一项，实际 %d", v.section)
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
