package tui

import (
	"fmt"
	"strings"
	"testing"

	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/ref"
)

// **先钉住这个跨包的假设**：quant 条目的 ID 是 "quant:" + 名字，
// 而名字必须恰好等于 model.Dtype 的字面值，否则反向定位永远匹配不上。
//
// 它一旦漂移（比如 dtype 改名成 "Q4_K_M"），表现是"本模型中没有"
// 而不是报错 —— 所以必须有东西看着它。
func TestJump_dtype名与速查表ID一致(t *testing.T) {
	// **把字面值本身钉住**：反向定位全靠 `"quant:" + string(dtype)` 匹配，
	// 而这个耦合没有任何编译期的东西看着它
	if got := string(model.DtypeQ4K); got != "Q4_K" {
		t.Errorf("model.DtypeQ4K 的字面值是 %q, want Q4_K —— 反向定位靠它匹配", got)
	}

	// 而且这些名字在速查表里真的查得到
	for _, id := range []string{"quant:Q4_K", "quant:Q6_K"} {
		if _, ok := ref.ByID(id); !ok {
			t.Errorf("速查表里没有 %q —— 那这个类型的张量永远找不到自己的条目", id)
		}
	}
}

// realSegments 从速查表里取 n 个**真实存在**的段名。
//
// **不许在测试里写死 "attn_q" 这种字面量**：那张表改一次名字
// （或者某个段被标成 segmentsNotInCorpus）这些测试就会假失败，
// 而它们想验的是"分段能匹配到张量"这件事本身，不是某张表的内容。
//
// **跳过 "blk" 与 "weight"**：这两个是结构性段，造出来的名字里
// 到处都是它们 —— 用它们当某一组的标记，另一组的计数会被带偏，
// 而失败信息只会说"计数不对"，看不出是段选错了。
// 跳过含 "." 的段同理：拆段是按 "." 切的，段里带点会多切出一段。
//
// **取出来的段还要两两不互为子串**：按段匹配是精确的，
// 但 `NewTensorsViewName` 的过滤是**子串**匹配 ——
// 取到 "attn" 与 "attn_q" 这一对时，按 "attn" 筛会把两组都算上。
func realSegments(t *testing.T, n int) []string {
	t.Helper()
	for _, tb := range ref.Tables() {
		if tb.ID != "tensors" {
			continue
		}
		out := make([]string, 0, n)
		for _, e := range tb.Entries {
			seg := strings.TrimPrefix(e.ID, "tensor:")
			if seg == "blk" || seg == "weight" || strings.Contains(seg, ".") {
				continue
			}
			overlap := false
			for _, prev := range out {
				if strings.Contains(prev, seg) || strings.Contains(seg, prev) {
					overlap = true
					break
				}
			}
			if overlap {
				continue
			}
			out = append(out, seg)
			if len(out) == n {
				return out
			}
		}
		t.Fatalf("tensors 表里可用的段只有 %d 个，取不出 %d 个", len(out), n)
	}
	t.Fatal("速查表里没有 tensors 表")
	return nil
}

// jumpModel 造一个有几类张量的模型，返回模型与用到的两个真实段名。
//
//   - segA 出现在 3 个 Q4_K 张量里
//   - segB 出现在 1 个 Q6_K 张量里
//
// 两个段都**从速查表里现取**（见 realSegments）。
func jumpModel(t *testing.T) (m *model.Model, segA, segB string) {
	t.Helper()
	segs := realSegments(t, 2)
	segA, segB = segs[0], segs[1]

	m = &model.Model{Path: "/x/m.gguf", Format: model.FormatGGUF, Version: "v3"}
	m.Metadata = []model.MetaKV{
		{Key: "general.file_type", Value: "15", Raw: uint32(15)},
		{Key: "custom.unknown", Value: "x"},
	}
	// **名字里只用 segA/segB 两个段（外加 "weight"）**，不带 "blk"、"#N" 那些：
	// 带上它们的话，按 segA 筛出来的个数会被结构段带偏，
	// 下面"3 个"和"1 个"两条断言就都不成立了
	for i := range 3 {
		m.Tensors = append(m.Tensors, &model.Tensor{
			Name: fmt.Sprintf("%s.weight.%d", segA, i), Dtype: model.DtypeQ4K,
			Dims: []int64{64}, ByteSize: 36, ParamCount: 64})
	}
	m.Tensors = append(m.Tensors, &model.Tensor{
		Name: segB + ".weight", Dtype: model.DtypeQ6K,
		Dims: []int64{64}, ByteSize: 52, ParamCount: 64})
	return m, segA, segB
}

func targetsOf(t *testing.T, m *model.Model, id string) []entryTarget {
	t.Helper()
	e, ok := ref.ByID(id)
	if !ok {
		t.Fatalf("速查表里没有 %q", id)
	}
	return NewEntryView(m, e).targets
}

// key 类条目要指出模型里对应的**元数据行**，而且跳回去时选中的是那一行。
func TestJump_key条目指向元数据行(t *testing.T) {
	m, _, _ := jumpModel(t)
	ts := targetsOf(t, m, "filetype:15")
	if len(ts) == 0 {
		t.Fatal("filetype:15 在本模型里没有位置 —— 而模型的 file_type 正是 15")
	}
	if !strings.Contains(ts[0].label, "general.file_type") {
		t.Errorf("指向的是 %q，不是那条元数据", ts[0].label)
	}
	if !ts[0].ok || ts[0].cmd == nil {
		t.Fatal("这条跳不过去")
	}

	// **命令发出来的必须是"回到 ModelView 并选中那一行"**，不是随便一个视图
	msg, ok := ts[0].cmd().(popToMsg)
	if !ok {
		t.Fatalf("发出的是 %T, want popToMsg", ts[0].cmd())
	}
	sel, ok := msg.msg.(selectMetaMsg)
	if !ok {
		t.Fatalf("popToMsg 里装的是 %T, want selectMetaMsg", msg.msg)
	}
	if sel.index != 0 {
		t.Errorf("选中第 %d 条, want 0", sel.index)
	}
}

// **反向用的是与正向同一个判定**：元数据里挂了 `◂` 的那些行，
// 从对应条目跳过来必须能找到它。
//
// 各写一套匹配的话会出现"有记号但跳不来"，而两边各自都自洽。
func TestJump_与正向判定一致(t *testing.T) {
	m, _, _ := jumpModel(t)
	mv := NewModelView(m)

	checked := 0
	for i, kv := range m.Metadata {
		e, ok := mv.metaEntry(kv)
		if !ok {
			continue // 这一行本来就没有关联
		}
		checked++
		ts := targetsOf(t, m, e.ID)
		found := false
		for _, tg := range ts {
			if msg, ok := tg.cmd().(popToMsg); ok {
				if sel, ok := msg.msg.(selectMetaMsg); ok && sel.index == i {
					found = true
				}
			}
		}
		if !found {
			t.Errorf("元数据第 %d 条 (%s) 指向 %s，但从那个条目跳不回来",
				i, kv.Key, e.ID)
		}
	}
	// **一行都没走到时这条测试是空转的**：没有这一句的话，把 jumpModel
	// 的 general.file_type 换成一个速查表里没有的键，循环零次、断言零次、
	// 测试照样绿 —— 而它想验的那件事一次都没验过。
	if checked == 0 {
		t.Fatal("这个模型里一行有关联的元数据都没有 —— 这条测试什么都没验")
	}
}

// quant 类条目要指出用这个类型的张量有几个、并能跳到过滤后的张量列表。
func TestJump_quant条目指向张量(t *testing.T) {
	m, _, _ := jumpModel(t)
	ts := targetsOf(t, m, "quant:Q4_K")
	if len(ts) == 0 {
		t.Fatal("quant:Q4_K 在本模型里没有位置 —— 而模型里有 3 个 Q4_K 张量")
	}
	if !strings.Contains(ts[0].label, "3") {
		t.Errorf("没说有几个: %q", ts[0].label)
	}
	msg, ok := ts[0].cmd().(pushMsg)
	if !ok {
		t.Fatalf("发出的是 %T, want pushMsg", ts[0].cmd())
	}
	tv, ok := msg.v.(TensorsView)
	if !ok {
		t.Fatalf("推入的是 %T, want TensorsView", msg.v)
	}
	// **必须只显示 Q4_K 的那 3 个**
	if got := len(tv.shown()); got != 3 {
		t.Errorf("过滤后有 %d 个张量, want 3", got)
	}
}

// tensor 段条目要按**名字拆段**匹配，不是按子串。
//
// 按子串的话 "q" 会匹配到一大堆名字里恰好含 q 的张量；
// 拆段之后每个段是一个完整的语义单元（blk / #N / attn_q / weight）。
func TestJump_tensor段条目指向张量(t *testing.T) {
	m, segA, segB := jumpModel(t)

	ts := targetsOf(t, m, "tensor:"+segA)
	if len(ts) == 0 {
		t.Fatalf("tensor:%s 在本模型里没有位置 —— 而模型里有 3 个含它的张量", segA)
	}
	if !strings.Contains(ts[0].label, "3") {
		t.Errorf("没说有几个: %q", ts[0].label)
	}

	// **segB 只出现在一个张量的名字里，不该被算进 segA 那一堆**
	ts2 := targetsOf(t, m, "tensor:"+segB)
	if len(ts2) == 0 {
		t.Fatalf("tensor:%s 在本模型里没有位置", segB)
	}
	if !strings.Contains(ts2[0].label, "1") {
		t.Errorf("%s 的定位不对（应当只算 1 个）: %+v", segB, ts2)
	}

	// 而且跳过去之后，张量列表里**只该有那 1 个**
	msg, ok := ts2[0].cmd().(pushMsg)
	if !ok {
		t.Fatalf("发出的是 %T, want pushMsg", ts2[0].cmd())
	}
	tv, ok := msg.v.(TensorsView)
	if !ok {
		t.Fatalf("推入的是 %T, want TensorsView", msg.v)
	}
	if got := len(tv.shown()); got != 1 {
		t.Errorf("按 %s 筛出 %d 个张量, want 1", segB, got)
	}
}

// 本模型里没有这个条目对应的东西时，**整段不显示** ——
// 留一行"0 处"的话，用户会以为是自己看漏了。
func TestJump_没有对应位置就不显示(t *testing.T) {
	m, segA, segB := jumpModel(t)
	segC := aSegmentNotUsed(t, m, segA, segB)

	ts := targetsOf(t, m, "tensor:"+segC)
	if len(ts) != 0 {
		t.Errorf("本模型里没有 %s，却给出了 %d 个位置", segC, len(ts))
	}
	out := NewEntryView(m, mustEntry(t, "tensor:"+segC)).View(100, 40)
	if strings.Contains(out, "在本模型中") {
		t.Errorf("没有对应位置却显示了那一节:\n%s", out)
	}
}

// 模型为 nil 时不该显示"在本模型中"，更不该崩。
func TestJump_没有模型上下文(t *testing.T) {
	seg := realSegments(t, 1)[0]
	out := NewEntryView(nil, mustEntry(t, "tensor:"+seg)).View(100, 40)
	if strings.Contains(out, "在本模型中") {
		t.Errorf("没有模型上下文却显示了那一节 —— 事实是「不知道」，不是「没有」:\n%s", out)
	}
}

// aSegmentNotUsed 找一个**速查表里有、但模型的名字里没有**的段。
//
// 与 realSegments 同一个理由：写死一个段名（比如 ffn_gate）的话，
// 那张表改了或者模型那边改了，这条测试就假失败。
func aSegmentNotUsed(t *testing.T, m *model.Model, used ...string) string {
	t.Helper()
	skip := make(map[string]bool, len(used))
	for _, s := range used {
		skip[s] = true
	}
	for _, tb := range ref.Tables() {
		if tb.ID != "tensors" {
			continue
		}
		for _, e := range tb.Entries {
			seg := strings.TrimPrefix(e.ID, "tensor:")
			if !skip[seg] && !nameContainsSegment(m, seg) {
				return seg
			}
		}
	}
	t.Fatal("速查表里每个段都在这个模型里出现了 —— 这条测试造不出场景")
	return ""
}

// nameContainsSegment 判断模型里有没有哪个张量名含这一段。
func nameContainsSegment(m *model.Model, seg string) bool {
	for _, tn := range m.Tensors {
		for _, s := range ref.SplitTensorName(tn.Name) {
			if s == seg {
				return true
			}
		}
	}
	return false
}

func mustEntry(t *testing.T, id string) ref.Entry {
	t.Helper()
	e, ok := ref.ByID(id)
	if !ok {
		t.Fatalf("速查表里没有 %q", id)
	}
	return e
}
