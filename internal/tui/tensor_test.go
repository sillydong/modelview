package tui

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/analyze"
	"github.com/sillydong/modelview/internal/model"
)

// fakeTensor 造一个 F32 张量，指向一个假模型。
func fakeTensor(name string) (*model.Model, *model.Tensor) {
	m := &model.Model{Path: "/x/m.gguf", Format: model.FormatGGUF, Version: "v3", Arch: "qwen2"}
	tn := &model.Tensor{
		Name: name, Dims: []int64{2048, 2048}, Dtype: model.DtypeQ4K,
		ByteSize: 2 << 20, ParamCount: 4194304,
	}
	m.Tensors = append(m.Tensors, tn)
	return m, tn
}

// stubScan 是一个不碰磁盘的 scan 实现。
func stubScan(err error, fill func(*model.Tensor)) func(context.Context, *model.Model, *model.Tensor) error {
	return func(_ context.Context, _ *model.Model, tn *model.Tensor) error {
		if fill != nil {
			fill(tn)
		}
		return err
	}
}

// expand 把 tea.Batch 包起来的命令展开成消息列表。
//
// **必须展开**：tea.Cmd 一次只返回一个消息，而 Init 用 tea.Batch
// 同时发起"扫描"与"计时"两件事 —— 直接对 cmd() 的结果做类型断言
// 拿到的是 tea.BatchMsg，断言会失败，而失败信息看着像"Init 没发起扫描"。
func expand(t *testing.T, cmd tea.Cmd) []tea.Msg {
	t.Helper()
	msg := cmd()
	if batch, ok := msg.(tea.BatchMsg); ok {
		out := make([]tea.Msg, 0, len(batch))
		for _, c := range batch {
			if c == nil {
				continue
			}
			out = append(out, c())
		}
		return out
	}
	return []tea.Msg{msg}
}

// 扫描在后台跑：Init 必须同时发起扫描与计时。
func TestTensorView_Init发起扫描(t *testing.T) {
	m, tn := fakeTensor("blk.0.attn_q.weight")
	v := NewTensorView(m, tn)
	v.scan = stubScan(nil, func(tn *model.Tensor) {
		tn.Stats = &model.Stats{Count: 100, Mean: 1}
	})

	cmd := v.Init()
	if cmd == nil {
		t.Fatal("Init 没有返回命令 —— 详情页会永远停在初始状态")
	}

	var scanned *tensorScannedMsg
	ticked := false
	for _, msg := range expand(t, cmd) {
		switch msg := msg.(type) {
		case tensorScannedMsg:
			scanned = &msg
		case tickMsg:
			ticked = true
		}
	}
	if scanned == nil {
		t.Fatal("Init 没有发起扫描")
	}
	if scanned.err != nil || scanned.stats == nil {
		t.Errorf("扫描结果 = %+v", *scanned)
	}
	if !ticked {
		t.Error("Init 没有启动计时 —— 扫描期间画面不会动，用户以为卡死了")
	}
}

// 已经算过的张量不该再扫一遍（也就不会闪一下"扫描中"）。
func TestTensorView_已算过不重扫(t *testing.T) {
	m, tn := fakeTensor("blk.0.attn_q.weight")
	tn.Stats = &model.Stats{Count: 1}
	// **Q4_K 的"算完了"是两件**：量化类型要 Stats 与 Quant 都非空
	//（浮点类型则是 Stats 与 QuantSims 两件，见 TestOne_已算完就不碰盘）。
	// 只设 Stats 的话 NeedsWork 仍为真 —— 计划在这里踩过一次。
	tn.Quant = &model.QuantInfo{Scheme: "Q4_K", BitsPerWeight: 4.5, SubBlocks: 1}
	v := NewTensorView(m, tn)

	if v.scanning {
		t.Error("已有统计的张量不该显示扫描中")
	}
	if cmd := v.Init(); cmd != nil {
		t.Error("已有统计的张量不该再发起扫描")
	}
}

// **`Stats` 有而 `Quant` 无 ⇒ 仍然要扫** —— 这条是 `analyze.NeedsWork` 存在的理由。
//
// analyze.go 的注释写着"不能只看 Stats != nil：那会把「统计算过了」
// 当成「分析做完了」……实测冷跑 181 个模拟 / 253 个诊断，热跑 0 / 0"。
//
// **这条语义在本计划里被误读过两次**（Task 4 与 Task 5 的 fixture 都只设了
// Stats、没设 Quant），说明它很容易被读漏 —— 所以必须有测试钉住它，
// 而不是只写在注释里。注释挡不住一个已经犯过两次的错误。
//
// 参考：量化类型的"算完了"是 Stats + Quant 两件；
// 浮点类型是 Stats + QuantSims 两件（后者见 TestOne_已算完就不碰盘）。
func TestTensorView_只算了统计仍要扫(t *testing.T) {
	m, tn := fakeTensor("blk.0.attn_q.weight") // Q4_K
	tn.Stats = &model.Stats{Count: 1}
	// Quant 故意留 nil

	if !analyze.NeedsWork(tn) {
		t.Fatal("只有 Stats 的量化张量被判成「算完了」—— 这正是 analyze 注释里点名的那个误读")
	}
	v := NewTensorView(m, tn)
	if !v.scanning {
		t.Error("只算了统计的量化张量不该显示为已完成")
	}
	if cmd := v.Init(); cmd == nil {
		t.Error("只算了统计的量化张量必须发起扫描 —— 块级诊断还没算")
	}
}

// **扫描中不读 tn.Stats** —— 那条 goroutine 正在往副本里写，
// 但主线程读的是**共享的 tn 字段**（副本的合并要等消息回来）。
// 这条规则一旦被破坏，表现是"偶尔显示半个统计"，且 -race 下才会报。
//
// 所以这里故意把 tn.Stats 设成非 nil 而 scanning 设成 true：
// 正确实现必须**看不见**它。
func TestTensorView_扫描中不读统计(t *testing.T) {
	m, tn := fakeTensor("blk.0.attn_q.weight")
	tn.Stats = &model.Stats{Count: 100, Mean: 1, Max: 7}
	v := NewTensorView(m, tn)
	v.scanning = true // 模拟"扫描在途"

	out := v.View(100, 30)
	if strings.Contains(out, "μ=") {
		t.Errorf("扫描中读了 tn.Stats —— 与那条 goroutine 是数据竞争:\n%s", out)
	}
	if !strings.Contains(out, "已用") {
		t.Errorf("扫描中没显示计时 —— 用户没法判断是在算还是卡死:\n%s", out)
	}
}

// 计时在动：时间推进后画面必须不一样（否则"在动"是假的）。
func TestTensorView_计时会变(t *testing.T) {
	m, tn := fakeTensor("blk.0.attn_q.weight")
	v := NewTensorView(m, tn)
	v.scanning = true

	first := v.View(100, 30)
	v.elapsed = 3 * time.Second
	if second := v.View(100, 30); second == first {
		t.Errorf("过了 3 秒画面一模一样 —— 用户会以为卡死了:\n%s", first)
	}
}

// 扫完之后：结果写回张量（列表里那行才会显示"已扫描"），并且**停止计时**。
//
// 不停的话每秒一次无谓重绘，一直烧 CPU 直到用户离开这一页。
func TestTensorView_扫完写回并停表(t *testing.T) {
	m, tn := fakeTensor("blk.0.attn_q.weight")
	v := NewTensorView(m, tn)

	stats := &model.Stats{Count: 100, Mean: 1, Max: 7}
	next, cmd := v.Update(tensorScannedMsg{name: tn.Name, stats: stats})
	v = next.(TensorView)

	if cmd != nil {
		t.Errorf("扫描完成后 Update 还返回了命令: %v", cmd)
	}
	if tn.Stats != stats {
		t.Error("扫描结果没写回张量 —— 列表里那行不会显示已扫描")
	}
	if v.scanning {
		t.Error("扫描完成了还在显示扫描中")
	}
	if cmd := v.tick(); cmd != nil {
		t.Error("扫描完成后计时还在继续 —— 会一直无谓重绘")
	}
	if out := v.View(100, 30); !strings.Contains(out, "μ=") {
		t.Errorf("扫描完成后没显示统计:\n%s", out)
	}
}

// 失败要说清楚原因，而且**不能再显示"扫描中"** ——
// 一个既说在扫又说失败了的界面，用户不知道信哪个。
func TestTensorView_失败要说明原因(t *testing.T) {
	m, tn := fakeTensor("blk.0.attn_q.weight")
	v := NewTensorView(m, tn)

	next, _ := v.Update(tensorScannedMsg{
		name: tn.Name, err: errors.New("张量不是连续布局，无法按线性顺序解码")})
	v = next.(TensorView)

	out := v.View(100, 30)
	if !strings.Contains(out, "连续布局") {
		t.Errorf("没显示失败原因:\n%s", out)
	}
	if strings.Contains(out, "已用") {
		t.Errorf("已经失败了还在显示扫描中:\n%s", out)
	}
	if tn.Stats != nil {
		t.Error("失败之后不该留下 Stats")
	}
}

// 消息的张量名对不上就**整个丢弃** —— 用户的 Esc 与扫描完成会撞在一起：
// 回到列表、又点开另一个张量，此时上一个的扫描结果才回来。
// 名字对不上还往当前张量上合并的话，显示的是**另一个张量**的统计。
func TestTensorView_名字对不上就丢弃(t *testing.T) {
	m, tn := fakeTensor("blk.0.attn_q.weight")
	v := NewTensorView(m, tn)

	v2, _ := v.Update(tensorScannedMsg{
		name:  "blk.9.something_else.weight",
		stats: &model.Stats{Count: 7, Mean: 3},
	})
	if tn.Stats != nil {
		t.Fatal("把别的张量的统计合并进来了 —— 界面会显示一个不属于它的分布")
	}
	if out := v2.(TensorView).View(100, 30); strings.Contains(out, "μ=3") {
		t.Errorf("显示了别的张量的统计:\n%s", out)
	}
}

// 直方图：桶数多于列数时按段取最大值合并，尖峰必须留得下。
//
// 抽稀（每 N 个取一个）会把窄峰整个漏掉 —— 而"有没有尖峰"
// 正是看这张图的原因。
func TestTensorView_直方图尖峰不丢(t *testing.T) {
	buckets := make([]int64, 64)
	for i := range buckets {
		buckets[i] = 1
	}
	// 尖峰放在**合并段中间**（41，不是 40）：40 恰好是第 10 段的第一个桶，
	// 而"取每段第一个桶"的错法在那个位置上输出与正确版**逐字节相同**
	//（实测：尖峰在 40 时，变异 ④ 漏网）。换个位置才看得出差别。
	buckets[41] = 1000 // 一个窄峰

	out := histogram(buckets, 16)
	if !strings.Contains(out, "█") {
		t.Errorf("尖峰在合并后消失了:\n%s", out)
	}
	// **要数"满高柱有几根"，不能只数"有没有满高柱"**：
	// 合并把尖峰漏掉之后每个桶都是 1、峰值也是 1，于是**每一列**都被画成
	// 满高 —— `Contains(out, "█")` 在漏掉尖峰的版本里照样为真（实测漏网）。
	if n := strings.Count(out, "█"); n != 1 {
		t.Errorf("满高柱有 %d 根, want 1 —— 尖峰被合并漏掉了（或整行都画满了）:\n%s", n, out)
	}
	// 每列一个字符，16 列不能画成 64 个字符
	if n := len([]rune(out)); n != 16 {
		t.Errorf("画了 %d 列, want 16:\n%s", n, out)
	}
}

// 桶数少于列数时不该被拉长，也不该报错。
func TestTensorView_直方图桶数少于列数(t *testing.T) {
	out := histogram([]int64{1, 2, 3}, 10)
	if n := len([]rune(out)); n != 3 {
		t.Errorf("画了 %d 列, want 3", n)
	}
}

// 全零桶（区间无意义的张量）不能除零，也不能画出全高的假柱。
func TestTensorView_直方图全零(t *testing.T) {
	out := histogram(make([]int64, 64), 16)
	if strings.Contains(out, "█") {
		t.Errorf("全零直方图画出了实心柱 —— 那是假的:\n%s", out)
	}
}

// **量化那两节不能走单行版** —— 这条测试就是为此存在的。
//
// 单行版（QuantExisting / QuantSim）是给 CLI 用的：CLI 输出到管道或文件，
// 宽度不受限。TUI 的宽度是**终端的宽度**，而 80 列是 spec 定的最小终端。
// 用单行版的话根视图的 padTo 会静默截断，砍掉的恰好是
// "⚠压平 N 子块 / 最扁 #N"——那两条在 render 和 model 的注释里
// 都被点名"必须显示"。
func TestTensorView_窄终端不超宽且不丢关键信息(t *testing.T) {
	m, tn := fakeTensor("blk.0.attn_q.weight")
	tn.Quant = &model.QuantInfo{
		Scheme: "Q4_K", BitsPerWeight: 4.5, SubBlocks: 124400,
		ScaleMin: 0.0001, ScaleMax: 1.234, ScaleMedian: 0.0268,
		ZeroScaleBlocks: 3, FlattestIndex: 12345, FlattestRatio: 0.02,
	}
	// Sampled 全为 true：**采样是常态**（DefaultSampleLimit 1000 万，
	// 一个 4096×4096 的 f16 矩阵就是 1677 万），所以下面必须断言
	// `≈` 出现在界面上 —— 分行版曾经把它静默丢掉，
	// 于是同一页的 CLI 说"≈模拟"、TUI 说得像精确值。
	tn.QuantSims = []model.QuantSim{
		{Target: "Q8_0", BitsPerWeight: 8.5, SNRDB: 30.1, Compression: 3.76, Sampled: true},
		{Target: "Q6_K", BitsPerWeight: 6.5625, SNRDB: 26.5, Compression: 4.88, Sampled: true},
		{Target: "Q4_K", BitsPerWeight: 4.5, SNRDB: 20.9, Compression: 7.11, Sampled: true},
	}
	v := NewTensorView(m, tn)
	next, _ := v.Update(tensorScannedMsg{
		name: tn.Name, stats: &model.Stats{Count: 100, Sampled: true},
		quant: tn.Quant, sims: tn.QuantSims})
	v = next.(TensorView)

	out := v.View(80, 40)
	for _, line := range strings.Split(out, "\n") {
		if w := displayWidth(line); w > 80 {
			t.Errorf("有一行宽 %d 列，超过 80 —— 会被静默截掉尾巴:\n%s", w, line)
		}
	}
	// 两条"承诺必须显示"的信息，加上位宽与采样标记
	for _, want := range []string{"压平", "最扁", "6.5625", "≈"} {
		if !strings.Contains(out, want) {
			t.Errorf("80 列下丢了 %q —— 那是注释里承诺要显示的东西:\n%s", want, out)
		}
	}
	// **表头那行的位宽要单独钉**（`类型 Q4_K（4.5 bit/权重）`）。
	//
	// 上面那串里的 "6.5625" 挡不住"把位宽换成 humanize.Float"的变异：
	// 实测 humanize.Float(6.5625) == "6.5625"，与 render.BitsPerWeight
	// **逐字节相同**（它是四位小数，那个格式器也留四位小数）——
	// 会打短位宽的是 %.4g，不是 humanize.Float。
	// 两者真正分叉的地方是**整数位宽**：humanize.Float(4.5) 是 "4.5000"。
	// 实测：只断言 "6.5625" 时，变异 ⑥ 漏网。
	if !strings.Contains(out, "Q4_K（4.5 bit/权重）") {
		t.Errorf("表头位宽不是精确值 —— 是不是走了 humanize.Float（4.5 会变成 4.5000）:\n%s", out)
	}
}
