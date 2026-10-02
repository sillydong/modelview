package tui

import (
	"context"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/analyze"
	"github.com/sillydong/modelview/internal/humanize"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/ref"
	"github.com/sillydong/modelview/internal/render"
)

// tickInterval 是"扫描中"那一行的重绘间隔。
//
// 取 120ms 是为了转圈看着顺（约 8 帧/秒），不是为了刷数字 ——
// 秒数一位小数，本来每 100ms 才变一次。
const tickInterval = 120 * time.Millisecond

// spinnerFrames 是转圈的帧。用盲文点阵而不是 |/-\\ ：
// 后者在等宽字体里宽度一致但在中文字体里会被当成全角，整行会跳。
var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// histogramBlocks 是直方图的八级柱高。
var histogramBlocks = []rune{'▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'}

// tensorScannedMsg 是一次单张量扫描的结果。
//
// **必须带张量名**：用户的 Esc 与扫描完成会撞在一起 ——
// 回到列表、点开另一个张量，此时上一个的扫描结果才回来。
// 不带名字的话，那份统计会被合并到**当前**这个张量上，
// 显示的是一个不属于它的分布，而且没有任何东西会红。
type tensorScannedMsg struct {
	name  string
	stats *model.Stats
	quant *model.QuantInfo
	sims  []model.QuantSim
	err   error
}

// tickMsg 驱动"扫描中"的计时。
type tickMsg time.Time

// TensorView 是单个张量的详情页。
//
// **它就地写 tn**（扫描结果合并到 m.Tensors 里那一个），
// 这是刻意的：回到列表时那一行要显示"已扫描"，
// 而列表与详情页共享同一批 *model.Tensor。
type TensorView struct {
	m  *model.Model
	tn *model.Tensor

	// scan 注入是为了测试不碰真实文件 —— 真的那个要读几秒磁盘。
	scan func(context.Context, *model.Model, *model.Tensor) error

	scanning bool
	elapsed  time.Duration
	err      error

	segCursor int
}

// tensorSegment 是张量名里的一个段，附带它对应的速查表条目。
type tensorSegment struct {
	seg string
	e   ref.Entry
}

// segments 列出名字里**能查到速查表条目**的那些段。
//
// 只列查得到的：把 "weight" 这种谁都认识的段也列上去的话，
// 用户真正关心的那个（`attn_q_norm` 那种）会被淹在里面。
// 查不到的段不是"错了"，只是这张表还没收录它 —— 不显示不等于不存在。
//
// 归一（"#12" → "#N"）交给 ref.LookupTensorSegment，**不在这里自己写**：
// 那段逻辑曾经只存在于测试文件里，于是生产路径上永远查不到带层号的段
// （ref 包里 normalizeLayer 的原话）。
func (v TensorView) segments() []tensorSegment {
	var out []tensorSegment
	for _, s := range ref.SplitTensorName(v.tn.Name) {
		if e, ok := ref.LookupTensorSegment(s); ok {
			out = append(out, tensorSegment{seg: s, e: e})
		}
	}
	return out
}

func NewTensorView(m *model.Model, tn *model.Tensor) TensorView {
	return TensorView{
		m: m, tn: tn, scan: analyze.One,
		// **扫描状态在构造时定，不在 Init 里定**：Init 是值接收者，
		// 改不了字段 —— 在它里面写 v.scanning = true 是改一个马上被丢掉的副本，
		// 界面会永远停在"没有扫描中"的那一版（这个错误静默且致命）。
		//
		// 判据直接用 analyze.NeedsWork：这一栏要答的是"点开这个张量
		// 还要不要等"，与 analyze.Analyze/One 的入口闸门是同一个问题。
		//（"什么时候该重扫"那三处表达的差异见 NeedsWork 的说明。）
		scanning: analyze.NeedsWork(tn),
	}
}

func (v TensorView) Title() string {
	return fmt.Sprintf("张量 · %s · %s · %s",
		humanize.Truncate(v.tn.Name, 48), v.tn.Dtype,
		humanize.Bytes(v.tn.ByteSize))
}

// Init 在后台发起扫描。
//
// 已经算过的（内存里有结果）直接返回 nil —— 不返回 nil 的话
// 界面会闪一下"扫描中"，而它其实什么都没扫。
func (v TensorView) Init() tea.Cmd {
	if !v.scanning {
		return nil
	}
	return tea.Batch(v.scanCmd(), v.tick())
}

// Modal 恒为 false：张量详情没有输入框，q 与 Esc 照旧归根视图。
func (v TensorView) Modal() bool { return false }

// scanCmd 在 goroutine 里扫一份**副本**，结果通过消息回到主线程。
//
// **必须扫副本**：analyze.One 会就地写 tn.Stats/Quant/QuantSims，
// 而主线程随时可能在渲染（用户按 Esc 回列表，列表每行都要看 tn.Stats）。
// 一写一读就是数据竞争 —— 表现是"偶尔显示半个统计"，
// 只有 -race 才能稳定看到。在副本上算、回主线程再合并，
// 共享可写状态就只剩 Update 这一处。
func (v TensorView) scanCmd() tea.Cmd {
	m, tn, scan := v.m, v.tn, v.scan
	return func() tea.Msg {
		cp := *tn
		err := scan(context.Background(), m, &cp)
		return tensorScannedMsg{
			name: tn.Name, stats: cp.Stats, quant: cp.Quant,
			sims: cp.QuantSims, err: err,
		}
	}
}

// tick 造一条计时消息。
func (v TensorView) tick() tea.Cmd {
	if !v.scanning {
		return nil
	}
	return tea.Tick(tickInterval, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (v TensorView) Update(msg tea.Msg) (View, tea.Cmd) {
	switch msg := msg.(type) {
	case tensorScannedMsg:
		// **名字对不上整个丢弃**（见 tensorScannedMsg 的说明）
		if msg.name != v.tn.Name {
			return v, nil
		}
		v.scanning, v.err = false, msg.err
		// **合并必须在主线程做** —— 这正是 scanCmd 扫副本的原因
		v.tn.Stats, v.tn.Quant, v.tn.QuantSims = msg.stats, msg.quant, msg.sims
		return v, nil

	case tickMsg:
		if !v.scanning {
			return v, nil
		}
		v.elapsed += tickInterval
		return v, v.tick()

	case tea.KeyMsg:
		segs := v.segments()
		switch msg.String() {
		case "up", "k":
			if v.segCursor > 0 {
				v.segCursor--
			}
		case "down", "j":
			if v.segCursor < len(segs)-1 {
				v.segCursor++
			}
		case "enter":
			// **必须判 `v.segCursor < len(segs)`**：名字里一个段都查不到时
			// `len(segs) == 0`，不判就是下标越界 panic
			if v.segCursor < len(segs) {
				return v, pushCmd(NewEntryView(v.m, segs[v.segCursor].e))
			}
		}
	}
	return v, nil
}

func (v TensorView) Help() []string {
	if len(v.segments()) > 0 {
		return []string{
			keyUp + " " + keyDown + " 选择名字里的段",
			keyEnter + " 查速查表",
			keyEsc + " 返回",
			keyQuit + " 退出",
		}
	}
	return []string{keyEsc + " 返回", keyQuit + " 退出"}
}

func (v TensorView) View(width, height int) string {
	var sb strings.Builder

	fmt.Fprintf(&sb, "%s\n", styleSection.Render(humanize.Truncate(v.tn.Name, width)))
	fmt.Fprintf(&sb, "形状     %s\n", humanize.Dims(v.tn.Dims))
	fmt.Fprintf(&sb, "类型     %s", v.tn.Dtype)
	if bpw := v.tn.Dtype.BitsPerWeight(); bpw > 0 {
		// **位宽走 render.BitsPerWeight，不走 humanize.Float**：
		// Float 是给统计量用的（[1e-3, 1e5) 内 4 位小数、区间外科学计数法），
		// 它**不是为精确性设计的**，而且会补尾零（Float(4.5) = "4.5000"）。
		// BitsPerWeight 按构造精确且去尾零 —— 位宽是文件里的精确值
		//（Q6_K 就是 6.5625），显示成别的数用户会拿它算文件大小。
		fmt.Fprintf(&sb, "（%s bit/权重）", render.BitsPerWeight(bpw))
	}
	sb.WriteString("\n")
	fmt.Fprintf(&sb, "元素     %s（%s）\n",
		humanize.Count(v.tn.ParamCount), humanize.Comma(v.tn.ParamCount))
	fmt.Fprintf(&sb, "占用     %s\n", humanize.Bytes(v.tn.ByteSize))
	if !v.tn.OffsetUnknown {
		fmt.Fprintf(&sb, "偏移     %#x\n", v.tn.Offset)
	}

	// **"名字构成"在扫描前后都在**：它只看名字，与扫描结果无关 ——
	// 放在 scanning 那一段之后的话，扫描中的几秒里这一节会消失、
	// 扫完又冒出来，用户会以为是自己看错了
	if segs := v.segments(); len(segs) > 0 {
		sb.WriteString("\n" + styleSection.Render("名字构成") + "\n")
		for i, s := range segs {
			line := fmt.Sprintf("  %-16s %s", s.seg, s.e.Title)
			if i == v.segCursor {
				sb.WriteString(styleSelected.Render("▸ "+fmt.Sprintf("%-16s %s", s.seg, s.e.Title)) + "\n")
				continue
			}
			sb.WriteString(line + "\n")
		}
		sb.WriteString(styleHint.Render(fmt.Sprintf("  按 %s 查这一段", keyEnter)) + "\n")
	}

	// **扫描中绝不读 tn 的统计字段**：那条 goroutine 正在算，
	// 合并要等消息回来。这里读就是数据竞争。
	if v.scanning {
		frame := spinnerFrames[int(v.elapsed/tickInterval)%len(spinnerFrames)]
		sb.WriteString("\n" + styleHint.Render(fmt.Sprintf(
			"%s 正在读取并计算（已用 %.1f 秒）…", frame, v.elapsed.Seconds())) + "\n")
		// **末尾不留换行**（下面那个 return 同理）：padTo 按 "\n" 切行，
		// 多出来的那个空元素让它多算一行 —— 内容放不下时
		// "…还有 N 行没显示"里的 N 会比实际丢掉的多一，等于屏幕上
		// 印一个错的数字；内容恰好放得下时还会被白白砍掉一行。
		// joinHorizontal 里记过同一条，Library 与这个详情页都踩过。
		return strings.TrimSuffix(sb.String(), "\n")
	}

	sb.WriteString("\n")
	sb.WriteString(v.results(width))
	return strings.TrimSuffix(sb.String(), "\n")
}

// results 渲染扫描结果那一段。**只在扫描结束后调用**。
func (v TensorView) results(width int) string {
	var sb strings.Builder
	switch {
	case v.err != nil:
		sb.WriteString(styleWarn.Render("统计失败："+v.err.Error()) + "\n")
		return sb.String()
	case v.tn.Stats == nil:
		// 理论上到不了（NeedsWork 为假且没错误）。
		// 留着是因为"什么都没有"比一句说明更难查 —— 用户会以为是界面坏了。
		sb.WriteString(styleHint.Render("这个张量没有可统计的数值") + "\n")
		return sb.String()
	}

	sb.WriteString(styleSection.Render("统计") + "\n")
	fmt.Fprintf(&sb, "  %s\n", render.Stats(v.tn.Stats))
	if v.tn.Stats.Sampled {
		// **采样说明必须单独一行**：与统计挤在同一行实测 95 列，
		// 80 列终端下会被根视图的 padTo 静默截断 —— 而截掉的恰好是
		// "这是采样值"那半句，剩下的数字看起来像精确值。
		// 那正是 model.Stats 的注释点名要避免的误导。
		fmt.Fprintf(&sb, "  %s\n", styleDim.Render(fmt.Sprintf(
			"采样 %s / %s 个元素（来自样本，非全量）",
			humanize.Count(v.tn.Stats.Count), humanize.Count(v.tn.ParamCount))))
	}
	if h := histogram(v.tn.Stats.Histogram, min(width-4, 64)); h != "" {
		fmt.Fprintf(&sb, "  %s\n", styleDim.Render(h))
	}

	if v.tn.Quant != nil {
		// **标题说"读遍全部块头"，不说"全部子块"**：前者是事实
		//（块级诊断按设计要读遍全部块头，不受采样上限约束），
		// 后者会被读成"这一节里每个数都是全量的"—— 而中位数是抽样来的
		//（ScaleMedianSampled，本机 qwen2.5:3b 上有 1 个张量触发）。
		// 那一处由 render 的 `≈…（抽样）` 自己标出来。
		sb.WriteString("\n" + styleSection.Render("量化诊断（读遍全部块头）") + "\n")
		// **用分行版，不用单行版**：单行版实测 105 列，80 列终端下
		// 被静默截断，砍掉的恰好是"⚠压平 N 子块 / 最扁 #N" ——
		// 而 render 里那两条的注释写着"必须显示"，
		// model.QuantInfo 的注释也承诺了界面上要显示它。
		// 承诺了又切掉，比不承诺更糟。
		for _, line := range render.QuantExistingLines(v.tn.Quant) {
			fmt.Fprintf(&sb, "  %s\n", line)
		}
	}
	if len(v.tn.QuantSims) > 0 {
		sb.WriteString("\n" + styleSection.Render("量化模拟") + "\n")
		// 三档压一行实测 104 列（同样会被截掉压缩最狠的 Q4_K 那档），
		// 所以用分行版。
		//
		// **绝不在 TUI 里拼格式串**：拼了就长回两份渲染，
		// internal/render 这个包存在的理由就没了 ——
		// 而两份渲染分叉时不报错，只是同一页出现两个不同的数。
		for _, line := range render.QuantSimLines(v.tn.QuantSims) {
			fmt.Fprintf(&sb, "  %s\n", line)
		}
		sb.WriteString(styleDim.Render(
			"  （按各格式真实编码器算，未用重要性矩阵加权）") + "\n")
	}
	return sb.String()
}

// histogram 把直方图桶画成一行柱子。
//
// **桶数多于列数时按段取最大值合并**，不是抽稀：抽稀（每 N 个取一个）
// 会把窄而高的峰整个漏掉，而"有没有尖峰"正是看这张图的原因。
// 合并之后的列数恰好等于 width，不会超宽。
func histogram(buckets []int64, width int) string {
	if len(buckets) == 0 || width <= 0 {
		return ""
	}
	if len(buckets) <= width {
		width = len(buckets)
	}
	cols := make([]int64, 0, width)
	var peak int64
	for i := range width {
		lo := i * len(buckets) / width
		hi := (i + 1) * len(buckets) / width
		if hi <= lo {
			hi = lo + 1
		}
		var m int64
		for _, c := range buckets[lo:hi] {
			if c > m {
				m = c
			}
		}
		cols = append(cols, m)
		if m > peak {
			peak = m
		}
	}
	// 全零：区间无意义（值全是 NaN/Inf）。画等高的虚柱会**看起来很真**，
	// 所以宁可什么都不画，让调用方那行直接空掉。
	if peak <= 0 {
		return ""
	}
	var sb strings.Builder
	for _, c := range cols {
		// -1 是为了让最小的非零桶也看得见：0 → '▁'，peak → '█'
		idx := int(c * int64(len(histogramBlocks)-1) / peak)
		sb.WriteRune(histogramBlocks[idx])
	}
	return sb.String()
}
