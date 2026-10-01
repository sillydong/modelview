package tui

import (
	"fmt"
	"sort"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/sillydong/modelview/internal/humanize"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/parser"
	"github.com/sillydong/modelview/internal/ref"
)

// modelLoadedMsg 是解析完成的消息。
type modelLoadedMsg struct {
	m   *model.Model
	err error
}

// section 是左栏的导航项。
type section int

const (
	sectionOverview section = iota
	sectionMetadata
	sectionTensors
	sectionQuantDist
	sectionRef
)

// sections 是导航项的名字。**顺序即显示顺序** ——
// 用户先要"这是什么模型"，再要细节。
var sections = []string{"概览", "元数据", "张量", "量化分布", "速查表"}

// ModelView 是单个模型的视图。
//
// **没有 section 字段**：栏目就是 section(cursor)，现算。
// 存两份迟早会不一致 —— 实测按一个与导航无关的键就能把 section
// 打回概览，因为 Update 末尾那句 `v.section = section(v.cursor)`
// 对每个按键都无条件执行。派生值不要单独存。
type ModelView struct {
	m           *model.Model
	pendingPath string // 非空表示还没解析
	name        string // 给用户看的名字，空则退回文件名
	err         error
	cursor      int
}

// NewModelView 用一个已解析的模型造视图（测试与"已经有 m"的场景用）。
func NewModelView(m *model.Model) ModelView {
	return ModelView{m: m}
}

// NewModelViewFromPath 从文件路径造视图，解析在 Init 里异步做。
//
// **解析必须异步**：读一个大模型的头部要几百毫秒到几秒，
// 放在 Update 里会让界面在切换时卡住。
//
// name 是**给用户看的名字**（模型库里那一列）。不传的话标题栏只能显示
// 路径的最后一段 —— 而 ollama 的路径是 blobs/sha256-7121486771cbfe2...，
// 一长串哈希对用户毫无意义（实测在真终端里就是这样）。
func NewModelViewFromPath(path, name string) ModelView {
	return ModelView{pendingPath: path, name: name}
}

func (v ModelView) Title() string {
	if v.err != nil {
		return "modelview · 读取失败"
	}
	if v.m == nil {
		if v.name != "" {
			return "modelview · " + v.name + " · 载入中"
		}
		return "modelview · 载入中"
	}
	return fmt.Sprintf("%s · %s %s · %s · %d 张量 · %s 参数",
		v.displayName(), v.m.Format, v.m.Version,
		humanize.Bytes(v.m.FileSize), len(v.m.Tensors),
		humanize.Count(v.m.TotalParams()))
}

func (v ModelView) Init() tea.Cmd {
	if v.pendingPath == "" {
		return nil
	}
	path := v.pendingPath
	return func() tea.Msg {
		m, err := parser.Parse(path)
		return modelLoadedMsg{m: m, err: err}
	}
}

func (v ModelView) Update(msg tea.Msg) (View, tea.Cmd) {
	switch msg := msg.(type) {
	case modelLoadedMsg:
		v.m, v.err = msg.m, msg.err
		return v, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if v.cursor > 0 {
				v.cursor--
			}
		case "down", "j":
			if v.cursor < len(sections)-1 {
				v.cursor++
			}
		case "enter":
			// **模型还没解析完时 Enter 什么也不做**：这时推入的任何子视图
			// 都会拿到一个 nil 的 *model.Model，而它们全都无保护地取
			// m.Tensors / m.Metadata —— 结果是 panic（实测复现过）。
			//
			// 放在这里统一早退，而不是每个分支各写一句 `v.m != nil`：
			// 后面每接一个栏目（速查表、元数据）都要记得加一次，
			// 漏一次就是一条崩溃路径，而它只在"进模型后立刻按 Enter"
			// 这个几秒的窗口里可达 —— 手测几乎撞不到。
			//
			// 界面此时显示的是"正在解析…"，所以用户看得出来为什么没反应。
			if v.m == nil {
				return v, nil
			}
			switch section(v.cursor) {
			case sectionTensors:
				return v, pushCmd(NewTensorsView(v.m))
			case sectionRef:
				return v, pushCmd(NewRefView(v.m))
			}
		}
	}
	return v, nil
}

// opensOnEnter 表示当前栏目的内容会在 Enter 时进子视图。
//
// 帮助栏只列真的按键 —— `keys.go` 里那条"列了不支持的等于骗用户按"
// 对这里同样成立。**每接上一个栏目的子视图，就在这里加一格**：
// 忘了加的表现是"按了有反应但帮助栏不写"（用户发现不了），
// 加多了的表现是"帮助栏写了一个按了没反应的键"（用户被指到死路）。
// 后者更糟，所以默认返回 false。
func (v ModelView) opensOnEnter() bool {
	return section(v.cursor) == sectionTensors || section(v.cursor) == sectionRef
}

func (v ModelView) Help() []string {
	bindings := []string{keyUp + " " + keyDown + " 切换栏目"}
	if v.opensOnEnter() {
		bindings = append(bindings, keyEnter+" 打开")
	}
	return append(bindings, keyEsc+" 返回", keyQuit+" 退出")
}

func (v ModelView) View(width, height int) string {
	if v.err != nil {
		return styleWarn.Render("读取失败：" + v.err.Error())
	}
	if v.m == nil {
		return styleHint.Render("正在解析…")
	}
	nav, navWidth := v.nav()
	// 右栏可用宽度 = 总宽 − 左栏 − 一个分隔空格。
	// **这里不做截断**：根视图的 padTo 已经按终端宽度统一截过了，
	// 各层再截一遍会出现"同一行在两层里算出的宽度不一样"。
	bodyWidth := width - navWidth - 1
	return joinHorizontal(nav, v.body(bodyWidth, height))
}

// nav 渲染左栏，返回内容与它的**显示宽度**。
//
// 必须返回宽度：右栏要对齐到它的右侧，而各栏目名字长短不一
// （"概览" vs "元数据 (52)"）—— 不补齐的话右栏每行的起始列都不一样，
// 真终端里看就是逐行错位（实测过）。
func (v ModelView) nav() (string, int) {
	labels := make([]string, len(sections))
	width := 0
	for i, name := range sections {
		label := name
		switch section(i) {
		case sectionMetadata:
			label = fmt.Sprintf("%s (%d)", name, len(v.m.Metadata))
		case sectionTensors:
			label = fmt.Sprintf("%s (%d)", name, len(v.m.Tensors))
		}
		labels[i] = label
		if w := lipgloss.Width(label) + 2; w > width { // +2 是「▸ 」或「  」
			width = w
		}
	}

	var sb strings.Builder
	for i, label := range labels {
		var line string
		if i == v.cursor {
			line = styleSelected.Render("▸ " + label)
		} else {
			line = "  " + label
		}
		// 补齐到列宽。用 lipgloss.Width 算显示宽度而不是 len() ——
		// 一个汉字 3 字节却只占 2 列，按字节补齐反而会错位。
		if pad := width - lipgloss.Width(line); pad > 0 {
			line += strings.Repeat(" ", pad)
		}
		sb.WriteString(line + "\n")
	}
	return sb.String(), width
}

func (v ModelView) body(width, height int) string {
	switch section(v.cursor) {
	case sectionOverview:
		return v.overview()
	case sectionMetadata:
		return v.metadata(height)
	case sectionTensors:
		return v.tensors()
	case sectionQuantDist:
		return quantDistText(v.m)
	default:
		return v.refHint()
	}
}

func (v ModelView) overview() string {
	m := v.m
	var sb strings.Builder
	fmt.Fprintf(&sb, "文件     %s\n", m.Path)
	fmt.Fprintf(&sb, "格式     %s %s\n", m.Format, m.Version)
	fmt.Fprintf(&sb, "大小     %s\n", humanize.Bytes(m.FileSize))
	fmt.Fprintf(&sb, "架构     %s\n", m.Arch)
	fmt.Fprintf(&sb, "张量     %d 个\n", len(m.Tensors))
	fmt.Fprintf(&sb, "总参数   %s（%s）\n",
		humanize.Count(m.TotalParams()), humanize.Comma(m.TotalParams()))
	fmt.Fprintf(&sb, "张量占用 %s", humanize.Bytes(m.TensorBytes()))

	// **存储字节与张量字节和是两回事**（权重绑定会让前者更小），
	// 差值要说出来 —— 否则用户看到两个不一样的数会以为哪边算错了
	if m.StorageBytes > 0 && m.StorageBytes != m.TensorBytes() {
		fmt.Fprintf(&sb, "，去重后仅占 %s（差值 %s 是共享的存储）",
			humanize.Bytes(m.StorageBytes),
			humanize.Bytes(m.TensorBytes()-m.StorageBytes))
	}
	sb.WriteString("\n")

	sb.WriteString(v.warnings())
	return sb.String()
}

// 告警最多显示这么多条，其余折叠成一行。
//
// **理由不是"会把界面顶穿"**：根视图的 padTo 会兜住行数，标题栏与帮助栏
// 不会被挤掉 —— 真正的后果是告警被**静默截到十几条**，用户不知道自己
// 漏看了多少（实测：把上限改成 1000，144 条告警仍然只渲染 24 行）。
// 所以上限本身是对的，但它要配一句"还有 N 条"，否则就是静默丢信息。
//
// 144 这个数是实测的：gpt-oss:20b 在补 MXFP4 之前正好 144 条（72 张量 × 2）。
const maxWarningsShown = 8

func (v ModelView) warnings() string {
	n := len(v.m.Warnings)
	if n == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("\n" + styleWarn.Render(fmt.Sprintf("告警 %d 条", n)) + "\n")
	for i, w := range v.m.Warnings {
		if i >= maxWarningsShown {
			sb.WriteString(styleWarn.Render(fmt.Sprintf("  还有 %d 条（用 --json 看全部）",
				n-maxWarningsShown)) + "\n")
			break
		}
		sb.WriteString(styleWarn.Render("  "+w) + "\n")
	}
	return sb.String()
}

// metadata 渲染元数据列表，**并在可解释的值旁边挂上速查表关联**。
//
// 关联是**现算**的（拿 key 调 ref.LookupKey），不是解析时写死字段 ——
// 写死的那种一旦速查表换了 ID 就全体指空，而且已经在 --json 里对外过一遍。
// metadata 渲染元数据列表。
//
// **一屏放不下时明说漏了多少**：④b-1 里这一栏还没有内容光标，
// 滚动要等 ④b-2 跟张量列表一起做。静默只显示前 N 条的话，
// 用户会以为这就是全部 —— 而"文件里看到的每个键都能查到"
// 正是这张表存在的理由（实测 gemma4:26b 有 52 条，一屏只放得下 26 条）。
func (v ModelView) metadata(height int) string {
	var sb strings.Builder
	sb.WriteString(styleSection.Render(fmt.Sprintf("元数据（%d 条）", len(v.m.Metadata))) + "\n\n")

	// 标题 2 行（含空行）+ 可能的汇总 1 行
	cap := height - 3
	if cap < 1 {
		cap = 1
	}
	shown, hidden := v.m.Metadata, 0
	if len(shown) > cap {
		hidden = len(shown) - cap
		shown = shown[:cap]
	}

	for _, kv := range shown {
		sb.WriteString(styleField.Render(kv.Key) + "  " + kv.Value)

		// ① "值本身就是个码"的，把码翻译出来（file_type 是典型）
		if e, ok := fileTypeEntry(kv); ok {
			sb.WriteString("  " + styleLink.Render("["+e.Title+"] ◂"))
			// **上游猜出来的档位要标出来**：它可能与该文件实际的张量类型
			// 不符，而用户会拿它当事实（ref 包里的原话）。
			// 不标的话，"文件里写的"与"工具猜的"在界面上长得一样。
			if code, ok := fileTypeCode(kv); ok && ref.FileTypeGuessed(code) {
				sb.WriteString(styleWarn.Render(" (上游猜的，未必与实际张量类型相符)"))
			}
		}

		// ② 键本身在速查表里有条目的，标一个可跳转的记号。
		//
		// **一个键最多一个记号**：file_type 那条已经挂了值释义，
		// 再挂一个键记号的话同一行会出现两个一模一样的 `◂`（实测），
		// 而它们指向两个不同的条目 —— ④b-2 接 Enter 跳转时
		// "跳哪一个"就没有答案。
		if _, hasValueRef := fileTypeEntry(kv); !hasValueRef {
			if _, ok := ref.LookupKey(kv.Key); ok {
				sb.WriteString("  " + styleLink.Render("◂"))
			}
		}
		sb.WriteString("\n")
	}
	if hidden > 0 {
		sb.WriteString(styleHint.Render(fmt.Sprintf(
			"…还有 %d 条没显示（滚动在 ④b-2 里做；现在可以用 modelview --json 看全部）",
			hidden)) + "\n")
	}
	return sb.String()
}

// fileTypeEntry 把 general.file_type 的值翻译成速查表条目。
//
// 单独一个函数是因为里面有三个边界：值可能不是整数、
// 可能是废弃编号、可能带 GUESSED 标志位（1024|15 这种）。
func fileTypeEntry(kv model.MetaKV) (ref.Entry, bool) {
	code, ok := fileTypeCode(kv)
	if !ok {
		return ref.Entry{}, false
	}
	return ref.FileTypeByCode(code)
}

// fileTypeCode 取出 general.file_type 的原始码（**不剥 GUESSED 标志位**）。
//
// 与 fileTypeEntry 分开是因为调用方有时需要原始值：
// FileTypeByCode 内部会把 1024 悄悄剥掉，剥掉之后就没法判断
// "这个档位是不是上游猜的了"。
//
// Raw 的类型随解析器而变：GGUF 的 uint32 是最常见的一种。
func fileTypeCode(kv model.MetaKV) (uint32, bool) {
	if kv.Key != "general.file_type" {
		return 0, false
	}
	switch t := kv.Raw.(type) {
	case uint32:
		return t, true
	case uint64:
		return uint32(t), true
	case int:
		return uint32(t), true
	default:
		return 0, false
	}
}

// tensors 是"张量"栏的右栏内容：**只是摘要**，完整列表按 Enter 进子视图。
//
// 不把列表直接铺在右栏：那条列表有自己的光标、过滤与滚动，
// 挤在右栏里既窄又要和左栏的 ↑↓ 抢按键。
func (v ModelView) tensors() string {
	var sb strings.Builder
	sb.WriteString(styleSection.Render(fmt.Sprintf("张量（%d 个）", len(v.m.Tensors))) + "\n\n")

	// **空模型要先挡住**：下面用 sum.largest 的两个字段，
	// 一个张量都没有时它是 nil —— 直接取就是解引用 panic，
	// 而"读到一个没有张量的文件"完全可能（parser 不保证非空）
	if len(v.m.Tensors) == 0 {
		sb.WriteString(styleHint.Render("这个文件里没有张量") + "\n")
		return sb.String()
	}

	sum := summarizeTensors(v.m)
	fmt.Fprintf(&sb, "总参数   %s\n", humanize.Count(v.m.TotalParams()))
	fmt.Fprintf(&sb, "类型     %s\n", sum.dtypeLine())
	fmt.Fprintf(&sb, "最大的   %s（%s）\n",
		humanize.Truncate(sum.largest.Name, 40), humanize.Bytes(sum.largest.ByteSize))
	sb.WriteString(styleHint.Render(fmt.Sprintf(
		"\n按 %s 打开完整列表（可按名字过滤）", keyEnter)) + "\n")
	return sb.String()
}

// summarizeTensors 汇总张量列表。
//
// **最大的那个要挑出来**：用户点进"张量"栏想知道的第一件事是
// "这个模型的钱花在哪了"，而答案就是这个。
// 这里不逐条列前 N 个 —— 那是完整列表的活，抄一份在右栏里，
// 两处的排序一旦不同就是两个互相矛盾的"前 5 个"。
func summarizeTensors(m *model.Model) tensorSummary {
	s := tensorSummary{}
	for _, tn := range m.Tensors {
		if tn.ByteSize > s.largestBytes {
			s.largest, s.largestBytes = tn, tn.ByteSize
		}
	}
	// **必须排序**：DtypeHistogram 返回的是 map，遍历顺序随机 ——
	// 不排的话同一屏每次刷新（任何一个按键都会触发重绘）
	// 类型的先后顺序都在变，看起来就是字在抖。
	// 按个数降序、同数按类型名升序：前者是用户关心的，
	// 后者只为让结果唯一（否则个数相同的两项顺序仍不确定）。
	for d, n := range m.DtypeHistogram() {
		s.dtypes = append(s.dtypes, dtypeCount{d: d, n: n})
	}
	sort.Slice(s.dtypes, func(i, j int) bool {
		if s.dtypes[i].n != s.dtypes[j].n {
			return s.dtypes[i].n > s.dtypes[j].n
		}
		return s.dtypes[i].d < s.dtypes[j].d
	})
	return s
}

type dtypeCount struct {
	d model.Dtype
	n int
}

type tensorSummary struct {
	dtypes       []dtypeCount
	largest      *model.Tensor
	largestBytes int64
}

// dtypeLine 拼一行类型分布。
func (s tensorSummary) dtypeLine() string {
	parts := make([]string, 0, len(s.dtypes))
	for _, dc := range s.dtypes {
		parts = append(parts, fmt.Sprintf("%s ×%d", dc.d, dc.n))
	}
	return strings.Join(parts, "  ")
}

// refHint 是"速查表"栏的右栏内容。
//
// 与张量栏同一个道理：249 条是一份要翻的列表，塞在右栏里
// 既窄又要和左栏抢按键，所以进子视图。
func (v ModelView) refHint() string {
	var sb strings.Builder
	sb.WriteString(styleSection.Render("速查表") + "\n\n")
	for _, tb := range ref.Tables() {
		fmt.Fprintf(&sb, "  %-24s %3d 条\n", tb.Title, len(tb.Entries))
	}
	sb.WriteString(styleHint.Render(fmt.Sprintf(
		"\n按 %s 打开（可跨表搜索）", keyEnter)) + "\n")
	return sb.String()
}

// joinHorizontal 把左栏与右栏拼起来，右栏各行对齐到左栏右侧。
//
// 用 strings.Builder 而不是 lipgloss.JoinHorizontal：后者按**行数**
// 对齐两侧，左栏只有几行而右栏几十行时，它会在左栏那侧补一堆空格 ——
// 那些空格会一路留到行尾，在终端里就是一片空白块。
//
// **左栏用完之后的行要补等宽的空格**（而不是什么都不写）：
// 不补的话，右栏比左栏长的那几行会从第 0 列开始 ——
// 实测概览里的"总参数 / 张量占用"两行就是这么错位的。
func joinHorizontal(left, right string) string {
	// **两边都要 TrimRight**：nav 与 body 都以 "\n" 结尾，
	// 不裁的话会多出一个空行 —— 而那个空行会挤掉真正的内容
	// （实测：metadata 自己算好了 12 行，join 出来是 13 行，
	// 根视图再按高度截一刀，用户看到的是"还有 3 行没显示"，
	// 而那句提示本该是"还有 54 条"）
	ll := strings.Split(strings.TrimRight(left, "\n"), "\n")
	rl := strings.Split(strings.TrimRight(right, "\n"), "\n")

	// 左栏的显示宽度（不是字节数）
	leftWidth := 0
	for _, l := range ll {
		if w := lipgloss.Width(l); w > leftWidth {
			leftWidth = w
		}
	}

	n := len(ll)
	if len(rl) > n {
		n = len(rl)
	}
	// **末尾不留换行**：留了的话按 "\n" 切会多出一个空元素，
	// 而根视图的 padTo 就是这么数的 —— 多出来的那一行会把视图
	// 自己算好的提示（"还有 N 条没显示"）挤掉，换成 padTo 的
	// "还有 N 行没显示"。两个提示都说得通，但用户看到的是措辞更差的那个。
	lines := make([]string, 0, n)
	for i := range n {
		var sb strings.Builder
		if i < len(ll) {
			sb.WriteString(ll[i])
			if pad := leftWidth - lipgloss.Width(ll[i]); pad > 0 {
				sb.WriteString(strings.Repeat(" ", pad))
			}
		} else {
			sb.WriteString(strings.Repeat(" ", leftWidth))
		}
		sb.WriteString(" ")
		if i < len(rl) {
			sb.WriteString(rl[i])
		}
		lines = append(lines, sb.String())
	}
	return strings.Join(lines, "\n")
}

// displayName 是标题栏里显示的名字：优先用模型库给的名字，
// 没有才退回文件名。
func (v ModelView) displayName() string {
	if v.name != "" {
		return v.name
	}
	return baseName(v.m.Path)
}

func baseName(p string) string {
	if i := strings.LastIndex(p, "/"); i >= 0 {
		return p[i+1:]
	}
	return p
}
