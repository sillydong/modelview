package tui

import (
	"context"
	"fmt"
	"sort"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/sillydong/modelview/internal/discover"
	"github.com/sillydong/modelview/internal/humanize"
)

// libraryLoadedMsg 是扫描完成的消息。
type libraryLoadedMsg struct{ res discover.Result }

// itemFilledMsg 是某个条目的格式/参数量补齐了。
//
// **必须带世代号**：Fill 要读文件头部，在途几百毫秒；
// 这期间用户按 r 重扫的话，旧结果回来时会覆盖新列表 ——
// 界面上会出现一个这次根本没扫到的模型，而且不报任何错。
// 实测复现过（两轮 scan + 在途 fill），见 TestLibrary_重扫丢弃在途的Fill结果。
type itemFilledMsg struct {
	index int
	item  discover.Item
	gen   int
}

// Library 是本机模型库视图。
//
// **scan / fill 是字段而不是直接调 discover**：测试注入假的实现，
// 界面就不用碰真实磁盘 —— 否则这条测试的结论会随开发机上
// 装了什么模型而变，而"本机只有 1/12 条路径存在"的机器上
// 它还会静默地什么都没验到。
type Library struct {
	scan func(context.Context, discover.Options) discover.Result
	fill func(*discover.Item) *discover.Item

	res    discover.Result
	items  []discover.Item
	cursor int
	loaded bool
	filled int

	// gen 是扫描世代号：每次 Scan 完成就自增。
	//
	// 在途的 Fill 命令出发时捕获当时的 gen，回来时不匹配就整个丢弃。
	// 没有它的话，重扫之后旧结果会按 index 覆盖新列表 ——
	// 显示的模型与磁盘上的对不上，而且没有任何提示。
	gen int
}

// NewLibrary 造一个模型库视图，用真实的 discover。
func NewLibrary() Library {
	return Library{
		scan: discover.Scan,
		fill: discover.Fill,
	}
}

func (l Library) Title() string { return "modelview · 模型库" }

// Init 只做 Scan（只 stat，实测 1.76 ms，瞬时返回），Fill 由后续命令逐个补。
//
// 一次读完再显示的话，HF 缓存里几百个文件会让用户盯着空屏几十秒。
func (l Library) Init() tea.Cmd {
	scan := l.scan
	return func() tea.Msg {
		return libraryLoadedMsg{res: scan(context.Background(), discover.Options{})}
	}
}

// Modal 恒为 false：模型库没有输入框，q 与 Esc 照旧归根视图。
func (l Library) Modal() bool { return false }

func (l Library) Update(msg tea.Msg) (View, tea.Cmd) {
	switch msg := msg.(type) {
	case libraryLoadedMsg:
		l.res = msg.res
		l.items = msg.res.Items
		sortItems(l.items)
		l.loaded = true
		// **这三行重置与 gen 自增缺一不可**：
		//   - cursor 归零不只是"回到顶部"，它还兼着防越界 ——
		//     从 30 条的列表重扫到 2 条之后，旧光标会让 Enter 索引越界
		//   - gen 自增让上一轮在途的 Fill 全部失效
		l.cursor = 0
		l.filled = 0
		l.gen++
		// 每个条目一条独立命令：先出来的先显示，
		// 一条大模型的 Fill 卡住不会拖住其它条目。
		//
		// **已知边界（没实测，属于推断）**：这里是无上限扇出 ——
		// HF 缓存那种几百个文件的场景下，会同时打开几百个文件
		// （每条 Fill 都要 parser.Parse 读头部），而 macOS 的 fd 软上限
		// 是 256。本机只有 5 个模型，碰不到。
		// 真要改的话是加一个容量固定的信号量（8–16），不是分批 ——
		// 分批会让"先出来的先显示"退化成"一批一批地显示"。
		cmds := make([]tea.Cmd, 0, len(l.items))
		for i := range l.items {
			cmds = append(cmds, l.fillCmd(i))
		}
		return l, tea.Batch(cmds...)

	case itemFilledMsg:
		// **先看世代**：过期的结果直接丢，不要碰 items 与 filled。
		// 顺序很重要 —— 写在写 items 之后就等于没写。
		if msg.gen != l.gen {
			return l, nil
		}
		if msg.index >= 0 && msg.index < len(l.items) {
			l.items[msg.index] = msg.item
		}
		l.filled++
		return l, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if l.cursor > 0 {
				l.cursor--
			}
		case "down", "j":
			if l.cursor < len(l.items)-1 {
				l.cursor++
			}
		case "r":
			l.loaded = false
			l.filled = 0
			return l, l.Init()
		case "enter":
			if l.canOpen() {
				it := l.items[l.cursor]
				return l, pushCmd(NewModelViewFromPath(it.Path, it.Name))
			}
		}
	}
	return l, nil
}

// canOpen 表示 Enter 现在真的有得看 —— **帮助栏与 Update 读同一个判据**。
//
// 两个条件都要：
//   - 列表为空（扫过的地方一个模型都没有）：没有可索引的条目，
//     而 cursor 会一直压在 0 上；
//   - `!loaded`：屏幕上是"正在扫描模型目录…"，列表**不在屏幕上** ——
//     重扫那一下列表字段还是上一轮那份，Enter 那时打开的是一个
//     用户看不见的条目（本仓为这条形状踩过好几次，见 View 里那段窗口说明）。
//
// 两处列/不列、开/不开都走这一个函数，所以不存在"帮助栏说能开、按下去没反应"
// 或者反过来"没列却按得开"。
func (l Library) canOpen() bool { return l.loaded && len(l.items) > 0 }

// fillCmd 造一条"补这个条目的格式与参数量"的命令。
//
// **复制一份再传进去**：Fill 是就地修改的，
// 直接把切片元素的地址交给另一个 goroutine，
// 而主线程同时在读它 —— 那是数据竞争。
func (l Library) fillCmd(i int) tea.Cmd {
	it := l.items[i]
	fill := l.fill
	gen := l.gen
	return func() tea.Msg {
		fill(&it)
		return itemFilledMsg{index: i, item: it, gen: gen}
	}
}

func (l Library) Help() []string {
	bindings := []string{keyUp + " " + keyDown + " 移动"}
	if l.canOpen() {
		bindings = append(bindings, keyEnter+" 查看")
	}
	return append(bindings, keyRescan+" 重扫", keyQuit+" 退出")
}

func (l Library) View(width, height int) string {
	if !l.loaded {
		return styleHint.Render("正在扫描模型目录…")
	}

	var sb strings.Builder

	// **安全提示放在列表之前**，与 runScan 的先后顺序一致。
	//
	// 放在列表之后的话，模型一多它们就落到屏幕外了 ——
	// 而"未完成的下载不是可回收空间"是全工具唯一一条会引导**破坏性操作**
	// 的提示（照着删会毁掉用户正在下的模型），它恰恰不能因为
	// 模型多就消失。实测：30 个模型、80×24 的终端里，
	// 原先把提示放在列表之后时它完全不可见。
	//
	// **空库也要走这一段**（所以这里不再有 `len(items)==0` 的早退）：
	// 那是同一件事的**另一个方向** —— 提示不能因为模型**多**而消失，
	// 同样不能因为模型**零**而消失。而且零这一头更隐蔽：空库看着像个
	// 干净状态，用户不会怀疑自己漏看了什么（孤儿 blob 与未完成的下载
	// 恰恰是"一个模型都没扫到、但磁盘上并不干净"时才会出现的组合）。
	head := l.notices(width)
	headLines := 0
	if head != "" {
		headLines = strings.Count(head, "\n")
		sb.WriteString(head)
	}

	// 空库：没有列表可滚，只有一段说明 —— 它照样要接 height，
	// 已用掉的提示行由 headLines 传进去扣（理由见 emptyView）
	if len(l.items) == 0 {
		sb.WriteString(l.emptyView(headLines, height))
		return sb.String()
	}

	// 给列表留的行数：总高 − 提示 − **进度行（如果有）**。
	//
	// Library 是唯一有**两个**可选尾行的视图（范围提示 + 在途进度），
	// 所以这一个 capacity 必须把进度行先让出来，范围提示那一行由
	// listWindow 自己再扣 —— 原先只按一个扣减、判断里又少了
	// "这一行放得下吗"那半句，矮终端下两行会一起被 padTo 顶掉，
	// 屏幕上换成"…还有 2 行没显示"（实测 30 个模型、终端高 7）。
	footerLines := 0
	if l.filled < len(l.items) {
		footerLines = 1
	}
	listCap := height - headLines - footerLines
	if listCap < 1 {
		listCap = 1
	}

	// 窗口与"要不要打范围提示"都交给 listWindow（先扣提示行、再算窗口，
	// 且只有真放得下才打）—— 理由写在它那里，不在这里重抄。
	start, end, showHint := listWindow(len(l.items), l.cursor, listCap)

	// **自己拼行、最后 Join，末尾不留换行** —— 这是 joinHorizontal 里
	// 记过的那条：留了的话 padTo 按 "\n" 切会多出一个空元素，
	// 多出来的那一行会把视图自己算好的提示挤掉，换成措辞更差的
	// "…还有 N 行没显示"（而且数字还大一）。Library 是**启动后的第一屏**，
	// 模型一多就撞上（HF 缓存那种几百个文件的场景）。
	lines := make([]string, 0, end-start+2)
	for i := start; i < end; i++ {
		lines = append(lines, l.row(i, l.items[i]))
	}
	if showHint {
		lines = append(lines, styleDim.Render(fmt.Sprintf(
			"  …共 %d 个，显示第 %d–%d 个", len(l.items), start+1, end)))
	}

	if l.filled < len(l.items) {
		lines = append(lines, styleHint.Render(fmt.Sprintf(
			"正在读取格式与参数量… %d/%d", l.filled, len(l.items))))
	}
	sb.WriteString(strings.Join(lines, "\n"))
	return sb.String()
}

// notices 是列表之前那一小段提示（未完成下载 / 孤儿 / 目录警告）。
//
// 单独一个函数是因为它必须在 View 里**先于**列表渲染：
// 见 View 里那段关于"破坏性提示不能因为模型多就消失"的说明。
func (l Library) notices(width int) string {
	var sb strings.Builder

	if n := len(l.res.InProgress); n > 0 {
		sb.WriteString(styleWarn.Render(fmt.Sprintf(
			"未完成的下载 %d 个（合计 %s）—— 不是可回收空间，删了会毁掉下载",
			n, humanize.Bytes(totalBytes(l.res.InProgress)))) + "\n")
	}
	if n := len(l.res.Orphans); n > 0 {
		sb.WriteString(styleSection.Render(fmt.Sprintf(
			"孤儿 blob %d 个（可回收 %s）", n,
			humanize.Bytes(totalBytes(l.res.Orphans)))) + "\n")
		for _, o := range l.res.Orphans {
			sb.WriteString(styleDim.Render(fmt.Sprintf("  %-64s %10s",
				humanize.Truncate(o.Name, 64), humanize.Bytes(o.Size))) + "\n")
		}
	}
	// 目录警告**要换行而不是让它被截断**：唯一一条破坏性操作警告的
	// 关键半句（"报成「可回收」可能让你删掉真实模型"）就在末尾，
	// 截掉它是把这句安全提示变成一句废话
	for _, e := range l.res.Errs {
		sb.WriteString(styleWarn.Render(wrapText(e, width)) + "\n")
	}
	return sb.String()
}

// wrapText 按**显示宽度**折行（不是按字节数）。
//
// 中文每字 3 字节却占 2 列 —— 按字节折会折出很短的宽窄不一的段落。
func wrapText(s string, width int) string {
	if width <= 0 || lipgloss.Width(s) <= width {
		return s
	}
	var out []string
	var cur strings.Builder
	curW := 0
	for _, r := range s {
		rw := lipgloss.Width(string(r))
		if curW+rw > width {
			out = append(out, cur.String())
			cur.Reset()
			curW = 0
		}
		cur.WriteRune(r)
		curW += rw
	}
	if cur.Len() > 0 {
		out = append(out, cur.String())
	}
	return strings.Join(out, "\n")
}

// row 渲染一行模型。
//
// 三段按"先看得见的、后读出来的"排列：名字与大小是 Scan 就有的，
// 格式/架构/参数量要 Fill 之后才有 —— 后者没读到时**不占位**，
// 免得出现一串空白让人以为读失败了。
func (l Library) row(i int, it discover.Item) string {
	marker := "  "
	if i == l.cursor {
		marker = "▸ "
	}
	if it.Err != "" {
		// **只显示原因，不带路径**：路径通常比一行还长，
		// 会把原因整个挤出屏幕（实测 80 列下只剩 "⚠/tmp/xxx/"）
		return styleWarn.Render(fmt.Sprintf("%s%-28s %-11s %10s  ⚠%s",
			marker, humanize.Truncate(it.Name, 28), it.Source,
			humanize.Bytes(it.Size), discover.ErrReason(it)))
	}
	line := fmt.Sprintf("%s%-28s %-11s %10s", marker,
		humanize.Truncate(it.Name, 28), it.Source, humanize.Bytes(it.Size))
	if it.Format != "" {
		line += "  " + string(it.Format)
	}
	if it.Arch != "" {
		line += " · " + it.Arch
	}
	if it.Params > 0 {
		line += " · " + humanize.Count(it.Params) + " 个参数"
	}
	if i == l.cursor {
		return styleSelected.Render(line)
	}
	return line
}

// emptyView 是空模型库那一屏的**说明部分**（安全提示由 View 先拼好，
// 用的行数从 headLines 传进来 —— 它不能在这里自己再调一次 notices：
// 两处各拼一次的话，"空库有没有提示"就又变成两个决定）。
//
// **它必须自己接住 height**：说明是定长的，但路径清单每条一行，
// `discover.Paths()` 有几条就有几条 —— 目录多、终端矮时整屏放不下，
// 超出的行会被根视图的 padTo 砍掉，砍掉的正是路径清单的尾部
// （用户看这一屏就是为了知道去哪儿放模型）。所以按剩余行数裁剪，
// 并留一行说"还有几行没列出来"。
//
// **末尾不留换行**（`joinHorizontal` 记过的那条）：留了的话 padTo 按 "\n"
// 切会多算一行，内容放不下时"…还有 N 行没显示"里的 N 比实际丢掉的**多一**，
// 内容恰好等于高度时还会白白砍掉一行。空库是**首次运行就会撞上的第一屏**
// （本机没装模型时），而这一支此前没人看过：跨视图的守卫 `allViews()`
// 用的是非空的假模型库。这里用 `strings.Join` 拼，所以末尾**构造上**
// 不可能多出换行（原先逐行 append "\n" 再 TrimRight，靠的是裁，不是构造）。
func (l Library) emptyView(headLines, height int) string {
	// 这一屏自己能用的行数。**至少留 1 行**：提示本身就可能占满整个矮终端
	//（那时 height-headLines ≤ 0），而这一屏不能什么都不说 ——
	// 剩下的交给 padTo 兜底。它是唯一能砍到安全提示的东西，也正因如此
	// notices 不在这里裁（"提示不能消失"是硬规矩，宁可让 padTo 留下记号）。
	budget := height - headLines
	if budget < 1 {
		budget = 1
	}

	head := []string{"没有发现模型文件。", "", "扫过这些目录（不存在的会被跳过）："}
	paths := discover.Paths()
	total := len(head) + len(paths)

	lines := make([]string, 0, max(total, budget))
	lines = append(lines, head...)
	for _, p := range paths {
		lines = append(lines, styleDim.Render(fmt.Sprintf("  %-12s %s", p.Source, p.Dir)))
	}

	// 放不下时**先砍路径清单、最后才砍说明**：说明是这一屏的结论
	//（"没有发现模型文件"），路径只是"去哪儿放模型"的细节。反过来截的话，
	// 矮终端上会剩一屏警告加一句"还有 N 行没显示"，用户不知道工具到底
	// 扫到东西没有。提示行照例算进 budget（先留出提示行，否则被顶掉的
	// 是真实内容）；budget 只剩 1 行时那一行给结论，隐藏的清单交给 padTo。
	switch {
	case total <= budget:
		// 全放得下，什么都不用砍
	case budget > len(head):
		room := budget - len(head) - 1 // 给提示行让出一行
		lines = append(lines[:len(head)+room], styleDim.Render(fmt.Sprintf(
			"  …还有 %d 个扫描目录没列出来", len(paths)-room)))
	case budget > 1:
		lines = append(lines[:budget-1], styleDim.Render(fmt.Sprintf(
			"  …还有 %d 行没显示（共 %d 个扫描目录）", total-budget+1, len(paths))))
	default:
		lines = lines[:budget]
	}
	return strings.Join(lines, "\n")
}

func totalBytes(items []discover.Item) int64 {
	var n int64
	for _, it := range items {
		n += it.Size
	}
	return n
}

// sortItems 按名字排序，保证顺序稳定。
//
// 界面每次刷新时列表乱跳是不可用的 —— 而 Scan 内部的排序按**路径**，
// ollama 的路径是 blobs/sha256-xxx，用户看到的顺序会是随机的。
func sortItems(items []discover.Item) {
	sort.SliceStable(items, func(i, j int) bool { return items[i].Name < items[j].Name })
}

// displayWidth 按**显示宽度**算一行有多宽（中文算 2 列）。
//
// 不能用 len()：一个汉字 3 字节却只占 2 列，用字节数判断会把
// 本来不超宽的行判成超宽。
func displayWidth(s string) int {
	return lipgloss.Width(s)
}
