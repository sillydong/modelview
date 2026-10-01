package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/sillydong/modelview/internal/humanize"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/ref"
)

// refFocus 是速查表视图里"↑↓ 现在归谁"。
type refFocus int

const (
	focusTables refFocus = iota
	focusEntries
)

// RefView 是速查表视图：左栏表、右栏条目。
type RefView struct {
	// m 是**当前正在看的模型**，跟着往下传给 EntryView。
	//
	// 它可以为 nil（单测里不给模型）：为 nil 时条目详情不显示
	// "在本模型中"那一节 —— 没有模型上下文时那一节无法计算，
	// 显示成空的话用户会以为"这个模型里没有"，而事实是"不知道"。
	//
	// **必须往下传而不是让 EntryView 自己去拿**：EntryView 是从
	// RefView 或 ModelView 推出来的，只有它们知道当前看的是哪个模型。
	m *model.Model

	// tables 做成字段而不是每次调 ref.Tables()：测试要用一张空表
	// 去验"空表不崩"，而内置的四张表都是非空的 ——
	// 不注入的话那条分支永远走不到。
	tables []ref.Table

	tableCursor int
	entryCursor int
	focus       refFocus

	// searching 表示正在输入搜索词；search 是已输入的词。
	searching bool
	search    string
}

// NewRefView 造一个速查表视图。m 可以为 nil（见 RefView.m 的说明）。
func NewRefView(m *model.Model) RefView {
	return RefView{m: m, tables: ref.Tables()}
}

func (v RefView) Title() string {
	if v.search != "" || v.searching {
		return fmt.Sprintf("modelview · 速查表 · 搜索 %q（%d 条）",
			v.search, len(v.entries()))
	}
	return fmt.Sprintf("modelview · 速查表 · %s（%d 条）",
		v.table().Title, len(v.entries()))
}

func (v RefView) Init() tea.Cmd { return nil }

// Modal 让搜索输入能拿到 q（见 TensorsView.Modal 的说明）。
func (v RefView) Modal() bool { return v.searching }

// table 返回当前选中的那张表；`tables` 为空时返回零值。
//
// **越界判据只有这一处**：原先 entries() 挡了、Title() 直接下标 ——
// 同一个前提（tables 可能为空）被两个读者读出两个结论，
// 而只有一个是对的（另一个真为空时 panic）。
func (v RefView) table() ref.Table {
	if v.tableCursor < 0 || v.tableCursor >= len(v.tables) {
		return ref.Table{}
	}
	return v.tables[v.tableCursor]
}

// entries 返回当前该显示的条目。
//
// **搜索时是跨表的**：ref.Find 搜的是全文，结果天然跨表 ——
// 只搜当前这张表的话，用户搜一个明明存在的键却被告知没有，
// 而他无从知道要先去左栏换一张表。
func (v RefView) entries() []ref.Entry {
	if v.search != "" {
		return ref.Find(v.search)
	}
	return v.table().Entries
}

// canOpen 表示 Enter 现在真的有得开 —— **与 Update 里那条判断同一句**：
// 各写一遍的话，会出现"帮助栏列了 Enter，按下去没反应"。
//
// 搜不到匹配是真会出现的（搜索是正常的操作，不是错误状态），
// `entryCursor` 那一半则是因为空列表下光标没有意义。
func (v RefView) canOpen() bool {
	es := v.entries()
	return len(es) > 0 && v.entryCursor < len(es)
}

func (v RefView) Update(msg tea.Msg) (View, tea.Cmd) {
	key, isKey := msg.(tea.KeyMsg)
	if !isKey {
		return v, nil
	}
	if v.searching {
		return v.updateSearching(key)
	}
	switch key.String() {
	case "tab":
		// 搜索态下没有"表"这一栏可切 —— 钉在条目上
		if v.search == "" {
			if v.focus == focusTables {
				v.focus = focusEntries
			} else {
				v.focus = focusTables
			}
		}
	case "/":
		// **保留已有的搜索词**：多打一个字收窄比重新敲一遍省事，
		// 与 TensorsView 的 `/` 同一个约定
		v.searching = true
		v.focus = focusEntries
	case "up", "k", "down", "j":
		v = v.move(key.String())
	case "backspace":
		// **已确认的搜索词也要能删字** —— 与 `TensorsView` 同一条路。
		//
		// 两个都是"列表 + 一个过滤词"的视图：同一个键在一处能删、
		// 在另一处静默失效，用户在一边学会的动作到另一边只会以为是自己按错了。
		// （原先只有"按 / 回输入态再删"这一条路，而屏幕上没说这件事 ——
		// 空结果那句提示原先只写"按 / 改搜索词"。）
		//
		// 删到空就是回到"当前表"那一态：右栏内容整个换了一份，
		// 条目光标必须跟着归零 —— 停在旧下标上要么越界，要么静默指到另一条。
		if v.search != "" {
			v.search = v.search[:len(v.search)-1]
			v.entryCursor = 0
		}
	// **没有"清搜索"的 esc 分支**：搜索态才是模态的，
	// 确认之后的 Esc 走根视图的"弹栈"，到不了这里 ——
	// 写了就是死代码（Task 2 修的就是同类死角）。
	// Esc 直接返回上一层，要改搜索词按 `/`。
	case "enter":
		es := v.entries()
		if v.canOpen() {
			return v, pushCmd(NewEntryView(v.m, es[v.entryCursor]))
		}
	}
	return v, nil
}

// move 把 ↑↓ 交给当前有焦点的那个光标，**返回改过的新视图**。
//
// 必须返回新值而不是就地改：接收者是值（视图在栈里也是值，
// 见 TensorsView.Modal 那条说明）。丢掉返回值的 `v.move(...)`
// 改的是副本 —— 界面照常重绘，光标却一动不动，
// 而"按了有反应但没反应"是最难从屏幕上看出原因的一种。
//
// 换表时**条目光标归零**：从 118 条的 keys 切到 21 条的 floats，
// 停在 100 的位置会越界 —— 而越界的那一下要么 panic，
// 要么（如果加了钳制）静默指到另一条上，用户按 Enter 打开的是别的东西。
func (v RefView) move(key string) RefView {
	up := key == "up" || key == "k"
	switch {
	case v.search != "" || v.focus == focusEntries:
		n := len(v.entries())
		if up {
			if v.entryCursor > 0 {
				v.entryCursor--
			}
		} else if v.entryCursor < n-1 {
			v.entryCursor++
		}
	case v.focus == focusTables && len(v.tables) > 0:
		if up {
			if v.tableCursor > 0 {
				v.tableCursor--
				v.entryCursor = 0
			}
		} else if v.tableCursor < len(v.tables)-1 {
			v.tableCursor++
			v.entryCursor = 0
		}
	}
	return v
}

// updateSearching 处理搜索输入态。
//
// 与 TensorsView.updateFiltering 同一套规则：Enter 确认（保留词）、
// Esc 取消（清词）、Backspace 退格、其余可打印字符进词。
// **没有 ctrl+c 分支** —— 根视图在模态判定之前就处理了它。
func (v RefView) updateSearching(msg tea.KeyMsg) (View, tea.Cmd) {
	switch msg.String() {
	case "enter":
		v.searching = false
		v.entryCursor = 0
	case "esc":
		v.searching, v.search, v.entryCursor = false, "", 0
	case "backspace":
		if v.search != "" {
			v.search = v.search[:len(v.search)-1]
			v.entryCursor = 0
		}
	default:
		// **判据是 `len(msg.Runes) > 0`，不是 `msg.Type == tea.KeyRunes`** ——
		// 与 TensorsView.updateFiltering 是同一个坑：单独一个空格被
		// bubbletea 设成 `Type: KeySpace`，但 `Runes` 仍然是 `[' ']`
		//（key.go:698-701），按 Type 判会把它静默丢掉 ——
		// 速查表里搜 "Q4 K" 就永远打不出那个空格。
		// 这个判据同时接住整段粘贴（粘贴的 `String()` 是 "[qwen]"，
		// 按 String() 相等判断同样接不住），并天然排除 Enter/Esc/Tab
		//（它们的 Runes 为空）。
		if len(msg.Runes) > 0 {
			v.search += string(msg.Runes)
			v.entryCursor = 0
		}
	}
	return v, nil
}

func (v RefView) Help() []string {
	if v.searching {
		return []string{"输入关键词", keyEnter + " 确认", keyEsc + " 取消", "Ctrl+C 退出"}
	}
	// 搜索态没有"切换栏" —— 左栏只有一行搜索结果
	bindings := []string{keyUp + " " + keyDown + " 移动"}
	if v.search == "" {
		bindings = append(bindings, keyTab+" 切换栏")
	}
	if v.canOpen() {
		bindings = append(bindings, keyEnter+" 详情")
	}
	if v.search != "" {
		return append(bindings, keyFilter+" 改搜索词", keyEsc+" 返回", keyQuit+" 退出")
	}
	return append(bindings, keyFilter+" 搜索", keyEsc+" 返回", keyQuit+" 退出")
}

func (v RefView) View(width, height int) string {
	nav, navWidth := v.nav(height)
	// **右栏宽度夹到 0 以上**：终端比左栏还窄时 `width-navWidth-1` 是负数，
	// 而搜索态那行会把它交给 humanize.Truncate —— 那个函数对负数是
	// `s[:cut]` 切片越界 panic，不是空串（实测：宽 0/1/10 三档全 panic）。
	// 0 宽在真终端里可达：窗口拖到 17 列以下，或 pty 报 0×0
	//（drive_tui 的文件头就写着默认 winsize 是 0×0）。
	body := v.body(max(width-navWidth-1, 0), height)
	return joinHorizontal(nav, body)
}

// nav 渲染左栏，返回内容与它的**显示宽度**。
//
// 不返回"占了几行"：右栏的高度是它自己那一块，与左栏行数无关 ——
// joinHorizontal 会给短的那一侧补等宽空行（见它的说明）。
// ④b-1 就是在这里踩过坑：让两侧共用一套行数口径，
// 结果一边算 12 行、拼出来 13 行，根视图再截一刀，
// 用户看到的是措辞更差的那个提示。
func (v RefView) nav(height int) (string, int) {
	labels := v.navLabels()
	width := 0
	for _, l := range labels {
		// +2 是「▸ 」或「  」两列
		if w := lipgloss.Width(l) + 2; w > width {
			width = w
		}
	}

	focusHere := v.focus == focusTables && v.search == ""
	start, end := window(len(labels), v.tableCursor, max(height-1, 1))

	lines := make([]string, 0, end-start)
	for i := start; i < end; i++ {
		var line string
		switch {
		case focusHere && i == v.tableCursor:
			line = styleSelected.Render("▸ " + labels[i])
		case i == v.tableCursor:
			// 不是焦点栏但仍是当前项：给一个弱记号 ——
			// 不标的话用户不知道右栏为什么是这张表
			line = styleDim.Render("▸ " + labels[i])
		default:
			line = "  " + labels[i]
		}
		if pad := width - lipgloss.Width(line); pad > 0 {
			line += strings.Repeat(" ", pad)
		}
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n"), width
}

// navLabels 是左栏每一行的文字。
//
// 搜索时塌成一行：搜索结果是跨表的，此时"选中的是第几张表"没有意义，
// 留着那张表的计数反而让用户以为自己只搜了一张。
func (v RefView) navLabels() []string {
	if v.search != "" || v.searching {
		return []string{fmt.Sprintf("搜索结果 (%d)", len(v.entries()))}
	}
	labels := make([]string, 0, len(v.tables))
	for _, tb := range v.tables {
		labels = append(labels, fmt.Sprintf("%s (%d)", tb.Title, len(tb.Entries)))
	}
	return labels
}

func (v RefView) body(width, height int) string {
	// 输入行占一行，右栏就少显示一条
	if v.searching {
		head := styleSection.Render(humanize.Truncate("搜索："+v.search+"▏", width))
		return head + "\n" + v.entryLines(width, height-1)
	}
	return v.entryLines(width, height)
}

func (v RefView) entryLines(width, listCap int) string {
	es := v.entries()
	if len(es) == 0 {
		switch {
		case v.search != "":
			// **提示里不写"按 Esc 清搜索"** —— Esc 在这一态是"返回上一层"，
			// 照着写的话用户按完发现回到了模型页，而搜索词也没了，
			// 分不清是"清掉了"还是"退出去了"
			//
			// 两个入口都写出来（与 `TensorsView` 的空结果提示同一份措辞）：
			// 只用 `/` 改词的话，用户得先回输入态才知道能删字
			return styleHint.Render(fmt.Sprintf(
				"没有匹配 %q 的条目（共 %d 条）。按 %s 删字，%s 改搜索词",
				v.search, totalEntries(v.tables), keyBackspace, keyFilter))
		case v.searching:
			return styleHint.Render("输入关键词，跨全部速查表搜索")
		default:
			return styleHint.Render("这张表没有条目")
		}
	}
	if listCap < 1 {
		listCap = 1
	}

	focusHere := v.search != "" || v.focus == focusEntries
	// **要写"显示第 N–M 条"就先把它那一行从容量里扣掉，再算窗口。**
	//
	// 反过来做（先按 listCap 取窗口、再砍掉一行放提示）砍掉的是窗口的
	// 最后一条 —— 而光标停在最后一条时，被砍掉的正是用户正选着的那一条。
	// 屏幕上看起来只是"少了一条"，根本看不出少的是选中的那条：
	// 用户按 Enter 打开的会是一个自己没看见的条目。
	// 这条是计划里没写到的，`TestRefView_选中项始终可见` 直接红了。
	rows := listCap
	if len(es) > rows && rows > 1 {
		rows--
	}
	start, end := window(len(es), v.entryCursor, rows)
	lines := make([]string, 0, end-start+1)
	for i := start; i < end; i++ {
		title := humanize.Truncate(es[i].Title, max(width-4, 8))
		if focusHere && i == v.entryCursor {
			lines = append(lines, styleSelected.Render("▸ "+title))
			continue
		}
		lines = append(lines, "  "+title)
	}
	// "显示第 N–M 条"要算进高度里，否则它会被根视图的 padTo 顶掉，
	// 换成一句信息量更差的"…还有 1 行没显示"
	if (start > 0 || end < len(es)) && len(lines) < listCap {
		lines = append(lines, styleDim.Render(fmt.Sprintf(
			"…显示第 %d–%d 条，共 %d 条", start+1, end, len(es))))
	}
	return strings.Join(lines, "\n")
}

func totalEntries(tables []ref.Table) int {
	n := 0
	for _, tb := range tables {
		n += len(tb.Entries)
	}
	return n
}
