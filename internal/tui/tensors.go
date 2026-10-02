package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/humanize"
	"github.com/sillydong/modelview/internal/model"
)

// TensorsView 是张量的完整列表。
//
// **没有"已加载"状态，也没有 loadedMsg**：m.Tensors 已经在内存里，
// 列表是同步可得的（异步加载的是张量**详情**里的统计，那是 Task 5）。
// 造一个只为对称而存在的消息类型，会让每个读到它的人多想一次
// "这里到底等什么"。
type TensorsView struct {
	m      *model.Model
	cursor int

	filtering bool   // 正在输入过滤词
	filter    string // 当前过滤词（空表示不过滤）

	// failed 是**统计失败**的张量名集合，由 ModelView 传进来（它是
	// 唯一知道哪张扫失败了的视图）。spec §10 要求"该项标红"。
	//
	// 用名字而不是下标：两边看到的是同一份 m.Tensors，但过滤/排序之后
	// 下标会错位 —— 而标错行比不标更糟。
	failed map[string]bool

	// dtype 非空时只显示这个类型的张量。
	//
	// 与 filter **互相独立**（不是二选一）：从"量化分布"跳过来时
	// 是按类型筛，而用户在列表里还能再按名字筛一遍，
	// 两个条件同时生效才符合直觉。
	dtype model.Dtype

	// seg 非空时只显示**名字拆段后含这一段**的张量。
	//
	// 与 filter **不是一回事**：那个是用户在列表里打的子串，
	// 这个是条目页跳过来时说的"段"（判据见 nameHasSegmentEntry）。
	// 与 dtype 一样只由构造函数给，用户改不了。
	seg string
}

func NewTensorsView(m *model.Model) TensorsView {
	return TensorsView{m: m}
}

// NewTensorsViewDtype 造一个只显示某个类型的张量列表。
//
// 与 NewTensorsView 分开而不是加参数：绝大多数调用点是"看全部"，
// 多加一个参数会让每个调用点都要想一下"这里该传什么类型"。
// 两个构造函数的差别恰好就是这里要说的事。
func NewTensorsViewDtype(m *model.Model, d model.Dtype) TensorsView {
	return TensorsView{m: m, dtype: d}
}

// NewTensorsViewName 造一个只显示名字里含关键词的张量列表。
func NewTensorsViewName(m *model.Model, keyword string) TensorsView {
	return TensorsView{m: m, filter: keyword}
}

// NewTensorsViewSegment 造一个只显示「名字里含有这一段」的张量列表。
//
// **与 NewTensorsViewName 不是一回事**：那个按**子串**（`strings.Contains`），
// 这个按**段**（`ref.SplitTensorName` + `ref.LookupTensorSegment`，与
// `segmentOccurrences` 数数用的是同一个判据）。
//
// 为什么必须有它：段条目那一节屏幕上是"72 个张量的名字含「attn_q」"，
// 按下去若走子串筛，两者在 `#N` 上会分叉成 **432 vs 0** ——
// **屏幕印一句数，按下去得到另一份结果**。
// 按段筛之后两边是同一个判据，**不可能不一致**（不是靠测试对齐，是靠构造）。
func NewTensorsViewSegment(m *model.Model, seg string) TensorsView {
	return TensorsView{m: m, seg: seg}
}

func (v TensorsView) Title() string {
	switch {
	case v.dtype != "" && v.filter != "":
		return fmt.Sprintf("张量 · %s · %s + %q（%d 个）",
			v.m.DisplayName(), v.dtype, v.filter, len(v.shown()))
	case v.dtype != "":
		return fmt.Sprintf("张量 · %s · %s（%d 个）",
			v.m.DisplayName(), v.dtype, len(v.shown()))
	case v.seg != "" && v.filter != "":
		return fmt.Sprintf("张量 · %s · 段 %s + %q（%d 个）",
			v.m.DisplayName(), v.seg, v.filter, len(v.shown()))
	case v.seg != "":
		return fmt.Sprintf("张量 · %s · 段 %s（%d 个）",
			v.m.DisplayName(), v.seg, len(v.shown()))
	case v.filter != "":
		return fmt.Sprintf("张量 · %s（过滤：%s）", v.m.DisplayName(), v.filter)
	}
	return fmt.Sprintf("张量 · %s（共 %d 个）", v.m.DisplayName(), len(v.m.Tensors))
}

func (v TensorsView) Init() tea.Cmd { return nil }

// Modal 让根视图把按键全部交给本视图 —— 只在**正在输入**过滤词的时候。
//
// **接收者必须是值，不能是指针**（`func (v *TensorsView) Modal() bool`）：
// 视图以值入栈（`pushCmd(NewTensorsView(m))`），指针接收者会让 TensorsView
// 不再实现 View —— pushCmd 那行直接编不过。加固之前这条路是**静默**的：
// 断言失败 → 非模态 → 用户在过滤框里敲 q 直接退出程序，
// 而所有单元测试全绿（它们直接调 Update，绕过了根 Model）。
// Task 2 的探针实测过这条。
//
// **已确认的过滤态（`filter != "" && !filtering`）刻意「不是」模态的**：
// 那样 q 才能退出程序。代价是清过滤词不能用 Esc（那会被根视图
// 拦成"返回上一层"），只能用 Backspace —— 表头的提示照这个写。
func (v TensorsView) Modal() bool { return v.filtering }

// shown 返回当前该显示的张量下标。
//
// **现算而不是存字段**：存一份 []*model.Tensor 的话，Task 5 的详情页扫完
// 统计之后列表里还是旧的那份（指针相同也没用 —— 列表要显示的是
// "这个张量扫过没有"，那是会变的）。每次现算的代价是 O(n) 次字符串比较，
// 434 个张量下可以忽略。
//
// 按段筛那一支要贵一档（每个张量每个段都要查一次速查表），
// 但它只在真的按段筛时才走 —— 空字段那一次 `v.seg != ""` 就短路了，
// 看全部这条最常走的路上不付这份钱。
func (v TensorsView) shown() []int {
	match := func(tn *model.Tensor) bool {
		if v.dtype != "" && tn.Dtype != v.dtype {
			return false
		}
		if v.seg != "" && !nameHasSegmentEntry(tn.Name, "tensor:"+v.seg) {
			return false
		}
		return v.filter == "" || strings.Contains(strings.ToLower(tn.Name), strings.ToLower(v.filter))
	}
	// **快路径的条件是"三个都没筛"，判据少一个就是静默失效**：
	// 漏掉哪一维，那一维的筛就会走这条捷径直接返回全部 ——
	// 列表显示全部，而用户以为自己筛过了。
	// 三个都要判，因为三个筛是**互相独立**的（见字段注释）。
	if v.filter == "" && v.dtype == "" && v.seg == "" {
		idx := make([]int, len(v.m.Tensors))
		for i := range idx {
			idx[i] = i
		}
		return idx
	}
	var idx []int
	for i, tn := range v.m.Tensors {
		if match(tn) {
			idx = append(idx, i)
		}
	}
	return idx
}

func (v TensorsView) Update(msg tea.Msg) (View, tea.Cmd) {
	keyMsg, isKey := msg.(tea.KeyMsg)
	if v.filtering && isKey {
		return v.updateFiltering(keyMsg)
	}
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			if v.cursor > 0 {
				v.cursor--
			}
		case "down", "j":
			if v.cursor < len(v.shown())-1 {
				v.cursor++
			}
		case "/":
			v.filtering = true
		// **这里曾经有一条 "backspace" 分支**（已确认过滤态下删一个字）。
		// spec §3 把 Backspace 定为全局的"返回上一层"之后它不可达了：
		// 确认过滤态不是模态的，根视图在模态判定之后先接走这个键并弹栈。
		// 留着它就是一段只被测试调用的死代码（本仓在审计里专门点过
		// 这类分支）。删掉之后清过滤词的两条路是：`/` 回输入态再删、
		// 或 `/` 之后 Esc 整个清掉。
		//
		// **没有非过滤态的 "esc" 分支**同理：未过滤时 Esc 走根视图的
		// "弹栈"，永远到不了这里。
		case "enter":
			if v.canOpen() {
				// `shown()` 返回的是**过滤后的下标**，所以 idx[cursor]
				// 正是屏幕上选中的那一个 —— 这也正是 shown() 存下标
				// 而不是存张量副本的理由。
				idx := v.shown()
				tn := v.m.Tensors[idx[v.cursor]]
				return v, pushCmd(NewTensorView(v.m, tn))
			}
		}
	}
	return v, nil
}

// updateFiltering 处理过滤态下的按键。
//
// 走到这里的前提是 Modal() 为真（根视图见它就把按键全交过来），
// 所以 q、Esc 都会到这儿 —— 用户想筛 "qwen"，敲 q 不该退出程序。
//
// **这里没有 ctrl+c 分支，是刻意的**：根视图在模态判定**之前**就处理了
// Ctrl+C，所以那句分支永远不会被执行。留着它不只是多余 ——
// 它会让下一个读代码的人以为"退出的保证在这里"，
// 于是把根视图那条删掉。保证只能有一个地方写。
func (v TensorsView) updateFiltering(msg tea.KeyMsg) (View, tea.Cmd) {
	switch msg.String() {
	case "enter":
		// 确认过滤：退出输入态但**保留过滤词**，光标归零
		v.filtering = false
		v.cursor = 0
		return v, nil
	case "esc":
		// 取消过滤：词也一起清掉
		v.filtering, v.filter, v.cursor = false, "", 0
		return v, nil
	case "backspace":
		if v.filter != "" {
			v.filter = backspaceOne(v.filter)
			v.cursor = 0
		}
		return v, nil
	}
	// **判据是 `len(msg.Runes) > 0`，不是 `msg.Type == tea.KeyRunes`。**
	//
	// bubbletea v1.3.10 把**单独一个空格**设成 `Type: KeySpace`，
	// 但 `Runes` 仍然是 `[' ']`（key.go:698-701）。按 Type 判的话
	// 空格会被静默丢掉 —— 张量名里没有空格所以这一版看不出来，
	// 但速查表与元数据的过滤会踩到（Task 7/9）。
	//
	// 这个判据同时兜住三件事：可打印字符、空格、多字符粘贴
	//（粘贴的 Type 是 KeyRunes 且 Runes 有多字符，整段追加是对的），
	// 并且天然排除 Enter/Esc/Tab 这些（它们的 Runes 为空）。
	if len(msg.Runes) > 0 {
		v.filter += string(msg.Runes)
		v.cursor = 0
	}
	return v, nil
}

// canOpen 表示 Enter 现在真的有得开 —— **帮助栏与 Update 读同一个判据**。
//
// 按名字/类型/段筛出空列表、或者模型压根没有张量时，Enter 什么都不做；
// 帮助栏那时还列"Enter 详情"就是在骗用户按（与 `ModelView.opensOnEnter`
// 按当前状态判定是同一条规矩）。
func (v TensorsView) canOpen() bool { return len(v.shown()) > 0 }

func (v TensorsView) Help() []string {
	if v.filtering {
		return []string{"输入过滤词", keyEnter + " 确认", keyEsc + " 取消", "Ctrl+C 退出"}
	}
	bindings := []string{keyUp + " " + keyDown + " 移动"}
	if v.canOpen() {
		// **接上了就必须列**：keys.go 那条"只列当前视图真的支持的键"
		// 是两个方向 —— 不支持的不能列（骗用户按），支持的不能漏
		//（用户不知道有这条路）。详情页在这里接的线。
		bindings = append(bindings, keyEnter+" 详情")
	}
	return append(bindings, keyFilter+" 过滤", keyHelp+" 速查表", keyEsc+" 返回", keyQuit+" 退出")
}

func (v TensorsView) View(width, height int) string {
	idx := v.shown()
	if len(idx) == 0 {
		if v.filter != "" {
			// **两个态的措辞必须分开**：还在输入时（filtering）Esc 走的是
			// "取消过滤"那条分支；已确认之后 Modal() 为假，Esc 会被根视图
			// 拦成"弹掉整个列表" —— 那时写"Esc 取消过滤"就是在说假话，
			// 用户按下去整个列表不见了，而屏幕上说它会取消过滤。
			// 与表头那句是同一条坑（updateFiltering 的注释里点过名），
			// 表头早就分成两句了，这里原先漏了。
			if v.filtering {
				return styleHint.Render(fmt.Sprintf(
					"没有匹配 %q 的张量（共 %d 个）。按 %s 取消过滤",
					v.filter, len(v.m.Tensors), keyEsc))
			}
			return styleHint.Render(fmt.Sprintf(
				"没有匹配 %q 的张量（共 %d 个）。按 %s 删字，%s 返回",
				v.filter, len(v.m.Tensors), keyBackspace, keyEsc))
		}
		// 按类型/按段筛出来的空列表**不能说"这个模型没有张量"** ——
		// 模型里有没有张量，与有没有这一类的张量是两回事
		if v.dtype != "" {
			return styleHint.Render(fmt.Sprintf("没有 %s 类型的张量（共 %d 个）",
				v.dtype, len(v.m.Tensors)))
		}
		if v.seg != "" {
			return styleHint.Render(fmt.Sprintf("没有名字里含「%s」这一段的张量（共 %d 个）",
				v.seg, len(v.m.Tensors)))
		}
		return styleHint.Render("这个模型没有张量")
	}

	var sb strings.Builder
	// 标题行（或过滤输入行）占 1 行
	//
	// **已确认过滤态不能写"Esc 取消"**：那时 `Modal()` 为假，
	// Esc 会被根视图拦下**弹掉整个列表**，永远到不了清过滤那条分支 ——
	// 写上就是一条照着按没反应的假提示（而且是"返回上一层"这种
	// 恰好看起来像生效了的行为，用户分不清是清掉了还是退出了）。
	// 清字的真实入口是 Backspace（输入态删一个字符、已确认态也删一个字符），
	// 提示必须照着这个写。
	head := fmt.Sprintf("张量（%d/%d）", len(idx), len(v.m.Tensors))
	switch {
	case v.filtering:
		head = fmt.Sprintf("过滤：%s▏  匹配 %d/%d",
			v.filter, len(idx), len(v.m.Tensors))
	case v.dtype != "" && v.filter != "":
		// **两个筛同时生效时两个都要写出来**：只写一个的话，
		// 用户看到的条数与他知道的那个筛不符，会以为工具算错了
		head = fmt.Sprintf("类型 %s · 过滤 %q：%d/%d（%s 删字，%s 返回）",
			v.dtype, v.filter, len(idx), len(v.m.Tensors), keyBackspace, keyEsc)
	case v.dtype != "":
		head = fmt.Sprintf("类型 %s：%d/%d", v.dtype, len(idx), len(v.m.Tensors))
	case v.seg != "" && v.filter != "":
		head = fmt.Sprintf("段 %s · 过滤 %q：%d/%d（%s 删字，%s 返回）",
			v.seg, v.filter, len(idx), len(v.m.Tensors), keyBackspace, keyEsc)
	case v.seg != "":
		head = fmt.Sprintf("段 %s：%d/%d", v.seg, len(idx), len(v.m.Tensors))
	case v.filter != "":
		head = fmt.Sprintf("过滤 %q：%d/%d（%s 删字，%s 返回）",
			v.filter, len(idx), len(v.m.Tensors), keyBackspace, keyEsc)
	}
	sb.WriteString(styleSection.Render(humanize.Truncate(head, width)) + "\n")

	// **名字列按实际宽度缩水**：80 列的终端里名字+形状+类型+大小
	// 一共要 ~85 列，硬编码的 44 会让每一行都被 padTo 从右边切掉 ——
	// 切掉的是最右边的大小那一列，而用户正需要它判断"这个大不大"。
	nameWidth := width - 45
	if nameWidth < 16 {
		nameWidth = 16
	}
	if nameWidth > 60 {
		nameWidth = 60
	}

	listCap := height - 1
	if listCap < 1 {
		listCap = 1
	}
	// 窗口与"要不要打范围提示"都交给 listWindow（先扣提示行、再算窗口，
	// 且只有真放得下才打）—— 那三条理由（砍掉选中项、提示的范围与屏幕
	// 不一致、极矮终端硬塞）写在那里，不在这里重抄一遍。
	start, end, showHint := listWindow(len(idx), v.cursor, listCap)
	lines := make([]string, 0, end-start+1)
	for i := start; i < end; i++ {
		lines = append(lines, v.row(i, v.m.Tensors[idx[i]], nameWidth))
	}
	if showHint {
		lines = append(lines, styleDim.Render(fmt.Sprintf(
			"…显示第 %d–%d 个，共 %d 个", start+1, end, len(idx))))
	}
	sb.WriteString(strings.Join(lines, "\n"))
	return sb.String()
}

// row 渲染一行张量。
//
// 列：名字（截断到 nameWidth）· 形状 · 类型 · 大小。**不显示统计量** ——
// 那是详情页的事，而且绝大多数张量还没扫过（Stats == nil）。
//
// 名字列用字节截断而不是按显示宽度：见 humanize.Truncate 的说明
// （中文名那列会偏宽，这是不引依赖的代价）。
func (v TensorsView) row(i int, tn *model.Tensor, nameWidth int) string {
	// 标记区是**两列、各管一件事**：第一列光标、第二列失败。
	//
	// 原来写的是 `else if`（光标优先，失败标记让位），实测发现问题：
	// 只有一张张量、而它恰好是失败的那张时，标记永远看不到 —— 而
	// 单张量模型里那张**必然**在光标下（列表刚打开时光标在第 0 行）。
	//
	// **失败标记是字符不是颜色**：lipgloss 在非终端（测试、管道）下
	// 不出颜色，只靠 styleWarn 的话这一行与普通行**逐字节相同** ——
	// 测试断言不出来，色盲用户与重定向到文件的场景也看不出来。
	cur, bad := " ", " "
	if i == v.cursor {
		cur = "▸"
	}
	if v.failed[tn.Name] {
		bad = "⚠"
	}
	marker := cur + bad
	line := fmt.Sprintf("%s%-*s %-18s %-7s %10s", marker,
		nameWidth, humanize.Truncate(tn.Name, nameWidth),
		humanize.Dims(tn.Dims), string(tn.Dtype), humanize.Bytes(tn.ByteSize))
	if i == v.cursor {
		return styleSelected.Render(line)
	}
	// **统计失败的那一张要标出来**（spec §10：「该项标红」）。
	// 只看进度行里那句"失败 N"的话，用户知道有几张失败却不知道是哪张，
	// 得逐张点开才知道。
	//
	// 选中态优先：光标停在这一行时它已经足够醒目，再叠一层警示色
	// 反而看不出选了哪个。
	if v.failed[tn.Name] {
		return styleWarn.Render(line)
	}
	return line
}
