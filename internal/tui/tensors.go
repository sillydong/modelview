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
}

func NewTensorsView(m *model.Model) TensorsView {
	return TensorsView{m: m}
}

func (v TensorsView) Title() string {
	if v.filter != "" {
		return fmt.Sprintf("张量 · %s（过滤：%s）", baseName(v.m.Path), v.filter)
	}
	return fmt.Sprintf("张量 · %s（共 %d 个）", baseName(v.m.Path), len(v.m.Tensors))
}

func (v TensorsView) Init() tea.Cmd { return nil }

// Modal 让根视图把按键全部交给本视图 —— 只在**正在输入**过滤词的时候。
//
// **接收者必须是值，不能是指针**（`func (v *TensorsView) Modal() bool`）：
// 视图以值入栈（`pushCmd(NewTensorsView(m))`），指针接收者会让
// `m.stack[i].(modalView)` 断言失败 → 静默变成非模态 →
// 用户在过滤框里敲 q 直接退出程序，而所有单元测试全绿
// （它们直接调 Update，绕过了根 Model）。Task 2 的探针实测过这条。
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
func (v TensorsView) shown() []int {
	if v.filter == "" {
		idx := make([]int, len(v.m.Tensors))
		for i := range idx {
			idx[i] = i
		}
		return idx
	}
	kw := strings.ToLower(v.filter)
	var idx []int
	for i, tn := range v.m.Tensors {
		if strings.Contains(strings.ToLower(tn.Name), kw) {
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
		case "backspace":
			// **已确认的过滤态下也要能删字** —— 否则清掉过滤词只有
			// "按 / 回输入态再删"这一条路，而屏幕上没有任何地方说这件事。
			//
			// 这里删的是**已确认的过滤词**，与输入态那条 backspace
			// 改的是同一个字段，所以两条路的行为看着一样、用户不用分。
			if v.filter != "" {
				v.filter = v.filter[:len(v.filter)-1]
				v.cursor = 0
			}
			// **没有非过滤态的 "esc" 分支**：过滤态才是模态的，所以
			// 未过滤时 Esc 会走根视图那条"弹栈"，永远到不了这里 ——
			// 写了也是死代码（这一版修的就是同类死角，见 Task 2）。
			// 清字的入口是 backspace，不是 Esc。
		case "enter":
			// `shown()` 返回的是**过滤后的下标**，所以 idx[cursor]
			// 正是屏幕上选中的那一个 —— 这也正是 shown() 存下标
			// 而不是存张量副本的理由。
			idx := v.shown()
			if len(idx) > 0 {
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
			v.filter = v.filter[:len(v.filter)-1]
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

func (v TensorsView) Help() []string {
	if v.filtering {
		return []string{"输入过滤词", keyEnter + " 确认", keyEsc + " 取消", "Ctrl+C 退出"}
	}
	return []string{
		keyUp + " " + keyDown + " 移动",
		// **接上了就必须列**：keys.go 那条"只列当前视图真的支持的键"
		// 是两个方向 —— 不支持的不能列（骗用户按），支持的不能漏
		//（用户不知道有这条路）。详情页在这里接的线。
		keyEnter + " 详情",
		keyFilter + " 过滤",
		keyEsc + " 返回",
		keyQuit + " 退出",
	}
}

func (v TensorsView) View(width, height int) string {
	idx := v.shown()
	if len(idx) == 0 {
		if v.filter != "" {
			return styleHint.Render(fmt.Sprintf(
				"没有匹配 %q 的张量（共 %d 个）。按 %s 取消过滤",
				v.filter, len(v.m.Tensors), keyEsc))
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
	start, end := window(len(idx), v.cursor, listCap)
	lines := make([]string, 0, end-start+1)
	for i := start; i < end; i++ {
		lines = append(lines, v.row(i, v.m.Tensors[idx[i]], nameWidth))
	}
	// **"显示第 N–M 个"要算进高度里**：不预留的话最后一行会被
	// 根视图的 padTo 顶掉，换上一句"…还有 1 行没显示" ——
	// 两个提示都说得通，但用户看到的是信息量更差的那个。
	if start > 0 || end < len(idx) {
		if len(lines) >= listCap {
			lines = lines[:listCap-1]
		}
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
	marker := "  "
	if i == v.cursor {
		marker = "▸ "
	}
	line := fmt.Sprintf("%s%-*s %-18s %-7s %10s", marker,
		nameWidth, humanize.Truncate(tn.Name, nameWidth),
		humanize.Dims(tn.Dims), string(tn.Dtype), humanize.Bytes(tn.ByteSize))
	if i == v.cursor {
		return styleSelected.Render(line)
	}
	return line
}
