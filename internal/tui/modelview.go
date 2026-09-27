package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

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
type ModelView struct {
	m           *model.Model
	pendingPath string // 非空表示还没解析
	name        string // 给用户看的名字，空则退回文件名
	err         error
	section     section
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
		}
		v.section = section(v.cursor)
	}
	return v, nil
}

func (v ModelView) Help() []string {
	return []string{
		keyUp + " " + keyDown + " 切换栏目",
		keyEsc + " 返回",
		keyQuit + " 退出",
	}
}

func (v ModelView) View(width, height int) string {
	if v.err != nil {
		return styleWarn.Render("读取失败：" + v.err.Error())
	}
	if v.m == nil {
		return styleHint.Render("正在解析…")
	}
	// 左栏固定 16 列，其余留给内容。
	//
	// 这里**不做截断**：根视图的 padTo 已经按终端宽度统一截过了，
	// 各层再截一遍就会出现"同一行在两层里算出的宽度不一样"。
	return joinHorizontal(v.nav(), v.body())
}

func (v ModelView) nav() string {
	var sb strings.Builder
	for i, name := range sections {
		label := name
		switch section(i) {
		case sectionMetadata:
			label = fmt.Sprintf("%s (%d)", name, len(v.m.Metadata))
		case sectionTensors:
			label = fmt.Sprintf("%s (%d)", name, len(v.m.Tensors))
		}
		if i == v.cursor {
			sb.WriteString(styleSelected.Render("▸ " + label))
		} else {
			sb.WriteString("  " + label)
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func (v ModelView) body() string {
	var out string
	switch v.section {
	case sectionOverview:
		out = v.overview()
	case sectionMetadata:
		out = v.metadata()
	case sectionTensors:
		out = v.tensors()
	case sectionQuantDist:
		out = v.quantDist()
	case sectionRef:
		out = styleHint.Render("按 ? 打开速查表")
	}
	return out
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
// **不设上限会把界面顶穿**：实测 gpt-oss:20b 在补 MXFP4 之前有 144 条告警，
// 全打出来的话标题栏和帮助栏都会被挤出屏幕。
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
func (v ModelView) metadata() string {
	var sb strings.Builder
	sb.WriteString(styleSection.Render(fmt.Sprintf("元数据（%d 条）", len(v.m.Metadata))) + "\n\n")

	for _, kv := range v.m.Metadata {
		sb.WriteString(styleField.Render(kv.Key) + "  " + kv.Value)

		// ① "值本身就是个码"的，把码翻译出来（file_type 是典型）
		if e, ok := fileTypeEntry(kv); ok {
			sb.WriteString("  " + styleLink.Render("["+e.Title+"] ◂"))
		}

		// ② 键本身在速查表里有条目的，标一个可跳转的记号
		if _, ok := ref.LookupKey(kv.Key); ok {
			sb.WriteString("  " + styleLink.Render("◂"))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// fileTypeEntry 把 general.file_type 的值翻译成速查表条目。
//
// 单独一个函数是因为里面有三个边界：值可能不是整数、
// 可能是废弃编号、可能带 GUESSED 标志位（1024|15 这种）。
func fileTypeEntry(kv model.MetaKV) (ref.Entry, bool) {
	if kv.Key != "general.file_type" {
		return ref.Entry{}, false
	}
	// Raw 的类型随解析器而变：GGUF 的 uint32 是最常见的一种
	var code uint32
	switch t := kv.Raw.(type) {
	case uint32:
		code = t
	case uint64:
		code = uint32(t)
	case int:
		code = uint32(t)
	default:
		return ref.Entry{}, false
	}
	e, ok := ref.FileTypeByCode(code)
	return e, ok
}

// tensors / quantDist 是 ④b-2 的内容。
//
// **明写"还没做"而不是留空白面板**：空白会让用户以为是加载失败，
// 而"还没做"至少是诚实的。左栏仍然列出这几项 ——
// 导航结构现在就定下来，④b-2 只填内容、不用再动骨架。
func (v ModelView) tensors() string {
	return styleHint.Render(fmt.Sprintf(
		"%d 个张量。列表与详情在计划 ④b-2 里实现；"+
			"现在可以用 modelview --json 看到全部张量", len(v.m.Tensors)))
}

func (v ModelView) quantDist() string {
	return styleHint.Render("量化分布在计划 ④b-2 里实现")
}

// joinHorizontal 把左栏与右栏拼起来。
//
// 用 strings.Builder 而不是 lipgloss.JoinHorizontal：后者按**行数**
// 对齐两侧，左栏只有几行而右栏几十行时，它会在左栏那侧补一堆空格 ——
// 那些空格会让"每行不超过宽度"的断言失败，也会在终端里留下空白块。
func joinHorizontal(left, right string) string {
	ll := strings.Split(left, "\n")
	rl := strings.Split(right, "\n")
	n := len(ll)
	if len(rl) > n {
		n = len(rl)
	}
	var sb strings.Builder
	for i := range n {
		if i < len(ll) {
			sb.WriteString(ll[i])
		}
		sb.WriteString(" ")
		if i < len(rl) {
			sb.WriteString(rl[i])
		}
		sb.WriteString("\n")
	}
	return sb.String()
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
