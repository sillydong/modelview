package tui

import (
	"context"
	"fmt"
	"sort"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/discover"
	"github.com/sillydong/modelview/internal/humanize"
)

// libraryLoadedMsg 是扫描完成的消息。
type libraryLoadedMsg struct{ res discover.Result }

// itemFilledMsg 是某个条目的格式/参数量补齐了。
type itemFilledMsg struct {
	index int
	item  discover.Item
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

func (l Library) Update(msg tea.Msg) (View, tea.Cmd) {
	switch msg := msg.(type) {
	case libraryLoadedMsg:
		l.res = msg.res
		l.items = msg.res.Items
		sortItems(l.items)
		l.loaded = true
		l.cursor = 0
		l.filled = 0
		// 每个条目一条独立命令：先出来的先显示，
		// 一条大模型的 Fill 卡住不会拖住其它条目
		cmds := make([]tea.Cmd, 0, len(l.items))
		for i := range l.items {
			cmds = append(cmds, l.fillCmd(i))
		}
		return l, tea.Batch(cmds...)

	case itemFilledMsg:
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
			if len(l.items) > 0 {
				it := l.items[l.cursor]
				return l, func() tea.Msg {
					return pushMsg{v: NewModelViewFromPath(it.Path, it.Name)}
				}
			}
		}
	}
	return l, nil
}

// fillCmd 造一条"补这个条目的格式与参数量"的命令。
//
// **复制一份再传进去**：Fill 是就地修改的，
// 直接把切片元素的地址交给另一个 goroutine，
// 而主线程同时在读它 —— 那是数据竞争。
func (l Library) fillCmd(i int) tea.Cmd {
	it := l.items[i]
	fill := l.fill
	return func() tea.Msg {
		fill(&it)
		return itemFilledMsg{index: i, item: it}
	}
}

func (l Library) Help() []string {
	return []string{
		keyUp + " " + keyDown + " 移动",
		keyEnter + " 查看",
		keyRescan + " 重扫",
		keyQuit + " 退出",
	}
}

func (l Library) View(width, height int) string {
	if !l.loaded {
		return styleHint.Render("正在扫描模型目录…")
	}
	if len(l.items) == 0 {
		return l.emptyView()
	}

	var sb strings.Builder
	for i, it := range l.items {
		sb.WriteString(l.row(i, it) + "\n")
	}

	// **未完成的下载单独说，且明说它不是可回收的。**
	//
	// ollama 是下完之后才写 manifest 的，所以"第一次 pull 下到一半"时
	// 这些文件不被任何模型引用 —— 曾经被算进"可回收 12.85 GiB"，
	// 而那正是用户下到一半的模型。照着删就毁了自己的下载。
	if n := len(l.res.InProgress); n > 0 {
		sb.WriteString("\n" + styleWarn.Render(fmt.Sprintf(
			"未完成的下载 %d 个（合计 %s）—— 不是可回收空间，删了会毁掉下载",
			n, humanize.Bytes(totalBytes(l.res.InProgress)))) + "\n")
	}
	if n := len(l.res.Orphans); n > 0 {
		sb.WriteString("\n" + styleSection.Render(fmt.Sprintf(
			"孤儿 blob %d 个（可回收 %s）", n,
			humanize.Bytes(totalBytes(l.res.Orphans)))) + "\n")
		for _, o := range l.res.Orphans {
			sb.WriteString(styleDim.Render(fmt.Sprintf("  %-64s %10s",
				humanize.Truncate(o.Name, 64), humanize.Bytes(o.Size))) + "\n")
		}
	}
	if n := len(l.res.Errs); n > 0 {
		sb.WriteString("\n" + styleWarn.Render(strings.Join(l.res.Errs, "\n")) + "\n")
	}

	if l.filled < len(l.items) {
		sb.WriteString("\n" + styleHint.Render(fmt.Sprintf(
			"正在读取格式与参数量… %d/%d", l.filled, len(l.items))))
	}
	return sb.String()
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
		return styleWarn.Render(fmt.Sprintf("%s%-28s %-11s %10s  ⚠%s",
			marker, humanize.Truncate(it.Name, 28), it.Source,
			humanize.Bytes(it.Size), it.Err))
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

func (l Library) emptyView() string {
	var sb strings.Builder
	sb.WriteString("没有发现模型文件。\n\n")
	sb.WriteString("扫过这些目录（不存在的会被跳过）：\n")
	for _, p := range discover.Paths() {
		sb.WriteString(styleDim.Render(fmt.Sprintf("  %-12s %s", p.Source, p.Dir)) + "\n")
	}
	return sb.String()
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
