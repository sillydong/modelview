package tui

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/sillydong/modelview/internal/analyze"
	"github.com/sillydong/modelview/internal/humanize"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/ref"
)

// modelLoadedMsg 是解析完成的消息。
type modelLoadedMsg struct {
	m   *model.Model
	err error
}

// batchScannedMsg 是「扫描全部」里一张张量扫完的结果。
//
// 与 tensorScannedMsg **分开而不是加一个 index 字段复用**：那个是
// TensorView 的消息（它按**张量名**认领），这个是 ModelView 的
// （它按**位次**认领）—— 判据不同，而混在一起的表现是"一条消息被
// 另一个视图收下"，合并到一张不相干的张量上，不报任何错。
//
// m 与 idx 一起构成认领判据：指针相同才算同一个模型（用户退回模型库、
// 换一个模型再进来时，上一轮在途的消息会落在新 ModelView 上）；
// 位次要正好是这一刻在等的那一个（取消后重启会留下上一轮的残影，
// 它的位次对不上，被丢掉）。
//
// **判据认不出的那一类**：残影的位次与新链在等的那一个**恰好相同**
// （取消、重启，两边都在等第 0 张）。它会被收下 —— 同一张量、同一个
// 计算，内容与新一轮本来要算的逐字节相同，收下无害，所以没有为此加
// 世代号。**将来若把合并换成非幂等的动作**（写缓存、累加计数、
// 换采样上限），这里必须先加一个世代号，否则残影会被当成这一轮的结果。
type batchScannedMsg struct {
	m   *model.Model
	idx int
	err error

	// 下面是副本上算出来的结果，**合并只在 Update（主线程）里做**。
	stats *model.Stats
	quant *model.QuantInfo
	sims  []model.QuantSim
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

// focus 是 ↑↓ 现在归左栏还是右栏。
type focus int

const (
	focusNav focus = iota
	focusBody
)

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
	focus       focus
	metaCursor  int

	// parse 是文件解析入口，注入进来的。
	//
	// **不直接调 parser.Parse**：spec §4.0 要求 tui 只知道 model.Model、
	// 不知道底层格式。界面层真正"读文件"的地方只有这一处，把它的来源
	// 换成注入值之后，tui 就没有到 internal/parser 的**直接** import 了。
	//
	// **但传递边还在**：`go list -deps ./internal/tui` 仍然能看到三个
	// 解析器，路径是 tui → discover → parser（discover.Fill 要读文件头）。
	// 这条边没有切，也不该在这里切 —— discover 是界面消费的领域层，
	// "它内部怎么读文件"是它自己的事。spec §4.0 那句"只依赖 model.Model"
	// 按字面在 discover 这一层仍然不成立，这里如实记下，别把它读成
	// "已经做到了"。
	//
	// 由 Library 在推入本视图时附上（见 Library.WithParse）。
	parse func(path string) (*model.Model, error)

	// scan 是单张量分析，注入是为了测试不碰真实文件
	//（与 TensorView.scan 同一个理由）。真的那个要读几秒磁盘。
	scan func(context.Context, *model.Model, *model.Tensor) error

	// 下面三个是「扫描全部」（`a`）的状态。**放在 ModelView 里而不是
	// 根 Model**：扫的是这个模型的张量、进度行也画在这一屏上，
	// 而根视图连"现在是哪个模型"都不知道（那正是 ContextModel 的理由）。
	//
	// 「扫完了」与「取消了」都不单独存字段：
	//   - 扫完 ⇔ scanDone == len(m.Tensors)。取消**不可能**留下这个状态 ——
	//     走满那一帧 scanning 已经被置回 false 了（见 Update 那条分支），
	//     所以取消之后按 a 是重新开始，不是"接着取消"；
	//   - 取消之后进度行整行消失（用户自己的动作，不需要收尾信号），
	//     不用再存一个"取消过"的标志。
	scanning   bool
	scanDone   int
	scanFailed int
}

// NewModelView 用一个已解析的模型造视图（测试与"已经有 m"的场景用）。
func NewModelView(m *model.Model) ModelView {
	return ModelView{m: m, scan: analyze.One}
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
	return ModelView{pendingPath: path, name: name, scan: analyze.One}
}

// ContextModel 实现 modelProvider：根视图按 `?` 时靠它把"当前模型"
// 带进速查表。
//
// **解析还没回来时返回 nil 是对的**：RefView 对 nil 的契约是
// "不显示"在本模型中"那一节"，而那一刻的事实是"还不知道"，
// 不是"这个模型里没有"。
func (v ModelView) ContextModel() *model.Model { return v.m }

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
	name := v.name
	parse := v.parse
	return func() tea.Msg {
		if parse == nil {
			// **报错而不是崩**：nil 解引用会掀掉整个 TUI，而这只可能是
			// 接线漏了（Library 没附上 parse）。做成一条能显示的
			// 读取失败，用户至少看得到原因。
			return modelLoadedMsg{err: errors.New("tui: 解析入口未注入（Library.WithParse）")}
		}
		m, err := parse(path)
		if err == nil && m != nil {
			// **名字写在模型上，下游视图才看得到**：张量列表与条目页
			// 拿到的都是这个 *Model，各自再传一遍名字要改四个构造函数的
			// 签名（含 jump.go 那两条没有名字可传的路径）。
			// 不写的话张量列表标题打的是 blob 名（`sha256-5ee4f07c…`），
			// 与上一屏的模型名对不上。
			m.Name = name
		}
		return modelLoadedMsg{m: m, err: err}
	}
}

// Modal 恒为 false：模型概览不输入文字，q 与 Esc 照旧归根视图。
func (v ModelView) Modal() bool { return false }

func (v ModelView) Update(msg tea.Msg) (View, tea.Cmd) {
	switch msg := msg.(type) {
	case modelLoadedMsg:
		v.m, v.err = msg.m, msg.err
		return v, nil

	case batchScannedMsg:
		// **三条认领判据缺一条就是一种静默串数据**（理由见 batchScannedMsg）。
		// 丢弃不是错误：取消之后回来的、别的模型的、上一轮重启留下的
		// 残影都走这里 —— 它们本来就该被丢掉，且不该让链继续。
		if msg.m != v.m || !v.scanning || msg.idx != v.scanDone {
			return v, nil
		}
		if msg.err != nil {
			// **失败的那一张什么都不合并**，只记一个失败数。
			//
			// 这条链的副本是**派发时**拷的，而同一张 tn 上还有第二个写者
			//（用户点开详情页时 TensorView 的合并）。把一份旧副本的结果
			// 无条件写回去，就可能用一个"空结果"把用户刚算出来的真结果
			// 清掉 —— 两边都没有版本号，判不出新旧，那就**别写**。
			// 于是失败一律不落地：与 CLI 路径（analyzeOne 直接写真实的 tn，
			// 失败前算出来那半份会留下）刻意不同，差别在于那边没有第二份
			// 副本可用，也没有第二个写者。
			//
			// 不落地等于保持"还没算"：NeedsWork 仍为真，再按一次 `a` 会重试。
			// 失败**计数**要显示出来，否则用户不知道有张量没算出来
			//（进度行里那句"失败 N"就是它唯一的显示位置）。
			v.scanFailed++
		} else {
			// **合并必须在主线程做** —— 这正是 scanOneCmd 先拷副本的原因。
			tn := v.m.Tensors[msg.idx]
			tn.Stats, tn.Quant, tn.QuantSims = msg.stats, msg.quant, msg.sims
		}
		v.scanDone++
		if v.scanDone == len(v.m.Tensors) {
			// **扫完必须停**：这里返回下一条命令的话，链会一直自我调度下去
			//（多走一轮永远走不完的空转），用户按不停，界面每帧都在重绘。
			v.scanning = false
			return v, nil
		}
		return v, v.scanOneCmd(v.scanDone)

	case selectMetaMsg:
		// **越界就整个丢弃**：从速查表跳回来时，模型理论上没变，
		// 但"理论上"不是保证 —— 越界索引会在下一帧 panic
		//
		// `v.m == nil` 那一半是**防御性的、不可达**（selectMetaMsg 只能由
		// 一个加载好的 ModelView 推出来的 EntryView 发出），但 View() 里
		// 早就有 `if v.m == nil` 的分支 —— "m 为 nil"在本仓是被承认
		// 可表示的状态，而一次 nil 解引用 panic 的代价远大于一行判空。
		if v.m == nil || msg.index < 0 || msg.index >= len(v.m.Metadata) {
			return v, nil
		}
		v.cursor = int(sectionMetadata)
		v.focus = focusBody
		v.metaCursor = msg.index
		return v, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "tab":
			// Tab 只在当前栏目的内容**有可选项**时切换焦点：
			// 概览、量化分布那几栏的右栏是一段定长文字，
			// 切过去会让 ↑↓ 突然不能切栏目，而屏幕上没有任何东西
			// 说明为什么（这时 Tab 什么都不做，帮助栏也不显示它）。
			if v.hasBodyCursor() {
				if v.focus == focusNav {
					v.focus = focusBody
				} else {
					v.focus = focusNav
				}
			}
		case "up", "k":
			v = v.move(-1)
		case "down", "j":
			v = v.move(1)
		case keyScanAll:
			// 扫描中再按一次 = 取消。**已经扫出来的结果留着**（One 是就地
			// 填 tn.Stats，天然如此）—— 回滚等于把用户等的时间丢掉。
			//
			// 取消只做两件事：不再调度下一条、进度行消失。在途那一条回来时
			// 会被 batchScannedMsg 那条判据丢掉（scanning 已经是 false）。
			// 没有 ctx 可以取消 —— 那要 One 的签名带一个能被外部取消的 ctx，
			// 而它一次只扫一张、最多几秒，丢弃结果比改签名划算。
			//
			// **只有站在这一屏才取消得了**（`a` 是 ModelView 的键）：扫描是
			// 后台的，用户在其它屏看东西时它照跑 —— 想停就回这一屏按 `a`。
			// 换句话说"离开这一屏"不是取消，也不是暂停，什么都不发生。
			//
			// # 这条路径与 CLI 的 --stats 不是一回事（取舍，别顺手"修"）
			//
			// 它走的是 analyze.One —— 那个函数按 Task 4 的契约**不碰缓存**，
			// 所以在这里扫完之后，CLI 的 `--stats` 下次仍然要重算（缓存里没有）。
			// 这是**刻意保留**的两条路径：
			//   - `--stats` 走 Analyze：整份模型一次算完、写缓存；
			//   - `a` 走 One：逐张量、要进度、要能取消，结果只在内存里
			//     （再点开详情页是瞬时的，因为 NeedsWork 看的是 tn.Stats）。
			//
			// 为什么不给 One 加缓存写入来"顺带解决"：往整份模型的缓存里
			// 写单张量就是**部分写入** —— 下次全量扫描会"看起来命中了、
			// 其实缺一半"，而那类故障不报错，只是数字少了几块
			//（one.go 里写着这条）。进度要的是逐张量粒度，缓存只认整份，
			// 两者的粒度对不上，不能混在一个写入点里。
			if v.scanning {
				v.scanning = false
				return v, nil
			}
			if !v.canScan() {
				return v, nil
			}
			v.scanning, v.scanDone, v.scanFailed = true, 0, 0
			return v, v.scanOneCmd(0)
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
			// 元数据栏的 Enter 是**跳到速查表条目**，不是推入子视图 ——
			// 所以它不必先按 Tab（帮助栏那句"Enter 打开"指的就是它）。
			// 跳不过去的行什么都不做：推一张空白卡片进去，用户只会
			// 以为是自己按错了。
			// 模型要跟着进条目页（`NewEntryView(v.m, e)`）：Task 10 的
			// "在本模型中"那一节是跳过去之后**回来**的那一半，
			// 它依赖 EntryView.m 非 nil —— 传 nil 的话这条路径是单向的。
			if e, ok := v.metaJumpEntry(); ok {
				return v, pushCmd(NewEntryView(v.m, e))
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

// canScan 表示 `a` 现在真的扫得动。
//
// 两个条件：模型解析出来了、且真的有张量。缺一个时按 `a` 什么也不做，
// 所以 Help() 读同一个判据决定列不列它 —— keys.go 那条"列了不支持的
// 等于骗用户按"对这里同样成立（空模型那一屏列"a 扫描全部"就是这句话）。
func (v ModelView) canScan() bool { return v.m != nil && len(v.m.Tensors) > 0 }

// scanOneCmd 造一条"扫第 i 个张量"的命令。
//
// **副本在构造命令时（主线程）就拷好，不放进闭包**：闭包由 bubbletea
// 在另一个 goroutine 里执行，而这张 tn 上的写者有**两个** —— 这条链
// 自己的合并，以及用户点开同一张量详情页时 TensorView 的合并，两个
// 都在主线程。在 goroutine 里拷就是在读一个可能正在被写的字段：
// 窗口极窄（两边各自都是一瞬间的事），但它不需要窗口多大才算竞争。
//
// TensorView.scanCmd 的形状与此完全一致（也是先拷再进闭包）——
// 它那一处是这一版顺手改的：这条链给共享的 tn 添了第二个写者。
//
// 一次一条、回来了再发下一条：没有长期存活的 goroutine、没有 channel，
// 共享可写状态也只剩两处 Update（这条链的、TensorView 的），
// 都在主线程 —— 两个写者之间没有版本号可判新旧，所以只在**成功**时写
// （见 Update 里那条分支），失败的旧副本不落地。
func (v ModelView) scanOneCmd(i int) tea.Cmd {
	m, scan := v.m, v.scan
	cp := *m.Tensors[i]
	return func() tea.Msg {
		err := scan(context.Background(), m, &cp)
		return batchScannedMsg{
			m: m, idx: i, err: err,
			stats: cp.Stats, quant: cp.Quant, sims: cp.QuantSims,
		}
	}
}

// scanHelp 是 `a` 那一格的帮助文字。
//
// 三个状态各一句：扫描中是"取消"、有张量可扫是"扫描全部"，
// 都没有（模型还没解析出来 / 这个文件里没有张量）**什么都不列**。
// （`a` 与 CLI 的 `--stats` 是两条不同的路径，取舍写在 Update 里
// 那条 case keyScanAll 分支上。）
func (v ModelView) scanHelp() string {
	switch {
	case v.scanning:
		return keyScanAll + " 取消"
	case v.canScan():
		return keyScanAll + " 扫描全部"
	}
	return ""
}

// scanLine 是内容区顶部那一行扫描状态；没扫过（或取消之后）返回空串。
//
// **它占满整个内容区一行**（跨左右两栏）：挤在某一栏的右栏里的话，
// 用户按 ↑↓ 换个栏目进度就没了 —— 而扫描是后台在跑的，看不见就等于
// 不知道它还在不在跑。
//
// **扫完之后不消失，换成一行"扫描完成"**：这是整条链唯一的收尾信号，
// 也是失败计数唯一的显示位置（张量列表与详情页都不显示"有 N 个没算出来"）。
// 直接消失的话，用户分不清"扫完了"与"还在跑"—— 扫描是**后台**的
// （根视图把消息路由到这一层，见 routeToModelView），用户完全可能在
// 它跑的时候翻到别的屏去，回来时这一行是他唯一的凭据；而失败信息
// 一消失就被静默丢掉了，那是本仓反复踩过的那类问题。代价是常驻一行，
// 再按 `a` 会刷新它。
//
// **取消之后不显示**：那是用户自己的动作，不需要收尾信号；已经扫出来的
// 结果就在张量里（详情页点开是瞬时的），没有信息被丢掉。
//
// 不做百分比（Task 5 已经写过理由）：剩余时间估不出来，而一个会跳的
// 百分比比"123/434"更容易被当成承诺。
func (v ModelView) scanLine() string {
	switch {
	case v.scanning:
		line := fmt.Sprintf("正在扫描张量… %d/%d", v.scanDone, len(v.m.Tensors))
		if v.scanFailed > 0 {
			line += fmt.Sprintf(" · 失败 %d", v.scanFailed)
		}
		return styleHint.Render(line + "（" + keyScanAll + " 取消）")
	case v.scanDone > 0 && v.scanDone == len(v.m.Tensors):
		line := fmt.Sprintf("扫描完成 %d/%d", v.scanDone, len(v.m.Tensors))
		if v.scanFailed > 0 {
			// **失败那半句不能只是灰色小字**：它是这一屏唯一一处
			// 说"有张量没算出来"的地方
			return styleWarn.Render(line + fmt.Sprintf(" · 失败 %d", v.scanFailed))
		}
		return styleHint.Render(line)
	}
	return ""
}

// hasBodyCursor 表示当前栏目的右栏内容**有可选项**。
//
// 概览与量化分布是定长文字，没有可选项 —— 在它们上面切到右栏，
// ↑↓ 会突然不能切栏目，而屏幕上没有任何东西说明为什么。
// 所以不是"能不能切"，而是"这一栏压根没有右栏焦点"。
//
// 元数据为空时同样没有可选项（safetensors 可以没有 __metadata__ 段）：
// 切过去只会让 ↑↓ 失效，而帮助栏还写着"↑↓ 选择" —— 那是同一个坑。
//
// **必须判 m 是不是 nil**：根视图每一帧都调 Help()，而解析还没回来时
// m 是 nil —— 用户在"正在解析…"上按几下 ↓ 走到这一栏就会撞上解引用 panic。
func (v ModelView) hasBodyCursor() bool {
	return section(v.cursor) == sectionMetadata && v.m != nil && len(v.m.Metadata) > 0
}

// move 把 ↑↓ 交给当前有焦点的那个光标，**返回改过的新视图**。
//
// 必须返回新值而不是就地改：接收者是值（视图在栈里也是值，
// 见 RefView.move 那条说明）。写成不返回的 `func (v ModelView) move(...)`，
// 调用点一句 `v.move(1)` 改的是副本 —— ↑↓ 整个失效，
// 而屏幕上只是"按了没反应"，从现象上完全看不出原因（实测踩过）。
//
// **换栏目时元数据光标归零**：不归零的话，从 60 条的元数据切到
// 张量栏再切回来，光标停在 55 —— 而提示里的范围是按光标算的，
// 用户会看到一段对不上的提示。越界也是同一个来源。
func (v ModelView) move(delta int) ModelView {
	if v.focus == focusBody {
		n := len(v.m.Metadata)
		if n == 0 {
			return v
		}
		v.metaCursor += delta
		if v.metaCursor < 0 {
			v.metaCursor = 0
		}
		if v.metaCursor > n-1 {
			v.metaCursor = n - 1
		}
		return v
	}
	v.cursor += delta
	if v.cursor < 0 {
		v.cursor = 0
	}
	if v.cursor > len(sections)-1 {
		v.cursor = len(sections) - 1
	}
	// 换栏目就把右栏光标归零（理由见函数注释）
	v.metaCursor = 0
	v.focus = focusNav
	return v
}

// metaJumpEntry 返回**光标当前那一行**能跳到的条目（跳不过去时 false）。
//
// 四个边界都收在这里，Update 里那条分支才能平铺成一句话：
//   - 模型还没解析出来（m == nil）：Update 的 Enter 分支有早退，
//     但 opensOnEnter 会在那之前走到这里 —— 少这一句就是一次
//     nil 解引用 panic（帮助栏每一帧都渲染）
//   - 栏目不是元数据（这一栏的 Enter 是别的动作）
//   - 元数据为空（safetensors 可以没有 __metadata__ 段）：光标压在 0 上
//     而一条都没有，直接取下标就是越界 panic
//   - 这一行没有关联：跳不过去
func (v ModelView) metaJumpEntry() (ref.Entry, bool) {
	if v.m == nil || section(v.cursor) != sectionMetadata ||
		v.metaCursor >= len(v.m.Metadata) {
		return ref.Entry{}, false
	}
	return v.metaEntry(v.m.Metadata[v.metaCursor])
}

// metaEntry 判断这条元数据能跳到速查表的哪一条。
//
// **与渲染时的判定必须是同一个函数**：渲染时决定挂不挂 `◂` 记号，
// 这里决定 Enter 跳不跳。两处各写一遍的话，会出现"有记号但跳不动"
// 或者更糟的"没记号却能跳"。任务描述里那句"一个键最多一个记号"
// 正是为了避免两个记号指向不同条目 —— 这里沿用同一个优先级：
// 值释义（file_type）优先于键释义。
func (v ModelView) metaEntry(kv model.MetaKV) (ref.Entry, bool) {
	if e, ok := fileTypeEntry(kv); ok {
		return e, true
	}
	return ref.LookupKey(kv.Key)
}

// opensOnEnter 表示当前栏目的内容会在 Enter 时**有动作**。
//
// 帮助栏只列真的按键 —— `keys.go` 里那条"列了不支持的等于骗用户按"
// 对这里同样成立。**每接上一个栏目的 Enter 动作，就在这里加一格**：
// 忘了加的表现是"按了有反应但帮助栏不写"（用户发现不了），
// 加多了的表现是"帮助栏写了一个按了没反应的键"（用户被指到死路）。
// 后者更糟，所以默认返回 false。
//
// **"有动作"不等于"会推入子视图"**：元数据栏的 Enter 是跳到速查表的
// 条目（见 Update 里那条分支），它同样要在帮助栏里出现。
//
// 元数据栏的判据是**光标当前那一行跳不跳得动**，不是"这一栏有没有条目"：
// 同一栏里 general.file_type 跳得动、它下面那些 custom.* 一条都跳不动，
// 而光标一往下走，"这一栏有条目"就与"这一行跳得动"分道扬镳 ——
// 从栏目级条件（曾经是 `hasBodyCursor`）派生的话，光标停在哪一行
// 帮助栏就在哪一行印假话（实测 metaModel：10 条里 9 条如此；
// safetensors 的 __metadata__ 更彻底，每一条都跳不动却每一帧都列 Enter）。
// 所以这里直接问 `metaJumpEntry` —— **与 Update 的 Enter 分支同一个函数**，
// 两处不可能漂移。
//
// 张量与速查表两栏要 `v.m != nil`：载入中（"正在解析…"）时 Update 开头
// 那句早退让 Enter 什么都不做，这时列"Enter 打开"是同一类假话 ——
// 而"正在解析"那几秒里用户恰好最容易按着不动。
func (v ModelView) opensOnEnter() bool {
	switch section(v.cursor) {
	case sectionMetadata:
		_, ok := v.metaJumpEntry()
		return ok
	case sectionTensors, sectionRef:
		return v.m != nil
	}
	return false
}

func (v ModelView) Help() []string {
	// `?` 与 `a` 两格由 scanHelp / keyHelp 提供：`a` 有时真的没有动作
	//（模型还没解析出来、或这个文件里没有张量），那时 scanHelp 返回空串。
	scan := v.scanHelp()
	if v.focus == focusBody {
		// **右栏焦点下的 Enter 也要走 opensOnEnter**：原先把"Enter 查速查表"
		// 写死在这一支里，于是光标往下走到某条 custom.*（跳不动）时，
		// 屏幕上那句"Enter 查速查表"照样在 —— 用户按下去的反馈是"什么都没发生"。
		// 左栏那一支本就按同一个判据列，两支分头判断才会出现这种一只眼。
		bindings := []string{keyUp + " " + keyDown + " 选择"}
		if v.opensOnEnter() {
			bindings = append(bindings, keyEnter+" 查速查表")
		}
		bindings = append(bindings, keyTab+" 回到栏目", keyHelp+" 速查表")
		if scan != "" {
			bindings = append(bindings, scan)
		}
		return append(bindings, keyEsc+" 返回", keyQuit+" 退出")
	}
	bindings := []string{keyUp + " " + keyDown + " 切换栏目"}
	if v.hasBodyCursor() {
		bindings = append(bindings, keyTab+" 选内容")
	}
	if v.opensOnEnter() {
		bindings = append(bindings, keyEnter+" 打开")
	}
	bindings = append(bindings, keyHelp+" 速查表")
	if scan != "" {
		bindings = append(bindings, scan)
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

	line := v.scanLine()
	if line == "" {
		return joinHorizontal(nav, v.body(bodyWidth, height))
	}
	// 进度行那一行是**从内容区里扣的**：这一层交给根视图的原始输出
	// 不能比给定的高度多一行（多了的话 padTo 会砍掉各栏目自己算好的
	// 最后一行 —— 元数据那一栏正是它的"显示第 N–M 条"，换上一句
	// 措辞更差的"…还有 N 行没显示"）。
	//
	// 扣掉的这一行**是看得出来的**：同一个高度下，元数据那一栏的范围
	// 提示从"1–17"变成"1–18"（条数差一条）。它只在专门盯这句话的
	// TestModelView_进度行从内容区里扣 里才红 —— 变异验证实测过：
	// 去掉这个 -1，其余测试（包括那些数行数的）全绿。
	//
	// **高度 1 是唯一的例外，而且这一行在那里保不住**：padTo 截断时
	// 保留的是前 h-1 行（h=1 时一行都不保留），所以进度行照样被它那句
	// "…还有 N 行没显示"顶掉。`max(..., 1)` 只是保证 body 拿到的不是 0，
	// 不是为了在高度 1 时显示进度行 —— 那种终端（总高 ≤3 行）里
	// 两者不可能同时显示，谁也救不了。
	bodyHeight := max(height-1, 1)
	return line + "\n" + joinHorizontal(nav, v.body(bodyWidth, bodyHeight))
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

	// **判据必须是 TiedGroups，不能是"两个数字不相等"**。
	//
	// StorageBytes 的语义随格式变：safetensors 下它是数据区跨度，而头部
	// 允许 data_offsets 不从 0 开始 —— 此时它**大于**张量字节和，
	// 打出来就是「去重后仅占 32 B（差值 -16 B 是共享的存储）」，
	// 自相矛盾（"去重后"比"去重前"更大，差值还是负的）。
	//
	// CLI 的 printSummary 用的是同一个判据，两处不能分叉 ——
	// 而 TUI 这一处原来就是没跟上 CLI 的那次修正。
	if len(m.TiedGroups) > 0 {
		fmt.Fprintf(&sb, "，去重后占 %s（差值 %s 是共享的存储）",
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

// metadata 渲染元数据列表，**带内容光标与滚动**，并在可解释的值旁边
// 挂上速查表关联。
//
// 关联是**现算**的（拿 key 调 ref.LookupKey），不是解析时写死字段 ——
// 写死的那种一旦速查表换了 ID 就全体指空，而且已经在 --json 里对外过一遍。
//
// ④b-1 里这一栏还没有内容光标，靠一句"还有 N 条没显示"顶着 ——
// 那句提示在滚动做出来之后就删了：滚动做了之后它就是一句假话。
func (v ModelView) metadata(height int) string {
	var sb strings.Builder
	sb.WriteString(styleSection.Render(fmt.Sprintf("元数据（%d 条）", len(v.m.Metadata))) + "\n\n")

	// 标题 2 行（含空行）+ 可能的"显示第 N–M 条"1 行
	listCap := height - 3
	if listCap < 1 {
		listCap = 1
	}
	// 窗口与"要不要打范围提示"都交给 listWindow（先扣提示行、再算窗口，
	// 且只有真放得下才打）—— 那三条理由写在那里，不在这里重抄一遍。
	n := len(v.m.Metadata)
	start, end, showHint := listWindow(n, v.metaCursor, listCap)
	focusHere := v.focus == focusBody
	lines := make([]string, 0, end-start+1)
	for i := start; i < end; i++ {
		lines = append(lines, v.metaLine(i, v.m.Metadata[i], focusHere))
	}
	if showHint {
		lines = append(lines, styleDim.Render(fmt.Sprintf(
			"…显示第 %d–%d 条，共 %d 条", start+1, end, n)))
	}
	// 末尾不留换行：`strings.Join` 天然不会多，但别改成逐行 append "\n"
	sb.WriteString(strings.Join(lines, "\n"))
	return sb.String()
}

// metaLine 渲染一行元数据。
//
// **记号规则原样照搬 ④b-1，一个字都不改**：它已经在真终端里看过、
// 也有测试盯着。这一版只是把"这一行归渲染"扩成"顺便算一下光标"，
// 顺手改渲染的话，改坏的是一处**与本次目标无关**的东西。
//
// 原来的三条规则：
//   - 值本身就是个码的（file_type），把码翻译出来：`[标题] ◂`
//   - 否则键本身在速查表里有条目的，标一个 `◂`
//   - **一个键最多一个记号**：file_type 那条已经挂了值释义，
//     再挂一个键记号的话同一行会出现两个一模一样的 `◂`（实测），
//     而它们指向两个不同的条目 —— Enter 跳哪一个就没有答案
func (v ModelView) metaLine(i int, kv model.MetaKV, focusHere bool) string {
	var sb strings.Builder
	sb.WriteString(styleField.Render(kv.Key) + "  " + oneLine(kv.Value))

	_, hasValueRef := fileTypeEntry(kv)
	if e, ok := fileTypeEntry(kv); ok {
		sb.WriteString("  " + styleLink.Render("["+e.Title+"] ◂"))
		// 上游猜出来的档位要标出来：它可能与该文件实际的张量类型不符，
		// 而用户会拿它当事实（ref 包里的原话）
		if code, ok := fileTypeCode(kv); ok && ref.FileTypeGuessed(code) {
			sb.WriteString(styleWarn.Render(" (上游猜的，未必与实际张量类型相符)"))
		}
	}
	if !hasValueRef {
		// 键释义这一支走 metaEntry（值释义已经在上面拦掉了，两者等价）——
		// "挂不挂记号"与"跳不跳"因此真的只有一份判定。
		if _, ok := v.metaEntry(kv); ok {
			sb.WriteString("  " + styleLink.Render("◂"))
		}
	}

	line := sb.String()
	if focusHere && i == v.metaCursor {
		return styleSelected.Render(line)
	}
	return line
}

// oneLine 把值里的换行换成可见记号，让**一条元数据永远只占一行**。
//
// 含换行的值（qwen2.5 的 tokenizer.chat_template 那种）一次渲染出来是
// 几十行，而这一栏的行数核账（listCap / window / "显示第 N–M 条"）
// 全按"一条一行"算 —— 实测 80×24 下那句范围提示被整个顶掉，屏幕上换成
// 根视图 padTo 的"…还有 54 行没显示"，用户既不知道自己在第几条，
// 也不知道后面还有几条。
//
// **不能静默丢内容**：直接删掉换行的话值会变成一行连在一起的假文本。
// ␊（U+240A，控制图片符）让"这里原本有换行"看得见，且只占一列宽
// （East Asian Width 是 Neutral），不会把那一行撑歪。
// \r 单独处理：CRLF 不先合成一个的话会显示成 ␍␊ 两个记号。
func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\n", "␊")
	return strings.ReplaceAll(s, "\r", "␍")
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

// displayName 是标题栏里显示的名字。
//
// **判据交给 model.Model.DisplayName**：张量列表与条目页也要显示
// 同一个名字，三处各写一份的话改一处漏一处不会有编译错误 ——
// 而表现是同一屏里同一个模型有两个名字（实测过：标题栏是 qwen2.5:3b、
// 点进张量列表变成 sha256-5ee4f07c…）。
//
// v.name 先判是因为这一屏在解析完成前也要能渲染（加载态还没有 m）。
func (v ModelView) displayName() string {
	if v.name != "" {
		return v.name
	}
	return v.m.DisplayName()
}
