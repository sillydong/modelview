// Package discover 扫描本机常见的模型存储目录。
//
// 设计要点：
//   - **目录不存在是常态，不是错误**。本机 12 条路径里通常只有一两条存在，
//     每条都报错的话输出会被噪音淹没
//   - 扫描分两层：Scan 只 stat（瞬时），Fill 才读头部补格式与参数量。
//     几百个文件的 HF 缓存里，一次读完再显示会让用户盯着空屏几十秒
//   - 结果顺序稳定（按路径排序）：界面每次刷新时列表乱跳是不可用的
package discover

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/parser"
)

// maxDepth 是通用目录的递归深度上限。
//
// 6 层足够覆盖 HF 缓存（models--org--name/snapshots/<sha>/model.safetensors，
// 4 层）与常见的 experiment/run/checkpoint 布局，又不会在
// 意外选到大目录时把整块盘走一遍。
const maxDepth = 6

// skipDirs 是不进入的目录名。
var skipDirs = map[string]bool{
	".git": true, "node_modules": true, ".venv": true,
	"__pycache__": true, ".modelview-cache": true,
}

// Source 是一个发现来源。
type Source string

const (
	SourceOllama      Source = "ollama"
	SourceHuggingFace Source = "huggingface"
	SourceLMStudio    Source = "lm-studio"
	SourceGPT4All     Source = "gpt4all"
	SourceLlamaCpp    Source = "llama.cpp"
	SourceTorchHub    Source = "torch-hub"
	SourceJan         Source = "jan"
	SourceModelScope  Source = "modelscope"
	SourceGeneric     Source = "通用目录"
)

// Path 是一条待扫描的目录。
type Path struct {
	Source Source
	Dir    string // 已经展开过 ~ 的绝对路径

	// manifest 非空表示这个来源要靠 manifest 解析（ollama），
	// 不做通用递归 —— 它的文件是内容寻址的，递归出来的是一堆 sha256-xxx。
	//
	// **把处理器放在这里，而不是放一个 `Manifest bool` 再去 Scan 里
	// 按 Source 分发**：一个"声明了要按 manifest 扫、却没有对应处理器"的
	// 条目在那种写法下是能写出来的，而它的表现是**静默扫不到任何东西**
	//（命中 bool、进不了 if、continue）—— 用户看到的是"我没有模型"。
	// 现在处理器就是字段本身，这种组合不存在。
	manifest func(root string) (items, orphans, inProgress []Item, errs []string)
}

// KV 是一条附加信息。
//
// **json tag 不能省**：省掉会打出 {"Key":…,"Value":…}，而同一份 JSON
// 里 Item 的其它字段全是 snake_case（source/name/path/size/format/
// arch/param_count）—— 消费方要为这一个字段破例。
// 与 model.Model 里那句"键名漂移在 map 里既无编译错误也无测试失败"
// 是同一类顾虑。
type KV struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// Item 是一个被发现的模型。
type Item struct {
	Source Source `json:"source"`
	// Name 是模型的显示名，**Scan 返回的每个条目都非空**——
	// ollama 从 manifest 推断（如 "qwen2.5:3b"，与 `ollama list` 一致），
	// 通用目录与孤儿 blob 用文件名。
	//
	// 这是给消费方的契约：界面直接用它当第一列，JSON 的消费方不必判空。
	// 由 TestScan_每个条目都有名字 在生产者这一侧钉住 ——
	// 在渲染那一侧写 `if Name == "" { 用文件名 }` 是兜不住的：
	// 生产者漏填时那个分支根本不会被执行到（实测过，见 scanLine 的说明）。
	Name string `json:"name"`
	Path string `json:"path"`
	Size int64  `json:"size"`

	// 以下由 Fill 填充，未填时是零值。
	// Format 用有类型的 model.Format 而不是裸 string：
	// 消费方（界面、JSON 的校验）都需要这个类型，而裸字符串没有编译期约束
	Format model.Format `json:"format,omitempty"`
	Arch   string       `json:"arch,omitempty"`
	Params int64        `json:"param_count,omitempty"`
	// Err 记录这个条目读失败的原因；不中断整体扫描。
	// **Fill 之后 Format 与 Err 必然有一个非空** —— 两者皆空意味着
	// 界面上会永远显示"读取中"。
	Err string `json:"err,omitempty"`

	// Extra 是来源特有的附加信息（ollama 的 template/system/license 层）
	Extra []KV `json:"extra,omitempty"`
}

// Result 是一次扫描的结果。
type Result struct {
	// Items 是模型条目，按路径排序。
	Items []Item `json:"items"`
	// Orphans 是没有被任何 manifest 引用的 blob（只有 ollama 有这个概念）。
	Orphans []Item `json:"orphans,omitempty"`
	// InProgress 是 ollama 还没下完的文件（<digest>-partial 及其分片）。
	//
	// **与 Orphans 分开是必须的**：它们看着占了一大片盘，但删了会毁掉
	// 用户正在进行的下载。实测拉 gpt-oss:20b 到一半时，它们曾被算进
	// "可回收 12.85 GiB" —— 那是把"下到一半"当成了"没人要"。
	InProgress []Item `json:"in_progress,omitempty"`
	// Errs 是各目录的失败原因。**不中断整体** —— 本机 12 条路径里
	// 通常只有一两条存在，目录不存在是常态不是错误，不进这里。
	Errs []string `json:"errs,omitempty"`
}

// Options 控制扫描行为。
type Options struct {
	// Home 用于展开 ~。空表示用 os.UserHomeDir()。
	// **测试必须能指定它** —— 否则测的是开发机的真实目录，
	// 结果随机器而变，而本机只有 1/12 条路径存在，覆盖不到别的来源。
	Home string
	// Only 非空时只扫这些来源。空表示全部。
	Only []Source
}

// Paths 返回待扫描的目录清单。
//
// 路径来自 spec §7 的表。**顺序即展示顺序**：先列推理框架，
// 再列通用目录 —— 用户先关心的是"我装的模型"，不是"我 clone 的仓库"。
func Paths() []Path {
	raw := []Path{
		{Source: SourceOllama, Dir: "~/.ollama/models", manifest: scanOllama},
		{Source: SourceHuggingFace, Dir: "~/.cache/huggingface/hub"},
		{Source: SourceLMStudio, Dir: "~/.cache/lm-studio"},
		{Source: SourceLMStudio, Dir: "~/.lmstudio/models"},
		{Source: SourceGPT4All, Dir: "~/.cache/gpt4all"},
		{Source: SourceGPT4All, Dir: "~/.local/share/nomic.ai/GPT4All"},
		{Source: SourceLlamaCpp, Dir: "~/.cache/llama.cpp"},
		{Source: SourceLlamaCpp, Dir: "~/Library/Caches/llama.cpp"},
		{Source: SourceTorchHub, Dir: "~/.cache/torch/hub/checkpoints"},
		{Source: SourceJan, Dir: "~/Library/Application Support/Jan"},
		{Source: SourceModelScope, Dir: "~/.cache/modelscope"},
		{Source: SourceGeneric, Dir: "~/models"},
	}
	return raw
}

// expandHome 展开前导 ~。**只展开开头的 `~/`**，中间的 ~ 是文件名的一部分。
//
// home 为空（拿不到用户目录）时原样返回：展开成相对路径比不展开更糟。
func expandHome(p, home string) string {
	if home == "" {
		return p
	}
	if p == "~" {
		return home
	}
	if strings.HasPrefix(p, "~/") {
		return filepath.Join(home, p[2:])
	}
	return p
}

// Scan 列出候选：只做目录遍历与 os.Stat，**不解析任何模型**，所以是瞬时的。
//
// 格式与参数量由 Fill 补。这一步刻意不解析：几百个文件的 HF 缓存里，
// 逐个读头部要几十秒，而用户想先看到的是"有哪些文件"。
//
// **例外是 ollama**：它走 manifest 那条路，要读 manifest 本身，
// 还要把 license / template / system 这几个文本层读出来放进 Extra
// （见 ollama.go 的 readLayerLabel）—— 那些层是几 KB 的文本，
// 不是权重，所以"瞬时"仍然成立。但这句话不能说成"不读任何文件内容"：
// 实测只调 Scan（不调 Fill），条目的 Extra 里已经带着完整的 license 与
// params 内容了。
func Scan(ctx context.Context, opts Options) Result {
	// Items 初始化成空切片而不是 nil：JSON 里 nil 会编码成 `null`，
	// 而 `null` 与 `[]` 对消费方是两回事 —— 前者要额外判空。
	// 契约应当是"没有模型时给一个空数组"，不是一个需要特判的空值。
	res := Result{Items: []Item{}}
	home := opts.Home
	if home == "" {
		home, _ = os.UserHomeDir()
	}
	only := map[Source]bool{}
	for _, s := range opts.Only {
		only[s] = true
	}

	for _, p := range Paths() {
		if len(only) > 0 && !only[p.Source] {
			continue
		}
		if err := ctx.Err(); err != nil {
			res.Errs = append(res.Errs, err.Error())
			return res
		}
		dir := expandHome(p.Dir, home)
		if _, err := os.Stat(dir); err != nil {
			// 目录不存在是常态：本机 12 条里只有 1 条存在。
			// 它不是错误，也不进 Errs —— 每条都报的话输出全是噪音
			continue
		}
		if p.manifest != nil {
			items, orphans, inProgress, errs := p.manifest(dir)
			res.Errs = append(res.Errs, errs...)
			res.Items = append(res.Items, items...)
			res.Orphans = append(res.Orphans, orphans...)
			res.InProgress = append(res.InProgress, inProgress...)
			continue
		}
		items, err := scanDir(ctx, dir, p.Source)
		if err != nil {
			res.Errs = append(res.Errs, fmt.Sprintf("%s: %v", dir, err))
			continue
		}
		res.Items = append(res.Items, items...)
	}

	// 顺序稳定：界面每次刷新时列表乱跳是不可用的
	sort.Slice(res.Items, func(i, j int) bool { return res.Items[i].Path < res.Items[j].Path })
	sort.Slice(res.Orphans, func(i, j int) bool { return res.Orphans[i].Path < res.Orphans[j].Path })
	return res
}

// modelSuffixes 是会被当成模型的扩展名。
//
// 按扩展名过滤是**第一层筛选**，不是判据：内容对不对由 Fill 说了算，
// 那时读不出格式的条目会带上 Err，界面上标红而不是被悄悄丢掉
// （用户看到"我明明有个模型，它没列出来"比看到一条红的更困惑）。
var modelSuffixes = []string{".gguf", ".safetensors", ".pt", ".pth", ".bin", ".onnx"}

func hasModelSuffix(name string) bool {
	lower := strings.ToLower(name)
	for _, suf := range modelSuffixes {
		if strings.HasSuffix(lower, suf) {
			return true
		}
	}
	return false
}

// scanDir 递归找模型文件。
func scanDir(ctx context.Context, root string, src Source) ([]Item, error) {
	var out []Item
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			// 权限不足的单个子目录不该让整棵树失败 —— 跳过这一项继续走
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if d.IsDir() {
			if path == root {
				// 根目录本身即便叫 .git 也要进（调用方明确指了它）
				return nil
			}
			if skipDirs[d.Name()] {
				return fs.SkipDir
			}
			// **`>` 而不是 `>=`**：maxDepth 是"文件所在目录的层数上限"，
			// 写成 `>=` 会把第 maxDepth 层整个跳过，实际只允许 maxDepth-1 层
			//（边界测试抓出来的 off-by-one）。
			if depthOf(root, path) > maxDepth {
				return fs.SkipDir
			}
			return nil
		}
		if !hasModelSuffix(d.Name()) {
			return nil
		}
		// **用 os.Stat 而不是 d.Info()**：WalkDir 给的 DirEntry 是
		// **lstat 语义**，对符号链接返回的是链接自身的信息 ——
		// HF 缓存的布局恰恰是 `snapshots/<sha>/model.safetensors` →
		// `blobs/<sha>` 的符号链接，于是首屏那一列全是几十字节，
		// 要等 Fill 读完头部才变成真大小，而解析失败的条目永远留着错值。
		// （这个函数的文档一直写着"只做 os.Stat"，是代码没跟上。）
		//
		// 仍然只是元数据查询，不读内容 —— 分层加载的前提不受影响。
		info, err := os.Stat(path)
		if err != nil {
			return nil // 断链的符号链接：跳过
		}
		out = append(out, Item{
			Source: src,
			Name:   d.Name(),
			Path:   path,
			Size:   info.Size(),
		})
		return nil
	})
	if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
		return nil, err
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

// depthOf 返回 path 相对 root 的层数（root 自己是 0）。
func depthOf(root, path string) int {
	rel, err := filepath.Rel(root, path)
	if err != nil || rel == "." {
		return 0
	}
	return strings.Count(rel, string(filepath.Separator)) + 1
}

// Fill 读一个条目的头部，补上 Format/Arch/Params。
//
// **就地修改**并返回 —— 调用方通常已经把它放进列表里了。
// 读失败**不返回 error**，而是写进 it.Err：一个坏文件不该让
// 整个列表都不能显示，界面按行标注即可。
func Fill(it *Item) *Item {
	if it.Err != "" {
		return it
	}
	m, err := parser.Parse(it.Path)
	if err != nil {
		it.Err = err.Error()
		return it
	}
	it.Format = m.Format
	it.Arch = m.Arch
	it.Params = m.TotalParams()
	// 大小以文件系统为准：manifest 里写的 size 可能与实际不符
	//（半途中断的下载就是这种），界面显示实际占用更有用
	if st, err := os.Stat(it.Path); err == nil {
		it.Size = st.Size()
	}
	return it
}

// ErrReason 从 Item.Err 里取"原因"部分，去掉开头的路径前缀。
//
// **只给渲染用，不改 Err 本身**：JSON 的 Err 保持原样（那里没有别的地方
// 显示路径，去掉就是信息丢失），但终端上路径通常比一行还长 ——
// 实测 80 列下整行只剩 "⚠/tmp/xxx/"，用户唯一得到的信息是"这个模型坏了"，
// 坏在哪不知道。而路径在 Item.Path 里本来就有，不必再占一次屏幕。
func ErrReason(it Item) string {
	prefix := it.Path + ": "
	return strings.TrimPrefix(it.Err, prefix)
}

// FillAll 逐个 Fill 并返回填好的切片。同步路径用它；
// 界面应当自己起 goroutine 逐个调用 Fill 并刷新 —— 那是分层加载的意义。
func FillAll(ctx context.Context, items []Item) []Item {
	for i := range items {
		if ctx.Err() != nil {
			break
		}
		Fill(&items[i])
	}
	return items
}

// ScanDir 递归扫描**任意目录**下的模型文件，返回条目与错误清单。
//
// 与 Scan 的分工：那个只看写死的十几个已知路径（spec §7 那张表），
// 这个是"就扫这里"。spec §3 把 `modelview <dir>` 与 `modelview scan`
// 并列，对应的就是这两条。
//
// **复用 scanDir 而不是自己走文件树**：模型后缀的判据、递归深度上限、
// 要跳过的目录名（.git / node_modules / 缓存目录）都已经在那里，各写
// 一份迟早分叉 —— 而分叉的表现是"同一个目录，两条命令列出不同的文件"，
// 用户无从判断哪个是对的。
//
// 来源标成 generic：这个目录不在已知路径表里，标成别的是撒谎。
//
// **"失败"与"空目录"必须由两个信号分别表达**：
//   - errs 非空 = 扫描失败，此时 items 是 nil
//   - errs 为空 = 扫描成功，items 是**非 nil** 的切片（空目录就是空切片）
//
// 不这么做的话调用方只能拿 `items == nil` 当失败判据，而空目录的
// items 也是 nil —— 实测过：空目录会打出"扫描失败"，用户以为路径写错了。
// 与 model.Tensor 的 SizeUnknown、MetaKV 空切片的取舍是同一个道理：
// 哨兵值必须显式且唯一。
func ScanDir(ctx context.Context, root string) ([]Item, []string) {
	items, err := scanDir(ctx, root, SourceGeneric)
	if err != nil {
		return nil, []string{err.Error()}
	}
	if items == nil {
		items = []Item{}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].Path < items[j].Path })
	return items, nil
}

// TotalSize 返回一组条目的字节总数。
//
// **原来有三份逐字相同的实现**：cmd 的 totalSize、tui 的 totalBytes、
// 以及 discover 自己测试里的一份。同一个语义写三遍，改一处漏一处不会
// 有编译错误 —— 而它算的是"能回收多少磁盘"，正是孤儿 blob 那一段
// 引导用户做删除决策的依据。合并到领域层，因为它问的是 discover.Item。
func TotalSize(items []Item) int64 {
	var n int64
	for _, it := range items {
		n += it.Size
	}
	return n
}
