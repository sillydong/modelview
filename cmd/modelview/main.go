// Command modelview 解析并展示模型文件的结构信息。
//
// 支持 GGUF、safetensors 与 PyTorch（.pt）三种格式。
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime/debug"
	"slices"
	"strings"

	"github.com/sillydong/modelview/internal/analyze"
	"github.com/sillydong/modelview/internal/humanize"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/parser"
	"github.com/sillydong/modelview/internal/render"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "modelview: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		asJSON      = flag.Bool("json", false, "以 JSON 输出（非交互）")
		showVersion = flag.Bool("version", false, "打印版本后退出")
		asStats     = flag.Bool("stats", false, "计算张量数值统计（慢，会读全部张量数据）")
		noCache     = flag.Bool("no-cache", false, "禁用缓存读写")
		// 0 表示不采样（读完整个张量）；默认 1e7 个元素约 40 MB
		sampleLimit = flag.Int("sample-limit", analyze.DefaultSampleLimit,
			"单张量统计的采样上限（元素数），0 表示不采样（会读完整个张量，"+
				"大模型上峰值内存可达数 GB）。"+
				"注意它只管统计：量化张量的块级诊断要读遍全部块头才能"+
				"找出被压得最狠的子块，不受这个上限约束")
	)
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, "用法: modelview [选项] <模型文件>\n"+
			"      modelview [选项] scan        扫描本机模型目录（无参数时也是它）\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	// **位置参数里出现 "-" 开头的东西 = 选项写在子命令后面了**。
	//
	// Go 的 flag 遇到第一个非选项参数就停止解析，于是
	// `modelview scan --json` 里的 --json **被静默丢掉**，
	// 打出来的是人类可读的列表 —— 脚本把它管道给 jq 才报错，
	// 而报错的地方离原因很远。实测过这个行为。
	//
	// 宁可报错也不要猜。
	//
	// **`--` 之后要豁免**：flag 会把 `--` 自己**消费掉**，`flag.Args()` 里
	// 仍然留着它后面的参数 —— 只看 Args() 的话守卫照样拒绝，而上面那句
	// 「写 `--` 可以显式终止选项解析」就成了一个**不存在的逃生口**。
	// 实测：`modelview --no-cache -- -weird.gguf` 被拒，
	// 于是名字以 - 开头的模型文件永远打不开，而提示语给的办法无效。
	//
	// 判据只能看 os.Args（flag 消费掉的东西在 Args() 里找不回来）。
	if !slices.Contains(os.Args, "--") {
		for _, a := range flag.Args() {
			if strings.HasPrefix(a, "-") && a != "-" {
				flag.Usage()
				return fmt.Errorf("选项 %q 写在位置参数后面了，不会被解析；"+
					"选项要写在前面，例如 modelview --json scan（"+
					"路径本身以 - 开头时写在 `--` 之后）", a)
			}
		}
	}

	if *showVersion {
		fmt.Println("modelview " + version())
		return nil
	}

	// 无参数或显式 scan 都进模型库。
	//
	// **--json 与非终端都走非交互**：JSON 是一个整体，不能流式拼；
	// 而重定向到管道或文件时 bubbletea 拿不到尺寸，画出来的东西是给
	// 看不到它的人准备的，ANSI 转义还会混进 jq 的输入里。
	//
	// **非终端这条原来没有**：那时它落到 runTUI，被"交互界面需要终端"
	// 挡回来 —— 于是 `modelview scan` 在管道/日志里完全用不了，
	// 而 runScan 那段人类可读输出（含孤儿 blob 与未完成下载的提示）
	// 已经写好、只被测试覆盖。
	//
	// 分流放在这里而不是让 runTUI 自己兜：runTUI 该做的就是"不能跑就
	// 说清楚为什么"，而"非终端下换成什么"是入口的决策。
	if flag.NArg() == 0 || flag.Arg(0) == "scan" {
		if *asJSON || !isTerminal(os.Stdout) {
			return runScan(context.Background(), *asJSON)
		}
		return runTUI()
	}

	// **这里没有"缺少参数"的分支**：NArg()==0 已经在上面走了 scan
	// （无参数 = 扫模型库），所以走到这里必然至少有一个位置参数。
	// 原先那段 `if flag.NArg() < 1` 是死代码 —— 一段永不执行的守卫
	// 读起来像"缺参数已经处理了"，实际永远不会触发。
	// 与 scanLine 里那段被删掉的兜底是同一类。

	// **目录参数走目录分支**：直接丢给 parser.Parse 会得到
	// "读取文件头: read /tmp/x: is a directory" 这种对用户没有帮助的
	// 报错，而 spec §3 明列了 `modelview <dir>`。
	if st, err := os.Stat(flag.Arg(0)); err == nil && st.IsDir() {
		return runDir(context.Background(), flag.Arg(0), *asJSON)
	}

	m, err := parser.Parse(flag.Arg(0))
	if err != nil {
		return err
	}

	if *asStats {
		warnNoSample(m, *sampleLimit)
		if err := runStats(m, *sampleLimit, *noCache); err != nil {
			return err
		}
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		// 匿名嵌入 Model 会把它的字段全部提升到顶层，再加一个 param_count，
		// 而不用给 Model 加一个会和 TotalParams() 漂移的字段。
		out := struct {
			*model.Model
			ParamCount int64 `json:"param_count"`
		}{m, m.TotalParams()}
		if err := enc.Encode(out); err != nil {
			return fmt.Errorf("序列化 JSON: %w", err)
		}
		return nil
	}

	printSummary(m, *asStats)
	return nil
}

// runStats 计算统计并把失败的张量报到 stderr。
//
// 失败不中断：解析或解码某个张量出错时，其余张量照常显示，
// 这里只负责把失败清单说清楚 —— 静默少几个张量的统计比报错更糟。
func runStats(m *model.Model, sampleLimit int, noCache bool) error {
	limit := sampleLimit
	if limit == 0 {
		limit = -1 // 0 表示不采样
	}
	errs, err := analyze.Analyze(context.Background(), m, analyze.Options{
		SampleLimit: limit,
		NoCache:     noCache,
	})
	if err != nil {
		return fmt.Errorf("统计: %w", err)
	}
	if len(errs) > 0 {
		fmt.Fprintf(os.Stderr, "modelview: %d 个张量统计失败\n", len(errs))
		names := make([]string, 0, len(errs))
		for name := range errs {
			names = append(names, name)
		}
		slices.Sort(names)
		for _, name := range names {
			fmt.Fprintf(os.Stderr, "  %s: %v\n", name, errs[name])
		}
	}
	return nil
}

// printSummary 打印人类可读的摘要。
// 元数据与张量全部列出，不做过滤。
func printSummary(m *model.Model, withStats bool) {
	fmt.Printf("文件     %s\n", m.Path)
	fmt.Printf("格式     %s %s\n", m.Format, m.Version)
	fmt.Printf("大小     %s\n", humanize.Bytes(m.FileSize))
	fmt.Printf("架构     %s\n", orDash(m.Arch))
	fmt.Printf("元数据   %d 条\n", len(m.Metadata))
	fmt.Printf("张量     %d 个\n", len(m.Tensors))
	// 精确值 + SI 前缀缩写。刻意不用 "B" ——
	// 上一行的 GiB 已经让读者把 B 理解成字节，3.086 B 会被读成"3 个字节"。
	fmt.Printf("总参数   %s（%s）\n", humanize.Count(m.TotalParams()), humanize.Comma(m.TotalParams()))

	if withStats {
		ok, failed := 0, 0
		for _, t := range m.Tensors {
			if t.Stats != nil {
				ok++
			} else {
				failed++
			}
		}
		fmt.Printf("统计     %d 个张量已统计", ok)
		if failed > 0 {
			fmt.Printf("，%d 个失败", failed)
		}
		fmt.Println()
	}

	if m.Alignment > 0 {
		fmt.Printf("对齐     %d 字节（数据区起点 %d）\n", m.Alignment, m.DataStart)
	}

	if n := len(m.Warnings); n > 0 {
		fmt.Printf("\n⚠ 告警（%d 条）\n", n)
		for _, w := range m.Warnings {
			fmt.Printf("  %s\n", w)
		}
	}

	// 存储块与权重绑定：只有真的存在别名时才展示，否则是噪音。
	//
	// 判据必须用 TiedGroups 而不是"两个数字不相等"。
	// StorageBytes 的语义随格式变（见 model.Model 的说明）：safetensors
	// 下它是数据区跨度，头部允许 data_offsets 不从 0 开始，所以它**可能
	// 大于**张量字节和 —— 那时打印出来就是"去重后比去重前更大"加一个
	// 负数差值，而下面一行权重绑定是空的，自相矛盾。
	//
	// 不解释这个差值，用户会以为算错了；解释错了更糟。
	if len(m.TiedGroups) > 0 {
		fmt.Printf("\n存储     张量逻辑字节 %s，去重后 %s（差 %s，即别名重复计入的部分）\n",
			humanize.Bytes(m.TensorBytes()), humanize.Bytes(m.StorageBytes),
			humanize.Bytes(m.TensorBytes()-m.StorageBytes))
		for _, g := range m.TiedGroups {
			fmt.Printf("         权重绑定: %s\n", strings.Join(g, " ≡ "))
		}
	}

	hist := m.DtypeHistogram()
	dtypes := make([]model.Dtype, 0, len(hist))
	for d := range hist {
		dtypes = append(dtypes, d)
	}
	slices.Sort(dtypes)
	fmt.Printf("\n类型分布\n")
	for _, d := range dtypes {
		note := ""
		if bpw := d.BitsPerWeight(); bpw > 0 {
			note = fmt.Sprintf("  %s bit/权重", render.BitsPerWeight(bpw))
		}
		fmt.Printf("  %-10s %4d 个张量%s\n", d, hist[d], note)
	}

	fmt.Printf("\n元数据（全部 %d 条）\n", len(m.Metadata))
	for _, kv := range m.Metadata {
		fmt.Printf("  %-44s %s\n", kv.Key, kv.Value)
	}

	fmt.Printf("\n张量（全部 %d 个）\n", len(m.Tensors))
	for _, t := range m.Tensors {
		if withStats && t.Stats != nil {
			extra := render.TensorQuant(t)
			if extra != "" {
				fmt.Printf("  %-56s %-20s %-8s %12s  %s  %s\n",
					t.Name, humanize.Dims(t.Dims), t.Dtype, humanize.Bytes(t.ByteSize),
					render.Stats(t.Stats), extra)
				continue
			}
			fmt.Printf("  %-56s %-20s %-8s %12s  %s\n",
				t.Name, humanize.Dims(t.Dims), t.Dtype, humanize.Bytes(t.ByteSize),
				render.Stats(t.Stats))
			continue
		}
		fmt.Printf("  %-56s %-20s %-8s %12s\n",
			t.Name, humanize.Dims(t.Dims), t.Dtype, humanize.Bytes(t.ByteSize))
	}
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

// readBuildInfo 是 debug.ReadBuildInfo 的间接层。
//
// 存在的唯一理由是**可测**。测试二进制里 Main.Version 恒为 "(devel)"、
// 也没有 vcs.revision（实测确认过），所以"版本号是从构建信息推出来的"
// 这件事在端到端层面**不可观测**：把实现换成一句硬编码的常量，
// `--version` 打出来的字符串一模一样，任何断言都区分不了。
// 留一个可替换的入口，测试才能把构造好的构建信息喂进**生产路径上
// 真的会被调用的那个函数**（而不是喂给一个只有测试在用的纯函数）。
var readBuildInfo = debug.ReadBuildInfo

// version 返回 --version 要打给用户的版本号。
//
// 原先这里是一句硬编码的 "dev"：发布出去之后，用户报 bug 时给出的版本
// 永远对应不到任何一个提交。
//
// **有版本号就只报版本号，没有才去拼提交号**。本仓库里 `go build`
// 出来的 Main.Version 不是 "(devel)"，而是 Go 合成的伪版本
// `v0.0.0-20261002132043-ec74684ce6d7`（实测），里面已经含了提交 ——
// 再附一个 " (ec74684)" 就是同一件事说两遍，实测打出过
// `modelview v0.0.0-20261002132043-ec74684ce6d7 (ec74684, dirty)`。
func version() string {
	bi, ok := readBuildInfo()
	if !ok {
		// 拿不到构建信息（极少见）时退到 "dev"，不编一个版本号出来
		return "dev"
	}
	v := bi.Main.Version
	if v != "" && v != "(devel)" && !isPseudoVersion(v) {
		return v
	}
	// 本地构建：报 bug 时"哪个提交、有没有改过"比"什么版本"有用
	rev := buildSetting(bi, "vcs.revision")
	if rev == "" {
		return "dev"
	}
	if len(rev) > 7 {
		rev = rev[:7]
	}
	if buildSetting(bi, "vcs.modified") == "true" {
		rev += ", dirty"
	}
	return "dev (" + rev + ")"
}

// isPseudoVersion 判断这是不是 Go 给"没有 tag 的提交"合成的伪版本。
//
// **只认最常见的 `v0.0.0-` 那一种**（仓库还没打过任何 tag 时）。
// tag 之后构建的伪版本形如 `v1.2.4-0.2026…-abc123`，这里放它过去 ——
// 那种写法本身就说明了"在 v1.2.3 之后、某个提交上"，不翻译反而信息更多。
func isPseudoVersion(v string) bool {
	return strings.HasPrefix(v, "v0.0.0-")
}

// buildSetting 取一条构建设置，没有就返回空串。
func buildSetting(bi *debug.BuildInfo, key string) string {
	for _, s := range bi.Settings {
		if s.Key == key {
			return s.Value
		}
	}
	return ""
}

// warnNoSample 在 `--sample-limit 0`（不采样）时先给出预估峰值。
//
// **不采样是唯一会爆炸的路径**：默认 1e7 采样时峰值与模型总大小无关
// （实测 qwen2.5:3b 全量 335 MiB），而不采样时峰值与**最大那张张量**
// 的元素数成正比。实测（优化后）3.11 亿元素的张量 → 2.44 GiB，
// 约 8 B/元素（out 的 4 B 加 GC 系数）。
//
// 帮助文本里那句"大模型上峰值内存可达数 GB"是泛泛的，用户在看到自己
// 这个文件的数字之前不会把它和自己联系起来。
//
// **只提示、不阻止**：语义是用户明确选的，拦下来反而堵死了"小模型上
// 确实想要全量精确值"这条正常用法。
//
// 阈值 1 GiB 是为了不让小模型也刷一行 —— 那时提示只是噪音。
func warnNoSample(m *model.Model, limit int) {
	if limit != 0 {
		return
	}
	var peak int64
	for _, tn := range m.Tensors {
		if tn.ParamCount > peak {
			peak = tn.ParamCount
		}
	}
	const bytesPerElem = 8 // 实测标定，见上
	if est := peak * bytesPerElem; est < 1<<30 {
		return
	}
	fmt.Fprintf(os.Stderr,
		"modelview: --sample-limit 0 不采样；最大张量 %s 个元素，预估峰值内存约 %s\n",
		humanize.Count(peak), humanize.Bytes(peak*bytesPerElem))
}
