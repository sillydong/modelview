// Command modelview 解析并展示模型文件的结构信息。
//
// 支持 GGUF、safetensors 与 PyTorch（.pt）三种格式。
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"

	"github.com/sillydong/modelview/internal/analyze"
	"github.com/sillydong/modelview/internal/humanize"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/parser"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "modelview: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		asJSON  = flag.Bool("json", false, "以 JSON 输出（非交互）")
		version = flag.Bool("version", false, "打印版本后退出")
		asStats = flag.Bool("stats", false, "计算张量数值统计（慢，会读全部张量数据）")
		noCache = flag.Bool("no-cache", false, "禁用缓存读写")
		// 0 表示不采样（读完整个张量）；默认 1e7 个元素约 40 MB
		sampleLimit = flag.Int("sample-limit", analyze.DefaultSampleLimit,
			"单张量统计的采样上限（元素数），0 表示不采样（会读完整个张量，"+
				"大模型上峰值内存可达数 GB）。"+
				"注意它**只管统计**：量化张量的块级诊断要读遍全部块头才能"+
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
	// 宁可报错也不要猜：写 `--` 可以显式终止选项解析。
	for _, a := range flag.Args() {
		if strings.HasPrefix(a, "-") && a != "-" {
			flag.Usage()
			return fmt.Errorf("选项 %q 写在位置参数后面了，不会被解析；"+
				"选项要写在前面，例如 modelview --json scan", a)
		}
	}

	if *version {
		fmt.Println("modelview dev")
		return nil
	}

	// 无参数或显式 scan 都进模型库。
	// ④b 会把这里换成 TUI；现在先给非交互输出
	if flag.NArg() == 0 || flag.Arg(0) == "scan" {
		if !*asJSON {
			fmt.Fprintln(os.Stderr, "提示：交互界面在计划 ④b；当前为列表输出")
		}
		return runScan(context.Background(), *asJSON)
	}

	// **这里没有"缺少参数"的分支**：NArg()==0 已经在上面走了 scan
	// （无参数 = 扫模型库），所以走到这里必然至少有一个位置参数。
	// 原先那段 `if flag.NArg() < 1` 是死代码 —— 一段永不执行的守卫
	// 读起来像"缺参数已经处理了"，实际永远不会触发。
	// 与 scanLine 里那段被删掉的兜底是同一类。

	m, err := parser.Parse(flag.Arg(0))
	if err != nil {
		return err
	}

	if *asStats {
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
			note = fmt.Sprintf("  %.4g bit/权重", bpw)
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
			extra := tensorQuantString(t)
			if extra != "" {
				fmt.Printf("  %-56s %-20s %-8s %12s  %s  %s\n",
					t.Name, humanize.Dims(t.Dims), t.Dtype, humanize.Bytes(t.ByteSize),
					statsString(t.Stats), extra)
				continue
			}
			fmt.Printf("  %-56s %-20s %-8s %12s  %s\n",
				t.Name, humanize.Dims(t.Dims), t.Dtype, humanize.Bytes(t.ByteSize),
				statsString(t.Stats))
			continue
		}
		fmt.Printf("  %-56s %-20s %-8s %12s\n",
			t.Name, humanize.Dims(t.Dims), t.Dtype, humanize.Bytes(t.ByteSize))
	}
}

// tensorQuantString 决定张量行末尾挂哪一段量化信息。
//
// 两者不该同时出现：浮点张量有模拟（QuantSims）、量化张量有诊断（Quant），
// 接入层保证了互斥。同时有值时优先显示模拟 —— 那说明数据来源不寻常，
// 但宁可显示一个也不要显示两个互相矛盾的数字。
func tensorQuantString(t *model.Tensor) string {
	if s := quantSimString(t.QuantSims); s != "" {
		return s
	}
	return quantExistingString(t.Quant)
}

// quantSimString 把三档模拟压成一行。
//
// 这是按各格式**真实编码器**算出来的误差，不是统一公式的估算。
//
// 仍然标出「模拟」二字，因为还有一处差别必须让用户知道：
// 模拟是等权编码，真实文件可能是用重要性矩阵加权编的 ——
// 本工具没有那份标定数据。实测在 qwen2.5:3b 上这没造成数值差异
// （真实块 40/40 都能被等权编码器精确复现），但换一个文件就未必。
//
// 采样出来的数字前面加 ≈，与 statsString 同一个约定：
// 把样本的数字当成精确值是误导，而一行里没地方写"这是采样值"。
func quantSimString(sims []model.QuantSim) string {
	if len(sims) == 0 {
		return ""
	}
	mark := ""
	if sims[0].Sampled {
		mark = "≈"
	}
	parts := make([]string, 0, len(sims))
	for _, s := range sims {
		// 用 "bit/权重" 而不是 "B/权重"：B 在这个输出里已经是**字节**
		// （同一行就有 243.43 MiB），且"类型分布"段用的就是 bit/权重。
		// %.4g 也与那一段一致，8.5 不会打成 8.5000
		parts = append(parts, fmt.Sprintf("%s %.4g bit/权重 %.1fdB ×%.2f",
			s.Target, s.BitsPerWeight, s.SNRDB, s.Compression))
	}
	return mark + "模拟[" + strings.Join(parts, " | ") + "]"
}

// quantExistingString 把块级诊断压成一行。
//
// 位宽与压缩比不在这里重复：类型分布那一节已经按类型给过 bit/权重。
func quantExistingString(q *model.QuantInfo) string {
	if q == nil {
		return ""
	}
	out := fmt.Sprintf("%s %.4g bit/权重 %s 子块 scale[%s, %s] 中位 %s",
		q.Scheme, q.BitsPerWeight, humanize.Count(q.SubBlocks),
		humanFloat(q.ScaleMin), humanFloat(q.ScaleMax), humanFloat(q.ScaleMedian))
	// 被压平的**子块**数是最直接的证据，必须显示。
	// 单位必须写"子块"而不是"块"：同一行前面刚写过"N 子块"，
	// 而一个块含 8 或 16 个子块，写成"块"会差一个数量级
	if q.ZeroScaleBlocks > 0 {
		out += fmt.Sprintf(" ⚠压平 %s 子块", humanize.Count(q.ZeroScaleBlocks))
	}
	// 被压得最狠的那个子块。model.QuantInfo 的注释承诺了界面上要显示它
	//（"第 N 个子块（张量内第 N×BlockElems 个权重）"），
	// 只进 JSON 不显示的话那个承诺就是假的
	if q.FlattestRatio < 1 && q.FlattestRatio >= 0 {
		out += fmt.Sprintf(" 最扁 #%d（比值 %s）",
			q.FlattestIndex, humanFloat(q.FlattestRatio))
	}
	return out
}

// statsString 把统计压成一行。
//
// 采样过的数字前面加 ≈ —— 把样本统计量当成全量是误导，
// 而一行里没地方写"这是采样值"。
func statsString(s *model.Stats) string {
	mark := ""
	if s.Sampled {
		mark = "≈"
	}
	out := fmt.Sprintf("%s[%s, %s] μ=%s σ=%s 零=%s 离群=%s",
		mark, humanFloat(s.Min), humanFloat(s.Max),
		humanFloat(s.Mean), humanFloat(s.Std),
		humanRatio(s.ZeroRatio), humanRatio(s.OutlierRatio))

	// 非有限值必须显示出来。
	//
	// 它们只被计数、不参与统计，所以一个**全是 NaN** 的张量
	// Min/Max/Mean/Std 全是零值，看起来跟真正的零张量一模一样。
	// 正是 model.Stats 注释里点名要避免的那种误导，只是换了个形式。
	if s.NaN > 0 || s.Inf > 0 {
		out += fmt.Sprintf(" ⚠NaN=%d Inf=%d", s.NaN, s.Inf)
	}
	return out
}

// humanFloat 用 4 位有效数字打印统计量。
//
// 权重的动态范围常常横跨好几个数量级（1e-5 到 1e-1），
// 定点格式会让小值全变成 0.0000。
func humanFloat(v float64) string {
	switch {
	case v == 0:
		return "0"
	case math.IsNaN(v):
		return "NaN"
	case math.IsInf(v, 1):
		return "+Inf"
	case math.IsInf(v, -1):
		return "-Inf"
	}
	if a := math.Abs(v); a >= 1e-3 && a < 1e5 {
		return strconv.FormatFloat(v, 'f', 4, 64)
	}
	return strconv.FormatFloat(v, 'g', 4, 64)
}

// humanRatio 把 0..1 的比例打成百分数。
func humanRatio(v float64) string {
	if v == 0 {
		return "0"
	}
	if v < 0.0001 {
		return fmt.Sprintf("%.2g%%", v*100)
	}
	return fmt.Sprintf("%.2f%%", v*100)
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
