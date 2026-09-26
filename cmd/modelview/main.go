// Command modelview 解析并展示模型文件的结构信息。
//
// 支持 GGUF、safetensors 与 PyTorch（.pt）三种格式。
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"

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
	)
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, "用法: modelview [选项] <模型文件>\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if *version {
		fmt.Println("modelview dev")
		return nil
	}

	if flag.NArg() < 1 {
		flag.Usage()
		return fmt.Errorf("缺少模型文件参数")
	}

	m, err := parser.Parse(flag.Arg(0))
	if err != nil {
		return err
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(m); err != nil {
			return fmt.Errorf("序列化 JSON: %w", err)
		}
		return nil
	}

	printSummary(m)
	return nil
}

// printSummary 打印人类可读的摘要。
// 元数据与张量全部列出，不做过滤。
func printSummary(m *model.Model) {
	fmt.Printf("文件     %s\n", m.Path)
	fmt.Printf("格式     %s %s\n", m.Format, m.Version)
	fmt.Printf("大小     %s\n", humanBytes(m.FileSize))
	fmt.Printf("架构     %s\n", orDash(m.Arch))
	fmt.Printf("元数据   %d 条\n", len(m.Metadata))
	fmt.Printf("张量     %d 个\n", len(m.Tensors))
	// 精确值 + SI 前缀缩写。刻意不用 "B" ——
	// 上一行的 GiB 已经让读者把 B 理解成字节，3.086 B 会被读成"3 个字节"。
	fmt.Printf("总参数   %s（%s）\n", humanCount(m.TotalParams()), commaInt(m.TotalParams()))

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
			humanBytes(m.TensorBytes()), humanBytes(m.StorageBytes),
			humanBytes(m.TensorBytes()-m.StorageBytes))
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
		fmt.Printf("  %-56s %-20s %-8s %12s\n",
			t.Name, dimsString(t.Dims), t.Dtype, humanBytes(t.ByteSize))
	}
}

func dimsString(dims []int64) string {
	var sb strings.Builder
	sb.WriteByte('[')
	for i, d := range dims {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(strconv.FormatInt(d, 10))
	}
	sb.WriteByte(']')
	return sb.String()
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	units := []string{"KiB", "MiB", "GiB", "TiB"}
	v := float64(n)
	for _, u := range units {
		v /= unit
		if v < unit {
			return fmt.Sprintf("%.2f %s", v, u)
		}
	}
	return fmt.Sprintf("%.2f PiB", v/unit)
}

// humanCount 用 SI 前缀（K/M/G/T，1000 进制）缩写参数量。
//
// 刻意不使用 "B"：摘要里同一屏的 humanBytes 用 "B" 表示字节，
// 两个 B 并排出现会让 "3.086 B" 被误读成三个字节。
func humanCount(n int64) string {
	if n < 1000 {
		return strconv.FormatInt(n, 10)
	}
	units := []string{"K", "M", "G", "T"}
	v := float64(n)
	for _, u := range units {
		v /= 1000
		if v < 1000 {
			return fmt.Sprintf("%.3f %s", v, u)
		}
	}
	return fmt.Sprintf("%.3f P", v/1000)
}

// commaInt 给整数加千位分隔符，用于展示精确值。
func commaInt(n int64) string {
	s := strconv.FormatInt(n, 10)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	var sb strings.Builder
	if neg {
		sb.WriteByte('-')
	}
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			sb.WriteByte(',')
		}
		sb.WriteRune(c)
	}
	return sb.String()
}
