// Command modelview 解析并展示模型文件的结构信息。
//
// 当前支持 GGUF；safetensors 与 PyTorch 见后续计划。
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
	fmt.Printf("总参数   %s\n", humanCount(m.TotalParams()))

	if v := m.Extra["alignment"]; v != "" {
		fmt.Printf("对齐     %s 字节（数据区起点 %s）\n", v, m.Extra["data_start"])
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

func humanCount(n int64) string {
	if n < 1000 {
		return strconv.FormatInt(n, 10)
	}
	units := []string{"K", "M", "B", "T"}
	v := float64(n)
	for _, u := range units {
		v /= 1000
		if v < 1000 {
			return fmt.Sprintf("%.3f %s", v, u)
		}
	}
	return fmt.Sprintf("%.3f P", v/1000)
}
