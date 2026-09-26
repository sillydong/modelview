//go:build ignore

// compare_dequant 把 internal/decode 的量化块解码结果与 llama.cpp 参考实现的真值
// 逐值比对。
//
// 用法：
//
//	go run tools/compare_dequant.go
//	go run tools/compare_dequant.go Q4_K      # 只比一种
//
// 真值由 tools/verify_ggml_dequant.py 生成（来源是 gguf.quants）。
// 日常回归由 internal/decode 的 vectors_test.go 读同一份 JSON 保证；
// 这个程序补的是"出错时能看出第几个元素不符"——断言只给行号，
// 排查块布局问题时不够用。
package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/sillydong/modelview/internal/decode"
	"github.com/sillydong/modelview/internal/model"
)

// tolerance 是逐值比对允许的绝对偏差。
//
// 参考实现走 numpy 的 float32 向量化运算，本实现是标量循环，
// 中间结果的舍入路径不同。实测两者在 1e-8 量级，1e-5 足够宽。
const tolerance = 1e-5

const vectorsPath = "internal/decode/testdata/vectors.json"

type vector struct {
	BlockSize int       `json:"block_size"`
	TypeSize  int       `json:"type_size"`
	Bytes     string    `json:"bytes"`
	Expect    []float64 `json:"expect"`
}

func main() {
	raw, err := os.ReadFile(vectorsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "读 %s 失败: %v\n", vectorsPath, err)
		fmt.Fprintln(os.Stderr, "用 uv run --with gguf --with numpy python3 tools/verify_ggml_dequant.py 生成")
		os.Exit(1)
	}
	var all map[string]vector
	if err := json.Unmarshal(raw, &all); err != nil {
		fmt.Fprintf(os.Stderr, "解析向量失败: %v\n", err)
		os.Exit(1)
	}

	names := make([]string, 0, len(all))
	for n := range all {
		if len(os.Args) > 1 && os.Args[1] != n {
			continue
		}
		names = append(names, n)
	}
	sort.Strings(names)
	if len(names) == 0 {
		fmt.Fprintln(os.Stderr, "没有可比的类型")
		os.Exit(1)
	}

	failed := 0
	for _, name := range names {
		if !compare(name, all[name]) {
			failed++
		}
	}

	if failed > 0 {
		fmt.Printf("\n%d/%d 种类型不符\n", failed, len(names))
		os.Exit(1)
	}
	fmt.Printf("\n%d 种类型全部吻合\n", len(names))
}

func compare(name string, v vector) bool {
	src, err := hex.DecodeString(v.Bytes)
	if err != nil {
		fmt.Printf("%-6s ✗ 向量不是合法 hex: %v\n", name, err)
		return false
	}
	dst := make([]float32, len(v.Expect))
	if err := decode.Decode(model.Dtype(name), src, dst); err != nil {
		fmt.Printf("%-6s ✗ %v\n", name, err)
		return false
	}

	bad, worst, worstAt := 0, 0.0, -1
	for i := range dst {
		d := math.Abs(float64(dst[i]) - v.Expect[i])
		if d > worst {
			worst, worstAt = d, i
		}
		if d > tolerance {
			if bad < 3 {
				fmt.Printf("        [%d] got %.9g want %.9g\n", i, dst[i], v.Expect[i])
			}
			bad++
		}
	}
	if bad == 0 {
		fmt.Printf("%-6s ✓ %d 个值全部吻合\n", name, len(dst))
		return true
	}
	fmt.Printf("%-6s ✗ %d/%d 个值不符，最大偏差 %.6g（下标 %d）\n",
		name, bad, len(dst), worst, worstAt)
	return false
}
