//go:build ignore

// compare_uniform_formula 量化「统一公式」与真实编码算法的差距，
// 用来核对 spec §6.2 里那句"必须用真实算法"的依据。
//
// 用法：
//
//	go run tools/compare_uniform_formula.go
//
// ## 背景
//
// spec §6.2 最初写的是一套统一公式：`scale = 块内最大绝对值 / 级数`。
// 改成真实编码算法（plan ③b）时给出的理由是"统一公式乐观 4.1 dB"。
// 本程序复核这个理由，结论是**当初那个数字不可靠**：
//
// 同一个张量、同一批权重，统一公式的结果随"级数怎么定、块多大"摆动 6 dB 以上：
//
//	31 级 / 32 元素块   31.27 dB   （比真实 Q6_K 差 2.28 dB）
//	63 级 / 32 元素块   37.53 dB   （当初用的，好 3.97 dB）
//	31 级 / 16 元素块   33.15 dB   （只差 0.41 dB）
//
// 关键在级数：Q6_K 的 6 位码是 [-32, 31] 的**有符号偏移**，
// 对称重建每侧只有 31~32 级；按 63 级算等于给了它实际两倍的精度。
//
// 所以采用真实算法的理由不是"公式乐观"，而是**公式的结果依赖一个
// 任意的建模选择**，摆动比它自身的误差还大 —— 拿它做量化决策没有意义。
// 真实编码算法没有这个自由度。
package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/sillydong/modelview/internal/decode"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/parser"
)

const (
	nBlocks = 512 // 取多少个 256 元素块
	elems   = 256
)

// uniformQuantizeN 是统一公式的通用版：块内最大绝对值 / levels，块大小可配。
func uniformQuantizeN(vals []float32, levels, block int) []float32 {
	out := make([]float32, len(vals))
	for b := 0; b+block <= len(vals); b += block {
		seg := vals[b : b+block]
		var amax float32
		for _, v := range seg {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		scale := float64(amax) / float64(levels)
		if scale == 0 {
			continue
		}
		for i, v := range seg {
			out[b+i] = float32(math.Round(float64(v)/scale) * scale)
		}
	}
	return out
}

// snr 返回（信噪比 dB, 最大绝对误差, 平均绝对误差）。
func snr(orig, back []float32) (float64, float64, float64) {
	var sumSq, sumErrSq, sumAbs, maxAbs float64
	for i := range orig {
		x := float64(orig[i])
		e := math.Abs(x - float64(back[i]))
		sumSq += x * x
		sumErrSq += e * e
		sumAbs += e
		maxAbs = math.Max(maxAbs, e)
	}
	if sumErrSq == 0 || sumSq == 0 {
		return 0, maxAbs, sumAbs / float64(len(orig))
	}
	return 10 * math.Log10(sumSq/sumErrSq), maxAbs, sumAbs / float64(len(orig))
}

// realEncode 用真实编码器编码再解码。
func realEncode(d model.Dtype, vals []float32) []float32 {
	perBlock, _ := d.BlockBytes()
	n := int64(len(vals)) / d.BlockElems()
	buf := make([]byte, n*perBlock)
	if err := decode.Quantize(d, vals, buf); err != nil {
		panic(err)
	}
	back := make([]float32, len(vals))
	if err := decode.Decode(d, buf, back); err != nil {
		panic(err)
	}
	return back
}

func main() {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}
	blobs, err := os.ReadDir(filepath.Join(home, ".ollama", "models", "blobs"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "读 ollama blobs 失败:", err)
		os.Exit(1)
	}

	// 找一个 F16 权重张量当语料
	var path string
	var tn *model.Tensor
	for _, e := range blobs {
		p := filepath.Join(home, ".ollama", "models", "blobs", e.Name())
		m, err := parser.Parse(p)
		if err != nil || m.Arch != "nomic-bert" {
			continue
		}
		for _, t := range m.Tensors {
			if t.Dtype == model.DtypeF16 && t.ParamCount >= int64(nBlocks*elems) {
				path, tn = p, t
				break
			}
		}
		if path != "" {
			break
		}
	}
	if path == "" {
		fmt.Fprintln(os.Stderr, "找不到 nomic-bert 的 F16 张量（需要本地 ollama 模型）")
		os.Exit(1)
	}

	f, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer func() { _ = f.Close() }()

	raw := make([]byte, nBlocks*elems*2)
	if _, err := f.ReadAt(raw, tn.Offset); err != nil {
		panic(err)
	}
	vals := make([]float32, nBlocks*elems)
	if err := decode.Decode(model.DtypeF16, raw, vals); err != nil {
		panic(err)
	}

	fmt.Printf("语料：%s，%d 个值（%d 个 256 元素块）\n\n", tn.Name, len(vals), nBlocks)

	q6 := realEncode(model.DtypeQ6K, vals)
	q6DB, q6Max, q6Mean := snr(vals, q6)
	q4 := realEncode(model.DtypeQ4K, vals)
	q4DB, _, _ := snr(vals, q4)

	fmt.Printf("%-34s %9s %12s %12s\n", "", "信噪比", "最大|误差|", "平均|误差|")
	fmt.Printf("%-34s %7.2f dB %12.6g %12.6g\n", "Q6_K 真实编码器", q6DB, q6Max, q6Mean)
	fmt.Printf("%-34s %7.2f dB\n", "Q4_K 真实编码器（对照）", q4DB)
	fmt.Println()

	for _, v := range []struct {
		name   string
		levels int
		block  int
	}{
		{"统一公式 31 级 / 32 元素块", 31, 32},
		{"统一公式 63 级 / 32 元素块", 63, 32},
		{"统一公式 31 级 / 16 元素块", 31, 16},
		{"统一公式 63 级 / 16 元素块", 63, 16},
		{"统一公式 64 级 / 32 元素块", 64, 32},
	} {
		dB, mx, mean := snr(vals, uniformQuantizeN(vals, v.levels, v.block))
		fmt.Printf("%-34s %7.2f dB %12.6g %12.6g  (差 %+.2f dB)\n",
			v.name, dB, mx, mean, dB-q6DB)
	}
	fmt.Println("\n结论：级数从 31 改成 63 就摆动 6.26 dB —— 比它自身的误差还大。")
	fmt.Println("不是「公式乐观」，是「公式的结果取决于一个任意选择」。")
}
