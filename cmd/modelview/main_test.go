package main

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

func TestHumanFloat(t *testing.T) {
	tests := []struct {
		in   float64
		want string
	}{
		{0, "0"},
		{1.5, "1.5000"},
		{-0.25, "-0.2500"},
		{1e-5, "1e-05"}, // 小值必须走科学计数，否则会打成 0.0000
		{1e6, "1e+06"},
	}
	for _, tt := range tests {
		if got := humanFloat(tt.in); got != tt.want {
			t.Errorf("humanFloat(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
	// 权重的典型量级必须打得出可读结果，不能全是 0.0000
	if got := humanFloat(0.026777247); got == "0.0000" {
		t.Errorf("humanFloat(0.026777247) = %q —— 权重级别的数值被打成了 0", got)
	}
	if got := humanFloat(math.NaN()); got != "NaN" {
		t.Errorf("humanFloat(NaN) = %q", got)
	}
}

func TestHumanRatio(t *testing.T) {
	tests := []struct {
		in   float64
		want string
	}{
		{0, "0"},
		{0.5, "50.00%"},
		{0.00007, "0.007%"},
		{1, "100.00%"},
	}
	for _, tt := range tests {
		if got := humanRatio(tt.in); got != tt.want {
			t.Errorf("humanRatio(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// ---- 以下 5 个测试来自 main，本分支误删过，已恢复 ----

func TestHumanBytes(t *testing.T) {
	tests := []struct {
		in   int64
		want string
	}{
		{0, "0 B"},
		{1, "1 B"},
		{1023, "1023 B"},
		{1024, "1.00 KiB"},
		{1536, "1.50 KiB"},
		{1 << 20, "1.00 MiB"},
		{1 << 30, "1.00 GiB"},
		{1929903008, "1.80 GiB"},
	}
	for _, tt := range tests {
		if got := humanBytes(tt.in); got != tt.want {
			t.Errorf("humanBytes(%d) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// humanCount 用 SI 前缀（1000 进制）且不使用 "B" ——
// 摘要里 "B" 已被 humanBytes 占用表示字节。
func TestHumanCount_不使用B后缀(t *testing.T) {
	tests := []struct {
		in   int64
		want string
	}{
		{0, "0"},
		{999, "999"},
		{1000, "1.000 K"},
		{3085938688, "3.086 G"},
		{25805936462, "25.806 G"},
		{1_000_000_000_000, "1.000 T"},
	}
	for _, tt := range tests {
		if got := humanCount(tt.in); got != tt.want {
			t.Errorf("humanCount(%d) = %q, want %q", tt.in, got, tt.want)
		}
	}

	// 显式防回归：任何量级都不应出现 "B"
	for _, n := range []int64{1e9, 1e10, 1e11, 1e12, 1e13} {
		if got := humanCount(n); got[len(got)-1] == 'B' {
			t.Errorf("humanCount(%d) = %q 以 B 结尾，与 humanBytes 的 B 冲突", n, got)
		}
	}
}

func TestCommaInt(t *testing.T) {
	tests := []struct {
		in   int64
		want string
	}{
		{0, "0"},
		{1, "1"},
		{999, "999"},
		{1000, "1,000"},
		{3085938688, "3,085,938,688"},
		{1234567, "1,234,567"},
		{-1234567, "-1,234,567"},
	}
	for _, tt := range tests {
		if got := commaInt(tt.in); got != tt.want {
			t.Errorf("commaInt(%d) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestOrDash(t *testing.T) {
	if got := orDash(""); got != "-" {
		t.Errorf("orDash(\"\") = %q, want -", got)
	}
	if got := orDash("qwen2"); got != "qwen2" {
		t.Errorf("orDash(qwen2) = %q", got)
	}
}

func TestDimsString(t *testing.T) {
	tests := []struct {
		in   []int64
		want string
	}{
		{nil, "[]"},
		{[]int64{}, "[]"},
		{[]int64{4}, "[4]"},
		{[]int64{2048, 151936}, "[2048, 151936]"},
	}
	for _, tt := range tests {
		if got := dimsString(tt.in); got != tt.want {
			t.Errorf("dimsString(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// 采样过的统计必须带 ≈ 标记 —— 把样本统计量当成全量是误导。
func TestStatsString(t *testing.T) {
	s := &model.Stats{Min: -1, Max: 1, Mean: 0, Std: 0.5, ZeroRatio: 0.1}
	// ≈ 是多字节字符，不能用 got[0] 比 —— 必须按前缀判断
	if got := statsString(s); strings.HasPrefix(got, "≈") {
		t.Errorf("未采样不该有 ≈ 前缀: %q", got)
	}

	s.Sampled = true
	if got := statsString(s); !strings.HasPrefix(got, "≈") {
		t.Errorf("采样过的统计应带 ≈ 前缀: %q", got)
	}
}

// 非有限值必须出现在人读输出里。
//
// 少了这个，一个**全是 NaN** 的张量会显示成 [0, 0] μ=0 σ=0 零=0，
// 与真正的零张量长得一模一样 —— 用户根本看不出这份统计是废的。
func TestStatsString_非有限值必须显示(t *testing.T) {
	s := &model.Stats{Min: 0, Max: 0, Mean: 0, Std: 0, NaN: 4}
	got := statsString(s)
	if !strings.Contains(got, "NaN=4") {
		t.Errorf("全 NaN 张量必须显示 NaN 计数，实际: %q", got)
	}

	s = &model.Stats{Min: 1, Max: 1, Mean: 1, Inf: 3}
	if got := statsString(s); !strings.Contains(got, "Inf=3") {
		t.Errorf("含 Inf 的张量必须显示 Inf 计数，实际: %q", got)
	}

	// 没有非有限值时不要加噪音
	s = &model.Stats{Min: 1, Max: 2, Mean: 1.5, Std: 0.5}
	if got := statsString(s); strings.Contains(got, "NaN") || strings.Contains(got, "Inf") {
		t.Errorf("无非有限值时不显示，实际: %q", got)
	}
}

// 模拟那一行必须带「模拟」字样 —— 它不含重要性矩阵加权，
// 不加标注用户会拿它去对真实文件。
func TestQuantSimString(t *testing.T) {
	sims := []model.QuantSim{
		{Target: "Q8_0", BitsPerWeight: 8.5, SNRDB: 45.2, Compression: 3.76},
		{Target: "Q6_K", BitsPerWeight: 6.5625, SNRDB: 35.1, Compression: 4.88},
		{Target: "Q4_K", BitsPerWeight: 4.5, SNRDB: 24.8, Compression: 7.11},
	}
	got := quantSimString(sims)
	if !strings.Contains(got, "模拟") {
		t.Errorf("必须标出这是模拟值（未用重要性矩阵）: %q", got)
	}
	for _, s := range sims {
		if !strings.Contains(got, s.Target) {
			t.Errorf("缺少 %s 档: %q", s.Target, got)
		}
		// 位宽要用 bit/权重 而不是 B/权重 —— B 在本输出里已经是字节
		if !strings.Contains(got, "bit/权重") {
			t.Errorf("%s 的位宽单位不是 bit/权重: %q", s.Target, got)
		}
		// 数值必须真的出现在输出里，不能只有一个档名
		if !strings.Contains(got, fmt.Sprintf("%.1f", s.SNRDB)) {
			t.Errorf("%s 的信噪比 %v 没出现在输出里: %q", s.Target, s.SNRDB, got)
		}
		if !strings.Contains(got, fmt.Sprintf("%.2f", s.Compression)) {
			t.Errorf("%s 的压缩比 %v 没出现在输出里: %q", s.Target, s.Compression, got)
		}
	}
	if quantSimString(nil) != "" {
		t.Error("没有模拟结果时不该输出内容")
	}
}

// 采样的模拟要带 ≈ —— 与 statsString 同一个约定。
func TestQuantSimString_采样标记(t *testing.T) {
	full := quantSimString([]model.QuantSim{{Target: "Q8_0", SNRDB: 45.2, Compression: 3.76}})
	if strings.Contains(full, "≈") {
		t.Errorf("全量模拟不该带 ≈: %q", full)
	}
	sampled := quantSimString([]model.QuantSim{
		{Target: "Q8_0", SNRDB: 45.2, Compression: 3.76, Sampled: true},
	})
	if !strings.Contains(sampled, "≈") {
		t.Errorf("采样模拟必须带 ≈: %q", sampled)
	}
}

func TestQuantExistingString(t *testing.T) {
	q := &model.QuantInfo{
		Scheme: "Q4_K", BitsPerWeight: 4.5, Blocks: 100, SubBlocks: 800,
		BlockElems: 32, ScaleMin: 0.001, ScaleMax: 0.5, ScaleMedian: 0.02,
		ZeroScaleBlocks: 3,
	}
	got := quantExistingString(q)
	if !strings.Contains(got, "Q4_K") || !strings.Contains(got, "4.5 bit/权重") {
		t.Errorf("缺少方案或位宽: %q", got)
	}
	if !strings.Contains(got, "800") {
		t.Errorf("子块数没显示: %q", got)
	}
	// 单位必须是"子块"：同一行前面刚写过"800 子块"，
	// 而一个块含 8 或 16 个子块，写成"块"会差一个数量级
	if !strings.Contains(got, "压平 3 子块") {
		t.Errorf("被压平的子块数必须显示且单位正确 —— 那是最直接的证据: %q", got)
	}
	// 被压得最狠的那个子块也必须显示（QuantInfo 的注释承诺了这一点）
	if !strings.Contains(got, "最扁") {
		t.Errorf("最扁的子块没显示: %q", got)
	}
	if quantExistingString(nil) != "" {
		t.Error("nil 时不该输出内容")
	}
	// 没有被压平的块时不该出现警告
	q2 := &model.QuantInfo{Scheme: "Q8_0", BitsPerWeight: 8.5, SubBlocks: 2,
		ScaleMin: 0.0625, ScaleMax: 0.5, ScaleMedian: 0.2}
	if got := quantExistingString(q2); strings.Contains(got, "⚠") {
		t.Errorf("没有被压平的块时不该有警告: %q", got)
	}
}

// 张量行末尾挂哪一段：模拟与诊断互斥，都为空时不留空档。
func TestTensorQuantString(t *testing.T) {
	sims := []model.QuantSim{{Target: "Q8_0", SNRDB: 45.2, Compression: 3.76}}
	quant := &model.QuantInfo{Scheme: "Q4_K", BitsPerWeight: 4.5, SubBlocks: 8,
		ScaleMin: 0.1, ScaleMax: 0.5, ScaleMedian: 0.2}

	// 浮点张量：只有模拟
	got := tensorQuantString(&model.Tensor{QuantSims: sims})
	if !strings.Contains(got, "模拟") || strings.Contains(got, "Q4_K ") {
		t.Errorf("浮点张量应只显示模拟: %q", got)
	}
	// 量化张量：只有诊断
	got = tensorQuantString(&model.Tensor{Quant: quant})
	if strings.Contains(got, "模拟") || !strings.Contains(got, "Q4_K") {
		t.Errorf("量化张量应只显示诊断: %q", got)
	}
	// 都没有：空串，调用方不该多打一个空格
	if got := tensorQuantString(&model.Tensor{}); got != "" {
		t.Errorf("两者皆无时应返回空串，实际 %q", got)
	}
	// 都有（不该发生）：显示模拟，不能同时给两个互相矛盾的数字
	got = tensorQuantString(&model.Tensor{QuantSims: sims, Quant: quant})
	if !strings.Contains(got, "模拟") {
		t.Errorf("两者都有时应优先显示模拟: %q", got)
	}
}
