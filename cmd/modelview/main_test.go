package main

import (
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
