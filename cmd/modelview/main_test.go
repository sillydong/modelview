package main

import "testing"

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
