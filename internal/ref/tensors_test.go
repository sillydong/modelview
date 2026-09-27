package ref

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// 本机真实模型里出现的张量段，每一个都要能查到释义。
//
// 清单是 tools/extract_tensor_segments.py 从真实文件生成的
// **入库产物**（internal/ref/testdata/real_segments.txt），与元数据键同一套做法。
//
// 这里原来是 23 个段的手写字面量，注释却写着"由脚本生成"——
// 脚本当时并不存在，而手写清单实测只覆盖 73 个真实段里的 24 个，
// 漏掉的正好是多模态塔、量化标定、Gemma 3n 结构件那几批。
//
// 与 real_test.go 那条的关系：那条拿**本机当前**的文件现场算一遍（能发现
// 上游格式变化），这条拿**入库的清单**（在没有模型的机器上也能跑）。
// 两条同时绿才说明"清单与当前文件一致、且表覆盖了清单"。
func TestTensorNaming_覆盖真实张量名(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "real_segments.txt"))
	if err != nil {
		t.Fatalf("读段名清单失败（用 tools/extract_tensor_segments.py 重新生成）: %v", err)
	}
	var missing []string
	total := 0
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		// 注释行是 "# "（井号加空格）；**不能只判 "#"** ——
		// "#N" 本身就是一个真实的段名（层号），判错了会把它静默跳过，
		// 而总数断言那时会红得莫名其妙
		if line == "" || strings.HasPrefix(line, "# ") {
			continue
		}
		total++
		if _, ok := lookupTensorSegment(line); !ok {
			missing = append(missing, line)
		}
	}
	if len(missing) > 0 {
		t.Errorf("真实文件里的张量段查不到释义（%d 个）：%s",
			len(missing), strings.Join(missing, "、"))
	}
	// 反向门禁：清单被截断时上面那条会"零缺失"地通过
	const wantSegs = 73 // 4 个本机模型实测的并集；清单变了核对后再改
	if total != wantSegs {
		t.Errorf("清单有 %d 段，预期 %d 段 —— 要么清单被截断了，"+
			"要么上游模型换了（那就重新跑一次脚本并核对这个数）", total, wantSegs)
	}
}

// 拆分器本身：数字段要被识别成**层号**而不是查询键。
func TestSplitTensorName(t *testing.T) {
	tests := []struct {
		in   string
		want []string
	}{
		{"blk.0.attn_q.weight", []string{"blk", "#0", "attn_q", "weight"}},
		{"blk.12.ffn_down.weight", []string{"blk", "#12", "ffn_down", "weight"}},
		{"token_embd.weight", []string{"token_embd", "weight"}},
		{"output_norm.bias", []string{"output_norm", "bias"}},
		{"blk.3.attn_norm.weight", []string{"blk", "#3", "attn_norm", "weight"}},
	}
	for _, tt := range tests {
		got := SplitTensorName(tt.in)
		if len(got) != len(tt.want) {
			t.Errorf("SplitTensorName(%q) = %v, want %v", tt.in, got, tt.want)
			continue
		}
		for i := range got {
			if got[i] != tt.want[i] {
				t.Errorf("SplitTensorName(%q) = %v, want %v", tt.in, got, tt.want)
				break
			}
		}
		// 拆出来的每一段都必须能查到。层号段是 "#12" 这种具体值，
		// 而表里的条目是 "#N" —— 查之前要归一，否则这条断言永远红
		for _, seg := range got {
			if _, ok := LookupTensorSegment(seg); !ok {
				t.Errorf("%q 拆出的段 %q 查不到释义", tt.in, seg)
			}
		}
	}
}

// 层号段必须有自己的释义（用户最常问"blk.12 是什么意思"）。
func TestTensorNaming_层号条目(t *testing.T) {
	e, ok := lookupTensorSegment("#N")
	if !ok {
		t.Fatal("没有层号段 #N 的释义")
	}
	if !strings.Contains(e.meaning, "层") {
		t.Errorf("层号段的释义不像在说层号: %q", e.meaning)
	}
}

// 空段要被丢掉，不能产生无意义的 "#"。
func TestSplitTensorName_空段(t *testing.T) {
	for _, in := range []string{"", ".", "..", "a..b"} {
		for _, seg := range SplitTensorName(in) {
			if seg == "" || seg == "#" {
				t.Errorf("SplitTensorName(%q) 产生了空段 %q", in, seg)
			}
		}
	}
}

// 未实测段清单必须与真实语料**精确**对得上。
//
// 这张清单是"哪些释义是实测的、哪些是抄来的"的唯一记录 ——
// 没有任何东西守着它的话，它会随着增删条目慢慢变成谎言，
// 而谎报"验证过"比不验证更糟。
func TestTensorNaming_未实测段清单准确(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "real_segments.txt"))
	if err != nil {
		t.Fatal(err)
	}
	real := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "# ") {
			real[line] = true
		}
	}

	// 现算一遍"表里有、语料里没有"的集合
	actual := map[string]bool{}
	for _, seg := range tensorSegments {
		if !real[seg.seg] {
			actual[seg.seg] = true
		}
	}
	for seg := range actual {
		if !segmentsNotInCorpus[seg] {
			t.Errorf("段 %q 在语料里没出现过，但没被记进 segmentsNotInCorpus —— "+
				"它的释义来自哪里？实测还是抄的？", seg)
		}
	}
	for seg := range segmentsNotInCorpus {
		if !actual[seg] {
			t.Errorf("段 %q 被记成「语料里没有」，但它（现在）出现了 —— "+
				"把它从 segmentsNotInCorpus 里去掉，并订正它的释义", seg)
		}
	}

	// **精确计数**：上面两条只保证"两个集合相等"，而"两边同时删一项"
	// 对相等关系没有影响 —— 实测：同时删掉 tensorSegments 的 laurel
	// 与 segmentsNotInCorpus 的 laurel，测试照样绿，而注释里写着
	// "多一个少一个都红"。规模要单独钉住。
	const (
		wantSegs      = 86 // tensorSegments 的条目数
		wantNotInCorp = 13 // 其中语料里没出现过的
	)
	if len(tensorSegments) != wantSegs {
		t.Errorf("张量段有 %d 条，预期 %d 条 —— 增删条目时同步改这里，"+
			"否则删掉一条不会有任何东西红", len(tensorSegments), wantSegs)
	}
	if len(segmentsNotInCorpus) != wantNotInCorp {
		t.Errorf("未实测清单有 %d 条，预期 %d 条", len(segmentsNotInCorpus), wantNotInCorp)
	}
}
