package render

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

// 采样过的统计必须带 ≈ 标记 —— 把样本统计量当成全量是误导。
func TestStatsString(t *testing.T) {
	s := &model.Stats{Min: -1, Max: 1, Mean: 0, Std: 0.5, ZeroRatio: 0.1}
	// ≈ 是多字节字符，不能用 got[0] 比 —— 必须按前缀判断
	if got := Stats(s); strings.HasPrefix(got, "≈") {
		t.Errorf("未采样不该有 ≈ 前缀: %q", got)
	}

	s.Sampled = true
	if got := Stats(s); !strings.HasPrefix(got, "≈") {
		t.Errorf("采样过的统计应带 ≈ 前缀: %q", got)
	}
}

// 非有限值必须出现在人读输出里。
//
// 少了这个，一个**全是 NaN** 的张量会显示成 [0, 0] μ=0 σ=0 零=0，
// 与真正的零张量长得一模一样 —— 用户根本看不出这份统计是废的。
func TestStatsString_非有限值必须显示(t *testing.T) {
	s := &model.Stats{Min: 0, Max: 0, Mean: 0, Std: 0, NaN: 4}
	got := Stats(s)
	if !strings.Contains(got, "NaN=4") {
		t.Errorf("全 NaN 张量必须显示 NaN 计数，实际: %q", got)
	}

	s = &model.Stats{Min: 1, Max: 1, Mean: 1, Inf: 3}
	if got := Stats(s); !strings.Contains(got, "Inf=3") {
		t.Errorf("含 Inf 的张量必须显示 Inf 计数，实际: %q", got)
	}

	// 没有非有限值时不要加噪音
	s = &model.Stats{Min: 1, Max: 2, Mean: 1.5, Std: 0.5}
	if got := Stats(s); strings.Contains(got, "NaN") || strings.Contains(got, "Inf") {
		t.Errorf("无非有限值时不显示，实际: %q", got)
	}
}

// **整串精确断言，不是 Contains**。
//
// 用 Contains 的话，把 μ 与 σ 的取值互换、把 Min 与 Max 互换、
// 把 "μ=" 改成 "x=" —— 三种都会全绿（实测），因为
// "这些数字出现过"与"这个数字挂在这个标签下"是两回事。
// 而这个包的**全部意义**就是"两处显示同一个数"，
// 挂错标签等于两个数。
func TestStats_整串精确(t *testing.T) {
	s := &model.Stats{Count: 8, Min: -1.5, Max: 2.5, Mean: 0.125, Std: 0.75,
		ZeroRatio: 0.25, OutlierRatio: 0.125}
	// want 是先按约定手算、再与实际输出核对得到的，不是从实现里抄的：
	// Float 在 [1e-3, 1e5) 内走定点 4 位小数（-1.5 → "-1.5000"，ASCII 减号），
	// Percent 走 "%.2f%%"（0.25 → "25.00%"）。
	// 手算时这里踩过两次：Min 不是 "-1.5"（少一位），减号也不是排印的 "−"。
	want := "[-1.5000, 2.5000] μ=0.1250 σ=0.7500 零=25.00% 离群=12.50%"
	if got := Stats(s); got != want {
		t.Errorf("Stats() =\n got %q\nwant %q —— 数字与标签的对应关系变了", got, want)
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
	got := QuantSim(sims)
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
	if QuantSim(nil) != "" {
		t.Error("没有模拟结果时不该输出内容")
	}
}

// 采样的模拟要带 ≈ —— 与 Stats 同一个约定。
func TestQuantSimString_采样标记(t *testing.T) {
	full := QuantSim([]model.QuantSim{{Target: "Q8_0", SNRDB: 45.2, Compression: 3.76}})
	if strings.Contains(full, "≈") {
		t.Errorf("全量模拟不该带 ≈: %q", full)
	}
	sampled := QuantSim([]model.QuantSim{
		{Target: "Q8_0", SNRDB: 45.2, Compression: 3.76, Sampled: true},
	})
	if !strings.Contains(sampled, "≈") {
		t.Errorf("采样模拟必须带 ≈: %q", sampled)
	}
}

// QuantSim 的整串精确断言：锁住分隔符、档序、以及每个数字挂在哪一档下。
//
// 与 Stats 那条同一个理由：Contains 拦不住"分隔符从 ' | ' 改成 ' / '"，
// 也拦不住两档之间的数字互换（实测都是绿的）。
func TestQuantSim_整串精确(t *testing.T) {
	sims := []model.QuantSim{
		{Target: "Q8_0", BitsPerWeight: 8.5, SNRDB: 45.2, Compression: 3.76},
		{Target: "Q6_K", BitsPerWeight: 6.5625, SNRDB: 35.1, Compression: 4.88},
		{Target: "Q4_K", BitsPerWeight: 4.5, SNRDB: 24.8, Compression: 7.11},
	}
	// 同样先手算：位宽是精确值（6.5625 不是 6.562），
	// 档与档之间是 " | "（竖线前后各一个空格）
	want := "模拟[Q8_0 8.5 bit/权重 45.2dB ×3.76 | " +
		"Q6_K 6.5625 bit/权重 35.1dB ×4.88 | " +
		"Q4_K 4.5 bit/权重 24.8dB ×7.11]"
	if got := QuantSim(sims); got != want {
		t.Errorf("QuantSim() =\n got %q\nwant %q", got, want)
	}

	// ≈ 的位置也是整串的一部分：必须在 "模拟[" 之前，不能只在串里某个地方
	sampled := []model.QuantSim{{Target: "Q8_0", BitsPerWeight: 8.5, SNRDB: 45.2,
		Compression: 3.76, Sampled: true}}
	wantSampled := "≈模拟[Q8_0 8.5 bit/权重 45.2dB ×3.76]"
	if got := QuantSim(sampled); got != wantSampled {
		t.Errorf("采样 QuantSim() =\n got %q\nwant %q", got, wantSampled)
	}
}

func TestQuantExistingString(t *testing.T) {
	q := &model.QuantInfo{
		Scheme: "Q4_K", BitsPerWeight: 4.5, Blocks: 100, SubBlocks: 800,
		BlockElems: 32, ScaleMin: 0.001, ScaleMax: 0.5, ScaleMedian: 0.02,
		ZeroScaleBlocks: 3,
	}
	got := QuantExisting(q)
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
	if QuantExisting(nil) != "" {
		t.Error("nil 时不该输出内容")
	}
	// 没有被压平的块时不该出现警告
	q2 := &model.QuantInfo{Scheme: "Q8_0", BitsPerWeight: 8.5, SubBlocks: 2,
		ScaleMin: 0.0625, ScaleMax: 0.5, ScaleMedian: 0.2}
	if got := QuantExisting(q2); strings.Contains(got, "⚠") {
		t.Errorf("没有被压平的块时不该有警告: %q", got)
	}
}

// 张量行末尾挂哪一段：模拟与诊断互斥，都为空时不留空档。
func TestTensorQuantString(t *testing.T) {
	sims := []model.QuantSim{{Target: "Q8_0", SNRDB: 45.2, Compression: 3.76}}
	quant := &model.QuantInfo{Scheme: "Q4_K", BitsPerWeight: 4.5, SubBlocks: 8,
		ScaleMin: 0.1, ScaleMax: 0.5, ScaleMedian: 0.2}

	// 浮点张量：只有模拟
	got := TensorQuant(&model.Tensor{QuantSims: sims})
	if !strings.Contains(got, "模拟") || strings.Contains(got, "Q4_K ") {
		t.Errorf("浮点张量应只显示模拟: %q", got)
	}
	// 量化张量：只有诊断
	got = TensorQuant(&model.Tensor{Quant: quant})
	if strings.Contains(got, "模拟") || !strings.Contains(got, "Q4_K") {
		t.Errorf("量化张量应只显示诊断: %q", got)
	}
	// 都没有：空串，调用方不该多打一个空格
	if got := TensorQuant(&model.Tensor{}); got != "" {
		t.Errorf("两者皆无时应返回空串，实际 %q", got)
	}
	// 都有（不该发生）：显示模拟，不能同时给两个互相矛盾的数字
	got = TensorQuant(&model.Tensor{QuantSims: sims, Quant: quant})
	if !strings.Contains(got, "模拟") {
		t.Errorf("两者都有时应优先显示模拟: %q", got)
	}
}

// 位宽是文件里的精确值，少一位就是错的 —— 这组是"%.4g 抹掉一位"的回归保护。
func TestBitsPerWeight(t *testing.T) {
	tests := []struct {
		in   float64
		want string
	}{
		// 下面四个都以 x.0625 / x.4375 结尾，正是 %.4g 抹掉一位的那批：
		// 6.5625→"6.562"、3.4375→"3.438"、2.0625→"2.062"、1.5625→"1.562"
		{6.5625, "6.5625"},
		{3.4375, "3.4375"},
		{2.0625, "2.0625"},
		{1.5625, "1.5625"},
		// 整数位宽不能因为去尾零而被打成 "32.0000"，也不能丢掉整数部分
		{4.5, "4.5"},
		{8.5, "8.5"},
		{32, "32"},
		{0, "0"},
		// 边界输入。位宽的前提是"正的、有限的、1/16 的整数倍"，
		// 这些都在前提之外，结果无意义 —— 但**必须钉住**，
		// 否则换实现时没人知道它们会变成什么：
		{math.NaN(), "NaN"},
		{math.Inf(1), "+Inf"},
		{math.Inf(-1), "-Inf"},
		{-6.5625, "-6.5625"}, // 负号会原样留着，不做 abs
		{5e-324, "0"},        // 次正规数被 4 位小数抹成 0：**非零打成零**
	}
	for _, tt := range tests {
		if got := BitsPerWeight(tt.in); got != tt.want {
			t.Errorf("BitsPerWeight(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// QuantSimLines 给 TUI 的列对齐版：档名左对齐、位宽右对齐。
//
// **档名必须长短不一**（IQ2_XXS 7 个字符，Q8_0/Q4_K 4 个）：
// 原先三个 fixture 档名都是 4 个字符，wTarget 恰好等于每行的档名长度，
// 于是把档名补齐整个删掉（`fmt.Sprintf("%-*s", wTarget, s.Target)`
// 换成 `s.Target`）测试**仍然全绿**（实测漏过一轮）—— 左对齐那一半
// 等于没有验。位宽右对齐那一半当时是被抓住的，漏的只有档名。
//
// IQ2_XXS 顺带把 2.0625 这个位宽拉进单元覆盖：本机没有 IQ 量化的
// 真实文件，它此前只有 TestBitsPerWeight 的表驱动覆盖。
func TestQuantSimLines(t *testing.T) {
	sims := []model.QuantSim{
		{Target: "IQ2_XXS", BitsPerWeight: 2.0625, SNRDB: 20.9, Compression: 7.11},
		{Target: "Q8_0", BitsPerWeight: 8.5, SNRDB: 30.1, Compression: 3.76},
		{Target: "Q4_K", BitsPerWeight: 4.5, SNRDB: 24.8, Compression: 5.33},
	}
	lines := QuantSimLines(sims)
	// want 由列宽手推，不是跑出来抄的：档名列宽 = max(7,4,4) = 7，
	// 位宽列宽 = max(len("2.0625"), len("8.5"), len("4.5")) = 6。
	// 所以短档名那两行是 3 个补齐空格 + 1 个分隔空格 + 3 个位宽右对齐空格
	// = 档名与数字之间 7 个空格；最长的 IQ2_XXS 两列都顶满，各只隔 1 个。
	want := []string{
		"IQ2_XXS 2.0625 bit/权重 20.9dB ×7.11",
		"Q8_0       8.5 bit/权重 30.1dB ×3.76",
		"Q4_K       4.5 bit/权重 24.8dB ×5.33",
	}
	if len(lines) != len(want) {
		t.Fatalf("每档一行，得到 %d 行: %q", len(lines), lines)
	}
	for i := range want {
		if lines[i] != want[i] {
			t.Errorf("第 %d 行 = %q，want %q（档名左对齐、位宽右对齐）",
				i, lines[i], want[i])
		}
		if strings.Contains(lines[i], "模拟[") || strings.Contains(lines[i], "≈") {
			t.Errorf("第 %d 行不该带包装或 ≈: %q", i, lines[i])
		}
	}
	// 采样时**每一行**都要带 ≈，且与全量版只差这一个前缀。
	//
	// 少了它就是把采样值当精确值显示，而采样是常态：默认上限 1e7 个元素，
	// 一个 4096×4096 的 f16 矩阵就有 1677 万。单行版 QuantSim 一直打着
	// "≈模拟[...]"，多行版漏掉过——两个入口对同一批数字给出不同结论。
	allSampled := make([]model.QuantSim, len(sims))
	copy(allSampled, sims)
	for i := range allSampled {
		allSampled[i].Sampled = true
	}
	sampledLines := QuantSimLines(allSampled)
	if len(sampledLines) != len(want) {
		t.Fatalf("采样时行数变了: %q", sampledLines)
	}
	for i, ln := range sampledLines {
		if want := "≈" + lines[i]; ln != want {
			t.Errorf("采样时第 %d 行 = %q，want %q —— 采样值不能打得跟精确值一样",
				i, ln, want)
		}
	}

	// 口径是 **sims[0]**（与单行版 QuantSim 一致），不是逐档判断：
	// 逐档判断会做出参差不齐的列，也与单行版口径不符。
	// 真实数据里三档出自同一次分析，Sampled 必然一致，这里是把口径钉死。
	firstOnly := make([]model.QuantSim, len(sims))
	copy(firstOnly, sims)
	firstOnly[0].Sampled = true
	for i, ln := range QuantSimLines(firstOnly) {
		// **整行精确**，不是 HasPrefix：只判前缀的话，"带上了 ≈
		// 但同一行的数字被改坏"照样绿（整串精确断言的理由见 TestStats_整串精确）
		if want := "≈" + lines[i]; ln != want {
			t.Errorf("sims[0] 采样时第 %d 行 = %q，want %q", i, ln, want)
		}
	}
	restOnly := make([]model.QuantSim, len(sims))
	copy(restOnly, sims)
	restOnly[1].Sampled = true
	restOnly[2].Sampled = true
	for i, ln := range QuantSimLines(restOnly) {
		if ln != lines[i] {
			t.Errorf("口径是 sims[0]（与单行版一致），第 %d 行 = %q，want %q",
				i, ln, lines[i])
		}
	}

	// 空输入返回 nil：调用方按 len()==0 判断"没有模拟"，不该拿到一个空串元素
	if got := QuantSimLines(nil); got != nil {
		t.Errorf("空输入应返回 nil，实际 %q", got)
	}
}

// QuantExistingLines 拆成能塞进 80 列的行；拼回来必须与单行版一字不差
// （唯一的例外是抽样中位数，两类输出都带标记，见下）。
func TestQuantExistingLines(t *testing.T) {
	// SubBlocks 刻意取到 124400（>1000）：真实量化张量的子块数是 10^5~10^7
	// 量级，会走 humanize.Count 的 "124.400 K" 缩写路径。原先写 800
	// 低于 1000 的阈值，把 Count(...) 换成裸整数输出一模一样、测试全绿
	// （实测漏网）—— 这条路径此前全仓没有覆盖。
	q := &model.QuantInfo{
		Scheme: "Q4_K", BitsPerWeight: 4.5, Blocks: 100, SubBlocks: 124400,
		BlockElems: 32, ScaleMin: 0.001, ScaleMax: 0.5, ScaleMedian: 0.02,
		ZeroScaleBlocks: 3, FlattestIndex: 7, FlattestRatio: 0.02,
	}
	lines := QuantExistingLines(q)
	want := []string{
		"Q4_K 4.5 bit/权重 124.400 K 子块",
		"scale[0.0010, 0.5000] 中位 0.0200",
		"⚠压平 3 子块 最扁 #7（比值 0.0200）",
	}
	if len(lines) != len(want) {
		t.Fatalf("得到 %d 行 %q，want %d 行", len(lines), lines, len(want))
	}
	for i := range want {
		if lines[i] != want[i] {
			t.Errorf("第 %d 行 = %q，want %q", i, lines[i], want[i])
		}
	}
	// 这条**看着**自我指涉，但**不能删**：它和上面的逐行 want 合起来才把
	// CLI 的单行输出钉成常量。删掉它，"join 分隔符从 " " 改成 "  "" 那条
	// 变异就漏网 —— 上面的 want 只钉每一行，分隔符只活在这一句里。
	if got, wantJoined := strings.Join(lines, " "), QuantExisting(q); got != wantJoined {
		t.Errorf("拆行拼回来与单行版不一致:\n got %q\nwant %q", got, wantJoined)
	}

	// 中位数可能来自抽样（子块数超预算时，见 model.QuantInfo 的说明），
	// 抽样值必须显示出来 —— 与 Stats 的 ≈ 同一个约定。
	// 这里要求：true 与 false 的输出必须不同，且**差别只在中位数那一处**。
	sampled := *q
	sampled.ScaleMedianSampled = true
	gotSampled := QuantExistingLines(&sampled)
	if len(gotSampled) != len(want) {
		t.Fatalf("只该中位数变，行数却变了: %q", gotSampled)
	}
	for i := range want {
		if i != 1 && gotSampled[i] != lines[i] {
			t.Errorf("只有第 2 行该变，第 %d 行也变了:\n 全量 %q\n 抽样 %q",
				i+1, lines[i], gotSampled[i])
		}
	}
	if gotSampled[1] == lines[1] {
		t.Error("抽样中位数必须显示出来 —— 否则用户会把它当全量精确值")
	}
	if wantMedian := "scale[0.0010, 0.5000] 中位 ≈0.0200（抽样）"; gotSampled[1] != wantMedian {
		t.Errorf("抽样中位数 = %q，want %q", gotSampled[1], wantMedian)
	}
	// 全量时**一个标记都不许有**，否则等于狼来了
	if strings.Contains(lines[1], "≈") || strings.Contains(lines[1], "抽样") {
		t.Errorf("全量中位数不该带抽样标记: %q", lines[1])
	}

	// 两个警告都没有时，第三行**不返回**（空行会让 TUI 白占一行）
	quiet := &model.QuantInfo{Scheme: "Q8_0", BitsPerWeight: 8.5, SubBlocks: 2,
		ScaleMin: 0.0625, ScaleMax: 0.5, ScaleMedian: 0.2, FlattestRatio: 1}
	if got := QuantExistingLines(quiet); len(got) != 2 {
		t.Errorf("没有警告时不该有第三行，实际 %q", got)
	}

	// 只有一个警告时，第三行只放那一个 —— 存在条件不能写反
	onlyFlat := &model.QuantInfo{Scheme: "Q4_K", BitsPerWeight: 4.5, SubBlocks: 8,
		ScaleMin: 0.1, ScaleMax: 0.5, ScaleMedian: 0.2,
		ZeroScaleBlocks: 5, FlattestRatio: 1}
	if got := QuantExistingLines(onlyFlat); len(got) != 3 ||
		got[2] != "⚠压平 5 子块" {
		t.Errorf("只该有压平那一行，实际 %q", got)
	}
	onlyThin := &model.QuantInfo{Scheme: "Q4_K", BitsPerWeight: 4.5, SubBlocks: 8,
		ScaleMin: 0.1, ScaleMax: 0.5, ScaleMedian: 0.2,
		FlattestIndex: 5562, FlattestRatio: 0}
	if got := QuantExistingLines(onlyThin); len(got) != 3 ||
		got[2] != "最扁 #5562（比值 0）" {
		t.Errorf("只该有最扁那一行，实际 %q", got)
	}

	// nil 仍返回空 —— 调用方（TUI）按 len()==0 判断"没有诊断"
	if got := QuantExistingLines(nil); len(got) != 0 {
		t.Errorf("nil 应返回空，实际 %q", got)
	}
	if got := QuantExisting(nil); got != "" {
		t.Errorf("nil 的单行输出仍是空串，实际 %q", got)
	}
}

// **单行版（CLI 用）的整串精确断言**，与 `TestStats_整串精确` 同一个理由。
//
// 原来那条 `TestQuantExistingString` 全是 Contains：把"⚠压平 3 子块"里的
// 数字换掉、把三个字段的先后顺序调换、把 join 的分隔符从一个空格改成两个 ——
// 只要那些词还出现过，它就全绿。而行与行之间的**分隔与顺序**
// 正是这条输出唯一不能错的东西（`QuantExistingLines` 拆行时按同一个顺序 join）。
//
// want 是按 QuantExistingLines 的约定手推的：Float 在 [1e-3,1e5) 内
// 4 位小数（0.5 → "0.5000"），Count 用 SI 缩写（800 → "800"、2000 → "2.000 K"），
// 两个警告之间、以及每一行之间都是**一个空格**。
func TestQuantExisting_整串精确(t *testing.T) {
	full := &model.QuantInfo{
		Scheme: "Q4_K", BitsPerWeight: 4.5, Blocks: 100, SubBlocks: 800,
		BlockElems: 32, ScaleMin: 0.001, ScaleMax: 0.5, ScaleMedian: 0.02,
		ZeroScaleBlocks: 3, FlattestIndex: 7, FlattestRatio: 0.02,
	}
	// 两个警告都有：它们的先后、以及"最扁"那一项里 #N 与比值的对应
	want := "Q4_K 4.5 bit/权重 800 子块 " +
		"scale[0.0010, 0.5000] 中位 0.0200 " +
		"⚠压平 3 子块 最扁 #7（比值 0.0200）"
	if got := QuantExisting(full); got != want {
		t.Errorf("QuantExisting() =\n got %q\nwant %q", got, want)
	}

	// 抽样中位数：单行版**必须**带上标记（TUI 的分行版有同一份标记），
	// 否则 CLI 把样本值打成精确值
	sampled := *full
	sampled.ScaleMedianSampled = true
	wantSampled := "Q4_K 4.5 bit/权重 800 子块 " +
		"scale[0.0010, 0.5000] 中位 ≈0.0200（抽样） " +
		"⚠压平 3 子块 最扁 #7（比值 0.0200）"
	if got := QuantExisting(&sampled); got != wantSampled {
		t.Errorf("抽样 QuantExisting() =\n got %q\nwant %q", got, wantSampled)
	}

	// 一个警告都没有：第三条整个不返回，单行版也就**不该多出分隔空格**
	//（多一个空格在终端里看不出来，但它是 CLI 输出的逐字节契约）
	quiet := &model.QuantInfo{Scheme: "Q8_0", BitsPerWeight: 8.5, SubBlocks: 2000,
		ScaleMin: 0.0625, ScaleMax: 0.5, ScaleMedian: 0.2, FlattestRatio: 1}
	wantQuiet := "Q8_0 8.5 bit/权重 2.000 K 子块 scale[0.0625, 0.5000] 中位 0.2000"
	if got := QuantExisting(quiet); got != wantQuiet {
		t.Errorf("QuantExisting() =\n got %q\nwant %q", got, wantQuiet)
	}
}

// **张量行末尾那一段的整串精确断言**（CLI 的每一行都以它结尾）。
//
// 这条钉的是**组装**：模拟优先于诊断、两者只出一个、都没有时不留空档。
// 原先全是 Contains，`TensorQuant` 把两个拼在一起（"模拟[...] Q4_K ..."）
// 或者多补一个前导空格，都拦不住。
func TestTensorQuant_整串精确(t *testing.T) {
	sims := []model.QuantSim{
		{Target: "Q8_0", BitsPerWeight: 8.5, SNRDB: 45.2, Compression: 3.76}}
	quant := &model.QuantInfo{Scheme: "Q4_K", BitsPerWeight: 4.5, SubBlocks: 8,
		ScaleMin: 0.1, ScaleMax: 0.5, ScaleMedian: 0.2, FlattestRatio: 1}

	wantSim := "模拟[Q8_0 8.5 bit/权重 45.2dB ×3.76]"
	if got := TensorQuant(&model.Tensor{QuantSims: sims}); got != wantSim {
		t.Errorf("浮点张量 =\n got %q\nwant %q（应当就是 QuantSim 那一行）", got, wantSim)
	}

	wantQuant := "Q4_K 4.5 bit/权重 8 子块 scale[0.1000, 0.5000] 中位 0.2000"
	if got := TensorQuant(&model.Tensor{Quant: quant}); got != wantQuant {
		t.Errorf("量化张量 =\n got %q\nwant %q（应当就是 QuantExisting 那一行）",
			got, wantQuant)
	}

	// 都有（不该发生）：**只出模拟那一份**，不是两份都出
	if got := TensorQuant(&model.Tensor{QuantSims: sims, Quant: quant}); got != wantSim {
		t.Errorf("两者都有时 =\n got %q\nwant %q（不能同时给两个互相矛盾的数字）",
			got, wantSim)
	}

	// 都没有：空串，调用方不该多打一个空格
	if got := TensorQuant(&model.Tensor{}); got != "" {
		t.Errorf("两者皆无时应是空串，实际 %q", got)
	}
}
