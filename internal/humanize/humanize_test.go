package humanize

import (
	"math"
	"testing"
	"unicode/utf8"
)

// 下面 7 个测试是从 cmd/modelview 搬过来的（原本测 humanBytes/humanCount/
// commaInt/dimsString/truncate），TestFloat / TestRatio 则是经 internal/render
// 二度搬来的（原本测 humanFloat/humanRatio）—— 它们收的是裸 float64、
// 不知道模型的存在，属于这个包而不是 render。**测试内容一字不改** ——
// 只改了函数名的首字母、包名与诊断文本里的函数名。它们的价值就在那些边界
// 和约定上：
//   - Count 那条断言"任何量级都不出现 B"
//   - Truncate 那几条中文用例是"切点必须回退到 rune 边界"的证据
//   - Float 那条"权重级别的数值不能打成 0.0000"
//
// 搬的时候如果把测试留在了原处，删函数就等于删掉这些守卫 ——
// 而它们守的正是 CLI 与 TUI 共用的显示格式。

func TestBytes(t *testing.T) {
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
		if got := Bytes(tt.in); got != tt.want {
			t.Errorf("Bytes(%d) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// Count 用 SI 前缀（1000 进制）且不使用 "B" ——
// 摘要里 "B" 已被 Bytes 占用表示字节。
func TestCount_不使用B后缀(t *testing.T) {
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
		if got := Count(tt.in); got != tt.want {
			t.Errorf("Count(%d) = %q, want %q", tt.in, got, tt.want)
		}
	}

	// 显式防回归：任何量级都不应出现 "B"
	for _, n := range []int64{1e9, 1e10, 1e11, 1e12, 1e13} {
		if got := Count(n); got[len(got)-1] == 'B' {
			t.Errorf("Count(%d) = %q 以 B 结尾，与 Bytes 的 B 冲突", n, got)
		}
	}
}

func TestComma(t *testing.T) {
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
		if got := Comma(tt.in); got != tt.want {
			t.Errorf("Comma(%d) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestDims(t *testing.T) {
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
		if got := Dims(tt.in); got != tt.want {
			t.Errorf("Dims(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// Truncate 只影响列宽，不能把信息截没了 —— 边界要测到。
func TestTruncate(t *testing.T) {
	tests := []struct {
		in   string
		n    int
		want string
	}{
		{"abc", 5, "abc"},        // 短于上限：原样
		{"abcde", 5, "abcde"},    // 正好等于上限：原样
		{"abcdef", 6, "abcdef"},  // 边界
		{"abcdefg", 6, "abc..."}, // 超出：截断加省略号
		{"abcdefghij", 6, "abc..."},
		{"ab", 2, "ab"},   // n 很小
		{"abc", 3, "abc"}, // n=3 时没有省略号的余地
		{"abcd", 3, "abc"},

		// **多字节字符**：中文模型名（如 "qwen2.5-3b-中文"）会让
		// 按 rune 计数的写法超长 —— 一个汉字 3 字节，按 rune 算不超上限
		// 但字节数会翻三倍，列就对齐不了。截断必须按**字节**。
		// 12 字节 → 留 3 给省略号 → 前 3 字节 = "中"；
		// 15 字节 → 留 3 → 前 6 字节 = "中文"
		{"中文名字", 6, "中..."},
		{"中文名字啊", 9, "中文..."},
		// **切点落在多字节字符中间**：必须回退到 rune 边界，
		// 否则会切出半个汉字（输出里出现替换字符）
		//
		// 注意 cut = n-3（给 "..." 留 3 字节），所以结果比直觉的短：
		{"abc中文", 5, "ab..."}, // cut=2，落在 ASCII 内 → 不回退
		{"ab中文", 4, "a..."},   // cut=1
		{"中中中", 7, "中..."},    // cut=4 落在第二个汉字中间 → 回退到 3
		// **切出非法 UTF-8 的真实用例**（审查时实测报出来的）：
		// 按字节硬切会得到 "中\xe6\x96..." 这种残缺序列，
		// 而这些 n 都不是 3 的倍数，正好踩在字符中间
		//
		// 期望值是**逐字节数出来的**（汉字 3 字节）：
		// cut = n-3，再往前回退到字符起点。手估很容易多算一两个字节，
		// 这几条我估错过三次 —— 好在下面那两条不变式（不超过 n、合法 UTF-8）
		// 与实现一致，所以红的是期望值而不是代码。
		{"中文模型名很长的名字abc", 8, "中..."},                    // cut=5→回退到 3
		{"中文模型名很长的名字abc", 5, "..."},                     // cut=2→回退到 0（放不下一个字）
		{"中文模型名很长的名字abc", 10, "中文..."},                  // cut=7→回退到 6
		{"中文模型名很长的名字abc", 12, "中文模..."},                 // cut=9，正好是"型"的起点
		{"qwen2.5-3b-中文微调.gguf", 18, "qwen2.5-3b-中..."}, // cut=15 = "文"的起点
	}
	for _, tt := range tests {
		got := Truncate(tt.in, tt.n)
		// **无论怎么截，字节数都不能超过上限** —— 这才是截断的定义，
		// 与输入是不是 ASCII 无关
		if len(got) > tt.n {
			t.Errorf("Truncate(%q, %d) = %q，字节数 %d 超过上限",
				tt.in, tt.n, got, len(got))
		}
		// **截断后必须仍是合法 UTF-8** —— 按字节切最容易在这里出错
		if !utf8.ValidString(got) {
			t.Errorf("Truncate(%q, %d) = %q，不是合法 UTF-8（切出了半个字符）",
				tt.in, tt.n, got)
		}
		if tt.want != "" && got != tt.want {
			t.Errorf("Truncate(%q, %d) = %q, want %q", tt.in, tt.n, got, tt.want)
		}
	}
}

// Float 在常用区间内用 4 位小数（区间外退回 4 位有效数字）：
// 权重的动态范围横跨好几个数量级，定点格式会让小值全变成 0.0000。
func TestFloat(t *testing.T) {
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
		if got := Float(tt.in); got != tt.want {
			t.Errorf("Float(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
	// 权重的典型量级必须打得出可读结果，不能全是 0.0000
	if got := Float(0.026777247); got == "0.0000" {
		t.Errorf("Float(0.026777247) = %q —— 权重级别的数值被打成了 0", got)
	}
	if got := Float(math.NaN()); got != "NaN" {
		t.Errorf("Float(NaN) = %q", got)
	}
}

// Percent 把 0..1 的比例打成百分数。
func TestRatio(t *testing.T) {
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
		if got := Percent(tt.in); got != tt.want {
			t.Errorf("Percent(%v) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
