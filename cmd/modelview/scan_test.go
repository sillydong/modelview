package main

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/sillydong/modelview/internal/discover"
	"github.com/sillydong/modelview/internal/model"
)

// 模型库每行的渲染：有格式信息与没有时都要正确。
func TestScanLine(t *testing.T) {
	// 还没读到头部：只显示名字/来源/路径/大小
	it := discover.Item{Source: discover.SourceOllama, Name: "qwen2.5:3b",
		Path: "/x/blobs/sha256-aa", Size: 1929903008}
	got := scanLine(it)
	for _, want := range []string{"qwen2.5:3b", "1.80 GiB", "ollama"} {
		if !strings.Contains(got, want) {
			t.Errorf("缺少 %q: %q", want, got)
		}
	}
	if strings.Contains(got, "参数") {
		t.Errorf("还没有参数量却显示了参数段: %q", got)
	}

	// 读到头部之后：格式、架构、参数量都要出现
	it.Format = model.FormatGGUF
	it.Arch = "qwen2"
	it.Params = 3085938688
	got = scanLine(it)
	// humanCount 用 SI 前缀（G 而不是 GiB），且**刻意不用 "B"** ——
	// 同一屏里 B 已经是字节的意思，3.086 B 会被读成"3 个字节"
	for _, want := range []string{"GGUF", "qwen2", "3.086 G 个参数"} {
		if !strings.Contains(got, want) {
			t.Errorf("缺少 %q: %q", want, got)
		}
	}

	// 读失败：要显示失败原因，而不是装作没这回事
	it = discover.Item{Name: "bad.gguf", Path: "/x/bad.gguf", Size: 12,
		Err: "不是已识别的模型格式"}
	got = scanLine(it)
	if !strings.Contains(got, "不是已识别的模型格式") {
		t.Errorf("读失败时没显示原因: %q", got)
	}
	// 失败时不该显示空的格式/参数段
	if strings.Contains(got, "个参数") {
		t.Errorf("读失败的条目不该显示参数量: %q", got)
	}

	// 第一列显示的是 Name，**不是从 Path 推出来的文件名**。
	//
	// 这条钉住的是一个很容易"顺手修"的错误：ollama 的 Path 是
	// blobs/sha256-5ee4f07c…，从它推出的名字对用户毫无意义，
	// 而 Name 才是与 `ollama list` 对得上的那个（qwen2.5:3b）。
	//
	// **断言的是前缀那一段，不能只说"行里含 qwen2.5:3b"** ——
	// 路径列里也含它，那样把第一列改成路径兜底测试照样绿
	//（变异验证时确认过：变异后测试不红）。
	it = discover.Item{Source: discover.SourceOllama, Name: "qwen2.5:3b",
		Path: "/x/blobs/sha256-other", Size: 1024}
	got = scanLine(it)
	if !strings.HasPrefix(got, "  qwen2.5:3b") {
		t.Errorf("第一列该显示 Name，实际: %q", got)
	}
	if strings.HasPrefix(got, "  sha256-other") {
		t.Errorf("第一列在用路径兜底，而不是 Name: %q", got)
	}
}

// 孤儿 blob 的行：要能看出是哪个文件、多大。
func TestOrphanLine(t *testing.T) {
	o := discover.Item{Path: "/x/blobs/sha256-abcdef0123456789", Size: 487}
	got := orphanLine(o)
	if !strings.Contains(got, "487 B") {
		t.Errorf("没显示大小: %q", got)
	}
	if !strings.Contains(got, "sha256-abcdef0123456789") {
		t.Errorf("没显示文件名: %q", got)
	}
}

// 渲染函数不能产生终端控制码。
//
// 早先 scan 用 `\x1b[1A` 上移一行再重画想"就地刷新"，而数据行有一百多
// 字符、在 80 列终端上占两三屏行 —— 上移一行落在上一行中间，
// 刚打印的那行被擦掉，人看到的与脚本看到的不是一回事。
// 现在改成逐行追加；这条断言"输出里不许有转义序列"守住它。
func TestScanLine_不含控制码(t *testing.T) {
	items := []discover.Item{
		{Source: discover.SourceOllama, Name: "m:1b", Path: "/x/a.gguf", Size: 100},
		{Source: discover.SourceOllama, Name: "m:2b", Path: "/x/b.gguf", Size: 200,
			Format: model.FormatGGUF, Arch: "qwen2", Params: 12345},
		{Name: "bad.gguf", Path: "/x/bad.gguf", Err: "读不了"},
		{Path: "/x/c.gguf", Size: 300},
	}
	lines := []string{orphanLine(discover.Item{Path: "/x/sha256-abc", Size: 5})}
	for _, it := range items {
		lines = append(lines, scanLine(it))
	}
	for _, l := range lines {
		if strings.ContainsAny(l, "\x1b\r") {
			t.Errorf("输出里有终端控制字符: %q", l)
		}
	}
}

// truncate 只影响列宽，不能把信息截没了 —— 边界要测到。
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
		got := truncate(tt.in, tt.n)
		// **无论怎么截，字节数都不能超过上限** —— 这才是截断的定义，
		// 与输入是不是 ASCII 无关
		if len(got) > tt.n {
			t.Errorf("truncate(%q, %d) = %q，字节数 %d 超过上限",
				tt.in, tt.n, got, len(got))
		}
		// **截断后必须仍是合法 UTF-8** —— 按字节切最容易在这里出错
		if !utf8.ValidString(got) {
			t.Errorf("truncate(%q, %d) = %q，不是合法 UTF-8（切出了半个字符）",
				tt.in, tt.n, got)
		}
		if tt.want != "" && got != tt.want {
			t.Errorf("truncate(%q, %d) = %q, want %q", tt.in, tt.n, got, tt.want)
		}
	}
}
