package main

import (
	"strings"
	"testing"

	"context"
	"os"
	"path/filepath"

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

// **没有模型时，未完成的下载也必须被说出来。**
//
// ollama 是下完之后才写 manifest 的（实测：manifest 的时间戳比 blob 晚几秒），
// 所以"第一次 pull 下到一半"这个最常见的情形里 Items 正好是空的。
// 这里原先在 len(Items)==0 时直接 return，把未完成下载那一段整个吞掉 ——
// 实测：10 MiB 占着盘，输出里一个字都没有，而那正是这条提示要保护的场景。
func TestRunScan_没有模型时也要提未完成的下载(t *testing.T) {
	home := t.TempDir()
	blobs := filepath.Join(home, ".ollama", "models", "blobs")
	if err := os.MkdirAll(blobs, 0o755); err != nil {
		t.Fatal(err)
	}
	// ollama 下载中的形状：<64位十六进制>-partial 与 -partial-<序号>
	hex64 := strings.Repeat("ab", 32)
	for name, size := range map[string]int{
		"sha256-" + hex64 + "-partial":        4096,
		"sha256-" + hex64 + "-partial-000003": 1024,
	} {
		if err := os.WriteFile(filepath.Join(blobs, name), make([]byte, size), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("HOME", home)

	out := captureStdout(t, func() {
		if err := runScan(context.Background(), false); err != nil {
			t.Fatalf("runScan 失败: %v", err)
		}
	})

	if !strings.Contains(out, "未完成的下载") {
		t.Errorf("有未完成的下载却没提 —— 用户找不到那 5 KiB 去哪了:\n%s", out)
	}
	// 也不能把"没有模型"这句弄丢
	if !strings.Contains(out, "没有发现模型文件") {
		t.Errorf("没有模型时该说「没有发现模型文件」:\n%s", out)
	}
}

// captureStdout 把 fn 执行期间的 stdout 抓回来。
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		var sb strings.Builder
		buf := make([]byte, 4096)
		for {
			n, err := r.Read(buf)
			sb.Write(buf[:n])
			if err != nil {
				break
			}
		}
		done <- sb.String()
	}()

	fn()

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	os.Stdout = old
	return <-done
}
