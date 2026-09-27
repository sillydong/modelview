package tui

import (
	"context"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/discover"
	"github.com/sillydong/modelview/internal/model"
)

// 测试注入假的数据源 —— 视图**不自己去扫磁盘**。
// 这既让测试与机器上有什么模型无关，也逼着 I/O 留在 tea.Cmd 里。
func fakeLibrary() (Library, *int) {
	scans := 0
	lib := NewLibrary()
	lib.scan = func(context.Context, discover.Options) discover.Result {
		scans++
		return discover.Result{
			// 故意给一个**不是按名字排好**的顺序
			Items: []discover.Item{
				{Source: discover.SourceOllama, Name: "b:2b", Path: "/b", Size: 2 << 20},
				{Source: discover.SourceOllama, Name: "a:1b", Path: "/a", Size: 1 << 20},
			},
			Orphans:    []discover.Item{{Name: "sha256-orphan", Path: "/blobs/sha256-orphan", Size: 4096}},
			InProgress: []discover.Item{{Name: "sha256-x-partial", Path: "/blobs/sha256-x-partial", Size: 1 << 30}},
		}
	}
	lib.fill = func(it *discover.Item) *discover.Item {
		it.Format = model.FormatGGUF
		it.Arch = "qwen2"
		it.Params = 1_000_000_000
		return it
	}
	return lib, &scans
}

// 首屏必须**立刻**出来（只 stat），格式与参数量随后补。
func TestLibrary_先出首屏再补详情(t *testing.T) {
	lib, _ := fakeLibrary()

	cmd := lib.Init()
	if cmd == nil {
		t.Fatal("Init 没有返回命令 —— 首屏永远不会加载")
	}

	lib2, fillCmd := lib.Update(cmd())
	lib = lib2.(Library)
	if len(lib.items) != 2 {
		t.Fatalf("Scan 之后有 %d 项, want 2", len(lib.items))
	}
	if out := lib.View(80, 20); !strings.Contains(out, "a:1b") {
		t.Errorf("首屏没有列出模型:\n%s", out)
	}
	// Scan 阶段还没有格式信息，界面要提示"正在读取"
	if out := lib.View(80, 20); !strings.Contains(out, "正在读取") {
		t.Errorf("还没 Fill 完就该提示正在读取:\n%s", out)
	}
	if fillCmd == nil {
		t.Fatal("Scan 完成后没有发出 Fill 命令 —— 格式永远不会被补上")
	}

	// tea.Batch 返回的是**一个** Cmd，它的消息是 BatchMsg（里面才是那批命令）
	batch, ok := fillCmd().(tea.BatchMsg)
	if !ok {
		t.Fatalf("Fill 命令返回的不是 BatchMsg: %T", fillCmd())
	}
	if len(batch) != 2 {
		t.Fatalf("发出了 %d 条 Fill 命令, want 2（每个条目一条）", len(batch))
	}

	// 跑完 Fill，格式/架构/参数量要出现
	for _, c := range batch {
		lib2, _ := lib.Update(c())
		lib = lib2.(Library)
	}
	out := lib.View(100, 20)
	for _, want := range []string{"GGUF", "qwen2", "1.000 G 个参数"} {
		if !strings.Contains(out, want) {
			t.Errorf("Fill 之后缺少 %q:\n%s", want, out)
		}
	}
}

// 结果按**名字**排序，不是按路径。
//
// discover.Scan 内部按路径排，而 ollama 的路径是 blobs/sha256-xxx ——
// 照那个顺序显示，用户看到的是按哈希排的列表，毫无规律。
func TestLibrary_按名字排序(t *testing.T) {
	lib, _ := fakeLibrary()
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	out := lib.View(80, 20)
	if strings.Index(out, "a:1b") > strings.Index(out, "b:2b") {
		t.Errorf("模型没有按名字排序:\n%s", out)
	}
}

// ↑↓ 与 jk 都要能移动。
func TestLibrary_移动(t *testing.T) {
	lib, _ := fakeLibrary()
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	if lib.cursor != 0 {
		t.Fatalf("初始光标在 %d, want 0", lib.cursor)
	}
	lib2, _ = lib.Update(key("down"))
	if got := lib2.(Library).cursor; got != 1 {
		t.Errorf("按 ↓ 之后光标在 %d, want 1", got)
	}
	lib2, _ = lib2.(Library).Update(key("up"))
	if got := lib2.(Library).cursor; got != 0 {
		t.Errorf("按 ↑ 之后光标在 %d, want 0", got)
	}
	lib2, _ = lib.Update(key("j"))
	if got := lib2.(Library).cursor; got != 1 {
		t.Errorf("按 j 之后光标在 %d, want 1", got)
	}
}

// **边界不能越界，也不能回绕**：回绕会让"到底了"这件事消失，
// 用户按着 ↓ 一路转圈却不知道自己已经在末尾。
func TestLibrary_移动不越界不回绕(t *testing.T) {
	lib, _ := fakeLibrary()
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	for range 5 {
		lib2, _ = lib.Update(key("down"))
		lib = lib2.(Library)
	}
	if lib.cursor != 1 {
		t.Errorf("一直按 ↓ 之后光标在 %d, want 1（末尾）", lib.cursor)
	}
	for range 5 {
		lib2, _ = lib.Update(key("up"))
		lib = lib2.(Library)
	}
	if lib.cursor != 0 {
		t.Errorf("一直按 ↑ 之后光标在 %d, want 0（开头）", lib.cursor)
	}
}

// 空列表上按 ↑↓ 不能崩（cursor 会越界到 -1 之类）。
func TestLibrary_空列表按方向键不崩(t *testing.T) {
	lib := NewLibrary()
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{Items: []discover.Item{}}
	}
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	for _, k := range []string{"down", "up", "j", "k"} {
		lib2, _ = lib.Update(key(k))
		lib = lib2.(Library)
		if lib.cursor != 0 {
			t.Errorf("空列表上按 %s 之后光标是 %d，应当是 0", k, lib.cursor)
		}
	}
}

// **未完成的下载不能出现在"可回收"里** —— 那是用户自己的下载，
// 照着删会毁掉它。实测过：拉 gpt-oss:20b 到一半时报"可回收 12.85 GiB"。
func TestLibrary_未完成的下载不算可回收(t *testing.T) {
	lib, _ := fakeLibrary()
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	out := lib.View(100, 30)
	if !strings.Contains(out, "未完成") {
		t.Errorf("界面没提未完成的下载 —— 用户找不到那 1 GiB 去哪了:\n%s", out)
	}

	// **断言的是数字，不是词**：那句提示本身写着"不是可回收空间"，
	// 按词判断会误伤。真正要验的是"可回收总量里没有算进那 1 GiB"。
	orphanLine, ok := "", false
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, "孤儿 blob") {
			orphanLine, ok = line, true
		}
	}
	if !ok {
		t.Fatalf("孤儿 blob 那一段没渲染:\n%s", out)
	}
	if !strings.Contains(orphanLine, "4.00 KiB") {
		t.Errorf("孤儿那一行的可回收量不对（应当只有 4 KiB 那个孤儿）: %q", orphanLine)
	}
	if strings.Contains(orphanLine, "GiB") {
		t.Errorf("孤儿那一行出现了 GiB —— 未完成的下载被算进可回收了: %q", orphanLine)
	}
}

// r 重扫：必须重新调数据源，而不是复用旧结果。
func TestLibrary_重扫(t *testing.T) {
	lib, scans := fakeLibrary()
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)
	before := *scans

	_, cmd := lib.Update(key("r"))
	if cmd == nil {
		t.Fatal("按 r 没有返回重扫命令")
	}
	if *scans != before {
		t.Error("按 r 的时候就把扫描跑了 —— 应该返回命令、由框架去跑")
	}
	lib.Update(cmd())
	if *scans != before+1 {
		t.Errorf("重扫后扫描次数 %d, want %d", *scans, before+1)
	}
}

// 空结果是正常状态（本机没装模型），要给出可操作的提示而不是空白。
func TestLibrary_空结果有提示(t *testing.T) {
	lib := NewLibrary()
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{Items: []discover.Item{}}
	}
	lib2, _ := lib.Update(lib.Init()())
	out := lib2.(Library).View(80, 20)
	if !strings.Contains(out, "没有发现") {
		t.Errorf("没有模型时界面没给提示:\n%s", out)
	}
	// 提示里要列出扫过哪些目录 —— 否则用户不知道去哪儿放模型
	if !strings.Contains(out, "扫过") {
		t.Errorf("提示里没列扫描路径:\n%s", out)
	}
}

// 帮助栏只列**这一层真的支持**的键。
func TestLibrary_帮助栏只列支持的键(t *testing.T) {
	lib, _ := fakeLibrary()
	help := strings.Join(lib.Help(), " ")
	for _, want := range []string{keyUp, keyDown, keyRescan, keyQuit} {
		if !strings.Contains(help, want) {
			t.Errorf("帮助栏少了 %s: %q", want, help)
		}
	}
	// **还没做的键不许列**：列了等于骗用户按。
	// Enter（进单模型）在 Task 4 接上，?（速查表）在 ④b-2
	for _, unwanted := range []string{"Enter", "?"} {
		if strings.Contains(help, unwanted) {
			t.Errorf("还没实现 %s，帮助栏不该列: %q", unwanted, help)
		}
	}
}

// 模型名超长时第一列不能撑破列宽 —— 名字来自文件名，可以任意长。
func TestLibrary_长名字被截断(t *testing.T) {
	lib := NewLibrary()
	long := strings.Repeat("很长的模型名字", 10)
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{Items: []discover.Item{
			{Source: discover.SourceGeneric, Name: long, Path: "/x/" + long, Size: 1024},
		}}
	}
	lib2, _ := lib.Update(lib.Init()())
	out := lib2.(Library).View(80, 10)
	if strings.Contains(out, long) {
		t.Errorf("超长模型名没被截断，会把整行撑破:\n%s", out)
	}
}
