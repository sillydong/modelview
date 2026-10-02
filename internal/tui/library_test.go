package tui

import (
	"context"
	"fmt"
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
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)
	help := strings.Join(lib.Help(), " ")
	for _, want := range []string{keyUp, keyDown, keyEnter, keyRescan, keyQuit} {
		if !strings.Contains(help, want) {
			t.Errorf("帮助栏少了 %s: %q", want, help)
		}
	}
	// **没有处理分支的键不许列**：列了等于骗用户按。
	// `?` 原先就是这个反例（根视图里没有那条全局分支）；现在它接上了，
	// 于是反过来 —— 这一屏按 `?` 真的开得出速查表，帮助栏就得列。
	if !strings.Contains(help, keyHelp+" 速查表") {
		t.Errorf("按 ? 能开速查表，帮助栏却没列 %s: %q", keyHelp, help)
	}
}

// **空库时帮助栏不能说"Enter 查看"** —— 一条模型都没有，按下去什么也不做。
//
// 空库是真会出现的（扫过的地方一个模型都没有），不是错误状态；
// 还没扫完时列表也是空的，同一个判据一起挡住。
func TestLibrary_空库帮助栏不列Enter(t *testing.T) {
	lib := NewLibrary()
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{}
	}
	// 还没扫完：界面显示的是"正在扫描…"，此时列 Enter 同样是骗用户按
	if help := strings.Join(lib.Help(), " "); strings.Contains(help, keyEnter) {
		t.Errorf("还没扫完就列了 %s: %q", keyEnter, help)
	}

	lib2, _ := lib.Update(lib.Init()())
	empty := lib2.(Library)
	if len(empty.items) != 0 {
		t.Fatalf("前提不成立：这个库有 %d 个模型", len(empty.items))
	}
	if help := strings.Join(empty.Help(), " "); strings.Contains(help, keyEnter) {
		t.Errorf("空库的帮助栏列了 %s —— 按下去没反应: %q", keyEnter, help)
	}
	if _, cmd := empty.Update(key("enter")); cmd != nil {
		t.Error("空库按 Enter 居然有动作 —— 帮助栏与 Update 读的不是同一个判据")
	}

	// 正对照：有模型时两个方向都要成立（列了、而且真的能开）
	full, _ := fakeLibrary()
	full2, _ := full.Update(full.Init()())
	full = full2.(Library)
	if help := strings.Join(full.Help(), " "); !strings.Contains(help, keyEnter) {
		t.Errorf("有模型时帮助栏少了 %s: %q", keyEnter, help)
	}
	if _, cmd := full.Update(key("enter")); cmd == nil {
		t.Error("有模型时按 Enter 没动作")
	}
}

// **重扫期间按 Enter 不该打开一个看不见的条目。**
//
// 屏幕上是"正在扫描模型目录…"，而 items 还是上一轮那份 ——
// 判据只写 `len(items) > 0` 的话，用户这一下会进到一个自己没看见的模型里
// （本仓为"打开看不见的那一条"踩过好几次，见 View 里那段窗口说明）。
func TestLibrary_重扫期间不列Enter也不打开(t *testing.T) {
	lib, _ := fakeLibrary()
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)
	if !lib.canOpen() {
		t.Fatal("前提不成立：扫完之后应当能开")
	}

	// 按 r 发起重扫：只把 loaded 打回假，列表字段还留着
	lib3, cmd := lib.Update(key("r"))
	lib = lib3.(Library)
	if cmd == nil {
		t.Fatal("按 r 没返回重扫命令")
	}
	if lib.loaded || len(lib.items) == 0 {
		t.Fatalf("前提不成立：loaded=%v, items=%d", lib.loaded, len(lib.items))
	}
	if help := strings.Join(lib.Help(), " "); strings.Contains(help, keyEnter) {
		t.Errorf("重扫期间帮助栏列了 %s，而屏幕上是「正在扫描」: %q", keyEnter, help)
	}
	if _, c := lib.Update(key("enter")); c != nil {
		t.Error("重扫期间按 Enter 打开了看不见的条目")
	}

	// 扫完之后同一个判据要放行
	lib4, _ := lib.Update(cmd())
	lib = lib4.(Library)
	if help := strings.Join(lib.Help(), " "); !strings.Contains(help, keyEnter) {
		t.Errorf("扫完之后帮助栏少了 %s: %q", keyEnter, help)
	}
	if _, c := lib.Update(key("enter")); c == nil {
		t.Error("扫完之后按 Enter 没动作")
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

// Enter 要能进单模型视图（走 pushMsg，由根 Model 压栈）。
func TestLibrary_Enter进入模型视图(t *testing.T) {
	lib, _ := fakeLibrary()
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	_, cmd := lib.Update(key("enter"))
	if cmd == nil {
		t.Fatal("按 Enter 没有返回命令")
	}
	msg, ok := cmd().(pushMsg)
	if !ok {
		t.Fatalf("按 Enter 返回的不是 pushMsg: %T", cmd())
	}
	if msg.v == nil {
		t.Error("pushMsg 里的视图是 nil")
	}
}

// **重扫之后，上一轮在途的 Fill 结果必须被丢弃。**
//
// Fill 要读文件头部，在途几百毫秒；这期间用户按 r 重扫，
// 旧结果回来时会按 index 覆盖新列表 —— 界面上出现一个这次根本没扫到的
// 模型，而且不报任何错。实测复现过（见下），两次审查也都独立报了这条。
func TestLibrary_重扫丢弃在途的Fill结果(t *testing.T) {
	round := 0
	lib := NewLibrary()
	lib.scan = func(context.Context, discover.Options) discover.Result {
		round++
		if round == 1 {
			return discover.Result{Items: []discover.Item{
				{Source: discover.SourceOllama, Name: "旧A", Path: "/a", Size: 1},
				{Source: discover.SourceOllama, Name: "旧B", Path: "/b", Size: 2},
			}}
		}
		return discover.Result{Items: []discover.Item{
			{Source: discover.SourceOllama, Name: "新C", Path: "/c", Size: 3},
		}}
	}
	lib.fill = func(it *discover.Item) *discover.Item {
		it.Format = model.FormatGGUF
		return it
	}

	// 第一轮：拿到 2 条在途的 Fill 命令，但**先不投递**
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)
	_, staleBatch := lib.Update(libraryLoadedMsg{res: discover.Result{
		Items: []discover.Item{
			{Source: discover.SourceOllama, Name: "旧A", Path: "/a", Size: 1},
			{Source: discover.SourceOllama, Name: "旧B", Path: "/b", Size: 2},
		},
	}})
	_ = staleBatch

	// 按 r 重扫，第二轮只返回 1 条
	_, rescanCmd := lib.Update(key("r"))
	lib2, _ = lib.Update(rescanCmd())
	lib = lib2.(Library)
	if len(lib.items) != 1 || lib.items[0].Name != "新C" {
		t.Fatalf("重扫后列表 = %v, want [新C]", itemNames(lib.items))
	}

	// 现在投递**第一轮**的 Fill 结果（用旧世代号）
	stale := itemFilledMsg{index: 0, gen: lib.gen - 1,
		item: discover.Item{Name: "旧A", Path: "/a", Size: 1, Format: model.FormatGGUF}}
	lib2, _ = lib.Update(stale)
	lib = lib2.(Library)

	if lib.items[0].Name != "新C" {
		t.Errorf("过期的 Fill 覆盖了新列表: %v —— 界面上会出现一个这次没扫到的模型",
			itemNames(lib.items))
	}
	if lib.filled > len(lib.items) {
		t.Errorf("filled=%d > len(items)=%d —— 进度会提前消失", lib.filled, len(lib.items))
	}
}

// 重扫要把光标与进度一起重置。
//
// cursor 归零不只是"回到顶部"：它还兼着防越界 ——
// 从多条的列表重扫到少条之后，旧光标会让 Enter 索引越界。
func TestLibrary_重扫后状态被重置(t *testing.T) {
	round := 0
	lib := NewLibrary()
	lib.scan = func(context.Context, discover.Options) discover.Result {
		round++
		n := 5
		if round > 1 {
			n = 2
		}
		items := make([]discover.Item, n)
		for i := range items {
			items[i] = discover.Item{Name: string(rune('a' + i)), Path: "/x", Size: int64(i)}
		}
		return discover.Result{Items: items}
	}
	lib.fill = func(it *discover.Item) *discover.Item { return it }

	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)
	// 把光标移到底部
	for range 4 {
		lib2, _ = lib.Update(key("down"))
		lib = lib2.(Library)
	}
	if lib.cursor != 4 {
		t.Fatalf("光标在 %d, want 4", lib.cursor)
	}

	_, cmd := lib.Update(key("r"))
	lib2, _ = lib.Update(cmd())
	lib = lib2.(Library)

	if lib.cursor != 0 {
		t.Errorf("重扫后光标在 %d —— 应当归零（否则 Enter 会索引越界）", lib.cursor)
	}
	if lib.filled != 0 {
		t.Errorf("重扫后 filled=%d —— 应当归零", lib.filled)
	}
	// 从 5 条缩到 2 条之后按 Enter 不能崩
	if _, c := lib.Update(key("enter")); c == nil {
		t.Error("重扫后在有效条目上按 Enter 应当返回命令")
	}
}

// 空库上按 Enter 不能崩 —— 它要索引 items[cursor]，而空列表里没有。
func TestLibrary_空库按Enter不崩(t *testing.T) {
	lib := NewLibrary()
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{Items: []discover.Item{}}
	}
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	_, cmd := lib.Update(key("enter"))
	if cmd != nil {
		if _, ok := cmd().(pushMsg); ok {
			t.Error("空列表上按 Enter 不该推入模型视图")
		}
	}
}

// 读失败的条目要显示原因，而且**原因不能被路径挤到屏幕外**。
//
// discover.Fill 写的 Err 是 "<完整路径>: <原因>"，
// 而路径通常比一行还长 —— 实测 80 列下整行只剩 "⚠/tmp/xxx/"，
// 用户唯一得到的信息是"这个模型坏了"，坏在哪不知道。
func TestLibrary_读失败显示原因而不是路径(t *testing.T) {
	long := "/Users/somebody/very/long/path/to/models/" + strings.Repeat("d", 60) + "/broken.gguf"
	lib := NewLibrary()
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{Items: []discover.Item{
			{Source: discover.SourceGeneric, Name: "broken.gguf", Path: long,
				Err: long + ": 无法识别的模型格式"},
		}}
	}
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	out := lib.View(80, 10)
	if !strings.Contains(out, "无法识别的模型格式") {
		t.Errorf("读失败时没把原因显示出来（被路径挤掉了？）:\n%s", out)
	}
	for _, line := range strings.Split(out, "\n") {
		if w := displayWidth(line); w > 80 {
			t.Errorf("行宽 %d 超过 80: %q", w, line)
		}
	}
}

// 各目录的警告（如"manifest 读不了，孤儿检测已跳过"）必须显示完整 ——
// 那是全工具唯一一条会引导破坏性操作的提示。
func TestLibrary_目录警告要显示关键半句(t *testing.T) {
	lib := NewLibrary()
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{
			Items: []discover.Item{{Name: "m", Path: "/m", Size: 1}},
			Errs: []string{"/blobs 下有 1 个 manifest 读不了，孤儿 blob 检测已跳过：" +
				"报成「可回收」可能让你删掉真实模型"},
		}
	}
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	out := lib.View(120, 20)
	if !strings.Contains(out, "可能让你删掉真实模型") {
		t.Errorf("破坏性操作警告的关键半句被截掉了:\n%s", out)
	}
}

// emptyLibraryWithWarnings 造一个"空库 + 有孤儿 + 有未完成下载 + 有目录警告"的视图。
//
// 这不是硬凑的边角场景：**一个模型都没扫到、但磁盘上并不干净**恰恰是
// 孤儿 blob 与未完成下载最容易出现的组合（模型目录配错、权限不对时，
// items 扫不到而 blobs 照样列得出来），而这两条提示是全工具仅有的
// 会引导破坏性操作的话。
func emptyLibraryWithWarnings() Library {
	lib := NewLibrary()
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{
			InProgress: []discover.Item{{Name: "sha256-x-partial", Path: "/blobs/sha256-x-partial", Size: 1 << 30}},
			Orphans:    []discover.Item{{Name: "sha256-orphan", Path: "/blobs/sha256-orphan", Size: 4096}},
			Errs: []string{"/blobs 下有 1 个 manifest 读不了，孤儿 blob 检测已跳过：" +
				"报成「可回收」可能让你删掉真实模型"},
		}
	}
	lib2, _ := lib.Update(lib.Init()())
	return lib2.(Library)
}

// **空库时安全提示同样不能消失** —— "空"是另一个方向的同一个问题。
//
// `notices` 的注释讲的是"提示不能因为模型**多**而消失"，
// 而这里是"因为模型**零**而消失"：更隐蔽，因为空库看着像个干净状态，
// 用户不会怀疑自己漏看了什么。原来的 `View` 在 `len(items)==0` 时
// 直接 `return l.emptyView()`，把那三条提示整个跳过了。
//
// 断言的是**原始输出**（不经根视图，与既有的 TestLibrary_* 一致）：
// 三条提示都在屏幕上、空库说明也在，且行数 ≤ height ——
// 拼起来之后超出的行会被 padTo 砍掉，砍掉的正是路径清单（用户看这一屏
// 就是为了知道去哪儿放模型）。
func TestLibrary_空库时安全提示仍在(t *testing.T) {
	lib := emptyLibraryWithWarnings()
	if len(lib.items) != 0 {
		t.Fatalf("前提不成立：这个库有 %d 个模型", len(lib.items))
	}

	for _, h := range []int{6, 8, 10, 20, 40} {
		out := lib.View(80, h)
		if strings.HasSuffix(out, "\n") {
			t.Errorf("高度 %d：原始输出以换行结尾（padTo 会多算一行）:\n%q", h, out)
		}
		if n := len(strings.Split(out, "\n")); n > h {
			t.Errorf("高度 %d：原始输出 %d 行 —— 超出的会被 padTo 砍掉:\n%s", h, n, out)
		}
		// 两条破坏性提示 + 孤儿那一行的名字（提示里列的是清单，每个孤儿一行）
		for _, want := range []string{"未完成的下载", "孤儿 blob", "sha256-orphan",
			"没有发现模型文件。"} {
			if !strings.Contains(out, want) {
				t.Errorf("高度 %d：%q 不在屏上 —— 提示因为模型**零**而消失了:\n%s",
					h, want, out)
			}
		}
		// 目录警告会按宽度折行，按词匹配会跨行漏掉 —— 先拼回一行再找
		if flat := strings.Join(strings.Fields(out), ""); !strings.Contains(flat, "可能让你删掉真实模型") {
			t.Errorf("高度 %d：目录警告的关键半句不见了:\n%s", h, out)
		}
	}
}

func itemNames(items []discover.Item) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = it.Name
	}
	return out
}

// **选中项必须始终可见。**
//
// 不滚的话，用户按 ↓ 越过一屏之后看不到自己在选什么，
// 此时按 Enter 打开的是一个看不见的模型。
func TestLibrary_选中项始终可见(t *testing.T) {
	lib := NewLibrary()
	items := make([]discover.Item, 50)
	for i := range items {
		items[i] = discover.Item{
			Source: discover.SourceGeneric,
			Name:   fmt.Sprintf("model-%02d.gguf", i),
			Path:   fmt.Sprintf("/x/%02d", i), Size: int64(i),
		}
	}
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{Items: items}
	}
	lib.fill = func(it *discover.Item) *discover.Item { return it }
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	// 一路按到第 40 个，每一步都要求选中的那一行在屏幕上
	for range 40 {
		lib2, _ = lib.Update(key("down"))
		lib = lib2.(Library)
		out := lib.View(80, 24)
		want := fmt.Sprintf("model-%02d.gguf", lib.cursor)
		if !strings.Contains(out, want) {
			t.Fatalf("光标在第 %d 项，但屏幕上没有 %s（没跟着滚）:\n%s",
				lib.cursor, want, out)
		}
	}
}

// **"未完成的下载不是可回收空间"这条提示，模型再多也不能消失。**
//
// 它是全工具唯一一条会引导破坏性操作的提示 —— 照着删会毁掉用户
// 正在下的模型。原先把提示放在列表之后，实测 30 个模型、80×24 的
// 终端里它完全不可见。
func TestLibrary_模型多时安全提示仍可见(t *testing.T) {
	lib := NewLibrary()
	items := make([]discover.Item, 40)
	for i := range items {
		items[i] = discover.Item{Name: fmt.Sprintf("m%02d", i), Path: "/x", Size: 1}
	}
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{
			Items:      items,
			InProgress: []discover.Item{{Name: "sha256-x-partial", Path: "/p", Size: 1 << 30}},
			Errs: []string{"/blobs 下有 1 个 manifest 读不了，孤儿 blob 检测已跳过：" +
				"报成「可回收」可能让你删掉真实模型"},
		}
	}
	lib.fill = func(it *discover.Item) *discover.Item { return it }
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)

	out := lib.View(80, 24)
	if !strings.Contains(out, "未完成的下载") {
		t.Errorf("40 个模型时，未完成下载的提示不见了:\n%s", out)
	}
	// **断言前要去掉换行与空格**：警告是折行的，跨行的子串当然找不到 ——
	// 第一版断言就是栽在这里，看起来像"被截断"，其实是匹配方式不对
	flat := strings.Join(strings.Fields(out), "")
	if !strings.Contains(flat, "可能让你删掉真实模型") {
		t.Errorf("破坏性警告的关键半句被截掉了:\n%s", out)
	}
}

// **列表满时最后一行不能被 padTo 挤掉 —— 这条必须经过根视图。**
//
// 直接调 lib.View() 看不出问题（实测：那样调用一切正常，选中项可见、
// 提示的数字也自洽）；症状由两处**独立的**成因叠加产生：
//   - 范围提示追加在 listCap 之外 —— 原始输出比高度多一行
//   - 列表末尾的 "\n" —— strings.Split 多算一个空元素
//
// 两者都只在根视图的 padTo 那一层才显形：它砍掉最后两行，
// 而光标停在末尾时，被砍掉的正是用户选着的那一条
// （实测 30 个模型、120×10/20/30 三档：屏幕上最后一个名字一直是倒数第二个）。
//
// 既有的 TestLibrary_* 全部直接调 lib.View —— 这个结构缺口与 Task 2 修的
// "单测绿而真终端坏"同一类：**验证路径绕过了出事的那一层**。
func TestLibrary_经根视图不丢最后一行(t *testing.T) {
	lib := NewLibrary()
	items := make([]discover.Item, 30)
	for i := range items {
		items[i] = discover.Item{
			Source: discover.SourceGeneric,
			Name:   fmt.Sprintf("model-%02d.gguf", i),
			Path:   fmt.Sprintf("/x/%02d", i), Size: int64(i),
		}
	}
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{Items: items}
	}
	lib.fill = func(it *discover.Item) *discover.Item { return it }
	lib2, _ := lib.Update(lib.Init()())
	lib = lib2.(Library)
	// 走真实的 Fill 消息把"正在读取…"那一行消掉：留着它，
	// 列表高度又少一行，测的就不是"列表满"这一条了
	for i := range items {
		lib2, _ = lib.Update(itemFilledMsg{index: i, item: items[i], gen: lib.gen})
		lib = lib2.(Library)
	}
	// 光标走到最后一项 —— 被 padTo 砍掉的那一行正是它
	for range len(items) - 1 {
		lib2, _ = lib.Update(key("down"))
		lib = lib2.(Library)
	}
	if lib.cursor != len(items)-1 {
		t.Fatalf("光标在第 %d 项，没走到最后一项", lib.cursor)
	}

	// 高度至少覆盖三档：这条与高度无关（列表满就丢）
	for _, h := range []int{10, 20, 30} {
		m, _ := New(lib).Update(tea.WindowSizeMsg{Width: 120, Height: h})
		out := m.(Model).View()

		want := "▸ " + lib.items[lib.cursor].Name
		if !strings.Contains(out, want) {
			t.Errorf("高度 %d：光标在最后一项，屏幕上却没有 %q —— 被 padTo 砍掉了:\n%s",
				h, want, out)
		}
		// 范围提示那一行也要活下来（它是第二个被砍的候选）
		if len(lib.items) > h-2 && !strings.Contains(out, "显示第") {
			t.Errorf("高度 %d：列表放不下，范围提示却被 padTo 砍掉了:\n%s", h, out)
		}
	}
}

// **矮终端下范围提示与进度行都要活下来 —— 容量只够一行时不能硬塞。**
//
// Library 是唯一有**两个**可选尾行的视图（范围提示 + 在途进度），
// 而原先只按一个扣减、判断里又少了"这一行放得下吗"那半句：
// 终端高 7、30 个模型、孤儿提示 + 在途进度下，范围提示与进度行会一起
// 被 padTo 顶掉，屏幕上换成"…还有 2 行没显示" —— 用户既不知道
// 自己看到的是第几个，也不知道后面还有没有（实测）。
//
// 两档分别守两件事：①够放时提示必须在（去掉 listWindow 里的 `rows--`
// 就没有提示了）；②不够放时不许硬塞（少了 `end-start < capacity`
// 那半句，padTo 会把提示与进度一起顶掉）。
func TestLibrary_矮终端范围提示与进度行都在(t *testing.T) {
	items := make([]discover.Item, 30)
	for i := range items {
		items[i] = discover.Item{Source: discover.SourceGeneric,
			Name: fmt.Sprintf("model-%02d.gguf", i),
			Path: fmt.Sprintf("/x/%02d", i), Size: int64(i)}
	}
	orphan := discover.Item{Name: "sha256-orphan", Path: "/blobs/sha256-orphan", Size: 4096}
	newLib := func(res discover.Result) Library {
		lib := NewLibrary()
		lib.scan = func(context.Context, discover.Options) discover.Result { return res }
		// 不投递任何 Fill 结果 —— 这样"正在读取…"那一行是在途的
		lib.fill = func(it *discover.Item) *discover.Item { return it }
		lib2, _ := lib.Update(lib.Init()())
		lib = lib2.(Library)
		// 光标走到中间：范围提示才一定有东西可说（start > 0）
		for range 10 {
			lib2, _ = lib.Update(key("down"))
			lib = lib2.(Library)
		}
		return lib
	}
	view := func(lib Library) string {
		m, _ := New(lib).Update(tea.WindowSizeMsg{Width: 120, Height: 7})
		return m.(Model).View()
	}
	selected := func(lib Library) string { return "▸ " + lib.items[lib.cursor].Name }

	// ① 孤儿提示 2 行 + 进度行：容量 2 → 一个模型 + 范围提示 + 进度行，正好塞满
	lib := newLib(discover.Result{Items: items, Orphans: []discover.Item{orphan}})
	out := view(lib)
	if strings.Contains(out, "没显示") {
		t.Errorf("屏幕上出现了 padTo 的「还有 N 行没显示」—— 视图没算准高度:\n%s", out)
	}
	if !strings.Contains(out, "显示第") {
		t.Errorf("列表没显示完，范围提示那一行不在屏上:\n%s", out)
	}
	if !strings.Contains(out, "正在读取格式与参数量") {
		t.Errorf("在途的进度行被顶掉了:\n%s", out)
	}
	if !strings.Contains(out, selected(lib)) {
		t.Errorf("光标选中的那一行不在屏上:\n%s", out)
	}

	// ② 再加一条"未完成的下载"（提示 3 行）：容量只剩 1 行，装不下范围提示 ——
	//    这时**只能**保住进度行；硬塞的话两行都会换成 padTo 那句更差的话
	lib = newLib(discover.Result{Items: items, Orphans: []discover.Item{orphan},
		InProgress: []discover.Item{{Name: "sha256-x-partial", Path: "/p", Size: 1 << 30}}})
	out = view(lib)
	if strings.Contains(out, "没显示") {
		t.Errorf("容量只剩 1 行时硬塞提示行，padTo 把提示与进度一起顶掉了:\n%s", out)
	}
	if strings.Contains(out, "显示第") {
		t.Errorf("容量只剩 1 行，范围提示放不下却还是塞进去了:\n%s", out)
	}
	if !strings.Contains(out, "正在读取格式与参数量") {
		t.Errorf("在途的进度行被顶掉了:\n%s", out)
	}
}

// **目录警告要折行，不能被终端的宽度截掉。**
//
// 这条必须**从根 Model 渲染**：Library.View 自己不截断，
// 是根视图的 padTo 按终端宽度截的 —— 直接调 lib.View(80,24)
// 的话，折行与不折行没有任何区别（变异验证确认过：把 wrapText
// 去掉，那条测试照样绿）。
func TestLibrary_目录警告经根视图不被截断(t *testing.T) {
	lib := NewLibrary()
	lib.scan = func(context.Context, discover.Options) discover.Result {
		return discover.Result{
			Items: []discover.Item{{Name: "m", Path: "/m", Size: 1}},
			Errs: []string{"/blobs 下有 1 个 manifest 读不了，孤儿 blob 检测已跳过：" +
				"无法确认这些 blob 是否被引用，报成「可回收」可能让你删掉真实模型"},
		}
	}
	lib.fill = func(it *discover.Item) *discover.Item { return it }

	m := New(lib)
	m2, _ := m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	m = m2.(Model)
	// **必须真的跑一次 Init**：不跑的话界面停在"正在扫描模型目录…"，
	// 这条测试就什么都没验到（第一版就是这样，红得莫名其妙）
	m3, _ := m.Update(m.Init()())
	m = m3.(Model)

	out := m.View()
	// ① 每行不超宽
	for i, l := range strings.Split(out, "\n") {
		if w := displayWidth(l); w > 80 {
			t.Errorf("第 %d 行宽 %d 超过 80: %q", i, w, l)
		}
	}
	// ② 关键半句仍在（折行会把词断开，所以先拼回去再匹配）
	flat := strings.Join(strings.Fields(out), "")
	if !strings.Contains(flat, "可能让你删掉真实模型") {
		t.Errorf("破坏性警告的关键半句被终端宽度截掉了:\n%s", out)
	}
}
