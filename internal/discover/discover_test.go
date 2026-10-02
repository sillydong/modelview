package discover

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeModel 在 dir 下写一个可被 detect 认出的最小 GGUF 文件。
func writeModel(t *testing.T, dir, name string) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, name)
	buf := make([]byte, 64)
	copy(buf, []byte{'G', 'G', 'U', 'F', 3, 0, 0, 0})
	if err := os.WriteFile(p, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// 通用目录要递归找到模型文件，且按扩展名过滤。
func TestScanDir_找到模型文件(t *testing.T) {
	root := t.TempDir()
	writeModel(t, filepath.Join(root, "a"), "m1.gguf")
	writeModel(t, filepath.Join(root, "b", "c"), "m2.gguf")
	// 非模型文件不该被收进来
	if err := os.WriteFile(filepath.Join(root, "readme.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	// 扩展名对但内容不对的也要能列出来（格式留空、Fill 时才报错）
	if err := os.WriteFile(filepath.Join(root, "bad.gguf"), []byte("not a model"), 0o644); err != nil {
		t.Fatal(err)
	}

	items, err := scanDir(context.Background(), root, SourceGeneric)
	if err != nil {
		t.Fatalf("scanDir 失败: %v", err)
	}
	if len(items) != 3 {
		t.Fatalf("找到 %d 个候选，want 3（两个好的 + 一个内容坏的）", len(items))
	}
	// 顺序必须稳定 —— 界面每次刷新时列表乱跳是不可用的
	for i := 1; i < len(items); i++ {
		if items[i-1].Path > items[i].Path {
			t.Errorf("结果没有按路径排序：%q 在 %q 之前", items[i-1].Path, items[i].Path)
			break
		}
	}
}

// 递归要有限度：.git 与 node_modules 不进去，深度超过 maxDepth 不进去。
func TestScanDir_跳过与限深(t *testing.T) {
	root := t.TempDir()
	writeModel(t, filepath.Join(root, ".git"), "a.gguf")
	writeModel(t, filepath.Join(root, "node_modules"), "b.gguf")
	writeModel(t, filepath.Join(root, "ok"), "c.gguf")

	// 深度用**字面量**，不引用 maxDepth ——
	// 引用的话变异把上限改成 100，测试也会跟着造一个 101 层的东西，
	// 边界就永远测不到（变异验证时确认过这条逃逸）
	const atLimit, overLimit = 6, 7
	if maxDepth != atLimit {
		t.Fatalf("maxDepth = %d，测试按 %d 写死了边界 —— 改上限时同步改这里",
			maxDepth, atLimit)
	}
	deep := root
	for i := 1; i <= overLimit; i++ {
		deep = filepath.Join(deep, fmt.Sprintf("d%d", i))
	}
	writeModel(t, deep, "over.gguf")

	// 正好在上限上的那一层要能收到
	atEdge := root
	for i := 1; i <= atLimit; i++ {
		atEdge = filepath.Join(atEdge, fmt.Sprintf("d%d", i))
	}
	writeModel(t, atEdge, "at.gguf")

	items, err := scanDir(context.Background(), root, SourceGeneric)
	if err != nil {
		t.Fatalf("scanDir 失败: %v", err)
	}
	names := map[string]bool{}
	for _, it := range items {
		names[filepath.Base(it.Path)] = true
	}
	if names["a.gguf"] {
		t.Error(".git 底下的文件被收进来了")
	}
	if names["b.gguf"] {
		t.Error("node_modules 底下的文件被收进来了")
	}
	if !names["c.gguf"] {
		t.Error("正常目录下的文件没被收进来")
	}
	if names["over.gguf"] {
		t.Errorf("超过 %d 层的文件被收进来了", atLimit)
	}
	if !names["at.gguf"] {
		t.Errorf("正好 %d 层的文件没被收进来 —— 边界取错了方向", atLimit)
	}
}

// 目录不存在时**不该报错**，也不该进 Result.Errs。
//
// 本机 12 条路径里只有 1 条存在 —— 另外 11 条都报错的话，
// 输出会被噪音淹没，而"没装那个框架"根本不是错误。
//
// 这条断言的是**整条扫描路径**（Scan 而不只是 scanDir）：
// scanDir 自己返回 nil 是不够的，Scan 里把它记进 Errs 一样是错的。
func TestScan_缺目录不进Errs(t *testing.T) {
	home := t.TempDir()
	writeModel(t, filepath.Join(home, "models"), "m.gguf")

	// 不限定 Only：12 条路径全走一遍，其中 11 条不存在
	res := Scan(context.Background(), Options{Home: home})
	if len(res.Errs) != 0 {
		t.Errorf("缺目录产生了 %d 条错误（应当一条都没有）: %v", len(res.Errs), res.Errs)
	}
	if len(res.Items) != 1 {
		t.Errorf("找到 %d 个条目，want 1", len(res.Items))
	}
}

// 目录不存在时**不该报错** —— 本机多半只装了一两个推理框架，
// 缺失的目录是常态不是异常。
func TestScanDir_目录不存在不报错(t *testing.T) {
	items, err := scanDir(context.Background(),
		filepath.Join(t.TempDir(), "没有这个目录"), SourceGeneric)
	if err != nil {
		t.Fatalf("目录不存在时不该报错: %v", err)
	}
	if len(items) != 0 {
		t.Errorf("找到 %d 个条目，应为 0", len(items))
	}
}

// 路径清单里的每条都必须说明来源，且路径要么是绝对路径要么以 ~ 开头。
func TestPaths_形状正确(t *testing.T) {
	ps := Paths()
	if len(ps) != 12 {
		t.Fatalf("只有 %d 条路径，spec 列了 12 条", len(ps))
	}
	seen := map[Source]bool{}
	for _, p := range ps {
		if p.Source == "" || p.Dir == "" {
			t.Errorf("路径条目缺字段: %+v", p)
		}
		if !filepath.IsAbs(p.Dir) && !strings.HasPrefix(p.Dir, "~") {
			t.Errorf("%s 的路径 %q 既不是绝对路径也不以 ~ 开头", p.Source, p.Dir)
		}
		seen[p.Source] = true
	}
	// 反向门禁：路径条数掉了说明清单被删了
	const wantPaths = 12
	if len(ps) != wantPaths {
		t.Errorf("路径有 %d 条，预期 %d 条 —— 清单变了，核对后再改这个数",
			len(ps), wantPaths)
	}
	// ollama 必须走 manifest（它的 blob 是内容寻址的，递归出来的是一堆哈希）
	found := false
	for _, p := range ps {
		if p.Source == SourceOllama {
			found = true
			if p.manifest == nil {
				t.Error("ollama 那条必须带 manifest 处理器 —— 否则会去递归 blobs/ 下的哈希文件")
			}
		}
	}
	if !found {
		t.Error("路径清单里没有 ollama")
	}
	// 反过来：不带处理器的条目不能扫出孤儿 blob 来（孤儿只有 ollama 有这个概念）
	for _, p := range ps {
		if p.Source != SourceOllama && p.manifest != nil {
			t.Errorf("%s 带了 manifest 处理器，但它不是内容寻址的存储", p.Source)
		}
	}
}

// expandHome 只展开开头的 ~/，中间的 ~ 是文件名的一部分。
func TestExpandHome(t *testing.T) {
	tests := []struct {
		in, home, want string
	}{
		{"~/models", "/h", "/h/models"},
		{"~", "/h", "/h"},
		{"/abs/path", "/h", "/abs/path"},
		{"~/a~b/c", "/h", "/h/a~b/c"}, // 中间的 ~ 不动
		{"~/x", "", "~/x"},            // 拿不到 home 时原样返回
	}
	for _, tt := range tests {
		if got := expandHome(tt.in, tt.home); got != tt.want {
			t.Errorf("expandHome(%q, %q) = %q, want %q", tt.in, tt.home, got, tt.want)
		}
	}
}

// Scan 必须只 stat、不读内容 —— 这是分层加载的前提。
//
// 判据不是"耗时小于 X 毫秒"（CI 上会飘，本机与 CI 的磁盘差很远），
// 而是**行为**：拿一个内容坏掉但扩展名对的文件，Scan 照样列出它
// 且不带 Err；Fill 之后才带 Err。Scan 阶段读内容的话，
// 它会在首屏就报错，异步分层也就名存实亡。
func TestScan_只stat不读内容(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, "models")
	writeModel(t, dir, "good.gguf")
	if err := os.WriteFile(filepath.Join(dir, "bad.gguf"), []byte("这不是模型"), 0o644); err != nil {
		t.Fatal(err)
	}

	res := Scan(context.Background(), Options{Home: home, Only: []Source{SourceGeneric}})
	if len(res.Items) != 2 {
		t.Fatalf("Scan 列出 %d 个条目，want 2（按扩展名就该收下）", len(res.Items))
	}
	for _, it := range res.Items {
		if it.Err != "" {
			t.Errorf("Scan 阶段就报了错（%q）—— 说明它读了内容，分层加载没生效", it.Err)
		}
		if it.Format != "" {
			t.Errorf("Scan 阶段就填了 Format（%q）—— 同上", it.Format)
		}
	}

	// Fill 之后：好的要有 Format，坏的要有 Err
	res.Items = FillAll(context.Background(), res.Items)
	byName := map[string]Item{}
	for _, it := range res.Items {
		byName[filepath.Base(it.Path)] = it
	}
	if got := byName["good.gguf"]; got.Format == "" {
		t.Errorf("good.gguf 读不出格式: Err=%q", got.Err)
	}
	if got := byName["bad.gguf"]; got.Err == "" {
		t.Error("bad.gguf 内容不是模型，Fill 之后应当带 Err")
	}
}

// Fill 对每个条目都必须留下"格式"或"错误"之一 —— 不能两者皆空。
//
// 两者皆空的话，界面上这一行看起来就是"还没读完"，
// 而实际上永远读不完了，用户会一直等。
func TestFill_不留悬空条目(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, "models")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// 空文件：既不是已知格式，也不至于读报错
	if err := os.WriteFile(filepath.Join(dir, "empty.gguf"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	res := Scan(context.Background(), Options{Home: home, Only: []Source{SourceGeneric}})
	for i := range res.Items {
		Fill(&res.Items[i])
		if res.Items[i].Format == "" && res.Items[i].Err == "" {
			t.Errorf("%s 既没有 Format 也没有 Err —— 界面上会永远显示「读取中」",
				res.Items[i].Path)
		}
	}
}

// 只扫指定来源时，别的来源不该出现。
func TestScan_Only过滤(t *testing.T) {
	home := t.TempDir()
	writeModel(t, filepath.Join(home, "models"), "m.gguf")
	res := Scan(context.Background(), Options{Home: home, Only: []Source{SourceGeneric}})
	for _, it := range res.Items {
		if it.Source != SourceGeneric {
			t.Errorf("只让扫 %s，却出现了 %s", SourceGeneric, it.Source)
		}
	}
	if len(res.Items) != 1 {
		t.Errorf("找到 %d 个，want 1", len(res.Items))
	}
}

// 多个来源的结果必须**全局**排序，不能按来源分段拼接。
//
// 这条是唯一能区分"排序"与"没排序"的用例：单个来源时，
// Paths() 的列举顺序恰好与路径的字典序一致（都是 ~/.xxx 在前、
// ~/models 在后），去掉排序照样是绿的。
//
// ollama 是唯一会打破这个巧合的来源 —— 它列在清单第一位，
// 而它展开后的 ~/.ollama 在字典序上排在 ~/.cache 之后。
func TestScan_多来源结果全局排序(t *testing.T) {
	home := t.TempDir()

	// 来源一：ollama（清单里排第 1，字典序上却在 ~/.cache 之后）
	ollamaRoot := filepath.Join(home, ".ollama", "models")
	if err := os.MkdirAll(filepath.Dir(ollamaRoot), 0o755); err != nil {
		t.Fatal(err)
	}
	src := fakeOllama(t)
	if err := os.Rename(src, ollamaRoot); err != nil {
		t.Fatal(err)
	}

	// 来源二：llama.cpp（清单里排第 7）
	writeModel(t, filepath.Join(home, ".cache", "llama.cpp"), "llama-model.gguf")

	res := Scan(context.Background(), Options{Home: home})
	if len(res.Items) != 3 {
		t.Fatalf("找到 %d 个条目，want 3（ollama 两个 + llama.cpp 一个）: %v",
			len(res.Items), names(res.Items))
	}
	for i := 1; i < len(res.Items); i++ {
		if res.Items[i-1].Path > res.Items[i].Path {
			t.Errorf("结果没有全局排序：\n  [%d] %s\n  [%d] %s\n"+
				"（按来源分段拼接的话就会这样 —— ollama 排在最前，但它的路径字典序靠后）",
				i-1, res.Items[i-1].Path, i, res.Items[i].Path)
			break
		}
	}
}

// Scan 返回的每个条目都必须有非空的 Name —— 这是给消费方的契约（见 Item.Name）。
//
// **在生产者这一侧钉，不在渲染那一侧兜。** 渲染侧写
// `if Name == "" { 用文件名 }` 看着更稳妥，但那条分支永远不会被执行到
// （现在没有生产者会产出空名字），于是它不提供任何保护，
// 只是让"生产者漏填"这件事看起来已经被处理了。
//
// 名字还必须**不是**从路径推的：ollama 的 Path 是 blobs/sha256-…，
// 拿它当名字，用户在界面上看到的就是一串哈希。
func TestScan_每个条目都有名字(t *testing.T) {
	home := t.TempDir()
	ollamaRoot := filepath.Join(home, ".ollama", "models")
	if err := os.MkdirAll(filepath.Dir(ollamaRoot), 0o755); err != nil {
		t.Fatal(err)
	}
	src := fakeOllama(t)
	if err := os.Rename(src, ollamaRoot); err != nil {
		t.Fatal(err)
	}
	writeModel(t, filepath.Join(home, ".cache", "llama.cpp"), "llama-model.gguf")

	res := Scan(context.Background(), Options{Home: home})
	if len(res.Items) == 0 {
		t.Fatal("一个条目都没扫到，这条测试没验到东西")
	}
	for _, it := range res.Items {
		if it.Name == "" {
			t.Errorf("%s 的 Name 是空的 —— 界面第一列会空着", it.Path)
		}
		if it.Name == filepath.Base(it.Path) && it.Source == SourceOllama {
			t.Errorf("ollama 条目的名字是文件名 %q —— 那是 blobs 下的哈希，"+
				"用户拿它去 `ollama list` 里找不到", it.Name)
		}
	}
	// 孤儿 blob 也要有名字（不然孤儿那一节整列是空的）
	for _, o := range res.Orphans {
		if o.Name == "" {
			t.Errorf("孤儿 blob %s 没有名字", o.Path)
		}
	}
}

// 没有找到任何模型时，Items 必须是空切片而不是 nil。
//
// JSON 里 nil 编码成 `null`，空切片编码成 `[]` —— 对消费方是两回事：
// 前者要额外判空。契约应当是"没有模型时给一个空数组"。
func TestScan_空结果不是nil(t *testing.T) {
	home := t.TempDir() // 一个模型都没有
	res := Scan(context.Background(), Options{Home: home})
	if res.Items == nil {
		t.Error("Items 是 nil，JSON 会输出 null —— 消费方要多写一个判空")
	}
	if len(res.Items) != 0 {
		t.Errorf("空目录里找到 %d 个条目", len(res.Items))
	}
	// 编码出来必须是 []，不是 null
	raw, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"items":[]`) {
		t.Errorf("JSON 里 items 不是空数组: %s", raw)
	}
}

// 符号链接的模型文件必须按**目标**的大小报告，不能按链接自身的字节数。
//
// HF 缓存的布局就是这样：`snapshots/<sha>/model.safetensors` 是指向
// `blobs/<sha>` 的符号链接（`maxDepth` 注释点名的首要场景）。
// WalkDir 给的 DirEntry 是 **lstat 语义** —— `d.Info()` 返回链接本身的
// 信息（几十字节），于是首屏那一列全是"48 B"，而 Fill 之后才变成真大小。
// "解析失败"的条目则永远留着这个错值。
func TestScanDir_符号链接按目标大小(t *testing.T) {
	root := t.TempDir()
	blobs := filepath.Join(root, "blobs")
	if err := os.MkdirAll(blobs, 0o755); err != nil {
		t.Fatal(err)
	}
	real := filepath.Join(blobs, "sha256-abc")
	body := make([]byte, 64<<10) // 64 KiB，与链接长度差三个数量级
	copy(body, []byte{'G', 'G', 'U', 'F', 3, 0, 0, 0})
	if err := os.WriteFile(real, body, 0o644); err != nil {
		t.Fatal(err)
	}
	snap := filepath.Join(root, "snapshots", "deadbeef")
	if err := os.MkdirAll(snap, 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(snap, "model.gguf")
	if err := os.Symlink(real, link); err != nil {
		t.Fatal(err)
	}

	items, err := scanDir(context.Background(), root, SourceHuggingFace)
	if err != nil {
		t.Fatal(err)
	}
	var got *Item
	for i := range items {
		if items[i].Path == link {
			got = &items[i]
		}
	}
	if got == nil {
		t.Fatalf("没扫到符号链接 %s，扫到的是 %v", link, names(items))
	}
	if got.Size != int64(len(body)) {
		t.Errorf("符号链接的大小是 %d，应当是目标的 %d（链接自身的长度才是 %d）",
			got.Size, len(body), len(link))
	}
}

// extra 的 JSON 键名要与同一份 JSON 里的其它字段一致（snake_case）。
//
// 原来 KV 没有 json tag，打出来是 {"Key":…,"Value":…}，而 Item 的
// 其它字段全是 snake_case（source/name/path/size/format/arch/
// param_count）。消费方要为这一个字段破例。
func TestItem_extra用snake_case(t *testing.T) {
	raw, err := json.Marshal(Item{Extra: []KV{{Key: "license", Value: "apache-2.0"}}})
	if err != nil {
		t.Fatal(err)
	}
	got := string(raw)
	if strings.Contains(got, `"Key"`) || strings.Contains(got, `"Value"`) {
		t.Errorf("extra 的键名还是 PascalCase:\n%s", got)
	}
	for _, want := range []string{`"key":"license"`, `"value":"apache-2.0"`} {
		if !strings.Contains(got, want) {
			t.Errorf("缺少 %s:\n%s", want, got)
		}
	}
	// 同一个结构体里的其它字段也要一起看 —— 只改 KV 不改别的。
	//
	// **不能拿 param_count 当代表**：它是 omitempty，零值时不出现
	//（第一版就是这么写的，于是断言恒假）。source 是必出的。
	if !strings.Contains(got, `"source"`) {
		t.Errorf("Item 的其它字段应当仍是 snake_case:\n%s", got)
	}
}
