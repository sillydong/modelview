package discover

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// fakeOllama 造一棵结构与本机真实 ollama 一致的目录树。
func fakeOllama(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	mk := func(rel string, data []byte) {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// 两个模型，共享一个 license 层 —— 共享是真实存在的（本机上 gemma4 的
	// e4b 与 26b 就共享 license 与 params），去重逻辑要能处理
	mk("blobs/sha256-aaaa", make([]byte, 1024)) // 模型 A
	mk("blobs/sha256-bbbb", make([]byte, 2048)) // 模型 B
	mk("blobs/sha256-cccc", []byte("license"))  // 共享层
	mk("blobs/sha256-dddd", []byte("template"))
	mk("blobs/sha256-eeee", []byte("孤儿")) // 没有任何 manifest 引用
	// 真实 manifest 是 Docker v2 结构，**有一个 config 字段** ——
	// 夹具必须照做，否则"config 的 blob 会不会被误判成孤儿"永远测不到
	mk("blobs/sha256-ffff", []byte("配置")) // config 层，被 alpha 引用
	mk("manifests/registry.ollama.ai/library/alpha/1b", []byte(`{
		"config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"sha256:ffff","size":6},
		"layers":[
		{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:aaaa","size":1024},
		{"mediaType":"application/vnd.ollama.image.license","digest":"sha256:cccc","size":7},
		{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:dddd","size":8}
	]}`))
	mk("manifests/registry.ollama.ai/library/beta/7b", []byte(`{
		"config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"sha256:ffff","size":6},
		"layers":[
		{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:bbbb","size":2048},
		{"mediaType":"application/vnd.ollama.image.license","digest":"sha256:cccc","size":7}
	]}`))
	return root
}

func names(items []Item) []string {
	out := make([]string, 0, len(items))
	for _, it := range items {
		out = append(out, it.Name)
	}
	sort.Strings(out)
	return out
}

// 模型名要从 manifest 路径推出来，blob 文件名本身不含模型信息。
func TestScanOllama_模型名与路径(t *testing.T) {
	root := fakeOllama(t)
	items, _, _, errs := scanOllama(root)
	if len(errs) != 0 {
		t.Fatalf("scanOllama 报了警告: %v", errs)
	}
	byName := map[string]Item{}
	for _, it := range items {
		byName[it.Name] = it
	}

	a, ok := byName["alpha:1b"]
	if !ok {
		t.Fatalf("没有 alpha:1b，有的是 %v", names(items))
	}
	if filepath.Base(a.Path) != "sha256-aaaa" {
		t.Errorf("alpha:1b 的路径 = %q，应指向 sha256-aaaa", a.Path)
	}
	if a.Size != 1024 {
		t.Errorf("alpha:1b 的大小 = %d, want 1024", a.Size)
	}
	if _, ok := byName["beta:7b"]; !ok {
		t.Errorf("没有 beta:7b，有的是 %v", names(items))
	}
	// 名字里不能带 registry 段
	for name := range byName {
		if strings.Contains(name, "registry") || strings.Contains(name, "library") {
			t.Errorf("模型名 %q 里带了 registry/路径段 —— 应当只留 <name>:<tag>", name)
		}
	}
	// 非模型层（license/template）不该被当成模型列出来
	for _, it := range items {
		base := filepath.Base(it.Path)
		if base == "sha256-cccc" || base == "sha256-dddd" {
			t.Errorf("%s 是非模型层，不该被列成模型", it.Path)
		}
	}
}

// 非模型层的内容要能读出来 —— 它是模型的"对话逻辑"（spec §7.1）。
func TestScanOllama_附加上层内容(t *testing.T) {
	root := fakeOllama(t)
	items, _, _, errs := scanOllama(root)
	if len(errs) != 0 {
		t.Fatalf("不该有警告: %v", errs)
	}
	var a Item
	for _, it := range items {
		if it.Name == "alpha:1b" {
			a = it
		}
	}
	got := map[string]string{}
	for _, kv := range a.Extra {
		got[kv.Key] = kv.Value
	}
	if got["license"] != "license" {
		t.Errorf("license 层的内容没读出来: %+v", a.Extra)
	}
	if got["template"] != "template" {
		t.Errorf("template 层的内容没读出来: %+v", a.Extra)
	}
	// beta:7b 只共享了 license，不该有 template
	for _, it := range items {
		if it.Name != "beta:7b" {
			continue
		}
		for _, kv := range it.Extra {
			if kv.Key == "template" {
				t.Errorf("beta:7b 没有 template 层，却读出了: %+v", it.Extra)
			}
		}
	}
}

// 孤儿 blob：没有被任何 manifest 引用的，要单独列出来并给出可回收空间。
func TestScanOllama_孤儿blob(t *testing.T) {
	root := fakeOllama(t)
	_, orphans, _, errs := scanOllama(root)
	if len(errs) != 0 {
		t.Fatalf("scanOllama 报了警告: %v", errs)
	}
	if len(orphans) != 1 {
		t.Fatalf("找到 %d 个孤儿，want 1（sha256-eeee）: %v", len(orphans), names(orphans))
	}
	if filepath.Base(orphans[0].Path) != "sha256-eeee" {
		t.Errorf("孤儿是 %q，want sha256-eeee", orphans[0].Path)
	}
	// 被两个 manifest 共享的 license 层**不是**孤儿
	for _, o := range orphans {
		if filepath.Base(o.Path) == "sha256-cccc" {
			t.Error("被共享引用的 license 层被误判成孤儿")
		}
		// **config 层的 blob 也不是孤儿** —— 真实 manifest 有 config 字段，
		// 漏读它会让每个模型都多报一个孤儿（实测本机 15 个 blob 里
		// 11 个在 layers、4 个是 config，正好少报 4 个）
		if filepath.Base(o.Path) == "sha256-ffff" {
			t.Error("config 层的 blob 被误判成孤儿 —— manifest 的 config 字段没读")
		}
	}
}

// manifest 损坏时不能中断整个扫描，也不能把已经解析出来的丢掉。
func TestScanOllama_坏manifest不影响其它(t *testing.T) {
	root := fakeOllama(t)
	bad := filepath.Join(root, "manifests/registry.ollama.ai/library/broken/1b")
	if err := os.MkdirAll(filepath.Dir(bad), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(bad, []byte("{ 不是 JSON"), 0o644); err != nil {
		t.Fatal(err)
	}
	items, orphans, _, errs := scanOllama(root)
	if len(items) != 2 {
		t.Errorf("找到 %d 个模型，want 2 —— 坏的 manifest 把好的也带下去了？%v",
			len(items), names(items))
	}
	// 坏 manifest 应当**报成警告**（不中断，但要说出来）
	if len(errs) == 0 {
		t.Error("有 manifest 读不了却没有任何警告 —— 用户会以为一切正常")
	}
	// **而且不能再报孤儿**：读不了的 manifest 引用了哪些 blob 无从知晓，
	// 把"不知道"当成"没被引用"，用户照着"可回收"去删就会删掉真实模型。
	// 实测过：一个截断的 manifest 会让它的大模型 blob 出现在孤儿列表里。
	if len(orphans) != 0 {
		t.Errorf("有 manifest 读不了时仍报了 %d 个孤儿 —— 那可能是真实模型：%v",
			len(orphans), names(orphans))
	}
}

// 没有坏 manifest 时，孤儿照常检测（不要为了防上面那种情况而永远不报）。
func TestScanOllama_正常时照常报孤儿(t *testing.T) {
	root := fakeOllama(t)
	items, orphans, _, errs := scanOllama(root)
	if len(errs) != 0 {
		t.Fatalf("不该有警告: %v", errs)
	}
	if len(items) != 2 {
		t.Fatalf("找到 %d 个模型，want 2", len(items))
	}
	if len(orphans) != 1 {
		t.Errorf("正常时孤儿 %d 个，want 1（sha256-eeee）", len(orphans))
	}
}

// manifest 指向的 blob 不在了：要报出来，不能装作模型还在。
func TestScanOllama_blob缺失要报错(t *testing.T) {
	root := fakeOllama(t)
	// 删掉模型 A 的 blob
	if err := os.Remove(filepath.Join(root, "blobs/sha256-aaaa")); err != nil {
		t.Fatal(err)
	}
	items, _, _, errs := scanOllama(root)
	if len(errs) != 0 {
		t.Fatalf("不该有警告: %v", errs)
	}
	found := false
	for _, it := range items {
		if it.Name == "alpha:1b" {
			found = true
			if it.Err == "" {
				t.Error("blob 不在磁盘上，条目却没有 Err —— 用户会以为模型还在")
			}
		}
	}
	if !found {
		t.Error("blob 缺失的条目整个消失了 —— 那用户看不到任何提示")
	}
}

// 目录不存在时返回空，不报错。
func TestScanOllama_目录不存在(t *testing.T) {
	items, orphans, _, errs := scanOllama(filepath.Join(t.TempDir(), "没有"))
	if len(items) != 0 || len(orphans) != 0 || len(errs) != 0 {
		t.Errorf("items=%v orphans=%v errs=%v，want 全空", items, orphans, errs)
	}
}

// manifestPathToName 的边界。
//
// 期望值**逐条对照上游 `DisplayShortest()` 的四条规则**（见 ollama.go 的
// 注释），不是照着本机文件推的 —— 本机 4 个模型全是
// `registry.ollama.ai/library/...`，只覆盖第一行那种最简情形。
//
// 这里曾经把 `registry.example.com/myorg/mymodel/1b` 的期望写成
// `myorg/mymodel:1b`，也就是把"无条件砍掉 host"这个偏差固化成了期望值 ——
// 于是测试守着的是实现，而不是 `ollama list` 的输出。
func TestManifestPathToName(t *testing.T) {
	dir := "/m"
	tests := []struct {
		path string
		want string
		ok   bool
	}{
		// host 与命名空间都是默认值 → 只剩 <名字>:<tag>
		{"/m/registry.ollama.ai/library/alpha/1b", "alpha:1b", true},
		{"/m/registry.ollama.ai/library/gemma4/26b", "gemma4:26b", true},
		// host 默认、命名空间非默认 → 保留命名空间
		{"/m/registry.ollama.ai/myorg/mymodel/1b", "myorg/mymodel:1b", true},
		// **host 非默认 → host 与命名空间都要打印**，即使命名空间是 library
		{"/m/registry.example.com/myorg/mymodel/1b",
			"registry.example.com/myorg/mymodel:1b", true},
		// 从 HuggingFace 直接拉的 GGUF 就是这种形状，很常见
		{"/m/hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF/Q4_K_M",
			"hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", true},
		// 三段（host/名字/tag）：命名空间按默认值补齐，仍然只剩 <名字>:<tag>
		{"/m/registry.ollama.ai/alpha/1b", "alpha:1b", true},
		// 多出来的层并进名字：上游的 Name 只有一个 model 字段，
		// 遇到这种目录结构，拼起来比丢掉更能让用户认出来是哪个
		{"/m/registry.ollama.ai/library/ns/inner/tag", "ns/inner:tag", true},
		{"/m/registry.ollama.ai/alpha", "", false}, // 层数不够
		{"/m/registry.ollama.ai", "", false},
	}
	for _, tt := range tests {
		got, ok := manifestPathToName(dir, tt.path)
		if ok != tt.ok || got != tt.want {
			t.Errorf("manifestPathToName(%q) = (%q, %v), want (%q, %v)",
				tt.path, got, ok, tt.want, tt.ok)
		}
	}
}

// Scan 走 ollama 来源时，孤儿也要带上。
func TestScan_Ollama孤儿进入结果(t *testing.T) {
	root := fakeOllama(t)
	// fakeOllama 造的树根就是 models/，把它挂到 <home>/.ollama/models 下
	home := t.TempDir()
	dst := filepath.Join(home, ".ollama", "models")
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(root, dst); err != nil {
		t.Fatal(err)
	}
	res := Scan(context.Background(), Options{Home: home, Only: []Source{SourceOllama}})
	if len(res.Errs) != 0 {
		t.Fatalf("扫描报错: %v", res.Errs)
	}
	if len(res.Items) != 2 {
		t.Errorf("找到 %d 个模型，want 2: %v", len(res.Items), names(res.Items))
	}
	if len(res.Orphans) != 1 {
		t.Errorf("孤儿 %d 个，want 1 —— Scan 没把孤儿接出来", len(res.Orphans))
	}
}

// 孤儿 blob 的大小必须按**目标**算，不是链接自身的字节数。
//
// 这一列要回答的是"能回收多少空间"。blob 是符号链接时报链接的字节数，
// 用户会以为没什么可清的 —— 同一个 lstat/stat 陷阱，
// 在 scanDir 那边有测试守着，这边一样要有。
func TestScanOllama_孤儿大小按目标(t *testing.T) {
	root := fakeOllama(t)
	// 把一个孤儿做成指向大文件的符号链接
	target := filepath.Join(root, "real-payload")
	big := make([]byte, 1<<20) // 1 MiB
	if err := os.WriteFile(target, big, 0o644); err != nil {
		t.Fatal(err)
	}
	orphan := filepath.Join(root, "blobs", "sha256-eeee")
	if err := os.Remove(orphan); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, orphan); err != nil {
		t.Fatal(err)
	}

	_, orphans, _, errs := scanOllama(root)
	if len(errs) != 0 {
		t.Fatalf("扫描报错: %v", errs)
	}
	var got *Item
	for i := range orphans {
		if orphans[i].Path == orphan {
			got = &orphans[i]
		}
	}
	if got == nil {
		t.Fatalf("孤儿 %s 没被列出来", orphan)
	}
	if got.Size != int64(len(big)) {
		t.Errorf("孤儿报的大小是 %d，应当是目标的 %d（链接自身才是 %d）",
			got.Size, len(big), len(orphan))
	}
}

// **未完成的下载不能被报成"可回收"。**
//
// ollama 下载时先建 `<digest>-partial`（预分配到最终大小，所以看着是满的），
// 另有一批 `<digest>-partial-<N>` 分片。实测拉 gpt-oss:20b 到一半时，
// 扫描输出写着"另有 17 个孤儿 blob 可回收 12.85 GiB"—— 而那 12.85 GiB
// 正是那个下到一半的模型。用户照着"可回收"去删，就把自己的下载毁了。
func TestScanOllama_未完成的下载不是孤儿(t *testing.T) {
	root := fakeOllama(t)
	digest := strings.Repeat("ab", 32) // 64 位十六进制
	partial := filepath.Join(root, "blobs", "sha256-"+digest+"-partial")
	chunk := filepath.Join(root, "blobs", "sha256-"+digest+"-partial-3")
	for _, p := range []string{partial, chunk} {
		if err := os.WriteFile(p, make([]byte, 4096), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	_, orphans, inProgress, errs := scanOllama(root)
	for _, o := range orphans {
		if o.Path == partial || o.Path == chunk {
			t.Errorf("未完成的下载被报成孤儿（可回收）: %s —— "+
				"用户照着删会毁掉自己的下载", filepath.Base(o.Path))
		}
	}
	// 也不能一声不吭：磁盘被占着，用户得有解释。
	// **断言的是数据（InProgress）而不是某句文案** —— 文案会改，
	// 而"这两个文件被认出来了"是事实
	got := map[string]bool{}
	for _, it := range inProgress {
		got[filepath.Base(it.Path)] = true
	}
	if !got[filepath.Base(partial)] || !got[filepath.Base(chunk)] {
		t.Errorf("未完成的下载没被单独列出来 —— 用户找不到那 12.85 GiB 去哪了: %v", got)
	}
	// 它也**不是"错误"**：Errs 是失败原因通道，把说明塞进去会让
	// "真实语料扫描不该报错"那条断言失效
	if len(errs) != 0 {
		t.Errorf("未完成的下载不该进 Errs（那是失败原因通道）: %v", errs)
	}
	// 正常孤儿照报（别为了防上面那种而少报）
	if len(orphans) != 1 || filepath.Base(orphans[0].Path) != "sha256-eeee" {
		t.Errorf("孤儿 %d 个：%v，want 只有 sha256-eeee", len(orphans), names(orphans))
	}
}
