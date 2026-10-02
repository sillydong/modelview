package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sillydong/modelview/internal/analyze"
	"github.com/sillydong/modelview/internal/parser"
)

func TestOrDash(t *testing.T) {
	if got := orDash(""); got != "-" {
		t.Errorf("orDash(\"\") = %q, want -", got)
	}
	if got := orDash("qwen2"); got != "qwen2" {
		t.Errorf("orDash(qwen2) = %q", got)
	}
}

func u32(v uint32) []byte { b := make([]byte, 4); binary.LittleEndian.PutUint32(b, v); return b }
func u64(v uint64) []byte { b := make([]byte, 8); binary.LittleEndian.PutUint64(b, v); return b }

func ggufStr(x string) []byte {
	b := []byte(x)
	return append(u64(uint64(len(b))), b...)
}

// buildMinimalGGUF 手写一个只含一个张量的 GGUF v3。
//
// 头部结构依次是：magic、version(u32)、tensor_count(u64)、kv_count(u64)、
// KV 对、张量描述、对齐填充，最后才是数据区。
//
// typeCode 是 ggml 的类型码（0 = F32、12 = Q4_K，见 model.Dtype 的 GGMLCode）。
func buildMinimalGGUF(name string, typeCode uint32, nElems int, raw []byte) []byte {
	var out []byte
	out = append(out, "GGUF"...)
	out = append(out, u32(3)...) // version
	out = append(out, u64(1)...) // tensor_count
	out = append(out, u64(2)...) // kv_count
	out = append(out, ggufStr("general.architecture")...)
	out = append(out, u32(8)...) // 值类型 string
	out = append(out, ggufStr("llama")...)
	out = append(out, ggufStr("general.alignment")...)
	out = append(out, u32(4)...) // 值类型 u32
	out = append(out, u32(32)...)
	out = append(out, ggufStr(name)...)
	out = append(out, u32(1)...) // n_dims
	out = append(out, u64(uint64(nElems))...)
	out = append(out, u32(typeCode)...)
	out = append(out, u64(0)...) // 数据区偏移
	for len(out)%32 != 0 {
		out = append(out, 0)
	}
	return append(out, raw...)
}

// writeNaNScaleGGUF 造一个含 Q4_K 张量的最小 GGUF，把一个块的
// **float16 scale 写成 NaN**（块头前 2 字节，小端 0x7e00）。
//
// 一个块是 144 字节 / 256 个权重；块头里的 d 是 float16，
// 它是这个块全部 8 个子块 scale 的公共因子，所以一个 NaN 会染到 8 个子块。
func writeNaNScaleGGUF(t *testing.T, dir string) string {
	t.Helper()
	const blkElems, blkBytes = 256, 144
	raw := make([]byte, blkBytes)
	raw[0], raw[1] = 0x00, 0x7e // float16 NaN，小端
	p := filepath.Join(dir, "nan_scale.gguf")
	if err := os.WriteFile(p, buildMinimalGGUF("t.q4k", 12 /* Q4_K */, blkElems, raw), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// 非有限值不得让整个 --json 序列化失败。
//
// 这是**最初那个症状**的回归：encoding/json 拒绝 NaN/Inf，而它们一旦
// 进入 QuantInfo 的 scale 字段，报错的是整个输出（不是这一张张量）。
// 实测在真实 qwen2.5:3b（1.9 GB）上改 6 个字节就能让 434 个张量的结果全丢。
func TestJSON_非有限值不再让整个输出失败(t *testing.T) {
	p := writeNaNScaleGGUF(t, t.TempDir())

	m, err := parser.Parse(p)
	if err != nil {
		t.Fatalf("解析失败: %v", err)
	}
	if _, err := analyze.Analyze(t.Context(), m, analyze.Options{SampleLimit: -1}); err != nil {
		t.Fatalf("分析失败: %v", err)
	}

	// 这一步就是原来会挂的地方
	out := struct {
		Metadata   any   `json:"metadata"`
		Tensors    any   `json:"tensors"`
		ParamCount int64 `json:"param_count"`
	}{m.Metadata, m.Tensors, m.TotalParams()}
	raw, err := json.Marshal(out)
	if err != nil {
		t.Fatalf("序列化失败（非有限值又漏出来了）: %v", err)
	}

	var back struct {
		Tensors []map[string]any `json:"tensors"`
	}
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	if len(back.Tensors) != 1 {
		t.Fatalf("张量数 = %d, want 1", len(back.Tensors))
	}
	q, ok := back.Tensors[0]["quant"].(map[string]any)
	if !ok {
		t.Fatalf("张量没有 quant 字段: %v", back.Tensors[0])
	}
	// **非有限值必须可见**，不能只是被悄悄丢掉。
	// 一个 Q4_K 块有 8 个子块，d 是它们的公共因子 —— 所以是 8 不是 1。
	if n, _ := q["non_finite_scales"].(float64); n != 8 {
		t.Errorf("non_finite_scales = %v, want 8（一个块的 8 个子块共用那个 NaN 的 d）",
			q["non_finite_scales"])
	}
	// 一个可用的 scale 都没有时 min/max 留零值，且**不能是 NaN**
	if v, ok := q["scale_min"].(float64); !ok || v != 0 {
		t.Errorf("scale_min = %v, want 0", q["scale_min"])
	}
}

// 含 NaN 的数据也要能写缓存 —— 原来 json.Marshal 失败后 save 直接
// return nil，缓存目录建了却永远是空的，用户不知道缓存已经失效。
func TestCache_非有限值的数据也能落盘(t *testing.T) {
	dir := t.TempDir()
	p := writeNaNScaleGGUF(t, dir)

	m, err := parser.Parse(p)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := analyze.Analyze(t.Context(), m, analyze.Options{SampleLimit: -1}); err != nil {
		t.Fatal(err)
	}

	entries, err := os.ReadDir(filepath.Join(dir, ".modelview-cache"))
	if err != nil {
		t.Fatalf("缓存目录不存在: %v", err)
	}
	if len(entries) == 0 {
		t.Error("缓存目录是空的 —— save 静默失败了（json.Marshal 又碰上了非有限值）")
	}
}

// resetFlags 重置全局 flag 集合并设置 os.Args。
//
// **必须做这件事**：run() 里调的是 flag.Parse()，而 flag.CommandLine 是
// 全局的 —— 同一个进程里跑第二次 run() 会因 flag 重复注册而 panic
// （"flag redefined: json"）。那个 panic 看起来像被测代码崩了，实际是
// 测试自己造成的，很容易误判。SetOutput(io.Discard) 是顺手把它在
// 用法错误时打的噪音收掉（要断言输出时用管道接 os.Stdout）。
func resetFlags(t *testing.T, args ...string) {
	t.Helper()
	old := os.Args
	flag.CommandLine = flag.NewFlagSet("modelview", flag.ContinueOnError)
	flag.CommandLine.SetOutput(io.Discard)
	os.Args = append([]string{"modelview"}, args...)
	t.Cleanup(func() { os.Args = old })
}

// `--` 之后的参数是纯位置参数，不该再套用"选项写在位置参数后面"的守卫。
//
// flag.Parse 会把 `--` 自己消费掉，所以 flag.Args() 里仍然留着它后面的
// `-weird.gguf` —— 只看 Args() 的话，原来的守卫照样拒绝，而**提示语教的
// 正是"写 `--` 可以显式终止选项解析"**：一个不存在的逃生口。
// 名字以 - 开头的模型文件因此永远打不开。
func TestRun_双横线后的参数被当作路径(t *testing.T) {
	dir := t.TempDir()
	name := "-weird.gguf"
	if err := os.WriteFile(filepath.Join(dir, name),
		buildMinimalGGUF("t", 0, 4, make([]byte, 16)), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)
	resetFlags(t, "--no-cache", "--", name)

	if err := run(); err != nil {
		t.Fatalf("`--` 之后的路径被当成选项了: %v", err)
	}
}

// 反面：**没有** `--` 时那条守卫必须照旧生效 ——
// 否则 `modelview scan --json` 里的 --json 会被静默丢掉，
// 打出来的人类可读列表被管道给 jq 才报错，而报错的地方离原因很远。
func TestRun_选项写在位置参数后面仍要报错(t *testing.T) {
	resetFlags(t, "scan", "--json")
	err := run()
	if err == nil {
		t.Fatal("`scan --json` 应当报错（选项会被静默丢掉）")
	}
	// **断言必须打在守卫特有的文案上**。
	// 第一版写的是 `strings.Contains(err.Error(), "--json")` ——
	// 而关掉守卫之后 run() 会返回「交互界面需要终端；非交互请用
	// modelview --json scan」，那句提示里**也含 --json**，
	// 于是测试照样通过（变异验证实测报「漏网」）。
	// 判据要挑一个只有守卫会说的词。
	if !strings.Contains(err.Error(), "写在位置参数后面") {
		t.Errorf("应当由那条守卫报错，得到: %v", err)
	}
	if !strings.Contains(err.Error(), "--json") {
		t.Errorf("错误信息应指明是哪个选项，得到: %v", err)
	}
}

// --sample-limit 0 时要先给出预估峰值。
//
// 帮助文本里那句"峰值内存可达数 GB"是泛泛的，用户在看到自己这个文件的
// 数字之前不会把它和自己联系起来。实测（优化后）最大张量 3.11 亿元素
// 时峰值 2.4 GiB，约 8 B/元素。
func TestRun_不采样时给出预估(t *testing.T) {
	dir := t.TempDir()
	// **必须大到超过预警阈值**：warnNoSample 对预估 < 1 GiB 的不提示
	//（小模型上那只是噪音）。按 8 B/元素算，1 GiB 对应约 1.34 亿元素 ——
	// 第一版用了 1e8（763 MiB），于是测试测的是"阈值以下"那条路径。
	p := filepath.Join(dir, "big.gguf")
	if err := os.WriteFile(p, buildMinimalGGUF("big", 0 /*F32*/, 200_000_000, nil), 0o644); err != nil {
		t.Fatal(err)
	}

	r, w, _ := os.Pipe()
	oldErr := os.Stderr
	os.Stderr = w
	t.Cleanup(func() { os.Stderr = oldErr })

	resetFlags(t, "--no-cache", "--stats", "--sample-limit", "0", p)
	// 统计会失败（数据区是空的），但那不是这条测试关心的 ——
	// 预警必须在**开始跑之前**就打出来
	_ = run()
	_ = w.Close() // 关掉写端让读端拿到 EOF；失败不影响断言
	got, _ := io.ReadAll(r)

	if !strings.Contains(string(got), "预估") {
		t.Errorf("--sample-limit 0 时没有给出预估:\n%s", got)
	}
	// 2e8 元素 × 8 B ≈ 1.49 GiB —— 量级要对得上
	if !strings.Contains(string(got), "GiB") {
		t.Errorf("预估里没有量级:\n%s", got)
	}
}

// 默认采样（不传 --sample-limit）时不打这条 —— 否则每次跑都多一行噪音。
//
// **fixture 必须与上面那条一样大**：第一版用了 1e8 元素（763 MiB），
// 低于 1 GiB 的预警阈值，于是"不预警"这个结果由**大小**就能解释 ——
// 把 `limit != 0` 那道守卫删掉它照样通过（变异验证会报「漏网」）。
// 用同样超阈值的输入，"不预警"才只能由守卫解释。
func TestRun_默认采样不预警(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "big.gguf")
	if err := os.WriteFile(p, buildMinimalGGUF("big", 0, 200_000_000, nil), 0o644); err != nil {
		t.Fatal(err)
	}

	r, w, _ := os.Pipe()
	oldErr := os.Stderr
	os.Stderr = w
	t.Cleanup(func() { os.Stderr = oldErr })

	resetFlags(t, "--no-cache", "--stats", p)
	_ = run()
	_ = w.Close() // 关掉写端让读端拿到 EOF；失败不影响断言
	got, _ := io.ReadAll(r)

	if strings.Contains(string(got), "预估") {
		t.Errorf("默认采样不该打预估:\n%s", got)
	}
}

// 小模型上 --sample-limit 0 不预警：按 8 B/元素算不到 1 GiB 时只是噪音。
//
// 没有这条的话，那道大小阈值就是一段**没人看着的守卫** ——
// 把它删掉（永远预警）不会有任何测试变红。
func TestRun_小模型不采样也不预警(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "small.gguf")
	// 1e6 元素 × 8 B = 8 MiB，远低于阈值
	if err := os.WriteFile(p, buildMinimalGGUF("small", 0, 1_000_000, nil), 0o644); err != nil {
		t.Fatal(err)
	}

	r, w, _ := os.Pipe()
	oldErr := os.Stderr
	os.Stderr = w
	t.Cleanup(func() { os.Stderr = oldErr })

	resetFlags(t, "--no-cache", "--stats", "--sample-limit", "0", p)
	_ = run()
	_ = w.Close() // 关掉写端让读端拿到 EOF；失败不影响断言
	got, _ := io.ReadAll(r)

	if strings.Contains(string(got), "预估") {
		t.Errorf("小模型不该打预估（那是噪音）:\n%s", got)
	}
}

// 非终端下 `scan` 要走人类可读的非交互输出，而不是报"交互界面需要终端"。
//
// 实测原来会报错 —— 于是管道/日志里完全用不了，而那段输出（含孤儿 blob
// 与未完成下载的提示，约 60 行）已经写好、只被测试覆盖。
func TestRun_scan在非终端下给人类可读输出(t *testing.T) {
	r, w, _ := os.Pipe()
	oldOut := os.Stdout
	os.Stdout = w
	t.Cleanup(func() { os.Stdout = oldOut })

	resetFlags(t, "scan")
	err := run()
	_ = w.Close()
	got, _ := io.ReadAll(r)

	if err != nil {
		t.Fatalf("非终端下 scan 报错了: %v", err)
	}
	if !strings.Contains(string(got), "模型") {
		t.Errorf("没有人类可读输出:\n%s", got)
	}
}

// 非终端下 --json scan 仍然是 JSON（不能因为分流改动而受影响）。
func TestRun_scan非终端下json仍是json(t *testing.T) {
	r, w, _ := os.Pipe()
	oldOut := os.Stdout
	os.Stdout = w
	t.Cleanup(func() { os.Stdout = oldOut })

	resetFlags(t, "--json", "scan")
	err := run()
	_ = w.Close()
	got, _ := io.ReadAll(r)

	if err != nil {
		t.Fatalf("--json scan 报错了: %v", err)
	}
	var v map[string]any
	if err := json.Unmarshal(got, &v); err != nil {
		t.Fatalf("不是合法 JSON: %v\n%s", err, got)
	}
}
