package main

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// modelview <dir> 必须列出目录里的模型，而不是报 "is a directory"。
//
// spec §3 明列了这一条，而实测原来会得到
// 「读取文件头: read /tmp/x: is a directory」—— 用户没有任何替代入口
// （--json scan 只认那十几个写死的路径）。
func TestRun_目录参数列出模型(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"a.gguf", "b.gguf"} {
		if err := os.WriteFile(filepath.Join(dir, n),
			buildMinimalGGUF(n, 0, 4, nil), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// 无关文件必须被后缀筛掉
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	r, w, _ := os.Pipe()
	oldOut := os.Stdout
	os.Stdout = w
	t.Cleanup(func() { os.Stdout = oldOut })

	resetFlags(t, "--json", dir)
	err := run()
	_ = w.Close()
	raw, _ := io.ReadAll(r)
	if err != nil {
		t.Fatalf("目录参数报错了: %v", err)
	}

	var got []map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("输出不是 JSON 数组: %v\n%s", err, raw)
	}
	if len(got) != 2 {
		t.Errorf("列了 %d 个条目，want 2（a.gguf / b.gguf）", len(got))
	}
	for _, it := range got {
		if strings.HasSuffix(it["path"].(string), "notes.txt") {
			t.Error("notes.txt 不该出现在结果里")
		}
	}
}

// 目录不存在时给出清晰报错，而不是"这个目录里没有模型"。
func TestRun_目录不存在(t *testing.T) {
	resetFlags(t, "--json", "/nonexistent/dir/xyz")
	if err := run(); err == nil {
		t.Error("不存在的目录应当报错，而不是说'没有模型'")
	}
}

// 传的是文件时仍走单文件分析（目录分支不能把文件也吞掉）。
func TestRun_文件参数仍走单文件(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(p, buildMinimalGGUF("m", 0, 4, nil), 0o644); err != nil {
		t.Fatal(err)
	}

	r, w, _ := os.Pipe()
	oldOut := os.Stdout
	os.Stdout = w
	t.Cleanup(func() { os.Stdout = oldOut })

	resetFlags(t, "--json", p)
	err := run()
	_ = w.Close()
	raw, _ := io.ReadAll(r)
	if err != nil {
		t.Fatalf("单文件报错了: %v", err)
	}
	var v map[string]any
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatalf("单文件输出应当是对象而不是数组: %v\n%s", err, raw)
	}
	if v["format"] != "GGUF" {
		t.Errorf("format = %v，看起来走了目录分支", v["format"])
	}
}

// 空目录是**正常结果**，不是失败 —— 要说"没有发现模型文件"。
//
// 第一版实现拿 `items == nil` 当失败判据，而空目录的 items 也是 nil，
// 于是打出"扫描失败"，用户会以为路径写错了。CLI 实测发现的。
func TestRun_空目录不算失败(t *testing.T) {
	dir := t.TempDir()

	r, w, _ := os.Pipe()
	oldOut := os.Stdout
	os.Stdout = w
	t.Cleanup(func() { os.Stdout = oldOut })

	resetFlags(t, dir)
	err := run()
	_ = w.Close()
	got, _ := io.ReadAll(r)

	if err != nil {
		t.Fatalf("空目录不该报错: %v", err)
	}
	if !strings.Contains(string(got), "没有发现") {
		t.Errorf("空目录应当说清楚是「没有发现模型」:\n%s", got)
	}
}

// 空目录的 --json 必须是 []，不能是 null（与 metadata 那条同一个道理）。
func TestRun_空目录json是空数组(t *testing.T) {
	dir := t.TempDir()

	r, w, _ := os.Pipe()
	oldOut := os.Stdout
	os.Stdout = w
	t.Cleanup(func() { os.Stdout = oldOut })

	resetFlags(t, "--json", dir)
	if err := run(); err != nil {
		_ = w.Close()
		t.Fatalf("空目录不该报错: %v", err)
	}
	_ = w.Close()
	got, _ := io.ReadAll(r)

	if strings.TrimSpace(string(got)) == "null" {
		t.Error("空目录的 --json 输出了 null，应当是 []")
	}
	var v []map[string]any
	if err := json.Unmarshal(got, &v); err != nil {
		t.Fatalf("不是 JSON 数组: %v\n%s", err, got)
	}
	if len(v) != 0 {
		t.Errorf("列了 %d 个条目，want 0", len(v))
	}
}
