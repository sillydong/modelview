package model

import (
	"reflect"
	"testing"
)

func TestFindTiedGroups(t *testing.T) {
	m := &Model{Tensors: []*Tensor{
		{Name: "b.weight", StorageKey: "0", ByteSize: 10},
		{Name: "a.weight", StorageKey: "0", ByteSize: 10}, // 与 b 共享
		{Name: "solo.weight", StorageKey: "1", ByteSize: 20},
		{Name: "c.weight", StorageKey: "2", ByteSize: 30},
		{Name: "d.weight", StorageKey: "2", ByteSize: 30}, // 与 c 共享
		{Name: "no_storage", StorageKey: ""},
	}}

	got := m.FindTiedGroups()
	want := [][]string{{"a.weight", "b.weight"}, {"c.weight", "d.weight"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("FindTiedGroups() = %v, want %v", got, want)
	}
}

func TestFindTiedGroups_无共享时返回空(t *testing.T) {
	m := &Model{Tensors: []*Tensor{
		{Name: "a", StorageKey: "0"},
		{Name: "b", StorageKey: "1"},
	}}
	if got := m.FindTiedGroups(); got != nil {
		t.Errorf("FindTiedGroups() = %v, want nil", got)
	}
}

func TestTensorBytes(t *testing.T) {
	m := &Model{Tensors: []*Tensor{
		{ByteSize: 10}, {ByteSize: 20}, {ByteSize: 0},
	}}
	if got := m.TensorBytes(); got != 30 {
		t.Errorf("TensorBytes() = %d, want 30", got)
	}
}

// 共享存储时，按存储块去重后的字节数必须小于张量字节之和 ——
// 这正是界面要解释的那个差值的来源。
//
// 只断言 TensorBytes，以及"去重后的值"是**从张量算出来**的：
// 直接写 `StorageBytes: 100` 再断言它等于 100，是在断言测试自己写的字面量，
// 永远不可能失败。
func TestStorageBytes与TensorBytes可不同(t *testing.T) {
	m := &Model{
		Tensors: []*Tensor{
			{Name: "tied_a", StorageKey: "0", ByteSize: 100},
			{Name: "tied_b", StorageKey: "0", ByteSize: 100},
		},
	}
	if got := m.TensorBytes(); got != 200 {
		t.Errorf("TensorBytes() = %d, want 200", got)
	}

	groups := m.FindTiedGroups()
	if len(groups) != 1 || len(groups[0]) != 2 {
		t.Fatalf("FindTiedGroups() = %v, want 一组两个", groups)
	}

	// 按存储块去重：每个 StorageKey 只计一次
	byKey := map[string]int64{}
	for _, tn := range m.Tensors {
		byKey[tn.StorageKey] = tn.ByteSize
	}
	var unique int64
	for _, b := range byKey {
		unique += b
	}
	if unique != 100 {
		t.Errorf("去重后 = %d, want 100", unique)
	}
	if unique >= m.TensorBytes() {
		t.Errorf("去重后(%d) 必须小于张量字节和(%d)，否则界面无从解释差异",
			unique, m.TensorBytes())
	}
}

// NonContiguous 与 OffsetUnknown 是两个独立的事实，不能互相顶替：
// 前者说「数据不是线性排的」，后者说「不知道数据在哪」。
func TestTensor_两个标记互相独立(t *testing.T) {
	tn := &Tensor{Name: "t", NonContiguous: true, OffsetUnknown: true}
	if !tn.NonContiguous || !tn.OffsetUnknown {
		t.Error("两个字段应能同时为 true")
	}
}

// Stats 必须能表达「采样过」—— 把样本统计量当成全量是误导。
func TestStats_采样标记与直方图(t *testing.T) {
	s := &Stats{Count: 100, Sampled: true, Histogram: make([]int64, 64)}
	if !s.Sampled {
		t.Error("Sampled 应为 true")
	}
	if len(s.Histogram) != 64 {
		t.Errorf("直方图应为 64 桶，实际 %d", len(s.Histogram))
	}
}

// 归档前缀只有 PyTorch 会填，其它格式保持空串。
func TestModel_归档前缀(t *testing.T) {
	m := &Model{Format: FormatGGUF}
	if m.ArchivePrefix != "" {
		t.Errorf("GGUF 不该有归档前缀，实际 %q", m.ArchivePrefix)
	}
	p := &Model{Format: FormatPyTorch, ArchivePrefix: "model"}
	if p.ArchivePrefix != "model" {
		t.Errorf("ArchivePrefix = %q", p.ArchivePrefix)
	}
}

// DisplayName 优先用发现层给的名字，没有才退回文件名。
//
// 这条是"同一屏里同一个模型有两个名字"的回归：原来标题栏用
// ModelView 的 displayName（优先用模型库给的名字），而张量列表用
// baseName(m.Path) —— ollama 的路径是 blobs/sha256-5ee4f07c…，
// 于是点进张量列表后标题从 qwen2.5:3b 变成了那串哈希。
func TestModel_DisplayName(t *testing.T) {
	blob := "/Users/x/.ollama/models/blobs/sha256-5ee4f07cdb9beadbbb293e85803c569b"
	tests := []struct {
		name string
		m    *Model
		want string
	}{
		{"有名字时用名字", &Model{Path: blob, Name: "qwen2.5:3b"}, "qwen2.5:3b"},
		{"没名字时退回文件名", &Model{Path: blob}, "sha256-5ee4f07cdb9beadbbb293e85803c569b"},
		{"没有目录的路径", &Model{Path: "model.gguf"}, "model.gguf"},
		{"空路径", &Model{}, ""},
		{"nil 接收者不 panic", nil, ""},
	}
	for _, tt := range tests {
		if got := tt.m.DisplayName(); got != tt.want {
			t.Errorf("%s: DisplayName() = %q, want %q", tt.name, got, tt.want)
		}
	}
}
