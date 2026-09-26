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
