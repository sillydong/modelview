package gguf

import (
	"bytes"
	"math"
	"testing"
)

// parseKVBytes 跑一遍 KV 解析，返回键值对与结束偏移。
func parseKVBytes(t *testing.T, data []byte, kvCount uint64) ([]rawKV, int64) {
	t.Helper()
	r := newReader(bytes.NewReader(data))
	if err := r.skip(4); err != nil { // magic
		t.Fatalf("跳过 magic 失败: %v", err)
	}
	if _, err := r.u32(); err != nil { // version
		t.Fatalf("读 version 失败: %v", err)
	}
	if _, err := r.u64(); err != nil { // tensor count
		t.Fatalf("读 tensor count 失败: %v", err)
	}
	if _, err := r.u64(); err != nil { // kv count
		t.Fatalf("读 kv count 失败: %v", err)
	}
	kvs, err := readKVs(r, kvCount)
	if err != nil {
		t.Fatalf("readKVs 失败: %v", err)
	}
	return kvs, r.offset()
}

func TestReadKVs_标量类型(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 8)
	b.kv("k_u8", typeUint8).u8(255)
	b.kv("k_i8", typeInt8).i8(-128)
	b.kv("k_u16", typeUint16).u16(65535)
	b.kv("k_i16", typeInt16).i16(-32768)
	b.kv("k_u32", typeUint32).u32(4294967295)
	b.kv("k_i32", typeInt32).i32(-2147483648)
	b.kv("k_f32", typeFloat32).f32(0.5)
	b.kv("k_f64", typeFloat64).f64(3.14159)

	kvs, _ := parseKVBytes(t, b.bytes(), 8)

	if len(kvs) != 8 {
		t.Fatalf("KV 个数 = %d, want 8", len(kvs))
	}

	checks := []struct {
		idx  int
		name string
		got  any
		want any
	}{
		{0, "u8", kvs[0].Value, uint8(255)},
		{1, "i8", kvs[1].Value, int8(-128)},
		{2, "u16", kvs[2].Value, uint16(65535)},
		{3, "i16", kvs[3].Value, int16(-32768)},
		{4, "u32", kvs[4].Value, uint32(4294967295)},
		{5, "i32", kvs[5].Value, int32(-2147483648)},
		{6, "f32", kvs[6].Value, float32(0.5)},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %v (%T), want %v (%T)", c.name, c.got, c.got, c.want, c.want)
		}
	}
	if got, ok := kvs[7].Value.(float64); !ok || math.Abs(got-3.14159) > 1e-12 {
		t.Errorf("f64 = %v, want 3.14159", kvs[7].Value)
	}
}

// typeBool = 7 只占 1 字节，最容易在类型分派里被漏掉。
func TestReadKVs_bool与字符串(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 3)
	b.kv("b_true", typeBool).u8(1)
	b.kv("b_false", typeBool).u8(0)
	b.kv("s", typeString).str("qwen2")

	kvs, _ := parseKVBytes(t, b.bytes(), 3)

	if kvs[0].Value != true {
		t.Errorf("b_true = %v, want true", kvs[0].Value)
	}
	if kvs[1].Value != false {
		t.Errorf("b_false = %v, want false", kvs[1].Value)
	}
	if kvs[2].Value != "qwen2" {
		t.Errorf("s = %v, want qwen2", kvs[2].Value)
	}
}

func TestReadKVs_u64与i64(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 2)
	b.kv("big_u", typeUint64).u64(math.MaxUint64)
	b.kv("big_i", typeInt64).i64(math.MinInt64)

	kvs, _ := parseKVBytes(t, b.bytes(), 2)

	if kvs[0].Value != uint64(math.MaxUint64) {
		t.Errorf("big_u = %v, want %v", kvs[0].Value, uint64(math.MaxUint64))
	}
	if kvs[1].Value != int64(math.MinInt64) {
		t.Errorf("big_i = %v, want %v", kvs[1].Value, int64(math.MinInt64))
	}
}

func TestReadKVs_数组(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 3)

	// u32 数组 [1, 2, 3]
	b.kv("nums", typeArray).u32(typeUint32).u64(3).u32(1).u32(2).u32(3)
	// 字符串数组
	b.kv("strs", typeArray).u32(typeString).u64(2).str("a").str("b")
	// 空数组
	b.kv("empty", typeArray).u32(typeUint32).u64(0)

	kvs, _ := parseKVBytes(t, b.bytes(), 3)

	nums, ok := kvs[0].Value.(arrayValue)
	if !ok {
		t.Fatalf("nums 类型 = %T, want arrayValue", kvs[0].Value)
	}
	if nums.Len != 3 || len(nums.NumElems) != 3 {
		t.Fatalf("nums.Len = %d, NumElems = %d, want 3/3", nums.Len, len(nums.NumElems))
	}
	if nums.NumElems[0] != uint32(1) || nums.NumElems[2] != uint32(3) {
		t.Errorf("nums 内容 = %v", nums.NumElems)
	}

	strs, ok := kvs[1].Value.(arrayValue)
	if !ok {
		t.Fatalf("strs 类型 = %T", kvs[1].Value)
	}
	if len(strs.StrElems) != 2 || strs.StrElems[0] != "a" || strs.StrElems[1] != "b" {
		t.Errorf("strs = %v", strs.StrElems)
	}

	empty, ok := kvs[2].Value.(arrayValue)
	if !ok {
		t.Fatalf("empty 类型 = %T", kvs[2].Value)
	}
	if empty.Len != 0 || len(empty.NumElems) != 0 {
		t.Errorf("empty 应为空数组")
	}
}

// 数组必须完整消费。若只读前几个元素就跳过，
// 文件指针会错位，后续所有字段都读错。
func TestReadKVs_数组完整消费(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 2)
	// 100 个元素的数组，后面紧跟另一条 KV
	b.kv("big", typeArray).u32(typeUint32).u64(100)
	for i := 0; i < 100; i++ {
		b.u32(uint32(i))
	}
	b.kv("after", typeString).str("sentinel")

	kvs, _ := parseKVBytes(t, b.bytes(), 2)

	if len(kvs) != 2 {
		t.Fatalf("KV 个数 = %d, want 2", len(kvs))
	}
	big, ok := kvs[0].Value.(arrayValue)
	if !ok {
		t.Fatalf("big 类型 = %T", kvs[0].Value)
	}
	if big.Len != 100 || len(big.NumElems) != 100 {
		t.Fatalf("big.Len = %d, NumElems = %d, want 100/100", big.Len, len(big.NumElems))
	}
	// 关键断言：数组读完后指针位置正确，下一条 KV 能正常读出
	if kvs[1].Key != "after" || kvs[1].Value != "sentinel" {
		t.Errorf("第二条 KV = %q/%v, want after/sentinel（指针错位）", kvs[1].Key, kvs[1].Value)
	}
}

func TestReadKVs_嵌套数组(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 1)
	// 2 个子数组，每个含 2 个 u32
	b.kv("nested", typeArray).u32(typeArray).u64(2)
	b.u32(typeUint32).u64(2).u32(7).u32(8)
	b.u32(typeUint32).u64(2).u32(9).u32(10)

	kvs, _ := parseKVBytes(t, b.bytes(), 1)

	nested, ok := kvs[0].Value.(arrayValue)
	if !ok {
		t.Fatalf("类型 = %T", kvs[0].Value)
	}
	if len(nested.Nested) != 2 {
		t.Fatalf("子数组个数 = %d, want 2", len(nested.Nested))
	}
	if nested.Nested[0].NumElems[0] != uint32(7) {
		t.Errorf("nested[0][0] = %v, want 7", nested.Nested[0].NumElems[0])
	}
	if nested.Nested[1].NumElems[1] != uint32(10) {
		t.Errorf("nested[1][1] = %v, want 10", nested.Nested[1].NumElems[1])
	}
}

func TestReadKVs_未知类型报错(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 1)
	b.kv("weird", 999).u64(0)

	r := newReader(bytes.NewReader(b.bytes()))
	mustSkipHeader(t, r)
	if _, err := readKVs(r, 1); err == nil {
		t.Fatal("未知类型应报错，实际为 nil")
	}
}

// 截断的文件必须报错，不能返回部分结果。
func TestReadKVs_截断报错(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 1)
	b.kv("truncated", typeString).u64(100) // 声称长度 100，实际没有数据

	r := newReader(bytes.NewReader(b.bytes()))
	mustSkipHeader(t, r)
	if _, err := readKVs(r, 1); err == nil {
		t.Fatal("截断数据应报错，实际为 nil")
	}
}

// 声称超大长度的字符串必须被拒绝，而不是尝试分配内存。
func TestReadKVs_超长字符串报错(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 1)
	b.kv("huge", typeString).u64(1 << 40) // 1 TiB

	r := newReader(bytes.NewReader(b.bytes()))
	mustSkipHeader(t, r)
	if _, err := readKVs(r, 1); err == nil {
		t.Fatal("超长字符串应报错，实际为 nil")
	}
}

// mustSkipHeader 跳过 GGUF 文件头（magic + version + 两个计数）。
func mustSkipHeader(t *testing.T, r *reader) {
	t.Helper()
	if err := r.skip(4); err != nil {
		t.Fatal(err)
	}
	if _, err := r.u32(); err != nil {
		t.Fatal(err)
	}
	if _, err := r.u64(); err != nil {
		t.Fatal(err)
	}
	if _, err := r.u64(); err != nil {
		t.Fatal(err)
	}
}
