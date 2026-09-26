package gguf

import (
	"bytes"
	"errors"
	"math"
	"runtime"
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
	_, err := readKVs(r, 1)
	// 断言具体哨兵：只断言 err != nil 的话，把阈值从 64 MiB 改成 128 MiB
	// 测试照样通过 —— 它只钉住了"存在某个 ≤ 1 TiB 的阈值"。
	if !errors.Is(err, ErrStringTooLong) {
		t.Fatalf("err = %v, 期望 ErrStringTooLong", err)
	}
}

// 阈值本身必须被钉住：稍微超一点也要拒绝。
func TestReadKVs_字符串阈值精确(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 1)
	b.kv("just_over", typeString).u64(maxStringLen + 1)

	r := newReader(bytes.NewReader(b.bytes()))
	mustSkipHeader(t, r)
	_, err := readKVs(r, 1)
	if !errors.Is(err, ErrStringTooLong) {
		t.Fatalf("长度 %d 应被拒绝，err = %v", maxStringLen+1, err)
	}
}

// 声称超大条数的元数据必须被拒绝，而不是尝试分配内存。
func TestReadKVs_条数超限报错(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 0)

	r := newReader(bytes.NewReader(b.bytes()))
	mustSkipHeader(t, r)
	_, err := readKVs(r, maxArrayLen+1)
	// 必须断言的是"哪个错误"：去掉上限检查后，循环会在读第一个条目时
	// 因为数据耗尽而报错 —— 那样的测试会因为错误的理由通过。
	if !errors.Is(err, ErrTooManyEntries) {
		t.Fatalf("err = %v, 期望 ErrTooManyEntries", err)
	}
}

// 声称超大长度的数组必须被拒绝，而不是尝试分配内存。
func TestReadKVs_数组长度超限报错(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 1)
	// 元素类型 u32，长度声称 1 亿 + 1
	b.kv("huge", typeArray).u32(typeUint32).u64(maxArrayLen + 1)

	r := newReader(bytes.NewReader(b.bytes()))
	mustSkipHeader(t, r)
	_, err := readKVs(r, 1)
	if !errors.Is(err, ErrTooManyEntries) {
		t.Fatalf("err = %v, 期望 ErrTooManyEntries（而不是读到一半才失败）", err)
	}
}

// 声称超大数量必须在**分配内存之前**被拦住。
//
// 这是一条防回归测试：曾经的写法是先做上限检查、再 make([]T, 0, n) 预分配，
// 结果一个几十字节的损坏文件声称 1 亿条，校验通过后立刻申请数 GiB 内存，
// 直到读第 0 个元素才因 EOF 失败 —— 守卫只拦下了报错，没拦下它声称要防的分配。
func TestReadKVs_超大声明不触发巨额分配(t *testing.T) {
	const limit = 64 << 20 // 64 MiB，远超正常头部，远低于 1e8 元素的开销

	tests := []struct {
		name  string
		build func() []byte
	}{
		{
			name: "元数据条数声称 1 亿",
			build: func() []byte {
				b := newBuilder()
				b.header(3, 0, 0)
				return b.bytes()
			},
		},
		{
			name: "嵌套数组长度声称 1 亿（arrayValue 元素最大）",
			build: func() []byte {
				b := newBuilder()
				b.header(3, 0, 1)
				b.kv("nested", typeArray).u32(typeArray).u64(maxArrayLen)
				return b.bytes()
			},
		},
		{
			name: "字符串数组长度声称 1 亿",
			build: func() []byte {
				b := newBuilder()
				b.header(3, 0, 1)
				b.kv("strs", typeArray).u32(typeString).u64(maxArrayLen)
				return b.bytes()
			},
		},
		{
			name: "数值数组长度声称 1 亿（any 元素 16 字节）",
			build: func() []byte {
				b := newBuilder()
				b.header(3, 0, 1)
				b.kv("nums", typeArray).u32(typeUint32).u64(maxArrayLen)
				return b.bytes()
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := tt.build()

			var before, after runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&before)

			r := newReader(bytes.NewReader(data))
			mustSkipHeader(t, r)
			// 无论报不报错，都不该发生巨额分配。
			_, _ = readKVs(r, maxArrayLen)

			runtime.ReadMemStats(&after)
			allocated := after.TotalAlloc - before.TotalAlloc
			if allocated > limit {
				t.Errorf("声明 %d 条时分配了 %.1f MiB（上限 %.0f MiB）——"+
					"预分配必须在取 min(声明值, preallocCap) 之后再 make",
					uint64(maxArrayLen), float64(allocated)/(1<<20), float64(limit)/(1<<20))
			}
		})
	}
}

// 声称超大个数的张量必须被拒绝。
func TestReadTensorInfos_个数超限报错(t *testing.T) {
	b := newBuilder()
	b.header(3, 0, 0)

	r := newReader(bytes.NewReader(b.bytes()))
	mustSkipHeader(t, r)
	_, err := readTensorInfos(r, maxArrayLen+1)
	if !errors.Is(err, ErrTooManyEntries) {
		t.Fatalf("err = %v, 期望 ErrTooManyEntries", err)
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
