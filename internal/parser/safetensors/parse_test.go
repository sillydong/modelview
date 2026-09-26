package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

// buildFile 按 safetensors 布局手工拼一个文件。
func buildFile(t *testing.T, tensors map[string]any, data []byte) []byte {
	t.Helper()
	hdr, err := json.Marshal(tensors)
	if err != nil {
		t.Fatal(err)
	}
	var buf []byte
	buf = binary.LittleEndian.AppendUint64(buf, uint64(len(hdr)))
	buf = append(buf, hdr...)
	buf = append(buf, data...)
	return buf
}

func writeFile(t *testing.T, name string, b []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestParse_最小文件(t *testing.T) {
	b := buildFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{4}, "data_offsets": []int64{0, 16}},
		"b": map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{16, 24}},
	}, make([]byte, 24))

	m, err := Parse(writeFile(t, "min.safetensors", b))
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}

	if m.Format != model.FormatSafeTensors {
		t.Errorf("Format = %q", m.Format)
	}
	if len(m.Tensors) != 2 {
		t.Fatalf("张量数 = %d, want 2", len(m.Tensors))
	}
	// 名字必须按字典序稳定输出 —— map 遍历无序，不排序的话测试会闪。
	if m.Tensors[0].Name != "b" || m.Tensors[1].Name != "w" {
		t.Errorf("张量顺序 = %q, %q；want b, w", m.Tensors[0].Name, m.Tensors[1].Name)
	}
	if got := m.Tensors[1].ParamCount; got != 4 {
		t.Errorf("w 元素数 = %d, want 4", got)
	}
	if got := m.Tensors[1].ByteSize; got != 16 {
		t.Errorf("w 字节数 = %d, want 16", got)
	}
	if m.HeaderBytes <= 0 {
		t.Error("HeaderBytes 未记录")
	}
	if m.DataStart != 8+m.HeaderBytes {
		t.Errorf("DataStart = %d, want %d（8 + 头部长度）", m.DataStart, 8+m.HeaderBytes)
	}
	if got := m.StorageBytes; got != 24 {
		t.Errorf("StorageBytes = %d, want 24", got)
	}
}

// data_offsets 是相对数据区的，绝对偏移必须加上 8 + 头部长度。
func TestParse_偏移是绝对的(t *testing.T) {
	data := make([]byte, 12)
	binary.LittleEndian.PutUint32(data[4:], 42)
	b := buildFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{3}, "data_offsets": []int64{0, 12}},
	}, data)

	p := writeFile(t, "x.safetensors", b)
	m, err := Parse(p)
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}
	tn := m.Tensors[0]

	f, err := os.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	//nolint:errcheck // 只读文件，Close 失败不影响测试
	defer f.Close()

	buf := make([]byte, 4)
	if _, err := f.ReadAt(buf, tn.Offset+4); err != nil {
		t.Fatalf("按偏移读取失败: %v", err)
	}
	if got := binary.LittleEndian.Uint32(buf); got != 42 {
		t.Errorf("偏移处读出 %d, want 42", got)
	}
}

func TestParse_未知dtype致命(t *testing.T) {
	b := buildFile(t, map[string]any{
		"w": map[string]any{"dtype": "NOPE", "shape": []int64{4}, "data_offsets": []int64{0, 16}},
	}, make([]byte, 16))
	_, err := Parse(writeFile(t, "x.safetensors", b))
	var e ErrUnknownDtype
	if !errors.As(err, &e) {
		t.Fatalf("err = %v, 期望 ErrUnknownDtype", err)
	}
}

// data_offsets 跨度与形状不符时必须报错 —— 否则偏移算出来是错的却静默通过。
func TestParse_跨度与形状不符报错(t *testing.T) {
	b := buildFile(t, map[string]any{
		// 4 个 F32 应该是 16 字节，声称 32
		"w": map[string]any{"dtype": "F32", "shape": []int64{4}, "data_offsets": []int64{0, 32}},
	}, make([]byte, 32))
	if _, err := Parse(writeFile(t, "x.safetensors", b)); err == nil {
		t.Fatal("跨度与形状不符应报错")
	}
}

func TestParse_头部长度超限报错(t *testing.T) {
	var buf []byte
	buf = binary.LittleEndian.AppendUint64(buf, maxHeaderLen+1)
	buf = append(buf, '{', '}')
	_, err := Parse(writeFile(t, "x.safetensors", buf))
	if err == nil {
		t.Fatal("头部长度超限应报错")
	}
	// 必须断言**是哪一个守卫**报的错。
	// 只判 "有错误" 是空转的：这个文件只有 10 字节，就算去掉 maxHeaderLen
	// 上限检查，"头部长度超过文件剩余大小" 那条也会报错，测试照样通过。
	if !strings.Contains(err.Error(), "超出合理范围") {
		t.Fatalf("错误应来自头部长度上限守卫，实际: %v", err)
	}
}

func TestParse_头部长度为零报错(t *testing.T) {
	var buf []byte
	buf = binary.LittleEndian.AppendUint64(buf, 0)
	_, err := Parse(writeFile(t, "x.safetensors", buf))
	if err == nil {
		t.Fatal("头部长度为 0 应报错")
	}
	// 必须断言是**长度守卫**报的错：去掉 headerLen == 0 之后，
	// 空头部会让 JSON 解析失败并报另一条错，测试照样通过。
	if !strings.Contains(err.Error(), "超出合理范围") {
		t.Fatalf("错误应来自头部长度守卫，实际: %v", err)
	}
}

func TestParse_文件截断报错(t *testing.T) {
	b := buildFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{4}, "data_offsets": []int64{0, 16}},
	}, make([]byte, 4)) // 声称 16 字节数据，实际只有 4
	if _, err := Parse(writeFile(t, "x.safetensors", b)); err == nil {
		t.Fatal("数据区不足应报错")
	}
}

func TestParse_文件不存在报错(t *testing.T) {
	if _, err := Parse(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("不存在的文件应报错")
	}
}

func TestParse_负维度报错(t *testing.T) {
	b := buildFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{-1}, "data_offsets": []int64{0, 16}},
	}, make([]byte, 16))
	_, err := Parse(writeFile(t, "x.safetensors", b))
	if err == nil {
		t.Fatal("负维度应报错")
	}
	// 必须断言是**负维度守卫**报的错：去掉它之后，跨度校验会因为
	// -1 × 4 = -4 ≠ 16 而报另一条错，测试照样通过。
	if !strings.Contains(err.Error(), "负维度") {
		t.Fatalf("错误应来自负维度守卫，实际: %v", err)
	}
}

func TestParse_零张量不报错(t *testing.T) {
	b := buildFile(t, map[string]any{}, nil)
	m, err := Parse(writeFile(t, "x.safetensors", b))
	if err != nil {
		t.Fatalf("空张量集合不应报错: %v", err)
	}
	if len(m.Tensors) != 0 {
		t.Errorf("张量数 = %d, want 0", len(m.Tensors))
	}
}

// data_offsets 不能为负：负值算出的绝对偏移会落进 JSON 头部内部。
func TestParse_负dataoffsets报错(t *testing.T) {
	b := buildFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{4}, "data_offsets": []int64{-16, 0}},
	}, make([]byte, 16))
	_, err := Parse(writeFile(t, "x.safetensors", b))
	if err == nil {
		t.Fatal("负 data_offsets 应报错")
	}
}

// 两个张量占用同一段字节时必须报错，否则头部与数据区对不上却静默通过。
func TestParse_数据区重叠报错(t *testing.T) {
	b := buildFile(t, map[string]any{
		"a": map[string]any{"dtype": "F32", "shape": []int64{4}, "data_offsets": []int64{0, 16}},
		"b": map[string]any{"dtype": "F32", "shape": []int64{4}, "data_offsets": []int64{0, 16}},
	}, make([]byte, 16))
	_, err := Parse(writeFile(t, "x.safetensors", b))
	if err == nil {
		t.Fatal("数据区重叠应报错")
	}
	if !strings.Contains(err.Error(), "重叠") {
		t.Fatalf("错误应来自重叠检测，实际: %v", err)
	}
}

// 形状连乘溢出必须报错，不能回绕成 0 后恰好满足跨度校验。
func TestParse_形状溢出报错(t *testing.T) {
	b := buildFile(t, map[string]any{
		"w": map[string]any{
			"dtype": "F32", "shape": []int64{1 << 32, 1 << 32},
			"data_offsets": []int64{0, 0},
		},
	}, nil)
	_, err := Parse(writeFile(t, "x.safetensors", b))
	if err == nil {
		t.Fatal("形状乘积溢出应报错")
	}
}

// 反向门禁：dtypeByCode 里能映射出的每个类型都必须能取到每元素字节数。
//
// parse.go 在取不到字节数时会直接报错（fail-closed）。如果某个 safetensors
// 合法的 dtype 映射到了量化类型或没有字节数的类型，那个文件就会**解析失败**
// 而不是给出结果 —— 这条测试让这种矛盾在加映射时就暴露，而不是等到用户手里。
func TestDtypeByCode_全部可取字节数(t *testing.T) {
	for code, d := range dtypeByCode {
		if _, ok := d.ByteSize(); !ok {
			t.Errorf("dtype %q 映射到 %s，但它没有每元素字节数；该类型的文件会解析失败",
				code, d)
		}
	}
}

// 正向门禁：规范里的 dtype 不能漏。
// 漏一个不会报错，只会让该类型的文件报"未知 dtype"，属于静默能力缺失。
func TestDtypeByCode_规范取值齐全(t *testing.T) {
	// safetensors 规范定义的完整取值列表
	want := []string{
		"BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3",
		"I16", "U16", "F16", "BF16",
		"I32", "U32", "F32",
		"I64", "U64", "F64",
	}
	for _, code := range want {
		if _, err := parseDtype(code); err != nil {
			t.Errorf("规范里的 dtype %q 未收录: %v", code, err)
		}
	}
	if len(dtypeByCode) != len(want) {
		t.Errorf("dtypeByCode 有 %d 项，规范是 %d 项 —— 多出来的要确认是否真的存在",
			len(dtypeByCode), len(want))
	}
}
