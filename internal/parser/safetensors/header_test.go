package safetensors

import (
	"errors"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

func TestParseDtype(t *testing.T) {
	tests := []struct {
		in   string
		want model.Dtype
	}{
		{"F32", model.DtypeF32},
		{"F16", model.DtypeF16},
		{"BF16", model.DtypeBF16},
		{"F64", model.DtypeF64},
		{"I8", model.DtypeI8},
		{"U8", model.DtypeU8},
		{"BOOL", model.DtypeBool},
		{"F8_E4M3", model.DtypeF8E4M3},
	}
	for _, tt := range tests {
		got, err := parseDtype(tt.in)
		if err != nil {
			t.Errorf("parseDtype(%q) 出错: %v", tt.in, err)
			continue
		}
		if got != tt.want {
			t.Errorf("parseDtype(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestParseDtype_未知码返回可识别错误(t *testing.T) {
	_, err := parseDtype("NOPE")
	var e ErrUnknownDtype
	if !errors.As(err, &e) {
		t.Fatalf("err = %v, 期望 ErrUnknownDtype", err)
	}
	if e.Code != "NOPE" {
		t.Errorf("Code = %q, want NOPE", e.Code)
	}
}

func TestParseHeader(t *testing.T) {
	// 真实文件的头部形状：名字 → {dtype, shape, data_offsets}
	raw := []byte(`{
		"down": {"dtype":"F32","shape":[1024,32],"data_offsets":[0,131072]},
		"gate": {"dtype":"F32","shape":[32,1024],"data_offsets":[131072,262144]},
		"up":   {"dtype":"F32","shape":[32,1024],"data_offsets":[262144,393216]}
	}`)

	hdr, err := parseHeader(raw)
	if err != nil {
		t.Fatalf("parseHeader 失败: %v", err)
	}
	if len(hdr.Tensors) != 3 {
		t.Fatalf("张量数 = %d, want 3", len(hdr.Tensors))
	}
	gate, ok := hdr.Tensors["gate"]
	if !ok {
		t.Fatal("缺少 gate")
	}
	// tensorEntry.Dtype 是 JSON 里的原始字符串，尚未映射到 model.Dtype
	if gate.Dtype != "F32" {
		t.Errorf("gate.Dtype = %q, want F32", gate.Dtype)
	}
	if len(gate.Shape) != 2 || gate.Shape[0] != 32 || gate.Shape[1] != 1024 {
		t.Errorf("gate.Shape = %v, want [32 1024]", gate.Shape)
	}
	if gate.DataOffsets[0] != 131072 || gate.DataOffsets[1] != 262144 {
		t.Errorf("gate.DataOffsets = %v", gate.DataOffsets)
	}
	// Names 必须有序 —— map 遍历无序，不排序会让输出不稳定
	want := []string{"down", "gate", "up"}
	for i, n := range want {
		if hdr.Names[i] != n {
			t.Errorf("Names[%d] = %q, want %q", i, hdr.Names[i], n)
		}
	}
}

func TestParseHeader_metadata被跳过(t *testing.T) {
	raw := []byte(`{
		"__metadata__": {"format": "pt"},
		"w": {"dtype":"F32","shape":[4],"data_offsets":[0,16]}
	}`)
	hdr, err := parseHeader(raw)
	if err != nil {
		t.Fatalf("parseHeader 失败: %v", err)
	}
	if len(hdr.Tensors) != 1 {
		t.Fatalf("张量数 = %d, want 1（__metadata__ 不是张量）", len(hdr.Tensors))
	}
	if hdr.Metadata["format"] != "pt" {
		t.Errorf("Metadata = %v, want format=pt", hdr.Metadata)
	}
}

// 无 __metadata__ 的头部必须正常工作 —— 本地 37 个文件全部如此。
func TestParseHeader_无metadata(t *testing.T) {
	raw := []byte(`{"w": {"dtype":"F32","shape":[4],"data_offsets":[0,16]}}`)
	hdr, err := parseHeader(raw)
	if err != nil {
		t.Fatalf("parseHeader 失败: %v", err)
	}
	if hdr.Metadata != nil {
		t.Errorf("Metadata = %v, want nil", hdr.Metadata)
	}
}

func TestParseHeader_非对象报错(t *testing.T) {
	if _, err := parseHeader([]byte(`[1,2,3]`)); err == nil {
		t.Fatal("顶层非对象应报错")
	}
}

func TestParseHeader_截断JSON报错(t *testing.T) {
	if _, err := parseHeader([]byte(`{"w": {"dtype"`)); err == nil {
		t.Fatal("截断的 JSON 应报错")
	}
}

func TestParseHeader_缺dtype报错(t *testing.T) {
	raw := []byte(`{"w": {"shape":[4],"data_offsets":[0,16]}}`)
	if _, err := parseHeader(raw); err == nil {
		t.Fatal("缺 dtype 应报错")
	}
}

func TestParseHeader_dataoffsets倒序报错(t *testing.T) {
	raw := []byte(`{"w": {"dtype":"F32","shape":[4],"data_offsets":[16,0]}}`)
	if _, err := parseHeader(raw); err == nil {
		t.Fatal("data_offsets 倒序应报错")
	}
}
