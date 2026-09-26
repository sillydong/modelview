package analyze

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

// rawFile 造一个裸数据文件，返回路径与内容。
func rawFile(t *testing.T, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "x.bin")
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// f32Bytes 把一串 float32 编成小端字节。
func f32Bytes(vs ...float32) []byte {
	out := make([]byte, 0, 4*len(vs))
	for _, v := range vs {
		out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
	}
	return out
}

// 造一个最小的 .pt：ZIP 里一个条目 model/data/0，内容是 data 里的 float32。
func ptFile(t *testing.T, data []byte, prefix string, keys ...string) string {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	if len(keys) == 0 {
		keys = []string{"0"}
	}
	for _, k := range keys {
		w, err := zw.Create(prefix + "/data/" + k)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(t.TempDir(), "x.pt")
	if err := os.WriteFile(p, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestFileSource_读张量字节(t *testing.T) {
	p := rawFile(t, f32Bytes(1.5, -2.5, 3.5))
	src, err := openSource(&model.Model{Path: p, Format: model.FormatSafeTensors})
	if err != nil {
		t.Fatalf("openSource 失败: %v", err)
	}
	defer src.Close() //nolint:errcheck // 测试清理

	tn := &model.Tensor{Name: "w", Dtype: model.DtypeF32, Offset: 4, ByteSize: 8}
	got, err := src.readRaw(tn, 0, 8)
	if err != nil {
		t.Fatalf("readRaw 失败: %v", err)
	}
	if v := math.Float32frombits(binary.LittleEndian.Uint32(got)); v != -2.5 {
		t.Errorf("偏移 4 处读到 %v, want -2.5", v)
	}

	// 带偏移的读取：从张量数据区再往后 4 字节
	got, err = src.readRaw(tn, 4, 4)
	if err != nil {
		t.Fatalf("readRaw 失败: %v", err)
	}
	if v := math.Float32frombits(binary.LittleEndian.Uint32(got)); v != 3.5 {
		t.Errorf("张量内偏移 4 处读到 %v, want 3.5", v)
	}
}

// 偏移未知的张量必须报错，不能从文件头开始读（那会读到完全无关的数据）。
func TestFileSource_偏移未知报错(t *testing.T) {
	p := rawFile(t, make([]byte, 16))
	src, err := openSource(&model.Model{Path: p, Format: model.FormatGGUF})
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close() //nolint:errcheck // 测试清理

	tn := &model.Tensor{Name: "w", Dtype: model.DtypeF32, OffsetUnknown: true}
	if _, err := src.readRaw(tn, 0, 4); err == nil {
		t.Fatal("偏移未知应报错")
	}
}

func TestZipSource_读张量字节(t *testing.T) {
	p := ptFile(t, f32Bytes(1.5, -2.5, 3.5), "model")
	src, err := openSource(&model.Model{
		Path: p, Format: model.FormatPyTorch, ArchivePrefix: "model",
	})
	if err != nil {
		t.Fatalf("openSource 失败: %v", err)
	}
	defer src.Close() //nolint:errcheck // 测试清理

	tn := &model.Tensor{
		Name: "w", Dtype: model.DtypeF32, ByteSize: 8, StorageKey: "0",
	}
	got, err := src.readRaw(tn, 0, 8)
	if err != nil {
		t.Fatalf("readRaw 失败: %v", err)
	}
	if v := math.Float32frombits(binary.LittleEndian.Uint32(got)); v != 1.5 {
		t.Errorf("读到 %v, want 1.5", v)
	}
}

// StorageOffset 的单位是**元素**不是字节 —— 差一个系数 4 是最容易犯的错。
func TestZipSource_存储偏移按元素计(t *testing.T) {
	p := ptFile(t, f32Bytes(1.5, -2.5, 3.5), "model")
	src, err := openSource(&model.Model{
		Path: p, Format: model.FormatPyTorch, ArchivePrefix: "model",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close() //nolint:errcheck // 测试清理

	// 第 2 个元素（下标 1）起读
	tn := &model.Tensor{
		Name: "w", Dtype: model.DtypeF32, ByteSize: 8,
		StorageKey: "0", StorageOffset: 1,
	}
	got, err := src.readRaw(tn, 0, 4)
	if err != nil {
		t.Fatalf("readRaw 失败: %v", err)
	}
	if v := math.Float32frombits(binary.LittleEndian.Uint32(got)); v != -2.5 {
		t.Errorf("读到 %v, want -2.5 —— StorageOffset 是元素数不是字节数", v)
	}
}

// 读越界必须报错，不能返回短读或越界切片。
func TestZipSource_越界报错(t *testing.T) {
	p := ptFile(t, f32Bytes(1.5), "model")
	src, err := openSource(&model.Model{
		Path: p, Format: model.FormatPyTorch, ArchivePrefix: "model",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close() //nolint:errcheck // 测试清理

	tn := &model.Tensor{Name: "w", Dtype: model.DtypeF32, StorageKey: "0"}
	if _, err := src.readRaw(tn, 0, 100); err == nil {
		t.Fatal("越界读取应报错")
	}
}

// 缺少归档前缀的 .pt 必须报错，而不是去猜一个前缀。
func TestZipSource_缺前缀报错(t *testing.T) {
	p := ptFile(t, f32Bytes(1.5), "model")
	_, err := openSource(&model.Model{Path: p, Format: model.FormatPyTorch})
	if err == nil {
		t.Fatal("缺归档前缀应报错")
	}
}

// 引用了不存在的存储块要报错，且错误里带上条目名便于排查。
func TestZipSource_存储块缺失报错(t *testing.T) {
	p := ptFile(t, f32Bytes(1.5), "model", "0")
	src, err := openSource(&model.Model{
		Path: p, Format: model.FormatPyTorch, ArchivePrefix: "model",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close() //nolint:errcheck // 测试清理

	tn := &model.Tensor{Name: "w", Dtype: model.DtypeF32, StorageKey: "9"}
	_, err = src.readRaw(tn, 0, 4)
	if err == nil {
		t.Fatal("存储块不存在应报错")
	}
	if want := "model/data/9"; !bytes.Contains([]byte(err.Error()), []byte(want)) {
		t.Errorf("错误信息应包含 %q，实际: %v", want, err)
	}
}

// 不支持的格式要明确报错。
func TestOpenSource_未知格式报错(t *testing.T) {
	if _, err := openSource(&model.Model{Path: "x", Format: model.FormatUnknown}); err == nil {
		t.Fatal("未知格式应报错")
	}
}

func TestOpenSource_文件不存在报错(t *testing.T) {
	if _, err := openSource(&model.Model{
		Path: filepath.Join(t.TempDir(), "missing"), Format: model.FormatGGUF,
	}); err == nil {
		t.Fatal("文件不存在应报错")
	}
}
