package analyze

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

// stFile 按 safetensors 布局拼一个文件，返回路径与数据区起点。
//
// 数据区起点 = 8 + 头部长度，头部长度取决于张量描述符，所以必须现场算 ——
// 写死一个数字会在改头部内容时静默失效。
func stFile(t *testing.T, tensors map[string]any, data []byte) (path string, dataStart int64) {
	t.Helper()
	hdr, err := json.Marshal(tensors)
	if err != nil {
		t.Fatal(err)
	}
	var buf []byte
	buf = binary.LittleEndian.AppendUint64(buf, uint64(len(hdr)))
	buf = append(buf, hdr...)
	buf = append(buf, data...)
	p := filepath.Join(t.TempDir(), "x.safetensors")
	if err := os.WriteFile(p, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	return p, int64(8 + len(hdr))
}

// stModel 造一个 F32 张量的 model.Model，张量指向数据区起点。
func stModel(t *testing.T, path string, dataStart int64, elems int64, names ...string) *model.Model {
	t.Helper()
	st, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	m := &model.Model{
		Path: path, Format: model.FormatSafeTensors,
		DataStart: dataStart, FileSize: st.Size(),
	}
	for _, n := range names {
		m.Tensors = append(m.Tensors, &model.Tensor{
			Name: n, Dims: []int64{elems}, Dtype: model.DtypeF32,
			Offset: dataStart, ByteSize: elems * 4, ParamCount: elems,
		})
	}
	return m
}

func putF32(dst []byte, v float32) {
	binary.LittleEndian.PutUint32(dst, math.Float32bits(v))
}

func TestAnalyze_算出统计(t *testing.T) {
	data := make([]byte, 8)
	putF32(data[0:], 1)
	putF32(data[4:], 3)
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{0, 8}},
	}, data)

	m := stModel(t, p, dataStart, 2, "w")
	if _, err := Analyze(context.Background(), m, Options{}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	s := m.Tensors[0].Stats
	if s == nil {
		t.Fatal("Stats 未填充")
	}
	if s.Count != 2 || s.Min != 1 || s.Max != 3 {
		t.Errorf("Count/Min/Max = %d/%v/%v, want 2/1/3", s.Count, s.Min, s.Max)
	}
	if math.Abs(s.Mean-2) > 1e-6 {
		t.Errorf("Mean = %v, want 2", s.Mean)
	}
	if s.Sampled {
		t.Error("元素数没超上限，不该标为采样")
	}
	if len(s.Histogram) != histogramBuckets {
		t.Errorf("直方图桶数 = %d, want %d", len(s.Histogram), histogramBuckets)
	}
}

// 单个张量失败不能中断整体 —— 其余张量照常出统计。
func TestAnalyze_单张量失败不中断(t *testing.T) {
	data := make([]byte, 8)
	putF32(data[0:], 5)
	putF32(data[4:], 7)
	p, dataStart := stFile(t, map[string]any{
		"good": map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{0, 8}},
	}, data)

	m := stModel(t, p, dataStart, 2, "bad", "good")
	m.Tensors[0].NonContiguous = true // 只让第一个失败

	errs, err := Analyze(context.Background(), m, Options{})
	if err != nil {
		t.Fatalf("Analyze 不该整体失败: %v", err)
	}
	if _, ok := errs["bad"]; !ok {
		t.Errorf("非连续张量应记进 errs，实际 errs=%v", errs)
	}
	if m.Tensors[0].Stats != nil {
		t.Error("非连续张量不该有统计 —— 按线性顺序解出来的分布是错的")
	}
	if m.Tensors[1].Stats == nil {
		t.Error("正常的张量必须仍然算出统计")
	}
}

// 未收录的类型只让那一个张量失败，不影响其它。
func TestAnalyze_未收录类型只失败自己(t *testing.T) {
	data := make([]byte, 8)
	putF32(data[0:], 1)
	putF32(data[4:], 2)
	p, dataStart := stFile(t, map[string]any{
		"a": map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{0, 8}},
	}, data)

	m := stModel(t, p, dataStart, 2, "iq", "ok")
	m.Tensors[0].Dtype = model.DtypeIQ2XXS

	errs, err := Analyze(context.Background(), m, Options{})
	if err != nil {
		t.Fatalf("Analyze 不该整体失败: %v", err)
	}
	if len(errs) != 1 {
		t.Errorf("应只有 1 个张量失败，实际 %v", errs)
	}
	if errs["iq"] == nil {
		t.Error("未收录类型应记进 errs")
	}
	if m.Tensors[1].Stats == nil {
		t.Error("正常的张量必须仍然算出统计")
	}
}

// SizeUnknown 的报错必须说清是"字节数未知"，而不是笼统的"类型未收录"。
//
// 这条分支在当前代码路径下与"类型未收录"重叠（未知存储类同时会让
// BlockBytes 失败），但报错信息不同：用户看到"字节数未知"才知道
// 是存储类没认出来，而不是张量类型本身有问题。
func TestAnalyze_字节数未知的报错说清原因(t *testing.T) {
	data := make([]byte, 4)
	putF32(data, 1)
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{0, 4}},
	}, data)

	m := stModel(t, p, dataStart, 1, "w")
	m.Tensors[0].SizeUnknown = true

	errs, err := Analyze(context.Background(), m, Options{})
	if err != nil {
		t.Fatalf("Analyze 不该整体失败: %v", err)
	}
	got := errs["w"]
	if got == nil {
		t.Fatal("SizeUnknown 的张量应记进 errs")
	}
	if !strings.Contains(got.Error(), "字节数未知") {
		t.Errorf("错误应说明字节数未知，实际: %v", got)
	}
}

// 采样上限很小时必须标 Sampled，且样本量不超过上限。
func TestAnalyze_触发采样(t *testing.T) {
	data := make([]byte, 400)
	for i := range 100 {
		putF32(data[i*4:], float32(i))
	}
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{100}, "data_offsets": []int64{0, 400}},
	}, data)

	m := stModel(t, p, dataStart, 100, "w")
	if _, err := Analyze(context.Background(), m, Options{SampleLimit: 10}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	s := m.Tensors[0].Stats
	if !s.Sampled {
		t.Error("超过上限必须标 Sampled")
	}
	if s.Count != 10 {
		t.Errorf("Count = %d, want 10", s.Count)
	}
	// 等距采样覆盖整个区间，极值应当接近 0 与 99
	if s.Min > 5 || s.Max < 90 {
		t.Errorf("Min/Max = %v/%v —— 采样没有铺开", s.Min, s.Max)
	}
}

// 负的 SampleLimit 表示不采样（--sample-limit 0 的处理交给 CLI，
// 这里定义 -1 为"不限制"）。
func TestAnalyze_负数上限表示不采样(t *testing.T) {
	data := make([]byte, 400)
	for i := range 100 {
		putF32(data[i*4:], float32(i))
	}
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{100}, "data_offsets": []int64{0, 400}},
	}, data)

	m := stModel(t, p, dataStart, 100, "w")
	if _, err := Analyze(context.Background(), m, Options{SampleLimit: -1}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	s := m.Tensors[0].Stats
	if s.Sampled {
		t.Error("负数上限表示不采样")
	}
	if s.Count != 100 {
		t.Errorf("Count = %d, want 100", s.Count)
	}
}

// ctx 取消要能及时退出，且作为整体性错误返回（不是"某个张量失败"）。
func TestAnalyze_ctx取消(t *testing.T) {
	data := make([]byte, 400)
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{100}, "data_offsets": []int64{0, 400}},
	}, data)

	m := stModel(t, p, dataStart, 100, "w")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := Analyze(ctx, m, Options{})
	if err == nil {
		t.Fatal("ctx 已取消应返回错误")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("错误应是 context.Canceled，实际 %v", err)
	}
}

// 已有 Stats 的张量不该被重算（缓存命中时的行为）。
func TestAnalyze_已有统计则跳过(t *testing.T) {
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{1}, "data_offsets": []int64{0, 4}},
	}, make([]byte, 4))

	m := stModel(t, p, dataStart, 1, "w")
	m.Tensors[0].Stats = &model.Stats{Count: 999, Min: -1}

	if _, err := Analyze(context.Background(), m, Options{}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	if m.Tensors[0].Stats.Count != 999 {
		t.Errorf("已有统计被覆盖了: %+v", m.Tensors[0].Stats)
	}
}

// 缓存必须真的被 Analyze 采用，而不是"写了但没人读"。
//
// 判据用过写入的**特征值**：先往缓存里塞 Count=999，
// 再跑 Analyze —— 拿到的若是 999 就说明走的是缓存，不是重算。
func TestAnalyze_采用缓存(t *testing.T) {
	data := make([]byte, 8)
	putF32(data[0:], 1)
	putF32(data[4:], 3)
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{0, 8}},
	}, data)

	m := stModel(t, p, dataStart, 2, "w")

	// 先正常跑一遍写出缓存
	if _, err := Analyze(context.Background(), m, Options{}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	if m.Tensors[0].Stats.Count != 2 {
		t.Fatalf("首次统计 Count = %d, want 2", m.Tensors[0].Stats.Count)
	}

	// 手工把缓存里的 Count 改成一个特征值
	c := newCache(p, false)
	cp := c.path()
	raw, err := os.ReadFile(cp)
	if err != nil {
		t.Fatalf("读缓存失败（路径 %s）: %v", cp, err)
	}
	patched := strings.Replace(string(raw), `"count":2`, `"count":999`, 1)
	if patched == string(raw) {
		t.Fatalf("缓存里没找到 count 字段，内容: %s", raw)
	}
	if err := os.WriteFile(cp, []byte(patched), 0o644); err != nil {
		t.Fatal(err)
	}

	// 换一个全新的 model 对象再跑：应当直接采用缓存
	fresh := stModel(t, p, dataStart, 2, "w")
	if _, err := Analyze(context.Background(), fresh, Options{}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	if fresh.Tensors[0].Stats.Count != 999 {
		t.Errorf("Count = %d, want 999 —— 说明没有走缓存", fresh.Tensors[0].Stats.Count)
	}
}

// NoCache 时不读也不写缓存。
func TestAnalyze_NoCache不碰缓存(t *testing.T) {
	data := make([]byte, 8)
	putF32(data[0:], 1)
	putF32(data[4:], 3)
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{2}, "data_offsets": []int64{0, 8}},
	}, data)

	m := stModel(t, p, dataStart, 2, "w")
	if _, err := Analyze(context.Background(), m, Options{NoCache: true}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	if m.Tensors[0].Stats == nil || m.Tensors[0].Stats.Count != 2 {
		t.Errorf("NoCache 时仍应算出统计: %+v", m.Tensors[0].Stats)
	}
	if c := newCache(p, false); c.exists() {
		t.Errorf("NoCache 时不该写出缓存文件: %s", c.path())
	}
}

// 跨分块的张量必须算对 —— 分块路径此前只被真实文件覆盖，
// 而本地语料里没有 BOOL 张量，也没有超过 8 MiB 的非量化张量。
//
// 实测的缺陷：BOOL 解码不写 false 元素，而 analyze 的 decoded 缓冲
// 跨分块复用，于是第二块往后每个 false 都保留上一轮的值。
// 8,388,609 个元素、只有一个 0 的张量算出了 zero_ratio=0、min=1。
func TestAnalyze_跨分块张量算得对(t *testing.T) {
	const n = maxReadChunk + 1 // 刚超过一个分块
	data := make([]byte, n)
	for i := range n {
		data[i] = 1
	}
	data[n-1] = 0 // 唯一的一个 false，落在第二个分块里

	p, dataStart := stFile(t, map[string]any{
		"b": map[string]any{"dtype": "BOOL", "shape": []int64{n}, "data_offsets": []int64{0, n}},
	}, data)

	m := &model.Model{Path: p, Format: model.FormatSafeTensors, DataStart: dataStart}
	m.Tensors = append(m.Tensors, &model.Tensor{
		Name: "b", Dims: []int64{n}, Dtype: model.DtypeBool,
		Offset: dataStart, ByteSize: n, ParamCount: n,
	})

	errs, err := Analyze(context.Background(), m, Options{NoCache: true})
	if err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	if len(errs) != 0 {
		t.Fatalf("统计失败: %v", errs)
	}

	s := m.Tensors[0].Stats
	if s.Count != n {
		t.Fatalf("Count = %d, want %d", s.Count, n)
	}
	if s.Min != 0 {
		t.Errorf("Min = %v, want 0 —— 唯一那个 false 没被读到", s.Min)
	}
	if s.Max != 1 {
		t.Errorf("Max = %v, want 1", s.Max)
	}
	wantZero := 1.0 / float64(n)
	if e := math.Abs(s.ZeroRatio - wantZero); e > 1e-12 {
		t.Errorf("ZeroRatio = %v, want %v（偏差 %g）—— 跨分块时元素没被写满",
			s.ZeroRatio, wantZero, e)
	}
}

// 改了采样上限必须重扫 —— 统计结果取决于它，是缓存身份的一部分。
//
// 实测的缺陷：缓存记录里没存采样上限，先跑 --sample-limit 10
// 再按默认跑，拿到的仍是 count=10 的那份，用户改参数看不到任何变化。
func TestAnalyze_采样上限变化会重扫(t *testing.T) {
	data := make([]byte, 400)
	for i := range 100 {
		putF32(data[i*4:], float32(i))
	}
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{100}, "data_offsets": []int64{0, 400}},
	}, data)

	// 先用 10 跑一遍，写进缓存
	m := stModel(t, p, dataStart, 100, "w")
	if _, err := Analyze(context.Background(), m, Options{SampleLimit: 10}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	if m.Tensors[0].Stats.Count != 10 {
		t.Fatalf("首次 Count = %d, want 10", m.Tensors[0].Stats.Count)
	}

	// 再用"不采样"跑：必须重扫，而不是沿用缓存里的 10
	fresh := stModel(t, p, dataStart, 100, "w")
	if _, err := Analyze(context.Background(), fresh, Options{SampleLimit: -1}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	s := fresh.Tensors[0].Stats
	if s.Count != 100 {
		t.Errorf("Count = %d, want 100 —— 采样上限变了缓存却没失效", s.Count)
	}
	if s.Sampled {
		t.Error("不采样时 Sampled 不该为 true —— 读到的是上一次的缓存")
	}
}
