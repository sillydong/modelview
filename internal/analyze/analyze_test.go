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

// 浮点张量必须得到三档模拟，且顺序是 Q8_0 / Q6_K / Q4_K。
func TestAnalyze_浮点张量做三档模拟(t *testing.T) {
	data := make([]byte, 128)
	for i := range 32 {
		putF32(data[i*4:], float32(i)/32-0.5)
	}
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{32}, "data_offsets": []int64{0, 128}},
	}, data)

	m := stModel(t, p, dataStart, 32, "w")
	if _, err := Analyze(context.Background(), m, Options{NoCache: true}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	sims := m.Tensors[0].QuantSims
	if len(sims) != 3 {
		t.Fatalf("模拟档数 = %d, want 3（Q8_0/Q6_K/Q4_K）", len(sims))
	}
	want := []string{"Q8_0", "Q6_K", "Q4_K"}
	for i, w := range want {
		if sims[i].Target != w {
			t.Errorf("档位[%d] = %q, want %q", i, sims[i].Target, w)
		}
	}
	// 误差必须随压缩加重而单调增大 —— 这是模拟自洽的基本性质
	for i := 1; i < len(sims); i++ {
		if sims[i].MaxAbsErr <= sims[i-1].MaxAbsErr {
			t.Errorf("%s 的最大绝对误差 %v 应大于 %s 的 %v",
				sims[i].Target, sims[i].MaxAbsErr, sims[i-1].Target, sims[i-1].MaxAbsErr)
		}
		if sims[i].Compression <= sims[i-1].Compression {
			t.Errorf("%s 的压缩比 %v 应大于 %s 的 %v",
				sims[i].Target, sims[i].Compression, sims[i-1].Target, sims[i-1].Compression)
		}
	}
	// F32 源：Q8_0 的压缩比应是 32/8.5
	if want := 32 / model.DtypeQ8_0.BitsPerWeight(); sims[0].Compression != want {
		t.Errorf("Q8_0 压缩比 = %v, want %v（源位宽应取自张量类型 F32）",
			sims[0].Compression, want)
	}
	// 浮点张量不该有块级诊断
	if m.Tensors[0].Quant != nil {
		t.Error("浮点张量不该有块级诊断")
	}
}

// 已量化的张量不做模拟（它没有"压到某档"的问题），但要有块级诊断。
func TestAnalyze_已量化张量做块级诊断(t *testing.T) {
	// 手工拼一个 Q8_0 张量：两个块、各 32 个权重，d 分别是 0.0625 与 0.5
	blk := func(d uint16) []byte {
		b := make([]byte, 34)
		binary.LittleEndian.PutUint16(b, d)
		for i := range 32 {
			b[2+i] = 0x40 // 量化码 64
		}
		return b
	}
	data := append(blk(0x2C00), blk(0x3800)...) // fp16 0.0625 / 0.5

	p, dataStart := stFile(t, map[string]any{
		"q": map[string]any{"dtype": "Q8_0", "shape": []int64{64}, "data_offsets": []int64{0, 68}},
	}, data)

	m := stModel(t, p, dataStart, 64, "q")
	m.Tensors[0].Dtype = model.DtypeQ8_0
	m.Tensors[0].ByteSize = 68

	if _, err := Analyze(context.Background(), m, Options{NoCache: true}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	tn := m.Tensors[0]
	if len(tn.QuantSims) != 0 {
		t.Errorf("已量化张量不该有模拟结果，实际 %d 档", len(tn.QuantSims))
	}
	q := tn.Quant
	if q == nil {
		t.Fatal("已量化张量应有块级诊断")
	}
	if q.SubBlocks != 2 {
		t.Errorf("SubBlocks = %d, want 2", q.SubBlocks)
	}
	if q.Blocks != 2 {
		t.Errorf("Blocks = %d, want 2", q.Blocks)
	}
	if q.ScaleMin != 0.0625 || q.ScaleMax != 0.5 {
		t.Errorf("ScaleMin/Max = %v/%v, want 0.0625/0.5", q.ScaleMin, q.ScaleMax)
	}
	if q.Scheme != "Q8_0" {
		t.Errorf("Scheme = %q", q.Scheme)
	}
	if q.BitsPerWeight != model.DtypeQ8_0.BitsPerWeight() {
		t.Errorf("BitsPerWeight = %v, want %v", q.BitsPerWeight, model.DtypeQ8_0.BitsPerWeight())
	}
	// Q8_0 是 32 元素一块、一块一个子块 → 每块一个子块
	if q.BlockElems != 32 {
		t.Errorf("BlockElems = %d, want 32", q.BlockElems)
	}
}

// 浮点源走模拟、量化源走诊断 —— 两条路必须互斥。
//
// 同一次 Analyze 里两种张量都要处理正确，这是接入层最容易接错的地方：
// 把量化张量送进模拟（会拿量化码当浮点值算）或者反过来，
// 都不会报错，只会给出一个看似合理的错数字。
func TestAnalyze_模拟与诊断互斥(t *testing.T) {
	qblk := make([]byte, 34)
	binary.LittleEndian.PutUint16(qblk, 0x2C00)
	for i := range 32 {
		qblk[2+i] = 0x40
	}
	data := make([]byte, 128+34)
	for i := range 32 {
		putF32(data[i*4:], float32(i)/32-0.5)
	}
	copy(data[128:], qblk)

	p, dataStart := stFile(t, map[string]any{
		"f": map[string]any{"dtype": "F32", "shape": []int64{32}, "data_offsets": []int64{0, 128}},
		"q": map[string]any{"dtype": "Q8_0", "shape": []int64{32}, "data_offsets": []int64{128, 162}},
	}, data)

	fm := stModel(t, p, dataStart, 32, "f")
	qm := stModel(t, p, dataStart, 32, "q")
	qm.Tensors[0].Dtype = model.DtypeQ8_0
	qm.Tensors[0].ByteSize = 34
	qm.Tensors[0].Offset = dataStart + 128
	fm.Tensors = append(fm.Tensors, qm.Tensors[0])

	if _, err := Analyze(context.Background(), fm, Options{NoCache: true}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	for _, tn := range fm.Tensors {
		switch tn.Name {
		case "f":
			if len(tn.QuantSims) != 3 {
				t.Errorf("浮点张量应有 3 档模拟，实际 %d", len(tn.QuantSims))
			}
			if tn.Quant != nil {
				t.Error("浮点张量不该有块级诊断")
			}
		case "q":
			if len(tn.QuantSims) != 0 {
				t.Errorf("量化张量不该有模拟，实际 %d 档", len(tn.QuantSims))
			}
			if tn.Quant == nil {
				t.Error("量化张量应有块级诊断")
			}
		}
	}
}

// 现有统计行为不能被这次接入改动 —— 回归守卫。
//
// analyzeOne 里的解码循环被提成了 decodeSampled，这一步是重构已验证的代码，
// 统计数值必须一字不变。
func TestAnalyze_接入量化分析后统计不变(t *testing.T) {
	data := make([]byte, 256)
	for i := range 64 {
		putF32(data[i*4:], float32(i)/64-0.5)
	}
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{64}, "data_offsets": []int64{0, 256}},
	}, data)

	m := stModel(t, p, dataStart, 64, "w")
	if _, err := Analyze(context.Background(), m, Options{NoCache: true}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	s := m.Tensors[0].Stats
	if s == nil {
		t.Fatal("Stats 未填充")
	}
	if s.Count != 64 {
		t.Errorf("Count = %d, want 64", s.Count)
	}
	// 值域 [-0.5, 0.484375]，均值 0 - 1/128
	if s.Min != -0.5 {
		t.Errorf("Min = %v, want -0.5", s.Min)
	}
	if math.Abs(s.Max-0.484375) > 1e-9 {
		t.Errorf("Max = %v, want 0.484375", s.Max)
	}
	if math.Abs(s.Mean-(-1.0/128)) > 1e-9 {
		t.Errorf("Mean = %v, want %v", s.Mean, -1.0/128)
	}
	// 直方图必须覆盖全部 64 个元素
	var total int64
	for _, c := range s.Histogram {
		total += c
	}
	if total != 64 {
		t.Errorf("直方图总计 %d != 64", total)
	}
}

// 模拟的 Sampled 标记必须跟着统计走 —— 采样出来的误差不能冒充全量结论。
func TestAnalyze_模拟的采样标记(t *testing.T) {
	data := make([]byte, 400)
	for i := range 100 {
		putF32(data[i*4:], float32(i)/100-0.5)
	}
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{100}, "data_offsets": []int64{0, 400}},
	}, data)

	// 采样：模拟的 Sampled 必须是 true
	m := stModel(t, p, dataStart, 100, "w")
	if _, err := Analyze(context.Background(), m, Options{SampleLimit: 10, NoCache: true}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	if len(m.Tensors[0].QuantSims) != 3 {
		t.Fatalf("模拟档数 = %d", len(m.Tensors[0].QuantSims))
	}
	for _, s := range m.Tensors[0].QuantSims {
		if !s.Sampled {
			t.Errorf("%s 的 Sampled = false，但只采样了 10 个元素", s.Target)
		}
	}

	// 不采样：必须是 false
	m2 := stModel(t, p, dataStart, 100, "w")
	if _, err := Analyze(context.Background(), m2, Options{SampleLimit: -1, NoCache: true}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	for _, s := range m2.Tensors[0].QuantSims {
		if s.Sampled {
			t.Errorf("%s 的 Sampled = true，但读的是全量", s.Target)
		}
	}
}

// 整数张量不做量化模拟 —— 把 I32 压成 Q8_0 没有意义，
// 给出来的误差数字会被当成有效结论。
func TestAnalyze_整数张量不模拟(t *testing.T) {
	data := make([]byte, 64)
	for i := range 16 {
		binary.LittleEndian.PutUint32(data[i*4:], uint32(i*100))
	}
	p, dataStart := stFile(t, map[string]any{
		"idx": map[string]any{"dtype": "I32", "shape": []int64{16}, "data_offsets": []int64{0, 64}},
	}, data)

	m := stModel(t, p, dataStart, 16, "idx")
	m.Tensors[0].Dtype = model.DtypeI32

	if _, err := Analyze(context.Background(), m, Options{NoCache: true}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	tn := m.Tensors[0]
	if tn.Stats == nil {
		t.Fatal("整数张量仍然要有统计")
	}
	if len(tn.QuantSims) != 0 {
		t.Errorf("整数张量不该有量化模拟，实际 %d 档：%+v", len(tn.QuantSims), tn.QuantSims)
	}
	if tn.Quant != nil {
		t.Error("整数张量不该有块级诊断")
	}
}

// 块级诊断必须读遍全部分块 —— scale 分布是"哪个子块被压得最狠"的依据，
// 少读一段就会漏掉被压得最狠的那些。
//
// 造一个跨 maxReadChunk 的 Q8_0 张量：34 字节一块，用 250000 块
// （8.5 MB）确保至少两个分块。与 TestAnalyze_跨分块张量算得对 同一手法。
func TestAnalyze_块级诊断跨分块(t *testing.T) {
	const nBlocks = 250_000 // 250000 × 34 = 8,500,000 字节 > maxReadChunk
	const perBlock = 34
	const elemsPerBlock = 32

	blocksPerChunk := int64(maxReadChunk) / perBlock
	if int64(nBlocks) <= blocksPerChunk {
		t.Fatalf("nBlocks = %d 不超过一个分块（%d）—— 这条测试就失去意义了",
			nBlocks, blocksPerChunk)
	}

	blk := make([]byte, perBlock)
	binary.LittleEndian.PutUint16(blk, 0x2C00) // fp16 0.0625
	for i := range 32 {
		blk[2+i] = 0x40
	}
	data := make([]byte, nBlocks*perBlock)
	for i := range nBlocks {
		copy(data[i*perBlock:], blk)
	}

	p, dataStart := stFile(t, map[string]any{
		"q": map[string]any{
			"dtype":        "Q8_0",
			"shape":        []int64{nBlocks * elemsPerBlock},
			"data_offsets": []int64{0, nBlocks * perBlock},
		},
	}, data)

	m := stModel(t, p, dataStart, nBlocks*elemsPerBlock, "q")
	m.Tensors[0].Dtype = model.DtypeQ8_0
	m.Tensors[0].ByteSize = nBlocks * perBlock

	// 采样上限给 1，让统计那一段几乎不花时间 —— 本测试只关心块级诊断
	if _, err := Analyze(context.Background(), m, Options{SampleLimit: 1, NoCache: true}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	q := m.Tensors[0].Quant
	if q == nil {
		t.Fatal("应有块级诊断")
	}
	if q.SubBlocks != nBlocks {
		t.Errorf("SubBlocks = %d, want %d —— 只读了部分分块（每分块 %d 块）",
			q.SubBlocks, nBlocks, blocksPerChunk)
	}
	if q.Blocks != nBlocks {
		t.Errorf("Blocks = %d, want %d", q.Blocks, nBlocks)
	}
	if q.ScaleMin != 0.0625 || q.ScaleMax != 0.0625 {
		t.Errorf("ScaleMin/Max = %v/%v, want 0.0625/0.0625", q.ScaleMin, q.ScaleMax)
	}
}

// 缓存命中时量化分析必须与冷跑一致。
//
// 这条是**回归守卫**：早先 cacheRecord 只存 Stats，而跳过判断是
// `Stats != nil` —— 于是同一个文件第一次跑有 181 个模拟 / 253 个诊断，
// 第二次变成 0 / 0，且不报任何错。既有的缓存测试只断言 Stats 被复用，
// 对这件事零敏感。
//
// 两种张量都要覆盖：浮点走模拟、量化走诊断，两条路都可能被跳过。
func TestAnalyze_缓存命中不丢量化分析(t *testing.T) {
	// 一个 F32 张量 + 一个 Q8_0 张量
	qblk := make([]byte, 34)
	binary.LittleEndian.PutUint16(qblk, 0x2C00)
	for i := range 32 {
		qblk[2+i] = 0x40
	}
	data := make([]byte, 128+34)
	for i := range 32 {
		putF32(data[i*4:], float32(i)/32-0.5)
	}
	copy(data[128:], qblk)

	p, dataStart := stFile(t, map[string]any{
		"f": map[string]any{"dtype": "F32", "shape": []int64{32}, "data_offsets": []int64{0, 128}},
		"q": map[string]any{"dtype": "Q8_0", "shape": []int64{32}, "data_offsets": []int64{128, 162}},
	}, data)

	var firstSims, firstQuants int
	for run := 1; run <= 2; run++ {
		// 每次都用**全新的** model 对象 —— 复用同一个对象会连同内存里的
		// 结果一起带过来，那样测的是"对象还在"，不是"缓存里有"
		m := stModel(t, p, dataStart, 32, "f")
		m.Tensors[0].Offset = dataStart
		q := &model.Tensor{
			Name: "q", Dims: []int64{32}, Dtype: model.DtypeQ8_0,
			Offset: dataStart + 128, ByteSize: 34, ParamCount: 32,
		}
		m.Tensors = append(m.Tensors, q)

		if _, err := Analyze(context.Background(), m, Options{}); err != nil {
			t.Fatalf("第 %d 次 Analyze 失败: %v", run, err)
		}

		sims, quants := 0, 0
		for _, tn := range m.Tensors {
			sims += len(tn.QuantSims)
			if tn.Quant != nil {
				quants++
			}
		}
		if run == 1 {
			firstSims, firstQuants = sims, quants
			if sims != 3 {
				t.Fatalf("冷跑：模拟档数 = %d, want 3", sims)
			}
			if quants != 1 {
				t.Fatalf("冷跑：诊断数 = %d, want 1", quants)
			}
			// 确认缓存真的写出来了，否则第二次跑的是"没缓存"而不是"缓存命中"，
			// 这条测试会静默变成空转
			if !newCache(p, false).exists() {
				t.Fatal("第一次跑没有写出缓存 —— 第二次跑就不是缓存命中了")
			}
			continue
		}

		// 第二次必须一模一样
		if sims != firstSims {
			t.Errorf("热跑：模拟档数 = %d，冷跑是 %d —— 缓存命中后量化分析丢了",
				sims, firstSims)
		}
		if quants != firstQuants {
			t.Errorf("热跑：诊断数 = %d，冷跑是 %d —— 缓存命中后量化分析丢了",
				quants, firstQuants)
		}
	}
}

// 调用方只预填了 Stats 时，量化分析必须补上。
//
// 这是 needsWork 存在的另一半理由（不只是缓存）："已经算过"是针对
// **某一件具体的事**说的。把 Stats 当成整体已完成的信号，
// 会让任何只有统计、没有量化分析的状态永远补不上 —— 缓存只是其中一种。
func TestAnalyze_预填Stats后仍补上量化分析(t *testing.T) {
	data := make([]byte, 128)
	for i := range 32 {
		putF32(data[i*4:], float32(i)/32-0.5)
	}
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{32}, "data_offsets": []int64{0, 128}},
	}, data)

	m := stModel(t, p, dataStart, 32, "w")
	// 模拟"调用方已经有一份统计"（缓存或外部传入）
	m.Tensors[0].Stats = &model.Stats{Count: 32, Min: -0.5, Max: 0.484375}
	preFilled := m.Tensors[0].Stats

	if _, err := Analyze(context.Background(), m, Options{NoCache: true}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	tn := m.Tensors[0]
	if len(tn.QuantSims) != 3 {
		t.Errorf("预填 Stats 后模拟档数 = %d, want 3 —— 量化分析被当成「做完了」跳过了",
			len(tn.QuantSims))
	}
	// 预填的 Stats 不该被覆盖（它来自缓存，重算会白费工夫）
	if tn.Stats != preFilled {
		t.Error("预填的 Stats 被重新计算并覆盖了 —— 缓存的意义就没了")
	}
}

// 量化张量也一样：预填了 Stats 不等于块级诊断已经算过。
//
// 单独一条是因为 needsWork 对量化张量与浮点张量走的是**两个分支**，
// 只测一条会让另一个分支的守卫无人看着（变异验证时确认过）。
func TestAnalyze_预填Stats后仍补上块级诊断(t *testing.T) {
	qblk := make([]byte, 34)
	binary.LittleEndian.PutUint16(qblk, 0x2C00)
	for i := range 32 {
		qblk[2+i] = 0x40
	}
	p, dataStart := stFile(t, map[string]any{
		"q": map[string]any{"dtype": "Q8_0", "shape": []int64{32}, "data_offsets": []int64{0, 34}},
	}, qblk)

	m := stModel(t, p, dataStart, 32, "q")
	m.Tensors[0].Dtype = model.DtypeQ8_0
	m.Tensors[0].ByteSize = 34
	m.Tensors[0].Stats = &model.Stats{Count: 32, Min: -0.5, Max: 0.5}
	preFilled := m.Tensors[0].Stats

	if _, err := Analyze(context.Background(), m, Options{NoCache: true}); err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	tn := m.Tensors[0]
	if tn.Quant == nil {
		t.Fatal("预填 Stats 后块级诊断缺失 —— 量化分析被当成「做完了」跳过了")
	}
	if tn.Quant.SubBlocks != 1 {
		t.Errorf("SubBlocks = %d, want 1", tn.Quant.SubBlocks)
	}
	if tn.Stats != preFilled {
		t.Error("预填的 Stats 被重新计算并覆盖了")
	}
	// 量化张量不该有模拟
	if len(tn.QuantSims) != 0 {
		t.Errorf("量化张量不该有模拟，实际 %d 档", len(tn.QuantSims))
	}
}

// 缓存里真的存了量化结果，并且第二次跑**复用它**而不是重算。
//
// 判据是"结果来自缓存而不是新数据"：先跑一遍写下缓存，
// 然后把文件内容改掉、把 mtime 与大小都还原（缓存仍然匹配），
// 再跑一遍。用缓存的话看到的是旧数据的结果，重算的话是新数据的。
// 只断言"两次输出相同"是不够的 —— 重算也能得到相同的输出。
func TestAnalyze_缓存命中复用量化结果(t *testing.T) {
	mkData := func(first float32) []byte {
		data := make([]byte, 128)
		for i := range 32 {
			putF32(data[i*4:], float32(i)/32*first)
		}
		return data
	}
	p, dataStart := stFile(t, map[string]any{
		"w": map[string]any{"dtype": "F32", "shape": []int64{32}, "data_offsets": []int64{0, 128}},
	}, mkData(1))

	m := stModel(t, p, dataStart, 32, "w")
	if errs, err := Analyze(context.Background(), m, Options{}); err != nil || len(errs) != 0 {
		t.Fatalf("Analyze 失败: %v / %v", err, errs)
	}
	want := m.Tensors[0].QuantSims
	if len(want) != 3 {
		t.Fatalf("首次模拟档数 = %d, want 3", len(want))
	}
	st, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if !newCache(p, false).exists() {
		t.Fatal("没有写出缓存 —— 这条测试就失去意义了")
	}

	// 只改**数据区**，保留头部。整文件重写会把 safetensors 头部一起截掉，
	// 那时第二个模型读到的是坏文件、分析直接报错 —— 而现象
	//（QuantSims 为空）与"缓存没存量化结果"一模一样
	orig, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	copy(orig[int64(len(orig))-int64(len(mkData(3))):], mkData(3))
	if err := os.WriteFile(p, orig, 0o644); err != nil {
		t.Fatal(err)
	}
	// 还原 mtime，让缓存仍然匹配（大小本来就一样）
	if err := os.Chtimes(p, st.ModTime(), st.ModTime()); err != nil {
		t.Fatal(err)
	}
	// 确认缓存真的还会命中 —— 不命中时第二次是重算，测的就不是缓存了
	if !newCache(p, false).matches(&cacheRecord{
		SchemaVersion: schemaVersion,
		FileSize:      st.Size(),
		FileMTime:     st.ModTime().UnixNano(),
		SampleLimit:   DefaultSampleLimit,
	}, DefaultSampleLimit) {
		t.Fatal("还原 mtime 后缓存仍然不命中（文件系统的时间戳精度不够？）—— " +
			"这条测试会变成「重算也得到同样结果」的空转")
	}

	fresh := stModel(t, p, dataStart, 32, "w")
	errs, err := Analyze(context.Background(), fresh, Options{})
	if err != nil {
		t.Fatalf("Analyze 失败: %v", err)
	}
	if len(errs) != 0 {
		// 漏断言 errs 会让"读失败"伪装成"缓存里没有量化结果"
		t.Fatalf("有张量失败: %v", errs)
	}
	got := fresh.Tensors[0].QuantSims
	if len(got) != 3 {
		t.Fatalf("热跑模拟档数 = %d, want 3 —— 缓存里没存量化结果", len(got))
	}
	// 缓存命中 → 拿到的是**旧数据**的结果
	if got[0].SNRDB != want[0].SNRDB || got[0].MaxAbsErr != want[0].MaxAbsErr {
		t.Errorf("热跑拿到的是新数据的结果（SNR %v vs %v）—— 说明重算了而不是用缓存",
			got[0].SNRDB, want[0].SNRDB)
	}
}
