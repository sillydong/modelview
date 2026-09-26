package pytorch

import (
	"archive/zip"
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

// buildPT 手工拼一个最小的 .pt：一个 2 元素的 F32 张量。
//
// 归档前缀由参数决定，用来验证前缀是**动态发现**而不是写死的。
func buildPT(t *testing.T, prefix string) []byte {
	t.Helper()
	return buildPTWith(t, prefix, "little", minimalPkl())
}

// ptSpec 描述要拼出的 .pt 的可变部分。
type ptSpec struct {
	prefix     string
	byteOrder  string
	storageCls string   // 存储类名，如 FloatStorage / WeirdStorage
	storageKey string   // data.pkl 里引用的存储块键
	dataKeys   []string // ZIP 里实际存在的 <前缀>/data/<N>；为空则用 storageKey
	pkl        []byte   // 非空则直接用这份 pickle，忽略上面几项
}

// buildPTWith 与 buildPT 相同，但可以指定 byteorder 与 data.pkl 的内容。
func buildPTWith(t *testing.T, prefix, byteOrder string, pkl []byte) []byte {
	t.Helper()
	return buildPTSpec(t, ptSpec{prefix: prefix, byteOrder: byteOrder, pkl: pkl})
}

// buildPTSpec 按描述拼出一个 .pt。
func buildPTSpec(t *testing.T, spec ptSpec) []byte {
	t.Helper()

	pkl := spec.pkl
	if pkl == nil {
		cls := spec.storageCls
		if cls == "" {
			cls = "FloatStorage"
		}
		key := spec.storageKey
		if key == "" {
			key = "0"
		}
		pkl = minimalPklSpec(cls, key)
	}
	order := spec.byteOrder
	if order == "" {
		order = "little"
	}
	keys := spec.dataKeys
	if keys == nil {
		keys = []string{spec.storageKey}
		if spec.storageKey == "" {
			keys = []string{"0"}
		}
	}

	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	add := func(name string, data []byte) {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	add(spec.prefix+"/data.pkl", pkl)
	add(spec.prefix+"/.format_version", []byte("1"))
	add(spec.prefix+"/byteorder", []byte(order))
	for _, k := range keys {
		add(spec.prefix+"/data/"+k, make([]byte, 8)) // 2 个 float32
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// minimalPkl 返回一个最小的 state dict pickle：一个 2 元素的 F32 张量。
//
// 抽出来是给"大端拒绝"用的 —— 那个测试要的是一份**能解析成功**的 pickle，
// 这样唯一的失败原因才只剩 byteorder。空 pickle 会因为"没解析出张量"
// 而报错，让测试因为别的原因通过。
func minimalPkl() []byte { return minimalPklSpec("FloatStorage", "0") }

// minimalPklSpec 与 minimalPkl 相同，但存储类名与存储块键可指定。
func minimalPklSpec(storageCls, storageKey string) []byte {
	return minimalPklStrided(storageCls, storageKey, []int64{2}, []int64{1})
}

// minimalPklStrided 与 minimalPklSpec 相同，但形状与步长可指定。
//
// 存在的理由：本地 2601 个真实 .pt 张量**全部是连续的**，
// 非连续布局（转置、切片视图）一个都没有。造一个转置张量
// （shape 2×2、stride 1,2）才能验证 NonContiguous 的接线是通的 ——
// 否则把那个表达式的值写死成 false 也不会有测试发现。
func minimalPklStrided(storageCls, storageKey string, shape, stride []int64) []byte {
	p := newPB()
	p.u8(opEMPTY_DICT).input(0) // 顶层 dict
	p.u8(opMARK)
	p.str("state").input(1)
	p.u8(opEMPTY_DICT).input(2)
	p.u8(opMARK)
	p.str("w.weight").input(3)
	// _rebuild_tensor_v2
	p.global("torch._utils", "_rebuild_tensor_v2").input(4)
	p.u8(opMARK) // args mark
	p.u8(opMARK) // persistent id mark
	p.str("storage")
	p.global("torch", storageCls)
	p.str(storageKey)
	p.str("cpu")
	numel := int64(1)
	for _, d := range shape {
		numel *= d
	}
	p.u8(opBININT1).u8(uint8(numel))
	p.u8(opTUPLE).input(5)
	p.u8(opBINPERSID).input(6) // storage
	p.u8(opBININT1).u8(0)      // storage_offset
	p.u8(opMARK)
	for _, d := range shape {
		p.u8(opBININT1).u8(uint8(d))
	}
	p.u8(opTUPLE).input(7) // size
	p.u8(opMARK)
	for _, d := range stride {
		p.u8(opBININT1).u8(uint8(d))
	}
	p.u8(opTUPLE) // stride
	p.u8(opNEWFALSE)
	p.global("collections", "OrderedDict").u8(opEMPTY_TUPLE).u8(opREDUCE)
	p.u8(opTUPLE)    // args 元组
	p.u8(opREDUCE)   // 调用 _rebuild_tensor_v2
	p.u8(opSETITEMS) // 写进 state 字典
	p.u8(opSETITEMS) // 写进顶层字典
	return p.done()
}

func write(t *testing.T, name string, b []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestParse_最小文件(t *testing.T) {
	// 文件名与归档前缀刻意不同
	p := write(t, "file.pt", buildPT(t, "weird-prefix"))

	m, err := Parse(p)
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}
	if m.Format != model.FormatPyTorch {
		t.Errorf("Format = %q", m.Format)
	}
	if len(m.Tensors) != 1 {
		t.Fatalf("张量数 = %d, want 1", len(m.Tensors))
	}
	tn := m.Tensors[0]
	// 名字含容器路径：这个最小文件的顶层键是 state，张量键是 w.weight
	if tn.Name != "state.w.weight" {
		t.Errorf("Name = %q, want state.w.weight", tn.Name)
	}
	if tn.Dtype != model.DtypeF32 {
		t.Errorf("Dtype = %q, want F32", tn.Dtype)
	}
	if tn.ParamCount != 2 {
		t.Errorf("ParamCount = %d, want 2", tn.ParamCount)
	}
	if tn.ByteSize != 8 {
		t.Errorf("ByteSize = %d, want 8", tn.ByteSize)
	}
	if tn.StorageKey != "0" {
		t.Errorf("StorageKey = %q, want 0", tn.StorageKey)
	}
	if m.StorageBytes != 8 {
		t.Errorf("StorageBytes = %d, want 8", m.StorageBytes)
	}
	if len(m.Warnings) != 0 {
		t.Errorf("正常文件不该有告警: %v", m.Warnings)
	}

	// ZIP 容器没有单一的数据区起点。model.Model 的约定是"0 表示无此概念"，
	// 不能另发明一个 -1 —— 两个哨兵值会让消费方把 -1 当成有效偏移。
	if m.DataStart != 0 {
		t.Errorf("DataStart = %d, want 0（该格式无此概念）", m.DataStart)
	}
	// 张量在 ZIP 条目里的绝对偏移不可得，必须显式标注，
	// 否则 JSON 里的 "offset": 0 会被读成"从文件第 0 字节开始"。
	if !tn.OffsetUnknown {
		t.Error("OffsetUnknown 应为 true —— .pt 的张量偏移拿不到")
	}
}

// 归档前缀必须动态发现 —— 实测 best-500k.pt 的前缀是 ln-c2-a0-500k。
func TestParse_归档前缀动态发现(t *testing.T) {
	for _, prefix := range []string{"model", "cell", "ln-c2-a0-500k", "任意中文前缀"} {
		t.Run(prefix, func(t *testing.T) {
			p := write(t, "x.pt", buildPT(t, prefix))
			m, err := Parse(p)
			if err != nil {
				t.Fatalf("Parse 失败: %v", err)
			}
			if len(m.Tensors) != 1 {
				t.Fatalf("张量数 = %d, want 1", len(m.Tensors))
			}
		})
	}
}

func TestParse_无dataPkl报错(t *testing.T) {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, err := zw.Create("something/else.txt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("hi")); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := Parse(write(t, "x.pt", buf.Bytes())); err == nil {
		t.Fatal("没有 data.pkl 应报错")
	}
}

func TestParse_大端拒绝(t *testing.T) {
	// 两个前提缺一不可，否则测的就不是大端守卫：
	//   1. ZIP 里要有 data.pkl —— 没有的话 findLayout 先返回，守卫执行不到；
	//   2. data.pkl 要能解析出张量 —— 空 pickle 会因为"没解析出张量"报错，
	//      测试照样绿。
	_, err := Parse(write(t, "x.pt", buildPTWith(t, "model", "big", minimalPkl())))
	if err == nil {
		t.Fatal("大端应报错")
	}
	// 必须断言是**大端守卫**报的错，否则换成别的错误也照样通过
	if !strings.Contains(err.Error(), "大端") {
		t.Fatalf("错误应来自大端守卫，实际: %v", err)
	}
}

func TestParse_文件不存在报错(t *testing.T) {
	if _, err := Parse(filepath.Join(t.TempDir(), "missing.pt")); err == nil {
		t.Fatal("不存在的文件应报错")
	}
}

func TestParse_非ZIP报错(t *testing.T) {
	if _, err := Parse(write(t, "x.pt", []byte("not a zip at all"))); err == nil {
		t.Fatal("非 ZIP 应报错")
	}
}

// 未收录的存储类只该让**字节数**不可信，参数个数必须照常计入。
//
// 留 0 会让 TotalParams() 少算这些张量，而摘要的头条数字正是 TotalParams()，
// 用户从"未收录的存储类"这句告警里看不出参数也被算错了。
func TestParse_未知存储类仍计参数(t *testing.T) {
	p := write(t, "x.pt", buildPTSpec(t, ptSpec{prefix: "model", storageCls: "WeirdStorage"}))
	m, err := Parse(p)
	if err != nil {
		t.Fatalf("未知存储类不该致命: %v", err)
	}
	if len(m.Tensors) != 1 {
		t.Fatalf("张量数 = %d, want 1", len(m.Tensors))
	}
	tn := m.Tensors[0]
	if tn.ParamCount != 2 {
		t.Errorf("ParamCount = %d, want 2 —— 它只依赖形状，与存储类无关", tn.ParamCount)
	}
	if got := m.TotalParams(); got != 2 {
		t.Errorf("TotalParams = %d, want 2", got)
	}
	if !tn.SizeUnknown {
		t.Error("SizeUnknown 应为 true —— 字节数确实不可信")
	}
	if !tn.OffsetUnknown {
		t.Error("OffsetUnknown 应为 true")
	}
	if tn.ByteSize != 0 {
		t.Errorf("ByteSize = %d, want 0", tn.ByteSize)
	}
	if len(m.Warnings) == 0 {
		t.Error("应有一条告警说明存储类未收录")
	}
}

// data.pkl 引用了归档里不存在的存储块时必须告警。
func TestParse_存储块缺失告警(t *testing.T) {
	p := write(t, "x.pt", buildPTSpec(t, ptSpec{
		prefix:     "model",
		storageKey: "9",           // data.pkl 引用 data/9
		dataKeys:   []string{"0"}, // 但 ZIP 里只有 data/0
	}))
	m, err := Parse(p)
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}
	if len(m.Warnings) == 0 {
		t.Fatal("存储块缺失应有告警")
	}
	found := false
	for _, w := range m.Warnings {
		if strings.Contains(w, "data/9") {
			found = true
		}
	}
	if !found {
		t.Errorf("告警应指出缺失的 data/9，实际: %v", m.Warnings)
	}
}

// .pt 必须记录归档前缀 —— 读张量数据要靠它拼出 <前缀>/data/<键>。
// 前缀不是固定的：实测 best-500k.pt 的前缀是 ln-c2-a0-500k。
func TestParse_记录归档前缀(t *testing.T) {
	for _, prefix := range []string{"model", "ln-c2-a0-500k", "任意中文前缀"} {
		t.Run(prefix, func(t *testing.T) {
			m, err := Parse(write(t, "x.pt", buildPT(t, prefix)))
			if err != nil {
				t.Fatalf("Parse 失败: %v", err)
			}
			if m.ArchivePrefix != prefix {
				t.Errorf("ArchivePrefix = %q, want %q", m.ArchivePrefix, prefix)
			}
		})
	}
}

// 连续布局的张量不能标成 NonContiguous —— 标错会让 analyze 白白拒绝统计。
func TestParse_连续张量不标记(t *testing.T) {
	m, err := Parse(write(t, "x.pt", buildPT(t, "model")))
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}
	for _, tn := range m.Tensors {
		if tn.NonContiguous {
			t.Errorf("张量 %s 被标为非连续，但 stride 是连续的", tn.Name)
		}
	}
}

// 真实文件同样：本地 2601 个张量全是连续的。
func TestParse_真实PT_都是连续张量(t *testing.T) {
	p := requireArtifact(t, "releases/v0.1/model.pt")
	m, err := Parse(p)
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}
	if m.ArchivePrefix != "model" {
		t.Errorf("ArchivePrefix = %q, want model", m.ArchivePrefix)
	}
	for _, tn := range m.Tensors {
		if tn.NonContiguous {
			t.Errorf("张量 %s 标为非连续，但实测本地文件全是连续的", tn.Name)
		}
	}
}

// 非连续布局（转置）必须被标记出来。
//
// 转置张量的字节数与形状都对，但元素在存储块里不是线性排列的 ——
// analyze 按线性顺序解码会得到顺序错乱但看着正常的分布，
// 所以必须能在解析阶段识别出来。
func TestParse_非连续张量被标记(t *testing.T) {
	// shape 2×2、stride (1,2)：这是转置，不是连续布局
	pkl := minimalPklStrided("FloatStorage", "0", []int64{2, 2}, []int64{1, 2})
	m, err := Parse(write(t, "x.pt", buildPTWith(t, "model", "little", pkl)))
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}
	if len(m.Tensors) != 1 {
		t.Fatalf("张量数 = %d, want 1", len(m.Tensors))
	}
	tn := m.Tensors[0]
	if !tn.NonContiguous {
		t.Errorf("转置张量（shape=%v stride=%v）应标为 NonContiguous",
			tn.Dims, []int64{1, 2})
	}
	// 形状与元素数仍是正确的 —— 非连续只影响元素顺序，不影响大小
	if tn.ParamCount != 4 {
		t.Errorf("ParamCount = %d, want 4", tn.ParamCount)
	}
}

// 对照组：连续的二维张量不该被标记。
func TestParse_连续二维张量不标记(t *testing.T) {
	pkl := minimalPklStrided("FloatStorage", "0", []int64{2, 2}, []int64{2, 1})
	m, err := Parse(write(t, "x.pt", buildPTWith(t, "model", "little", pkl)))
	if err != nil {
		t.Fatalf("Parse 失败: %v", err)
	}
	if m.Tensors[0].NonContiguous {
		t.Errorf("shape=[2 2] stride=[2 1] 是连续布局，不该被标记")
	}
}
