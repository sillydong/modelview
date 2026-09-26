package pytorch

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

func TestParseStorageClass(t *testing.T) {
	tests := []struct {
		in   string
		want model.Dtype
	}{
		// 点号分隔：本包的 GLOBAL 解析产出这种形式
		{"torch.FloatStorage", model.DtypeF32},
		{"torch.HalfStorage", model.DtypeF16},
		// 空格分隔：pickletools 的显示形式
		{"torch FloatStorage", model.DtypeF32},
		{"torch HalfStorage", model.DtypeF16},
		// 多级模块名
		{"torch.storage.DoubleStorage", model.DtypeF64},
		{"torch.BFloat16Storage", model.DtypeBF16},
		// 裸类名
		{"FloatStorage", model.DtypeF32},
	}
	for _, tt := range tests {
		got, err := parseStorageClass(tt.in)
		if err != nil {
			t.Errorf("parseStorageClass(%q) 出错: %v", tt.in, err)
			continue
		}
		if got != tt.want {
			t.Errorf("parseStorageClass(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestParseStorageClass_未知类返回可识别错误(t *testing.T) {
	_, err := parseStorageClass("torch WeirdStorage")
	var e ErrUnknownStorage
	if !errors.As(err, &e) {
		t.Fatalf("err = %v, 期望 ErrUnknownStorage", err)
	}
	if e.Class != "WeirdStorage" {
		t.Errorf("Class = %q", e.Class)
	}
}

// mkTensorCall 构造一条与真实文件同形的 _rebuild_tensor_v2 调用。
func mkTensorCall(storageKey string, numel int64, offset int64, shape, stride []int64) reduceCall {
	toAny := func(v []int64) []any {
		out := make([]any, len(v))
		for i, x := range v {
			out[i] = x
		}
		return out
	}
	return reduceCall{
		Func: rebuildTensorFunc,
		Args: []any{
			persistentRef{Value: []any{
				"storage", globalRef{Name: "torch FloatStorage"}, storageKey, "cpu", numel,
			}},
			offset,
			toAny(shape),
			toAny(stride),
			false,
			reduceCall{Func: "collections.OrderedDict"},
		},
	}
}

func TestExtractTensors(t *testing.T) {
	// 顶层只有一个根（就是 torch.save 的对象），所以它的键不加任何前缀 ——
	// state_dict 的点分键名必须原样保留。
	name := "model.layers.1.block_sparse_moe.experts::down_proj"
	call := mkTensorCall("1", 1048576, 0, []int64{2, 1024, 512}, []int64{524288, 512, 1})
	stack := []any{map[string]any{name: call}}

	recs, err := extractTensors(stack)
	if err != nil {
		t.Fatalf("extractTensors 失败: %v", err)
	}
	if len(recs) != 1 {
		t.Fatalf("提取到 %d 条记录, want 1", len(recs))
	}
	r := recs[0]
	if r.Name != name {
		t.Errorf("Name = %q, want %q", r.Name, name)
	}
	if r.StorageKey != "1" {
		t.Errorf("StorageKey = %q, want 1", r.StorageKey)
	}
	if r.StorageNumel != 1048576 {
		t.Errorf("StorageNumel = %d, want 1048576", r.StorageNumel)
	}
	if !reflect.DeepEqual(r.Shape, []int64{2, 1024, 512}) {
		t.Errorf("Shape = %v", r.Shape)
	}
	if !reflect.DeepEqual(r.Stride, []int64{524288, 512, 1}) {
		t.Errorf("Stride = %v", r.Stride)
	}
	if r.StorageClass != "torch FloatStorage" {
		t.Errorf("StorageClass = %q", r.StorageClass)
	}
}

// 非张量的 REDUCE（如 collections.OrderedDict）不能被当成张量。
func TestExtractTensors_忽略非张量调用(t *testing.T) {
	stack := []any{map[string]any{
		"state": map[string]any{"hooks": reduceCall{Func: "collections.OrderedDict"}},
	}}
	recs, err := extractTensors(stack)
	if err != nil {
		t.Fatalf("extractTensors 失败: %v", err)
	}
	if len(recs) != 0 {
		t.Errorf("提取到 %d 条记录, want 0", len(recs))
	}
}

// 结果必须按名字排序，否则输出不稳定。
func TestExtractTensors_结果有序(t *testing.T) {
	stack := []any{map[string]any{
		"z.weight": mkTensorCall("0", 4, 0, []int64{4}, []int64{1}),
		"a.weight": mkTensorCall("1", 4, 0, []int64{4}, []int64{1}),
		"m.weight": mkTensorCall("2", 4, 0, []int64{4}, []int64{1}),
	}}
	recs, err := extractTensors(stack)
	if err != nil {
		t.Fatalf("extractTensors 失败: %v", err)
	}
	want := []string{"a.weight", "m.weight", "z.weight"}
	for i, w := range want {
		if recs[i].Name != w {
			t.Errorf("recs[%d].Name = %q, want %q", i, recs[i].Name, w)
		}
	}
}

// 同名键出现在不同容器下时必须各自保留 —— 这是真实文件暴露的缺陷。
//
// 训练检查点把优化器状态存在 optimizer.state.<参数下标> 下，
// 每个参数都重复使用同一组键名 step / exp_avg / exp_avg_sq。
// 实测 transformer-s-10m.pt：173 个张量只有 47 个唯一裸键名，
// 按裸键名去重会静默丢掉 126 个真实张量。
func TestExtractTensors_同名键在不同容器下不互相覆盖(t *testing.T) {
	perParam := func(key string) map[string]any {
		return map[string]any{
			"step":       mkTensorCall("0", 1, 0, []int64{1}, []int64{1}),
			"exp_avg":    mkTensorCall("1", 8, 0, []int64{8}, []int64{1}),
			"exp_avg_sq": mkTensorCall("2", 8, 0, []int64{8}, []int64{1}),
		}
	}
	stack := []any{map[string]any{
		"weight": mkTensorCall("3", 8, 0, []int64{8}, []int64{1}),
		"optimizer": map[string]any{
			"state": map[string]any{
				"0": perParam("0"),
				"1": perParam("1"),
				"2": perParam("2"),
			},
		},
	}}

	recs, err := extractTensors(stack)
	if err != nil {
		t.Fatalf("extractTensors 失败: %v", err)
	}
	// 1 个权重 + 3 个参数 × 3 个状态张量
	if len(recs) != 10 {
		names := make([]string, len(recs))
		for i, r := range recs {
			names[i] = r.Name
		}
		t.Fatalf("提取到 %d 条记录, want 10；实际 %v", len(recs), names)
	}

	for _, want := range []string{
		"weight",
		"optimizer.state.0.step",
		"optimizer.state.0.exp_avg",
		"optimizer.state.2.exp_avg_sq",
	} {
		if !slices.ContainsFunc(recs, func(r tensorRecord) bool { return r.Name == want }) {
			t.Errorf("缺少记录 %q", want)
		}
	}
}

func TestNumel(t *testing.T) {
	tests := []struct {
		shape []int64
		want  int64
	}{
		{[]int64{4}, 4},
		{[]int64{2, 1024, 512}, 1048576},
		{[]int64{}, 1},
		{[]int64{0, 5}, 0},
	}
	for _, tt := range tests {
		got, err := numel(tt.shape)
		if err != nil {
			t.Errorf("numel(%v) 出错: %v", tt.shape, err)
			continue
		}
		if got != tt.want {
			t.Errorf("numel(%v) = %d, want %d", tt.shape, got, tt.want)
		}
	}
	if _, err := numel([]int64{-1}); err == nil {
		t.Error("负维度应报错")
	}
}

func TestIsContiguous(t *testing.T) {
	tests := []struct {
		name          string
		shape, stride []int64
		want          bool
	}{
		{"连续二维", []int64{2, 3}, []int64{3, 1}, true},
		{"转置", []int64{2, 3}, []int64{1, 2}, false},
		{"一维", []int64{4}, []int64{1}, true},
		{"带广播维", []int64{1, 4}, []int64{4, 1}, true},
		{"维度不匹配", []int64{2, 3}, []int64{1}, false},
		{"三维连续", []int64{2, 1024, 512}, []int64{524288, 512, 1}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isContiguous(tt.shape, tt.stride); got != tt.want {
				t.Errorf("isContiguous(%v, %v) = %v, want %v",
					tt.shape, tt.stride, got, tt.want)
			}
		})
	}
}

// 形状连乘溢出必须报错：回绕成负数后 ParamCount 会被当成总参数打印出来。
func TestNumel_溢出报错(t *testing.T) {
	if _, err := numel([]int64{1 << 21, 1 << 21, 1 << 21}); err == nil {
		t.Error("2^63 应报溢出")
	}
	// 边界之内不能误报
	if n, err := numel([]int64{1 << 20, 1 << 20, 1 << 20}); err != nil || n != 1<<60 {
		t.Errorf("2^60 不该报错: n=%d err=%v", n, err)
	}
}

// 路径是拼出来的字符串，不同结构可能拼出同一个值：
// 扁平键 "a.b" 与嵌套 a→b 都得到 "a.b"。
// 只按路径去重会静默丢掉其中一个，且丢哪个取决于 map 遍历顺序。
func TestExtractTensors_路径别名冲突不丢张量(t *testing.T) {
	for range 5 { // map 遍历无序，多跑几次确认结果稳定
		stack := []any{map[string]any{
			"a":   map[string]any{"b": mkTensorCall("0", 8, 0, []int64{8}, []int64{1})},
			"a.b": mkTensorCall("1", 8, 0, []int64{8}, []int64{1}),
		}}
		recs, err := extractTensors(stack)
		if err != nil {
			t.Fatalf("extractTensors 失败: %v", err)
		}
		if len(recs) != 2 {
			names := make([]string, len(recs))
			for i, r := range recs {
				names[i] = r.Name
			}
			t.Fatalf("提取到 %d 条, want 2；实际 %v", len(recs), names)
		}
		// 两个都要留下，且名字必须唯一
		if recs[0].Name == recs[1].Name {
			t.Fatalf("名字重复: %q", recs[0].Name)
		}
	}
}

// 同一个张量被引用两次时必须只保留一条，否则张量数虚高。
func TestExtractTensors_同一张量重复引用被去重(t *testing.T) {
	call := mkTensorCall("0", 8, 0, []int64{8}, []int64{1})
	stack := []any{map[string]any{
		"x": call,
		"y": call, // 内容完全相同，只是路径不同
	}}
	// 路径不同 → 名字不同 → 这是两个名字指向同一份数据，应当都保留
	recs, err := extractTensors(stack)
	if err != nil {
		t.Fatalf("extractTensors 失败: %v", err)
	}
	if len(recs) != 2 {
		t.Errorf("提取到 %d 条, want 2", len(recs))
	}

	// 真正的重复：同一个 map 被两处引用，路径也相同
	inner := map[string]any{"w": call}
	recs, err = extractTensors([]any{map[string]any{"m": inner, "m2": inner}})
	if err != nil {
		t.Fatalf("extractTensors 失败: %v", err)
	}
	if len(recs) != 2 { // m.w 与 m2.w 路径不同
		t.Errorf("提取到 %d 条, want 2", len(recs))
	}
}
