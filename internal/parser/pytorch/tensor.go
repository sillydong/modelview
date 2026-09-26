package pytorch

import (
	"fmt"
	"math"
	"slices"
	"strconv"
)

// tensorRecord 是从 pickle 里提取出的一条张量记录。
type tensorRecord struct {
	Name string
	// StorageKey 对应 ZIP 里的 <前缀>/data/<StorageKey>
	StorageKey string
	// StorageNumel 是该存储块的**总**元素数，不是这个张量的元素数。
	// 多个张量可共享一个存储块（权重绑定）。
	StorageNumel int64
	// StorageOffset 是张量数据在存储块内的元素偏移
	StorageOffset int64
	StorageClass  string
	Shape         []int64
	Stride        []int64
}

// rebuildTensorFunc 是 PyTorch 用来重建张量的全局函数名。
const rebuildTensorFunc = "torch._utils._rebuild_tensor_v2"

// extractTensors 遍历 pickle 求值结果，找出所有 _rebuild_tensor_v2 调用，
// 并把它们与所在容器的路径配对。
//
// PyTorch 用 SETITEMS 把 (名字, 张量对象) 成对写进字典，所以只要递归找出
// 「字典里值是张量记录」的条目即可。
//
// **名字必须是容器路径，不能只用裸键名。**
// 训练检查点里除了模型权重还存优化器状态，形如
//
//	optimizer.state.<参数下标>.{step, exp_avg, exp_avg_sq}
//
// 每个参数一份，键名全是同一组。实测 transformer-s-10m.pt：173 个张量
// 只有 47 个唯一裸键名，`step`/`exp_avg`/`exp_avg_sq` 各重复 43 次。
// 按裸键名去重会静默吞掉 126 个真实张量，参数总量随之算错。
//
// 路径从顶层键开始逐级拼。实测四个文件的包装键各不相同 ——
// model_state / model_state_dict / state_dict / state —— 所以不认名字、
// 不做特例，一律按实际嵌套拼。代价是名字带一层包装前缀，换来的是
// 无损、无歧义，且模型权重与优化器状态天然分开。
//
// 结果按名字排序，保证输出稳定、可复现。
func extractTensors(stack []any) ([]tensorRecord, error) {
	out := make([]tensorRecord, 0, 64)
	// emitted 是「已发出的名字 → 那条记录」，用于区分两种情况：
	// 路径重复且记录相同（同一个张量被 memo 引用多次，该丢），
	// 与路径重复但记录不同（不同结构拼出了同一个字符串，不能丢）。
	emitted := map[string]tensorRecord{}

	// 给记录分配一个未被占用的名字。
	//
	// 路径是拼出来的字符串，不同结构可能拼出同一个值：扁平键 "a.b"
	// 与嵌套 a→b 都是 "a.b"。只按路径去重会静默丢掉其中一个，
	// 且丢哪个取决于 map 遍历顺序（不可复现）。
	nameFor := func(p string, rec tensorRecord) (string, bool) {
		prev, taken := emitted[p]
		if !taken {
			return p, true
		}
		if sameTensor(prev, rec) {
			return "", false // 同一个张量的重复引用
		}
		for i := 2; ; i++ {
			cand := fmt.Sprintf("%s#%d", p, i)
			if _, used := emitted[cand]; !used {
				return cand, true
			}
		}
	}

	var walk func(v any, path string) error
	walk = func(v any, path string) error {
		switch x := v.(type) {
		case map[string]any:
			for k, val := range x {
				p := joinPath(path, k)
				if rec, ok := asTensor(val); ok {
					name, keep := nameFor(p, rec)
					if !keep {
						continue
					}
					rec.Name = name
					emitted[name] = rec
					out = append(out, rec)
					continue
				}
				if err := walk(val, p); err != nil {
					return err
				}
			}
		case []any:
			for i, e := range x {
				if err := walk(e, indexPath(path, i)); err != nil {
					return err
				}
			}
		case reduceCall:
			// 张量可能是别的调用的参数（例如模块对象把参数张量存在 args 里）
			for i, a := range x.Args {
				if err := walk(a, indexPath(path, i)); err != nil {
					return err
				}
			}
		case persistentRef:
			if err := walk(x.Value, joinPath(path, "value")); err != nil {
				return err
			}
		}
		return nil
	}

	// 正常情况栈顶只有一个元素 —— 就是 torch.save 的那个对象。
	// 此时它就是根，键不加位置前缀，state_dict 的点分键名保持原样。
	// 只有在栈里确实留下多个根时才用 [i] 区分。
	root := func(i int) string {
		if len(stack) == 1 {
			return ""
		}
		return indexPath("", i)
	}
	for i, v := range stack {
		if err := walk(v, root(i)); err != nil {
			return nil, err
		}
	}

	slices.SortFunc(out, func(a, b tensorRecord) int {
		switch {
		case a.Name < b.Name:
			return -1
		case a.Name > b.Name:
			return 1
		}
		return 0
	})
	return out, nil
}

// sameTensor 判断两条记录描述的是不是同一个张量。
//
// 用逐字段比较而不是整体 ==：记录里有切片，Go 不允许直接比较。
func sameTensor(a, b tensorRecord) bool {
	return a.StorageKey == b.StorageKey &&
		a.StorageNumel == b.StorageNumel &&
		a.StorageOffset == b.StorageOffset &&
		a.StorageClass == b.StorageClass &&
		slices.Equal(a.Shape, b.Shape) &&
		slices.Equal(a.Stride, b.Stride)
}

// joinPath 拼字典键路径：父路径为空时直接用键名，否则点号连接。
func joinPath(parent, key string) string {
	if parent == "" {
		return key
	}
	return parent + "." + key
}

// indexPath 拼列表下标路径：父路径为空时是 "[i]"，否则 "父[i]"。
func indexPath(parent string, i int) string {
	return parent + "[" + strconv.Itoa(i) + "]"
}

// asTensor 判断一个值是否是 _rebuild_tensor_v2 的调用结果，并转成记录。
func asTensor(v any) (tensorRecord, bool) {
	call, ok := v.(reduceCall)
	if !ok || call.Func != rebuildTensorFunc {
		return tensorRecord{}, false
	}
	// 参数：(storage, storage_offset, size, stride, requires_grad, backward_hooks)
	if len(call.Args) < 4 {
		return tensorRecord{}, false
	}

	pid, ok := call.Args[0].(persistentRef)
	if !ok {
		return tensorRecord{}, false
	}
	// PersistentId = ('storage', 存储类, 键, 位置, 元素数) —— 5 元组
	tup, ok := pid.Value.([]any)
	if !ok || len(tup) < 5 {
		return tensorRecord{}, false
	}

	rec := tensorRecord{}
	if cls, ok := tup[1].(globalRef); ok {
		rec.StorageClass = cls.Name
	}
	rec.StorageKey = keyString(tup[2])
	if n, ok := tup[4].(int64); ok {
		rec.StorageNumel = n
	}

	if n, ok := call.Args[1].(int64); ok {
		rec.StorageOffset = n
	}
	shape, ok := intSlice(call.Args[2])
	if !ok {
		return tensorRecord{}, false
	}
	rec.Shape = shape
	if stride, ok := intSlice(call.Args[3]); ok {
		rec.Stride = stride
	}
	return rec, true
}

// intSlice 把 []any 里的整数转成 []int64。
func intSlice(v any) ([]int64, bool) {
	items, ok := v.([]any)
	if !ok {
		return nil, false
	}
	out := make([]int64, 0, len(items))
	for _, it := range items {
		n, ok := it.(int64)
		if !ok {
			return nil, false
		}
		out = append(out, n)
	}
	return out, true
}

// numel 返回形状的元素总数。
//
// 必须查溢出：损坏或伪造的头部可能给出 shape=[2^21, 2^21, 2^21]，
// 连乘回绕成负数，ParamCount 就成了 -9223372036854775808 并被当成
// 总参数打印出来。NumElements 这个量本身没有意义了，只能报错。
func numel(shape []int64) (int64, error) {
	n := int64(1)
	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("负维度 %d", d)
		}
		if d != 0 && n > math.MaxInt64/d {
			return 0, fmt.Errorf("形状 %v 的元素总数超出 int64 上限", shape)
		}
		n *= d
	}
	return n, nil
}

// isContiguous 判断张量是否是连续布局。
//
// 只有连续张量的数据才在存储块里紧密排列；非连续（转置、切片视图）
// 需要用 stride 才能定位。本工具不解码数据，只标注出来。
func isContiguous(shape, stride []int64) bool {
	if len(shape) != len(stride) {
		return false
	}
	expect := int64(1)
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] != 1 {
			if stride[i] != expect {
				return false
			}
			expect *= shape[i]
		}
	}
	return true
}
