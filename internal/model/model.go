// Package model 定义与文件格式无关的模型数据模型。
//
// 所有格式解析器都把自己的结果归一化成这里的类型；
// 上层（分析、界面、JSON 输出）只依赖本包，不依赖任何具体格式。
package model

import (
	"cmp"
	"slices"
)

// Format 是模型文件的容器格式。
type Format string

const (
	FormatGGUF        Format = "GGUF"
	FormatSafeTensors Format = "SafeTensors"
	FormatPyTorch     Format = "PyTorch"
	FormatUnknown     Format = ""
)

// MetaKV 是一条元数据。保持原始顺序，不做过滤或截断。
type MetaKV struct {
	Key   string `json:"key"`
	Value string `json:"value"`         // 已格式化为可读字符串
	Raw   any    `json:"raw,omitempty"` // 原始值，供程序化消费
	Ref   string `json:"ref,omitempty"` // 关联的速查表条目 ID，空表示无
}

// QuantInfo 描述一个已量化张量的块结构，由 analyze 包填充；解析阶段为 nil。
type QuantInfo struct {
	BlockSize int     `json:"block_size"`
	NumBlocks int64   `json:"num_blocks"`
	ScaleMin  float64 `json:"scale_min"`
	ScaleMax  float64 `json:"scale_max"`
}

// Tensor 是一个张量。
//
// 解析阶段只填名称、形状、类型、位置；Quant 由 analyze 包懒加载填充。
type Tensor struct {
	Name       string  `json:"name"`
	Dims       []int64 `json:"dims"`
	Dtype      Dtype   `json:"dtype"`
	Offset     int64   `json:"offset"`      // 数据在文件中的绝对偏移
	ByteSize   int64   `json:"byte_size"`   // 数据占用的字节数
	ParamCount int64   `json:"param_count"` // 元素个数 = dims 连乘

	// OffsetUnknown 表示 Offset 无意义（真值为 0 而非"从第 0 字节开始"）。
	//
	// PyTorch 的 .pt 是 ZIP 容器：张量数据在 <前缀>/data/<N> 条目里，
	// 条目的绝对偏移不可得（archive/zip 不暴露本地文件头位置），
	// 只有 StorageKey + StorageOffset 能定位。不加这个字段的话，
	// 消费方无法把"从文件头开始"和"不知道在哪"区分开。
	//
	// 与 SizeUnknown 同一个道理：哨兵值必须显式且唯一。
	OffsetUnknown bool `json:"offset_unknown,omitempty"`

	// StorageKey 标识数据所在的存储块。PyTorch 的 .pt 里多个张量可以共享
	// 同一存储块（权重绑定），此时它们的 StorageKey 相同。其它格式为空。
	StorageKey string `json:"storage_key,omitempty"`

	// StorageOffset 是张量数据在存储块内的**元素**偏移（不是字节）。
	StorageOffset int64 `json:"storage_offset,omitempty"`

	// SizeUnknown 表示 ByteSize 无法计算（如未收录的量化类型）。
	// 单独一个字段是必要的：否则 ByteSize==0 会被读成"这个张量真的是 0 字节"。
	SizeUnknown bool `json:"size_unknown,omitempty"`

	Quant *QuantInfo `json:"quant,omitempty"`
}

// Model 是一个已解析的模型文件。
type Model struct {
	Path     string    `json:"path"`
	Format   Format    `json:"format"`
	Version  string    `json:"version"`
	FileSize int64     `json:"file_size"`
	Arch     string    `json:"arch"`
	Metadata []MetaKV  `json:"metadata"`
	Tensors  []*Tensor `json:"tensors"`

	// 数据区布局。各格式按需填充，0 表示该格式无此概念。
	//
	// 这些是有类型的具名字段，不是 map —— 键名漂移在 map 里既无编译错误
	// 也无测试失败，只会让消费方静默读到零值。
	Alignment   int64 `json:"alignment,omitempty"`
	DataStart   int64 `json:"data_start,omitempty"`
	HeaderBytes int64 `json:"header_bytes,omitempty"`

	// StorageBytes 是文件里被张量占用的存储的字节数。
	//
	// 注意它**不等于**所有张量 ByteSize 之和：多个张量可共享一个存储块
	//（典型例子是语言模型的 token embedding 与 lm_head 权重绑定）。
	// 实测 model.pt 的张量逻辑字节和为 10,479,616，而去重后的存储只有
	// 9,431,040 字节 —— 界面必须展示这个区别，否则用户会以为算错了。
	//
	// **两种格式下的确切含义不同**，消费方按格式理解：
	//   - PyTorch：按 StorageKey 去重后的存储块字节数之和。
	//     因此它 ≤ 张量字节和，差值即别名重复计入的部分。
	//   - safetensors：数据区跨度（数据区末尾 − 数据区起点）。
	//     头部允许 data_offsets 不从 0 开始，所以它**可能大于**张量字节和。
	StorageBytes int64 `json:"storage_bytes,omitempty"`

	// TiedGroups 记录共享同一存储块的张量名分组，每组按名称排序。
	// 只有真正的共享（组内 ≥2 个张量）才会列在这里。
	TiedGroups [][]string `json:"tied_groups,omitempty"`

	// Warnings 记录解析过程中被降级处理、但用户应当知道的问题。
	// 例如未收录的量化类型导致占用大小算不出来。
	Warnings []string `json:"warnings,omitempty"`
}

// TotalParams 返回所有张量的元素总数。
func (m *Model) TotalParams() int64 {
	var n int64
	for _, t := range m.Tensors {
		n += t.ParamCount
	}
	return n
}

// TensorBytes 返回所有张量逻辑字节数之和。
//
// 与 StorageBytes 的区别见 StorageBytes 的说明：共享存储块时本值更大。
func (m *Model) TensorBytes() int64 {
	var n int64
	for _, t := range m.Tensors {
		n += t.ByteSize
	}
	return n
}

// FindTiedGroups 按 StorageKey 分组，返回共享存储块的张量名分组。
// 组内按名称排序；只有 ≥2 个张量的组会被返回，结果按首元素名排序。
func (m *Model) FindTiedGroups() [][]string {
	byKey := make(map[string][]string)
	for _, t := range m.Tensors {
		if t.StorageKey == "" {
			continue
		}
		byKey[t.StorageKey] = append(byKey[t.StorageKey], t.Name)
	}
	var out [][]string
	for _, names := range byKey {
		if len(names) < 2 {
			continue
		}
		slices.Sort(names)
		out = append(out, names)
	}
	slices.SortFunc(out, func(a, b []string) int { return cmp.Compare(a[0], b[0]) })
	return out
}

// DtypeHistogram 返回类型到张量个数的分布。
func (m *Model) DtypeHistogram() map[Dtype]int {
	h := make(map[Dtype]int, 8)
	for _, t := range m.Tensors {
		h[t.Dtype]++
	}
	return h
}
