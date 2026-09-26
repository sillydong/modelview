// Package model 定义与文件格式无关的模型数据模型。
//
// 所有格式解析器都把自己的结果归一化成这里的类型；
// 上层（分析、界面、JSON 输出）只依赖本包，不依赖任何具体格式。
package model

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
// 解析阶段只填名称、形状、类型、位置；Stats 与 Quant 由 analyze 包懒加载填充。
type Tensor struct {
	Name       string  `json:"name"`
	Dims       []int64 `json:"dims"`
	Dtype      Dtype   `json:"dtype"`
	Offset     int64   `json:"offset"`      // 数据在文件中的绝对偏移
	ByteSize   int64   `json:"byte_size"`   // 数据占用的字节数
	ParamCount int64   `json:"param_count"` // 元素个数 = dims 连乘

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

	// Extra 保存格式特有的附加信息，如 ZIP 条目数、数据区起点。
	Extra map[string]string `json:"extra,omitempty"`
}

// TotalParams 返回所有张量的元素总数。
func (m *Model) TotalParams() int64 {
	var n int64
	for _, t := range m.Tensors {
		n += t.ParamCount
	}
	return n
}

// DtypeHistogram 返回类型到张量个数的分布。
func (m *Model) DtypeHistogram() map[Dtype]int {
	h := make(map[Dtype]int, 8)
	for _, t := range m.Tensors {
		h[t.Dtype]++
	}
	return h
}
