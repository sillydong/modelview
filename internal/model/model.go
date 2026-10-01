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
//
// **没有"关联的速查表条目"字段**。曾经有一个 `Ref`，由解析器按 key 填
// 一个硬编码的表名；那张表后来真的建出来时用的是另一套 ID，
// 于是这个字段指向的全是空气，而它已经在 --json 里对外了。
//
// 关联应当由展示层现算：拿到 key 调 `ref.LookupKey(k)`，它知道
// 精确键与按后缀匹配两套规则。解析器不该知道速查表的内部 ID。
type MetaKV struct {
	Key   string `json:"key"`
	Value string `json:"value"`         // 已格式化为可读字符串
	Raw   any    `json:"raw,omitempty"` // 原始值，供程序化消费
}

// QuantInfo 是一个已量化张量的块级诊断，由 analyze 包懒加载填充。
//
// 只读块头就能算出来（scale 与 min 都在块头里），所以它覆盖**全部**子块，
// 不受统计采样影响 ——「哪个子块被压得最狠」必须看全量，
// 采样会把这个结论变得不可信。
type QuantInfo struct {
	Scheme        string  `json:"scheme"`          // Q4_K / Q6_K / …
	BitsPerWeight float64 `json:"bits_per_weight"` // 该格式真实的每权重位宽
	Blocks        int64   `json:"blocks"`          // 块数
	SubBlocks     int64   `json:"sub_blocks"`      // 子块总数
	BlockElems    int     `json:"block_elems"`     // 一个（子）块覆盖的权重数

	// scale 的分布取的是**绝对值**。
	//
	// 决定「能表示多细」的是 |scale|（它就是该子块的量化步长），
	// 符号只是解码时的约定。实测真实文件里 Q6_K 的 int8 子 scale 确实有负值 ——
	// 按带符号的值统计，「最小 scale」会变成绝对值最大的那个，结论完全反了。
	ScaleMin    float64 `json:"scale_min"`
	ScaleMax    float64 `json:"scale_max"`
	ScaleMean   float64 `json:"scale_mean"`
	ScaleMedian float64 `json:"scale_median"`
	// ScaleMedianSampled 表示中位数来自等距抽样而非全量。
	//
	// 中位数是唯一必须看到全部值才能精确算出的量（其余统计量都是流式精确的），
	// 而子块数可以到千万级 —— 超预算后改为等距抽样并置这个标记。
	// min/max/均值/零计数/最扁**任何时候都是全量精确的**。
	ScaleMedianSampled bool `json:"scale_median_sampled,omitempty"`

	// ZeroScaleBlocks 是 scale 恒为 0 的子块数 —— 这些子块的权重
	// 全落在一个量化级上，信息被压没了。是「压得最狠」的直接证据。
	ZeroScaleBlocks int64 `json:"zero_scale_blocks"`

	// FlattestRatio / FlattestScale / FlattestIndex 描述「被压得最狠的子块」。
	//
	// 判据是 **|scale| / 同一个块内最大的 |scale|**，不是绝对 scale：
	// scale = d × 子scale，而 d 是整个块共用的。跨块直接比绝对 scale，
	// 会把「这个块的数值本来就小」误判成「被压平」。
	// 归一后的比值才表示编码器把这个子块挤到了多窄（0 = 完全压平）。
	//
	// FlattestIndex 是**子块序号**（张量内的顺序，0 基），不是字节偏移 ——
	// 界面上的显示形如「最扁 #7（比值 0.0200）」：`#N` 就是这个序号。
	//
	// **它曾经承诺显示成「第 N 个子块（张量内第 N×BlockElems 个权重）」，
	// 那句话没有实现，而且不该实现 —— 量过宽度了**：
	// 最坏情况下加这段字之后是 89 显示列（真实形状：本仓实测过一个
	// 3.11 亿权重的量化张量，序号 8 位、权重偏移 9 位），再加 TUI 的
	// 2 列缩进是 91 列 —— 而 80 列是 spec 定的最小终端，TUI 的
	// truncateLines 从**右边**切，切掉的恰好是末尾那个比值。
	// 比值才是"被压得最狠"的唯一证据（render 里那两条注释写着必须显示），
	// 用它换一个可以从序号乘出来的偏移量，是拿结论换中间量。
	// 序号与 BlockElems 都在（--json 里是原值），要换算的用户自己乘。
	FlattestRatio float64 `json:"flattest_ratio"`
	FlattestScale float64 `json:"flattest_scale"`
	FlattestIndex int64   `json:"flattest_index"`
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

	// NonContiguous 表示张量的数据在存储块里不是线性排列的
	//（转置、切片视图）。只有 PyTorch 会产生这种张量。
	//
	// 单独一个字段是因为它不是"异常"而是"读法不同"：解码器按线性顺序
	// 读会得到元素顺序错乱的数据，统计结果看着正常但全是错的。
	// analyze 遇到这类张量必须明确拒绝，不能给一个看似合理的分布。
	NonContiguous bool `json:"non_contiguous,omitempty"`

	// Stats 是张量数值的统计结果，由 analyze 包懒加载填充；nil 表示未扫描。
	Stats *Stats `json:"stats,omitempty"`

	Quant *QuantInfo `json:"quant,omitempty"`

	// QuantSims 是对浮点张量做的量化模拟，三档并排（Q8_0/Q6_K/Q4_K）。
	//
	// 已量化的张量不填这个字段 —— 它们没有「压到某档」的问题，看 Quant。
	// 两者互斥：浮点走模拟、量化走诊断。
	QuantSims []QuantSim `json:"quant_sims,omitempty"`
}

// QuantSim 是把一个浮点张量模拟量化到某一档的结果。
//
// **按该格式真实的编码算法模拟**（编码 → 解码 → 与原值比对），
// 不是「块内最大绝对值 / 级数」那种统一公式 —— 那套东西的结果取决于
// 「级数怎么定、块多大」这两个任意选择，实测同一批真实 F16 权重上
// 光是级数从 31 改成 63 就摆动 6.26 dB，比它自身的误差还大。
type QuantSim struct {
	Target string `json:"target"` // Q8_0 / Q6_K / Q4_K

	MaxAbsErr  float64 `json:"max_abs_err"`
	MaxRelErr  float64 `json:"max_rel_err"`
	MeanAbsErr float64 `json:"mean_abs_err"`
	SNRDB      float64 `json:"snr_db"`

	// BitsPerWeight 是该目标格式**真实**的每权重位宽
	//（Q8_0 是 8.5、Q6_K 是 6.5625、Q4_K 是 4.5，含块头开销）。
	BitsPerWeight float64 `json:"bits_per_weight"`
	// Compression 是相对源类型的压缩比（源位宽 / 目标位宽）。
	Compression float64 `json:"compression"`

	// Sampled 表示这些数字来自采样而非全量。
	Sampled bool `json:"sampled"`
}

// Stats 是一个张量数值的统计结果。
//
// 全部字段都基于**采样后**的那批值计算 —— Sampled 为 true 时，
// 下面的数字是样本的统计量，不是全量的。界面必须把这一点显示出来，
// 否则用户会把采样值当成精确值。
//
// NaN 与 Inf 只计数，不参与 Min/Max/Mean/Std —— 让它们参与的话
// 整条统计会变成 NaN，而界面上看起来只是"有点怪"。
type Stats struct {
	Count int64   `json:"count"` // 参与统计的元素个数（采样时小于参数总量）
	Min   float64 `json:"min"`
	Max   float64 `json:"max"`
	Mean  float64 `json:"mean"`
	Std   float64 `json:"std"`

	NaN int64 `json:"nan"`
	Inf int64 `json:"inf"`

	// ZeroRatio 是精确等于 0 的比例；OutlierRatio 是超出 mean±3σ 的比例。
	// 两者的分母都是参与统计的**有效值**个数（不含 NaN/Inf）。
	ZeroRatio    float64 `json:"zero_ratio"`
	OutlierRatio float64 `json:"outlier_ratio"`

	// Histogram 是 64 个等宽桶的计数，覆盖区间 [Min, Max]。
	//
	// 不再单独存直方图区间：原先的 HistMin/HistMax 在所有路径上都
	// 恒等于 Min/Max（fillHistogram 就是直接赋值），而注释给的理由
	// ——"Min/Max 会被 NaN 影响"——是错的：NaN/Inf 已经被排除在
	// Min/Max 之外了。同一个区间被两个字段表达，其中一个是冗余的。
	//
	// 全部值都是 NaN/Inf 时直方图为空（区间无意义）。
	Histogram []int64 `json:"histogram"`

	// Sampled 为 true 时上面的数字来自等距采样，不是全量。
	Sampled bool `json:"sampled"`
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

	// ArchivePrefix 是归档格式里数据块的路径前缀，只有 PyTorch 会填。
	//
	// 张量数据在 <ArchivePrefix>/data/<Tensor.StorageKey> 这个 ZIP 条目里，
	// 条目的绝对偏移拿不到（archive/zip 不暴露本地文件头位置），
	// 所以要读数据必须重新打开归档并拼出这个路径。
	ArchivePrefix string `json:"archive_prefix,omitempty"`

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
