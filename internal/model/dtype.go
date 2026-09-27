package model

import "iter"

// Dtype 是与文件格式无关的数据类型。
type Dtype string

const (
	DtypeUnknown Dtype = "?"

	DtypeF64    Dtype = "F64"
	DtypeF32    Dtype = "F32"
	DtypeF16    Dtype = "F16"
	DtypeBF16   Dtype = "BF16"
	DtypeF8E4M3 Dtype = "F8_E4M3"
	DtypeF8E5M2 Dtype = "F8_E5M2"

	DtypeI64 Dtype = "I64"
	DtypeI32 Dtype = "I32"
	DtypeI16 Dtype = "I16"
	DtypeI8  Dtype = "I8"

	DtypeU64  Dtype = "U64"
	DtypeU32  Dtype = "U32"
	DtypeU16  Dtype = "U16"
	DtypeU8   Dtype = "U8"
	DtypeBool Dtype = "BOOL"

	// GGML 量化类型。
	DtypeQ4_0   Dtype = "Q4_0"
	DtypeQ4_1   Dtype = "Q4_1"
	DtypeQ5_0   Dtype = "Q5_0"
	DtypeQ5_1   Dtype = "Q5_1"
	DtypeQ8_0   Dtype = "Q8_0"
	DtypeQ8_1   Dtype = "Q8_1"
	DtypeQ2K    Dtype = "Q2_K"
	DtypeQ3K    Dtype = "Q3_K"
	DtypeQ4K    Dtype = "Q4_K"
	DtypeQ5K    Dtype = "Q5_K"
	DtypeQ6K    Dtype = "Q6_K"
	DtypeQ8K    Dtype = "Q8_K"
	DtypeIQ2XXS Dtype = "IQ2_XXS"
	DtypeIQ2XS  Dtype = "IQ2_XS"
	DtypeIQ3XXS Dtype = "IQ3_XXS"
	DtypeIQ1S   Dtype = "IQ1_S"
	DtypeIQ4NL  Dtype = "IQ4_NL"
	DtypeIQ3S   Dtype = "IQ3_S"
	DtypeIQ2S   Dtype = "IQ2_S"
	DtypeIQ4XS  Dtype = "IQ4_XS"
	DtypeIQ1M   Dtype = "IQ1_M"
)

// dtypeProps 描述一种类型的存储属性。
type dtypeProps struct {
	// BitsPerWeight 是每个权重占用的比特数。
	// 非量化类型是整数（F32=32）；量化类型是均值（Q4_K=4.5）。
	BitsPerWeight float64
	// IsQuantized 表示该类型带块级 scale，需要按块解码。
	IsQuantized bool
	// BlockSize 是一个量化块包含的权重个数，非量化类型为 1。
	BlockSize int
	// BlockBytes 是一个块占用的字节数（含块头里的 scale / min）。
	// 非量化类型的块就是单个元素，等于每元素字节数。
	// 0 表示未收录 —— 解码时会明确报错，而不是给出可能错误的数字。
	BlockBytes int64
	// IsFloat 表示可按 IEEE 浮点解码。
	IsFloat bool

	// GGMLCode 是这个类型在 GGUF 里的类型码（ggml_type 的取值）。
	//
	// **-1 表示没有类型码**。哨兵值必须显式且唯一：0 是合法的码
	//（F32），拿它当"没有"会让 DtypeUnknown 被当成 F32 ——
	// 一个不认识的类型悄悄变成"最普通的那个"，编译器和测试都不会响。
	//
	// 它放在这里而不是 parser/gguf 里：类型码是 Dtype 自己的属性，
	// 而速查表要按码列出全部类型 —— 让数据包去 import 一个解析器
	// 只为拿一张常量表是本末倒置。
	GGMLCode int
}

var dtypeTable = map[Dtype]dtypeProps{
	DtypeF64:    {BitsPerWeight: 64, IsFloat: true, BlockSize: 1, BlockBytes: 8, GGMLCode: 28},
	DtypeF32:    {BitsPerWeight: 32, IsFloat: true, BlockSize: 1, BlockBytes: 4, GGMLCode: 0},
	DtypeF16:    {BitsPerWeight: 16, IsFloat: true, BlockSize: 1, BlockBytes: 2, GGMLCode: 1},
	DtypeBF16:   {BitsPerWeight: 16, IsFloat: true, BlockSize: 1, BlockBytes: 2, GGMLCode: 30},
	DtypeF8E4M3: {BitsPerWeight: 8, IsFloat: true, BlockSize: 1, BlockBytes: 1, GGMLCode: -1},
	DtypeF8E5M2: {BitsPerWeight: 8, IsFloat: true, BlockSize: 1, BlockBytes: 1, GGMLCode: -1},

	DtypeI64:  {BitsPerWeight: 64, BlockSize: 1, BlockBytes: 8, GGMLCode: 27},
	DtypeI32:  {BitsPerWeight: 32, BlockSize: 1, BlockBytes: 4, GGMLCode: 26},
	DtypeI16:  {BitsPerWeight: 16, BlockSize: 1, BlockBytes: 2, GGMLCode: 25},
	DtypeI8:   {BitsPerWeight: 8, BlockSize: 1, BlockBytes: 1, GGMLCode: 24},
	DtypeU64:  {BitsPerWeight: 64, BlockSize: 1, BlockBytes: 8, GGMLCode: -1},
	DtypeU32:  {BitsPerWeight: 32, BlockSize: 1, BlockBytes: 4, GGMLCode: -1},
	DtypeU16:  {BitsPerWeight: 16, BlockSize: 1, BlockBytes: 2, GGMLCode: -1},
	DtypeU8:   {BitsPerWeight: 8, BlockSize: 1, BlockBytes: 1, GGMLCode: -1},
	DtypeBool: {BitsPerWeight: 8, BlockSize: 1, BlockBytes: 1, GGMLCode: -1},

	DtypeQ4_0: {BitsPerWeight: 4.5, IsQuantized: true, BlockSize: 32, BlockBytes: 18, GGMLCode: 2},
	DtypeQ4_1: {BitsPerWeight: 5.0, IsQuantized: true, BlockSize: 32, BlockBytes: 20, GGMLCode: 3},
	DtypeQ5_0: {BitsPerWeight: 5.5, IsQuantized: true, BlockSize: 32, BlockBytes: 22, GGMLCode: 6},
	DtypeQ5_1: {BitsPerWeight: 6.0, IsQuantized: true, BlockSize: 32, BlockBytes: 24, GGMLCode: 7},
	DtypeQ8_0: {BitsPerWeight: 8.5, IsQuantized: true, BlockSize: 32, BlockBytes: 34, GGMLCode: 8},
	DtypeQ8_1: {BitsPerWeight: 9.0, IsQuantized: true, BlockSize: 32, BlockBytes: 36, GGMLCode: 9},
	// 以下位宽由块字节数推导：bits = BlockBytes × 8 / BlockSize。
	// 数值经真实 GGUF 文件反推验证（见 gguf 包的 TestBlockTable_与真实文件吻合）。
	// 块字节数原先在 parser/gguf 里按 GGML 类型码另存一份，现已收敛到本表。
	DtypeQ2K: {BitsPerWeight: 2.625, IsQuantized: true, BlockSize: 256, BlockBytes: 84, GGMLCode: 10},
	DtypeQ3K: {BitsPerWeight: 3.4375, IsQuantized: true, BlockSize: 256, BlockBytes: 110, GGMLCode: 11},
	DtypeQ4K: {BitsPerWeight: 4.5, IsQuantized: true, BlockSize: 256, BlockBytes: 144, GGMLCode: 12},
	DtypeQ5K: {BitsPerWeight: 5.5, IsQuantized: true, BlockSize: 256, BlockBytes: 176, GGMLCode: 13},
	DtypeQ6K: {BitsPerWeight: 6.5625, IsQuantized: true, BlockSize: 256, BlockBytes: 210, GGMLCode: 14},
	DtypeQ8K: {BitsPerWeight: 9.125, IsQuantized: true, BlockSize: 256, BlockBytes: 292, GGMLCode: 15},

	// IQ 系列（i-quants）的块结构复杂且未在本项目中验证，
	// 因此不给位宽 —— 计算占用大小时会明确报"未收录"，而不是给出可能错误的数字。
	DtypeIQ2XXS: {IsQuantized: true, BlockSize: 256, GGMLCode: 16},
	DtypeIQ2XS:  {IsQuantized: true, BlockSize: 256, GGMLCode: 17},
	DtypeIQ3XXS: {IsQuantized: true, BlockSize: 256, GGMLCode: 18},
	DtypeIQ1S:   {IsQuantized: true, BlockSize: 256, GGMLCode: 19},
	DtypeIQ4NL:  {IsQuantized: true, BlockSize: 32, GGMLCode: 20},
	DtypeIQ3S:   {IsQuantized: true, BlockSize: 256, GGMLCode: 21},
	DtypeIQ2S:   {IsQuantized: true, BlockSize: 256, GGMLCode: 22},
	DtypeIQ4XS:  {IsQuantized: true, BlockSize: 256, GGMLCode: 23},
	DtypeIQ1M:   {IsQuantized: true, BlockSize: 256, GGMLCode: 29},
}

// Props 返回该类型的存储属性。未知类型返回零值。
func (d Dtype) Props() dtypeProps {
	return dtypeTable[d]
}

// BitsPerWeight 返回每权重占用的比特数，未知类型返回 0。
func (d Dtype) BitsPerWeight() float64 { return dtypeTable[d].BitsPerWeight }

// IsQuantized 表示该类型是否带块级 scale。
func (d Dtype) IsQuantized() bool { return dtypeTable[d].IsQuantized }

// IsFloat 表示该类型是否为 IEEE 浮点。
func (d Dtype) IsFloat() bool { return dtypeTable[d].IsFloat }

// GGMLCode 返回这个类型在 GGUF 里的类型码。
//
// 第二个返回值为 false 表示**没有类型码**，与"类型码是 0"是两回事 ——
// 0 是 F32 的合法码。调用方不能拿零值当"没有"。
func (d Dtype) GGMLCode() (uint32, bool) {
	p, ok := dtypeTable[d]
	if !ok || p.GGMLCode < 0 {
		return 0, false
	}
	return uint32(p.GGMLCode), true
}

// AllDtypes 遍历所有已收录的类型，顺序不保证。
//
// 导出是为了让**按属性反查**的调用方（如 parser 按 GGML 类型码反查，
// 速查表按位宽排序）不必自己维护一份平行的清单 ——
// 那正是本次收敛要消灭的东西。
func AllDtypes() iter.Seq[Dtype] {
	return func(yield func(Dtype) bool) {
		for d := range dtypeTable {
			if !yield(d) {
				return
			}
		}
	}
}

// Known 表示该类型是否在位宽表内。
func (d Dtype) Known() bool {
	_, ok := dtypeTable[d]
	return ok
}

// safetensorsDtypes 是 safetensors 头部 dtype 字段的**全部合法取值**，
// 顺序即规范里的书写顺序。
//
// 这些拼写与本包的 Dtype 字面量恰好相同，但那是巧合而非契约：
// 所以这里写全，而不是用 `Dtype(code).Known()` 去推导 ——
// 推导会把 Q4_K 这种 safetensors 里根本不存在的名字也判为合法，
// 于是"文件头部写错了"这个真实的错误会被静默接受。
//
// 放在 model 而不是某个解析器里：解析器（认识它）与速查表（展示它）
// 都要用同一份清单，各写一份的话，将来 safetensors 加一个新类型时
// 必然只改一处 —— 而漂移的表现是速查表里少一行，没有任何东西会红。
var safetensorsDtypes = []Dtype{
	DtypeF64, DtypeF32, DtypeF16, DtypeBF16, DtypeF8E4M3, DtypeF8E5M2,
	DtypeI64, DtypeI32, DtypeI16, DtypeI8,
	DtypeU64, DtypeU32, DtypeU16, DtypeU8,
	DtypeBool,
}

// SafetensorsDtypes 按规范顺序返回 safetensors 支持的全部 dtype。
//
// 返回副本：调用方拿到的是自己的切片，改它不会污染这张表。
func SafetensorsDtypes() []Dtype {
	return append([]Dtype(nil), safetensorsDtypes...)
}

// DtypeFromSafetensors 把 safetensors 头部的 dtype 字符串映射到 Dtype。
func DtypeFromSafetensors(code string) (Dtype, bool) {
	for _, d := range safetensorsDtypes {
		if string(d) == code {
			return d, true
		}
	}
	return DtypeUnknown, false
}

// ByteSize 返回每元素占用的字节数。
//
// 只对**非量化且位宽已知**的类型成立 —— 量化类型（Q4_K 等）按块存储，
// 没有"每元素字节数"这个概念，返回 false。
//
// 这是全项目唯一的 dtype→字节数 出处。各解析器不要再自己写一份 switch：
// 那种拷贝漏改时不会编译报错，只会让调用点的完整性校验
// （如 safetensors 的"data_offsets 跨度必须等于形状 × 字节数"）**静默失效**。
func (d Dtype) ByteSize() (int64, bool) {
	p, ok := dtypeTable[d]
	if !ok || p.IsQuantized || p.BlockSize != 1 || p.BitsPerWeight <= 0 {
		return 0, false
	}
	if bits := p.BitsPerWeight; bits == float64(int64(bits)) {
		return int64(bits) / 8, true
	}
	return 0, false
}

// BlockBytes 返回一个块占用的字节数。
//
// 非量化类型的块就是单个元素，等于每元素字节数；量化类型是若干个权重
// 加块头（scale / min）的总字节数。
//
// 第二返回值为 false 表示该类型的块结构未收录（IQ 系列），
// 调用方**必须报错**而不是当成 0 —— 0 会被读成"这个张量不占空间"。
func (d Dtype) BlockBytes() (int64, bool) {
	p, ok := dtypeTable[d]
	if !ok || p.BlockBytes <= 0 {
		return 0, false
	}
	return p.BlockBytes, true
}

// BlockElems 返回一个块包含的权重个数。非量化类型是 1，未知类型是 0。
func (d Dtype) BlockElems() int64 {
	p, ok := dtypeTable[d]
	if !ok {
		return 0
	}
	return int64(p.BlockSize)
}
