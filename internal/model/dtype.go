package model

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
}

var dtypeTable = map[Dtype]dtypeProps{
	DtypeF64:    {BitsPerWeight: 64, IsFloat: true, BlockSize: 1, BlockBytes: 8},
	DtypeF32:    {BitsPerWeight: 32, IsFloat: true, BlockSize: 1, BlockBytes: 4},
	DtypeF16:    {BitsPerWeight: 16, IsFloat: true, BlockSize: 1, BlockBytes: 2},
	DtypeBF16:   {BitsPerWeight: 16, IsFloat: true, BlockSize: 1, BlockBytes: 2},
	DtypeF8E4M3: {BitsPerWeight: 8, IsFloat: true, BlockSize: 1, BlockBytes: 1},
	DtypeF8E5M2: {BitsPerWeight: 8, IsFloat: true, BlockSize: 1, BlockBytes: 1},

	DtypeI64:  {BitsPerWeight: 64, BlockSize: 1, BlockBytes: 8},
	DtypeI32:  {BitsPerWeight: 32, BlockSize: 1, BlockBytes: 4},
	DtypeI16:  {BitsPerWeight: 16, BlockSize: 1, BlockBytes: 2},
	DtypeI8:   {BitsPerWeight: 8, BlockSize: 1, BlockBytes: 1},
	DtypeU64:  {BitsPerWeight: 64, BlockSize: 1, BlockBytes: 8},
	DtypeU32:  {BitsPerWeight: 32, BlockSize: 1, BlockBytes: 4},
	DtypeU16:  {BitsPerWeight: 16, BlockSize: 1, BlockBytes: 2},
	DtypeU8:   {BitsPerWeight: 8, BlockSize: 1, BlockBytes: 1},
	DtypeBool: {BitsPerWeight: 8, BlockSize: 1, BlockBytes: 1},

	DtypeQ4_0: {BitsPerWeight: 4.5, IsQuantized: true, BlockSize: 32, BlockBytes: 18},
	DtypeQ4_1: {BitsPerWeight: 5.0, IsQuantized: true, BlockSize: 32, BlockBytes: 20},
	DtypeQ5_0: {BitsPerWeight: 5.5, IsQuantized: true, BlockSize: 32, BlockBytes: 22},
	DtypeQ5_1: {BitsPerWeight: 6.0, IsQuantized: true, BlockSize: 32, BlockBytes: 24},
	DtypeQ8_0: {BitsPerWeight: 8.5, IsQuantized: true, BlockSize: 32, BlockBytes: 34},
	DtypeQ8_1: {BitsPerWeight: 9.0, IsQuantized: true, BlockSize: 32, BlockBytes: 36},
	// 以下位宽由块字节数推导：bits = BlockBytes × 8 / BlockSize。
	// 数值经真实 GGUF 文件反推验证（见 gguf 包的 TestBlockTable_与真实文件吻合）。
	// 块字节数原先在 parser/gguf 里按 GGML 类型码另存一份，现已收敛到本表。
	DtypeQ2K: {BitsPerWeight: 2.625, IsQuantized: true, BlockSize: 256, BlockBytes: 84},
	DtypeQ3K: {BitsPerWeight: 3.4375, IsQuantized: true, BlockSize: 256, BlockBytes: 110},
	DtypeQ4K: {BitsPerWeight: 4.5, IsQuantized: true, BlockSize: 256, BlockBytes: 144},
	DtypeQ5K: {BitsPerWeight: 5.5, IsQuantized: true, BlockSize: 256, BlockBytes: 176},
	DtypeQ6K: {BitsPerWeight: 6.5625, IsQuantized: true, BlockSize: 256, BlockBytes: 210},
	DtypeQ8K: {BitsPerWeight: 9.125, IsQuantized: true, BlockSize: 256, BlockBytes: 292},

	// IQ 系列（i-quants）的块结构复杂且未在本项目中验证，
	// 因此不给位宽 —— 计算占用大小时会明确报"未收录"，而不是给出可能错误的数字。
	DtypeIQ2XXS: {IsQuantized: true, BlockSize: 256},
	DtypeIQ2XS:  {IsQuantized: true, BlockSize: 256},
	DtypeIQ3XXS: {IsQuantized: true, BlockSize: 256},
	DtypeIQ1S:   {IsQuantized: true, BlockSize: 256},
	DtypeIQ4NL:  {IsQuantized: true, BlockSize: 32},
	DtypeIQ3S:   {IsQuantized: true, BlockSize: 256},
	DtypeIQ2S:   {IsQuantized: true, BlockSize: 256},
	DtypeIQ4XS:  {IsQuantized: true, BlockSize: 256},
	DtypeIQ1M:   {IsQuantized: true, BlockSize: 256},
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

// Known 表示该类型是否在位宽表内。
func (d Dtype) Known() bool {
	_, ok := dtypeTable[d]
	return ok
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
