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
	// IsFloat 表示可按 IEEE 浮点解码。
	IsFloat bool
}

var dtypeTable = map[Dtype]dtypeProps{
	DtypeF64:    {BitsPerWeight: 64, IsFloat: true, BlockSize: 1},
	DtypeF32:    {BitsPerWeight: 32, IsFloat: true, BlockSize: 1},
	DtypeF16:    {BitsPerWeight: 16, IsFloat: true, BlockSize: 1},
	DtypeBF16:   {BitsPerWeight: 16, IsFloat: true, BlockSize: 1},
	DtypeF8E4M3: {BitsPerWeight: 8, IsFloat: true, BlockSize: 1},
	DtypeF8E5M2: {BitsPerWeight: 8, IsFloat: true, BlockSize: 1},

	DtypeI64:  {BitsPerWeight: 64, BlockSize: 1},
	DtypeI32:  {BitsPerWeight: 32, BlockSize: 1},
	DtypeI16:  {BitsPerWeight: 16, BlockSize: 1},
	DtypeI8:   {BitsPerWeight: 8, BlockSize: 1},
	DtypeU64:  {BitsPerWeight: 64, BlockSize: 1},
	DtypeU32:  {BitsPerWeight: 32, BlockSize: 1},
	DtypeU16:  {BitsPerWeight: 16, BlockSize: 1},
	DtypeU8:   {BitsPerWeight: 8, BlockSize: 1},
	DtypeBool: {BitsPerWeight: 8, BlockSize: 1},

	DtypeQ4_0:   {BitsPerWeight: 4.5, IsQuantized: true, BlockSize: 32},
	DtypeQ4_1:   {BitsPerWeight: 5.0, IsQuantized: true, BlockSize: 32},
	DtypeQ5_0:   {BitsPerWeight: 5.5, IsQuantized: true, BlockSize: 32},
	DtypeQ5_1:   {BitsPerWeight: 6.0, IsQuantized: true, BlockSize: 32},
	DtypeQ8_0:   {BitsPerWeight: 8.5, IsQuantized: true, BlockSize: 32},
	DtypeQ8_1:   {BitsPerWeight: 9.0, IsQuantized: true, BlockSize: 32},
	DtypeQ2K:    {BitsPerWeight: 2.5625, IsQuantized: true, BlockSize: 256},
	DtypeQ3K:    {BitsPerWeight: 3.4375, IsQuantized: true, BlockSize: 256},
	DtypeQ4K:    {BitsPerWeight: 4.5, IsQuantized: true, BlockSize: 256},
	DtypeQ5K:    {BitsPerWeight: 5.5, IsQuantized: true, BlockSize: 256},
	DtypeQ6K:    {BitsPerWeight: 6.5625, IsQuantized: true, BlockSize: 256},
	DtypeQ8K:    {BitsPerWeight: 8.5, IsQuantized: true, BlockSize: 256},
	DtypeIQ2XXS: {BitsPerWeight: 2.0625, IsQuantized: true, BlockSize: 256},
	DtypeIQ2XS:  {BitsPerWeight: 2.3125, IsQuantized: true, BlockSize: 256},
	DtypeIQ3XXS: {BitsPerWeight: 3.0625, IsQuantized: true, BlockSize: 256},
	DtypeIQ1S:   {BitsPerWeight: 1.5625, IsQuantized: true, BlockSize: 256},
	DtypeIQ4NL:  {BitsPerWeight: 4.5, IsQuantized: true, BlockSize: 32},
	DtypeIQ3S:   {BitsPerWeight: 3.4375, IsQuantized: true, BlockSize: 256},
	DtypeIQ2S:   {BitsPerWeight: 2.5, IsQuantized: true, BlockSize: 256},
	DtypeIQ4XS:  {BitsPerWeight: 4.25, IsQuantized: true, BlockSize: 256},
	DtypeIQ1M:   {BitsPerWeight: 1.75, IsQuantized: true, BlockSize: 256},
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
