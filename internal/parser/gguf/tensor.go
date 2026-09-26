package gguf

import "fmt"

// tensorInfo 是 GGUF 里的张量描述符（不含数据本身）。
type tensorInfo struct {
	Name   string
	Dims   []int64 // GGUF 按「最快变化的维度在前」存储
	Type   uint32
	Offset int64 // 相对数据区起点的偏移
}

// maxDims 是张量维数上限。GGUF 实际最多 4 维，留些余量。
const maxDims = 8

// readTensorInfos 读 n 个张量描述符。
func readTensorInfos(r *reader, n uint64) ([]tensorInfo, error) {
	if n > maxArrayLen {
		return nil, fmt.Errorf("张量个数 %d: %w（上限 %d）", n, ErrTooManyEntries, maxArrayLen)
	}
	out := make([]tensorInfo, 0, capFor(n))
	for i := uint64(0); i < n; i++ {
		name, err := r.str()
		if err != nil {
			return nil, fmt.Errorf("第 %d 个张量的名称: %w", i, err)
		}
		nd, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("张量 %q 的维数: %w", name, err)
		}
		if nd > maxDims {
			return nil, fmt.Errorf("张量 %q 维数 %d 超出上限 %d", name, nd, maxDims)
		}
		dims := make([]int64, nd)
		for d := range dims {
			v, err := r.u64()
			if err != nil {
				return nil, fmt.Errorf("张量 %q 第 %d 维: %w", name, d, err)
			}
			dims[d] = int64(v)
		}
		t, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("张量 %q 的类型: %w", name, err)
		}
		off, err := r.u64()
		if err != nil {
			return nil, fmt.Errorf("张量 %q 的偏移: %w", name, err)
		}
		out = append(out, tensorInfo{Name: name, Dims: dims, Type: t, Offset: int64(off)})
	}
	return out, nil
}

// alignUp 把 off 向上对齐到 align 的整数倍。
//
// 这一步不能省：GGUF 的数据区起点是「张量描述符结束位置」向上对齐后的位置，
// 直接拿描述符结束偏移当数据区起点会读出垃圾数据。
func alignUp(off, align int64) int64 {
	if align <= 1 {
		return off
	}
	rem := off % align
	if rem == 0 {
		return off
	}
	return off + (align - rem)
}

// blockBytes 是各类型「一个块」占用的字节数（含块头里的 scale / min）。
//
// 非量化类型是单元素字节数。
//
// 表中数值经四个真实 GGUF 文件反推验证：按此表累加所有张量的字节数，
// 最后一个张量的结束位置恰好等于文件大小（差值 0 字节）。
// 未列入的类型（IQ 系列）块结构复杂且未经本项目验证，
// tensorByteSize 会明确报错，而不是给出可能错误的数字。
var blockBytes = map[uint32]int64{
	0:  4,   // F32
	1:  2,   // F16
	2:  18,  // Q4_0:  2 (d) + 16 (4bit × 32)
	3:  20,  // Q4_1:  2 (d) + 2 (m) + 16
	6:  22,  // Q5_0:  2 (d) + 4 (qh) + 16
	7:  24,  // Q5_1:  2 (d) + 2 (m) + 4 (qh) + 16
	8:  34,  // Q8_0:  2 (d) + 32 (int8 × 32)
	9:  36,  // Q8_1:  2 (d) + 2 (s) + 32
	10: 84,  // Q2_K:  16 (scales) + 64 (qs) + 2 (d) + 2 (dmin)
	11: 110, // Q3_K:  32 (hmask) + 64 (qs) + 12 (scales) + 2 (d)
	12: 144, // Q4_K:  2 (d) + 2 (dmin) + 12 (scales) + 128 (qs)
	13: 176, // Q5_K:  2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (qs)
	14: 210, // Q6_K:  128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
	15: 292, // Q8_K:  4 (d) + 256 (qs) + 32 (bsums)
	24: 1,   // I8
	25: 2,   // I16
	26: 4,   // I32
	27: 8,   // I64
	28: 8,   // F64
	30: 2,   // BF16
}

// blockElemCount 是各量化类型「一个块」包含的权重个数。
func blockElemCount(t uint32) int64 {
	switch t {
	case 2, 3, 6, 7, 8, 9: // Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1
		return 32
	case 10, 11, 12, 13, 14, 15: // Q2_K … Q8_K
		return 256
	}
	return 1
}

// ErrUnknownBlockType 表示该 GGML 类型的块结构未收录，无法计算占用大小。
type ErrUnknownBlockType struct{ Code uint32 }

func (e ErrUnknownBlockType) Error() string {
	return fmt.Sprintf("GGML 类型码 %d 的块结构未收录，无法计算占用大小", e.Code)
}

// tensorByteSize 计算一个张量占用的字节数。
//
// 非量化类型 = 元素数 × 每元素字节数。
// 量化类型 = 块数 × 每块字节数，且元素数必须是块大小的整数倍。
func tensorByteSize(dims []int64, dtype uint32) (int64, error) {
	elems := int64(1)
	for _, d := range dims {
		if d < 0 {
			return 0, fmt.Errorf("负的维度 %d", d)
		}
		elems *= d
	}

	perBlock, ok := blockBytes[dtype]
	if !ok {
		return 0, ErrUnknownBlockType{Code: dtype}
	}

	blockElems := blockElemCount(dtype)
	if blockElems == 1 {
		return elems * perBlock, nil
	}

	if elems%blockElems != 0 {
		return 0, fmt.Errorf("元素数 %d 不是块大小 %d 的整数倍", elems, blockElems)
	}
	return (elems / blockElems) * perBlock, nil
}
