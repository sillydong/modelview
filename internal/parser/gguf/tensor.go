package gguf

import (
	"fmt"
	"math"
)

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
			return nil, fmt.Errorf("张量 %q 维数 %d: %w（上限 %d）", name, nd, ErrTooManyDims, maxDims)
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

// ErrUnknownBlockType 表示该 GGML 类型的块结构未收录，无法计算占用大小。
type ErrUnknownBlockType struct{ Code uint32 }

func (e ErrUnknownBlockType) Error() string {
	return fmt.Sprintf("GGML 类型码 %d 的块结构未收录，无法计算占用大小", e.Code)
}

// elemCount 把形状的各个维度连乘，溢出时报错。
//
// **只留一份**：tensorByteSize 与解析循环都要这个数，而原先只有前者
// 查了溢出 —— 后者回绕成负数，于是同一个文件里一处报「元素总数超出
// int64 上限」、另一处把回绕值当事实打印给用户（实测 dims=[3, 2^62]
// 会打出「总参数 -4611686018427387904」，CLI/JSON/TUI 三处都是）。
//
// safetensors 与 pytorch 两个解析器都各自查了溢出；GGUF 这边合并成
// 一个函数，是因为它自己内部就有两处要算元素总数。
func elemCount(dims []int64) (int64, error) {
	n := int64(1)
	for _, d := range dims {
		// 负维度要先挡住：`math.MaxInt64/d` 对负数是负数，
		// 后面那条溢出判据会先命中，报出来的原因就变成了溢出
		if d < 0 {
			return 0, fmt.Errorf("负的维度 %d", d)
		}
		if d != 0 && n > math.MaxInt64/d {
			return 0, fmt.Errorf("形状 %v 的元素总数超出 int64 上限", dims)
		}
		n *= d
	}
	return n, nil
}

// tensorByteSize 计算一个张量占用的字节数。
//
// 非量化类型 = 元素数 × 每元素字节数。
// 量化类型 = 块数 × 每块字节数，且元素数必须是块大小的整数倍。
//
// 块尺寸来自 model.Dtype —— 全项目唯一出处。这里保留 code→Dtype 的转换，
// 是因为错误信息里要带上原始的 GGML 类型码，便于对着规范查。
func tensorByteSize(dims []int64, code uint32) (int64, error) {
	elems, err := elemCount(dims)
	if err != nil {
		return 0, err
	}

	dtype, known := ggmlDtype(code)
	if !known {
		return 0, ErrUnknownBlockType{Code: code}
	}
	perBlock, ok := dtype.BlockBytes()
	if !ok {
		return 0, ErrUnknownBlockType{Code: code}
	}
	if h := dtype.BlockElems(); h > 1 {
		if elems%h != 0 {
			return 0, fmt.Errorf("元素数 %d 不是块大小 %d 的整数倍", elems, h)
		}
		return (elems / h) * perBlock, nil
	}
	return elems * perBlock, nil
}
