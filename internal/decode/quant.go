package decode

import (
	"fmt"

	"github.com/sillydong/modelview/internal/model"
)

// decodeQuant 处理量化类型。
//
// 块校验放在这里而不是各 dq* 函数里：十种类型的校验规则完全相同，
// 分散写十遍只会有十处可以写错的地方。
func decodeQuant(d model.Dtype, src []byte, dst []float32) error {
	perBlock, ok := d.BlockBytes()
	if !ok {
		return ErrUnsupported{Dtype: d}
	}
	elems := d.BlockElems()
	if elems <= 0 {
		return ErrUnsupported{Dtype: d}
	}

	need := int64(len(dst))
	if need%elems != 0 {
		return fmt.Errorf("目标长度 %d 不是块大小 %d 的整数倍", need, elems)
	}
	if int64(len(src)) < need/elems*perBlock {
		return fmt.Errorf("字节不足：%d 个元素需要 %d 字节，只有 %d",
			need, need/elems*perBlock, len(src))
	}

	return decodeBlocks(d, src, dst)
}

// dequantizers 把类型映射到它的块解码函数。
//
// 用映射表而不是 switch：类型是分批实现的（32 元素块、超级块、更复杂的超级块），
// 每批只往这里加几行，不必反复改同一个 switch —— 改 switch 时容易漏掉
// 某个 case，而漏掉的后果是"解码静默失败"或"返回一堆 0"。
//
// 不在表里的类型一律报 ErrUnsupported。Q8_1 与 Q8_K 就属于这类：
// 块字节数收录了（算占用大小用得上），但没有独立参照验证过内部布局。
var dequantizers = map[model.Dtype]func([]byte, []float32) error{
	model.DtypeQ4_0: dqQ4_0,
	model.DtypeQ4_1: dqQ4_1,
	model.DtypeQ5_0: dqQ5_0,
	model.DtypeQ5_1: dqQ5_1,
	model.DtypeQ8_0: dqQ8_0,
	model.DtypeQ2K:  dqQ2K,
	model.DtypeQ3K:  dqQ3K,
	model.DtypeQ4K:  dqQ4K,
	model.DtypeQ5K:  dqQ5K,
	model.DtypeQ6K:  dqQ6K,
}

// decodeBlocks 查表分发。表里没有就明确报错，不给 0。
func decodeBlocks(d model.Dtype, src []byte, dst []float32) error {
	fn, ok := dequantizers[d]
	if !ok {
		return ErrUnsupported{Dtype: d}
	}
	return fn(src, dst)
}
