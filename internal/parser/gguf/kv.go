package gguf

import "fmt"

// GGUF 元数据值类型码。
//
// 注意 typeBool = 7 只占 1 字节，容易在类型分派里被漏掉；
// 一旦漏掉，后续所有字段都会错位。
const (
	typeUint8   uint32 = 0
	typeInt8    uint32 = 1
	typeUint16  uint32 = 2
	typeInt16   uint32 = 3
	typeUint32  uint32 = 4
	typeInt32   uint32 = 5
	typeFloat32 uint32 = 6
	typeBool    uint32 = 7
	typeString  uint32 = 8
	typeArray   uint32 = 9
	typeUint64  uint32 = 10
	typeInt64   uint32 = 11
	typeFloat64 uint32 = 12
)

// typeName 返回类型码的可读名。
func typeName(t uint32) string {
	switch t {
	case typeUint8:
		return "u8"
	case typeInt8:
		return "i8"
	case typeUint16:
		return "u16"
	case typeInt16:
		return "i16"
	case typeUint32:
		return "u32"
	case typeInt32:
		return "i32"
	case typeFloat32:
		return "f32"
	case typeBool:
		return "bool"
	case typeString:
		return "string"
	case typeArray:
		return "array"
	case typeUint64:
		return "u64"
	case typeInt64:
		return "i64"
	case typeFloat64:
		return "f64"
	}
	return fmt.Sprintf("unknown(%d)", t)
}

// rawKV 是一条原始元数据。
type rawKV struct {
	Key   string
	Value any
}

// maxArrayLen 是数组元素个数的上限（1 亿），
// 以及元数据条数、张量个数的上限。
// 防止损坏文件声称一个巨大的数量导致尝试分配内存。
const maxArrayLen = 100_000_000

// scalar 读取一个非数组、非字符串的标量值。
func (r *reader) scalar(t uint32) (any, error) {
	switch t {
	case typeUint8:
		return r.u8()
	case typeInt8:
		return r.i8()
	case typeUint16:
		return r.u16()
	case typeInt16:
		return r.i16()
	case typeUint32:
		return r.u32()
	case typeInt32:
		return r.i32()
	case typeFloat32:
		return r.f32()
	case typeBool:
		v, err := r.u8()
		if err != nil {
			return nil, err
		}
		return v != 0, nil
	case typeUint64:
		return r.u64()
	case typeInt64:
		return r.i64()
	case typeFloat64:
		return r.f64()
	}
	return nil, fmt.Errorf("不支持的标量类型 %s", typeName(t))
}

// readKVs 读取 n 条元数据。
//
// 数组的元素类型是独立字段，必须完整消费 —— 不能只取前几个就跳过，
// 否则文件指针会错位，后续所有字段都会读错。
func readKVs(r *reader, n uint64) ([]rawKV, error) {
	if n > maxArrayLen {
		return nil, fmt.Errorf("元数据条数 %d 异常", n)
	}
	out := make([]rawKV, 0, n)
	for i := uint64(0); i < n; i++ {
		key, err := r.str()
		if err != nil {
			return nil, fmt.Errorf("第 %d 条元数据的键: %w", i, err)
		}
		vtype, err := r.u32()
		if err != nil {
			return nil, fmt.Errorf("第 %d 条元数据的类型: %w", i, err)
		}
		val, err := readValue(r, vtype)
		if err != nil {
			return nil, fmt.Errorf("元数据 %q: %w", key, err)
		}
		out = append(out, rawKV{Key: key, Value: val})
	}
	return out, nil
}

// readValue 读一个值（可能是字符串或数组，也可能是标量）。
func readValue(r *reader, t uint32) (any, error) {
	switch t {
	case typeString:
		return r.str()
	case typeArray:
		return readArray(r)
	default:
		return r.scalar(t)
	}
}

// arrayValue 是一个已解析的数组。
type arrayValue struct {
	ElemType uint32
	Len      uint64
	// StrElems 在元素为字符串时有值（全部元素，不截断）。
	StrElems []string
	// NumElems 在元素为数值时有值（全部元素，不截断）。
	NumElems []any
	// Nested 在元素为数组时有值。
	Nested []arrayValue
}

// readArray 读一个数组：元素类型 u32 + 元素个数 u64 + 元素。
func readArray(r *reader) (arrayValue, error) {
	elemType, err := r.u32()
	if err != nil {
		return arrayValue{}, err
	}
	n, err := r.u64()
	if err != nil {
		return arrayValue{}, err
	}
	if n > maxArrayLen {
		return arrayValue{}, fmt.Errorf("数组长度 %d 超出上限 %d", n, maxArrayLen)
	}
	a := arrayValue{ElemType: elemType, Len: n}
	switch elemType {
	case typeString:
		a.StrElems = make([]string, 0, n)
		for i := uint64(0); i < n; i++ {
			s, err := r.str()
			if err != nil {
				return a, fmt.Errorf("字符串数组第 %d 项: %w", i, err)
			}
			a.StrElems = append(a.StrElems, s)
		}
	case typeArray:
		a.Nested = make([]arrayValue, 0, n)
		for i := uint64(0); i < n; i++ {
			sub, err := readArray(r)
			if err != nil {
				return a, fmt.Errorf("嵌套数组第 %d 项: %w", i, err)
			}
			a.Nested = append(a.Nested, sub)
		}
	default:
		a.NumElems = make([]any, 0, n)
		for i := uint64(0); i < n; i++ {
			v, err := r.scalar(elemType)
			if err != nil {
				return a, fmt.Errorf("数组第 %d 项: %w", i, err)
			}
			a.NumElems = append(a.NumElems, v)
		}
	}
	return a, nil
}
