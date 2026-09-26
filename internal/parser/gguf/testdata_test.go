package gguf

import (
	"bytes"
	"encoding/binary"
)

// builder 用来手工拼接 GGUF 字节流，供测试构造确定性样本。
type builder struct {
	buf bytes.Buffer
}

func newBuilder() *builder { return &builder{} }

func (b *builder) bytes() []byte { return b.buf.Bytes() }

func (b *builder) raw(v []byte) *builder { b.buf.Write(v); return b }

func (b *builder) u8(v uint8) *builder { b.buf.WriteByte(v); return b }

func (b *builder) u16(v uint16) *builder {
	_ = binary.Write(&b.buf, binary.LittleEndian, v)
	return b
}

func (b *builder) u32(v uint32) *builder {
	_ = binary.Write(&b.buf, binary.LittleEndian, v)
	return b
}

func (b *builder) u64(v uint64) *builder {
	_ = binary.Write(&b.buf, binary.LittleEndian, v)
	return b
}

func (b *builder) i8(v int8) *builder { b.buf.WriteByte(byte(v)); return b }

func (b *builder) i16(v int16) *builder {
	_ = binary.Write(&b.buf, binary.LittleEndian, v)
	return b
}

func (b *builder) i32(v int32) *builder {
	_ = binary.Write(&b.buf, binary.LittleEndian, v)
	return b
}

func (b *builder) i64(v int64) *builder {
	_ = binary.Write(&b.buf, binary.LittleEndian, v)
	return b
}

func (b *builder) f32(v float32) *builder {
	_ = binary.Write(&b.buf, binary.LittleEndian, v)
	return b
}

func (b *builder) f64(v float64) *builder {
	_ = binary.Write(&b.buf, binary.LittleEndian, v)
	return b
}

// str 写 GGUF 字符串：u64 长度 + 字节。
func (b *builder) str(s string) *builder {
	b.u64(uint64(len(s)))
	b.buf.WriteString(s)
	return b
}

// header 写 GGUF 文件头。
func (b *builder) header(version uint32, tensorCount, kvCount uint64) *builder {
	b.raw([]byte{'G', 'G', 'U', 'F'})
	b.u32(version)
	b.u64(tensorCount)
	b.u64(kvCount)
	return b
}

// kv 写一条元数据的键和类型（值需由调用方接着写）。
func (b *builder) kv(key string, vtype uint32) *builder {
	b.str(key)
	b.u32(vtype)
	return b
}
