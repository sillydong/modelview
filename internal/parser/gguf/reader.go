// Package gguf 解析 GGUF 格式（llama.cpp / ollama 使用的模型容器）。
//
// 文件布局：
//
//	magic[4] "GGUF" | version u32 | tensor_count u64 | kv_count u64
//	kv:     { key string, value_type u32, value }         × kv_count
//	tensor: { name string, n_dims u32, dims u64[n_dims],
//	          type u32, offset u64 }                       × tensor_count
//	<padding 到 general.alignment>
//	tensor data
//
// 所有多字节整数为小端。
package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// reader 是从文件头部顺序读取的小端读取器，自己跟踪偏移量。
//
// 不把整个文件读进内存，只顺序消费头部区域 —— 因此对 17 GiB 的文件也能秒级完成。
type reader struct {
	br  *bufio.Reader
	off int64
}

func newReader(ra io.ReaderAt) *reader {
	const bufSize = 1 << 20 // 1 MiB
	return &reader{
		// SectionReader 从偏移 0 读到文件末尾；MaxInt64 只表示"不设上限"。
		br: bufio.NewReaderSize(io.NewSectionReader(ra, 0, math.MaxInt64), bufSize),
	}
}

// offset 返回当前已消费的字节数。
func (r *reader) offset() int64 { return r.off }

func (r *reader) read(p []byte) error {
	n, err := io.ReadFull(r.br, p)
	r.off += int64(n)
	if err != nil {
		return fmt.Errorf("偏移 %d 处读取 %d 字节: %w", r.off-int64(n), len(p), err)
	}
	return nil
}

// skip 跳过 n 字节。n 为负时报错。
//
// **目前只有测试在调**：生产路径按偏移定位后再读，不需要线性跳过。
func (r *reader) skip(n int64) error {
	if n < 0 {
		return fmt.Errorf("skip 负数: %d", n)
	}
	got, err := r.br.Discard(int(n))
	r.off += int64(got)
	if err != nil {
		return fmt.Errorf("偏移 %d 处跳过 %d 字节: %w", r.off-int64(got), n, err)
	}
	return nil
}

func (r *reader) u8() (uint8, error) {
	var b [1]byte
	if err := r.read(b[:]); err != nil {
		return 0, err
	}
	return b[0], nil
}

func (r *reader) u16() (uint16, error) {
	var b [2]byte
	if err := r.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(b[:]), nil
}

func (r *reader) u32() (uint32, error) {
	var b [4]byte
	if err := r.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (r *reader) u64() (uint64, error) {
	var b [8]byte
	if err := r.read(b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (r *reader) i8() (int8, error) {
	v, err := r.u8()
	return int8(v), err
}

func (r *reader) i16() (int16, error) {
	v, err := r.u16()
	return int16(v), err
}

func (r *reader) i32() (int32, error) {
	v, err := r.u32()
	return int32(v), err
}

func (r *reader) i64() (int64, error) {
	v, err := r.u64()
	return int64(v), err
}

func (r *reader) f32() (float32, error) {
	v, err := r.u32()
	return math.Float32frombits(v), err
}

func (r *reader) f64() (float64, error) {
	v, err := r.u64()
	return math.Float64frombits(v), err
}

// ErrStringTooLong 表示字符串声称的长度超过 maxStringLen。
//
// 同样是"可识别错误"：否则测试只能钉住"存在一个 ≤ 1 TiB 的阈值"，
// 钉不住这个阈值具体是 64 MiB。
var ErrStringTooLong = errors.New("字符串长度超过上限")

// maxStringLen 是单个字符串的长度上限（64 MiB）。
// 用于在文件损坏导致指针错位时尽早失败，而不是尝试分配巨大内存。
const maxStringLen = 64 << 20

// str 读一个 GGUF 字符串：u64 长度 + 字节内容。
func (r *reader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	if n > maxStringLen {
		return "", fmt.Errorf("偏移 %d 处字符串长度 %d: %w（上限 %d）",
			r.off, n, ErrStringTooLong, maxStringLen)
	}
	if n == 0 {
		return "", nil
	}
	buf := make([]byte, n)
	if err := r.read(buf); err != nil {
		return "", err
	}
	return string(buf), nil
}
