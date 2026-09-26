package safetensors

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"slices"

	"github.com/sillydong/modelview/internal/model"
)

// Parse 解析一个 safetensors 文件。
//
// 只读取文件头部，不读取张量数据本身 —— 大文件也能秒级返回。
func Parse(path string) (*model.Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("打开 %s: %w", path, err)
	}
	//nolint:errcheck // 只读文件，Close 失败不影响解析结果
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}

	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		return nil, fmt.Errorf("读取头部长度: %w", err)
	}
	headerLen := binary.LittleEndian.Uint64(lenBuf[:])
	if headerLen == 0 || headerLen > maxHeaderLen {
		return nil, fmt.Errorf("头部长度 %d 超出合理范围（1..%d）", headerLen, maxHeaderLen)
	}
	if int64(headerLen) > st.Size()-8 {
		return nil, fmt.Errorf("头部长度 %d 超过文件剩余大小 %d", headerLen, st.Size()-8)
	}

	raw := make([]byte, headerLen)
	if _, err := io.ReadFull(f, raw); err != nil {
		return nil, fmt.Errorf("读取头部: %w", err)
	}

	hdr, err := parseHeader(raw)
	if err != nil {
		return nil, err
	}

	dataStart := int64(8) + int64(headerLen)

	m := &model.Model{
		Path:        path,
		Format:      model.FormatSafeTensors,
		Version:     "v1",
		FileSize:    st.Size(),
		Tensors:     make([]*model.Tensor, 0, len(hdr.Tensors)),
		DataStart:   dataStart,
		HeaderBytes: int64(headerLen),
	}
	if len(hdr.Metadata) > 0 {
		for k, v := range hdr.Metadata {
			m.Metadata = append(m.Metadata, model.MetaKV{Key: k, Value: v, Raw: v})
		}
		slices.SortFunc(m.Metadata, func(a, b model.MetaKV) int { return cmp.Compare(a.Key, b.Key) })
	}

	var maxEnd int64
	for _, name := range hdr.Names {
		e := hdr.Tensors[name]

		dtype, err := parseDtype(e.Dtype)
		if err != nil {
			return nil, fmt.Errorf("张量 %q: %w", name, err)
		}

		// 连乘必须查溢出：回绕成负数或 0 之后，下面那条
		// "跨度必须等于形状 × 字节数" 的校验会被 0 * 4 == 0 恰好满足，
		// 整条防线就绕过去了（实测 shape=[2^32, 2^32] 会得到 params=0）。
		params := int64(1)
		for _, d := range e.Shape {
			if d < 0 {
				return nil, fmt.Errorf("张量 %q 有负维度 %d", name, d)
			}
			if d != 0 && params > math.MaxInt64/d {
				return nil, fmt.Errorf("张量 %q 的形状 %v 元素总数超出 int64 上限", name, e.Shape)
			}
			params *= d
		}

		size := e.DataOffsets[1] - e.DataOffsets[0]

		// data_offsets 的跨度必须与「形状 × 元素字节数」相符。
		// 不符说明头部与数据区对不上（文件损坏或我们理解错了格式），
		// 此时算出来的偏移没有意义，必须报错而不是继续。
		//
		// 取不到字节数时**必须报错**，不能跳过校验：跳过等于让这条防线
		// 在遇到新类型时静默消失，而它消失的方式是无错通过。
		eb, ok := dtype.ByteSize()
		if !ok {
			return nil, fmt.Errorf("张量 %q 的类型 %s 没有每元素字节数，无法校验 data_offsets",
				name, dtype)
		}
		if want := params * eb; size != want {
			return nil, fmt.Errorf(
				"张量 %q 的 data_offsets 跨度 %d 字节，与 %v × %s 推得的 %d 字节不符",
				name, size, e.Shape, dtype, want)
		}

		start := dataStart + e.DataOffsets[0]
		if end := start + size; end > maxEnd {
			maxEnd = end
		}

		m.Tensors = append(m.Tensors, &model.Tensor{
			Name:       name,
			Dims:       e.Shape,
			Dtype:      dtype,
			Offset:     start,
			ByteSize:   size,
			ParamCount: params,
		})
	}

	// 数据区必须完整落在文件内
	if maxEnd > st.Size() {
		return nil, fmt.Errorf("数据区结束于 %d，超过文件大小 %d", maxEnd, st.Size())
	}

	// 数据区不允许重叠：两个张量声称占用同一段字节，
	// 说明头部与数据区对不上，此时各自算出的偏移都是错的。
	// 与 tools/verify_safetensors.py 的参照实现保持同一套校验。
	spans := make([]tensorSpan, 0, len(m.Tensors))
	for _, tn := range m.Tensors {
		spans = append(spans, tensorSpan{tn.Name, tn.Offset, tn.Offset + tn.ByteSize})
	}
	slices.SortFunc(spans, func(a, b tensorSpan) int { return cmp.Compare(a.begin, b.begin) })
	for i := 1; i < len(spans); i++ {
		if spans[i].begin < spans[i-1].end {
			return nil, fmt.Errorf("张量 %q 与 %q 的数据区重叠（%d < %d）",
				spans[i].name, spans[i-1].name, spans[i].begin, spans[i-1].end)
		}
	}
	if len(m.Tensors) > 0 {
		m.StorageBytes = maxEnd - dataStart
	}
	// safetensors 头部不携带架构信息，m.Arch 保持零值（空串）

	return m, nil
}

// tensorSpan 是一个张量在文件里占用的绝对字节区间，用于重叠检测。
type tensorSpan struct {
	name       string
	begin, end int64
}
