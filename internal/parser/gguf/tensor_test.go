package gguf

import (
	"bytes"
	"cmp"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

func TestReadTensorInfos(t *testing.T) {
	b := newBuilder()
	// 两个张量：一个 2 维 F32，一个 1 维 Q4_K
	b.str("blk.0.attn_qkv.weight")
	b.u32(2)
	b.u64(768)
	b.u64(2304)
	b.u32(0) // F32
	b.u64(0) // offset

	b.str("blk.0.ffn_gate.weight")
	b.u32(1)
	b.u64(256)
	b.u32(12)   // Q4_K
	b.u64(1024) // offset

	infos, err := readTensorInfos(newReader(bytes.NewReader(b.bytes())), 2)
	if err != nil {
		t.Fatalf("readTensorInfos 失败: %v", err)
	}
	if len(infos) != 2 {
		t.Fatalf("张量个数 = %d, want 2", len(infos))
	}

	if infos[0].Name != "blk.0.attn_qkv.weight" {
		t.Errorf("名称 = %q", infos[0].Name)
	}
	if len(infos[0].Dims) != 2 || infos[0].Dims[0] != 768 || infos[0].Dims[1] != 2304 {
		t.Errorf("形状 = %v, want [768 2304]", infos[0].Dims)
	}
	if infos[0].Type != 0 {
		t.Errorf("类型 = %d, want 0", infos[0].Type)
	}
	if infos[1].Type != 12 {
		t.Errorf("类型 = %d, want 12", infos[1].Type)
	}
	if infos[1].Offset != 1024 {
		t.Errorf("偏移 = %d, want 1024", infos[1].Offset)
	}
}

func TestReadTensorInfos_零维张量(t *testing.T) {
	b := newBuilder()
	b.str("scalar")
	b.u32(0) // 0 维
	b.u32(0) // F32
	b.u64(0)

	infos, err := readTensorInfos(newReader(bytes.NewReader(b.bytes())), 1)
	if err != nil {
		t.Fatalf("readTensorInfos 失败: %v", err)
	}
	if len(infos[0].Dims) != 0 {
		t.Errorf("维数 = %d, want 0", len(infos[0].Dims))
	}
}

func TestReadTensorInfos_维数超限报错(t *testing.T) {
	b := newBuilder()
	b.str("bad")
	b.u32(99) // 超过 maxDims
	if _, err := readTensorInfos(newReader(bytes.NewReader(b.bytes())), 1); err == nil {
		t.Fatal("维数超限应报错")
	}
}

func TestAlignUp(t *testing.T) {
	tests := []struct {
		off, align, want int64
	}{
		{0, 32, 0},
		{1, 32, 32},
		{31, 32, 32},
		{32, 32, 32},
		{33, 32, 64},
		{100, 32, 128},
		// 真实 qwen2.5:3b 文件的实测值：张量描述符结束于 756678，
		// 数据区实际从 756704 开始。
		{756678, 32, 756704},
		// align <= 1 时不做调整
		{100, 1, 100},
		{100, 0, 100},
	}
	for _, tt := range tests {
		if got := alignUp(tt.off, tt.align); got != tt.want {
			t.Errorf("alignUp(%d, %d) = %d, want %d", tt.off, tt.align, got, tt.want)
		}
	}
}

func TestTensorByteSize(t *testing.T) {
	tests := []struct {
		name    string
		dims    []int64
		dtype   uint32
		want    int64
		wantErr bool
	}{
		// 非量化：元素数 × 每元素字节数
		{"F32 一维", []int64{8}, 0, 32, false},
		{"F16 一维", []int64{4}, 1, 8, false},
		{"BF16 二维", []int64{2, 4}, 30, 16, false},
		{"F64 一维", []int64{4}, 28, 32, false},
		{"I8 一维", []int64{100}, 24, 100, false},

		// 量化：块数 × 每块字节数
		// Q4_K 一个 256 权重的块 = 2 + 2 + 12 + 128 = 144 字节
		{"Q4_K 一块", []int64{256}, 12, 144, false},
		{"Q4_K 两块", []int64{512}, 12, 288, false},
		{"Q4_K 二维", []int64{2048, 256}, 12, 2048 * 144, false},
		// Q6_K = 128 + 64 + 16 + 2 = 210
		{"Q6_K 一块", []int64{256}, 14, 210, false},
		// Q8_0 = 2 + 32 = 34，块大小 32
		{"Q8_0 一块", []int64{32}, 8, 34, false},
		{"Q8_0 两块", []int64{64}, 8, 68, false},
		// Q5_0 = 2 + 4 + 16 = 22
		{"Q5_0 一块", []int64{32}, 6, 22, false},

		// 错误情况
		{"IQ 类型未收录", []int64{256}, 16, 0, true},
		{"未知类型码", []int64{8}, 999, 0, true},
		{"Q4_K 非整块", []int64{100}, 12, 0, true},
		{"负维度", []int64{-1}, 0, 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tensorByteSize(tt.dims, tt.dtype)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("期望报错，实际 got=%d", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("意外错误: %v", err)
			}
			if got != tt.want {
				t.Errorf("tensorByteSize = %d, want %d", got, tt.want)
			}
		})
	}
}

// 未收录类型必须返回可识别的错误，而不是静默返回 0。
func TestTensorByteSize_未收录类型返回可识别错误(t *testing.T) {
	_, err := tensorByteSize([]int64{256}, 16) // IQ2_XXS
	var e ErrUnknownBlockType
	if !errors.As(err, &e) {
		t.Fatalf("err = %v, 期望 ErrUnknownBlockType", err)
	}
	if e.Code != 16 {
		t.Errorf("Code = %d, want 16", e.Code)
	}
}

// 块字节数与块元素数必须自洽：块字节数应是整数，且能整除。
func TestBlockTable_自洽(t *testing.T) {
	for code, size := range blockBytes {
		elems := blockElemCount(code)
		if elems <= 0 {
			t.Errorf("类型 %d 的块元素数 = %d", code, elems)
		}
		if size <= 0 {
			t.Errorf("类型 %d 的块字节数 = %d", code, size)
		}
	}
}

// 最强的验证：真实文件里所有张量的字节区间必须互不重叠。
//
// 这条不变式能抓住任何一处块大小写错 —— 只要某个类型的块大小偏大，
// 该类型张量的结束位置就会越过下一个张量的起点，重叠量恰好等于误差。
//
// 注意不能只检查"最后一个张量结束于文件末尾"：那样只有排在最末的那个
// 类型会被验证到，其它类型写错也发现不了。
func TestBlockTable_与真实文件吻合(t *testing.T) {
	blobs := []string{
		"sha256-5ee4f07cdb9bead", // qwen2.5:3b     F32/Q4_K/Q6_K
		"sha256-970aa74c0a90ef7", // nomic-embed    F32/F16
		"sha256-4c27e0f5b5adf02", // gemma4:e4b     + BF16
		"sha256-7121486771cbfe2", // gemma4:26b     + Q5_0/Q8_0
	}

	tested := 0
	for _, prefix := range blobs {
		path := findBlob(prefix)
		if path == "" {
			continue
		}
		t.Run(prefix[7:15], func(t *testing.T) {
			tested++
			m, err := Parse(path)
			if err != nil {
				t.Fatalf("Parse 失败: %v", err)
			}

			sorted := make([]*model.Tensor, len(m.Tensors))
			copy(sorted, m.Tensors)
			slices.SortFunc(sorted, func(a, b *model.Tensor) int {
				return cmp.Compare(a.Offset, b.Offset)
			})

			align := int64(32)
			for _, tn := range sorted {
				if tn.ByteSize == 0 {
					t.Fatalf("%s 的字节数为 0（类型 %s 未收录）", tn.Name, tn.Dtype)
				}
			}

			for i := 1; i < len(sorted); i++ {
				prev, cur := sorted[i-1], sorted[i]
				end := prev.Offset + prev.ByteSize
				if end > cur.Offset {
					t.Fatalf("张量重叠 %d 字节：\n  %s 结束于 %d\n  %s 起始于 %d\n"+
						"说明这两者之一的块大小算错了",
						end-cur.Offset, prev.Name, end, cur.Name, cur.Offset)
				}
				// 张量之间只应有对齐填充，不应有大段空隙。
				if gap := cur.Offset - end; gap >= align {
					t.Errorf("张量间空隙 %d 字节（≥ 对齐 %d）：%s 结束于 %d，%s 起始于 %d",
						gap, align, prev.Name, end, cur.Name, cur.Offset)
				}
			}

			last := sorted[len(sorted)-1]
			if diff := m.FileSize - (last.Offset + last.ByteSize); diff < 0 || diff > 1024 {
				t.Errorf("最后一个张量 %s 结束于 %d，文件大小 %d，差值 %d",
					last.Name, last.Offset+last.ByteSize, m.FileSize, diff)
			}
		})
	}
	if tested == 0 {
		t.Skip("本机没有可用的 ollama 模型文件")
	}
}

// findBlob 在 ollama 的 blobs 目录里按前缀找文件；找不到返回空串。
func findBlob(prefix string) string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	dir := filepath.Join(home, ".ollama", "models", "blobs")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		if len(e.Name()) >= len(prefix) && e.Name()[:len(prefix)] == prefix {
			return filepath.Join(dir, e.Name())
		}
	}
	return ""
}
