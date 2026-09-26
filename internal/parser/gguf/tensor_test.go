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
	_, err := readTensorInfos(newReader(bytes.NewReader(b.bytes())), 1)
	// 必须断言是哪个错误：把守卫改成永假之后，错误会改由
	// "读到第 3 维时 EOF"产生 —— 只断言 err != nil 的测试照样通过。
	if !errors.Is(err, ErrTooManyDims) {
		t.Fatalf("err = %v, 期望 ErrTooManyDims", err)
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

// 覆盖块表里的**每一个**类型码。
//
// 期望值独立于实现：直接来自 llama.cpp 的 block_* 结构体定义，算式写在注释里。
// 不从 blockBytes 反算 —— 否则就是同义反复，表写错了测试跟着一起错。
//
// 注意：下面这张表的存在意义是"换台机器也能验证"。真实文件回归测试
// 依赖 ~/.ollama 目录，在 CI 或别人机器上会整个 SKIP。
func TestTensorByteSize_覆盖全部类型码(t *testing.T) {
	const (
		qk4 = 32  // QK4_0 / QK5_0 / QK8_0 系列块大小
		qkK = 256 // QK_K 系列块大小
	)

	tests := []struct {
		code uint32
		name string
		// dtype 是期望映射到的类型
		dtype model.Dtype
		// perUnit 是「一个单位」占用的字节数：
		// 非量化类型 = 单个元素；量化类型 = 一个块。
		perUnit int64
		// unit 是「一个单位」包含多少个权重：
		// 非量化类型 = 1；量化类型 = 块大小。
		unit int64
	}{
		// 非量化：perUnit = 每元素字节数，unit = 1
		{0, "F32", model.DtypeF32, 4, 1},
		{1, "F16", model.DtypeF16, 2, 1},
		{28, "F64", model.DtypeF64, 8, 1},
		{30, "BF16", model.DtypeBF16, 2, 1},
		{24, "I8", model.DtypeI8, 1, 1},
		{25, "I16", model.DtypeI16, 2, 1},
		{26, "I32", model.DtypeI32, 4, 1},
		{27, "I64", model.DtypeI64, 8, 1},

		// 量化（块 32）：2(d) + [scale/min/qh] + qs
		{2, "Q4_0", model.DtypeQ4_0, 2 + 16, qk4},
		{3, "Q4_1", model.DtypeQ4_1, 2 + 2 + 16, qk4},
		{6, "Q5_0", model.DtypeQ5_0, 2 + 4 + 16, qk4},
		{7, "Q5_1", model.DtypeQ5_1, 2 + 2 + 4 + 16, qk4},
		{8, "Q8_0", model.DtypeQ8_0, 2 + 32, qk4},
		{9, "Q8_1", model.DtypeQ8_1, 2 + 2 + 32, qk4},

		// 量化（块 256）
		{10, "Q2_K", model.DtypeQ2K, 16 + 64 + 2 + 2, qkK},
		{11, "Q3_K", model.DtypeQ3K, 32 + 64 + 12 + 2, qkK},
		{12, "Q4_K", model.DtypeQ4K, 2 + 2 + 12 + 128, qkK},
		{13, "Q5_K", model.DtypeQ5K, 2 + 2 + 12 + 32 + 128, qkK},
		{14, "Q6_K", model.DtypeQ6K, 128 + 64 + 16 + 2, qkK},
		{15, "Q8_K", model.DtypeQ8K, 4 + 256 + 32, qkK},
	}

	// IQ 系列（i-quants）：块结构未收录，算不出字节数。
	// 但**类型码到名字的映射**必须准确 —— 这是这些张量在界面上唯一的输出。
	// perUnit/unit 为 0 表示"不该算得出来"。
	iqCodes := []struct {
		code  uint32
		dtype model.Dtype
	}{
		{16, model.DtypeIQ2XXS}, {17, model.DtypeIQ2XS}, {18, model.DtypeIQ3XXS},
		{19, model.DtypeIQ1S}, {20, model.DtypeIQ4NL}, {21, model.DtypeIQ3S},
		{22, model.DtypeIQ2S}, {23, model.DtypeIQ4XS}, {29, model.DtypeIQ1M},
	}
	for _, iq := range iqCodes {
		got, ok := ggmlDtype(iq.code)
		if !ok {
			t.Errorf("IQ 类型码 %d 未识别", iq.code)
			continue
		}
		if got != iq.dtype {
			t.Errorf("ggmlDtype(%d) = %q, want %q", iq.code, got, iq.dtype)
		}
		if _, err := tensorByteSize([]int64{256}, iq.code); err == nil {
			t.Errorf("IQ 类型码 %d 不该算得出字节数（块结构未收录）", iq.code)
		}
	}

	// 这张表必须覆盖块表的全部条目 —— 新增类型时同步补进来。
	if len(tests) != len(blockBytes) {
		t.Errorf("本表覆盖 %d 个类型，块表有 %d 个 —— 有类型没被验证到",
			len(tests), len(blockBytes))
	}

	// want 按元素数算出期望字节数。
	want := func(perUnit, unit, elems int64) int64 {
		return (elems / unit) * perUnit
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 元素数取块大小的整数倍，量化与非量化都能用同一个算式
			for _, mult := range []int64{1, 3} {
				elems := tt.unit * mult
				got, err := tensorByteSize([]int64{elems}, tt.code)
				if err != nil {
					t.Fatalf("tensorByteSize(%d 个元素) 失败: %v", elems, err)
				}
				if w := want(tt.perUnit, tt.unit, elems); got != w {
					t.Errorf("%d 个元素 → %d 字节, want %d", elems, got, w)
				}
			}

			// 类型码必须映射到预期的 Dtype
			dt, ok := ggmlDtype(tt.code)
			if !ok {
				t.Fatalf("类型码 %d 未识别", tt.code)
			}
			if dt != tt.dtype {
				t.Errorf("ggmlDtype(%d) = %q, want %q", tt.code, dt, tt.dtype)
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

// 用真实文件验证张量的字节区间互不重叠。
//
// 只要某个类型的块大小偏大，该类型张量的结束位置就会越过下一个张量的起点，
// 重叠量恰好等于误差；偏小则表现为空隙。两个方向都杀得死。
//
// **覆盖边界**：这条不变式只能验证**这四个文件里实际出现的类型**
// （F32/F16/BF16/Q4_K/Q5_0/Q6_K/Q8_0 共 7 项）。其余 13 项
// （Q4_0/Q4_1/Q5_1/Q8_1/Q2_K/Q3_K/Q5_K/Q8_K/I8/I16/I32/I64/F64）
// 没有任何真实文件覆盖，只能靠 TestTensorByteSize_覆盖全部类型码
// 逐值钉住 —— 那张表的期望值来自结构体定义，是第三处手抄，
// 三处一致地写错仍然拦不住。
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

	covered := map[model.Dtype]bool{}
	for _, prefix := range blobs {
		path := findBlob(prefix)
		if path == "" {
			// 不能静默 continue：少一个文件，测试照样全绿，
			// 该文件覆盖的类型（如 Q5_0/Q8_0）就凭空消失了。
			t.Errorf("找不到 %s*，该文件覆盖的类型未被验证", prefix)
			continue
		}
		t.Run(prefix[7:15], func(t *testing.T) {
			m, err := Parse(path)
			if err != nil {
				t.Fatalf("Parse 失败: %v", err)
			}

			sorted := make([]*model.Tensor, len(m.Tensors))
			copy(sorted, m.Tensors)
			slices.SortFunc(sorted, func(a, b *model.Tensor) int {
				return cmp.Compare(a.Offset, b.Offset)
			})

			for _, tn := range sorted {
				covered[tn.Dtype] = true
			}

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

			// 实测四个文件的差值**恰好为 0**（GGUF 数据区紧排到文件末尾）。
			// 这里不放宽容差：1024 字节的口子意味着末尾张量的块大小偏小
			// 一千字节以内永远抓不到，而注释声称的是"差值 0"。
			last := sorted[len(sorted)-1]
			if end := last.Offset + last.ByteSize; end != m.FileSize {
				t.Errorf("最后一个张量 %s 结束于 %d，文件大小 %d，差值 %d",
					last.Name, end, m.FileSize, m.FileSize-end)
			}
		})
	}
	// 真实文件能覆盖到的类型集合必须至少包含这几个 ——
	// 块表里其余项由 TestTensorByteSize_覆盖全部类型码 兜底。
	want := []model.Dtype{
		model.DtypeF32, model.DtypeF16, model.DtypeBF16,
		model.DtypeQ4K, model.DtypeQ5_0, model.DtypeQ6K, model.DtypeQ8_0,
	}
	var missing []string
	for _, d := range want {
		if !covered[d] {
			missing = append(missing, string(d))
		}
	}
	if len(missing) > 0 {
		t.Errorf("真实文件未覆盖到预期类型 %v —— 覆盖范围缩水了", missing)
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
