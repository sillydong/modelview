package analyze

import (
	"context"
	"errors"

	"github.com/sillydong/modelview/internal/decode"
	"github.com/sillydong/modelview/internal/model"
)

// maxReadChunk 是单次读取的字节上限（8 MiB）。
//
// 分块读而不是一次读完：Q6_K 的 token_embd 有 311M 个元素、243 MiB 原始数据，
// 一次读完会在内存里同时存在原始字节和解码结果。
const maxReadChunk = 8 << 20

// DefaultSampleLimit 是默认的采样上限。
//
// 1e7 个 float32 约 40 MB，一次只驻留一个张量的样本；
// 再大就与"交互式浏览"的定位不符了。
//
// **这只是采样缓冲的大小，不是进程峰值内存**。不采样（SampleLimit < 0）
// 时峰值由张量本身决定：实测 qwen2.5:3b（3.08G 参数）跑
// `--stats --sample-limit 0` 峰值 RSS 7–11 GB，而默认采样只有 478 MB。
// 差异来自 out（全部元素 × 4 字节）加 computeStats 里 float64 的 valid 副本。
// 更大的模型按元素数线性外推。
const DefaultSampleLimit = 10_000_000

// Options 控制统计的行为。
type Options struct {
	// SampleLimit 是单个张量参与统计的元素数上限，超过则等距采样。
	//
	// 0 表示用 DefaultSampleLimit —— 零值可用是刻意的：
	// Options{} 不该意味着"把 3 亿个元素全读进内存"（实测 7–11 GB）。
	// 负数表示不设上限（读完整个张量，内存代价见 DefaultSampleLimit 的说明）。
	//
	// **注意 CLI 的 --sample-limit 用的是另一套：0 表示不采样。**
	// 不一致是因为两边"留空"的含义不同：CLI 用户不传这个参数时用的是
	// 默认值，敲 0 想表达的是"别采样"；而库调用方留空 Options 时
	// 不该得到最耗内存的行为。CLI 侧在 runStats 里把 0 翻成 -1。
	SampleLimit int

	// NoCache 为 true 时既不读也不写缓存。
	NoCache bool
}

// Analyze 计算 m 里所有张量的统计量，就地填入 Tensor.Stats。
//
// 单个张量失败**不中断整体**：把它记进返回的 map 并继续，
// 界面上该张量标为"统计失败"，其余照常显示。
// 一个损坏的张量不该让整个模型看不了。
//
// 返回的错误（第二个值）只表示整体性问题：ctx 取消、源打不开。
func Analyze(ctx context.Context, m *model.Model, opts Options) (map[string]error, error) {
	limit := opts.SampleLimit
	if limit == 0 {
		limit = DefaultSampleLimit
	}

	c := newCache(m.Path, opts.NoCache)
	//nolint:errcheck // load 只在读文件，任何问题都静默跳过并重扫
	c.load(m, limit)

	src, err := openSource(m)
	if err != nil {
		return nil, err
	}
	//nolint:errcheck // 只读源，Close 失败不改变已算出的结果
	defer src.Close()

	errs := make(map[string]error)
	for _, tn := range m.Tensors {
		if err := ctx.Err(); err != nil {
			// 取消不是"某个张量失败"，要单独报出来让调用方知道是主动中断
			return errs, err
		}
		if tn.Stats != nil {
			continue // 已经算过（来自缓存或调用方预填）
		}
		if err := analyzeOne(src, tn, limit); err != nil {
			errs[tn.Name] = err
		}
	}

	//nolint:errcheck // 缓存写失败不影响本次结果，只影响下次是否要重扫
	c.save(m, limit)
	return errs, nil
}

// analyzeOne 计算单个张量的统计量。
func analyzeOne(src source, tn *model.Tensor, limit int) error {
	// 非连续张量的数据在存储块里不是线性排列的，按线性顺序解码会得到
	// **元素顺序错乱但看着正常**的分布。宁可不给，也不给错的。
	if tn.NonContiguous {
		return errors.New("张量不是连续布局，无法按线性顺序解码")
	}
	if tn.SizeUnknown {
		return errors.New("字节数未知，无法读取数据")
	}
	if tn.ParamCount == 0 {
		tn.Stats = computeStats(nil, histogramBuckets, false)
		return nil
	}

	perBlock, ok := tn.Dtype.BlockBytes()
	if !ok {
		return decode.ErrUnsupported{Dtype: tn.Dtype}
	}
	elems := tn.Dtype.BlockElems()
	if elems <= 0 {
		elems = 1
	}

	idx := pickIndices(int(tn.ParamCount), limit)
	if len(idx) == 0 {
		tn.Stats = computeStats(nil, histogramBuckets, false)
		return nil
	}

	out := make([]float32, len(idx))
	// decoded 跨块复用：每轮按需扩容，避免每个块都分配一次
	var decoded []float32

	blocksPerChunk := int64(maxReadChunk) / perBlock
	if blocksPerChunk < 1 {
		blocksPerChunk = 1
	}

	for start := 0; start < len(idx); {
		// 本轮的块区间：从第一个采样点所在的块，到 blocksPerChunk 个块之后
		firstBlock := idx[start] / elems
		end := start
		for end < len(idx) && idx[end]/elems-firstBlock < blocksPerChunk {
			end++
		}
		nBlocks := idx[end-1]/elems - firstBlock + 1

		raw, err := src.readRaw(tn, firstBlock*perBlock, nBlocks*perBlock)
		if err != nil {
			return err
		}
		need := int(nBlocks) * int(elems)
		if cap(decoded) < need {
			decoded = make([]float32, need)
		}
		decoded = decoded[:need]
		if err := decode.Decode(tn.Dtype, raw, decoded); err != nil {
			return err
		}

		for ; start < end; start++ {
			out[start] = decoded[idx[start]-firstBlock*elems]
		}
	}

	tn.Stats = computeStats(out, histogramBuckets, len(idx) < int(tn.ParamCount))
	return nil
}
