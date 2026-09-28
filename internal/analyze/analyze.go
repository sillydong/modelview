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
// `--stats --sample-limit 0` 峰值 RSS 7–11 GB。
// 差异来自 out（全部元素 × 4 字节）加 computeStats 里 float64 的 valid 副本。
// 更大的模型按元素数线性外推。
//
// **它也管不到块级诊断**。量化张量的块级诊断按设计要读遍全部块头
// （「哪个子块被压得最狠」抽样会把这个结论变得不可信），
// 所以 --sample-limit 对它没有约束力。实测 qwen2.5:3b：
//
//	--sample-limit 1   块头扫描仍然跑满 → 峰值 116 MB / 9.1 s
//
// 代价是 O(1) 内存（流式聚合，见 quantAgg），但时间与张量总大小成正比。
// 内存峰值本身与采样上限无关：默认采样下实测 435 MB，
// 而不做块级诊断的版本是 477 MB。
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
		if !NeedsWork(tn) {
			continue // 本次输出需要的东西都在（来自缓存或调用方预填）
		}
		if err := analyzeOne(src, tn, limit); err != nil {
			errs[tn.Name] = err
		}
	}

	//nolint:errcheck // 缓存写失败不影响本次结果，只影响下次是否要重扫
	c.save(m, limit)
	return errs, nil
}

// NeedsWork 判断这个张量还有没有要算的。
//
// 导出是给界面用的：详情页要按同一个判据决定"要不要显示扫描中" ——
// 两边各写一份的话，对"什么时候该重扫"的理解迟早漂移。
//
// **不能只看 Stats != nil**：那会把「统计算过了」当成「分析做完了」。
// 量化分析是后加的，旧缓存里没有 —— 用 Stats 当判据时同一个文件
// 第二次跑会整块跳过量化分析，输出与第一次不同且不报错。
// 实测：冷跑 181 个模拟 / 253 个诊断，热跑 0 / 0。
func NeedsWork(tn *model.Tensor) bool {
	if tn.Stats == nil {
		return true
	}
	if tn.Dtype.IsQuantized() {
		// 只有**真能算出诊断**的类型才等它。未收录块头布局的类型
		// 永远算不出来，拿 Quant == nil 当判据会让它每次运行都重扫
		// 整个张量的数据。
		return decode.ScalesSupported(tn.Dtype) && tn.Quant == nil
	}
	if tn.Dtype.IsFloat() {
		return len(tn.QuantSims) == 0
	}
	return false
}

// analyzeOne 计算单个张量缺的那部分，并补上量化分析。
func analyzeOne(src source, tn *model.Tensor, limit int) error {
	// 非连续张量的数据在存储块里不是线性排列的，按线性顺序解码会得到
	// **元素顺序错乱但看着正常**的分布。宁可不给，也不给错的。
	if tn.NonContiguous {
		return errors.New("张量不是连续布局，无法按线性顺序解码")
	}
	if tn.SizeUnknown {
		return errors.New("字节数未知，无法读取数据")
	}

	// 需要逐值解码的只有两件事：统计、以及浮点张量的量化模拟。
	// 量化张量的块级诊断只读块头，不解码权重。
	needVals := tn.Stats == nil || (tn.Dtype.IsFloat() && len(tn.QuantSims) == 0)
	if needVals {
		vals, err := decodeSampled(src, tn, limit)
		if err != nil {
			return err
		}
		if tn.Stats == nil {
			tn.Stats = computeStats(vals, histogramBuckets, len(vals) < int(tn.ParamCount))
		}
		if tn.Dtype.IsFloat() && tn.Dtype.BitsPerWeight() > 0 && len(vals) > 0 {
			sims := simulateAll(vals, tn.Dtype.BitsPerWeight())
			sampled := len(vals) < int(tn.ParamCount)
			for i := range sims {
				sims[i].Sampled = sampled
			}
			tn.QuantSims = sims
		}
	}

	if tn.Dtype.IsQuantized() && tn.Quant == nil {
		q, err := aggregateScales(src, tn)
		// 未收录块头布局的类型（Q8_1、IQ 系列）算不出诊断。这不是错误 ——
		// 报出去会被 CLI 说成"统计失败"并给出"解码未实现"，
		// 而统计就在同一行显示着，真实故障反而被淹没。
		if errors.As(err, &decode.ErrUnsupported{}) {
			return nil
		}
		if err != nil {
			return err
		}
		if q.SubBlocks > 0 {
			tn.Quant = &q
		}
	}
	return nil
}

// decodeSampled 按等距采样取出 tn 的数值。
//
// analyzeOne 与量化模拟共用它 —— 两处各写一遍分块逻辑，
// 迟早会漂移成「统计看到的样本」与「模拟看到的样本」不是同一批。
func decodeSampled(src source, tn *model.Tensor, limit int) ([]float32, error) {
	if tn.ParamCount == 0 {
		return nil, nil
	}
	perBlock, ok := tn.Dtype.BlockBytes()
	if !ok {
		return nil, decode.ErrUnsupported{Dtype: tn.Dtype}
	}
	elems := tn.Dtype.BlockElems()
	if elems <= 0 {
		elems = 1
	}

	idx := pickIndices(int(tn.ParamCount), limit)
	if len(idx) == 0 {
		return nil, nil
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
			return nil, err
		}
		need := int(nBlocks) * int(elems)
		if cap(decoded) < need {
			decoded = make([]float32, need)
		}
		decoded = decoded[:need]
		if err := decode.Decode(tn.Dtype, raw, decoded); err != nil {
			return nil, err
		}

		for ; start < end; start++ {
			out[start] = decoded[idx[start]-firstBlock*elems]
		}
	}
	return out, nil
}

// aggregateScales 分块读遍整个张量，只提取块头的 scale/min，边读边聚合。
//
// **不把全部子块留下**：3.11 亿权重的 Q6_K 有 1944 万个子块，
// 全留下是 311 MB，而 --sample-limit 对这条路径没有任何约束力 ——
// 实测 --sample-limit 1 时峰值内存 806 MB，而不做这个分析的 main 只有 41 MB。
// 聚合后峰值只与分块大小有关。
func aggregateScales(src source, tn *model.Tensor) (model.QuantInfo, error) {
	d := tn.Dtype
	perBlock, ok := d.BlockBytes()
	if !ok {
		return model.QuantInfo{}, decode.ErrUnsupported{Dtype: d}
	}
	blocksPerChunk := int64(maxReadChunk) / perBlock
	if blocksPerChunk < 1 {
		blocksPerChunk = 1
	}
	totalBlocks := tn.ByteSize / perBlock
	if totalBlocks == 0 {
		return model.QuantInfo{}, nil
	}

	// 先问一个块「每块几个子块」：中位数的抽样步长要知道总量，
	// 而总量 = 块数 × 每块子块数。读一个块（144~210 字节）比事后补救便宜得多
	probe, err := src.readRaw(tn, 0, perBlock)
	if err != nil {
		return model.QuantInfo{}, err
	}
	probeSubs, err := decode.Scales(d, probe)
	if err != nil {
		return model.QuantInfo{}, err
	}
	if len(probeSubs) == 0 {
		return model.QuantInfo{}, nil
	}

	agg := newQuantAgg(d, totalBlocks*int64(len(probeSubs)))
	for first := int64(0); first < totalBlocks; first += blocksPerChunk {
		n := min(blocksPerChunk, totalBlocks-first)
		raw, err := src.readRaw(tn, first*perBlock, n*perBlock)
		if err != nil {
			return model.QuantInfo{}, err
		}
		subs, err := decode.Scales(d, raw)
		if err != nil {
			return model.QuantInfo{}, err
		}
		agg.add(subs)
	}
	return agg.finish(), nil
}
