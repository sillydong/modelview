package analyze

import (
	"math"

	"github.com/sillydong/modelview/internal/model"
)

// histogramBuckets 是直方图的桶数。
//
// 64 桶在 80 列的终端里两个字符一桶正好铺满，且足够看出双峰。
const histogramBuckets = 64

// computeStats 计算一批数值的统计量。
//
// 前提：vals 里可能有 NaN / Inf（量化模型里少见但存在），
// 它们只被计数，**不参与** min/max/mean/std —— 否则整条统计会变成 NaN，
// 而界面上看起来只是"有点怪"。
//
// Count 是这批值的总数（含 NaN/Inf）；ZeroRatio 与 OutlierRatio 的分母
// 则是**有效值**个数 —— 一个 NaN 不是零，也不是离群点。
func computeStats(vals []float32, buckets int, sampled bool) *model.Stats {
	s := &model.Stats{
		Count:     int64(len(vals)),
		Histogram: make([]int64, buckets),
		Sampled:   sampled,
	}
	if len(vals) == 0 {
		return s
	}

	// **三遍直接扫 vals，不复制一份 []float64 的"有效值"**。
	// 那份副本是 8 B/元素：默认 1e7 采样时 76 MiB，--sample-limit 0 时
	// 与张量元素数同阶（3.11 亿元素的张量上 2.32 GB）。而它只是一份
	// 过滤后的视图，过滤条件（非 NaN/非 Inf）每一遍现判即可。
	//
	// 代价是累加顺序与"先过滤再累加"不同，浮点求和可能有末位差异 ——
	// 所以有一条独立实现的对照测试盯着（TestComputeStats_与独立实现
	// 逐值一致，容差 1e-12 相对值），数值口径不变。

	// 第一遍：计数 + 极值 + 零计数 + 和
	var sum, zeroCount float64
	var n int64
	minV, maxV := math.Inf(1), math.Inf(-1)
	for _, v := range vals {
		f := float64(v)
		switch {
		case math.IsNaN(f):
			s.NaN++
			continue
		case math.IsInf(f, 0):
			s.Inf++
			continue
		}
		n++
		if f == 0 {
			zeroCount++
		}
		if f < minV {
			minV = f
		}
		if f > maxV {
			maxV = f
		}
		sum += f
	}

	if n == 0 {
		// 全是 NaN/Inf：极值留零值，直方图留空。Count/NaN/Inf 已经记下了。
		return s
	}

	s.Min, s.Max = minV, maxV
	s.Mean = sum / float64(n)

	// 第二遍：方差
	var sq float64
	for _, v := range vals {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			continue
		}
		d := f - s.Mean
		sq += d * d
	}
	s.Std = math.Sqrt(sq / float64(n))

	// 第三遍：离群计数 + 直方图
	limitLo := s.Mean - 3*s.Std
	limitHi := s.Mean + 3*s.Std
	var outliers int64
	span := s.Max - s.Min
	width := span / float64(buckets)
	for _, v := range vals {
		f := float64(v)
		if math.IsNaN(f) || math.IsInf(f, 0) {
			continue
		}
		if f < limitLo || f > limitHi {
			outliers++
		}
		if span <= 0 {
			continue
		}
		idx := int((f - s.Min) / width)
		// 最大值恰好落在右边界，算出来是 buckets，必须钳回最后一桶，
		// 否则越界 panic —— 或者更糟，静默丢掉这个计数。
		if idx >= buckets {
			idx = buckets - 1
		}
		// 不需要判 idx < 0：width > 0（span > 0）且 f >= Min，
		// 所以 (f-Min)/width 恒 >= 0。加了也没测试能覆盖它。
		s.Histogram[idx]++
	}
	if span <= 0 {
		// min == max（常量张量）：不能按 (v-min)/(max-min) 算下标 —— 会除零。
		// 全部计入中间那一桶。
		//
		// 放在循环**之后**覆盖而不是之前早退：那样要重复一遍计数，
		// 而这里只需要知道总数 n。
		s.Histogram[buckets/2] = n
	}
	s.OutlierRatio = float64(outliers) / float64(n)
	s.ZeroRatio = zeroCount / float64(n)
	return s
}

// sampleStep 返回等距采样的**点数与步长**，不返回下标切片。
//
// 等距而不是随机：量化权重是有序的（同一行的权重同分布、不同行可能差很多），
// 随机采样会引入不可复现的抖动，等距采样对有序数据更稳健，结果也稳定。
//
// **不落地下标是内存优化**：`[]int64` 是 8 B/元素，默认 1e7 采样时占
// 76 MiB；不采样（--sample-limit 0）时等于张量元素数 × 8 —— 一个
// 3.11 亿元素的张量上光这一项就是 2.32 GB。而下标是等差序列
// `int64(i*step)`，随时可以算出来，没有理由驻留。
//
// **步长必须保持 float64**：原实现就是 `total/limit` 的浮点除法，
// 换成整数除法会取到不同的元素（total=1009、limit=100 时步长
// 10.09 vs 10，取到的位置就此分叉）。这是纯内存优化，数值一个字
// 都不许变 —— TestDecodeSampled_采样值等于全量对应位置 钉住这一点。
//
// 元素数不超过 limit 时 step=1、count=total，即不采样。
// limit <= 0 同样视为不限制。
func sampleStep(total, limit int) (count int, step float64) {
	if total <= 0 {
		return 0, 1
	}
	if limit <= 0 || total <= limit {
		return total, 1
	}
	return limit, float64(total) / float64(limit)
}
