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

	// 第一遍：计数 + 极值，同时把有效值收进一个紧凑切片
	valid := make([]float64, 0, len(vals))
	var sum, zeroCount float64
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
		valid = append(valid, f)
	}

	if len(valid) == 0 {
		// 全是 NaN/Inf：极值留零值，直方图留空。Count/NaN/Inf 已经记下了。
		return s
	}

	s.Min, s.Max = minV, maxV
	s.Mean = sum / float64(len(valid))

	// 第二遍：方差与离群计数
	var sq float64
	for _, f := range valid {
		d := f - s.Mean
		sq += d * d
	}
	s.Std = math.Sqrt(sq / float64(len(valid)))

	limitLo := s.Mean - 3*s.Std
	limitHi := s.Mean + 3*s.Std
	var outliers int64
	for _, f := range valid {
		if f < limitLo || f > limitHi {
			outliers++
		}
	}
	s.OutlierRatio = float64(outliers) / float64(len(valid))
	s.ZeroRatio = zeroCount / float64(len(valid))

	fillHistogram(s, valid, buckets)
	return s
}

// fillHistogram 把有效值装进等宽桶。
//
// min == max（常量张量）时不能按 (v-min)/(max-min) 算下标 —— 会除零。
// 这时全部计入中间那一桶。
func fillHistogram(s *model.Stats, valid []float64, buckets int) {
	span := s.Max - s.Min
	if span <= 0 {
		s.Histogram[buckets/2] = int64(len(valid))
		return
	}
	width := span / float64(buckets)
	for _, f := range valid {
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
}

// pickIndices 返回等距采样的元素下标。
//
// 等距而不是随机：量化权重是有序的（同一行的权重同分布、不同行可能差很多），
// 随机采样会引入不可复现的抖动，等距采样对有序数据更稳健，结果也稳定。
//
// 元素数不超过 limit 时返回全部下标 —— 此时不采样，结果就是精确值。
// limit <= 0 同样视为不限制。
func pickIndices(total int, limit int) []int64 {
	if total <= 0 {
		return nil
	}
	if limit <= 0 || total <= limit {
		out := make([]int64, total)
		for i := range out {
			out[i] = int64(i)
		}
		return out
	}
	step := float64(total) / float64(limit)
	out := make([]int64, limit)
	for i := range out {
		// 用乘法而不是累加：累加 float64 会让误差随下标增长，
		// 采到后面可能重复或跳号。
		out[i] = int64(float64(i) * step)
	}
	return out
}
