package analyze

import (
	"math"
	"testing"
)

func TestComputeStats_基本量(t *testing.T) {
	s := computeStats([]float32{1, 2, 3, 4, 5}, histogramBuckets, false)

	if s.Count != 5 {
		t.Errorf("Count = %d, want 5", s.Count)
	}
	if s.Min != 1 || s.Max != 5 {
		t.Errorf("Min/Max = %v/%v, want 1/5", s.Min, s.Max)
	}
	if math.Abs(s.Mean-3) > 1e-6 {
		t.Errorf("Mean = %v, want 3", s.Mean)
	}
	// 总体标准差（除以 n）而不是样本标准差
	if math.Abs(s.Std-math.Sqrt(2)) > 1e-6 {
		t.Errorf("Std = %v, want √2 ≈ %v", s.Std, math.Sqrt(2))
	}
	if s.ZeroRatio != 0 || s.OutlierRatio != 0 {
		t.Errorf("ZeroRatio/OutlierRatio = %v/%v, want 0/0", s.ZeroRatio, s.OutlierRatio)
	}
	if s.NaN != 0 || s.Inf != 0 {
		t.Errorf("NaN/Inf = %d/%d, want 0/0", s.NaN, s.Inf)
	}
	if s.Sampled {
		t.Error("Sampled 应由调用方指定，这里传的是 false")
	}
}

// NaN 与 Inf 必须只计数，不参与 min/max/mean/std —— 否则整条统计全是 NaN，
// 而界面上看起来只是"有点怪"。
func TestComputeStats_NaN与Inf只计数(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	s := computeStats([]float32{1, nan, 3, inf, 5, negInf}, histogramBuckets, false)

	if s.NaN != 1 {
		t.Errorf("NaN = %d, want 1", s.NaN)
	}
	// +Inf 与 -Inf 都算 Inf
	if s.Inf != 2 {
		t.Errorf("Inf = %d, want 2", s.Inf)
	}
	if s.Min != 1 || s.Max != 5 {
		t.Errorf("Min/Max = %v/%v, want 1/5 —— NaN/Inf 不能污染极值", s.Min, s.Max)
	}
	if math.Abs(s.Mean-3) > 1e-6 {
		t.Errorf("Mean = %v, want 3（只有 1,3,5 参与）", s.Mean)
	}
	// Count 是参与统计的总数（含 NaN/Inf），有效值个数是 Count - NaN - Inf
	if s.Count != 6 {
		t.Errorf("Count = %d, want 6", s.Count)
	}
}

func TestComputeStats_零值比例与离群比例(t *testing.T) {
	vals := make([]float32, 100)
	for i := 50; i < 100; i++ {
		vals[i] = float32(i) / 100
	}
	s := computeStats(vals, histogramBuckets, false)

	if math.Abs(s.ZeroRatio-0.5) > 1e-9 {
		t.Errorf("ZeroRatio = %v, want 0.5", s.ZeroRatio)
	}
	if s.OutlierRatio < 0 || s.OutlierRatio > 1 {
		t.Errorf("OutlierRatio = %v，应在 0..1", s.OutlierRatio)
	}
}

// 常量张量（min == max）不能让直方图除零，也不能丢计数。
func TestComputeStats_常量张量(t *testing.T) {
	vals := make([]float32, 100)
	for i := range vals {
		vals[i] = 7
	}
	s := computeStats(vals, histogramBuckets, false)

	if len(s.Histogram) != histogramBuckets {
		t.Fatalf("直方图桶数 = %d, want %d", len(s.Histogram), histogramBuckets)
	}
	var total int64
	for _, c := range s.Histogram {
		total += c
	}
	if total != 100 {
		t.Errorf("直方图总计 = %d, want 100 —— 常量张量也不能丢计数", total)
	}
	if s.Min != 7 || s.Max != 7 {
		t.Errorf("Min/Max = %v/%v, want 7/7", s.Min, s.Max)
	}
	if s.Std != 0 {
		t.Errorf("Std = %v, want 0", s.Std)
	}
	// 必须断言落在**中间那一桶**，不能只断言总数。
	// 去掉 span<=0 的专门分支后，(f-min)/0 得到 NaN，int(NaN) 在 amd64 上
	// 变成极大负数，随后被 idx<0 的兜底挪到 0 桶 —— 总数仍然是 100，
	// 只断言总数的测试发现不了。
	mid := histogramBuckets / 2
	if s.Histogram[mid] != 100 {
		t.Errorf("常量张量应全部落在中间桶 %d，实际该桶 %d，0 桶 %d",
			mid, s.Histogram[mid], s.Histogram[0])
	}
	// 直方图区间就是 [Min, Max]，不再单独存字段
	if s.Min != 7 || s.Max != 7 {
		t.Errorf("直方图区间应为 [7, 7]，实际 [%v, %v]", s.Min, s.Max)
	}
}

func TestComputeStats_空输入(t *testing.T) {
	s := computeStats(nil, histogramBuckets, false)
	if s.Count != 0 {
		t.Errorf("Count = %d, want 0", s.Count)
	}
	if len(s.Histogram) != histogramBuckets {
		t.Errorf("空输入也要有 %d 个桶，实际 %d", histogramBuckets, len(s.Histogram))
	}
}

// 全是 NaN/Inf 时不能崩，也不能给出无意义的极值。
func TestComputeStats_全是无效值(t *testing.T) {
	nan := float32(math.NaN())
	s := computeStats([]float32{nan, nan, nan}, histogramBuckets, false)

	if s.NaN != 3 {
		t.Errorf("NaN = %d, want 3", s.NaN)
	}
	if len(s.Histogram) != histogramBuckets {
		t.Errorf("直方图桶数 = %d", len(s.Histogram))
	}
	var total int64
	for _, c := range s.Histogram {
		total += c
	}
	if total != 0 {
		t.Errorf("没有有效值，直方图应为空，实际总计 %d", total)
	}
}

// 直方图必须覆盖全部有效值 —— 计数总和等于有效值个数。
// 最大值恰好落在右边界时容易越界，这条同时盯着那个钳位。
func TestComputeStats_直方图不丢计数(t *testing.T) {
	vals := make([]float32, 1000)
	for i := range vals {
		vals[i] = float32(i%37) * 0.1
	}
	vals[999] = 1e6 // 一个极远的离群点，逼出右边界情况

	s := computeStats(vals, histogramBuckets, false)
	var total int64
	for _, c := range s.Histogram {
		total += c
	}
	if total != int64(len(vals)) {
		t.Errorf("直方图总计 = %d, want %d", total, len(vals))
	}
	if s.Max != 1e6 {
		t.Errorf("Max = %v, want 1e6", s.Max)
	}
}

// 等距采样：点数与步长必须让采样铺开到首尾。
//
// 原来这条测的是"两次调用结果相同"（可复现）—— 那条对纯函数是
// 恒真的，而 pickIndices 被 sampleStep 取代之后下标不再落地，
// **真正的守卫挪到了 TestDecodeSampled_采样值等于全量对应位置**：
// 那里断言的是取到的值，不是内部下标。
func TestSampleStep_铺开到首尾(t *testing.T) {
	count, step := sampleStep(1000, 100)
	if count != 100 {
		t.Fatalf("点数 = %d, want 100", count)
	}
	if step != 10 {
		t.Fatalf("步长 = %v, want 10", step)
	}
	// 首尾都要取到
	if idx := int64(float64(0) * step); idx != 0 {
		t.Errorf("首个下标 = %d, want 0", idx)
	}
	if idx := int64(float64(count-1) * step); idx < 900 {
		t.Errorf("末个下标 = %d —— 采样没有铺开到尾部", idx)
	}
}

// 元素数不超过上限时不采样：步长 1、点数=总数。
func TestSampleStep_不超限则全取(t *testing.T) {
	count, step := sampleStep(3, 10)
	if count != 3 || step != 1 {
		t.Errorf("sampleStep(3, 10) = (%d, %v), want (3, 1)", count, step)
	}
}

// limit <= 0 视为不限制。
func TestSampleStep_上限为零表示不限制(t *testing.T) {
	if count, step := sampleStep(5, 0); count != 5 || step != 1 {
		t.Errorf("sampleStep(5, 0) = (%d, %v), want (5, 1)", count, step)
	}
}

// 空输入与负输入：点数 0，且步长不能是 0（调用方会拿它做除法）。
func TestSampleStep_空输入(t *testing.T) {
	for _, total := range []int{0, -1} {
		count, step := sampleStep(total, 100)
		if count != 0 {
			t.Errorf("sampleStep(%d, 100) 的点数 = %d, want 0", total, count)
		}
		if step == 0 {
			t.Errorf("sampleStep(%d, 100) 的步长是 0 —— 调用方会除零", total)
		}
	}
}

// 采样下标必须严格递增且不重复 —— 重复会让某些值被计两次。
//
// 下标不再落地，所以这条改成"按公式算出来的下标"来验。
func TestSampleStep_严格递增不重复(t *testing.T) {
	const total, limit = 1_000_000, 1000
	count, step := sampleStep(total, limit)
	if count != limit {
		t.Fatalf("点数 = %d, want %d", count, limit)
	}
	prev := int64(-1)
	for i := 0; i < count; i++ {
		idx := int64(float64(i) * step)
		if idx <= prev {
			t.Fatalf("下标未递增: [%d]=%d <= %d", i, idx, prev)
		}
		prev = idx
	}
	if prev >= total {
		t.Errorf("末个下标 %d 越界（总数 %d）", prev, total)
	}
}
