package analyze

import (
	"math"
	"testing"

	"github.com/sillydong/modelview/internal/decode"
	"github.com/sillydong/modelview/internal/model"
)

// 三档的误差必须随压缩加重单调增大，信噪比单调下降。
func TestSimulate_三档单调(t *testing.T) {
	vals := make([]float32, 256)
	for i := range vals {
		vals[i] = float32(math.Sin(float64(i)*0.07)) * 0.2
	}
	sims := simulateAll(vals, 16) // 源是 F16
	if len(sims) != 3 {
		t.Fatalf("档数 = %d, want 3", len(sims))
	}
	want := []string{"Q8_0", "Q6_K", "Q4_K"}
	for i, w := range want {
		if sims[i].Target != w {
			t.Errorf("档位[%d] = %q, want %q", i, sims[i].Target, w)
		}
	}
	for i := 1; i < len(sims); i++ {
		if sims[i].MaxAbsErr <= sims[i-1].MaxAbsErr {
			t.Errorf("%s 的最大误差 %v 应大于 %s 的 %v",
				sims[i].Target, sims[i].MaxAbsErr, sims[i-1].Target, sims[i-1].MaxAbsErr)
		}
		if sims[i].SNRDB >= sims[i-1].SNRDB {
			t.Errorf("%s 的信噪比 %v 应低于 %s 的 %v",
				sims[i].Target, sims[i].SNRDB, sims[i-1].Target, sims[i-1].SNRDB)
		}
		if sims[i].Compression <= sims[i-1].Compression {
			t.Errorf("%s 的压缩比 %v 应大于 %s 的 %v",
				sims[i].Target, sims[i].Compression, sims[i-1].Target, sims[i-1].Compression)
		}
	}
	// 位宽必须来自类型表，不是猜的
	if sims[0].BitsPerWeight != model.DtypeQ8_0.BitsPerWeight() {
		t.Errorf("Q8_0 位宽 = %v, want %v", sims[0].BitsPerWeight, model.DtypeQ8_0.BitsPerWeight())
	}
	// 压缩比必须真的由源位宽算出来 —— 传 16（F16）时 Q8_0 应是 16/8.5
	if want := 16 / model.DtypeQ8_0.BitsPerWeight(); sims[0].Compression != want {
		t.Errorf("Q8_0 压缩比 = %v, want %v", sims[0].Compression, want)
	}
	// 用不同源位宽再算一次，确认这个字段真的随参数变
	if got := simulateAll(vals, 32)[0].Compression; got <= sims[0].Compression {
		t.Errorf("源位宽 32 的压缩比 %v 应大于源位宽 16 的 %v", got, sims[0].Compression)
	}
}

// 模拟必须走真实编码器 —— 用它的误差与直接调 encode/decode 的结果对齐。
//
// 这条是「模拟没有偷偷换成别的算法」的门禁：如果哪天有人把 simulateTarget
// 改成统一公式，这里的数字会对不上。
func TestSimulate_走的是真实编码器(t *testing.T) {
	vals := make([]float32, 256)
	for i := range vals {
		vals[i] = float32(i)/128 - 1
	}
	sims := simulateAll(vals, 16)
	for i, d := range []model.Dtype{model.DtypeQ8_0, model.DtypeQ6K, model.DtypeQ4K} {
		perBlock, _ := d.BlockBytes()
		n := int64(len(vals)) / d.BlockElems()
		buf := make([]byte, n*perBlock)
		if err := decode.Quantize(d, vals, buf); err != nil {
			t.Fatalf("%s Quantize 失败: %v", d, err)
		}
		back := make([]float32, len(vals))
		if err := decode.Decode(d, buf, back); err != nil {
			t.Fatalf("%s Decode 失败: %v", d, err)
		}
		var maxErr, sumAbs float64
		for j := range vals {
			e := math.Abs(float64(vals[j] - back[j]))
			maxErr = math.Max(maxErr, e)
			sumAbs += e
		}
		if sims[i].MaxAbsErr != maxErr {
			t.Errorf("%s: 模拟的最大误差 %v 与直接编解码的 %v 不一致 —— "+
				"模拟没有走真实编码器", d, sims[i].MaxAbsErr, maxErr)
		}
		if want := sumAbs / float64(len(vals)); sims[i].MeanAbsErr != want {
			t.Errorf("%s: 模拟的平均误差 %v 与直接编解码的 %v 不一致",
				d, sims[i].MeanAbsErr, want)
		}
	}
}

// 全零张量不能产生 NaN。
func TestSimulate_全零(t *testing.T) {
	for _, s := range simulateAll(make([]float32, 256), 16) {
		if math.IsNaN(s.MaxAbsErr) || math.IsNaN(s.SNRDB) || math.IsNaN(s.MeanAbsErr) {
			t.Errorf("%s 产生 NaN: %+v", s.Target, s)
		}
		if s.MaxAbsErr != 0 {
			t.Errorf("%s 全零输入误差应为 0，实际 %v", s.Target, s.MaxAbsErr)
		}
		if s.SNRDB != 0 {
			t.Errorf("%s 全零输入没有信号也没有噪声，信噪比应记 0 而不是算出来，实际 %v",
				s.Target, s.SNRDB)
		}
	}
}

// 相对误差的分母接近 0 时不能变成 Inf。
func TestSimulate_微小值不炸相对误差(t *testing.T) {
	vals := append(make([]float32, 128), 1e-30)
	vals = append(vals, make([]float32, 127)...)
	for _, s := range simulateAll(vals, 16) {
		if math.IsInf(s.MaxRelErr, 0) || math.IsNaN(s.MaxRelErr) {
			t.Errorf("%s 的相对误差是 %v", s.Target, s.MaxRelErr)
		}
	}
}

// 元素数不是块大小的整数倍时要补零凑整，且补的零不能算进误差。
func TestSimulate_元素数不是块大小整数倍(t *testing.T) {
	// 300 个元素：Q8_0 块大小 32 → 需要补到 320；Q6_K/Q4_K 是 256 → 补到 512
	vals := make([]float32, 300)
	for i := range vals {
		vals[i] = float32(i)/300 - 0.5
	}
	sims := simulateAll(vals, 16)
	if len(sims) != 3 {
		t.Fatalf("档数 = %d", len(sims))
	}
	for _, s := range sims {
		if math.IsNaN(s.MaxAbsErr) || s.MaxAbsErr <= 0 {
			t.Errorf("%s 的最大误差是 %v —— 补零后编码应当仍然有效", s.Target, s.MaxAbsErr)
		}
	}
}

// 空输入不能崩，也不能给出误导性的非零数字。
func TestSimulate_空输入(t *testing.T) {
	for _, s := range simulateAll(nil, 16) {
		if s.MaxAbsErr != 0 || s.SNRDB != 0 {
			t.Errorf("%s 空输入应全 0，实际 %+v", s.Target, s)
		}
	}
}

// 块级诊断：scale 分布与「被压得最狠」的子块。
//
// 用 2 的幂当 scale，让所有期望值都是精确的二进制小数，
// 断言可以按位比 —— 避免浮点字面量带来的假失败。
func TestQuantDiag_scale分布(t *testing.T) {
	// 两个块，每块 8 个子块（Q4_K 的结构），每子块 32 个权重
	//
	//   块 0：|-0.5| 0.25 0.125 0.0625 0.03125 0.015625 0.5 0.5  → 块内最大 0.5
	//   块 1：全 2^-7                                             → 块内最大 2^-7
	//
	// 块内归一：块 0 的第 5 个子块比值 2^-5/2^-1 = 1/16 最扁；
	//          块 1 的每个子块比值都是 1（整块均匀，没被压）
	// 跨块比绝对：块 1 的 2^-7 看着比块 0 的都小，会选到块 1 —— 那是错的，
	//          块 1 只是整个块的数值都小，编码器并没有挤压它
	//
	// 这条用例就是拿这个分水岭来钉住「按块内归一」的。
	subs := []decode.SubScale{
		{Scale: -0.5, Elems: 32}, {Scale: 0.25, Elems: 32},
		{Scale: 0.125, Elems: 32}, {Scale: 0.0625, Elems: 32},
		{Scale: 0.03125, Elems: 32}, {Scale: 0.015625, Elems: 32},
		{Scale: 0.5, Elems: 32}, {Scale: 0.5, Elems: 32},
	}
	for range 8 {
		subs = append(subs, decode.SubScale{Scale: 0.0078125, Elems: 32})
	}

	d := quantDiag(model.DtypeQ4K, subs)

	if d.SubBlocks != 16 {
		t.Errorf("SubBlocks = %d, want 16", d.SubBlocks)
	}
	if d.Blocks != 2 {
		t.Errorf("Blocks = %d, want 2", d.Blocks)
	}
	if d.BlockElems != 32 {
		t.Errorf("BlockElems = %d, want 32", d.BlockElems)
	}
	if d.BitsPerWeight != model.DtypeQ4K.BitsPerWeight() {
		t.Errorf("BitsPerWeight = %v, want %v", d.BitsPerWeight, model.DtypeQ4K.BitsPerWeight())
	}
	// 绝对值分布
	if d.ScaleMin != 0.0078125 || d.ScaleMax != 0.5 {
		t.Errorf("ScaleMin/Max = %v/%v, want 0.0078125/0.5", d.ScaleMin, d.ScaleMax)
	}
	if d.ScaleMedian != 0.01171875 {
		t.Errorf("ScaleMedian = %v, want 0.01171875", d.ScaleMedian)
	}
	// 块 0 的 |scale| 和 = 0.5+0.25+0.125+0.0625+0.03125+0.015625+0.5+0.5 = 1.984375
	// 块 1 每个 0.0078125，8 个共 0.0625
	if d.ScaleMean != 2.046875/16 {
		t.Errorf("ScaleMean = %v, want %v", d.ScaleMean, 2.046875/16)
	}
	if d.ZeroScaleBlocks != 0 {
		t.Errorf("ZeroScaleBlocks = %d, want 0", d.ZeroScaleBlocks)
	}
	// 最扁的是块 0 的第 5 个子块
	if d.FlattestIndex != 5 {
		t.Errorf("FlattestIndex = %d, want 5 —— 跨块比绝对 scale 的话会选到块 1 里"+
			"那些（它们只是整块数值小，没被挤压）", d.FlattestIndex)
	}
	if d.FlattestRatio != 0.03125 {
		t.Errorf("FlattestRatio = %v, want 0.03125", d.FlattestRatio)
	}
	if d.FlattestScale != 0.015625 {
		t.Errorf("FlattestScale = %v, want 0.015625", d.FlattestScale)
	}
	// 负数 scale 没被算成"最小的"
	if d.ScaleMin < 0 {
		t.Errorf("ScaleMin = %v，负数 scale 应按绝对值参与统计", d.ScaleMin)
	}
}

// scale 为 0 的子块必须被选为最扁的（它的权重全落在一个量化级上）。
func TestQuantDiag_零scale子块最扁(t *testing.T) {
	subs := []decode.SubScale{
		{Scale: 0.5, Elems: 32}, {Scale: 0.25, Elems: 32},
		{Scale: 0.125, Elems: 32}, {Scale: 0.0625, Elems: 32},
		{Scale: 0.03125, Elems: 32}, {Scale: 0, Elems: 32},
		{Scale: 0.5, Elems: 32}, {Scale: 0.5, Elems: 32},
	}
	d := quantDiag(model.DtypeQ4K, subs)
	if d.FlattestIndex != 5 || d.FlattestRatio != 0 {
		t.Errorf("Flattest = (%d, %v), want (5, 0)", d.FlattestIndex, d.FlattestRatio)
	}
	if d.ZeroScaleBlocks != 1 {
		t.Errorf("ZeroScaleBlocks = %d, want 1", d.ZeroScaleBlocks)
	}
}

// 负数 scale 必须按绝对值参与统计 —— 真实文件里确实有。
func TestQuantDiag_负scale取绝对值(t *testing.T) {
	subs := []decode.SubScale{
		{Scale: -0.5, Elems: 32}, {Scale: -0.25, Elems: 32},
		{Scale: -0.125, Elems: 32}, {Scale: -0.0625, Elems: 32},
		{Scale: -0.03125, Elems: 32}, {Scale: -0.015625, Elems: 32},
		{Scale: -0.0078125, Elems: 32}, {Scale: -0.5, Elems: 32},
	}
	d := quantDiag(model.DtypeQ4K, subs)
	if d.ScaleMin < 0 {
		t.Errorf("ScaleMin = %v，负数 scale 应按绝对值参与统计", d.ScaleMin)
	}
	// 全部为负时，若按带符号比较，「最小」会是 -0.5（绝对值最大）——
	// 那条路会让 FlattestScale 指向最不扁的子块
	if d.FlattestScale != 0.0078125 {
		t.Errorf("FlattestScale = %v, want 0.0078125（绝对值最小的那个）", d.FlattestScale)
	}
}

// 偶数个样本时中位数取中间两个的平均。
func TestQuantDiag_中位数(t *testing.T) {
	subs := []decode.SubScale{
		{Scale: 1, Elems: 32}, {Scale: 2, Elems: 32},
		{Scale: 3, Elems: 32}, {Scale: 4, Elems: 32},
	}
	d := quantDiag(model.DtypeQ4K, subs)
	if d.ScaleMedian != 2.5 {
		t.Errorf("ScaleMedian = %v, want 2.5", d.ScaleMedian)
	}
	if d.FlattestIndex != 0 {
		t.Errorf("FlattestIndex = %d, want 0", d.FlattestIndex)
	}
	// 一个块里只有 4 个子块，不满 Q4_K 的 8 个 —— 不能崩，也不能算成 2 个块
	if d.Blocks != 1 {
		t.Errorf("Blocks = %d, want 1（不满一块按一块算）", d.Blocks)
	}
}

// 空输入不能崩。
func TestQuantDiag_空输入(t *testing.T) {
	d := quantDiag(model.DtypeQ4K, nil)
	if d.SubBlocks != 0 || d.Blocks != 0 || d.FlattestIndex != 0 {
		t.Errorf("空输入应全 0，实际 %+v", d)
	}
	// 位宽来自类型表，与有没有子块无关
	if d.BitsPerWeight != model.DtypeQ4K.BitsPerWeight() {
		t.Errorf("位宽应来自类型表，实际 %v", d.BitsPerWeight)
	}
}

// 整块 scale 全为 0 时不能除零，且这些子块应当被判为最扁。
func TestQuantDiag_整块全零(t *testing.T) {
	subs := make([]decode.SubScale, 8)
	for i := range subs {
		subs[i] = decode.SubScale{Elems: 32}
	}
	d := quantDiag(model.DtypeQ4K, subs)
	if math.IsNaN(d.FlattestRatio) || math.IsNaN(d.ScaleMean) {
		t.Errorf("全零块产生了 NaN: %+v", d)
	}
	if d.FlattestRatio != 0 {
		t.Errorf("FlattestRatio = %v, want 0", d.FlattestRatio)
	}
	if d.ZeroScaleBlocks != 8 {
		t.Errorf("ZeroScaleBlocks = %d, want 8", d.ZeroScaleBlocks)
	}
}

// 分块喂与一次喂必须得到完全相同的结果。
//
// 聚合器从"先收全部子块再统计"改成流式，这条钉住改写没有改变语义。
// 分块边界是块级诊断最容易写错的地方（块内归一要求先看完整块）。
func TestQuantAgg_分块与整体一致(t *testing.T) {
	// 3 个块 × 8 个子块（Q4_K 结构），每块的 scale 分布不同
	var subs []decode.SubScale
	for blk := range 3 {
		base := float32(uint(1) << (blk + 1)) // 1, 2, 4
		for j := range 8 {
			subs = append(subs, decode.SubScale{
				Scale: base / float32(uint(1)<<uint(j)), Elems: 32,
			})
		}
	}

	whole := quantDiag(model.DtypeQ4K, subs)

	// 按各种切法分块喂：1 个、3 个、7 个、17 个一组
	for _, chunk := range []int{1, 3, 7, 17} {
		a := newQuantAgg(model.DtypeQ4K, int64(len(subs)))
		for i := 0; i < len(subs); i += chunk {
			a.add(subs[i:min(i+chunk, len(subs))])
		}
		got := a.finish()
		if got != whole {
			t.Errorf("按 %d 个一组分块喂，结果与整体喂不同：\n got %+v\nwant %+v",
				chunk, got, whole)
		}
	}
	if whole.SubBlocks != 24 || whole.Blocks != 3 {
		t.Fatalf("整体喂的结果就不对：%d 子块 / %d 块", whole.SubBlocks, whole.Blocks)
	}
}

// 中位数的缓冲必须有上限，且抽样不能与数据的周期混叠。
//
// 上限存在但不变的话，一个 3 亿权重的张量会吃掉 78 MB。
// 混叠是抽样最容易踩的坑：早先用等距抽样（步长 = ceil(总数/预算)），
// 这条用例的数据是"每 10 个一个大值"，步长正好 10 —— 抽到的**全是大值**，
// 中位数从 1 变成 4。改成蓄水池抽样后才对。
func TestQuantAgg_中位数有预算上限(t *testing.T) {
	// 90% 小值 + 10% 大值：中位数必须是 1，混叠时会被抽成 4
	const n = quantMedianBudget*3 + 7
	const bigEvery = 10

	countBig := int64(0)
	a := newQuantAgg(model.DtypeQ8_0, n)
	for i := int64(0); i < n; i++ {
		if i%bigEvery == 0 {
			countBig++
			a.add([]decode.SubScale{{Scale: 4, Elems: 32}})
		} else {
			a.add([]decode.SubScale{{Scale: 1, Elems: 32}})
		}
	}
	q := a.finish()
	small := n - countBig

	if len(a.medians) > quantMedianBudget {
		t.Errorf("中位数缓冲 %d 超过了预算 %d", len(a.medians), quantMedianBudget)
	}
	if !q.ScaleMedianSampled {
		t.Error("超过预算却没有标 ScaleMedianSampled —— 用户会把抽样值当成全量")
	}
	// 其余统计量仍然是全量精确的
	if q.SubBlocks != n {
		t.Errorf("SubBlocks = %d, want %d —— 抽样不该影响计数", q.SubBlocks, n)
	}
	if q.ScaleMin != 1 || q.ScaleMax != 4 {
		t.Errorf("ScaleMin/Max = %v/%v, want 1/4 —— 极值必须来自全量", q.ScaleMin, q.ScaleMax)
	}
	wantMean := (float64(small)*1 + float64(countBig)*4) / float64(n)
	if math.Abs(q.ScaleMean-wantMean) > 1e-9 {
		t.Errorf("ScaleMean = %v, want %v —— 均值必须来自全量", q.ScaleMean, wantMean)
	}
	if q.ScaleMedian != 1 {
		t.Errorf("ScaleMedian = %v, want 1 —— 抽样与数据的周期混叠了"+
			"（大值每 %d 个出现一次，抽样的步长也是 %d 的倍数）",
			q.ScaleMedian, bigEvery, bigEvery)
	}

	// 没超预算时不标抽样，且样本就是全量
	a2 := newQuantAgg(model.DtypeQ8_0, 10)
	a2.add([]decode.SubScale{{Scale: 1, Elems: 32}, {Scale: 4, Elems: 32}})
	if q2 := a2.finish(); q2.ScaleMedianSampled {
		t.Error("没超预算却标了 ScaleMedianSampled")
	}
	if q2 := a2.finish(); q2.ScaleMedian != 2.5 {
		t.Errorf("没超预算时中位数 = %v, want 2.5（应为全量精确值）", q2.ScaleMedian)
	}
}
