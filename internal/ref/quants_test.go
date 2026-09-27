package ref

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/sillydong/modelview/internal/decode"
	"github.com/sillydong/modelview/internal/model"
)

// 每个"已识别的量化类型"都必须有条目 —— 否则用户看到文件里的 Q5_K，
// 速查表里查不到。
func TestQuants_覆盖全部量化类型(t *testing.T) {
	have := map[string]bool{}
	for _, e := range quantsTable().Entries {
		have[e.ID] = true
	}
	covered := 0
	for d := range model.AllDtypes() {
		if !d.IsQuantized() {
			continue
		}
		covered++
		if !have["quant:"+string(d)] {
			t.Errorf("量化类型 %s 没有速查条目", d)
		}
	}
	// 反向门禁：覆盖数掉了说明类型表变了，核对后再改这个数。
	//
	// **23 是实测值**：经典 6（Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1）
	// + K 系列 6（Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K）+ IQ 系列 9
	// + MXFP4/NVFP4 2 = 23。
	const wantCovered = 23
	if covered != wantCovered {
		t.Errorf("量化类型有 %d 个，预期 %d 个 —— 类型表变了，核对后再改这个数",
			covered, wantCovered)
	}
}

// 位宽与块结构必须从 model 派生，不能另写一份。
//
// 另写一份的后果：改了解码器的块布局而忘了改速查表，
// 界面会照着一个错数字解释用户的文件，而且没有任何东西会失败。
func TestQuants_位宽与块结构来自类型表(t *testing.T) {
	checked := 0
	for _, e := range quantsTable().Entries {
		name := strings.TrimPrefix(e.ID, "quant:")
		d := model.Dtype(name)
		if !d.IsQuantized() {
			continue
		}
		checked++
		// **必须走 bpwOf，不能直接比 d.BitsPerWeight()**。
		// 这里原来就是那么写的，于是 9 个 IQ 类型（d.BitsPerWeight()==0）
		// 两边同时为 0，断言被退化值满足 —— 「0 bit/权重」和「+Inf 压缩比」
		// 就这么一路绿着进了速查表。
		bpw, ok := bpwOf(name)
		if !ok {
			t.Errorf("%s 取不到位宽", name)
			continue
		}
		if got, want := e.Field("位宽"), humanFloat(bpw)+" bit/权重"; got != want {
			t.Errorf("%s 的位宽写的是 %q，算出来是 %q", name, got, want)
		}
		bb, hasBB := d.BlockBytes()
		switch {
		case decode.ScalesSupported(d):
			want := fmtBlock(d.BlockElems(), bb, d.BlockElems()/subElemsOf(name))
			if got := e.Field("块结构"); got != want {
				t.Errorf("%s 的块结构写的是 %q，算出来是 %q", name, got, want)
			}
		case hasBB:
			// 块大小收了但块头布局没验证：元素数与字节数可以说，
			// 子块划分必须标成未验证 —— 不能给一个看起来像事实的数字
			got := e.Field("块结构")
			if !strings.Contains(got, "未验证") {
				t.Errorf("%s 的块头布局未经验证，块结构字段却没标出来: %q", name, got)
			}
			if !strings.Contains(got, fmt.Sprintf("%d 个权重", d.BlockElems())) {
				t.Errorf("%s 的块结构没写元素数（那是已验证的事实）: %q", name, got)
			}
		}
	}
	if checked == 0 {
		t.Fatal("一个量化条目都没查 —— 这条测试会变成空转")
	}
}

// 压缩比必须由位宽算出（相对 F16）。
func TestQuants_压缩比(t *testing.T) {
	e, ok := ByID("quant:Q4_K")
	if !ok {
		t.Fatal("没有 Q4_K 条目")
	}
	// F16 是 16 bit/权重，Q4_K 是 4.5 → 3.56 倍
	want := humanFloat(16 / model.DtypeQ4K.BitsPerWeight())
	if got := e.Field("相对 F16 的压缩比"); got != want {
		t.Errorf("Q4_K 压缩比写的是 %q，算出来是 %q", got, want)
	}
}

// GGML 类型码必须来自 model.Dtype.GGMLCode()。
func TestQuants_类型码来自类型表(t *testing.T) {
	checked := 0
	for _, e := range quantsTable().Entries {
		name := strings.TrimPrefix(e.ID, "quant:")
		d := model.Dtype(name)
		code, ok := d.GGMLCode()
		if !ok {
			continue
		}
		checked++
		want := strconv.FormatUint(uint64(code), 10)
		if got := e.Field("GGML 类型码"); got != want {
			t.Errorf("%s 的类型码写的是 %q，类型表里是 %q", name, got, want)
		}
	}
	// 精确值：23 个量化类型都有类型码（下限的话少查几个不会有东西红）
	if checked != 23 {
		// 用变量而不是写死数字：写死的话改判据时忘了改文案，
		// 失败信息会把人引去核对一个旧的数
		t.Fatalf("查了 %d 个类型码，预期 %d 个 —— 增删类型时同步改这里",
			checked, 23)
	}
}

// 速查表里的"一个子块覆盖多少权重"必须与解码器的事实一致。
//
// 这是同一份事实的第二处表达（第一处在 decode 的 scalesExtractors 表里）——
// 不钉住的话，改了解码器的子块划分而速查表还写着旧的，
// 界面会照着一个错数字给用户解释文件。
func TestQuants_子块大小与解码器一致(t *testing.T) {
	checked := 0
	for name, want := range subElemsOfMap {
		d := model.Dtype(name)
		bb, ok := d.BlockBytes()
		if !ok {
			t.Errorf("%s: 好端端的块字节数取不到", name)
			continue
		}
		// 没验证过块头布局的类型不该出现在这张表里 ——
		// 那意味着速查表给出了一个无从核实的子块数
		if !decode.ScalesSupported(d) {
			t.Errorf("%s 的块头布局未经验证，不该出现在 subElemsOfMap 里"+
				"（写的是 %d）—— 速查表会把它当成事实展示", name, want)
			continue
		}
		// 造一个全 0 的块就够：我们只问它切出几个子块、每个多大
		subs, err := decode.Scales(d, make([]byte, bb))
		if err != nil {
			t.Errorf("%s: 解码器说支持却报错 —— %v", name, err)
			continue
		}
		checked++
		if len(subs) == 0 {
			t.Errorf("%s: 解码器切出 0 个子块", name)
			continue
		}
		got := int64(subs[0].Elems)
		if got != want {
			t.Errorf("%s: 速查表写着一个子块 %d 个权重，解码器切出来是 %d 个",
				name, want, got)
		}
		if int64(len(subs))*got != d.BlockElems() {
			t.Errorf("%s: %d 个子块 × %d 个权重 = %d，与块元素数 %d 不符",
				name, len(subs), got, int64(len(subs))*got, d.BlockElems())
		}
	}
	// 反向门禁：解码器**支持**的类型都必须在这张表里，
	// 否则速查表会少报子块数（把 16 个子块说成 1 个）
	for d := range model.AllDtypes() {
		if !decode.ScalesSupported(d) {
			continue
		}
		if _, ok := subElemsOfMap[string(d)]; !ok {
			t.Errorf("%s 的块头解码器能读，速查表却没收录它的子块大小", d)
		}
	}
	// 精确值：解码器能读块头的有 10 个类型（经典 5 + K 系列 5，Q8_1/Q8_K 未验证）
	if checked != len(subElemsOfMap) {
		t.Fatalf("查了 %d 个类型的子块大小，表里有 %d 个", checked, len(subElemsOfMap))
	}
}

// 位宽必须来自**恰好一个**地方，而且不能是 0。
//
// 修掉的那个 bug：9 个 IQ 类型在 model 的 dtypeTable 里没有块结构
// （本工具不解码 IQ，decode 拿 BlockBytes()==0 当"不支持"的闸门），
// 于是 BitsPerWeight() 返回 0 —— 速查表渲染成「0 bit/权重」与
// 「+Inf 压缩比」，而同一个条目的说明里写着「1.5625 bit/权重」。
// 它一直绿着，因为测试断言的是 `field == humanFloat(d.BitsPerWeight())`：
// 两边同时为 0，镜像断言被退化值满足。
func TestQuants_位宽来源唯一(t *testing.T) {
	// 上游块的 (元素数, 字节数)。独立抄自 ggml 的 type_size / type_blck_size
	// （QK_K=256），**不是**从 quantBpwFallback 读的 —— 那样就成了同义反复。
	// 值本身从除法算出来，复核时能一眼看出对错。
	upstream := map[string][2]int{
		"IQ1_S": {256, 50}, "IQ1_M": {256, 56},
		"IQ2_XXS": {256, 66}, "IQ2_XS": {256, 74}, "IQ2_S": {256, 82},
		"IQ3_XXS": {256, 98}, "IQ3_S": {256, 110},
		"IQ4_NL": {32, 18}, "IQ4_XS": {256, 136},
	}

	// ① 两个来源必须互斥且穷尽：有 model 值的不能有兜底，反之亦然
	for name := range quantEditorial {
		inModel := model.Dtype(name).BitsPerWeight() > 0
		_, inFallback := quantBpwFallback[name]
		switch {
		case inModel && inFallback:
			t.Errorf("%s 两个来源都有位宽 —— 改一处另一处不会红", name)
		case !inModel && !inFallback:
			t.Errorf("%s 两个来源都没有位宽 —— 会渲染成「0 bit/权重」", name)
		}
	}
	// 兜底表里不能有 quantEditorial 之外的键（写错名字的表现是静默无效）
	for name := range quantBpwFallback {
		if _, ok := quantEditorial[name]; !ok {
			t.Errorf("兜底表里的 %s 在量化方案表里没有条目", name)
		}
	}

	// ② 兜底值必须等于上游的 字节数×8/元素数（独立重算）
	if len(quantBpwFallback) != len(upstream) {
		t.Errorf("兜底表 %d 项，上游清单 %d 项", len(quantBpwFallback), len(upstream))
	}
	for name, eb := range upstream {
		want := float64(eb[1]) * 8 / float64(eb[0])
		got, ok := quantBpwFallback[name]
		if !ok {
			t.Errorf("%s 缺少兜底位宽", name)
			continue
		}
		if got != want {
			t.Errorf("%s 的位宽是 %v，按上游 %d 元素 / %d 字节算是 %v",
				name, got, eb[0], eb[1], want)
		}
	}

	// ③ 表里渲染出来的位宽/压缩比必须与 bpwOf 一致，且位宽 > 0、压缩比有限
	for _, e := range quantsTable().Entries {
		name := strings.TrimPrefix(e.ID, "quant:")
		bpw, ok := bpwOf(name)
		if !ok || bpw <= 0 {
			t.Errorf("%s 取不到位宽", name)
			continue
		}
		if got, want := e.Field("位宽"), humanFloat(bpw)+" bit/权重"; got != want {
			t.Errorf("%s 的位宽写的是 %q，算出来是 %q", name, got, want)
		}
		got := e.Field("相对 F16 的压缩比")
		if got == "+Inf" || got == "0" || got == "" {
			t.Errorf("%s 的压缩比是 %q —— 那不是一个能用的数", name, got)
		}
		if want := humanFloat(16 / bpw); got != want {
			t.Errorf("%s 的压缩比写的是 %q，算出来是 %q", name, got, want)
		}
	}
}

// 表必须按位宽**严格升序**（相等时按名字）。
//
// 修 bug 前 9 个 IQ 的位宽全是 0，它们一起挤到最前面 ——
// 「按位宽升序」这个说法在那 9 行上是假的。
func TestQuants_按位宽升序(t *testing.T) {
	entries := quantsTable().Entries
	if len(entries) != 23 {
		t.Fatalf("量化条目 %d 个，预期 23 个 —— 这条排序断言要覆盖全表", len(entries))
	}
	for i := 1; i < len(entries); i++ {
		prev, cur := entries[i-1], entries[i]
		bp, _ := bpwOf(strings.TrimPrefix(prev.ID, "quant:"))
		bc, _ := bpwOf(strings.TrimPrefix(cur.ID, "quant:"))
		if bc < bp {
			t.Errorf("顺序反了：%s(%v) 排在 %s(%v) 之后", prev.ID, bp, cur.ID, bc)
		}
		if bc == bp && cur.ID < prev.ID {
			t.Errorf("位宽相同的 %s 与 %s 没按名字排", prev.ID, cur.ID)
		}
	}
}
