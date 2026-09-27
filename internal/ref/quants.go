package ref

import (
	"fmt"
	"sort"
	"strconv"

	"github.com/sillydong/modelview/internal/decode"
	"github.com/sillydong/modelview/internal/model"
)

// quantEditorial 是每种量化方案里**只能靠人写**的部分。
//
// 键是 Dtype 字符串。**缺项不是错误**：新收录一个类型时先让它有条目、
// 只是没有编辑性说明，好过因为没写说明而整个类型查不到。
//
// 位宽与压缩比不在这里 —— 它们从 model.Dtype 派生（见 bpwOf）。
var quantEditorial = map[string]struct {
	family string // 方案族
	loss   string // 典型质量损失
	use    string // 用途
	notes  string
}{
	"Q4_0": {"经典（Q4_0/Q4_1/Q5_x/Q8_0）", "明显", "最早的 4 位方案，老 ggml 模型里常见",
		"32 元素一块、每块一个 fp16 scale。**码不是线性排列的**：低 4 位放前 16 个元素的码，" +
			"高 4 位放后 16 个 —— 按顺序读会得到错乱的权重。"},
	"Q4_1": {"经典（Q4_0/Q4_1/Q5_x/Q8_0）", "明显", "比 Q4_0 多一个 min，非对称",
		"多存一个 min（2 字节），能表示不以 0 为中心的分布，代价是每块从 18 字节涨到 20。"},
	"Q5_0": {"经典（Q4_0/Q4_1/Q5_x/Q8_0）", "较小", "在 4 位上再加 1 个比特",
		"第 5 位单独存在一块 qh 里，**按位一一对应**：元素 j 用 qh 的第 j 位" +
			"（不是低 16 位给前半、高 16 位给后半）。易错的是忘了把它左移 4 位再或进去 —— " +
			"那会让一半的值错、另一半对。"},
	"Q5_1": {"经典（Q4_0/Q4_1/Q5_x/Q8_0）", "较小", "5 位 + min",
		"Q5_0 的非对称版本。"},
	"Q8_0": {"经典（Q4_0/Q4_1/Q5_x/Q8_0）", "几乎无损", "接近无损的整数量化，常作对照基准",
		"32 元素一块、每块一个 fp16 scale、8 位有符号码。实测信噪比通常在 40 dB 以上。"},
	"Q8_1": {"经典（Q4_0/Q4_1/Q5_x/Q8_0）", "几乎无损", "带额外求和量的 8 位，中间产物",
		"比 Q8_0 多存一个块内和（用于某些 kernel 的快速内积），文件里少见。"},
	"Q2_K": {"K 系列（超级块）", "很大", "极致压缩，通常只用于对质量不敏感的场景",
		"256 元素超级块，16 个子块各 16 个权重。每个 scale 字节**低 4 位是 scale、高 4 位是 min**，" +
			"且每一半 16 元素各用一个独立的 scale 字节。"},
	"Q3_K": {"K 系列（超级块）", "大", "2 位与 4 位之间",
		"6 位子 scale 的打包方式极其反直觉：低 4 位在 scales[0..7]、高 2 位在 scales[8..11]，" +
			"且是按 32 位字重新分组的。"},
	"Q4_K": {"K 系列（超级块）", "中等", "**最常用的 4 位方案**，Q4_K_M 的一半组成",
		"256 元素超级块、8 个子块各 32 个权重。每子块 6 位 scale + 6 位 min，分两级缩放 —— " +
			"这是 K 系列与 Q4_0 的根本区别。"},
	"Q5_K": {"K 系列（超级块）", "较小", "5 位，Q5_K_M 的组成",
		"结构同 Q4_K，多一个 32 字节的 qh 存第 5 位。"},
	"Q6_K": {"K 系列（超级块）", "小", "高质量量化，常与 Q4_K 混用来拉高关键层",
		"256 元素超级块、16 个子块各 16 个权重。**d 存在块尾（偏移 208），不在块头** —— " +
			"与 Q2_K（偏移 80）、Q3_K（偏移 108）一样；只有 Q4_K/Q5_K 把 d 放在块头。"},
	"Q8_K": {"K 系列（超级块）", "几乎无损", "K 系列的中间产物，不直接作为权重格式",
		"256 元素一块，带 fp32 的 d 与块内和。主要用于把激活量化后做点积，文件里少见。"},
	// MXFP4 / NVFP4 是 4 位**浮点**块格式，与上面那些"整数码 + scale"不同。
	// 块的总字节数来自 ggml 的 type_size（32 个权重 17 字节、64 个权重 36 字节）。
	//
	// **钉住它们的是另外两条，不是 TestQuants_位宽来源唯一** ——
	// 那条测试的独立重算清单只覆盖 IQ 系列（见 quants_test.go 的 upstream）。
	//   - internal/parser/gguf 的 TestTensorByteSize_覆盖全部类型码：
	//     独立抄录的 {39: 17/32, 40: 36/64}
	//   - 同包的 TestBlockTable_与真实文件吻合：拿真实的 gpt-oss:20b 验
	//     张量跨度（实测加进语料之前，"17 改 18"连测试期望一起改是全绿的）
	//
	// 块内怎么分（多少字节放码、多少放共享指数）来自各自的规范，
	// 本工具**不解码**它们，所以那部分不在这里当成事实陈述。
	"MXFP4": {"OCP MX（4 位浮点 + 共享指数）", "中等", "gpt-oss 系列的专家层用的就是它",
		"32 个权重共用一份 8 位共享指数（E8M0），权重本身是 4 位浮点（E2M1）。" +
			"OCP Microscaling Formats v1.0 定义。**本工具不解码它**，只算占用大小。"},
	"NVFP4": {"NVIDIA（4 位浮点 + FP8 共享指数）", "较小", "NVIDIA 的 4 位方案，块比 MXFP4 更细",
		"64 个权重一块，块内再按 16 个一组配 FP8（E4M3）共享指数 —— " +
			"块更细所以同样 4 位下精度更好、开销也更大。" +
			"**本工具不解码它**，只算占用大小。"},
	"IQ1_S":   {"IQ 系列（码本）", "极大", "极限压缩（1.5625 bit/权重）", "基于码本查表，不是线性缩放。"},
	"IQ1_M":   {"IQ 系列（码本）", "极大", "极限压缩（1.75 bit/权重）", "基于码本查表。"},
	"IQ2_XXS": {"IQ 系列（码本）", "很大", "2 位档的极端版本（2.0625 bit/权重）", "基于码本查表。"},
	"IQ2_XS":  {"IQ 系列（码本）", "很大", "2 位档（2.3125 bit/权重）", "基于码本查表。"},
	"IQ2_S":   {"IQ 系列（码本）", "很大", "2 位档（2.5625 bit/权重）", "基于码本查表。"},
	"IQ3_XXS": {"IQ 系列（码本）", "大", "3 位档的极端版本（3.0625 bit/权重）", "基于码本查表。"},
	"IQ3_S":   {"IQ 系列（码本）", "大", "3 位档（3.4375 bit/权重）", "基于码本查表。"},
	"IQ4_NL":  {"IQ 系列（码本）", "中等", "4 位非线性的，介于 Q4_0 与 K 系列之间", "基于码本查表。"},
	"IQ4_XS":  {"IQ 系列（码本）", "中等", "4 位档（4.25 bit/权重）", "基于码本查表。"},
}

// quantBpwFallback 是 model.Dtype 里**没有块结构**的那些类型的位宽。
//
// 为什么需要它：IQ 系列没进 model 的 dtypeTable（本工具不解码 IQ，
// decode 拿 `BlockBytes()==0` 当"不支持"的闸门），于是
// `Dtype.BitsPerWeight()` 返回 0 —— 速查表渲染出来是
// 「0 bit/权重」与「+Inf 压缩比」，而同一行的说明里写着「1.5625 bit/权重」。
// **同一屏里自相矛盾，而且那个 0 看起来像个事实。**
//
// 值来自上游的块结构（ggml 的 type_size/type_blck_size，QK_K=256），
// 由 bpw = 字节数 × 8 / 元素数 算出：
//
//	IQ1_S   256/50  IQ1_M  256/56  IQ2_XXS 256/66  IQ2_XS 256/74
//	IQ2_S   256/82  IQ3_XXS 256/98 IQ3_S  256/110 IQ4_XS 256/136
//	IQ4_NL   32/18
//
// 推导过程写在这里而不是直接抄小数：口径（含不含块头、块多大）
// 是这些数字全部的意义所在，只留一个小数就没法复核了。
// TestQuants_位宽来源唯一 拿这张表与 model 的值做互斥校验，
// 并独立重算一遍这些除法。
var quantBpwFallback = map[string]float64{
	"IQ1_S":   1.5625,
	"IQ1_M":   1.75,
	"IQ2_XXS": 2.0625,
	"IQ2_XS":  2.3125,
	"IQ2_S":   2.5625,
	"IQ3_XXS": 3.0625,
	"IQ3_S":   3.4375,
	"IQ4_NL":  4.5,
	"IQ4_XS":  4.25,
}

// subElemsOfMap 是**解码器能读块头**的那些类型一个子块覆盖的权重数。
//
// 这里写一份，并由 TestQuants_子块大小与解码器一致 拿真实解码结果
// 交叉钉住 —— 两份表达 + 一条一致性断言，改了一处另一处会红。
//
// **只收解码器支持的类型**：Q8_1 与 Q8_K 的块头布局本工具没有独立参照
// 验证过（见 decode/scales.go 的说明），它们的子块划分也就无从核实。
// 把没验证的数字写进速查表，等于把一个猜测当成事实展示给用户 ——
// 那两个类型在表里走"子块划分未验证"的分支。
var subElemsOfMap = map[string]int64{
	"Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32, "Q8_0": 32,
	"Q4_K": 32, "Q5_K": 32,
	"Q2_K": 16, "Q3_K": 16, "Q6_K": 16,
}

// fmtBlock 把一个块的结构写成一行。
//
// 生成与测试共用它：两处各写一遍格式化的话，对照会因格式差异假失败，
// 掩盖真正的漂移。
func fmtBlock(elems, bytes, subs int64) string {
	return fmt.Sprintf("%d 个权重 → %d 字节（%d 个子块）", elems, bytes, subs)
}

// subElemsOf 返回一个子块覆盖的权重数；未收录时返回 1（"没有子块划分"）。
func subElemsOf(name string) int64 {
	if n, ok := subElemsOfMap[name]; ok {
		return n
	}
	return 1
}

// bpwOf 返回用于展示的位宽：优先 model.Dtype（Q 系列，解码器验证过），
// 其次 quantBpwFallback（IQ 系列，model 里没有块结构）。
//
// 第二返回值为 false 表示两个来源都没有 —— 调用方**不能拿 0 顶上**：
// 「0 bit/权重」是个看起来像事实的值，压缩比还会算出 +Inf。
// 这正是这一版修掉的 bug。
func bpwOf(name string) (float64, bool) {
	if b := model.Dtype(name).BitsPerWeight(); b > 0 {
		return b, true
	}
	if b, ok := quantBpwFallback[name]; ok && b > 0 {
		return b, true
	}
	return 0, false
}

func quantsTable() Table {
	// 按位宽升序：从压得最狠的开始读。位宽相同时按名字，保证顺序稳定。
	//
	// **排序也要用 bpwOf，不能直接用 d.BitsPerWeight()**：后者对 9 个 IQ
	// 全是 0，它们会一起挤到最前面，而"升序"这个说法就成了假的。
	names := make([]string, 0, len(quantEditorial))
	for n := range quantEditorial {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool {
		bi, _ := bpwOf(names[i])
		bj, _ := bpwOf(names[j])
		if bi != bj {
			return bi < bj
		}
		return names[i] < names[j]
	})

	tb := Table{ID: "quants", Title: "量化方案"}
	for _, name := range names {
		d := model.Dtype(name)
		ed := quantEditorial[name]
		fields := []Field{
			{"方案族", ed.family},
			{"典型质量损失", ed.loss},
			{"适用场景", ed.use},
		}
		// 位宽与压缩比成对出现：只有一个会让用户去心算另一个，
		// 而心算 16/1.5625 正是他打开这张表想避免的事
		if bpw, ok := bpwOf(name); ok {
			fields = append(fields,
				Field{"位宽", humanFloat(bpw) + " bit/权重"},
				Field{"相对 F16 的压缩比", humanFloat(16 / bpw)})
		} else {
			// 两个来源都没有。**不能填 0** —— 那看起来像个事实。
			// 目前不会走到这里（TestQuants_位宽来源唯一 保证划分完整），
			// 但少一个数字好过给一个假的
			fields = append(fields, Field{"位宽", "未收录"})
		}
		if code, ok := d.GGMLCode(); ok {
			fields = append(fields, Field{"GGML 类型码", strconv.FormatUint(uint64(code), 10)})
		}
		// 三种情形：解码器能读块头 / 只知道块大小 / 连块大小都没有。
		// 先把值取出来再判断 —— 原先这里把 hasBB 写成了一个立即调用的
		// 函数字面量塞在 case 的位置上（`case func() bool {…}():`），
		// 读者得先解析那个 IIFE 才知道这个分支是什么意思。
		bb, hasBB := d.BlockBytes()
		elems := d.BlockElems()
		switch {
		case decode.ScalesSupported(d):
			fields = append(fields, Field{"块结构",
				fmtBlock(elems, bb, elems/subElemsOf(name))})
		case hasBB:
			// 块大小收了，但块头布局没验证过 —— 元素数与字节数可以说
			//（它们来自已验证的块表），子块划分不能说
			fields = append(fields, Field{"块结构",
				fmt.Sprintf("%d 个权重 → %d 字节（子块划分未验证）", elems, bb)})
		default:
			// IQ 系列连块大小都没收录 —— 明说，不要留空让人以为是漏了
			fields = append(fields, Field{"块结构", "未收录（本工具不解码 IQ 系列）"})
		}
		tb.Entries = append(tb.Entries, Entry{
			ID:      "quant:" + name,
			Title:   name,
			Fields:  fields,
			Notes:   ed.notes,
			SeeAlso: []string{"float:F16"},
		})
	}
	return tb
}
