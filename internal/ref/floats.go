package ref

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/sillydong/modelview/internal/model"
)

// floatFormat 是一种 IEEE 浮点格式的位布局。
//
// 数值属性全部由这三个数算出来，不写死 —— 见 maxFinite。
type floatFormat struct {
	name    string
	sign    int // 符号位，总是 1，列出来是为了自解释
	exp     int
	mant    int
	aliases []string // 别称（safetensors 的 dtype 名、行业习惯叫法）
	notes   string

	// maxOverride 是**公式不适用**时的最大有限值。0 表示用 maxFinite 算。
	//
	// 只有 F8_E4M3FN 需要它，而原因是数据不是公式：
	// maxFinite 假定"指数域全 1 留给 Inf/NaN"（IEEE 的规矩），
	// 但这个格式**没有 Inf** —— 只有 S.1111.111 这一个位型是 NaN，
	// 于是指数 1111 对有限值是可用的，最大值是 1.110b × 2^8 = **448**。
	// 按 IEEE 的假定算出来是 240（1.875 × 2^7），差了一倍。
	//
	// 名字里的 FN 就是 "finite only"。三个独立来源核对过：
	//   - torch.finfo(torch.float8_e4m3fn).max == 448
	//   - 按位枚举这个格式的全部位型取最大值 == 448
	//   - 本分支的计划文档写的就是"最大 448"（代码当时没跟上）
	// **不要改 maxFinite 去凑 448**：它对 IEEE 格式是对的
	//（E5M2 = 57344 已独立核对），差异来自"哪些位型被留作特殊值"。
	maxOverride float64
}

// maxFinite 返回 (1, exp, mant) 布局下最大的有限值。
//
// 指数域全 1 留给 Inf/NaN，所以最大指数是 2^exp - 2；
// 去掉偏置 b = 2^(exp-1) - 1 后是 2^exp - 1 - b。尾数取全 1 时
// 有效数字是 2 - 2^-mant。
//
// 逐位算而不是抄教科书：抄的话 F16 写成 65535 不会有任何东西发现。
func maxFinite(exp, mant int) float64 {
	bias := (1 << (exp - 1)) - 1
	maxExp := (1 << exp) - 2 - bias
	return math.Ldexp(2-math.Ldexp(1, -mant), maxExp)
}

// minNormal 返回最小正规数 2^(1-bias)。
func minNormal(exp int) float64 {
	bias := (1 << (exp - 1)) - 1
	return math.Ldexp(1, 1-bias)
}

// maxOf 返回一个格式的最大有限值：有覆盖值就用它，否则按位布局算。
func maxOf(f floatFormat) float64 {
	if f.maxOverride != 0 {
		return f.maxOverride
	}
	return maxFinite(f.exp, f.mant)
}

// floatFormatByID 是位布局的唯一出处，floatsTable() 与测试都读它。
//
// 键是条目 ID，与 floatsTable() 里生成的 ID 一致 ——
// 两处不一致的话，测试会静默跳过（它用 `if !ok { continue }`），
// 所以 TestFloatFormats_表与算法一致 里有覆盖数下限兜底。
var floatFormatByID = map[string]floatFormat{
	"float:F64": {name: "F64", sign: 1, exp: 11, mant: 52,
		aliases: []string{"float64", "double", "双精度"},
		notes: "模型权重里极少见；出现通常是某个中间量被意外存了下来。" +
			"推理时它比 F32 慢得多，而模型精度并不会因此提高。"},
	"float:F32": {name: "F32", sign: 1, exp: 8, mant: 23,
		aliases: []string{"float32", "float", "单精度"},
		notes: "模型的通用工作精度。GGUF 里 norm/bias 这类小张量常见，" +
			"权重本身很少用 —— 同样大小下参数只有 F16 的一半。"},
	"float:BF16": {name: "BF16", sign: 1, exp: 8, mant: 7,
		aliases: []string{"bfloat16", "bf16"},
		notes: "指数位与 F32 相同（范围一样大），尾数只有 7 位（精度约 2 位十进制）。" +
			"训练时不用担心溢出，代价是同样的数在 F16 里更精确。"},
	"float:F16": {name: "F16", sign: 1, exp: 5, mant: 10,
		aliases: []string{"float16", "half", "fp16", "半精度"},
		notes: "训练与推理的常用精度。指数位只有 5 位，超过最大值就溢出成 Inf，" +
			"太小的数掉进次正规数——**范围比精度更容易出事**，" +
			"混合精度训练里的 loss scale 就是为了绕开这一点。"},
	"float:F8_E5M2": {name: "F8_E5M2", sign: 1, exp: 5, mant: 2,
		aliases: []string{"fp8_e5m2"},
		notes: "5 位指数 + 2 位尾数，范围与 F16 相当但精度只有 2 位尾数。" +
			"用作梯度侧（宁可糙也不能溢出）。"},
	"float:F8_E4M3": {name: "F8_E4M3", sign: 1, exp: 4, mant: 3,
		maxOverride: 448, // 这个格式没有 Inf，见 maxOverride 的说明
		aliases:     []string{"fp8_e4m3", "float8_e4m3fn"},
		notes: "4 位指数 + 3 位尾数，范围很窄。精度极低（约 1 位十进制），" +
			"只在有 per-block scale 的量化方案里可用。"},
}

// stDtypeNotes 是 safetensors dtype 条目里**只能靠人写**的那部分。
//
// 拼写、字节数都不在这里：清单取自 model.SafetensorsDtypes（与解析器同一份），
// 字节数由 model.Dtype.ByteSize 算出来。写死"4 字节"的散文看着无害，
// 但它与 Dtype 表是两份表达，改了一处另一处不会红 ——
// 而这张表是给用户查的，错一个字节数就是给了一个假事实。
//
// 只写"光看名字看不出来"的那些；没写的条目就没有补充说明。
var stDtypeNotes = map[string]string{
	"F32":  "safetensors 里最常见的权重精度。",
	"BF16": "与 F16 同宽但取舍不同：指数位多、尾数少，范围大而精度低。",
	"I32":  "常见于整数化的权重、以及索引类张量。",
}

func floatsTable() Table {
	// 条目顺序固定：按位宽从大到小，读起来是一条谱。
	//
	// **顺序从表里派生，不手写一个 order 清单**。手写那版有个反向缺口：
	// 漏写一个 ID 的表现是"那一整条静默消失"，而下面那句 panic 只挡住
	// "order 引用了未定义的布局"这一个方向 —— 实测删掉 "float:F16" 后，
	// 用户看到的表少了整条 F16（位布局/最大值/别称/loss scale 说明），
	// 而全部测试照样绿（checked 从 6 掉到 5，没触发下限）。
	order := make([]string, 0, len(floatFormatByID))
	for id := range floatFormatByID {
		order = append(order, id)
	}
	sort.Slice(order, func(i, j int) bool {
		fi, fj := floatFormatByID[order[i]], floatFormatByID[order[j]]
		// 位宽（符号+指数+尾数）大的在前；同宽按名字，保证顺序稳定
		if wi, wj := fi.exp+fi.mant, fj.exp+fj.mant; wi != wj {
			return wi > wj
		}
		return order[i] < order[j]
	})
	tb := Table{
		ID:    "floats",
		Title: "数值格式",
		// 这个限定语必须写在表上：下面那 15 条**不是** safetensors 的全部取值，
		// 而是本工具认识的全部。规范里还有 F8_E4M3FNUZ / F8_E5M2FNUZ /
		// F8_E8M0 / F4 / C64 —— 遇到它们，解析器报"未知 dtype"。
		// 只列 15 条而不说明，用户会以为文件里不可能出现别的。
		Notes: "safetensors 小节列的是**本工具认识**的取值（15 个）；" +
			"规范共 20 个，另有 F8_E4M3FNUZ / F8_E5M2FNUZ / F8_E8M0 / F4 / C64，" +
			"遇到会报「未知的 safetensors dtype」。",
	}
	for _, id := range order {
		f, ok := floatFormatByID[id]
		if !ok {
			// 表与定义脱节时必须显式失败，不能静默少一条 ——
			// 少一条的表现是界面上查不到，而没有任何东西会红
			panic("floatsTable 引用了未定义的位布局: " + id)
		}
		tb.Entries = append(tb.Entries, Entry{
			ID:    id,
			Title: f.name,
			Fields: []Field{
				{"位布局", fmt.Sprintf("%d 符号 + %d 指数 + %d 尾数", f.sign, f.exp, f.mant)},
				{"指数偏置", fmt.Sprintf("%d", (1<<(f.exp-1))-1)},
				{"最大值", humanFloat(maxOf(f))},
				{"最小正规数", humanFloat(minNormal(f.exp))},
				{"相对精度", humanFloat(math.Ldexp(1, -f.mant))},
				{"别称", strings.Join(f.aliases, "、")},
			},
			Notes: f.notes,
		})
	}
	// safetensors 的 dtype 名并进来：同一个东西在不同格式里拼写不同，
	// 用户看到的是文件里的那个拼写。
	//
	// 清单与解析器共用 model.SafetensorsDtypes —— 两份清单漂移的表现是
	// 速查表里少一行，没有任何东西会红。
	for _, d := range model.SafetensorsDtypes() {
		st := string(d)
		fields := []Field{{"safetensors 写法", st}}
		if n, ok := d.ByteSize(); ok {
			fields = append(fields, Field{"字节数", strconv.FormatInt(n, 10)})
		}
		e := Entry{
			ID:     "dtype:" + st,
			Title:  "safetensors dtype: " + st,
			Fields: fields,
			Notes:  stDtypeNotes[st],
		}
		if e.Notes == "" {
			e.Notes = "safetensors 头部的 dtype 字段取值。"
		}
		// 浮点类型再挂上对应的位布局：用户查 "F8_E4M3 是什么" 时，
		// 想知道的是指数/尾数怎么分，而不只是几个字节
		//
		// 键统一是 "float:" + Dtype 字面量 —— 与上面 order 里的写法同源，
		// 拼错一个字符的表现是这一栏**静默消失**（ok 为 false 就跳过），
		// 所以 IDs 不允许各写各的，见 TestFloats_ID与Dtype一致
		if f, ok := floatFormatByID["float:"+st]; ok {
			e.Fields = append(e.Fields,
				Field{"对应", f.name},
				Field{"最大值", humanFloat(maxOf(f))})
			e.SeeAlso = []string{"float:" + st}
		}
		tb.Entries = append(tb.Entries, e)
	}
	return tb
}
