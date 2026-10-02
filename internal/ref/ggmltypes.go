package ref

import (
	"sort"
	"strconv"

	"github.com/sillydong/modelview/internal/model"
)

// removedCodes 是**上游已删除**的 GGML 类型码。
//
// 4/5 是 Q4_2/Q4_3，llama.cpp 早就把它们删了，现代文件里不该出现。
// 留着条目是为了让"GGML 类型码 4 是什么"有一个**正确答案**：没有条目
// 的话，用户搜 4 只会命中 general.file_type 的 4（MOSTLY_Q4_1_SOME_F16），
// 那是**另一个命名空间**的值，同号不同义 —— 解析器报的"未知的 GGML
// 类型码 4"会被引到一个不相干的答案上。
var removedCodes = map[int]string{
	4: "Q4_2",
	5: "Q4_3",
}

// ggmlTypeTable 是「GGML 类型码 → 类型名」的全表。
//
// spec §9 要的是「GGML 类型码 0–30 全表」，而原先只有**量化**条目带码
// （quants 表里那一栏）：码 0/1/28/30 落在浮点表里却没有编号，
// 24–27（整数）两边都没有，4/5 完全没有条目。
//
// **与 general.file_type 是两套编号**：那边是"整份文件按哪个档位产出"，
// 这边是"这个张量用什么块布局"。数字重叠而含义无关 —— 实测搜
// `filetype:4` 得到 MOSTLY_Q4_1_SOME_F16，而 ggml_type 4 是 Q4_2。
// 两套的 ID 前缀不同（filetype: / ggmltype:），搜索结果里一眼分得开。
//
// 名字取自 model（**唯一出处**），只有上游已删的那两个写死在这里 ——
// 它们不在 model.Dtype 里，这是刻意的：解析器遇到它们要走"未收录类型"
// 那条告警路径，而不是假装认识。
func ggmlTypeTable() Table {
	type item struct {
		code int
		name string
		ok   bool // 名字来自 model（false 表示是已删的历史类型码）
	}
	var items []item
	for d := range model.AllDtypes() {
		code, ok := d.GGMLCode()
		if !ok {
			// GGMLCode 返回 false 的是 safetensors 独有的浮点格式
			//（F8_E4M3 等）与无符号整数：它们**没有** GGML 类型码，
			// 编一个出来就是撒谎
			continue
		}
		items = append(items, item{code: int(code), name: string(d), ok: true})
	}
	for code, name := range removedCodes {
		items = append(items, item{code: code, name: name})
	}
	sort.Slice(items, func(i, j int) bool { return items[i].code < items[j].code })

	// **不能用 ByID**：它走 Tables() → tablesOnce，而这个函数本身就是
	// 那个 OnceValue 里被调的 —— sync.Once 在函数返回前不放锁，
	// 于是**自己等自己，直接死锁**（不是报错，是界面永远刷不出来）。
	// 直接构造那两张表来查 ID：它们不经过 Once。
	known := make(map[string]bool)
	for _, tb := range []Table{floatsTable(), quantsTable()} {
		for _, e := range tb.Entries {
			known[e.ID] = true
		}
	}

	entries := make([]Entry, 0, len(items))
	for _, it := range items {
		code := strconv.Itoa(it.code)
		e := Entry{
			ID:    "ggmltype:" + code,
			Title: "GGML 类型码 " + code + "：" + it.name,
			Fields: []Field{
				{"类型码", code},
				{"类型名", it.name},
			},
		}
		if it.ok {
			// **位宽从 ref 自己的 bpwOf 取，不用 Dtype.BitsPerWeight()**：
			// IQ 系列的 BitsPerWeight() 是 0（model 那张表没填它们），
			// 真实位宽在 ref 这边 —— 用后者的话那九个条目会**静默少一栏**。
			// 与 quants 表同一个来源，同一个数。
			if bpw, ok := bpwOf(it.name); ok {
				e.Fields = append(e.Fields, Field{"位宽", humanFloat(bpw) + " bit/权重"})
			}
			// 交叉引用到量化/浮点那两张表里的详条目 ——
			// 这张表只回答"码 N 是什么"，细节在那边
			if known["quant:"+it.name] {
				e.SeeAlso = append(e.SeeAlso, "quant:"+it.name)
			}
			if known["float:"+it.name] {
				e.SeeAlso = append(e.SeeAlso, "float:"+it.name)
			}
		} else {
			e.Notes = "**上游已删除的历史类型码**，现代文件里不该出现。" +
				"解析器遇到它会给「未知的 GGML 类型码」告警，并把该张量标为大小未知。"
		}
		entries = append(entries, e)
	}

	return Table{
		ID:    "ggml-types",
		Title: "GGML 类型码",
		Notes: "**与 `general.file_type` 是两套编号**：这里说的是「这个张量用什么块布局」，那边说的是「整份文件按哪个档位产出」。" +
			"数字重叠而含义无关 —— 例如 4 在这里是已删除的 Q4_2，" +
			"在那边是 MOSTLY_Q4_1_SOME_F16。",
		Entries: entries,
	}
}
