// Package ref 提供静态速查表数据。
//
// ## 什么该算、什么该写
//
// 凡是可以由别处的权威数据算出来的，一律**算**：
// 位宽与块结构来自 model.Dtype，浮点的极值与精度来自位布局。
// 只有编辑性内容（这个格式是干什么的、典型损失多少、为什么这么设计）
// 才写成字面量。
//
// 理由：手抄的数字漂移时不会有任何东西失败 —— 界面照样显示，
// 只是显示的是一年前的答案。算出来的不可能漂移。
package ref

import (
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// Field 是条目里的一个「字段 → 值」。
//
// 用有序的切片而不是 map：速查表的展示顺序是内容的一部分
// （先看位宽再看用途），而 map 的遍历顺序是随机的。
type Field struct {
	Key   string
	Value string
}

// Entry 是一条速查表条目。
type Entry struct {
	// ID 稳定且唯一，是上下文关联的键，形如 "quant:Q4_K"、"key:general.file_type"。
	ID    string
	Title string
	// Fields 按展示顺序排列。
	Fields []Field
	// Notes 是补充说明，纯文本。
	Notes string
	// SeeAlso 是相关条目的 ID。**不校验存在性** —— 指向未来的条目是允许的
	//（跳过去显示"没有这条"比拒绝开局好），但测试会断言它不指向自己。
	SeeAlso []string
}

// Field 按 key 取值；不存在返回空串。
func (e Entry) Field(key string) string {
	for _, f := range e.Fields {
		if f.Key == key {
			return f.Value
		}
	}
	return ""
}

// Table 是一组条目。
type Table struct {
	ID    string
	Title string
	// Notes 是**整张表**的限定语（如"这里列的是本工具支持的子集"）。
	// 有它才不会被迫把同一句话抄进每一条 —— 抄进去的后果是
	// 改一处漏一处，而且用户要在每条里读一遍重复的话。
	Notes   string
	Entries []Entry
}

// tablesOnce 缓存一次构造的结果 —— 界面会反复调用 Tables()，
// 每次都重建等于每帧重算全表。
//
// **用 sync.OnceValue 而不是裸的 `if cache == nil`**：④b 的界面会一边
// 渲染速查表一边刷新模型库（discover 的注释里也明说会让调用方起 goroutine），
// 两个 goroutine 同时第一次调 Tables() 就是一次数据竞争。
// 没有同步的话 -race 会报（见 TestTables_并发安全）。
var tablesOnce = sync.OnceValue(func() []Table {
	return []Table{
		floatsTable(),
		quantsTable(),
		ggufKeysTable(),
		tensorNamingTable(),
	}
})

// Tables 返回全部速查表，顺序固定。
//
// 返回的切片与其中的条目都是**只读**的：它们是共享的，改它会污染
// 后续所有调用方（界面上表现为"刷新几次之后表就变了"）。
func Tables() []Table { return tablesOnce() }

// ByID 按条目 ID 取一条。
func ByID(id string) (Entry, bool) {
	for _, tb := range Tables() {
		for _, e := range tb.Entries {
			if e.ID == id {
				return e, true
			}
		}
	}
	return Entry{}, false
}

// Find 按关键词搜条目。大小写不敏感，空白关键词返回空。
//
// 搜的是**全文**（标题 + 全部字段的键与值 + Notes），不只是标题 ——
// 用户记得住的是"那个讲子块的"，不是条目的标题。
func Find(keyword string) []Entry {
	kw := strings.ToLower(strings.TrimSpace(keyword))
	if kw == "" {
		return nil
	}
	var out []Entry
	for _, tb := range Tables() {
		for _, e := range tb.Entries {
			if e.matches(kw) {
				out = append(out, e)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func (e Entry) matches(kw string) bool {
	if strings.Contains(strings.ToLower(e.ID), kw) ||
		strings.Contains(strings.ToLower(e.Title), kw) ||
		strings.Contains(strings.ToLower(e.Notes), kw) {
		return true
	}
	for _, f := range e.Fields {
		if strings.Contains(strings.ToLower(f.Key), kw) ||
			strings.Contains(strings.ToLower(f.Value), kw) {
			return true
		}
	}
	return false
}

// humanFloat 把浮点数打成可读字符串。
//
// 它同时被**生成**表内容的代码与**测试**使用 —— 两处用同一个格式化函数，
// 对照才有意义（一个用 %v 一个用 %g 会得出必然不同的字符串，
// 那样测试会因为格式化差异而假失败，掩盖真正的漂移）。
func humanFloat(v float64) string {
	switch {
	case v == 0:
		return "0"
	case math.IsInf(v, 1):
		return "+Inf"
	case math.IsInf(v, -1):
		return "-Inf"
	case math.IsNaN(v):
		return "NaN"
	}
	// 极值与次正规数需要科学计数，普通量级用定点。
	// 阈值的选取：1e-4 以下的定点会打成一串 0，1e6 以上的定点读不出量级
	if a := math.Abs(v); a >= 1e6 || a < 1e-4 {
		return strconv.FormatFloat(v, 'g', 6, 64)
	}
	return strconv.FormatFloat(v, 'g', 10, 64)
}
