package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/ref"
)

// selectMetaMsg 让 ModelView 把光标移到某一条元数据上。
//
// 由 popToMsg 转发，所以 ModelView 收到它时**已经在栈顶了**。
type selectMetaMsg struct{ index int }

// occurrences 返回一个条目在当前模型里出现的位置。
//
// **正向与反向必须是同一个判定**：`key:` 这一支用的是
// ModelView.metaEntry —— 与元数据栏决定"挂不挂 ◂ 记号"的是同一个函数。
// 各写一套的话会出现"有记号但跳不过来"，而两边各自都自洽，
// 没有任何东西会红。
//
// m 为 nil 时返回 nil：没有模型上下文时"不知道"，而不是"没有"。
func occurrences(m *model.Model, e ref.Entry) []entryTarget {
	if m == nil {
		return nil
	}
	switch {
	case strings.HasPrefix(e.ID, "key:"),
		strings.HasPrefix(e.ID, "keysuffix:"),
		strings.HasPrefix(e.ID, "filetype:"):
		return metaOccurrences(m, e)
	case strings.HasPrefix(e.ID, "quant:"),
		strings.HasPrefix(e.ID, "dtype:"):
		// dtype 名就是 ID 冒号后面那一段
		name := e.ID[strings.Index(e.ID, ":")+1:]
		return dtypeOccurrences(m, model.Dtype(name))
	case strings.HasPrefix(e.ID, "tensor:"):
		return segmentOccurrences(m, e)
	}
	return nil
}

// metaOccurrences 找元数据里指向这条的那些行。
func metaOccurrences(m *model.Model, e ref.Entry) []entryTarget {
	mv := NewModelView(m)
	var out []entryTarget
	for i, kv := range m.Metadata {
		got, ok := mv.metaEntry(kv)
		if !ok || got.ID != e.ID {
			continue
		}
		idx := i // 闭包捕获：直接用 i 的话所有目标都会指向最后一行
		out = append(out, entryTarget{
			group: "在本模型中",
			label: fmt.Sprintf("元数据 %s = %s", kv.Key, kv.Value),
			ok:    true,
			cmd: func() tea.Msg {
				return popToMsg{msg: selectMetaMsg{index: idx}}
			},
		})
	}
	return out
}

// dtypeOccurrences 找用这个类型的张量。
func dtypeOccurrences(m *model.Model, d model.Dtype) []entryTarget {
	var n int
	for _, tn := range m.Tensors {
		if tn.Dtype == d {
			n++
		}
	}
	if n == 0 {
		// **0 个就整段不显示**：留一行"0 处"的话用户会以为是自己看漏了
		return nil
	}
	return []entryTarget{{
		group: "在本模型中",
		label: fmt.Sprintf("%d 个张量用了 %s", n, d),
		ok:    true,
		cmd:   pushCmd(NewTensorsViewDtype(m, d)),
	}}
}

// segmentOccurrences 找名字里含这一段（**按段匹配，不是子串**）的张量。
//
// 按子串的话 "q" 会匹配到一大堆名字里恰好含 q 的张量；
// 拆段之后每个段是一个完整的语义单元（blk / #N / attn_q / weight）。
// 归一（#12 → #N）由 ref.LookupTensorSegment 负责 ——
// **不在这里自己写**：那段归一逻辑曾经只存在于测试文件里，
// 于是生产路径上永远查不到带层号的段（ref 包里的原话）。
func segmentOccurrences(m *model.Model, e ref.Entry) []entryTarget {
	seg := e.ID[strings.Index(e.ID, ":")+1:]
	var n int
	for _, tn := range m.Tensors {
		for _, s := range ref.SplitTensorName(tn.Name) {
			if got, ok := ref.LookupTensorSegment(s); ok && got.ID == e.ID {
				n++
				break
			}
		}
	}
	if n == 0 {
		return nil
	}
	return []entryTarget{{
		group: "在本模型中",
		label: fmt.Sprintf("%d 个张量的名字含「%s」", n, seg),
		ok:    true,
		cmd:   pushCmd(NewTensorsViewName(m, seg)),
	}}
}
