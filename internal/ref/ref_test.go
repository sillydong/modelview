package ref

import (
	"strings"
	"sync"
	"testing"
)

// 每张表都要有条目、有唯一 ID，且 ID 之间不重复。
//
// 唯一性不是洁癖：ID 是上下文关联的键（"这个 file_type 值对应哪条速查"），
// 重复的 ID 会让跳转随机命中一个。
func TestTables_结构自洽(t *testing.T) {
	tables := Tables()
	if len(tables) == 0 {
		t.Fatal("没有任何速查表")
	}
	seen := map[string]string{} // id -> 它在哪张表里
	entries := 0
	for _, tb := range tables {
		if tb.ID == "" || tb.Title == "" {
			t.Errorf("表 %+v 缺 ID 或 Title", tb)
		}
		if len(tb.Entries) == 0 {
			t.Errorf("表 %s 没有条目", tb.ID)
		}
		for _, e := range tb.Entries {
			entries++
			if e.ID == "" || e.Title == "" {
				t.Errorf("表 %s 有条目缺 ID 或 Title: %+v", tb.ID, e)
			}
			if prev, dup := seen[e.ID]; dup {
				t.Errorf("条目 ID %q 重复（%s 与 %s）—— 跳转会随机命中一个",
					e.ID, prev, tb.ID)
			}
			seen[e.ID] = tb.ID
			// 条目必须全文可搜：ID 与标题都要能被搜到
			if got := Find(e.ID); len(got) == 0 {
				t.Errorf("按 ID %q 搜不到自己", e.ID)
			}
			// 关联条目不允许指向自己
			for _, ref := range e.SeeAlso {
				if ref == e.ID {
					t.Errorf("%s 的 SeeAlso 指向了自己", e.ID)
				}
			}
			// **字段不能有空的键或值**。
			//
			// "覆盖率 100%" 原先说的是"有条目"，不是"有内容" ——
			// 实测 93 个真实键里有 2 个命中的条目释义栏是空的，
			// 而覆盖率测试照样报 100%。空的释义栏与"查不到"一样没用，
			// 区别只是它看起来像查到了。
			for _, f := range e.Fields {
				if f.Key == "" {
					t.Errorf("%s 有个字段没有键名", e.ID)
				}
				if strings.TrimSpace(f.Value) == "" {
					t.Errorf("%s 的字段 %q 是空的 —— 用户查到的是个空壳", e.ID, f.Key)
				}
			}
		}
	}
	if entries == 0 {
		t.Fatal("一个条目都没有")
	}
	// 反向门禁：条目总数掉了说明表被删了
	const wantEntries = 249
	if entries != wantEntries {
		t.Errorf("条目总数 %d，预期 %d —— 增删条目时同步改这里", entries, wantEntries)
	}
}

// 按 ID 取条目：命中与不命中都要正确。
func TestByID(t *testing.T) {
	e, ok := ByID("quant:Q4_K")
	if !ok {
		t.Fatal("找不到 quant:Q4_K")
	}
	if e.Title == "" {
		t.Error("条目没有标题")
	}
	if _, ok := ByID("不存在的ID"); ok {
		t.Error("不存在的 ID 不该命中")
	}
}

// 搜索：关键词要能命中标题、也要能命中正文。
func TestFind(t *testing.T) {
	// 标题命中
	if got := Find("Q4_K"); len(got) == 0 {
		t.Error("搜 Q4_K 没有结果")
	}
	// 大小写不敏感
	lower := Find("q4_k")
	upper := Find("Q4_K")
	if len(lower) != len(upper) {
		t.Errorf("大小写敏感了：%d vs %d", len(lower), len(upper))
	}
	// 字段命中：这个关键词**只出现在字段里**（实测 ID/标题/Notes 都是 0 命中）。
	//
	// 早先这里用的是"子块"，而它其实是被 Notes 命中的 ——
	// 变异验证时把字段那段改成只做等值比较，测试照样绿，
	// 才发现"搜索覆盖字段"这条根本没被验到。
	if got := Find("指数偏置"); len(got) == 0 {
		t.Error("搜只在字段里出现的词没有结果 —— 搜索没覆盖字段？")
	}
	// 字段的值也要能搜到（键与值是两处，漏一处就可能漏另一处）
	if got := Find("1.5625 bpw"); len(got) == 0 {
		t.Error("搜只在字段值里出现的词没有结果")
	}
	// 正文命中：Notes 也要覆盖
	if got := Find("按后缀匹配的释义"); len(got) == 0 {
		t.Error("搜只在 Notes 里出现的词没有结果 —— 搜索没覆盖正文？")
	}
	// 空关键词返回空，不是全量
	if got := Find("   "); len(got) != 0 {
		t.Errorf("空白关键词返回了 %d 条，应为 0", len(got))
	}
}

// **SeeAlso 图必须无环** —— 环会让界面栈溢出，而且失败点离原因很远。
//
// `tui.NewEntryView` 对 SeeAlso 是**急切递归**构造子视图
// （`pushCmd(NewEntryView(m, target))` 在构造时就求值），
// 所以 A→B→A 会让"打开 A"直接栈溢出 —— **不是显示错，是崩**。
//
// 既有的 `ref_test.go` 只挡了"指向自己"，挡不住 A→B→A 或更长的环。
//
// 实测（2026-10）：249 条、有 SeeAlso 的 157 条、边 157、**最长链 0**、环 0 ——
// 今天是安全的。这条测试是给**将来往数据里加 SeeAlso 的人**的：
// 加一个跨条目的互指就会红，而不是等到用户点开那一页崩掉。
func TestSeeAlso_无环(t *testing.T) {
	const (
		white = 0 // 还没进过
		gray  = 1 // 在当前这条 DFS 路径上
		black = 2 // 这条路径已经走完
	)
	color := map[string]int{}
	maxDepth, edges := 0, 0

	var dfs func(id string, depth int, path []string)
	dfs = func(id string, depth int, path []string) {
		if depth > maxDepth {
			maxDepth = depth
		}
		color[id] = gray
		e, ok := ByID(id)
		if !ok {
			// 查不到的目标**是允许的**（ref.Entry 的注释：指向未来的条目是允许的），
			// 界面上显示成"（速查表里没有这条）"。它没有出边，走不到环。
			color[id] = black
			return
		}
		for _, next := range e.SeeAlso {
			edges++
			switch color[next] {
			case gray:
				t.Fatalf("%s → %s 成环：%v —— 急切递归构造子视图时这会栈溢出",
					id, next, append(path, next))
			case white:
				dfs(next, depth+1, append(path, next))
			}
		}
		color[id] = black
	}

	for _, tb := range Tables() {
		for _, e := range tb.Entries {
			if color[e.ID] == white {
				dfs(e.ID, 0, []string{e.ID})
			}
		}
	}
	// **最长链只打日志、不判红**：链长不致命，它只影响构造代价 ——
	// 每深一层就多构造一份子视图（急切递归构造的是"路径"而不是"节点"，
	// 所以扇出比链长放大得更快）。判红会让一条合理的扩展
	//（比如 Q4_K → Q6_K → F16）卡住，而它并没有坏处。
	// 但 0 变成几十时值得看一眼：那说明有人把条目串成了一条链。
	t.Logf("SeeAlso 图：最长链 %d、边 %d、条目 %d", maxDepth, edges, len(color))
}

// 并发首次调用 Tables() 必须是安全的。
//
// ④b 的界面会一边渲染速查表一边刷新模型库，两个 goroutine 同时第一次
// 调 Tables() —— 懒加载那个包级变量如果没有同步，-race 下就是数据竞争。
// 这条测试要配合 `go test -race` 才有意义（普通模式跑不出竞争）。
func TestTables_并发安全(t *testing.T) {
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				if got := len(Tables()); got == 0 {
					t.Error("Tables() 返回空")
					return
				}
			}
		}()
	}
	wg.Wait()
}
