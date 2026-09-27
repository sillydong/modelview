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
