package tui

import (
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"

	"github.com/sillydong/modelview/internal/model"
)

// ansiRE 匹配 ANSI 转义序列。
var ansiRE = regexp.MustCompile(`\x1b\[[0-9;]*m`)

// plainText 去掉 ANSI 转义。
//
// 断言"Q4_K 出现在 Q6_K 之前"必须用纯文本：样式是逐段包在转义码里的，
// 带着转义码找子串，找到的位置是**转义码的位置**，
// 而转义码的顺序未必等于可见文本的顺序。
func plainText(s string) string { return ansiRE.ReplaceAllString(s, "") }

// quantModel 造一个有指定类型分布的模型。
//
// 字节数按类型真实的 bit/权重 算，让"整体位宽"那一条能算出确定的值。
func quantModel(byDtype map[model.Dtype][2]int64) *model.Model {
	m := &model.Model{Path: "/x/m.gguf", Format: model.FormatGGUF, Version: "v3"}
	i := 0
	for d, spec := range byDtype {
		count, elems := spec[0], spec[1]
		for range count {
			m.Tensors = append(m.Tensors, &model.Tensor{
				Name: fmt.Sprintf("blk.%d.w", i), Dtype: d,
				Dims: []int64{elems}, ParamCount: elems,
				ByteSize: int64(float64(elems) * d.BitsPerWeight() / 8),
			})
			i++
		}
	}
	return m
}

// 每个类型一行，且**按占用字节降序** —— 用户想知道的是"钱花在哪"。
//
// 顺序必须确定：来源是 map，遍历顺序随机，不排序的话同一屏
// 每次刷新行的先后都在变，看起来就是字在抖。
func TestQuantDist_按占用降序(t *testing.T) {
	m := quantModel(map[model.Dtype][2]int64{
		model.DtypeQ4K: {5, 1 << 20}, // 4.5 bpw → 占用最大
		model.DtypeQ6K: {3, 1 << 20}, // 6.5625 bpw → 次之
		model.DtypeF32: {1, 1024},    // 32 bpw 但元素极少 → 最小
	})
	out := plainText(quantDistText(m))

	iQ4 := strings.Index(out, "Q4_K")
	iQ6 := strings.Index(out, "Q6_K")
	iF32 := strings.Index(out, "F32")
	if iQ4 < 0 || iQ6 < 0 || iF32 < 0 {
		t.Fatalf("没列出全部类型:\n%s", out)
	}
	if iQ4 >= iQ6 || iQ6 >= iF32 {
		t.Errorf("顺序不是按占用降序（Q4_K=%d Q6_K=%d F32=%d）:\n%s", iQ4, iQ6, iF32, out)
	}
}

// 整体 bit/权重 = 总字节 × 8 / 总参数，含块头开销。
//
// **用混合档位的模型**：单类型的模型整体位宽恰好等于那一行的位宽，
// 断言会退化成"那一行显示了 4.5"，验不到整体那条算术。
func TestQuantDist_整体位宽(t *testing.T) {
	m := quantModel(map[model.Dtype][2]int64{
		model.DtypeQ4K: {1, 1 << 16}, // 4.5 bpw → 36864 字节
		model.DtypeF32: {1, 1 << 16}, // 32 bpw → 262144 字节
	})
	// (36864 + 262144) × 8 / 131072 = 18.25
	out := plainText(quantDistText(m))
	if !strings.Contains(out, "18.25") {
		t.Errorf("整体 bit/权重 不是两个档位之间（want 18.25）:\n%s", out)
	}
}

// **位宽那一列必须用精确格式器** —— 这条测的是"这一栏用对了格式器"，
// 不是"格式器本身对"（后者由 render 的 TestBitsPerWeight 管）。
//
// **断言必须挑一个两种格式器输出不同的值。** 计划最初写的是
// `strings.Contains(out, "6.5625")`，那是**空转的**：Task 5 执行时实测
// `humanize.Float(6.5625)` 也是 `"6.5625"`，与 `render.BitsPerWeight`
// 逐字节相同 —— 换错格式器照样绿。
//
// 能区分的是**整数位宽**：`BitsPerWeight(4.5)` = `"4.5"`，
// 而 `Float(4.5)` = `"4.5000"`（它补尾零）。
// 而 `%.4g` 那条错路要另一个值才抓得到：`%.4g(6.5625)` = `"6.562"`。
// 所以这条测试两个方向都验：
func TestQuantDist_位宽用精确格式器(t *testing.T) {
	// ① 不许补尾零（humanize.Float 那条错路）
	m := quantModel(map[model.Dtype][2]int64{model.DtypeQ4K: {1, 1 << 16}})
	out := plainText(quantDistText(m))
	if strings.Contains(out, "4.5000") {
		t.Errorf("位宽被补了尾零 —— 这一栏用了 humanize.Float:\n%s", out)
	}

	// ② 精确值不许被打短（%.4g 那条错路）
	//
	// **必须用混合模型**：单类型时"整体 bit/权重"恰好等于那一行的位宽，
	// 而整体那行**总是**走 BitsPerWeight —— 行内换成 %.4g 之后，
	// `strings.Contains(out, "6.5625")` 照样为真，断言就是空转的。
	// 混一个 F32 进来，两者就分开了（实测：行内 6.562、整体 19.2812）。
	m2 := quantModel(map[model.Dtype][2]int64{
		model.DtypeQ6K: {1, 1 << 16},
		model.DtypeF32: {1, 1 << 16},
	})
	if out := plainText(quantDistText(m2)); !strings.Contains(out, "6.5625") {
		t.Errorf("Q6_K 的位宽被打了短 —— 这一栏用了 %%.4g:\n%s", out)
	}
}

// 文件声明的档位要显示出来，但不判定它与实际分布是否一致。
func TestQuantDist_显示声明档位(t *testing.T) {
	m := quantModel(map[model.Dtype][2]int64{model.DtypeQ4K: {2, 1 << 16}})
	m.Metadata = []model.MetaKV{
		{Key: "general.file_type", Value: "15", Raw: uint32(15)},
	}
	out := plainText(quantDistText(m))
	if !strings.Contains(out, "15") {
		t.Errorf("没显示 file_type 的码:\n%s", out)
	}
	// **断言的是 e.Title 里的枚举名，不是 display 列**：同一个 file_type
	// 在元数据栏里显示的也是这个串（modelview_test.go 有同样的断言），
	// 两处必须一致 —— 而这个项目从 internal/render 立项起就在对抗
	// "同一件事两种渲染，分叉时不报错"。
	//
	// display 列（"Q4_K - Medium"）看着更好读，但它**不是每条都有**：
	// 废弃档位（4/5/6）没有这个 field，Entry.Field 查不到时静默返回空串，
	// gpt-oss:20b（file_type 恰好是 4）就会显示成 `[]`。
	if !strings.Contains(out, "MOSTLY_Q4_K_M") {
		t.Errorf("没显示档位名:\n%s", out)
	}
}

// 没有 file_type 元数据时不该凭空造一行 —— "没有"要说成没有。
func TestQuantDist_没有声明档位(t *testing.T) {
	m := quantModel(map[model.Dtype][2]int64{model.DtypeQ4K: {2, 1 << 16}})
	out := plainText(quantDistText(m))
	if strings.Contains(out, "file_type") {
		t.Errorf("没有 file_type 却显示了那一行:\n%s", out)
	}
}

// 空模型不能除零，也不能给一张空表。
func TestQuantDist_空模型(t *testing.T) {
	m := &model.Model{Path: "/x/m.gguf"}
	out := plainText(quantDistText(m))
	if !strings.Contains(out, "没有张量") {
		t.Errorf("空模型没给说明:\n%s", out)
	}
}

// **80 列终端下这一栏不能被截，且表头与数据行必须对齐。**
//
// 80×24 是 spec 定的最小终端。实测（按字节补齐的旧版）：表头 85 列、
// 数据行 73 列 —— 根视图的 truncateLines 会切掉最后 5 列，
// 恰好是 "bit/权重" 这个列标题的尾部。
//
// 两条断言各守一件事：
//   - 不超宽：否则终端自动折行、整个界面错位
//   - 表头与数据行等宽：中文标签 3 字节只占 2 列，按字节补齐时
//     表头会比数据行宽 12 列，列标题与数据**对不上**
func TestQuantDist_表头不超宽且与数据行等宽(t *testing.T) {
	m := quantModel(map[model.Dtype][2]int64{
		model.DtypeQ4K: {3, 1 << 16},
		model.DtypeF32: {1, 1 << 16},
	})
	out := quantDistText(m)

	// 这一栏是右栏内容，可用宽度 = 总宽 − nav 宽 − 1。
	// **67 = 80 列终端 − nav 最窄 12 列 − 1 个分隔空格**，是个与数据
	// 无关的上界：nav 最窄就是 12（实测本模型是 `元数据 (0)` 那个标签
	// 定的 —— 标签 10 列 + 前缀 2 列；元数据数是 0 时最小），
	// 所以超过 67 列的行**任何模型都放不下**，不是"本模型量出来是 67"。
	// 实测新版：表头与数据行都是 45 列，离上界还有余量。
	for _, line := range strings.Split(out, "\n") {
		if w := lipgloss.Width(line); w > 67 {
			t.Errorf("有一行宽 %d 列 —— 80 列终端下这一栏拿不到这么宽:\n%s", w, line)
		}
	}

	// 表头与数据行等宽 ⇒ 各列边界对齐
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	var header, data string
	for _, l := range lines {
		if strings.Contains(l, "类型") {
			header = l
		}
		if strings.Contains(l, "Q4_K") || strings.Contains(l, "F32") {
			data = l
		}
	}
	if header == "" || data == "" {
		t.Fatalf("没找到表头或数据行:\n%s", out)
	}
	if lipgloss.Width(header) != lipgloss.Width(data) {
		t.Errorf("表头 %d 列、数据行 %d 列 —— 列不对齐:\n%s\n%s",
			lipgloss.Width(header), lipgloss.Width(data), header, data)
	}
}

// **这一栏的每一行都要落在真实可用的宽度里**（80×24 是 spec 定的最小终端）。
//
// 这条守的不是表格（那是上一条的事），而是 `文件声明` 那节 ——
// 它是这一栏里**唯一随文件内容变长**的文本：档位码能翻译出多长的条目名
// 由文件决定，表格那几列的名字都是固定的。
//
// 上界 **58 = 80 列终端 − nav 最宽可达 15 列 − 1 个分隔空格 − 6 列余量**。
// nav 最宽是 `元数据 (1000)` 那种四位数的条目数：标签 13 列 + 前缀 2 列；
// 实测本仓三个真实模型的 nav 是 12–13（可用 66–67），58 说的是
// "最坏情况下也还剩 6 列"，不是当前值。
//
// 五条用例对应文件里可能写的东西，**最长的那条是废弃档位**：
// `MOSTLY_Q4_1_SOME_F16` 光名字就 21 列，条目名整行实测 **57 列**
// （GUESSED 那条警告另起一行后 36 列，不再是瓶颈）。
// **不重复印 `key = value` 之前它是 67 列** —— 比 nav=13 时的可用宽度
// 还宽 1 列，根视图的 truncateLines 会把收尾的 `]` 切掉（真终端上实测过：
// qwen2.5:3b 的 nav 是 13，原始字节里 `]` 从来没有被写出去）。
func TestQuantDist_文件声明行留有余量(t *testing.T) {
	cases := []struct {
		name string
		kv   model.MetaKV
		// guessed 表示这条用例**必须**走到 GUESSED 分支：那条路径在别处
		// 没有任何测试走过，警告没出现的话下面的宽度断言就是空转的
		guessed bool
	}{
		{"常见档位", model.MetaKV{Key: "general.file_type", Value: "15", Raw: uint32(15)}, false},
		{"已废弃档位", model.MetaKV{Key: "general.file_type", Value: "4", Raw: uint32(4)}, false},
		{"上游猜的档位", model.MetaKV{Key: "general.file_type", Value: "1028", Raw: uint32(1028)}, true},
		{"表里没有的码", model.MetaKV{Key: "general.file_type", Value: "999", Raw: uint32(999)}, false},
		{"值不是整数", model.MetaKV{Key: "general.file_type", Value: "abc", Raw: "abc"}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := quantModel(map[model.Dtype][2]int64{
				model.DtypeQ4K: {3, 1 << 16},
				model.DtypeF32: {1, 1 << 16},
			})
			m.Metadata = []model.MetaKV{c.kv}
			out := plainText(quantDistText(m))

			// 前提一：那一节真的渲染出来了 —— 没渲染出来的话上界是空转的
			if !strings.Contains(out, "general.file_type") {
				t.Fatalf("文件声明那节没渲染出来，这条用例的前提不成立:\n%s", out)
			}
			// 前提二：该走 GUESSED 的必须真的走了
			if got := strings.Contains(out, "上游猜的"); got != c.guessed {
				t.Fatalf("GUESSED 警告出现 = %v，want %v（这条用例的前提不成立）:\n%s",
					got, c.guessed, out)
			}

			for _, line := range strings.Split(out, "\n") {
				if w := lipgloss.Width(line); w > 58 {
					t.Errorf("有一行宽 %d 列 —— 80 列终端（nav 最宽 15）下余量不足:\n%s",
						w, line)
				}
			}
		})
	}
}
