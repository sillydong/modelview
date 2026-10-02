package ref

import (
	"strconv"
	"strings"
	"testing"
)

// spec §9 要的是「GGML 类型码 0–30 全表」—— 把它机械化。
//
// 原来只有量化条目带码（quants 表那一栏）：0/1/28/30 落在浮点表里没有
// 编号，24–27 两边都没有，4/5 完全没有条目。
func TestGGMLTypes_0到30全覆盖(t *testing.T) {
	var missing []string
	for code := 0; code <= 30; code++ {
		if _, ok := ByID("ggmltype:" + strconv.Itoa(code)); !ok {
			missing = append(missing, strconv.Itoa(code))
		}
	}
	if len(missing) > 0 {
		t.Errorf("这些类型码在速查表里查不到：%s", strings.Join(missing, ", "))
	}
}

// **两个命名空间不能混**：ggml_type 的 4 是 Q4_2（上游已删），
// general.file_type 的 4 是 MOSTLY_Q4_1_SOME_F16。同号不同义。
//
// 没有这条的话，解析器报「未知的 GGML 类型码 4」时用户搜 4 只会命中
// file_type 那条，得到一个不相干的答案。
func TestGGMLTypes_与file_type的码不混淆(t *testing.T) {
	ggml, ok := ByID("ggmltype:4")
	if !ok {
		t.Fatal("ggmltype:4 不存在")
	}
	ft, ok := ByID("filetype:4")
	if !ok {
		t.Fatal("filetype:4 不存在")
	}
	if !strings.Contains(ggml.Title, "Q4_2") {
		t.Errorf("ggmltype:4 应当是 Q4_2，得到 %q", ggml.Title)
	}
	if !strings.Contains(ft.Title, "MOSTLY_Q4_1_SOME_F16") {
		t.Errorf("filetype:4 应当是 MOSTLY_Q4_1_SOME_F16，得到 %q", ft.Title)
	}
	// 搜裸数字时必须**两条都能搜到** —— 只出一条就是把用户引向错的命名空间
	var sawGGML, sawFileType bool
	for _, e := range Find("4") {
		if e.ID == "ggmltype:4" {
			sawGGML = true
		}
		if e.ID == "filetype:4" {
			sawFileType = true
		}
	}
	if !sawGGML || !sawFileType {
		t.Errorf("搜 4 应当同时命中两个命名空间，得到 ggml=%v filetype=%v", sawGGML, sawFileType)
	}
}

// 已删的两个码要有**说明**，不能只留一个名字 —— 用户看到"Q4_2"会以为
// 能用，而答案是"上游删了，现代文件里不该出现"。
func TestGGMLTypes_已删的码要说明(t *testing.T) {
	for _, code := range []string{"4", "5"} {
		e, ok := ByID("ggmltype:" + code)
		if !ok {
			t.Fatalf("ggmltype:%s 不存在", code)
		}
		if !strings.Contains(e.Notes, "删除") {
			t.Errorf("ggmltype:%s 的说明里没写它是已删的：%q", code, e.Notes)
		}
	}
}
