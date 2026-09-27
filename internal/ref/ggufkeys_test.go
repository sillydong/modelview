package ref

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// 本机真实模型里出现的所有键，都必须能查到释义。
//
// 清单来自 testdata/real_keys.txt —— **用脚本从真实文件生成**，不是手写的。
// 第一版是手列的 68 个，而实测并集是 93 个：漏掉的正好是最少见的那批
// （gemma4 的 MoE/vision/rope 键、若干个 tokenizer 开关、general.parameter_count），
// 而"最少见的键查不到"恰恰是这个功能最该解决的问题。手写的清单
// 会按"我记得的"收敛，生成的不可能少。
//
// 重新生成：python3 tools/extract_metadata_keys.py
func TestGGUFKeys_覆盖真实模型的键(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "real_keys.txt"))
	if err != nil {
		t.Fatalf("读 testdata/real_keys.txt 失败（用 python3 "+
			"tools/extract_metadata_keys.py 生成）: %v", err)
	}
	var keys []string
	for _, line := range strings.Split(string(raw), "\n") {
		if line = strings.TrimSpace(line); line != "" && !strings.HasPrefix(line, "#") {
			keys = append(keys, line)
		}
	}
	if len(keys) != 108 {
		t.Fatalf("真值只有 %d 个键 —— 夹具被删过或生成得不完整，"+
			"这条测试会静默变成低标准", len(keys))
	}

	missing := 0
	for _, k := range keys {
		if _, ok := ByID("key:" + k); ok {
			continue
		}
		// 允许按后缀退化：架构专属键可以只给后缀条目
		if _, ok := lookupKeyPrefix(k); ok {
			continue
		}
		t.Errorf("键 %s 查不到释义（也没有匹配的后缀条目）", k)
		missing++
	}
	if missing > 0 {
		t.Errorf("%d/%d 个真实键没有释义", missing, len(keys))
	}
}

// file_type 的取值必须能查到 —— 它是界面上最常被问"15 是什么意思"的键。
func TestGGUFKeys_file_type取值(t *testing.T) {
	e, ok := ByID("key:general.file_type")
	if !ok {
		t.Fatal("没有 general.file_type 条目")
	}
	if !strings.Contains(e.Field("含义"), "档位") {
		t.Errorf("file_type 的说明不像在说档位: %q", e.Field("含义"))
	}
	// 取值表必须是可查的条目，不是一段散文
	for _, code := range []string{"0", "1", "7", "15", "41"} {
		if _, ok := ByID("filetype:" + code); !ok {
			t.Errorf("没有 file_type 取值 %s 的条目", code)
		}
	}
	// 已废弃的编号在这里**不再单独断言** —— 它们由
	// TestFileType_废弃编号查得到 覆盖（走 FileTypeByCode，是这里
	// 这个 ByID 版本的超集：多了上游原名、现役档的反面对照、以及
	// 带 GUESSED 标志位的 1028）。同一个决定留两份守卫的话，
	// 将来改废弃清单要在两处改，漏一处会以"两条测试都红了"的形式出现，
	// 容易被当成两个问题。
}

// file_type 的档位名要与实际张量类型分布自洽。
//
// 这是拿真实文件之间的**互相印证**：file_type 是上游写的一个数字，
// 而类型分布是我们自己数出来的。两者对不上就说明要么读错了键、
// 要么档位表抄错了 —— 而单独看任何一个都发现不了。
func TestGGUFKeys_file_type与类型分布自洽(t *testing.T) {
	e, ok := ByID("filetype:15")
	if !ok {
		t.Fatal("没有 file_type=15 的条目")
	}
	// 本机 qwen2.5:3b 实测：file_type=15，Q4_K 216 / Q6_K 37 / F32 181
	if !strings.Contains(e.Title, "MOSTLY_Q4_K_M") {
		t.Errorf("档位 15 的标题 = %q，应为 MOSTLY_Q4_K_M", e.Title)
	}
	// _M 档的特征是"混了更高精度的张量"，纯 _S 档不会出现 Q6_K
	if !strings.Contains(e.Notes, "Q6_K") {
		t.Errorf("档位 15 的说明没有提到混用 Q6_K —— 那正是它与 _S 的区别: %q", e.Notes)
	}
}

// 后缀查找必须从最长试起，否则更具体的后缀会被更泛的盖掉。
func TestGGUFKeys_后缀匹配从最长开始(t *testing.T) {
	// rope.freq_base_swa 比 freq_base 更具体，必须命中前者
	e, exact, ok := lookupKey("gemma4.rope.freq_base_swa")
	if !ok {
		t.Fatal("gemma4.rope.freq_base_swa 查不到")
	}
	if exact {
		t.Error("它不是精确条目，不该标成精确命中")
	}
	if !strings.Contains(e.key, "freq_base_swa") {
		t.Errorf("命中的是 %q，应当命中更具体的 freq_base_swa", e.key)
	}

	// **这一条才是真正区分两种顺序的**：
	// 表里同时有 "attention.causal" 与 "causal"，含义不同（后者只是指向前者）。
	// 从最长开始 → 命中 attention.causal；从最短开始 → 命中 causal。
	//
	// 前一条（freq_base_swa）看似在测顺序，其实两种顺序都命中同一条 ——
	// 变异验证时"改成从最短开始"照样绿，才发现它没在测它声称的东西。
	e3, _, ok := lookupKey("qwen2.attention.causal")
	if !ok {
		t.Fatal("qwen2.attention.causal 查不到")
	}
	if e3.key != "attention.causal" {
		t.Errorf("qwen2.attention.causal 命中了 %q —— 更长、更具体的后缀应当优先于更短的",
			e3.key)
	}

	// 多段后缀：audio.attention.head_count 要能命中 attention.head_count
	e2, _, ok := lookupKey("gemma4.audio.attention.head_count")
	if !ok || e2.key != "attention.head_count" {
		t.Errorf("gemma4.audio.attention.head_count 命中了 %q（ok=%v）", e2.key, ok)
	}

	// 完全不认识的键要明确返回 false，不能给一个空释义
	if _, _, ok := lookupKey("gemma4.完全没见过的键"); ok {
		t.Error("不认识的键不该命中")
	}
}

// 上游"猜出来的"档位值要能查到，且要能标出来。
//
// 老文件缺 general.file_type 时，上游会按张量类型分布反推一个值，
// 并与 LLAMA_FTYPE_GUESSED（1024）按位或。不剥这个位的话 1039 查不到 ——
// 而它其实就是 Q4_K_M。
func TestGGUFKeys_猜出来的档位(t *testing.T) {
	guessed := uint32(15) | 1024 // 1039
	e, ok := FileTypeByCode(guessed)
	if !ok {
		t.Fatal("1039 查不到 —— 没剥 LLAMA_FTYPE_GUESSED 位？")
	}
	if !strings.Contains(e.Title, "MOSTLY_Q4_K_M") {
		t.Errorf("1039 查到了 %q，应为 MOSTLY_Q4_K_M", e.Title)
	}
	if !FileTypeGuessed(guessed) {
		t.Error("1039 应当被标成「上游猜的」")
	}
	// 没置位的不能被误标
	if FileTypeGuessed(15) {
		t.Error("15 不该被标成猜的")
	}
	if e2, ok := FileTypeByCode(15); !ok || e2.ID != e.ID {
		t.Errorf("15 与 1039 应查到同一条: %q vs %q", e2.ID, e.ID)
	}
}

// 每条键释义都必须有"含义"与"取值"。
//
// **这条必须查源头 keyEntry，不能查成品 Entry**：entryOfKey 会把空字段
// 丢掉，所以释义为空的条目在成品里根本看不出问题 ——
// 它只是少了一栏，而覆盖率测试只看"条目在不在"。
// 实测：93 个真实键里有 2 个命中的条目释义栏是空的，
// 覆盖率却报 100%；把某条的释义改成空串，全部测试照样绿。
func TestGGUFKeys_每条都有释义与取值(t *testing.T) {
	check := func(where string, list []keyEntry) {
		for _, e := range list {
			if strings.TrimSpace(e.meaning) == "" {
				t.Errorf("%s 的 %s 没有含义", where, e.key)
			}
			if strings.TrimSpace(e.values) == "" {
				t.Errorf("%s 的 %s 没有取值说明", where, e.key)
			}
			if strings.TrimSpace(e.title) == "" {
				t.Errorf("%s 的 %s 没有标题", where, e.key)
			}
		}
	}
	check("keyExact", keyExact)
	check("keySuffixes", keySuffixes)
	// 精确值，不是下限：下限的话删掉十条都还是绿的
	const wantKeys = 76 // keyExact 31 + keySuffixes 45（实测）
	if got := len(keyExact) + len(keySuffixes); got != wantKeys {
		t.Fatalf("键释义有 %d 条，预期 %d 条 —— 增删条目时同步改这里", got, wantKeys)
	}
}

// 张量段同理：seg 是查表的键，meaning 是唯一的内容。
func TestTensorNaming_每段都有释义(t *testing.T) {
	for _, s := range tensorSegments {
		if strings.TrimSpace(s.seg) == "" {
			t.Error("有个张量段没有段名")
		}
		if strings.TrimSpace(s.title) == "" {
			t.Errorf("张量段 %s 没有标题", s.seg)
		}
		if strings.TrimSpace(s.meaning) == "" {
			t.Errorf("张量段 %s 没有释义 —— 用户查到的是个空壳", s.seg)
		}
	}
	if len(tensorSegments) != 87 {
		t.Fatalf("张量段 %d 条，预期 %d 条 —— 增删时同步改这里", len(tensorSegments), 87)
	}
}

// LookupKey 返回的条目必须与**速查表里那条逐字段相同**。
//
// 两处各拼一遍 ID 的后果实测过：`LookupKey("qwen2.block_count")` 给出
// `key:block_count`，而表里叫 `keysuffix:block_count` 并且多一句
// 「按后缀匹配的释义」—— 同一个键，两条路径两个对象。界面拿 ID 去跳转
// （SeeAlso 是 ID 语义）会跳空，还静默丢掉那句限定语。
func TestGGUFKeys_LookupKey与表一致(t *testing.T) {
	byID := map[string]Entry{}
	for _, tb := range Tables() {
		for _, e := range tb.Entries {
			byID[e.ID] = e
		}
	}
	// 精确键 + 一个只有靠后缀才命中的真实键。
	// **不能用裸的 "block_count"**：后缀匹配要求有 "." 分隔，
	// 裸段名本来就不该命中（那是我的期望写错了，不是代码的问题）
	for _, key := range []string{"general.file_type", "qwen2.block_count", "gemma4.rope.freq_base"} {
		got, ok := LookupKey(key)
		if !ok {
			t.Errorf("LookupKey(%q) 没找到", key)
			continue
		}
		want, inTable := byID[got.ID]
		if !inTable {
			t.Errorf("LookupKey(%q) 给的 ID %q 在表里查不到", key, got.ID)
			continue
		}
		if got.Title != want.Title || got.Notes != want.Notes ||
			len(got.Fields) != len(want.Fields) {
			t.Errorf("LookupKey(%q) 与表里的 %q 不是同一个条目：\n  查到 %+v\n  表里 %+v",
				key, got.ID, got, want)
			continue
		}
		for i := range got.Fields {
			if got.Fields[i] != want.Fields[i] {
				t.Errorf("LookupKey(%q) 的第 %d 个字段 %+v，表里是 %+v",
					key, i, got.Fields[i], want.Fields[i])
			}
		}
	}
	// 后缀命中的必须是 keysuffix: 前缀（否则界面上会以为存在裸键名条目）
	e, exact, ok := LookupKeyExact("qwen2.block_count")
	if !ok || exact {
		t.Fatalf("qwen2.block_count 应当是后缀命中：ok=%v exact=%v", ok, exact)
	}
	if !strings.HasPrefix(e.ID, "keysuffix:") {
		t.Errorf("后缀命中的 ID 是 %q，应当是 keysuffix: 开头", e.ID)
	}
	if e.Notes == "" {
		t.Error("后缀命中的条目没有「按后缀匹配」的限定说明 —— 界面会把它当成确定释义")
	}
	// 精确命中的必须是 key: 前缀
	e, exact, ok = LookupKeyExact("general.file_type")
	if !ok || !exact {
		t.Fatalf("general.file_type 应当是精确命中：ok=%v exact=%v", ok, exact)
	}
	if !strings.HasPrefix(e.ID, "key:") {
		t.Errorf("精确命中的 ID 是 %q，应当是 key: 开头", e.ID)
	}
}

// 已废弃的 file_type 编号必须**查得到**，而且要说清它已废弃。
//
// 实测：ollama 的 gpt-oss:20b 里 general.file_type 写的就是 4 ——
// 一个真实、流行的模型。而 4 是上游 llama.h 里注释掉的废弃编号。
// 这里原先完全不列废弃编号（当时的理由是"列出来会让用户以为那些档还在产出"，
// 但没人想到真实文件里会出现），于是 FileTypeByCode(4) 返回"查不到"，
// 界面上什么都不显示 —— 用户对着一个真实的值得不到任何解释。
func TestFileType_废弃编号查得到(t *testing.T) {
	// 逐条对照上游 llama.h 里被注释掉的那几行（含各自的废弃原因）
	deprecated := map[uint32]string{
		4:  "MOSTLY_Q4_1_SOME_F16",
		5:  "MOSTLY_Q4_2",
		6:  "MOSTLY_Q4_3",
		33: "MOSTLY_Q4_0_4_4",
		34: "MOSTLY_Q4_0_4_8",
		35: "MOSTLY_Q4_0_8_8",
	}
	for code, enum := range deprecated {
		e, ok := FileTypeByCode(code)
		if !ok {
			t.Errorf("file_type %d（%s）查不到 —— 真实文件里会出现这个值", code, enum)
			continue
		}
		if !strings.Contains(e.Title, "已废弃") {
			t.Errorf("file_type %d 的标题是 %q，没说明它已废弃 —— "+
				"用户会以为这是一种现役量化档", code, e.Title)
		}
		if !strings.Contains(e.Title, enum) {
			t.Errorf("file_type %d 的标题里没有上游原名 %s: %q", code, enum, e.Title)
		}
	}
	// 反面对照：现役档**不能**被标成已废弃
	for _, code := range []uint32{0, 1, 15, 38} {
		e, ok := FileTypeByCode(code)
		if !ok {
			t.Fatalf("现役档 %d 查不到", code)
		}
		if strings.Contains(e.Title, "已废弃") {
			t.Errorf("现役档 %d 被标成了已废弃: %q", code, e.Title)
		}
	}
	// 带 GUESSED 标志位的也要能查到（1028 = 1024|4）
	if e, ok := FileTypeByCode(1028); !ok || !strings.Contains(e.Title, "已废弃") {
		t.Errorf("1028（1024|4）应当剥掉标志位后查到废弃档 4，实际: ok=%v %q", ok, e.Title)
	}
}
