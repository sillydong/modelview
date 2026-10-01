package ref

import (
	"sort"
	"strconv"
	"strings"
)

// keyEntry 是一条元数据键的释义。
type keyEntry struct {
	key     string // 完整键名（keyExact）或以 "." 结尾的后缀（keySuffixes）
	title   string
	meaning string
	values  string // 取值范围 / 单位
}

// sepTokenIDKey 是 SEP 的键。**上游把它拼错了一个字母**（少了个 a），
// 但只能照抄 —— 改了就跟真实文件里的键对不上，查不到，
// 而这正是速查表要解决的问题。
const sepTokenIDKey = "tokenizer.ggml.seperator_token_id"

// keyExact 是必须逐字给出的键。
//
// general.* 与 tokenizer.* 是跨架构通用的，含义固定、不随架构变化，
// 而且**它们的名字推不出含义**（"tokenizer.ggml.pre" 是什么？），
// 所以必须逐个写。
var keyExact = []keyEntry{
	{"general.architecture", "架构", "决定其余架构专属键的前缀。如 qwen2 表示键是 qwen2.*", "字符串"},
	{"general.name", "模型名", "训练时给的模型名。**老文件里可能缺失**，缺了不该崩", "字符串"},
	{"general.type", "模型类型", "如 model（推理模型）", "字符串"},
	{"general.basename", "基础名", "上游转换工具记录的原始模型名", "字符串"},
	{"general.finetune", "微调标识", "微调版本的名字，如 qwen2.5 的 3b-instruct 变体", "字符串"},
	{"general.size_label", "规格标签", "如 3B、7B，给人看的参数量标签", "字符串"},
	{"general.license", "许可协议", "如 apache-2.0、gemma", "字符串"},
	{"general.license.name", "许可名称", "许可协议的全名", "字符串"},
	{"general.license.link", "许可链接", "许可协议的原文地址", "URL"},
	{"general.tags", "标签", "模型分类标签，如 text-generation、conversational", "字符串数组"},
	{"general.languages", "语言", "模型支持的语言代码", "字符串数组"},
	{"general.file_type", "量化档位",
		"**整份文件按哪个量化档位产出**。它的取值表见 filetype:<码>。" +
			"注意它只是「档位名」，实际每个张量用什么类型要看张量自己",
		"整数，见 filetype:* 条目"},
	{"general.quantization_version", "量化版本", "量化器的版本号。跨版本量化结果不可比", "整数"},
	{"general.parameter_count", "参数量", "上游算好的参数总数。缺失时本工具自己数（形状连乘）", "整数"},
	{"general.base_model.count", "基座模型个数", "这个模型是从几个基座微调来的", "整数"},
	{"tokenizer.ggml.model", "分词器类型", "如 gpt2（BPE）、llama（SentencePiece）、bert", "字符串"},
	{"tokenizer.ggml.pre", "预处理规则", "分词前的文本预处理规则名，如 qwen2、llama-bpe、default", "字符串"},
	{"tokenizer.ggml.tokens", "词表", "**几十万项的字符串数组**，体积最大的一条元数据", "字符串数组"},
	{"tokenizer.ggml.scores", "词元得分", "SentencePiece 的分词器用；BPE 不填", "浮点数组"},
	{"tokenizer.ggml.token_type", "词元类型", "每个 token 的类型（普通/未知/控制/字节）", "整数数组"},
	{"tokenizer.ggml.token_type_count", "词元类型种数", "token_type 的取值个数", "整数"},
	{"tokenizer.ggml.merges", "BPE 合并规则", "BPE 的合并表；SentencePiece 不填", "字符串数组"},
	{"tokenizer.ggml.bos_token_id", "BOS 的 id", "序列起始标记", "整数"},
	{"tokenizer.ggml.eos_token_id", "EOS 的 id", "序列结束标记", "整数"},
	{"tokenizer.ggml.eos_token_ids", "多个 EOS 的 id", "有些模型有多个结束符（如同时有 <|endoftext|> 与 <|im_end|>）", "整数数组"},
	{"tokenizer.ggml.padding_token_id", "PAD 的 id", "批推理时对齐长度用", "整数"},
	{"tokenizer.ggml.unknown_token_id", "UNK 的 id", "词表外 token", "整数"},
	{sepTokenIDKey, "SEP 的 id", "句子分隔", "整数"},
	{"tokenizer.ggml.cls_token_id", "CLS 的 id", "分类标记，编码器类模型用", "整数"},
	{"tokenizer.ggml.mask_token_id", "MASK 的 id", "掩码标记，编码器类模型用", "整数"},
	{"tokenizer.chat_template", "对话模板",
		"**模型的「对话逻辑」**：把消息列表渲染成提示词的 Jinja 模板", "字符串"},
}

// keySuffixes 是**按后缀**给出的释义，用于架构专属键。
//
// 为什么不全列完整键：llama.* 有几十个键、每个新架构又来一批，
// 逐个列既列不全也维护不动。而 "qwen2.attention.head_count" 这类键的含义
// 可以按后缀推出来 —— 前提是后缀本身有释义。
//
// 查找顺序见 lookupKeyPrefix。**匹配从最长的后缀组合试起**：
// "rope.freq_base" 比 "freq_base" 更具体，先命中前者才对。
var keySuffixes = []keyEntry{
	{"block_count", "层数", "Transformer 块的个数，决定模型深度", "正整数"},
	{"context_length", "上下文长度", "训练时的最大上下文（token 数）；推理时可外推但要付质量代价", "token 数"},
	{"embedding_length", "隐藏层维度", "token 嵌入与每层输出的向量维度", "正整数"},
	{"feed_forward_length", "前馈层中间维度", "FFN 的中间宽度", "正整数"},
	{"attention.head_count", "注意力头数", "query 的头数", "正整数"},
	{"attention.head_count_kv", "KV 头数", "key/value 的头数。小于 head_count 即为 GQA", "正整数"},
	{"attention.key_length", "K 的每头维度", "不填时按 embedding_length / head_count 推", "正整数"},
	{"attention.value_length", "V 的每头维度", "不填时按 embedding_length / head_count 推", "正整数"},
	{"attention.layer_norm_epsilon", "LayerNorm 的 eps", "分母加的小量，防除零。nomic-bert 用这个拼写", "浮点，典型 1e-5"},
	{"attention.layer_norm_rms_epsilon", "RMSNorm 的 eps", "RMSNorm 的 eps。llama 系架构用这个拼写", "浮点，典型 1e-5"},
	{"attention.causal", "是否因果注意力", "true 表示只能看前面的 token（语言模型）；false 表示双向（BERT 类编码器）", "true / false"},
	{"attention.sliding_window", "滑动窗口大小", "局部注意力的窗口宽度；超过它的 token 不直接互相注意", "token 数"},
	{"attention.sliding_window_pattern", "滑动窗口的分层模式", "指定哪些层用滑动窗口、哪些用全注意力", "整数序列"},
	{"attention.shared_kv_layers", "共享 KV 的层数", "多少层共用一份 KV；0 表示不共享", "正整数"},
	{"attention.key_length_swa", "滑动窗口层的 K 维度", "对使用滑动窗口的那几层生效的 K 维度", "正整数"},
	{"attention.value_length_swa", "滑动窗口层的 V 维度", "对使用滑动窗口的那几层生效的 V 维度", "正整数"},
	{"attention.head_count_swa", "滑动窗口层的头数", "对使用滑动窗口的那几层生效的头数", "正整数"},
	{"rope.freq_base", "RoPE 的基频", "旋转位置编码的基频。默认 10000，长上下文模型常调到几十万", "浮点"},
	{"rope.freq_base_swa", "滑动窗口层的 RoPE 基频", "对使用滑动窗口的那几层生效的基频", "浮点"},
	// ── 长上下文用的 RoPE 缩放 ──
	//
	// 取值名照 gguf 包的 RopeScalingType：none / linear / yarn / longrope。
	//
	// **来源要分清，两份真实文件不是同一组键**（实测）：
	//   - 本机 ollama 的 gpt-oss:20b（gptoss 前缀）：只有 factor 与
	//     original_context_length —— **没有 type**，也没有 yarn_beta_*
	//   - ggml-org 的官方 gpt-oss GGUF（gpt-oss 前缀）：另有 type=yarn
	//     与 yarn_beta_fast / yarn_beta_slow
	// 所以下面这五条里，只有前两条被本机语料验证过；
	// 后三条只被那份官方文件（不在本机）验证过。
	{"rope.scaling.type", "RoPE 缩放方式", "none=不缩放、linear=线性插值、" +
		"yarn=YaRN（按频率分段插值，长上下文最常用）、longrope=LongRoPE。" +
		"**这个键可能整条缺失**（本机 ollama 的 gpt-oss:20b 就没有）—— " +
		"缺了不代表没缩放，别从 factor 反推类型",
		"none / linear / yarn / longrope"},
	{"rope.scaling.factor", "RoPE 缩放倍数", "上下文被拉长的倍数。" +
		"与 rope.scaling.original_context_length 一起读：训练时 4096、推理拉到 131072 就是 32 倍", "浮点（如 32）"},
	{"rope.scaling.original_context_length", "缩放前的上下文长度",
		"模型训练时的原始上下文长度。**它才是「缩放前」的基准**，" +
			"与当前的 context_length 相除就是 rope.scaling.factor", "正整数"},
	{"rope.scaling.yarn_beta_fast", "YaRN 快边界",
		"YaRN 里区分「高频不插值 / 低频插值」的分界，越大插值的高频越多", "浮点"},
	{"rope.scaling.yarn_beta_slow", "YaRN 慢边界", "同上的另一侧边界；" +
		"fast/slow 之间做平滑过渡，避免频率处理突变", "浮点"},

	{"rope.dimension_count", "RoPE 的维度数", "参与旋转的维度个数；不填时按每头维度推", "正整数"},
	{"rope.dimension_count_swa", "滑动窗口层的 RoPE 维度数",
		"滑动窗口注意力层单独用的 RoPE 维度数；与 rope.dimension_count 的关系同 " +
			"rope.freq_base_swa 之于 rope.freq_base", "正整数"},
	{"embedding_length_per_layer_input", "逐层输入的嵌入维度", "Gemma 3n 这类架构给每层额外输入的维度", "正整数"},
	{"expert_count", "专家个数", "MoE 的专家总数", "正整数"},
	{"expert_used_count", "每 token 激活的专家数", "MoE 里每个 token 走几个专家；远小于总数时才省算力", "正整数"},
	{"expert_feed_forward_length", "专家的 FFN 宽度", "MoE 里每个专家自己的中间维度，通常比稠密模型小", "正整数"},
	{"final_logit_softcapping", "logits 软上限", "对最终 logits 做 tanh 软截断的上界；0 表示不截", "浮点"},
	{"num_channels", "输入通道数", "视觉塔的输入通道（RGB 是 3）", "正整数"},
	{"patch_size", "patch 边长", "视觉塔把图切成多大一块", "正整数"},
	{"projector.scale_factor", "投影缩放因子", "多模态投影器的缩放", "浮点"},
	{"pooling_type", "池化方式", "把序列压成一个向量的方式：-1=未指定、0=不池化、" +
		"1=均值、2=CLS（取首 token）、3=最后一个 token、4=RANK（重排序模型用）。" +
		"上游枚举是 llama_pooling_type", "-1..4"},
	{"causal", "是否因果", "同 attention.causal", "true / false"},
	{"add_bos_token", "是否自动加 BOS", "分词时是否自动在前面加 BOS", "true / false"},
	{"add_eos_token", "是否自动加 EOS", "分词时是否自动在末尾加 EOS", "true / false"},
	{"add_mask_token", "是否自动加 MASK", "编码器类模型用", "true / false"},
	{"add_padding_token", "是否自动加 PAD", "批推理时对齐长度用", "true / false"},
	{"add_unknown_token", "是否自动加 UNK", "分词时是否自动加未知词 token", "true / false"},
	{"conv_kernel_size", "卷积核边长", "音频塔的卷积核大小", "正整数"},
	// 带序号的键：数字段在归一后变成 #N，所以一条规则覆盖 .0. / .1. / …
	{"base_model.#N.name", "基座模型名", "从哪个基座微调来的（第 N 个）", "字符串"},
	{"base_model.#N.organization", "基座模型的组织", "基座模型的发布方", "字符串"},
	{"base_model.#N.repo_url", "基座模型的仓库地址", "基座模型的原始仓库", "URL"},
}

// lookupKeyPrefix 按后缀查一条键的释义。
//
// 规则：从**最长**的后缀组合开始试，直到命中。
// "gemma4.audio.attention.head_count" 会依次试
// "audio.attention.head_count" → "attention.head_count" → "head_count"，
// 第一个命中的胜出。
//
// 从最长开始是必须的：从最短开始的话 "rope.freq_base_swa" 会先命中
// "freq_base"（不存在），而 "freq_base_swa" 这类更具体的不该被更泛的盖掉。
func lookupKeyPrefix(key string) (keyEntry, bool) {
	parts := strings.Split(key, ".")
	// 去掉第一段（架构名），从剩下的最长后缀开始试。
	//
	// 每一步同时试**原样**与**序号归一**两种形态：
	// general.base_model.0.name 与 general.base_model.1.name 是同一类键，
	// 不该给每个序号各写一条 —— 归一后一条 base_model.#N.name 就够。
	for start := 1; start < len(parts); start++ {
		suffix := strings.Join(parts[start:], ".")
		norm := strings.Join(normalizeIndex(parts[start:]), ".")
		for _, e := range keySuffixes {
			if e.key == suffix || e.key == norm {
				return e, true
			}
		}
	}
	return keyEntry{}, false
}

// normalizeIndex 把纯数字段换成 "#N"。
//
// 与张量名里的层号是同一个做法：数字是**序号**，不是键名的一部分。
// 不归一的话，general.base_model.0.name / .1.name / .2.name
// 会被当成三条互不相干的键，要么各写一条（写不完），要么全都查不到。
func normalizeIndex(parts []string) []string {
	out := make([]string, len(parts))
	for i, p := range parts {
		if isAllDigits(p) {
			out[i] = "#N"
			continue
		}
		out[i] = p
	}
	return out
}

// LookupKey 按元数据键取释义，返回的条目**与速查表里那条逐字段相同**
// （含 ID 与那句"按后缀匹配"的限定说明），可以直接拿 ID 去跳转。
//
// 第二个返回值是"找没找到"。
//
// **要区分"精确命中"与"按后缀推测"请用 LookupKeyExact** ——
// 这里原先的注释把第二个返回值说成"是精确命中还是按后缀推测的"，
// 那是另一个函数的语义（注释与签名不符，正是本分支花了一轮修的那类问题）。
func LookupKey(key string) (Entry, bool) {
	e, exact, ok := lookupKey(key)
	if !ok {
		return Entry{}, false
	}
	return entryOfKey(e, !exact), true
}

// LookupKeyExact 是三返回值版本：entry、是否精确命中、是否找到。
//
// **两个 bool 相邻，调用方写反了编译器不会拦** ——
// 所以默认用 LookupKey（只要条目）或这个函数的具名变量赋值。
func LookupKeyExact(key string) (Entry, bool, bool) {
	e, exact, ok := lookupKey(key)
	if !ok {
		return Entry{}, false, false
	}
	return entryOfKey(e, !exact), exact, true
}

// entryOfKey 把一条键释义变成速查表条目。
//
// **标题用 e.key + e.title**：手写的短标签（"层数"、"注意力头数"）
// 是这张表里最好用的部分，曾经因为没有读它而白白躺在数据里，
// 条目标题退化成裸键名。
//
// **空字段会被丢掉，而不是留成空字段** —— 这一条有代价：
// "释义为空"在成品条目里是看不见的，表现为那一栏**不存在**
// （用户看到的是"这条怎么没有释义"，而"覆盖率"这类测试只看条目在不在，
// 于是报 100%）。所以空不空要在源头 keyEntry 上守，
// 见 TestGGUFKeys_每条都有释义与取值。
//
// viaSuffix 表示这条是按后缀命中的 —— 它决定 ID 与那句限定说明，
// **必须与表里的构造一致**：两处各拼一遍的话，"LookupKey 查到的条目"
// 与"表里列出的条目"会变成两个不同的对象（实测漂移过：
// 查到的 ID 是 key:block_count，而表里叫 keysuffix:block_count，
// 于是界面拿 ID 去跳转会跳空，还丢掉那句"这只是按后缀推测"）。
func entryOfKey(e keyEntry, viaSuffix bool) Entry {
	title := e.key
	if e.title != "" {
		title = e.key + " — " + e.title
	}
	fields := []Field{}
	if e.meaning != "" {
		fields = append(fields, Field{"含义", e.meaning})
	}
	if e.values != "" {
		fields = append(fields, Field{"取值", e.values})
	}
	if viaSuffix {
		return Entry{
			ID:     "keysuffix:" + e.key,
			Title:  title,
			Fields: fields,
			Notes:  "按后缀匹配的释义：任何以 ." + e.key + " 结尾的键都适用。",
		}
	}
	return Entry{ID: "key:" + e.key, Title: title, Fields: fields}
}

// lookupKey 先精确匹配，再按后缀退化。
//
// 第二个返回值区分"精确命中"与"按后缀退化命中"。
func lookupKey(key string) (keyEntry, bool, bool) {
	for _, e := range keyExact {
		if e.key == key {
			return e, true, true
		}
	}
	e, ok := lookupKeyPrefix(key)
	return e, false, ok
}

// fileTypeValues 是 general.file_type 的取值表。
//
// **码与名称都来自 llama.cpp 上游**（include/llama.h 的 enum llama_ftype +
// src/llama-model-loader.cpp 的 llama_ftype_name），2026-09-27 逐条核过。
// 复核用 tools/verify_ftype_table.py。
//
// 这里用**枚举名**（MOSTLY_Q4_K_M）而不是显示名（"Q4_K - Medium"）：
// 枚举名跨版本稳定，显示名上游改过措辞。display 字段给界面用。
//
// 中间几个编号（4/5/6、33/34/35）上游注释掉了但保留编号 ——
// 它们单列在下面的 deprecatedFileTypes 里。
//
// **原先完全不列，理由变了**：那时候的判断是"列出来会让用户以为
// 那些档还在产出"。但实测 ollama 的 gpt-oss:20b 里 general.file_type
// 写的就是 **4**（一个真实、流行的模型），而 FileTypeByCode(4) 当时
// 返回"查不到" —— 消费方拿到的是一个空条目，对着一个真实的值得不到任何解释。
//
// 现在的做法：照列，但把上游的移除原因写在 note 里，
// 读起来是"这个编号已废弃"而不是"这是一种现役量化档"。
//
// **现在有真实消费者了**：TUI 的元数据栏经 `tui.fileTypeEntry` 调它 ——
// qwen2.5:3b 上那一行是
// `general.file_type  15  [general.file_type = 15（MOSTLY_Q4_K_M）] ◂`
// （方括号里是本条目的标题），按 Enter 能进它的详情页。
// 所以上面说的"列出来才有解释"不再是"为展示层准备好了"，
// 而是一条走得到的路。
var fileTypeValues = []struct {
	code    uint32
	enum    string
	display string
	note    string
}{
	{0, "ALL_F32", "all F32", "全部 F32"},
	{1, "MOSTLY_F16", "F16", "大部分 F16，1 维张量例外"},
	{2, "MOSTLY_Q4_0", "Q4_0", "大部分 Q4_0"},
	{3, "MOSTLY_Q4_1", "Q4_1", "大部分 Q4_1"},
	{7, "MOSTLY_Q8_0", "Q8_0", "大部分 Q8_0"},
	{8, "MOSTLY_Q5_0", "Q5_0", "大部分 Q5_0"},
	{9, "MOSTLY_Q5_1", "Q5_1", "大部分 Q5_1"},
	{10, "MOSTLY_Q2_K", "Q2_K - Medium", "Q2_K 中档"},
	{11, "MOSTLY_Q3_K_S", "Q3_K - Small", "Q3_K 小档"},
	{12, "MOSTLY_Q3_K_M", "Q3_K - Medium", "Q3_K 中档"},
	{13, "MOSTLY_Q3_K_L", "Q3_K - Large", "Q3_K 大档"},
	{14, "MOSTLY_Q4_K_S", "Q4_K - Small", "Q4_K 小档"},
	{15, "MOSTLY_Q4_K_M", "Q4_K - Medium",
		"**最常见的 4 位档**。_M 档会把关键张量升到 Q6_K，" +
			"这正是它与 _S 的区别 —— 所以看到这个档位，" +
			"张量类型分布里通常同时有 Q4_K 与 Q6_K"},
	{16, "MOSTLY_Q5_K_S", "Q5_K - Small", "Q5_K 小档"},
	{17, "MOSTLY_Q5_K_M", "Q5_K - Medium", "Q5_K 中档"},
	{18, "MOSTLY_Q6_K", "Q6_K", "大部分 Q6_K"},
	{19, "MOSTLY_IQ2_XXS", "IQ2_XXS - 2.0625 bpw", "IQ2 的极端版本"},
	{20, "MOSTLY_IQ2_XS", "IQ2_XS - 2.3125 bpw", ""},
	{21, "MOSTLY_Q2_K_S", "Q2_K - Small", ""},
	{22, "MOSTLY_IQ3_XS", "IQ3_XS - 3.3 bpw", ""},
	{23, "MOSTLY_IQ3_XXS", "IQ3_XXS - 3.0625 bpw", ""},
	{24, "MOSTLY_IQ1_S", "IQ1_S - 1.5625 bpw", ""},
	{25, "MOSTLY_IQ4_NL", "IQ4_NL - 4.5 bpw", ""},
	{26, "MOSTLY_IQ3_S", "IQ3_S - 3.4375 bpw", ""},
	{27, "MOSTLY_IQ3_M", "IQ3_S mix - 3.66 bpw", "注意显示名里写的是 IQ3_S mix，不是 IQ3_M"},
	{28, "MOSTLY_IQ2_S", "IQ2_S - 2.5 bpw", ""},
	{29, "MOSTLY_IQ2_M", "IQ2_M - 2.7 bpw", ""},
	{30, "MOSTLY_IQ4_XS", "IQ4_XS - 4.25 bpw", ""},
	{31, "MOSTLY_IQ1_M", "IQ1_M - 1.75 bpw", ""},
	{32, "MOSTLY_BF16", "BF16", "大部分 BF16"},
	{36, "MOSTLY_TQ1_0", "TQ1_0 - 1.69 bpw ternary", "三值量化"},
	{37, "MOSTLY_TQ2_0", "TQ2_0 - 2.06 bpw ternary", "三值量化"},
	{38, "MOSTLY_MXFP4_MOE", "MXFP4 MoE", "MoE 专用的 4 位微缩放格式"},
	{39, "MOSTLY_NVFP4", "NVFP4", "NVIDIA 的 4 位格式"},
	{40, "MOSTLY_Q1_0", "Q1_0", "1 位"},
	{41, "MOSTLY_Q2_0", "Q2_0", "2 位（非 K 系列）"},
}

// deprecatedFileTypes 是上游**已废弃/移除**的档位编号。
//
// 名称与原因逐条抄自 llama.h 里被注释掉的那几行（2026-09-27 核过）：
//
//	// LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
//	// LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
//	// LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
//	//LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
//
// 它们**会出现在真实文件里**（实测 ollama 的 gpt-oss:20b 写的是 4），
// 所以必须查得到 —— 查不到就是"用户看到一个值，工具一句话都不说"。
var deprecatedFileTypes = []struct {
	code uint32
	enum string
	why  string
}{
	{4, "MOSTLY_Q4_1_SOME_F16",
		"上游注释里写的是「Q4_1，但 token_embeddings.weight 与 output.weight 是 F16」，" +
			"该编号已废弃"},
	{5, "MOSTLY_Q4_2", "上游已移除支持"},
	{6, "MOSTLY_Q4_3", "上游已移除支持"},
	{33, "MOSTLY_Q4_0_4_4", "已从 gguf 文件里移除，改用 Q4_0 + 运行时重排"},
	{34, "MOSTLY_Q4_0_4_8", "已从 gguf 文件里移除，改用 Q4_0 + 运行时重排"},
	{35, "MOSTLY_Q4_0_8_8", "已从 gguf 文件里移除，改用 Q4_0 + 运行时重排"},
}

// ftypeGuessedFlag 是 llama.h 里 LLAMA_FTYPE_GUESSED 的取值。
//
// 它与档位码按位或在一起：读到 general.file_type = 1039（1024|15）时，
// 真实档位是 15，但上游的显示名会多一个 "(guessed) " 前缀 ——
// 意思是这个值是上游**猜**的（老文件缺这个键时按张量类型分布反推的）。
// 直接拿它当码去查表会落空。
const ftypeGuessedFlag = 1024

// FileTypeByCode 按 general.file_type 的值取档位条目。
//
// **会先剥掉 LLAMA_FTYPE_GUESSED 标志位**。不剥的话，一个"上游猜出来的"
// file_type（1039）会查不到任何东西，而它其实就是 Q4_K_M。
func FileTypeByCode(code uint32) (Entry, bool) {
	return ByID("filetype:" + strconv.FormatUint(uint64(code&^ftypeGuessedFlag), 10))
}

// FileTypeGuessed 表示这个 file_type 的值是上游猜的，不是文件里写死的。
//
// 界面上应当标出来：猜出来的档位可能与该文件实际的张量类型不符，
// 而用户会拿它当事实。
func FileTypeGuessed(code uint32) bool { return code&ftypeGuessedFlag != 0 }

func ggufKeysTable() Table {
	tb := Table{ID: "keys", Title: "GGUF 元数据键"}

	entries := append([]keyEntry{}, keyExact...)
	sort.Slice(entries, func(i, j int) bool { return entries[i].key < entries[j].key })
	for _, e := range entries {
		// 走 entryOfKey 而不是在这里另拼一遍：LookupKey 也用它，
		// 两处拼两遍的话，"查到的条目"与"表里列出的条目"会漂移
		tb.Entries = append(tb.Entries, entryOfKey(e, false))
	}

	// file_type 的取值另起一组：它们是**值**不是键，
	// ID 用 filetype:<码>，与 key:<键名> 分开，避免两类混在一起
	for _, v := range fileTypeValues {
		e := Entry{
			ID:    "filetype:" + strconv.FormatUint(uint64(v.code), 10),
			Title: "general.file_type = " + strconv.FormatUint(uint64(v.code), 10) + "（" + v.enum + "）",
			Fields: []Field{
				{"码", strconv.FormatUint(uint64(v.code), 10)},
				{"枚举名", v.enum},
				{"llama.cpp 显示名", v.display},
			},
			Notes:   v.note,
			SeeAlso: []string{"key:general.file_type"},
		}
		tb.Entries = append(tb.Entries, e)
	}

	// 已废弃的编号也列出来 —— 它们会出现在真实文件里（见 deprecatedFileTypes）。
	// **标题里就写明"已废弃"**，不给它现役档的待遇
	for _, v := range deprecatedFileTypes {
		code := strconv.FormatUint(uint64(v.code), 10)
		tb.Entries = append(tb.Entries, Entry{
			ID:    "filetype:" + code,
			Title: "general.file_type = " + code + "（已废弃：" + v.enum + "）",
			Fields: []Field{
				{"码", code},
				{"原名", v.enum},
				{"状态", "上游已废弃/移除"},
			},
			Notes:   v.why + "。文件里读到这个值，说明写入方用了旧编号 —— 档位不可信，以张量实际类型为准。",
			SeeAlso: []string{"key:general.file_type"},
		})
	}

	// 后缀表也做成条目：用户看到 qwen2.attention.head_count 时，
	// 按后缀命中的那条要能直接跳过去看
	for _, e := range keySuffixes {
		// 后缀条目的 ID 用 keysuffix:（key:block_count 会让人以为
		// 存在一个叫 block_count 的键）—— 这件事由 entryOfKey 负责，
		// 这里只是说"这条是按后缀来的"，不再自己拼 ID
		tb.Entries = append(tb.Entries, entryOfKey(e, true))
	}
	return tb
}
