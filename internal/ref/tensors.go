package ref

import "strings"

// tensorSegment 是张量名里的一段。
type tensorSegment struct {
	seg     string // 段名；层号用 "#N" 占位
	title   string
	meaning string
	shape   string // 形状规律
}

// segmentsNotInCorpus 是**本机语料里一次都没出现过**的段。
//
// 它们的释义来自上游的命名约定（llama.cpp 的转换脚本、各家架构的
// 命名习惯），而不是实测 —— 可信度比另外 60 个低一档：
//
//   - `laurel` 尤其可疑：Gemma 3n 上游用的是 `laurel_l` / `laurel_r`，
//     按 "." 切开得到的是 "laurel_l" 而不是 "laurel"，
//     所以这一条很可能永远匹配不上任何东西。保留是因为
//     **不知道**它到底怎么切才对，删掉反而会丢掉这条线索。
//   - `attn`/`ffn`/`audio`/`vision` 这几个"通用前缀"同理：只有当名字里
//     真的出现独立的 `attn.` 段时才会命中。
//
// 有这张清单的意义：**"哪些是实测的、哪些是抄来的"不会随代码漂移**。
// TestTensorNaming_未实测段清单准确 拿 real_segments.txt 现算一遍并逐项比对，
// 多一个少一个都红 —— 将来某个模型开始产出这些段时，
// 它会提醒我们把清单（和这条注释）一起更新。
//
// 表里这些条目的说明后面会挂一句"本机语料未出现"，用户看得到。
var segmentsNotInCorpus = map[string]bool{
	"altup": true, "attn": true, "attn_linear": true, "attn_norm_2": true,
	"audio": true, "encoder": true, "ffn": true,
	"laurel": true, "projector": true, "vision": true,
}

// tensorSegments 是张量名各段的释义。
//
// 段名来自本机 5 个真实模型（gpt-oss:20b / qwen2.5:3b / gemma4:e4b /
// gemma4:26b / nomic-embed-text 与项目 artifacts）的张量名并集：
// tools/extract_tensor_segments.py 生成 testdata/real_segments.txt，
// TestTensorNaming_覆盖真实张量名 拿它校验覆盖率。
var tensorSegments = []tensorSegment{
	{"blk", "Transformer 块前缀", "后面跟层号，如 blk.7 表示第 7 层（从 0 数）", ""},
	{"#N", "层号", "第 N 层，从 0 开始。层号相同的张量属于同一个 Transformer 块", ""},
	{"token_embd", "词嵌入", "token id → 向量。GGUF 里形状是 [n_embd, n_vocab]（转置的）", "[n_embd, n_vocab]"},
	{"token_embd_norm", "嵌入后的归一化", "编码器类模型（BERT）在嵌入之后做一次归一化", "[n_embd]"},
	{"token_types", "token 类型嵌入", "BERT 的 segment embedding，区分句子 A/B", "[n_embd, 2]"},
	{"output", "输出投影", "最后一层隐藏态 → 词表 logits。**可能与 token_embd 权重绑定**（共享同一块存储）", "[n_vocab, n_embd]"},
	{"output_norm", "输出前的归一化", "最后的 norm，之后接 output 投影", "[n_embd]"},
	{"encoder", "编码器权重", "编码器类模型的输入编码", ""},
	{"attn_norm", "注意力前的归一化", "进入注意力之前的 norm（pre-norm 结构）", "[n_embd]"},
	{"attn_q", "注意力 Query 投影", "隐藏态 → query", "[n_embd, n_head*head_dim]"},
	{"attn_k", "注意力 Key 投影", "隐藏态 → key。**GQA 下它的输出维度小于 attn_q**", "[n_embd, n_head_kv*head_dim]"},
	{"attn_v", "注意力 Value 投影", "隐藏态 → value。维度同 attn_k", "[n_embd, n_head_kv*head_dim]"},
	{"attn_qkv", "Q/K/V 合并投影", "一些架构把三个矩阵拼成一个，省一次矩阵乘", "[n_embd, (n_head+2*n_head_kv)*head_dim]"},
	{"attn_output", "注意力输出投影", "注意力结果 → 隐藏态", "[n_head*head_dim, n_embd]"},
	{"attn_linear", "线性注意力的投影", "线性注意力变体的额外投影", ""},
	{"attn_norm_2", "第二处注意力归一化", "少数架构（如 Gemma）在注意力里做两次归一化", "[n_embd]"},
	{"ffn_norm", "前馈前的归一化", "进入 FFN 之前的 norm", "[n_embd]"},
	{"ffn_gate", "FFN 门控投影", "门控 FFN 的门（SwiGLU 的 W1）", "[n_embd, n_ff]"},
	{"ffn_up", "FFN 上投影", "门控 FFN 的值分支（SwiGLU 的 W3）", "[n_embd, n_ff]"},
	{"ffn_down", "FFN 下投影", "FFN 回到隐藏维度（SwiGLU 的 W2）", "[n_ff, n_embd]"},
	{"ffn_gate_inp", "MoE 路由", "MoE 的专家选择器，输出每个专家的权重", "[n_embd, n_expert]"},
	{"ffn_gate_exps", "MoE 专家门控", "全部专家的门控矩阵堆在一起", "[n_embd, n_ff, n_expert]"},
	{"ffn_down_exps", "MoE 专家下投影", "全部专家的下投影堆在一起", "[n_ff, n_embd, n_expert]"},
	{"ffn_up_exps", "MoE 专家上投影", "全部专家的上投影堆在一起", "[n_embd, n_ff, n_expert]"},
	{"weight", "权重", "乘性的参数。大部分张量是它", ""},
	{"bias", "偏置", "加性的偏置项。**量化模型里 bias 通常是 F32** —— 它数量少、影响直接", ""},
	{"scale", "缩放系数", "逐通道或逐元素的缩放", ""},
	{"norm", "归一化参数", "LayerNorm/RMSNorm 的缩放（weight）或偏移（bias）", ""},
	{"attn", "注意力", "注意力子层的通用前缀", ""},
	{"ffn", "前馈网络", "FFN 子层的通用前缀", ""},
	{"audio", "音频塔", "多模态模型里处理音频的分支", ""},
	{"vision", "视觉塔", "多模态模型里处理图像的分支", ""},
	{"mm", "多模态投影", "把视觉/音频塔的输出投影到文本嵌入空间", ""},
	{"per_layer_token_embd", "逐层 token 嵌入", "Gemma 3n 这类架构给每层一份额外的嵌入", ""},
	{"altup", "ALTUP 投影", "MatFormer/ALTUP 架构的预测-修正投影", ""},
	{"laurel", "Laurel 块", "Gemma 3n 的线性注意力增强块", ""},

	// ── 多模态塔（本机实测：gemma4 的 v./a. 前缀出现上千次）──
	//
	// v 与 a 是**塔的前缀**：v.blk.0.attn_q 是视觉塔第 0 层的 Q 投影，
	// 与文本塔的 blk.0.attn_q 结构相同但参数独立。
	//
	// 计数按**全部真实模型求和**（tools 的段名清单也是这个口径）：
	// v 段 1013 次（gemma4:e4b 658 + gemma4:26b 355）、a 段 752 次（都在 e4b）。
	// 这里原先写"v.* 有 658 段、a.* 有 752 段"—— 658 只是 e4b 一个模型的数，
	// 两个数字来自不同口径，并列摆着看起来像是同一层意思。
	{"v", "视觉塔前缀", "视觉编码器（ViT）的子树。v.blk.N.* 是它的第 N 层", ""},
	{"a", "音频塔前缀", "音频编码器的子树。a.blk.N.* 是它的第 N 层", ""},
	{"patch_embd", "patch 嵌入", "视觉塔把图切成 patch 后的线性嵌入", "[n_embd, patch_size^2*channels]"},
	{"position_embd", "位置嵌入", "显式存下来的位置编码表（不是 RoPE 那种算出来的）", "[n_embd, n_pos]"},
	{"input_projection", "输入投影", "把输入投到模型维度", ""},
	{"pre_encode", "编码前处理", "进编码器之前的预处理段", ""},
	{"linear_pos", "线性位置编码", "用线性变换产生的位置信息", ""},
	{"per_dim_scale", "逐维缩放", "对每个维度单独缩放，多用于视觉塔", ""},
	{"std_bias", "标准化偏置", "标准化（均值/方差）用的加性项", ""},
	{"std_scale", "标准化缩放", "标准化用的乘性项", ""},
	{"conv1d", "一维卷积", "音频塔的卷积层", ""},
	{"conv_dw", "深度可分离卷积", "depthwise：每个通道各自卷，参数量远小于普通卷积", ""},
	{"conv_pw1", "逐点卷积（升维）", "pointwise 的第一层，通常把通道数放大", ""},
	{"conv_pw2", "逐点卷积（降维）", "pointwise 的第二层，把放大后的通道压回去", ""},
	{"conv_norm", "卷积后的归一化", "归一化紧跟在卷积之后", ""},
	{"norm_conv", "归一化后的卷积", "归一化与卷积的顺序与 conv_norm 相反", ""},
	{"fc", "全连接层", "多模态投影器里常见的普通线性层", ""},

	// ── 激活范围记录（本机实测：**只出现在 gemma4:e4b**，232 组）──
	//
	// **这四条要标明哪部分是事实、哪部分是推断**：
	//   - 确凿：**只有 gemma4:e4b 有**（2131 个张量里 232 个各带一组；
	//     另外 3 个模型 0 个）。原先写"本机 4 个模型里成对出现（各 232 段）"
	//     与"gemma4 每个张量配一对"—— 两个说法都不对
	//   - 确凿：形状 [1]、F32、4 字节；四个标量**连续成一组**
	//     （input_max→input_min→output_max→output_min），
	//     并且**紧挨在同模块的 `.weight` 之前**（实测：232/232 组之后
	//     第 4 个张量就是该模块的 weight；"weight 之后才是标量"的有 0 组）。
	//     原先写成"紧跟在权重之后"，方向是反的
	//   - 确凿：上游的 conversion/gemma.py 会把它们写进 GGUF，
	//     注释称它们是 "scalar tensors (input/output_mix/max)"
	//   - **推断（未证实）**：它们是"量化标定数据"、用于静态量化范围、
	//     "推理时不再统计" —— 上游 src/ 与 gguf-py/ 里没有任何地方读它们，
	//     llama.cpp 的张量表也不认这些名字。这三句是从"名字 + 每模块一个标量"
	//     推出来的，写成一个断言就是给了用户一个我们并不知道的事实。
	//
	// 所以下面只说能证实的那部分：名字、形状、成对出现。
	{"input_max", "输入激活最大值记录", "该模块**输入**激活的最大值，与 input_min 成对出现。" +
		"形状 [1]、F32。上游把它写进文件但推理时并不读 —— " +
		"它的用途（推测是给静态量化做标定）未经证实", "标量 [1]"},
	{"input_min", "输入激活最小值记录", "该模块输入激活的最小值", "标量 [1]"},
	{"output_max", "输出激活最大值记录", "该模块**输出**激活的最大值", "标量 [1]"},
	{"output_min", "输出激活最小值记录", "该模块输出激活的最小值", "标量 [1]"},

	// ── Gemma 3 / 3n 的结构件 ──
	{"attn_q_norm", "Q 的归一化", "对 query 做归一化（QK-norm）。Gemma 3 / Qwen3 这类架构用它稳定注意力", "[head_dim]"},
	{"attn_k_norm", "K 的归一化", "对 key 做归一化", "[head_dim]"},
	{"attn_sinks", "注意力汇点", "**GPT-OSS 特有**：每个注意力头一个可学习的标量，" +
		"拼在注意力的 key/value 前面当「汇点」用 —— 让注意力有权把概率放在" +
		"「什么都不看」上，而不必摊到真实 token 上。形状 [n_head]（本机 gpt-oss:20b 是 [64]）", "[n_head]"},
	{"attn_out", "注意力输出投影", "与 attn_output 同一个东西。" +
		"本机实测出现在 gemma4 的视觉/音频塔里（v./a.blk.N.attn_out），文本塔用 attn_output", ""},
	{"attn_output_norm", "注意力输出的归一化", "注意力结果投影之后的 norm", "[n_embd]"},
	{"attn_post_norm", "注意力后的归一化", "post-norm 结构：注意力输出与本层输入相加之后再做一次 norm", "[n_embd]"},
	{"post_attention_norm", "注意力后的归一化", "同 attn_post_norm", "[n_embd]"},
	{"layer_pre_norm", "层前的归一化", "进入整个 Transformer 块之前的 norm", "[n_embd]"},
	{"layer_output_norm", "层输出的归一化", "整个块输出之后的 norm", "[n_embd]"},
	{"layer_output_scale", "层输出缩放", "给每层输出乘一个可学习的标量", "标量"},
	{"inp_gate", "输入门控", "对输入做门控的线性层", ""},
	{"post_norm", "输出后的归一化", "一般指子层输出与本层输入相加后的 norm", "[n_embd]"},
	{"post_ffw_norm", "FFN 后的归一化", "FFN 输出相加之后的 norm", "[n_embd]"},
	{"ffn_post_norm", "FFN 后的归一化", "同 post_ffw_norm", "[n_embd]"},
	{"ffn_gate_up_exps", "MoE 门控与上投影合并", "把 gate 与 up 两个矩阵拼在一起存，省一次读取", ""},
	{"rope_freqs", "RoPE 频率表", "**把 RoPE 的频率表当张量存下来**，而不是每次现算。" +
		"Gemma 3 这么存，是为了让频率可以被微调", "[head_dim/2]"},
	{"projector", "投影器", "多模态塔到文本空间的投影段", ""},
	{"out", "输出层", "塔或子模块的最后一段输出投影", ""},
	{"proj", "投影层", "把一个空间的表示投到另一个空间。" +
		"多模态里是塔→文本空间，Mamba 类架构里是状态→输出", ""},
	{"ln1", "第一处 LayerNorm", "BERT 类编码器的经典命名：注意力之前那一处", "[n_embd]"},
	{"ln2", "第二处 LayerNorm", "BERT 类编码器：FFN 之前那一处", "[n_embd]"},
	{"per_layer_model_proj", "逐层模型投影", "Gemma 3n 的逐层嵌入机制：给每层一份额外的投影", ""},
	{"per_layer_proj_norm", "逐层投影归一化", "上面那个投影的归一化", "[n_embd]"},

	// ── 编号后缀 ──
	//
	// 一个块里有多个同类子层时，用 _1/_2 区分（Gemma 3n 的 FFN 有两组）。
	// 用户看到 ffn_down_1 会疑惑"和 ffn_down 什么关系"，这条就是答案。
	{"ffn_down_1", "FFN 下投影（第 1 组）", "一个块里有两组 FFN 时，这是第二组的下投影", ""},
	{"ffn_up_1", "FFN 上投影（第 1 组）", "第二组 FFN 的上投影", ""},
	{"ffn_norm_1", "FFN 归一化（第 1 组）", "第二组 FFN 的 norm", ""},
	{"ffn_post_norm_1", "FFN 后归一化（第 1 组）", "第二组 FFN 输出后的 norm", ""},
	{"post_ffw_norm_1", "FFN 后归一化（第 1 组）", "同 ffn_post_norm_1", ""},
	{"post_ffw_norm_2", "FFN 后归一化（第 2 组）", "第三组", ""},
	{"pre_ffw_norm_2", "FFN 前归一化（第 2 组）", "第三组 FFN 之前的 norm", ""},
}

// tensorNamingTable 把张量段表变成速查表。
func tensorNamingTable() Table {
	tb := Table{ID: "tensors", Title: "张量命名"}
	for _, seg := range tensorSegments {
		fields := []Field{
			{"段名", seg.seg},
			{"含义", seg.meaning},
		}
		if seg.shape != "" {
			fields = append(fields, Field{"形状", seg.shape})
		}
		e := Entry{
			ID:     "tensor:" + seg.seg,
			Title:  seg.seg + " — " + seg.title,
			Fields: fields,
		}
		// 实测过与没实测过要看得出来：用户拿这条去解释自己的模型时，
		// "本机语料未出现"意味着这条释义可能是错的
		if segmentsNotInCorpus[seg.seg] {
			e.Notes = "**本机语料未出现**：这条释义来自上游命名约定，未经真实文件验证。"
		}
		// 除 blk 自己外都关联到 blk：名字里出现层号的场合最多
		if seg.seg != "blk" {
			e.SeeAlso = []string{"tensor:blk"}
		}
		tb.Entries = append(tb.Entries, e)
	}
	return tb
}

// normalizeLayer 把 "#12" 归一成 "#N"。
//
// 拆出来的段带具体层号（"#12"），而表里的条目是通配的 "#N" ——
// 两者不是同一个键，查之前必须归一。**这个函数原先只存在于测试文件里**，
// 于是"按 Segments 的结果查表"在生产路径上永远查不到层号段，
// 而那恰恰是测试注释里自称"用户最常问"的段。
func normalizeLayer(seg string) string {
	if strings.HasPrefix(seg, "#") && isAllDigits(seg[1:]) {
		return "#N"
	}
	return seg
}

// LookupTensorSegment 按段名取释义。
//
// **这是消费方应当用的入口**：它自己做层号归一，
// 所以 SplitTensorName 的输出可以直接喂进来。
func LookupTensorSegment(seg string) (Entry, bool) {
	return LookupTensorSegmentEntry(normalizeLayer(seg))
}

// LookupTensorSegmentEntry 与 LookupTensorSegment 相同，
// 只是不做层号归一 —— 给已经在用归一后段名的调用方。
func LookupTensorSegmentEntry(seg string) (Entry, bool) {
	for _, e := range tensorNamingTable().Entries {
		if strings.TrimPrefix(e.ID, "tensor:") == seg {
			return e, true
		}
	}
	return Entry{}, false
}

// lookupTensorSegment 返回段本身（不只是条目），供包内使用。
func lookupTensorSegment(seg string) (tensorSegment, bool) {
	for _, s := range tensorSegments {
		if s.seg == normalizeLayer(seg) {
			return s, true
		}
	}
	return tensorSegment{}, false
}

// SplitTensorName 把张量名拆成可查的段，数字段统一成 "#N"。
//
// 返回值可以直接喂给 LookupTensorSegment（它负责层号归一），
// 或喂给 LookupTensorSegmentEntry（要求调用方自己归一）。
//
// 数字要单独识别出来：它是**层号**，含义与普通段不同，
// 而且层号本身还有一层含义（同一层的张量属于同一个 Transformer 块）。
func SplitTensorName(name string) []string {
	raw := strings.Split(name, ".")
	out := make([]string, 0, len(raw))
	for _, s := range raw {
		if s == "" {
			continue
		}
		if isAllDigits(s) {
			out = append(out, "#"+s)
			continue
		}
		out = append(out, s)
	}
	return out
}

// isAllDigits 判断一个段是不是纯数字（层号）。
//
// 空串返回 false —— 空段在 SplitTensorName 里已经被丢掉了，
// 这里返回 true 的话会多出一个无意义的 "#"。
func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
