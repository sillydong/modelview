# modelview

看清一个模型文件里到底是什么。

Go 写的模型文件分析器，**TUI 界面 + 非交互 CLI** 两用，支持 **GGUF、safetensors、PyTorch（.pt）**。
全程只读模型文件（唯一的写操作是缓存），不加载模型、不跑推理。

它回答这些问题：

- 这个文件是什么格式、多大、什么架构、多少参数？
- 里面有哪些张量，各自什么形状、什么精度、占多少空间？
- 权重数值实际长什么样（分布、离群值、动态范围）？
- 如果把它量化会损失多少？**它已经被量化得有多狠？**
- 我本机装了哪些模型，分别在哪？哪些磁盘可以回收？

不做的事：推理、修改模型文件、无头部的裸二进制（没有自描述信息，无法可靠识别）、图表（直方图用字符画 `▁▂▅█`）。

## 安装

预编译二进制在 [Releases](https://github.com/sillydong/modelview/releases)：
Linux / macOS / Windows，amd64 与 arm64。

或者用 Go 1.26+ 自己装：

```bash
go install github.com/sillydong/modelview/cmd/modelview@latest
```

从源码构建：

```bash
git clone https://github.com/sillydong/modelview.git && cd modelview
go build -o modelview ./cmd/modelview
```

产出的二进制是自包含的，运行时不需要任何外部文件。依赖只有 `bubbletea`（TUI 框架）与 `lipgloss`（样式），其余全是标准库。

## 支持什么

格式**按 magic bytes 判断，不看扩展名**：

| 特征 | 判定 |
|---|---|
| `GGUF` | GGUF |
| 前 8 字节是 < 100 MiB 的小端 u64，后接 `{` | safetensors |
| `PK\x03\x04` | PyTorch（ZIP 容器） |

GGUF 的数据区起点按 `general.alignment` 对齐（默认 32）；
PyTorch 走自己实现的精简 pickle 解析器（只覆盖 `torch._utils._rebuild_tensor_v2` 用到的 opcode 子集，不引入第三方 pickle 库），
遇到不认识的 opcode 会报出是哪个字节（`未支持的 pickle 操作码 0x..`）而不是崩溃。

| 能力 | 覆盖的 dtype |
|---|---|
| 解码出数值（统计 / 量化模拟） | F64 F32 F16 BF16、I64 I32 I16 I8、U64 U32 U16 U8、BOOL，以及 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K |
| 只读块头（量化诊断） | 同上那 10 个量化类型 |
| 只收录块结构（能算占用大小） | MXFP4 NVFP4 Q8_1 Q8_K，以及 IQ1_S IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ4_NL IQ4_XS IQ1_M |

IQ 系列与 MXFP4 / NVFP4 **没有解码器**，也不给位宽：算占用大小时明确报「未收录」，
而不是给出一个可能错误的数字。同理，Q8_1 / Q8_K 的块字节数收录了，但内部布局没有独立参照验证过，所以不解码。

## 用法

### 模型库：看本机装了什么

```bash
modelview          # 无参数 → 模型库
modelview scan     # 同上，显式写法
```

```
modelview · 模型库
▸ gemma4:26b                   ollama       16.75 GiB  GGUF · gemma4 · 25.806 G 个参数
  gemma4:e4b                   ollama        8.95 GiB  GGUF · gemma4 · 7.996 G 个参数
  gpt-oss:20b                  ollama       12.85 GiB  GGUF · gptoss · 20.915 G 个参数
  nomic-embed-text:latest      ollama      261.58 MiB  GGUF · nomic-bert · 136.727 M 个参数
  qwen2.5:3b                   ollama        1.80 GiB  GGUF · qwen2 · 3.086 G 个参数

↑/k ↓/j 移动  Enter 查看  r 重扫  ? 速查表  q 退出
```

扫描这些目录（不存在的跳过），每行给出名字、来源、大小、格式、架构、参数量：

| 来源 | 路径 |
|---|---|
| ollama | `~/.ollama/models` |
| HuggingFace Hub | `~/.cache/huggingface/hub` |
| LM Studio | `~/.cache/lm-studio`、`~/.lmstudio/models` |
| GPT4All | `~/.cache/gpt4all`、`~/.local/share/nomic.ai/GPT4All` |
| llama.cpp | `~/.cache/llama.cpp`、`~/Library/Caches/llama.cpp` |
| PyTorch Hub | `~/.cache/torch/hub/checkpoints` |
| Jan | `~/Library/Application Support/Jan` |
| ModelScope | `~/.cache/modelscope` |
| 通用 | `~/models` |

通用目录递归最多 6 层，跳过 `.git` / `node_modules` / `.venv` / `__pycache__` / `.modelview-cache`。

**ollama 是特殊路径**：blob 文件名是内容哈希（`sha256-<hex>`），不含模型信息，必须遍历 `manifests/` 才能反查出 `qwen2.5:3b` 这样的名字。顺带给出两项磁盘信息：

- **孤儿 blob** —— 没有被任何 manifest 引用，列出可回收空间
- **未完成的下载** —— blob 已落盘、manifest 还没写（ollama 下完才写）。这一类单独提示「不算可回收」：**照着删会毁掉正在进行的下载**

### 分析单个文件

```bash
modelview model.gguf           # 人类可读摘要（元数据与张量全部列出）
modelview --stats model.gguf   # 附带数值统计与量化诊断
modelview ./models/            # 列出任意目录下的模型文件
```

```
文件     /Users/…/blobs/sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6
格式     GGUF v3
大小     1.80 GiB
架构     qwen2
元数据   35 条
张量     434 个
总参数   3.086 G（3,085,938,688）
对齐     32 字节（数据区起点 5956512）

类型分布
  F32         181 个张量  32 bit/权重
  Q4_K        216 个张量  4.5 bit/权重
  Q6_K         37 个张量  6.5625 bit/权重

元数据（全部 35 条）
  general.architecture                         qwen2
  general.name                                 Qwen2.5 3B Instruct
  …
```

### JSON 输出

```bash
modelview --json model.gguf                 # 结构信息
modelview --json --stats model.gguf > a.json  # 附带 stats / quant / quant_sims
```

选项**必须写在位置参数前面**。`modelview scan --json` 里的 `--json` 会被 Go 的 flag 静默丢弃，
所以本程序宁可报错也不猜；路径本身以 `-` 开头时写在 `--` 之后。

顶层形状（取自真实输出，`tensors` 数组在此省略；每项的字段见表）：

```json
{
  "path": "…/blobs/sha256-5ee4f07cdb9be…",
  "format": "GGUF", "version": "v3", "file_size": 1929903008,
  "arch": "qwen2", "param_count": 3085938688,
  "metadata": [{"key": "general.architecture", "value": "qwen2", "raw": "qwen2"}],
  "tensors": [ … ]
}
```

`tensors[*]` 的字段：

| 字段 | 含义 |
|---|---|
| `name` `dims` `dtype` `offset` `byte_size` `param_count` | 结构信息 |
| `stats` | 数值统计，字段见下 |
| `quant` | 量化诊断。**只有已量化的张量有**，字段见「量化分析」 |
| `quant_sims` | 量化模拟三档。**只有浮点张量有** |

`stats` 的字段：

| 字段 | 含义 |
|---|---|
| `count` `min` `max` `mean` `std` | 样本数与基本统计量（`count` 含 NaN/Inf） |
| `nan` `inf` | 非有限值计数，只计数、不参与上面的统计 |
| `zero_ratio` `outlier_ratio` | 精确为零、超出 μ±3σ 的比例 |
| `histogram` | 64 桶计数，界面上的字符画就是它 |
| `sampled` | 这份统计是否来自采样 |

拿单个张量的完整结构：

```bash
modelview --json --stats model.gguf | jq '.tensors[0]'
```

### 选项

| 选项 | 默认 | 说明 |
|---|---|---|
| `--json` | 关 | 以 JSON 输出（非交互）。`modelview scan` 的输出重定向到管道或文件时也走非交互，不会去开 TUI |
| `--stats` | 关 | 计算数值统计与量化诊断（慢，要读张量数据） |
| `--sample-limit N` | `10000000` | 单张量参与统计的元素数上限，超过则等距采样。**`0` 表示不采样**（读完整个张量，内存见下） |
| `--no-cache` | 关 | 既不读也不写缓存 |
| `--version` | | 打印版本后退出 |

## 交互界面

终端最小 80×24。左栏导航、右栏内容，`Esc` / `Backspace` 逐层返回。

```
模型库
 └→ 模型概览 → 元数据 / 张量 / 量化分布
      └→ 张量详情 → 基本信息 / 数值统计 / 量化模拟
           └→ 速查表（`?` 随处可开，自带上下文）
```

| 键 | 作用 |
|---|---|
| `↑↓` / `kj` | 移动 |
| `Enter` | 深入当前项 |
| `Esc` / `Backspace` | 返回上一层 |
| `Tab` | 左栏 / 右栏焦点切换 |
| `/` | 过滤（张量列表）或搜索（速查表） |
| `a` | 后台扫描全部张量，带进度（扫描中变成「取消」）。只在模型视图有 |
| `r` | 重扫。模型库：重新扫描目录；模型视图：清空已有结果重算 |
| `?` | 速查表 |
| `q` / `Ctrl+C` | 退出 |

帮助栏只列**当前视图真的支持**的键，不列按了没反应的。

### 张量列表与详情

```
张量 · qwen2.5:3b（共 434 个）
张量（434/434）
▸ token_embd.weight                   [2048, 151936]     Q6_K    243.43 MiB
  blk.0.attn_norm.weight              [2048]             F32       8.00 KiB
  blk.0.ffn_down.weight               [11008, 2048]      Q6_K     17.64 MiB
  blk.0.ffn_gate.weight               [2048, 11008]      Q4_K     12.09 MiB
  …
↑/k ↓/j 移动  Enter 详情  / 过滤  ? 速查表  Esc 返回  q 退出
```

```
张量 · blk.0.attn_norm.weight · F32 · 8.00 KiB
blk.0.attn_norm.weight
形状     [2048]
类型     F32（32 bit/权重）
元素     2.048 K（2,048）
占用     8.00 KiB
偏移     0xf91bba0

名字构成
▸ blk              blk — Transformer 块前缀
  #0               #N — 层号
  attn_norm        attn_norm — 注意力前的归一化
  weight           weight — 权重
  按 Enter 查这一段

统计
  [-0.8750, 1.7500] μ=0.3215 σ=0.1393 零=0 离群=1.71%
  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄█▆▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁

量化模拟
  Q8_0    8.5 bit/权重 45.5dB ×3.76
  Q6_K 6.5625 bit/权重 35.6dB ×4.88
```

### 量化分布

一屏看完一个文件的量化构成，并与文件自己声明的 `general.file_type` 对照：

```
qwen2.5:3b · GGUF v3 · 1.80 GiB · 434 张量 · 3.086 G 参数
  概览        量化分布（按类型）
  元数据 (35)
  张量 (434)  类型 张量      参数       占用  占比 bit/权重
▸ 量化分布    Q4_K  216   2.359 G   1.24 GiB 69.0%      4.5
  速查表      Q6_K   37 726.401 M 568.27 MiB 31.0%   6.5625
              F32   181 241.664 K 944.00 KiB  0.1%       32

              整体       4.9876 bit/权重（含块头开销）

              文件声明
                [general.file_type = 15（MOSTLY_Q4_K_M）]
                这是文件自己写的声明，不代表实际分布 —— 上表才是实际的
```

### 速查表

五组静态数据（`数值格式` / `量化方案` / `GGUF 元数据键` / `张量命名` / `GGML 类型码`），
从元数据或张量名可以 `Enter` 跳进对应条目，条目里再 `Enter` 反查「本模型里有哪些张量用了它」。

```
modelview · 速查表 · 数值格式（21 条）
▸ 数值格式 (21)         F64
  量化方案 (23)         F32
  GGUF 元数据键 (118)   BF16
  张量命名 (87)         F16
  GGML 类型码 (33)      F8_E4M3
↑/k ↓/j 移动  Tab 切换栏  Enter 详情  / 搜索  ? 速查表  Esc 返回  q 退出
```

## 分析能力

### 数值统计

解码张量数据后给出 `min` / `max` / `mean` / `std`、`nan` / `inf` 计数、
精确为零的比例、超出 μ±3σ 的比例，以及 64 桶直方图（界面上的字符画就是它）。

NaN / Inf **只计数，不参与** min / max / mean / std —— 否则一个 Inf 就能把整条统计变成 NaN，
而屏幕上看起来只是「有点怪」；`zero_ratio` 与 `outlier_ratio` 的分母也是有效值个数
（一个 NaN 既不是零，也不是离群点）。

元素数超过 `--sample-limit` 时按**等距**采样（每 `count/limit` 个取一个），
结果可复现，且对有序数据比随机采样稳健；此时 `sampled` 标记为 `true`。
`min` / `max` 也基于采样值 —— 界面与 JSON 都如实标注。

### 量化分析

**这是本工具的核心。** 分两种情况，判据是张量本身是不是浮点。

**情况 A：浮点张量（F32 / F16 / BF16）→ 量化模拟**

用**各格式真实的编码算法**跑一遍「编码 → 解码 → 与原值比对」，
三档并排（Q8_0 / Q6_K / Q4_K），给出最大绝对误差、平均误差、信噪比与压缩比。

不用「块内最大绝对值 ÷ 级数」那种统一公式，因为它的结论取决于两个**任意选择**——
级数怎么定、块多大。实测同一个真实 F16 张量上，光把级数从 31 改成 63 信噪比就摆动 6.26 dB，
比它自身的误差还大。真实编码器没有这个自由度。

最大相对误差给**两个口径**：`max_rel_err` 是全体元素上的最大值，
但它由最接近 0 的那个元素决定（贴近 0 的值被量化成 0，相对误差恒为 1，
实测 nomic-embed 的 336 条模拟里 168 条恰好等于 1）；
`max_rel_err_sig` 只统计 |x| 大于该张量峰值 5e-2 的元素，才是能用来比较量化质量的数。

**情况 B：已量化张量（Q4_K / Q6_K / Q8_0 等）→ 量化诊断（反解块结构）**

只读块头里的 `scale` / `min`，不反量化权重，所以能覆盖**全量**子块而不受采样影响：

- scale 的分布：min / max / 均值 / 中位数
- **被压得最惨的子块**：判据是 `|scale| / 同一个块内最大的 |scale|`。不跨块比绝对 scale——
  `scale = d × 子scale` 而 `d` 是整个块共用的，跨块比会把「这个块的数值本来就小」误判成「被压平」
- 有多少子块的 scale 为 0（权重全塌在一个量化级上，最直接的证据）
- 该张量实际的每权重平均位宽（Q4_K 实际是 4.5 bit，不是 4）

中位数是唯一必须看到全部值才能精确算出的量，子块数可达千万级，
因此设了 8 MiB 的样本预算（`1<<21` 个样本 × 4 字节，蓄水池抽样、种子固定），超预算时 JSON 里置
`scale_median_sampled: true`。**min / max / 均值 / 零计数 / 最扁 任何时候都是全量精确的。**

> 本节引用的实测数字（统一公式的 6.26 dB 摆动、`max_rel_err` 的 168/336、块复现率）出自
> [设计文档 §6.2](docs/superpowers/specs/2026-09-26-modelview-design.md)，
> 那里有完整测量过程与复现命令；这里只留结论。

### 内存

- **默认**：峰值由采样上限决定，与模型总大小基本无关。
  实测 qwen2.5:3b（434 张量 / 1.80 GiB）：峰值 337 MiB
- **`--sample-limit 0`（不采样）**：峰值 ∝ 最大那张张量的元素数，实测约 **8 B/元素**——
  3.11 亿元素的张量峰值 2.4 GiB。预估超过 1 GiB 时程序会在开始前把峰值打到 stderr
- **块级诊断**总是流式扫完全部块头（内存 O(1)），耗时 ∝ 张量数据总量，**不受采样上限约束**——
  `--sample-limit` 只管统计，这一点写在了 `--help` 里

### 缓存

**只有 CLI 的 `--stats` 用缓存**。TUI 不读也不写：它按需扫描单个张量（点开哪张扫哪张），
结果留在内存里，理由是把单张量结果写进「整份模型」的缓存会造成**部分写入**，
而部分写入正是下次全量扫描时「看起来命中了、其实缺一半」的来源——那类故障不报错，只是数字少了几块。

缓存内容是整份模型的统计与量化诊断，按张量名索引，位置按顺序取第一个可写的：

1. `<模型文件所在目录>/.modelview-cache/<模型文件名>.json` —— 模型目录整体拷走时缓存跟着走
2. 目录不可写（只读挂载、权限不足）→ `~/.cache/modelview/<sha256(绝对路径)>.json`

第 2 条用**路径哈希**而不是文件名：不同目录下的同名文件（实测有 37 份都叫 `mutation.safetensors`）
按名字存会互相覆盖。

失效判据四者缺一不可：结构版本、文件大小、修改时间、采样上限。
不缓存原始权重数据，`--no-cache` 时既不读也不写。

## 代码结构

```
cmd/modelview/       CLI 入口（main / scan / dir / tui）
internal/
  model/             统一数据模型 —— 所有格式归一到这里
  detect/            格式探测（magic bytes）
  parser/            gguf / safetensors / pytorch，各自独立
  decode/            位级解码：浮点、整数、量化块，以及只读块头的 scale 提取
  analyze/           统计、量化模拟、量化诊断、缓存
  discover/          模型库发现、ollama manifest、孤儿 blob
  ref/               速查表静态数据（Go 字面量，无外部文件）
  render/ humanize/  命令行渲染与数字格式化
  tui/               界面
```

核心约束：`parser/*` 各自解析后归一化成同一个 `model.Model`，
`tui/` 与 `analyze/` 只依赖 `model.Model`，不知道底层格式。
新增格式时只改 `detect/` + `parser/`，其余不动。

TUI 的 `Update` / `View` 是纯函数，I/O 走 `tea.Cmd`；后台扫描是自续式的
`tea.Cmd` 链而不是常驻 goroutine —— 取舍与理由写在 `internal/tui` 的注释里，
那里是唯一出处。

## 开发

```bash
go build ./...
go vet ./...
gofmt -l .                              # 应无输出
golangci-lint run ./...
go test ./...
go test ./... -race
python3 tools/check_readme.py           # 本文档与代码的名字/路径是否一致
```

### 真实语料

日常回归用的是**入库的小样本与真值**（`internal/decode/testdata/`、`internal/ref/testdata/`，
从真实文件截取），所以 `go test` 不需要网络、不需要 Python。

更完整的一组语料是**作者本机的** ollama 模型与另一个项目的 artifacts，
**不在仓库里**，别人 clone 下来也没有——那类测试默认跳过（`go test -v` 里能看到
跳过了多少条），设 `MODELVIEW_REAL=1` 则把「语料缺失」判为失败
（用来确认它们真的跑过，而不是静默跳过）：

```bash
MODELVIEW_REAL=1 go test ./...
```

### 交叉验证

`tools/` 下是开发期用的验证脚本，不参与构建：

| 脚本 | 用途 |
|---|---|
| `verify_ggml_dequant.py` | 用 llama.cpp 官方 `gguf.quants` 生成量化解码真值 → `internal/decode/testdata/vectors.json` |
| `verify_gguf_blocks.py` | 用真实 GGUF 文件反推验证块大小表 |
| `verify_gguf_scales.py` | 用独立来源验证块头 scale 提取 |
| `verify_gguf_quant_sim.py` | 量化模拟的字节级真值（含专门造出的舍入平局与全零块） |
| `verify_ggml_types.py` | GGML 类型码表与上游逐条对照 |
| `verify_ftype_table.py` | `general.file_type` 取值表与 llama.cpp 上游逐条对照 |
| `verify_safetensors.py` | safetensors 的独立解析参照（只用标准库，代码路径与 Go 版完全不同） |
| `verify_pytorch.py` | PyTorch .pt 的独立解析参照（`pickletools` 反汇编 + 自己的栈机） |
| `extract_real_blocks.py`、`extract_metadata_keys.py`、`extract_tensor_segments.py` | 从本机模型导出测试样本，以及速查表覆盖率的真值 |
| `compare_dequant.go`、`compare_uniform_formula.go` | 比对解码结果；复现「统一公式为什么不可信」的那组数据 |
| `mutate.py` | 变异验证：破坏一处实现 → 确认测试变红 → 按 SHA-256 还原（自带 `mutate_test.py`） |
| `drive_tui.py` | 在真 pty 里驱动 TUI 并逐步断言，含最小 vt100 屏幕模型（自带 `drive_tui_test.py`） |
| `check_real_switch.sh` | 验证 `MODELVIEW_REAL=1` 真的拦得住「静默跳过」 |
| `check_readme.py` | 核对本文档里可机械验证的名字与路径（选项名 / 仓库路径 / 扫描根 / 脚本名 / 速查表组名）与代码一致 |

验证产物**入库**（`internal/decode/testdata/`、`internal/ref/testdata/`），
所以日常 `go test` 既不需要网络也不需要 Python —— 重跑上面这些脚本是为了**重新生成**它们。

## 已知限制

- **不支持没有头部的裸权重**（`raw.bin` 之类），无法可靠识别格式
- **不做推理**，不对比输出质量；量化模拟给的是数值误差，不是下游任务指标
- **不做原始字节 hex dump**：`--json` 里有 `offset` 与 `byte_size`，配 `dd` / `xxd` 比在界面里翻页直接
- K 系列量化模拟的门禁是**解码值复现率**而非字节一致率：子块存在符号规范自由度
  （`(d, sc, q)` 与 `(-d, -sc, 64-q)` 是同一批数值），实测 Q6_K 40/40、Q4_K 39/40 个真实块数值完全相同
- IQ 系列与 NVFP4 / MXFP4 只收录块结构，不解码

## 设计文档

- [设计文档](docs/superpowers/specs/2026-09-26-modelview-design.md) —— 目标、非目标、各格式解析要点、判据取舍与实测依据
- [实施计划](docs/superpowers/plans/) —— 分阶段的实现计划

## 许可证

[MIT](LICENSE)

仓库里入库的验证产物（`internal/decode/testdata/`、`internal/ref/testdata/`）派生自
[llama.cpp 的 `gguf` 包](https://github.com/ggml-org/llama.cpp)（MIT）与
真实模型文件的少量字节（用于对齐上游行为），随本项目一同按 MIT 分发。
