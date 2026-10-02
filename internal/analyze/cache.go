package analyze

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/sillydong/modelview/internal/model"
)

// schemaVersion 是缓存内容的版本号。
//
// **任何会让统计数值变化的改动都必须递增**，不只是 model.Stats 的结构：
// 结构变了会静默读到零值；而块布局修正（比如 Q3_K 的某个位移写错了）
// 会让旧缓存里的数字**错但看着正常**，且缓存的设计原则是静默使用，
// 没有任何测试或日志能发现。
//
// 具体清单：Stats 结构、decode 的块布局或解码算法、采样策略。
// 2：删了 Stats.HistMin/HistMax、修了 BOOL 解码、采样上限进了缓存身份。
// 3：缓存内容从「只有 Stats」改成「Stats + Quant + QuantSims」——
//
//	同一份旧缓存用新代码读会命中，但量化分析整块缺失且不报错。
//
// 4：QuantSim 加了 MaxRelErrSig。
//
// **改了算法却忘了递增，症状是"改动看起来完全没生效"**。
// 实测过：给 max_rel_err 换四种阈值分别构建、跑同一个真实模型，
// 四次输出逐字节相同 —— 命中的都是旧缓存里的值，与改动无关。
// 那一次是**测量**被缓存骗了（加 --no-cache 才看到真实差异）；
// 换成用户升级程序，同样的机制会让新算法算出的数一个都到不了手上，
// 而屏幕上没有任何东西说明这件事。
//
// 所以：改完算法做前后对比时**必须加 --no-cache**；改了会让数值
// 变化的算法，这里要 +1。这两句是给下一个人的操作说明，不是背景介绍。
const schemaVersion = 4

// cacheDirName 是放在模型文件旁边的缓存目录名。
//
// **带前导点**，与 .gitignore 的规则和设计文档 spec §6.3 一致 ——
// 模型文件常常放在 git 仓库里（本项目自己的 artifacts 就是），
// 目录名不带点会让缓存出现在 git status 里。
const cacheDirName = ".modelview-cache"

// cache 是两级缓存：优先放在模型文件旁边，不可写时退到用户缓存目录。
//
// 优先放旁边是为了"把模型目录整体拷走时缓存跟着走"；
// 退到用户缓存目录是为了只读挂载（容器、只读硬盘）下仍然能缓存。
type cache struct {
	modelPath string
	disabled  bool
	// path 解析一次就记住：writable 探测有副作用（会创建目录），
	// 每次调用都探一遍没必要，也可能在两次调用之间结果不同。
	resolved string
}

func newCache(modelPath string, disabled bool) *cache {
	return &cache{modelPath: modelPath, disabled: disabled}
}

// cacheRecord 是缓存文件的内容。
type cacheRecord struct {
	SchemaVersion int   `json:"schema_version"`
	FileSize      int64 `json:"file_size"`
	FileMTime     int64 `json:"file_mtime"` // Unix 纳秒

	// SampleLimit 是生成这份统计时用的采样上限。
	//
	// **必须进缓存身份**：统计结果取决于它，同一份文件用不同上限跑出来的
	// 是两份不同的数据。实测踩过：先跑 --sample-limit 10 再按默认跑，
	// 拿到的仍是 count=10 的那份，用户改参数看不到任何变化，
	// 而 CLI 帮助文字承诺的语义被缓存悄悄推翻了。
	SampleLimit int `json:"sample_limit"`

	Tensors map[string]*cachedTensor `json:"tensors"`
}

// cachedTensor 是一个张量的全部派生结果。
//
// **三件必须一起存取**：Stats 与 Quant/QuantSims 都是同一个张量的分析产物，
// 而"缓存命中"与"这个张量不用再算了"必须是同一件事。
// 只缓存 Stats 会让跳过判断（`Stats != nil`）与真实意图
// （本次输出所需的东西都在）不再等价 —— 实测表现是同一个文件
// 第一次跑有 181 个模拟 / 253 个诊断，第二次整列变成 0 / 0，且不报错。
type cachedTensor struct {
	Stats     *model.Stats     `json:"stats,omitempty"`
	Quant     *model.QuantInfo `json:"quant,omitempty"`
	QuantSims []model.QuantSim `json:"quant_sims,omitempty"`
}

// primaryPath 是首选的缓存路径：模型文件旁边的 .modelview-cache/<名字>.json。
func (c *cache) primaryPath() string {
	return filepath.Join(filepath.Dir(c.modelPath), cacheDirName,
		filepath.Base(c.modelPath)+".json")
}

// fallbackPath 是退路：<用户缓存目录>/modelview/<绝对路径的 sha256>.json。
//
// 用路径哈希而不是文件名：实测 artifacts 里有 37 份都叫
// mutation.safetensors 的文件，内容两两不同，按名字存会互相覆盖。
func (c *cache) fallbackPath() string {
	base, err := os.UserCacheDir()
	if err != nil {
		base = os.TempDir()
	}
	abs, err := filepath.Abs(c.modelPath)
	if err != nil {
		abs = c.modelPath
	}
	sum := sha256.Sum256([]byte(abs))
	return filepath.Join(base, "modelview", hex.EncodeToString(sum[:])+".json")
}

// path 返回实际使用的缓存路径；禁用缓存或都不可写时返回空串。
//
// 解析一次后记住结果 —— 判断可写性的方式是按需创建目录，
// 每次调用都做一遍会产生无谓的副作用。
func (c *cache) path() string {
	if c.disabled {
		return ""
	}
	if c.resolved != "" {
		return c.resolved
	}
	if dirWritable(filepath.Dir(c.primaryPath())) {
		c.resolved = c.primaryPath()
		return c.resolved
	}
	if dirWritable(filepath.Dir(c.fallbackPath())) {
		c.resolved = c.fallbackPath()
		return c.resolved
	}
	return ""
}

// dirWritable 确认目录可写：不存在就建，存在就**实际探测一次**。
//
// 不能只看 os.MkdirAll 的返回值 —— 它对**已存在**的目录直接返回 nil，
// 不检查可写性。后果是：缓存目录一旦建成，path() 就永远解析到主路径，
// 写失败被静默吞掉，**永不回退**到用户缓存目录。
//
// 实测：删掉缓存文件后 chmod 555 两个目录（模拟只读镜像），
// 既不写主缓存也不写退路，而注释承诺的正是"只读挂载下仍然能缓存"。
// 只在"从没缓存过"时回退成立 —— 那不是设计，是巧合。
func dirWritable(dir string) bool {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return false
	}
	f, err := os.CreateTemp(dir, ".probe-*")
	if err != nil {
		return false
	}
	name := f.Name()
	//nolint:errcheck // 探测文件的清理，失败不影响可写性判断
	f.Close()
	//nolint:errcheck // 同上
	os.Remove(name)
	return true
}

// exists 判断缓存文件是否已经写出来。
//
// **目前只有测试在调**：生产路径不需要问"有没有" —— load 自己会静默
// 跳过缺失/损坏的缓存。留着是因为测试要断言"写成功了吗"，而
// c.path() 为空（禁用缓存、两处都不可写）时它也答得对。
func (c *cache) exists() bool {
	p := c.path()
	if p == "" {
		return false
	}
	_, err := os.Stat(p)
	return err == nil
}

// load 读缓存并就地填入 Tensor.Stats。
//
// **任何问题都静默跳过**：缓存是加速手段，损坏、缺失、过期都不该让
// 主流程报错 —— 重扫一遍即可。这与"解析失败必须报错"是两回事：
// 缓存是我们自己写的派生数据，模型文件才是事实来源。
func (c *cache) load(m *model.Model, sampleLimit int) error {
	p := c.path()
	if p == "" {
		return nil
	}
	raw, err := os.ReadFile(p)
	if err != nil {
		return nil
	}
	var rec cacheRecord
	if err := json.Unmarshal(raw, &rec); err != nil {
		return nil
	}
	if !c.matches(&rec, sampleLimit) {
		return nil
	}
	for _, tn := range m.Tensors {
		ct, ok := rec.Tensors[tn.Name]
		if !ok {
			continue
		}
		tn.Stats = ct.Stats
		tn.Quant = ct.Quant
		tn.QuantSims = ct.QuantSims
	}
	return nil
}

// save 写出缓存。写失败不报错 —— 只读目录下缓存本来就是尽力而为。
func (c *cache) save(m *model.Model, sampleLimit int) error {
	p := c.path()
	if p == "" {
		return nil
	}
	st, err := os.Stat(c.modelPath)
	if err != nil {
		return nil
	}

	rec := cacheRecord{
		SchemaVersion: schemaVersion,
		FileSize:      st.Size(),
		FileMTime:     st.ModTime().UnixNano(),
		SampleLimit:   sampleLimit,
		Tensors:       make(map[string]*cachedTensor, len(m.Tensors)),
	}
	for _, tn := range m.Tensors {
		if tn.Stats == nil && tn.Quant == nil && len(tn.QuantSims) == 0 {
			continue
		}
		rec.Tensors[tn.Name] = &cachedTensor{
			Stats:     tn.Stats,
			Quant:     tn.Quant,
			QuantSims: tn.QuantSims,
		}
	}
	if len(rec.Tensors) == 0 {
		return nil
	}

	raw, err := json.Marshal(rec)
	if err != nil {
		return nil
	}
	// 先写临时文件再改名：写到一半被打断时不会留下半截 JSON，
	// 下次读到的要么是完整的旧内容，要么是完整的新内容。
	tmp := p + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return nil
	}
	if err := os.Rename(tmp, p); err != nil {
		//nolint:errcheck // 改名失败后清理临时文件，失败不影响结论
		os.Remove(tmp)
		return nil
	}
	return nil
}

// matches 判断缓存是否仍然对应这个模型文件。
//
// 四个条件缺一不可：结构版本、文件大小、修改时间、采样上限。
// 只比大小会漏掉"内容变了但大小恰好相同"，只比 mtime 会在拷贝文件时失效，
// 不比采样上限会让 --sample-limit 在第二次运行时静默失效。
//
// 大小与时间都从**文件本身**取，不用 model.Model 里的 FileSize ——
// 那要求调用方一定填对，而这里是防"缓存张冠李戴"的最后一道闸门，
// 不该依赖上游的自觉。实测踩过：调用方漏填时缓存永远不命中，
// 而且不报错，只是每次都重扫。
func (c *cache) matches(rec *cacheRecord, sampleLimit int) bool {
	if rec.SchemaVersion != schemaVersion {
		return false
	}
	if rec.SampleLimit != sampleLimit {
		return false
	}
	st, err := os.Stat(c.modelPath)
	if err != nil {
		return false
	}
	if st.Size() != rec.FileSize {
		return false
	}
	return st.ModTime().UnixNano() == rec.FileMTime
}
