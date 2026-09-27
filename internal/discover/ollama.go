package discover

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ollamaMediaModel 是模型权重那一层。只有它代表模型文件本身。
const ollamaMediaModel = "application/vnd.ollama.image.model"

// manifestLayer 是 manifest 里的一层。
type manifestLayer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
}

// ollamaManifest 是 manifest 文件的结构（Docker manifest v2）。
//
// **`config` 字段不能漏**：它指向一个几百字节的 blob，代表这个模型在
// ollama 里的镜像配置。只看 `layers` 的话，那 4 个 config blob 会被
// 当成"没有被任何 manifest 引用"的孤儿 —— 实测本机 15 个 blob 里
// 恰好 11 个在 layers、4 个是 config，漏掉 config 就凭空多报 4 个孤儿，
// 而"哪些磁盘可以释放"正是这个功能给用户的结论。
type ollamaManifest struct {
	Config manifestLayer   `json:"config"`
	Layers []manifestLayer `json:"layers"`
}

// scanOllama 解析 manifests 目录，返回模型条目、孤儿 blob 与警告。
//
// 单个 manifest 坏掉**不中断整体**，也不丢掉已经解析出来的 ——
// 检查器的职责就是报告，不能因为一个文件坏了就闭嘴。
//
// **但只要有一个 manifest 读不了或解析不了，就完全不报孤儿**。
// 读不了的 manifest 到底引用了哪些 blob 是无从知晓的，把"不知道"
// 当成"没被引用"，用户照着"可回收"去删就会删掉真实模型 ——
// 这是本工具唯一一条会引导破坏性操作的路径，宁可少报也不能错报。
// 实测：把一个 manifest 截断（模拟写到一半被打断），它引用的 blob
// 立刻出现在"孤儿 blob（可回收）"里。
func scanOllama(root string) (items, orphans []Item, errs []string) {
	manifestsDir := filepath.Join(root, "manifests")

	// referenced 收集全部被引用过的 digest。**用集合而不是计数**：
	// 共享层（本机实测 gemma4 的 e4b 与 26b 共享 license 与 params）
	// 会被多个 manifest 引用，按次数算的话第二次就把它算成别的了
	referenced := map[string]bool{}
	// unparsed 是"看起来是 manifest 但读不了"的个数
	unparsed := 0

	err := filepath.WalkDir(manifestsDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil || d.IsDir() {
			return nil // 单个条目读不了就跳过
		}
		// 先判断它**看起来是不是一个 manifest**：不是（如 .DS_Store、
		// 目录里误放的其他文件）就不该影响孤儿判定
		name, ok := manifestPathToName(manifestsDir, path)
		if !ok {
			return nil // 层数不够，不是一个 model/tag 文件
		}
		raw, readErr := os.ReadFile(path)
		if readErr != nil {
			unparsed++
			return nil
		}
		var mf ollamaManifest
		if json.Unmarshal(raw, &mf) != nil {
			// 坏 manifest（写到一半被打断、被截断）：它引用了哪些 blob
			// 无从知晓 —— 记一笔，最后据此决定要不要报孤儿
			unparsed++
			return nil
		}

		// config 里的 blob 也是被引用的，不是孤儿
		if mf.Config.Digest != "" {
			referenced[mf.Config.Digest] = true
		}

		var it Item
		var extra []KV
		for _, l := range mf.Layers {
			referenced[l.Digest] = true
			p := blobPath(root, l.Digest)
			if l.MediaType == ollamaMediaModel {
				it = Item{Source: SourceOllama, Name: name, Path: p, Size: l.Size}
				continue
			}
			// 非模型层：把内容读出来（template / system / params 是文本，
			// license 可能较长 —— 截断显示但标明总长，界面要能展开看全文）
			if kv, ok := readLayerLabel(p, l.MediaType); ok {
				extra = append(extra, kv)
			}
		}
		if it.Path == "" {
			return nil // 没有 model 层的 manifest 不是模型
		}
		if st, statErr := os.Stat(it.Path); statErr == nil {
			it.Size = st.Size() // 实际占用以文件系统为准
		} else {
			// manifest 指向的 blob 不在了：这是要报的 ——
			// 用户会以为模型还在
			it.Err = "blob 缺失：" + it.Path
		}
		if len(extra) > 0 {
			sort.Slice(extra, func(i, j int) bool { return extra[i].Key < extra[j].Key })
			it.Extra = extra
		}
		items = append(items, it)
		return nil
	})
	if err != nil {
		return nil, nil, []string{fmt.Sprintf("%s: %v", manifestsDir, err)}
	}

	if unparsed > 0 {
		// 不报孤儿，并说清为什么 —— 静默不报的话，用户会以为
		// "没有孤儿"是算出来的结论
		errs = append(errs, fmt.Sprintf(
			"%s 下有 %d 个 manifest 读不了，孤儿 blob 检测已跳过："+
				"无法确认这些 blob 是否被引用，报成「可回收」可能让你删掉真实模型",
			manifestsDir, unparsed))
		return items, nil, errs
	}
	return items, findOrphans(root, referenced), errs
}

// readLayerLabel 把非模型层的内容读成一条 KV。读不了就返回 false。
//
// 键用 mediaType 的最后一段（template / system / license / params），
// 它比完整 mediaType 好读，又不会在层种类变化时撞车。
func readLayerLabel(path, mediaType string) (KV, bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return KV{}, false
	}
	key := mediaType
	if i := strings.LastIndex(mediaType, "."); i >= 0 {
		key = mediaType[i+1:]
	}
	const maxShow = 4096
	v := string(raw)
	if len(v) > maxShow {
		v = v[:maxShow] + fmt.Sprintf("…（共 %d 字节）", len(raw))
	}
	return KV{Key: key, Value: v}, true
}

// findOrphans 返回没有被任何 manifest 引用的 blob。
//
// spec §7.2：这是纯增量价值 —— 直接告诉用户哪些磁盘可以释放。
func findOrphans(root string, referenced map[string]bool) []Item {
	dir := blobsDir(root)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []Item
	for _, e := range entries {
		if e.IsDir() || !strings.HasPrefix(e.Name(), "sha256-") {
			continue
		}
		// 目录项名是 sha256-<hex>，digest 是 sha256:<hex> ——
		// 只看名字能不能对上引用集合，不做字符串反变换（等价但更脆）
		digest := "sha256:" + strings.TrimPrefix(e.Name(), "sha256-")
		if referenced[digest] {
			continue
		}
		// 同 scanDir：ReadDir 的 DirEntry 是 lstat 语义，
		// 而这一列要回答的是"能回收多少空间" —— 报链接自身的字节数
		// 会让用户以为没什么可清的
		info, err := os.Stat(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		out = append(out, Item{
			Source: SourceOllama,
			Name:   e.Name(),
			Path:   filepath.Join(dir, e.Name()),
			Size:   info.Size(),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

// defaultHost 是 ollama 的默认 registry。
//
// 名字里的 host 与 namespace 一样，**只在等于默认值时才省略** ——
// 这与 Docker Hub 的约定一样（library 是官方镜像的默认命名空间）。
const defaultHost = "registry.ollama.ai"

// defaultNamespace 是 ollama 的默认命名空间。
//
// 本机实测：路径 registry.ollama.ai/library/qwen2.5/3b 对应的模型名是
// qwen2.5:3b，而不是 library/qwen2.5:3b。`ollama list` 显示的也是前者。
const defaultNamespace = "library"

// manifestPathToName 从 manifests/<host>/<命名空间>/<名字>/<tag>
// 还原用户可见的模型名。
//
// 规则照抄上游 `types/model.Name.DisplayShortest()`（那是 `ollama list`
// 显示名字的来源）：
//
//	host 不是默认值 → "<host>/<命名空间>/<名字>:<tag>"
//	host 是默认值、命名空间不是 → "<命名空间>/<名字>:<tag>"
//	两者都是默认值 → "<名字>:<tag>"
//
// **注意 host 不是默认值时命名空间照样要打印**（上游那个分支里
// 没有"命名空间等于 library 就省掉"的判断）。早先这里无条件砍掉
// 第一段，于是 `hf.co/org/repo:Q4_K_M` 被显示成 `org/repo:Q4_K_M` ——
// 而 ollama 自己显示的是完整的三段。从 HuggingFace 直接拉 GGUF
// （`ollama pull hf.co/...`）是很常见的用法，用户拿我们的名字去
// `ollama list` 或 `ollama rm` 里对，会对不上。
//
// 本机 4 个模型全是默认 host + library，所以这个偏差在本机观察不到 ——
// 结论来自上游源码，不是本机实测。
func manifestPathToName(manifestsDir, filePath string) (string, bool) {
	rel, err := filepath.Rel(manifestsDir, filePath)
	if err != nil {
		return "", false
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	if len(parts) < 3 {
		return "", false // 至少要 <host>/<名字>/<tag>
	}
	host := parts[0]
	tag := parts[len(parts)-1]
	middle := parts[1 : len(parts)-1]
	// <host>/<名字>/<tag> 三段时命名空间是默认值（ollama 的解析也是这样补的）
	namespace, model := defaultNamespace, middle[0]
	if len(middle) >= 2 {
		namespace, model = middle[0], strings.Join(middle[1:], "/")
	}
	if model == "" || tag == "" || host == "" {
		return "", false
	}
	var b strings.Builder
	if !strings.EqualFold(host, defaultHost) {
		b.WriteString(host)
		b.WriteByte('/')
		b.WriteString(namespace)
		b.WriteByte('/')
	} else if !strings.EqualFold(namespace, defaultNamespace) {
		b.WriteString(namespace)
		b.WriteByte('/')
	}
	b.WriteString(model)
	b.WriteByte(':')
	b.WriteString(tag)
	return b.String(), true
}

// blobsDir 返回 <root>/blobs。ollama 的实测布局是 models/ 下并列
// manifests/ 与 blobs/。
func blobsDir(root string) string { return filepath.Join(root, "blobs") }

// blobPath 把 digest "sha256:abcd" 变成 blobs/sha256-abcd。
//
// ollama 把冒号换成了短横 —— 这是它的存储约定，不是我们选的。
// **只替换第一个冒号**：digest 里不会再有冒号。
func blobPath(root, digest string) string {
	return filepath.Join(blobsDir(root), strings.Replace(digest, ":", "-", 1))
}
