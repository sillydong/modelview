// Package safetensors 解析 safetensors 格式。
//
// 文件布局：
//
//	[8 字节小端 u64 = 头部长度 N][N 字节 JSON 头部][裸张量数据]
//
// 头部形如 {"名字": {"dtype": "F32", "shape": [...], "data_offsets": [起, 止]}}，
// 另有可选的 "__metadata__" 键（字符串到字符串的映射，不是张量）。
//
// data_offsets 是**相对数据区**的偏移，
// 绝对偏移 = 8 + N + data_offsets[i]。
package safetensors

import (
	"encoding/json"
	"fmt"
	"slices"
)

// tensorEntry 是头部里一个张量的描述。
type tensorEntry struct {
	Dtype       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// header 是解析后的头部。
type header struct {
	Tensors  map[string]tensorEntry
	Metadata map[string]string
	// Names 是按字典序排序的张量名。map 遍历无序，
	// 不排序的话输出不稳定，测试会间歇性失败。
	Names []string
}

// maxHeaderLen 是头部长度的合理上限（100 MiB）。
// safetensors 头部只存形状与偏移，正常在 MB 以内。
const maxHeaderLen = 100 << 20

// parseHeader 解析头部 JSON。
func parseHeader(raw []byte) (*header, error) {
	// 用 RawMessage 中转：__metadata__ 的值是 map[string]string，
	// 其它键的值是 tensorEntry，类型不同不能一次解出来。
	var rawMap map[string]json.RawMessage
	if err := json.Unmarshal(raw, &rawMap); err != nil {
		return nil, fmt.Errorf("解析头部 JSON: %w", err)
	}

	h := &header{Tensors: make(map[string]tensorEntry, len(rawMap))}
	for name, msg := range rawMap {
		if name == "__metadata__" {
			var meta map[string]string
			if err := json.Unmarshal(msg, &meta); err != nil {
				return nil, fmt.Errorf("解析 __metadata__: %w", err)
			}
			h.Metadata = meta
			continue
		}
		var e tensorEntry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, fmt.Errorf("解析张量 %q: %w", name, err)
		}
		if e.Dtype == "" {
			return nil, fmt.Errorf("张量 %q 缺少 dtype", name)
		}
		if e.DataOffsets[0] < 0 || e.DataOffsets[1] < e.DataOffsets[0] {
			return nil, fmt.Errorf("张量 %q 的 data_offsets 非法: %v", name, e.DataOffsets)
		}
		h.Tensors[name] = e
	}

	h.Names = make([]string, 0, len(h.Tensors))
	for name := range h.Tensors {
		h.Names = append(h.Names, name)
	}
	slices.Sort(h.Names)
	return h, nil
}
