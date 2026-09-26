package pytorch

import (
	"archive/zip"
	"fmt"
	"io"
	"strings"
)

// layout 描述一个 .pt ZIP 容器的内部布局。
type layout struct {
	// Prefix 是归档前缀，如 "model" / "cell" / "ln-c2-a0-500k"。
	//
	// **它不是固定的 "model"** —— 实测 best-500k.pt 的前缀是
	// "ln-c2-a0-500k"，既不是 "model" 也不是当前文件名。
	// PyTorch 保存时把当时的模型名写进归档名，之后文件被改名也不会跟着变。
	// 只能扫 ZIP 条目找以 "/data.pkl" 结尾的那个来动态发现。
	Prefix string
	// DataFiles 是 <Prefix>/data/<N> 的 N → 完整条目名映射。
	DataFiles map[string]string
	// FormatVersion 是 .format_version 的内容
	FormatVersion string
	// ByteOrder 是 byteorder 的内容，应为 "little"
	ByteOrder string
}

const dataPklSuffix = "/data.pkl"

// findLayout 扫描 ZIP 条目，发现归档前缀与数据文件。
func findLayout(zr *zip.Reader) (*layout, error) {
	var pklName string
	for _, f := range zr.File {
		if strings.HasSuffix(f.Name, dataPklSuffix) {
			pklName = f.Name
			break
		}
	}
	if pklName == "" {
		return nil, fmt.Errorf("ZIP 里找不到 *%s，可能不是 PyTorch 保存的文件", dataPklSuffix)
	}
	prefix := strings.TrimSuffix(pklName, dataPklSuffix)

	l := &layout{Prefix: prefix, DataFiles: map[string]string{}}
	dataPrefix := prefix + "/data/"
	for _, f := range zr.File {
		if key, ok := strings.CutPrefix(f.Name, dataPrefix); ok && key != "" {
			l.DataFiles[key] = f.Name
		}
	}

	l.FormatVersion = readZipString(zr, prefix+"/.format_version")
	l.ByteOrder = readZipString(zr, prefix+"/byteorder")

	// 大端存储需要翻转每个数值，本实现不支持。
	// 实测所有本地文件都是 little。
	if l.ByteOrder != "" && l.ByteOrder != "little" {
		return nil, fmt.Errorf("不支持的大端存储（byteorder=%q）", l.ByteOrder)
	}
	return l, nil
}

// readZipString 读取一个 ZIP 条目为字符串；条目不存在时返回空串。
func readZipString(zr *zip.Reader, name string) string {
	f, err := zr.Open(name)
	if err != nil {
		return ""
	}
	//nolint:errcheck // 只读，Close 失败无影响
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, 4096))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}
