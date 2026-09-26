// Package detect 按文件内容（magic bytes）判断模型格式，不看扩展名。
package detect

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/sillydong/modelview/internal/model"
)

var (
	ggufMagic = []byte{'G', 'G', 'U', 'F'}
	zipMagic  = []byte{'P', 'K', 3, 4}
)

// maxHeaderLen 是 safetensors 头部长度的合理上限（100 MiB）。
// 超过此值说明不是 safetensors，避免把随机字节误判成合法头部。
const maxHeaderLen = 100 << 20

// Probe 读取文件头部判断格式。文件不存在或无读取权限时返回错误。
func Probe(path string) (model.Format, error) {
	f, err := os.Open(path)
	if err != nil {
		return model.FormatUnknown, fmt.Errorf("打开文件: %w", err)
	}
	defer f.Close()

	var head [8]byte
	n, err := io.ReadFull(f, head[:])
	if err != nil && err != io.ErrUnexpectedEOF && err != io.EOF {
		return model.FormatUnknown, fmt.Errorf("读取文件头: %w", err)
	}
	if n < 4 {
		return model.FormatUnknown, nil
	}

	switch {
	case bytes.Equal(head[:4], ggufMagic):
		return model.FormatGGUF, nil
	case bytes.Equal(head[:4], zipMagic):
		return model.FormatPyTorch, nil
	}

	// safetensors: 前 8 字节是小端 u64 的头部长度，紧接着是 JSON 对象的 '{'。
	if n == 8 {
		hlen := binary.LittleEndian.Uint64(head[:])
		if hlen > 0 && hlen < maxHeaderLen {
			var first [1]byte
			if _, err := f.ReadAt(first[:], 8); err == nil && first[0] == '{' {
				return model.FormatSafeTensors, nil
			}
		}
	}

	return model.FormatUnknown, nil
}
