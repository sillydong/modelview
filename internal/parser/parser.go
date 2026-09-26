// Package parser 按格式把文件分发给具体的格式解析器。
package parser

import (
	"errors"
	"fmt"

	"github.com/sillydong/modelview/internal/detect"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/parser/gguf"
	"github.com/sillydong/modelview/internal/parser/pytorch"
	"github.com/sillydong/modelview/internal/parser/safetensors"
)

// ErrUnsupported 表示格式无法识别。
// 调用方可用 errors.Is 判断。
var ErrUnsupported = errors.New("无法识别的模型格式")

// Parse 探测格式并解析。
func Parse(path string) (*model.Model, error) {
	format, err := detect.Probe(path)
	if err != nil {
		return nil, err
	}

	switch format {
	case model.FormatGGUF:
		return gguf.Parse(path)
	case model.FormatSafeTensors:
		return safetensors.Parse(path)
	case model.FormatPyTorch:
		return pytorch.Parse(path)
	default:
		return nil, fmt.Errorf("%s: %w", path, ErrUnsupported)
	}
}
