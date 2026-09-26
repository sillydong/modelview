// Package parser 按格式把文件分发给具体的格式解析器。
package parser

import (
	"errors"
	"fmt"

	"github.com/sillydong/modelview/internal/detect"
	"github.com/sillydong/modelview/internal/model"
	"github.com/sillydong/modelview/internal/parser/gguf"
)

// ErrUnsupported 表示格式无法识别或尚未实现。
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
		// 计划 ② 实现
		return nil, fmt.Errorf("SafeTensors: %w", ErrUnsupported)
	case model.FormatPyTorch:
		// 计划 ② 实现
		return nil, fmt.Errorf("PyTorch: %w", ErrUnsupported)
	default:
		return nil, fmt.Errorf("%s: %w", path, ErrUnsupported)
	}
}
