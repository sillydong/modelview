// Package analyze 在已解析的 model.Model 之上做数值统计与缓存。
//
// 本包只认识 model.Model，不知道底层是 GGUF、safetensors 还是 PyTorch ——
// 格式差异全部收敛在 openSource 里。
package analyze

import (
	"archive/zip"
	"fmt"
	"io"
	"os"

	"github.com/sillydong/modelview/internal/model"
)

// source 按需提供张量的原始字节。
type source interface {
	// readRaw 从张量 t 的数据区偏移 off 处读 n 个字节。
	readRaw(t *model.Tensor, off, n int64) ([]byte, error)
	Close() error
}

// openSource 按格式选择定位方式。
//
// 两种形态：
//   - GGUF / safetensors：张量在文件里有绝对偏移，直接 ReadAt
//   - PyTorch：数据在 ZIP 条目 <前缀>/data/<StorageKey> 里，要先打开条目
func openSource(m *model.Model) (source, error) {
	switch m.Format {
	case model.FormatPyTorch:
		return newZipSource(m)
	case model.FormatGGUF, model.FormatSafeTensors:
		return newFileSource(m)
	default:
		return nil, fmt.Errorf("格式 %s 不支持读取张量数据", m.Format)
	}
}

// fileSource 用于张量在文件里有绝对偏移的格式。
type fileSource struct{ f *os.File }

func newFileSource(m *model.Model) (source, error) {
	f, err := os.Open(m.Path)
	if err != nil {
		return nil, fmt.Errorf("打开 %s: %w", m.Path, err)
	}
	return &fileSource{f: f}, nil
}

func (s *fileSource) readRaw(t *model.Tensor, off, n int64) ([]byte, error) {
	if t.OffsetUnknown {
		return nil, fmt.Errorf("张量 %s 的偏移未知，无法定位数据", t.Name)
	}
	buf := make([]byte, n)
	if _, err := s.f.ReadAt(buf, t.Offset+off); err != nil {
		return nil, fmt.Errorf("读张量 %s 偏移 %d 处 %d 字节: %w", t.Name, t.Offset+off, n, err)
	}
	return buf, nil
}

func (s *fileSource) Close() error { return s.f.Close() }

// zipSource 用于 PyTorch 的 .pt。
//
// 每个张量的数据在独立条目 <前缀>/data/<StorageKey> 里，
// 张量在条目内的偏移是 StorageOffset 个**元素**（不是字节）。
type zipSource struct {
	zr     *zip.Reader
	f      *os.File
	prefix string

	// 单槽缓存：只保留**最近一次**读过的存储块。
	//
	// 权重绑定（两个张量共享一块存储）在 .pt 里很常见 —— 实测 model.pt 的
	// token_embedding 与 lm_head 就是，两者相邻访问，单槽足够省掉那次解压。
	//
	// 但**不能缓存全部**：ZIP 条目无法随机访问，只能整块解压（io.ReadAll），
	// 缓存全部会让 --stats 一个大型 .pt 的峰值内存等于该文件全部张量数据
	// （7B fp16 检查点约 14 GB）。那与"读路径内存有界"的设计前提矛盾，
	// 而 GGUF / safetensors 两条路径都只读需要的 8 MiB。
	lastKey  string
	lastData []byte
}

func newZipSource(m *model.Model) (source, error) {
	if m.ArchivePrefix == "" {
		return nil, fmt.Errorf("%s 缺少归档前缀，无法定位张量数据", m.Path)
	}
	f, err := os.Open(m.Path)
	if err != nil {
		return nil, fmt.Errorf("打开 %s: %w", m.Path, err)
	}
	st, err := f.Stat()
	if err != nil {
		//nolint:errcheck // 打开失败后的清理，Close 的错误不改变结论
		f.Close()
		return nil, fmt.Errorf("stat %s: %w", m.Path, err)
	}
	zr, err := zip.NewReader(f, st.Size())
	if err != nil {
		//nolint:errcheck // 打开失败后的清理，Close 的错误不改变结论
		f.Close()
		return nil, fmt.Errorf("打开 ZIP: %w", err)
	}
	return &zipSource{zr: zr, f: f, prefix: m.ArchivePrefix}, nil
}

func (s *zipSource) readRaw(t *model.Tensor, off, n int64) ([]byte, error) {
	eb, ok := t.Dtype.ByteSize()
	if !ok {
		// 量化类型在 .pt 里不会出现（PyTorch 用 float 存储），
		// 真出现说明我们对格式的理解有偏差。
		return nil, fmt.Errorf("张量 %s 的类型 %s 没有每元素字节数", t.Name, t.Dtype)
	}
	start := t.StorageOffset*eb + off

	data := s.lastData
	if s.lastKey != t.StorageKey || data == nil {
		name := s.prefix + "/data/" + t.StorageKey
		rc, err := s.zr.Open(name)
		if err != nil {
			return nil, fmt.Errorf("打开 %s: %w", name, err)
		}
		//nolint:errcheck // 只读条目，Close 失败不影响数据完整性
		defer rc.Close()
		data, err = io.ReadAll(rc)
		if err != nil {
			return nil, fmt.Errorf("读取 %s: %w", name, err)
		}
		s.lastKey, s.lastData = t.StorageKey, data
	}

	if start < 0 || start+n > int64(len(data)) {
		return nil, fmt.Errorf("张量 %s 要读 [%d,%d)，存储块只有 %d 字节",
			t.Name, start, start+n, len(data))
	}
	return data[start : start+n], nil
}

func (s *zipSource) Close() error {
	s.lastData = nil
	return s.f.Close()
}
