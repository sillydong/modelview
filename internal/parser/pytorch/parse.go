package pytorch

import (
	"archive/zip"
	"fmt"
	"io"
	"os"

	"github.com/sillydong/modelview/internal/model"
)

// maxPklSize 是 data.pkl 的大小上限（128 MiB）。
// 它只存张量元数据，正常在几十 KB 量级。
const maxPklSize = 128 << 20

// Parse 解析一个 PyTorch .pt 文件。
//
// 只读取 ZIP 目录与 data.pkl，不读取张量数据本身。
func Parse(path string) (*model.Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("打开 %s: %w", path, err)
	}
	//nolint:errcheck // 只读文件，Close 失败不影响解析结果
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", path, err)
	}

	zr, err := zip.NewReader(f, st.Size())
	if err != nil {
		return nil, fmt.Errorf("打开 ZIP: %w", err)
	}

	lay, err := findLayout(zr)
	if err != nil {
		return nil, err
	}

	pkl, err := readPkl(zr, lay.Prefix+dataPklSuffix)
	if err != nil {
		return nil, err
	}

	stack, err := runPickle(pkl)
	if err != nil {
		return nil, fmt.Errorf("执行 pickle: %w", err)
	}

	recs, err := extractTensors(stack)
	if err != nil {
		return nil, fmt.Errorf("提取张量: %w", err)
	}
	if len(recs) == 0 {
		return nil, fmt.Errorf("%s 里没有解析出任何张量", path)
	}

	m := &model.Model{
		Path:     path,
		Format:   model.FormatPyTorch,
		Version:  "zip/" + orDash(lay.FormatVersion),
		FileSize: st.Size(),
		Tensors:  make([]*model.Tensor, 0, len(recs)),
		// DataStart 留 0：ZIP 容器没有单一的数据区起点，
		// 按 model.Model 的约定 0 即表示"该格式无此概念"。
	}

	storageBytes := map[string]int64{}
	var warnings []string

	for _, rec := range recs {
		// 存储块必须真的存在于 ZIP 里。引用了不存在的 data/N，
		// 说明 data.pkl 与数据区对不上，此时偏移没有意义。
		if _, ok := lay.DataFiles[rec.StorageKey]; !ok {
			warnings = append(warnings, fmt.Sprintf(
				"张量 %s 引用的存储块 data/%s 不在归档里", rec.Name, rec.StorageKey))
		}

		dtype, err := parseStorageClass(rec.StorageClass)
		if err != nil {
			// 未知存储类不致命：保留张量并标注**字节数**不可信。
			//
			// 但 ParamCount 必须照填 —— 它只依赖形状，与存储类无关。
			// 留 0 会让 TotalParams() 少算这些张量，而摘要的头条数字
			// 正是 TotalParams()，用户从告警里看不出参数也被算错了。
			n, err := numel(rec.Shape)
			if err != nil {
				return nil, fmt.Errorf("张量 %s: %w", rec.Name, err)
			}
			warnings = append(warnings, fmt.Sprintf("张量 %s：%v", rec.Name, err))
			m.Tensors = append(m.Tensors, &model.Tensor{
				Name:          rec.Name,
				Dims:          rec.Shape,
				Dtype:         model.DtypeUnknown,
				ParamCount:    n,
				StorageKey:    rec.StorageKey,
				StorageOffset: rec.StorageOffset,
				OffsetUnknown: true,
				SizeUnknown:   true,
			})
			continue
		}

		n, err := numel(rec.Shape)
		if err != nil {
			return nil, fmt.Errorf("张量 %s: %w", rec.Name, err)
		}
		eb, ok := dtype.ByteSize()
		if !ok {
			// 存储类认得、但没有每元素字节数（不该发生：.pt 只有非量化类型）。
			// 与其算个 0 字节的假值，不如明确报出来。
			return nil, fmt.Errorf("张量 %s: 存储类 %q 映射到 %s，但它没有每元素字节数",
				rec.Name, rec.StorageClass, dtype)
		}
		size := n * eb

		if !isContiguous(rec.Shape, rec.Stride) {
			warnings = append(warnings, fmt.Sprintf(
				"张量 %s 非连续布局（stride=%v），其字节数按逻辑形状计算", rec.Name, rec.Stride))
		}
		if end := rec.StorageOffset + n; end > rec.StorageNumel {
			warnings = append(warnings, fmt.Sprintf(
				"张量 %s 的数据越出存储块：offset %d + %d 元素 > 存储块 %d 元素",
				rec.Name, rec.StorageOffset, n, rec.StorageNumel))
		}

		storageBytes[rec.StorageKey] = rec.StorageNumel * eb

		m.Tensors = append(m.Tensors, &model.Tensor{
			Name:          rec.Name,
			Dims:          rec.Shape,
			Dtype:         dtype,
			ByteSize:      size,
			ParamCount:    n,
			StorageKey:    rec.StorageKey,
			StorageOffset: rec.StorageOffset,
			OffsetUnknown: true,
		})
	}

	for _, v := range storageBytes {
		m.StorageBytes += v
	}

	// 权重绑定只填 TiedGroups，不进 Warnings。
	// 它是正常且预期的结构（语言模型的 token embedding 与 lm_head 常年绑定），
	// 既不是降级处理也不是问题；何况 TiedGroups 已是结构化字段，
	// 再写一份到 Warnings 会同时污染人类摘要和 JSON 消费方。
	m.TiedGroups = m.FindTiedGroups()
	m.Warnings = warnings

	return m, nil
}

func readPkl(zr *zip.Reader, name string) ([]byte, error) {
	f, err := zr.Open(name)
	if err != nil {
		return nil, fmt.Errorf("打开 %s: %w", name, err)
	}
	//nolint:errcheck // 只读，Close 失败无影响
	defer f.Close()

	b, err := io.ReadAll(io.LimitReader(f, maxPklSize+1))
	if err != nil {
		return nil, fmt.Errorf("读取 %s: %w", name, err)
	}
	if len(b) > maxPklSize {
		return nil, fmt.Errorf("%s 超过大小上限 %d", name, maxPklSize)
	}
	return b, nil
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
