package gguf

import (
	"errors"
	"fmt"
	"os"

	"github.com/sillydong/modelview/internal/model"
)

// Parse 解析一个 GGUF 文件。
//
// 只读取文件头部区域（元数据 + 张量描述符），不读取张量数据本身。
// 因此对 17 GiB 的文件也能秒级返回。
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

	r := newReader(f)

	magic := make([]byte, 4)
	if err := r.read(magic); err != nil {
		return nil, fmt.Errorf("读取 magic: %w", err)
	}
	if string(magic) != "GGUF" {
		return nil, fmt.Errorf("%s 不是 GGUF 文件（magic=%q）", path, magic)
	}

	version, err := r.u32()
	if err != nil {
		return nil, fmt.Errorf("读取版本: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("不支持的 GGUF 版本 %d（仅支持 v2/v3）", version)
	}

	tensorCount, err := r.u64()
	if err != nil {
		return nil, fmt.Errorf("读取张量个数: %w", err)
	}
	kvCount, err := r.u64()
	if err != nil {
		return nil, fmt.Errorf("读取元数据条数: %w", err)
	}

	kvs, err := readKVs(r, kvCount)
	if err != nil {
		return nil, fmt.Errorf("解析元数据: %w", err)
	}

	infos, err := readTensorInfos(r, tensorCount)
	if err != nil {
		return nil, fmt.Errorf("解析张量描述符: %w", err)
	}

	// 数据区起点：张量描述符结束位置向上对齐到 general.alignment（默认 32）。
	// 省掉这一步会读出垃圾数据。
	//
	// alignment 取值异常时记录告警而不是静默退回默认值 ——
	// 否则所有张量的 offset 会整片偏移，界面上却看不出任何异常。
	alignment := int64(32)
	var warnings []string
	for _, kv := range kvs {
		if kv.Key != "general.alignment" {
			continue
		}
		switch v := kv.Value.(type) {
		case uint32:
			if v == 0 {
				warnings = append(warnings,
					"general.alignment 为 0，已按默认值 32 处理（该文件的张量偏移可能与实际不符）")
			} else {
				alignment = int64(v)
			}
		default:
			warnings = append(warnings,
				fmt.Sprintf("general.alignment 类型异常（%T），已按默认值 32 处理（该文件的张量偏移可能与实际不符）", v))
		}
	}
	headerEnd := r.offset()
	dataStart := alignUp(headerEnd, alignment)

	m := &model.Model{
		Path:        path,
		Format:      model.FormatGGUF,
		Version:     fmt.Sprintf("v%d", version),
		FileSize:    st.Size(),
		Metadata:    make([]model.MetaKV, 0, len(kvs)),
		Tensors:     make([]*model.Tensor, 0, len(infos)),
		Alignment:   alignment,
		DataStart:   dataStart,
		HeaderBytes: headerEnd,
	}

	for _, kv := range kvs {
		m.Metadata = append(m.Metadata, model.MetaKV{
			Key:   kv.Key,
			Value: formatValue(kv.Value),
			Raw:   rawForJSON(kv.Value),
			Ref:   refFor(kv.Key),
		})
		if kv.Key == "general.architecture" {
			if s, ok := kv.Value.(string); ok {
				m.Arch = s
			}
		}
	}

	for _, info := range infos {
		dtype, known := ggmlDtype(info.Type)
		if !known {
			dtype = model.DtypeUnknown
			warnings = append(warnings,
				fmt.Sprintf("张量 %s 使用未知的 GGML 类型码 %d", info.Name, info.Type))
		}

		// 单个张量算不出大小不影响整体解析，但必须让用户看得见 ——
		// 静默给 0 会让人以为"这个张量真的是 0 字节"。
		size, err := tensorByteSize(info.Dims, info.Type)
		sizeUnknown := err != nil
		if err != nil {
			var unknownBlock ErrUnknownBlockType
			if errors.As(err, &unknownBlock) {
				warnings = append(warnings,
					fmt.Sprintf("张量 %s 的类型 %s 块结构未收录，无法计算占用大小", info.Name, dtype))
			} else {
				// 非「未收录」的错误意味着文件本身有问题（维度为负、元素数不是块整数倍）
				warnings = append(warnings,
					fmt.Sprintf("张量 %s 的大小计算失败：%v", info.Name, err))
			}
			size = 0
		}

		var params int64 = 1
		for _, d := range info.Dims {
			params *= d
		}

		m.Tensors = append(m.Tensors, &model.Tensor{
			Name:        info.Name,
			Dims:        info.Dims,
			Dtype:       dtype,
			Offset:      dataStart + info.Offset,
			ByteSize:    size,
			ParamCount:  params,
			SizeUnknown: sizeUnknown,
		})
	}

	m.Warnings = warnings
	return m, nil
}

// rawForJSON 把内部值转成可被 encoding/json 序列化的形式。
func rawForJSON(v any) any {
	switch x := v.(type) {
	case arrayValue:
		switch {
		case x.StrElems != nil:
			return x.StrElems
		case x.NumElems != nil:
			out := make([]any, len(x.NumElems))
			for i, e := range x.NumElems {
				out[i] = rawForJSON(e)
			}
			return out
		case x.Nested != nil:
			out := make([]any, len(x.Nested))
			for i, e := range x.Nested {
				out[i] = rawForJSON(e)
			}
			return out
		}
		return nil
	default:
		return v
	}
}

// refFor 返回元数据键关联的速查表条目 ID（计划 ④ 填充表数据）。
func refFor(key string) string {
	switch key {
	case "general.file_type":
		return "quant-schemes"
	case "general.architecture", "general.alignment":
		return "gguf-keys"
	default:
		return ""
	}
}
