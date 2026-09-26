package pytorch

import "testing"

// 反向门禁：storageDtype 里能映射出的每个类型都必须能取到每元素字节数。
//
// Parse 在取不到字节数时会直接报错。如果某个存储类映射到了没有字节数的
// 类型，那个文件就会解析失败 —— 这条测试让矛盾在加映射时就暴露。
func TestStorageDtype_全部可取字节数(t *testing.T) {
	for class, d := range storageDtype {
		if _, ok := d.ByteSize(); !ok {
			t.Errorf("存储类 %q 映射到 %s，但它没有每元素字节数；该类型的文件会解析失败",
				class, d)
		}
	}
}

// 正向门禁：PyTorch 的存储类不能漏。
//
// 漏一个不会报错，只会让该类型的文件降级成"未知存储类 + SizeUnknown"，
// 参数总量仍然对但字节数不可用 —— 属于静默能力缺失。
// 这份列表来自 PyTorch 的 torch.storage 命名，本地语料只覆盖了
// FloatStorage(21 个文件) 与 HalfStorage(1 个文件)。
func TestStorageDtype_命名齐全(t *testing.T) {
	want := []string{
		"DoubleStorage", "FloatStorage", "HalfStorage", "BFloat16Storage",
		"ByteStorage", "CharStorage", "ShortStorage", "IntStorage",
		"LongStorage", "BoolStorage",
	}
	for _, class := range want {
		if _, err := parseStorageClass("torch." + class); err != nil {
			t.Errorf("存储类 %q 未收录: %v", class, err)
		}
	}
	if len(storageDtype) != len(want) {
		t.Errorf("storageDtype 有 %d 项，列表是 %d 项", len(storageDtype), len(want))
	}
}
