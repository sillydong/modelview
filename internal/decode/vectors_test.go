package decode

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

// vector 是 tools/verify_ggml_dequant.py 生成的一条真值。
type vector struct {
	BlockSize int       `json:"block_size"`
	TypeSize  int       `json:"type_size"`
	Bytes     string    `json:"bytes"`
	Expect    []float64 `json:"expect"`
}

// tolerance 是逐值比对允许的绝对偏差。
//
// 参考实现走 numpy 的 float32 向量化运算，本实现是标量循环，
// 中间结果的舍入路径不同。实测两者差在 1e-8 量级，1e-5 足够宽。
const tolerance = 1e-5

// loadVectors 读真值。文件缺失或损坏时**判失败**而不是跳过 ——
// 没有真值就等于量化解码完全没有守卫。
func loadVectors(t *testing.T) map[string]vector {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "vectors.json"))
	if err != nil {
		t.Fatalf("读 testdata/vectors.json 失败（用 uv run --with gguf --with numpy "+
			"python3 tools/verify_ggml_dequant.py 生成）: %v", err)
	}
	var all map[string]vector
	if err := json.Unmarshal(raw, &all); err != nil {
		t.Fatalf("解析 vectors.json: %v", err)
	}
	if len(all) == 0 {
		t.Fatal("vectors.json 是空的 —— 比对会变成空转")
	}
	return all
}

// 每种量化类型的解码结果都必须与参考实现逐值吻合。
//
// 这是量化解码唯一的外部证据：参考实现来自 llama.cpp，
// 不是本项目自己算的 —— 自己算的就是同一份理解写两遍，两边一起错时不会有提示。
func TestDecode_量化类型与参考实现吻合(t *testing.T) {
	all := loadVectors(t)
	for name, v := range all {
		t.Run(name, func(t *testing.T) {
			src, err := hex.DecodeString(v.Bytes)
			if err != nil {
				t.Fatalf("向量不是合法 hex: %v", err)
			}
			dst := make([]float32, len(v.Expect))
			if err := Decode(model.Dtype(name), src, dst); err != nil {
				t.Fatalf("Decode 失败: %v", err)
			}

			// 全部值都比一遍再报第一个 —— 只看第一个不符的值，
			// 遇到"一半对一半错"这种典型症状时定位不到规律。
			bad, worst, worstAt := 0, 0.0, -1
			for i := range dst {
				d := math.Abs(float64(dst[i]) - v.Expect[i])
				if d > worst {
					worst, worstAt = d, i
				}
				if d > tolerance {
					if bad < 3 {
						t.Errorf("[%d] = %.9g, want %.9g", i, dst[i], v.Expect[i])
					}
					bad++
				}
			}
			if bad > 0 {
				t.Fatalf("%d/%d 个值不符，最大偏差 %.6g（下标 %d）",
					bad, len(dst), worst, worstAt)
			}
		})
	}
}

// 反向门禁：真值文件里的每个类型都必须真的被解码支持。
//
// 少了这条，往 vectors.json 里加一种类型而解码没实现时，
// 测试会以「Decode 失败」的名义红 —— 但如果有人把失败改成跳过，
// 就会静默变成空转。这条让"支持的类型集合"与"真值集合"必须对齐。
func TestDecode_真值覆盖的类型都已实现(t *testing.T) {
	all := loadVectors(t)
	for name := range all {
		d := model.Dtype(name)
		if !d.IsQuantized() {
			t.Errorf("%s 不是量化类型，不该出现在量化块真值里", name)
			continue
		}
		src := make([]byte, 4096)
		dst := make([]float32, int(d.BlockElems()))
		err := Decode(d, src, dst)
		if errors.As(err, &ErrUnsupported{}) {
			t.Errorf("%s 有真值但解码报未收录 —— 要么实现它，要么从真值里去掉", name)
		}
	}
}

// 真值文件自身的结构必须与块表吻合 —— 换了 gguf 包版本导致块尺寸变了要能发现。
func TestVectors_与块表自洽(t *testing.T) {
	all := loadVectors(t)
	for name, v := range all {
		d := model.Dtype(name)
		if got := int(d.BlockElems()); got != v.BlockSize {
			t.Errorf("%s: 真值块元素数 %d，块表 %d", name, v.BlockSize, got)
		}
		bb, ok := d.BlockBytes()
		if !ok {
			t.Errorf("%s: 块表里没有块字节数", name)
			continue
		}
		if int(bb) != v.TypeSize {
			t.Errorf("%s: 真值块字节数 %d，块表 %d", name, v.TypeSize, bb)
		}
	}
}
