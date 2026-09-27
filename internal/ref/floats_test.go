package ref

import (
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/sillydong/modelview/internal/model"
)

// 浮点的数值属性必须由位布局**算**出来，且与教科书值吻合。
//
// 写死字面量的话，F16 的最大值抄成 65535 也不会有任何东西发现；
// 算出来的则不可能错 —— 它由 exp/mantissa 位数唯一决定。
func TestFloatFormats_数值属性(t *testing.T) {
	tests := []struct {
		name    string
		exp     int
		mant    int
		wantMax float64
	}{
		{"F32", 8, 23, 3.4028234663852886e+38},
		{"F16", 5, 10, 65504},
		{"BF16", 8, 7, 3.3895313892515355e+38},
		{"F64", 11, 52, 1.7976931348623157e+308},
		// E5M2 是标准 IEEE（指数全 1 留给 Inf），公式适用：1.11b × 2^15
		{"F8_E5M2", 5, 2, 57344},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := maxFinite(tt.exp, tt.mant); got != tt.wantMax {
				t.Errorf("maxFinite(%d, %d) = %v, want %v", tt.exp, tt.mant, got, tt.wantMax)
			}
			// 相对精度就是 2^-mantissa，这里只确认它不是 0 也不是 1
			if eps := math.Ldexp(1, -tt.mant); eps <= 0 || eps >= 1 {
				t.Errorf("2^-%d = %v，不在 (0,1) 内", tt.mant, eps)
			}
		})
	}
}

// 表里每个条目的字段必须与**独立常量**一致。
//
// 原先这里比的是 `e.Field("最大值")` 与 `humanFloat(maxFinite(...))`，
// 而那个 Field 本身就是用同一个表达式生成的 —— 两边同一个式子，
// 永远不可能红。实测它漏掉了 F8_E4M3 的「最大值 240」（正确值 448）。
//
// 现在期望值写死在 wantMax 里，来源标在注释上：F8 两条来自
// torch.finfo 与按位枚举（见 TestFloatFormats_F8的最大值），
// 其余四条是教科书值。
func TestFloatFormats_表与算法一致(t *testing.T) {
	wantMax := map[string]float64{
		"float:F64":     1.7976931348623157e+308,
		"float:F32":     3.4028234663852886e+38,
		"float:BF16":    3.3895313892515355e+38,
		"float:F16":     65504,
		"float:F8_E5M2": 57344,
		"float:F8_E4M3": 448, // FN 变体，没有 Inf
	}
	checked := 0
	for _, e := range floatsTable().Entries {
		f, ok := floatFormatByID[e.ID]
		if !ok {
			continue // safetensors dtype 条目，没有位布局定义
		}
		checked++
		want, ok := wantMax[e.ID]
		if !ok {
			t.Errorf("%s 没有独立期望值 —— 这张表就没被真正校验过", e.ID)
			continue
		}
		if got := e.Field("最大值"); got != humanFloat(want) {
			t.Errorf("%s 的最大值写的是 %s，独立期望值是 %s",
				e.ID, got, humanFloat(want))
		}
		if got, want2 := maxOf(f), want; got != want2 {
			t.Errorf("%s 的 maxOf = %v，独立期望值 %v", e.ID, got, want2)
		}
		if got, want := e.Field("相对精度"), humanFloat(math.Ldexp(1, -f.mant)); got != want {
			t.Errorf("%s 的相对精度写的是 %s，算出来是 %s", e.ID, got, want)
		}
		if got, want := e.Field("最小正规数"), humanFloat(minNormal(f.exp)); got != want {
			t.Errorf("%s 的最小正规数写的是 %s，算出来是 %s", e.ID, got, want)
		}
	}
	// 精确值：漏一条的表现是"用户那里少一整条"，必须红
	if want := len(floatFormatByID); checked != want {
		t.Fatalf("只查了 %d 条，表里有 %d 条 —— 有条目没被校验", checked, want)
	}
	if got := len(wantMax); got != len(floatFormatByID) {
		t.Fatalf("独立期望值有 %d 条，布局表有 %d 条 —— 两边必须一一对应",
			got, len(floatFormatByID))
	}
}

// 浮点条目的 ID 必须是 "float:" + Dtype 字面量。
//
// 这条钉住的是一处**静默失效**：safetensors 小节靠 `floatFormatByID["float:"+st]`
// 反查位布局，键拼错一个字符就查不到 —— 而查不到的表现只是那一栏不见了
// （「对应」「最大值」消失、SeeAlso 为空），不会有任何东西报错。
// 实测：F8 系列原先写成 float:F8E4M3（少了 Dtype 里的下划线），
// 于是这两个最常见于量化方案的格式，用户恰恰看不到它们的指数/尾数构成。
func TestFloats_ID与Dtype一致(t *testing.T) {
	for id := range floatFormatByID {
		d := model.Dtype(strings.TrimPrefix(id, "float:"))
		if want := "float:" + string(d); id != want {
			t.Errorf("浮点条目的 ID 是 %q，按 Dtype 字面量应当是 %q", id, want)
		}
		if !d.IsFloat() {
			t.Errorf("%s 在位布局表里，但 Dtype 不认为它是浮点", id)
		}
	}
}

// safetensors 小节必须与解析器认的清单**逐项对齐**，字节数必须与 Dtype 表一致。
//
// 两份清单漂移的表现是速查表里少一行、或者字节数写错 —— 前者没有任何东西
// 会红，后者更糟：那是把假的数字当事实展示给用户。
func TestFloats_safetensors小节与Dtype一致(t *testing.T) {
	byID := map[string]Entry{}
	for _, e := range floatsTable().Entries {
		byID[e.ID] = e
	}
	checked := 0
	for _, d := range model.SafetensorsDtypes() {
		st := string(d)
		e, ok := byID["dtype:"+st]
		if !ok {
			t.Errorf("safetensors dtype %q 在速查表里没有条目", st)
			continue
		}
		checked++
		// 字节数不能写死在散文里：必须与 Dtype 表算出来的一致
		n, ok := d.ByteSize()
		if !ok {
			t.Errorf("%s 取不到字节数，速查表却给它列了字节数", st)
			continue
		}
		if got, want := e.Field("字节数"), strconv.FormatInt(n, 10); got != want {
			t.Errorf("%s 的字节数写的是 %q，Dtype 表算出来是 %q", st, got, want)
		}
	}
	if checked != len(model.SafetensorsDtypes()) {
		t.Fatalf("只对上 %d 项，清单有 %d 项", checked, len(model.SafetensorsDtypes()))
	}
	// 浮点类型必须挂上位布局的链接（这正是 ID 拼错时会消失的那一栏）
	for _, st := range []string{"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"} {
		if len(byID["dtype:"+st].SeeAlso) == 0 {
			t.Errorf("%s 没有关联到位布局条目 —— 用户看不到它的指数/尾数构成", st)
		}
	}
}

// 表格里的位布局定义必须是自洽的（位数与总宽吻合、别称不全为空）。
func TestFloatFormats_定义自洽(t *testing.T) {
	for id, f := range floatFormatByID {
		if f.sign != 1 {
			t.Errorf("%s 的符号位是 %d，应为 1", id, f.sign)
		}
		if f.exp <= 0 || f.mant <= 0 {
			t.Errorf("%s 的位布局不完整: exp=%d mant=%d", id, f.exp, f.mant)
		}
		if len(f.aliases) == 0 {
			t.Errorf("%s 没有别称 —— 用户看到的往往是别称而不是规范名", id)
		}
		if f.notes == "" {
			t.Errorf("%s 没有说明", id)
		}
	}
}

// F8_E4M3FN 的最大值必须是 448，不能是公式算出来的 240。
//
// 这条**独立于 maxFinite**：按位枚举这个格式的全部位型取最大值。
// 为什么需要单独一条：maxFinite 假定"指数域全 1 留给 Inf"（IEEE 的规矩），
// 而 E4M3FN 没有 Inf —— 只有 S.1111.111 是 NaN，指数 1111 对有限值可用。
// 按那个假定算出来是 240，只有真值的一半，而它会以「最大值 240」的
// 样子展示给用户，看起来像个算出来的事实。
//
// 交叉来源：torch.finfo(torch.float8_e4m3fn).max == 448（实测一致）。
func TestFloatFormats_F8的最大值(t *testing.T) {
	// 按位枚举：1 符号 + 4 指数 + 3 尾数，bias = 2^(4-1)-1 = 7
	const expBits, mantBits = 4, 3
	bias := (1 << (expBits - 1)) - 1
	max := 0.0
	// 正规数：指数 1..2^expBits-1（**不排掉全 1**，这个格式里它可用）
	for e := 1; e <= (1<<expBits)-1; e++ {
		for m := 0; m < 1<<mantBits; m++ {
			// 唯一的 NaN 位型：指数全 1 且尾数全 1
			if e == (1<<expBits)-1 && m == (1<<mantBits)-1 {
				continue
			}
			v := (1 + float64(m)/float64(int(1)<<mantBits)) * math.Ldexp(1, e-bias)
			if v > max {
				max = v
			}
		}
	}
	if max != 448 {
		t.Fatalf("按位枚举算出 %v，预期 448 —— 枚举本身写错了", max)
	}
	if got := maxOf(floatFormatByID["float:F8_E4M3"]); got != 448 {
		t.Errorf("F8_E4M3 的 maxOf = %v，应为 448（按位枚举与 torch.finfo 都是它）", got)
	}
	// E5M2 是标准 IEEE，公式适用 —— 这条是反面对照，
	// 防止有人为了修上面那条去改 maxFinite
	if got := maxFinite(5, 2); got != 57344 {
		t.Errorf("maxFinite(5,2) = %v，IEEE 的 E5M2 应为 57344", got)
	}
}
