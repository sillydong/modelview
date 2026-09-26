package pytorch

import (
	"strings"
	"testing"
)

// pb 用来手工拼 pickle 字节流。opcode 常量由 pickle.go 定义（同包），
// 这里不重复声明。
type pb struct{ buf []byte }

func (b *pb) raw(v ...byte) *pb { b.buf = append(b.buf, v...); return b }

func (b *pb) u8(v uint8) *pb { return b.raw(v) }

func (b *pb) u32le(v uint32) *pb {
	return b.raw(byte(v), byte(v>>8), byte(v>>16), byte(v>>24))
}

// str 写 BINUNICODE：'X' + u32 长度 + utf8
func (b *pb) str(s string) *pb {
	b.u8(opBINUNICODE)
	b.u32le(uint32(len(s)))
	b.raw([]byte(s)...)
	return b
}

// global 写 GLOBAL：'c' + 模块名 + '\n' + 类名 + '\n'
func (b *pb) global(module, name string) *pb {
	b.u8(opGLOBAL)
	b.raw([]byte(module)...)
	b.u8('\n')
	b.raw([]byte(name)...)
	b.u8('\n')
	return b
}

func (b *pb) input(n uint32) *pb { b.u8(opBINPUT); return b.u8(uint8(n)) }
func (b *pb) get(n uint32) *pb   { b.u8(opBINGET); return b.u8(uint8(n)) }

func (b *pb) binint(v int32) *pb { b.u8(opBININT); return b.u32le(uint32(v)) }

func (b *pb) done() []byte { return append(b.buf, opSTOP) }

func newPB() *pb { return &pb{buf: []byte{opPROTO, 2}} }

func topValue(t *testing.T, data []byte) any {
	t.Helper()
	vals, err := runPickle(data)
	if err != nil {
		t.Fatalf("runPickle 失败: %v", err)
	}
	if len(vals) == 0 {
		t.Fatal("结果栈为空")
	}
	return vals[len(vals)-1]
}

func TestRunPickle_标量(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		want any
	}{
		{"BININT1", newPB().u8(opBININT1).u8(7).done(), int64(7)},
		{"BININT2", newPB().u8(opBININT2).raw(0x01, 0x02).done(), int64(0x0201)},
		{"BININT 负数", newPB().binint(-12345).done(), int64(-12345)},
		{"NEWFALSE", newPB().u8(opNEWFALSE).done(), false},
		{"NEWTRUE", newPB().u8(opNEWTRUE).done(), true},
		{"NONE", newPB().u8(opNONE).done(), nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := topValue(t, tt.data)
			if got != tt.want {
				t.Errorf("got %#v (%T), want %#v (%T)", got, got, tt.want, tt.want)
			}
		})
	}
}

func TestRunPickle_字符串与memo(t *testing.T) {
	// push "hello"，存进 memo[0]；再 push 9；再 BINGET 0 取回 "hello"。
	// 结束时栈上是 ["hello", 9, "hello"] —— BINPUT 不压栈，只记录。
	data := newPB().str("hello").input(0).u8(opBININT1).u8(9).get(0).done()
	vals, err := runPickle(data)
	if err != nil {
		t.Fatalf("runPickle 失败: %v", err)
	}
	if len(vals) != 3 {
		t.Fatalf("栈上 %d 个值, want 3", len(vals))
	}
	if vals[0] != "hello" {
		t.Errorf("vals[0] = %v, want hello", vals[0])
	}
	if vals[1] != int64(9) {
		t.Errorf("vals[1] = %v, want 9", vals[1])
	}
	if vals[2] != "hello" {
		t.Errorf("BINGET 取回 %v, want hello（memo 未生效）", vals[2])
	}
}

func TestRunPickle_引用不存在的memo报错(t *testing.T) {
	if _, err := runPickle(newPB().get(99).done()); err == nil {
		t.Fatal("引用不存在的 memo 应报错")
	}
}

func TestRunPickle_元组构造(t *testing.T) {
	// TUPLE3: 1 2 3 → (1,2,3)
	data := newPB().u8(opBININT1).u8(1).u8(opBININT1).u8(2).u8(opBININT1).u8(3).
		u8(opTUPLE3).done()
	got, ok := topValue(t, data).([]any)
	if !ok {
		t.Fatalf("类型 = %T, want []any", topValue(t, data))
	}
	if len(got) != 3 || got[0] != int64(1) || got[2] != int64(3) {
		t.Errorf("TUPLE3 = %v", got)
	}
}

func TestRunPickle_MARK与TUPLE(t *testing.T) {
	data := newPB().u8(opMARK).
		u8(opBININT1).u8(1).u8(opBININT1).u8(2).u8(opBININT1).u8(3).
		u8(opTUPLE).done()
	got, ok := topValue(t, data).([]any)
	if !ok || len(got) != 3 {
		t.Errorf("MARK/TUPLE = %v, want 3 项", got)
	}
}

// 嵌套 MARK：内层先闭合，外层仍能正确取到自己的范围。
func TestRunPickle_嵌套MARK(t *testing.T) {
	// MARK MARK 1 2 TUPLE 3 TUPLE
	// 内层 (1,2)，外层 ((1,2), 3)
	data := newPB().u8(opMARK).u8(opMARK).
		u8(opBININT1).u8(1).u8(opBININT1).u8(2).u8(opTUPLE).
		u8(opBININT1).u8(3).
		u8(opTUPLE).done()
	outer, ok := topValue(t, data).([]any)
	if !ok || len(outer) != 2 {
		t.Fatalf("外层 = %v, want 2 项", outer)
	}
	inner, ok := outer[0].([]any)
	if !ok || len(inner) != 2 || inner[0] != int64(1) {
		t.Errorf("内层 = %v, want [1 2]", outer[0])
	}
	if outer[1] != int64(3) {
		t.Errorf("外层第二项 = %v, want 3", outer[1])
	}
}

func TestRunPickle_无匹配MARK报错(t *testing.T) {
	if _, err := runPickle(newPB().u8(opBININT1).u8(1).u8(opTUPLE).done()); err == nil {
		t.Fatal("无 MARK 的 TUPLE 应报错")
	}
}

func TestRunPickle_SETITEMS(t *testing.T) {
	data := newPB().u8(opEMPTY_DICT).u8(opMARK).
		str("a").u8(opBININT1).u8(1).
		str("b").u8(opBININT1).u8(2).
		u8(opSETITEMS).done()
	got, ok := topValue(t, data).(map[string]any)
	if !ok {
		t.Fatalf("类型 = %T, want map", topValue(t, data))
	}
	if got["a"] != int64(1) || got["b"] != int64(2) {
		t.Errorf("SETITEMS = %v", got)
	}
}

func TestRunPickle_SETITEM单条(t *testing.T) {
	data := newPB().u8(opEMPTY_DICT).str("k").u8(opBININT1).u8(5).u8(opSETITEM).done()
	got := topValue(t, data).(map[string]any)
	if got["k"] != int64(5) {
		t.Errorf("SETITEM = %v", got)
	}
}

func TestRunPickle_APPEND(t *testing.T) {
	// APPEND 把栈顶元素追加到它下方的列表：[] append 1 → [1]
	data := newPB().u8(opEMPTY_LIST).u8(opBININT1).u8(1).u8(opAPPEND).done()
	got, ok := topValue(t, data).([]any)
	if !ok || len(got) != 1 || got[0] != int64(1) {
		t.Errorf("APPEND = %v", got)
	}
}

// REDUCE 的弹栈顺序是**先弹参数、再弹可调用对象**。
// 写反了会拿到完全错误的结果 —— 这个顺序是实测确认的。
func TestRunPickle_REDUCE弹栈顺序(t *testing.T) {
	// GLOBAL mod fn  MARK 7 8 TUPLE  REDUCE
	data := newPB().
		global("mod", "fn").
		u8(opMARK).u8(opBININT1).u8(7).u8(opBININT1).u8(8).u8(opTUPLE).
		u8(opREDUCE).done()

	vals, err := runPickle(data)
	if err != nil {
		t.Fatalf("runPickle 失败: %v", err)
	}
	call, ok := vals[len(vals)-1].(reduceCall)
	if !ok {
		t.Fatalf("栈顶类型 = %T, want reduceCall", vals[len(vals)-1])
	}
	if call.Func != "mod.fn" {
		t.Errorf("Func = %q, want mod.fn", call.Func)
	}
	if len(call.Args) != 2 || call.Args[0] != int64(7) || call.Args[1] != int64(8) {
		t.Errorf("Args = %v, want [7 8]", call.Args)
	}
}

func TestRunPickle_GLOBAL名字(t *testing.T) {
	got := topValue(t, newPB().global("torch._utils", "_rebuild_tensor_v2").done())
	g, ok := got.(globalRef)
	if !ok {
		t.Fatalf("类型 = %T", got)
	}
	if g.Name != "torch._utils._rebuild_tensor_v2" {
		t.Errorf("Name = %q", g.Name)
	}
}

func TestRunPickle_BINPERSID(t *testing.T) {
	// MARK "storage" GLOBAL torch\nFloatStorage "0" "cpu" 5 TUPLE  BINPERSID
	data := newPB().u8(opMARK).
		str("storage").
		global("torch", "FloatStorage").
		str("0").str("cpu").u8(opBININT1).u8(5).
		u8(opTUPLE).
		u8(opBINPERSID).done()

	got := topValue(t, data)
	pid, ok := got.(persistentRef)
	if !ok {
		t.Fatalf("类型 = %T, want persistentRef", got)
	}
	tup, ok := pid.Value.([]any)
	if !ok || len(tup) != 5 {
		t.Fatalf("PersistentId = %v, want 5 元组", pid.Value)
	}
	if tup[0] != "storage" || tup[2] != "0" || tup[4] != int64(5) {
		t.Errorf("PersistentId = %v", tup)
	}
}

func TestRunPickle_BUILD弹掉参数(t *testing.T) {
	// EMPTY_DICT MARK SETITEMS  然后 BUILD
	data := newPB().u8(opEMPTY_DICT).u8(opMARK).
		str("k").u8(opBININT1).u8(1).u8(opSETITEMS).
		u8(opEMPTY_TUPLE).u8(opBUILD).done()
	got, ok := topValue(t, data).(map[string]any)
	if !ok {
		t.Fatalf("BUILD 后栈顶类型 = %T, want map", topValue(t, data))
	}
	if got["k"] != int64(1) {
		t.Errorf("字典内容 = %v", got)
	}
}

func TestRunPickle_未知opcode返回可识别错误(t *testing.T) {
	_, err := runPickle([]byte{opPROTO, 2, 0xFE})
	if err == nil {
		t.Fatal("未知 opcode 应报错")
	}
	var e ErrUnknownOpcode
	if !asErr(err, &e) {
		t.Fatalf("err = %v, 期望 ErrUnknownOpcode", err)
	}
	if e.Op != 0xFE {
		t.Errorf("Op = 0x%02x, want 0xfe", e.Op)
	}
}

func TestRunPickle_截断报错(t *testing.T) {
	// 声称字符串长度 100，实际只有 1 字节
	if _, err := runPickle([]byte{opPROTO, 2, opBINUNICODE, 100, 0, 0, 0, 'a'}); err == nil {
		t.Fatal("截断应报错")
	}
}

func TestRunPickle_超长字符串报错(t *testing.T) {
	// 声称 2 GiB 的字符串（BINUNICODE 的长度字段是 u32）
	data := []byte{opPROTO, 2, opBINUNICODE, 0xff, 0xff, 0xff, 0x7f}
	_, err := runPickle(data)
	if err == nil {
		t.Fatal("超长字符串应报错")
	}
	// 必须断言是**长度上限守卫**报的错：这个输入本来就只有 7 字节，
	// 就算去掉上限检查，"需要 N 字节但只剩 M" 那条也会报错，测试照样通过。
	if !strings.Contains(err.Error(), "上限") {
		t.Fatalf("错误应来自长度上限守卫，实际: %v", err)
	}
}

func TestRunPickle_空栈POP报错(t *testing.T) {
	if _, err := runPickle(newPB().u8(opPOP).done()); err == nil {
		t.Fatal("空栈 POP 应报错")
	}
}

func TestRunPickle_DUP(t *testing.T) {
	data := newPB().u8(opBININT1).u8(3).u8(opDUP).done()
	vals, err := runPickle(data)
	if err != nil {
		t.Fatalf("runPickle 失败: %v", err)
	}
	if len(vals) != 2 || vals[0] != int64(3) || vals[1] != int64(3) {
		t.Errorf("DUP = %v", vals)
	}
}

// asErr 是 errors.As 的薄包装，避免在本文件引入额外 import。
func asErr(err error, target *ErrUnknownOpcode) bool {
	e, ok := err.(ErrUnknownOpcode)
	if ok {
		*target = e
	}
	return ok
}

// SETITEMS 的元素个数必须成对，奇数个说明字节流损坏。
// 静默丢掉最后一项会让张量数凭空少一个。
func TestRunPickle_SETITEMS奇数元素报错(t *testing.T) {
	// EMPTY_DICT MARK "a" 1 "b" SETITEMS —— "b" 没有对应的值
	data := newPB().u8(opEMPTY_DICT).u8(opMARK).
		str("a").u8(opBININT1).u8(1).
		str("b").
		u8(opSETITEMS).done()
	_, err := runPickle(data)
	if err == nil {
		t.Fatal("奇数个 SETITEMS 元素应报错")
	}
	if !strings.Contains(err.Error(), "不成对") {
		t.Fatalf("错误应来自成对校验，实际: %v", err)
	}
}

// OrderedDict 直接带参数构造的形式（[(k,v), ...]）。
// PyTorch 实际走的是"空 OrderedDict + SETITEMS"，这条是标准 pickle 的
// 另一种写法，本地语料里没有，所以必须靠单元测试钉住。
func TestOrderedDictFrom_键值对形式(t *testing.T) {
	got := orderedDictFrom([]any{[]any{
		[]any{"a", int64(1)},
		[]any{"b", int64(2)},
	}})
	if len(got) != 2 || got["a"] != int64(1) || got["b"] != int64(2) {
		t.Errorf("orderedDictFrom = %v", got)
	}
	// 空参数（PyTorch 的常见形式）得到空字典
	if got := orderedDictFrom(nil); len(got) != 0 {
		t.Errorf("空参数应得空字典，实际 %v", got)
	}
}

// SHORT_BINUNICODE 与 STACK_GLOBAL 本地 22 个文件里一次都没出现，
// 但两者都实现出来了 —— 没有测试的实现等于零证据代码。
func TestRunPickle_SHORT_BINUNICODE(t *testing.T) {
	// newPB 的 str 用的是 BINUNICODE，这里手工拼一个 SHORT_BINUNICODE
	raw := []byte{opPROTO, 2, opSHORT_BINUNICODE, 3, 'a', 'b', 'c', opSTOP}
	stack, err := runPickle(raw)
	if err != nil {
		t.Fatalf("runPickle 失败: %v", err)
	}
	if len(stack) != 1 || stack[0] != "abc" {
		t.Errorf("stack = %v, want [abc]", stack)
	}
}

// STACK_GLOBAL 从栈上取模块名与类名拼出全局引用（协议 4 的写法）。
func TestRunPickle_STACK_GLOBAL(t *testing.T) {
	raw := []byte{opPROTO, 4,
		opSHORT_BINUNICODE, 5, 't', 'o', 'r', 'c', 'h',
		opSHORT_BINUNICODE, 12, 'F', 'l', 'o', 'a', 't', 'S', 't', 'o', 'r', 'a', 'g', 'e',
		opSTACK_GLOBAL, opSTOP}
	stack, err := runPickle(raw)
	if err != nil {
		t.Fatalf("runPickle 失败: %v", err)
	}
	if len(stack) != 1 {
		t.Fatalf("stack = %v", stack)
	}
	g, ok := stack[0].(globalRef)
	if !ok {
		t.Fatalf("栈顶不是 globalRef: %T", stack[0])
	}
	if g.Name != "torch.FloatStorage" {
		t.Errorf("Name = %q, want torch.FloatStorage", g.Name)
	}
}

// BINPUT 把栈顶存进 memo，栈空说明字节流是坏的。
// 少了这个判断会直接越界 panic —— 解析器遇到畸形文件必须返回错误，不能崩。
func TestRunPickle_BINPUT空栈报错(t *testing.T) {
	raw := []byte{opPROTO, 2, opBINPUT, 0, opSTOP}
	_, err := runPickle(raw)
	if err == nil {
		t.Fatal("BINPUT 空栈应报错")
	}
	if !strings.Contains(err.Error(), "栈为空") {
		t.Fatalf("错误应说明栈为空，实际: %v", err)
	}
}
