// Package pytorch 解析 PyTorch 的 .pt 文件。
package pytorch

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// reduceCall 是 REDUCE 操作的解析结果：一个待调用的函数与其参数。
//
// 我们不需要真的调用它 —— 只需要认出 _rebuild_tensor_v2 并取它的参数。
type reduceCall struct {
	Func string
	Args []any
}

// globalRef 是被 GLOBAL / STACK_GLOBAL 引用的全局对象（模块.类名）。
type globalRef struct{ Name string }

// persistentRef 是 BINPERSID 解析出的持久化对象引用。
// Value 是那个 5 元组 ('storage', 存储类, 键, 位置, 元素数)。
type persistentRef struct{ Value any }

// mark 是栈上的标记对象，用于 TUPLE / SETITEMS 的多元素弹出。
type mark struct{}

// pickle 协议 2 的 opcode。
//
// 覆盖 PyTorch 实际用到的 opcode，另含几个标准 pickle 里常见、真实文件暂未出现
// 但明确实现出来的（SHORT_BINUNICODE / STACK_GLOBAL / POP / DUP）。
//
// 实测 22 个真实 .pt 的 data.pkl 共出现 30 种 opcode，全部在表内：
//
//	APPEND APPENDS BINFLOAT BINGET BININT BININT1 BININT2 BINPERSID BINPUT
//	BINUNICODE BUILD EMPTY_DICT EMPTY_LIST EMPTY_TUPLE GLOBAL LONG_BINGET
//	LONG_BINPUT MARK NEWFALSE NEWTRUE NONE PROTO REDUCE SETITEM SETITEMS
//	STOP TUPLE TUPLE1 TUPLE2 TUPLE3
//
// 遇到其它 opcode 会返回 ErrUnknownOpcode。
const (
	opPROTO            = 0x80
	opSTOP             = '.'
	opMARK             = '('
	opEMPTY_TUPLE      = ')'
	opEMPTY_DICT       = '}'
	opEMPTY_LIST       = ']'
	opBININT           = 'J'
	opBININT1          = 'K'
	opBININT2          = 'M'
	opBINFLOAT         = 'G'
	opBINUNICODE       = 'X'
	opSHORT_BINUNICODE = 0x8c
	opBINPUT           = 'q'
	opBINGET           = 'h'
	opLONG_BINPUT      = 'r'
	opLONG_BINGET      = 'j'
	opTUPLE            = 't'
	opTUPLE1           = 0x85
	opTUPLE2           = 0x86
	opTUPLE3           = 0x87
	opSETITEM          = 's'
	opSETITEMS         = 'u'
	opAPPEND           = 'a'
	opAPPENDS          = 'e'
	opGLOBAL           = 'c'
	opSTACK_GLOBAL     = 0x93
	opREDUCE           = 'R'
	opBUILD            = 'b'
	opNEWTRUE          = 0x88
	opNEWFALSE         = 0x89
	opBINPERSID        = 'Q'
	opNONE             = 'N'
	opPOP              = '0'
	opDUP              = '2'
)

// orderedDictFunc 是 PyTorch 用作 state_dict 容器的类型。
const orderedDictFunc = "collections.OrderedDict"

// orderedDictFrom 把 OrderedDict(...) 的参数转成字典。
//
// 参数形如 [("k1", v1), ("k2", v2)] 时逐对并入；空参数得到空字典。
func orderedDictFrom(args []any) map[string]any {
	m := map[string]any{}
	// 参数可能是一层列表包着若干对，也可能直接是若干对；
	// 递归展开，遇到 (键, 值) 形状的二元组就并入。
	var merge func(v any)
	merge = func(v any) {
		pair, ok := v.([]any)
		if !ok {
			return
		}
		if len(pair) == 2 {
			if _, isKey := pair[0].(string); isKey {
				m[keyString(pair[0])] = pair[1]
				return
			}
		}
		for _, e := range pair {
			merge(e)
		}
	}
	for _, a := range args {
		merge(a)
	}
	return m
}

// maxStringLen 是 pickle 里单个字符串的长度上限。
const maxStringLen = 64 << 20

// maxStackDepth 是操作数栈的深度上限，防止损坏数据导致无限增长。
const maxStackDepth = 1 << 20

// ErrUnknownOpcode 表示遇到了本实现未覆盖的 pickle 操作码。
//
// PyTorch 只使用 pickle 的一个受限子集；遇到其它 opcode 说明
// 文件不是 PyTorch 保存的，或者用了我们没见过的版本。
type ErrUnknownOpcode struct{ Op byte }

func (e ErrUnknownOpcode) Error() string {
	return fmt.Sprintf("未支持的 pickle 操作码 0x%02x (%q)", e.Op, string(rune(e.Op)))
}

// runPickle 执行一段 pickle 字节码，返回结束时的操作数栈。
//
// 这是一个**受限的**解释器：不构造真正的 Python 对象，
// 只把栈上的值组织成 Go 的 any
// （string / int64 / float64 / bool / nil / []any / map[string]any /
// reduceCall / globalRef / persistentRef）。
func runPickle(data []byte) ([]any, error) {
	var (
		stack []any
		memo  = map[uint64]any{}
		pos   int
	)

	push := func(v any) error {
		if len(stack) >= maxStackDepth {
			return fmt.Errorf("操作数栈超过上限 %d", maxStackDepth)
		}
		stack = append(stack, v)
		return nil
	}
	pop := func() (any, error) {
		if len(stack) == 0 {
			return nil, fmt.Errorf("偏移 %d 处栈为空，无法弹出", pos)
		}
		v := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		return v, nil
	}
	need := func(n int) error {
		if pos+n > len(data) {
			return fmt.Errorf("偏移 %d 处需要 %d 字节，只剩 %d", pos, n, len(data)-pos)
		}
		return nil
	}

	for pos < len(data) {
		op := data[pos]
		pos++

		switch op {
		case opPROTO:
			if err := need(1); err != nil {
				return nil, err
			}
			pos++ // 协议版本，不校验

		case opSTOP:
			return stack, nil

		case opMARK:
			if err := push(mark{}); err != nil {
				return nil, err
			}

		case opNONE:
			if err := push(nil); err != nil {
				return nil, err
			}

		case opNEWTRUE:
			if err := push(true); err != nil {
				return nil, err
			}
		case opNEWFALSE:
			if err := push(false); err != nil {
				return nil, err
			}

		case opBININT1:
			if err := need(1); err != nil {
				return nil, err
			}
			if err := push(int64(data[pos])); err != nil {
				return nil, err
			}
			pos++

		case opBININT2:
			if err := need(2); err != nil {
				return nil, err
			}
			if err := push(int64(binary.LittleEndian.Uint16(data[pos:]))); err != nil {
				return nil, err
			}
			pos += 2

		case opBININT:
			if err := need(4); err != nil {
				return nil, err
			}
			v := int64(int32(binary.LittleEndian.Uint32(data[pos:])))
			if err := push(v); err != nil {
				return nil, err
			}
			pos += 4

		case opBINFLOAT:
			if err := need(8); err != nil {
				return nil, err
			}
			bits := binary.BigEndian.Uint64(data[pos:]) // BINFLOAT 是大端
			if err := push(math.Float64frombits(bits)); err != nil {
				return nil, err
			}
			pos += 8

		case opBINUNICODE, opSHORT_BINUNICODE:
			var n uint64
			if op == opSHORT_BINUNICODE {
				if err := need(1); err != nil {
					return nil, err
				}
				n = uint64(data[pos])
				pos++
			} else {
				if err := need(4); err != nil {
					return nil, err
				}
				n = uint64(binary.LittleEndian.Uint32(data[pos:]))
				pos += 4
			}
			if n > maxStringLen {
				return nil, fmt.Errorf("偏移 %d 处字符串长度 %d 超出上限 %d", pos, n, maxStringLen)
			}
			if err := need(int(n)); err != nil {
				return nil, err
			}
			if err := push(string(data[pos : pos+int(n)])); err != nil {
				return nil, err
			}
			pos += int(n)

		case opEMPTY_TUPLE:
			if err := push([]any{}); err != nil {
				return nil, err
			}
		case opEMPTY_DICT:
			if err := push(map[string]any{}); err != nil {
				return nil, err
			}
		case opEMPTY_LIST:
			if err := push([]any{}); err != nil {
				return nil, err
			}

		case opTUPLE1, opTUPLE2, opTUPLE3:
			n := int(op-opTUPLE1) + 1
			if len(stack) < n {
				return nil, fmt.Errorf("偏移 %d 处栈不足 %d 项", pos, n)
			}
			items := append([]any(nil), stack[len(stack)-n:]...)
			stack = stack[:len(stack)-n]
			if err := push(items); err != nil {
				return nil, err
			}

		case opTUPLE:
			items, err := popMarked(&stack, pos)
			if err != nil {
				return nil, err
			}
			if err := push(items); err != nil {
				return nil, err
			}

		case opAPPEND, opAPPENDS:
			if err := doAppend(&stack, op == opAPPENDS, pos); err != nil {
				return nil, fmt.Errorf("偏移 %d: %w", pos, err)
			}

		case opSETITEM, opSETITEMS:
			if err := doSetItems(&stack, op == opSETITEMS, pos); err != nil {
				return nil, fmt.Errorf("偏移 %d: %w", pos, err)
			}

		case opBINPUT, opLONG_BINPUT:
			n, err := readMemoIndex(data, &pos, op == opLONG_BINPUT)
			if err != nil {
				return nil, err
			}
			// BINPUT 把栈顶存进 memo，栈空说明字节流是坏的。
			// 这里必须报错而不是跳过：跳过等于静默接受畸形输入，
			// 而少了这个判断会让下面一行直接越界 panic。
			if len(stack) == 0 {
				return nil, fmt.Errorf("偏移 %d 处 BINPUT 栈为空，无法记忆栈顶", pos)
			}
			memo[n] = stack[len(stack)-1]

		case opBINGET, opLONG_BINGET:
			n, err := readMemoIndex(data, &pos, op == opLONG_BINGET)
			if err != nil {
				return nil, err
			}
			v, ok := memo[n]
			if !ok {
				return nil, fmt.Errorf("偏移 %d 处引用不存在的 memo[%d]", pos, n)
			}
			if err := push(v); err != nil {
				return nil, err
			}

		case opGLOBAL:
			// 模块名 \n 类名 \n
			i := indexByte(data, pos, '\n')
			if i < 0 {
				return nil, fmt.Errorf("偏移 %d 处 GLOBAL 缺少模块名终止符", pos)
			}
			j := indexByte(data, i+1, '\n')
			if j < 0 {
				return nil, fmt.Errorf("偏移 %d 处 GLOBAL 缺少类名终止符", pos)
			}
			name := string(data[pos:i]) + "." + string(data[i+1:j])
			pos = j + 1
			if err := push(globalRef{Name: name}); err != nil {
				return nil, err
			}

		case opSTACK_GLOBAL:
			a, err := pop()
			if err != nil {
				return nil, err
			}
			b, err := pop()
			if err != nil {
				return nil, err
			}
			if err := push(globalRef{Name: fmt.Sprintf("%v.%v", b, a)}); err != nil {
				return nil, err
			}

		case opBUILD:
			// BUILD 只是给对象设置状态，弹掉参数即可
			if _, err := pop(); err != nil {
				return nil, err
			}

		case opREDUCE:
			// REDUCE 的栈序：可调用对象先压栈，参数后压栈。
			// 所以**先弹参数，再弹可调用对象** ——
			// 顺序反了会拿到完全错误的结果（实测确认）。
			argsVal, err := pop()
			if err != nil {
				return nil, err
			}
			fnVal, err := pop()
			if err != nil {
				return nil, err
			}
			call := reduceCall{Func: fmt.Sprint(fnVal)}
			if g, ok := fnVal.(globalRef); ok {
				call.Func = g.Name
			}
			if items, ok := argsVal.([]any); ok {
				call.Args = items
			}

			// collections.OrderedDict(...) 的结果必须当成**字典**，不是调用记录。
			//
			// PyTorch 用它做 state_dict 的容器：先 REDUCE 出一个空 OrderedDict，
			// 紧接着 MARK + 成百条 (键, 值) + SETITEMS 往里填。
			// 若产出的是 reduceCall，SETITEMS 就会因为"目标不是字典"而失败 ——
			// 这是实测 model.pt 时暴露的根因：state_dict 容器先 REDUCE 成空 OrderedDict，
			// 随后一个 SETITEMS 要填入整份权重（实测最大 252 项，即 126 对键值）。
			if call.Func == orderedDictFunc {
				if err := push(orderedDictFrom(call.Args)); err != nil {
					return nil, err
				}
				continue
			}

			if err := push(call); err != nil {
				return nil, err
			}

		case opBINPERSID:
			v, err := pop()
			if err != nil {
				return nil, err
			}
			if err := push(persistentRef{Value: v}); err != nil {
				return nil, err
			}

		case opPOP:
			if _, err := pop(); err != nil {
				return nil, err
			}

		case opDUP:
			if len(stack) == 0 {
				return nil, fmt.Errorf("偏移 %d 处 DUP 栈为空", pos)
			}
			if err := push(stack[len(stack)-1]); err != nil {
				return nil, err
			}

		default:
			return nil, ErrUnknownOpcode{Op: op}
		}
	}

	// 有些写入器省略 STOP。不视为错误，返回当前栈。
	return stack, nil
}

// popMarked 弹出最近一个 MARK 之上的所有元素（不含 MARK 本身）。
func popMarked(stack *[]any, pos int) ([]any, error) {
	s := *stack
	for i := len(s) - 1; i >= 0; i-- {
		if _, ok := s[i].(mark); ok {
			items := append([]any(nil), s[i+1:]...)
			*stack = s[:i]
			return items, nil
		}
	}
	return nil, fmt.Errorf("偏移 %d 处没有匹配的 MARK", pos)
}

func readMemoIndex(data []byte, pos *int, isLong bool) (uint64, error) {
	if isLong {
		if *pos+4 > len(data) {
			return 0, fmt.Errorf("偏移 %d 处 memo 索引越界", *pos)
		}
		n := uint64(binary.LittleEndian.Uint32(data[*pos:]))
		*pos += 4
		return n, nil
	}
	if *pos >= len(data) {
		return 0, fmt.Errorf("偏移 %d 处 memo 索引越界", *pos)
	}
	n := uint64(data[*pos])
	*pos++
	return n, nil
}

func indexByte(data []byte, from int, b byte) int {
	for i := from; i < len(data); i++ {
		if data[i] == b {
			return i
		}
	}
	return -1
}

// doAppend 实现 APPEND 与 APPENDS。**两者语义不同**：
//
//	APPEND  —— 弹一个元素，追加到其下方的列表
//	APPENDS —— 弹 MARK 之上的**全部**元素，批量追加到 MARK 下方的列表
//
// 把两者当成一回事会在真实文件上立刻失败：
// 实测 transformer-s-10m.pt 的偏移 175 处就是一个带 5 个整数的 APPENDS。
func doAppend(stack *[]any, many bool, pos int) error {
	s := *stack

	var items []any
	if many {
		var err error
		items, err = popMarked(&s, pos)
		if err != nil {
			return fmt.Errorf("APPENDS: %w", err)
		}
	} else {
		if len(s) < 2 {
			return fmt.Errorf("APPEND 栈不足 2 项")
		}
		items = []any{s[len(s)-1]}
		s = s[:len(s)-1]
	}

	if len(s) < 1 {
		return fmt.Errorf("APPEND 缺少目标列表")
	}
	lst, ok := s[len(s)-1].([]any)
	if !ok {
		return fmt.Errorf("APPEND 的目标不是列表（%T）", s[len(s)-1])
	}
	s[len(s)-1] = append(lst, items...)
	*stack = s
	return nil
}

func doSetItems(stack *[]any, many bool, pos int) error {
	s := *stack
	var items []any
	if many {
		var err error
		items, err = popMarked(&s, pos)
		if err != nil {
			return err
		}
	} else {
		if len(s) < 2 {
			return fmt.Errorf("SETITEM 栈不足 2 项")
		}
		items = []any{s[len(s)-2], s[len(s)-1]}
		s = s[:len(s)-2]
	}
	if len(s) < 1 {
		return fmt.Errorf("SETITEMS 缺少目标字典")
	}
	d, ok := s[len(s)-1].(map[string]any)
	if !ok {
		return fmt.Errorf("SETITEMS 的目标不是字典（%T）", s[len(s)-1])
	}
	// 键值必须成对。奇数个元素说明字节流损坏 ——
	// 静默丢掉最后一项会让张量数凭空少一个，且没有任何提示。
	// Python 的 pickle 在这里抛 ValueError，行为一致。
	if len(items)%2 != 0 {
		return fmt.Errorf("SETITEMS 有 %d 个元素，键值不成对", len(items))
	}
	for i := 0; i+1 < len(items); i += 2 {
		d[keyString(items[i])] = items[i+1]
	}
	*stack = s
	return nil
}

// keyString 把字典键转成字符串。PyTorch 的张量名字都是字符串。
func keyString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return strings.TrimSpace(fmt.Sprint(v))
}
