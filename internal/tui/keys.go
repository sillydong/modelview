package tui

import (
	"strings"
	"unicode/utf8"
)

// 按键常量。**帮助栏与测试共用这一份** ——
// 帮助栏写死字符串的话，改了键位就会显示错的提示，
// 而用户照着按没反应。
const (
	keyUp        = "↑/k"
	keyDown      = "↓/j"
	keyEsc       = "Esc"
	keyEnter     = "Enter"
	keyFilter    = "/"
	keyBackspace = "Backspace"
	keyTab       = "Tab"
	keyRescan    = "r"
	keyQuit      = "q"
	// `?` 现在有处理分支了：根视图在**模态判定之后**接它（app.go），
	// 随处可开、带上下文模型。原先这段注释写的是反面 ——
	// "在接上那条分支之前，任何 Help() 都不许列它，列了就是骗用户按"；
	// 所以现在每个非模态的 Help() 都列上了它，**模态那一支不列**
	//（输入态下 `?` 是用户要打的一个字符，不是快捷键）。
	keyHelp = "?"

	// keyScanAll 是"扫描全部张量"。**只有模型视图有那条分支**，
	// 所以只有 ModelView 的 Help() 列它 —— 别的视图列了就是骗用户按。
	keyScanAll = "a"
)

// helpLine 是底部的按键提示。
//
// **只列当前视图真的支持的键**：列了不支持的等于骗用户按 ——
// 所以每个视图的 Help() 返回自己那一份，而不是从一张全局表里挑。
//
// 常量集随视图增长 —— 用到才加，提前定义会让那一版 lint 报 unused。
func helpLine(bindings []string) string {
	return styleHelp.Render(strings.Join(bindings, "  "))
}

// backspaceOne 删掉 s 的最后一个**字符**（rune），不是最后一个字节。
//
// 直接写 `s[:len(s)-1]` 会把多字节字符切成半个：输入「中」按一次退格
// 得到 "\xe4\xb8"，界面显示替换字符，而下游的 strings.ToLower /
// ref.Find 拿到的是被 U+FFFD 污染的串 —— 匹配静默失效，没有任何报错。
//
// 与 humanize.Truncate「切点必须回退到 rune 边界」是同一条约定。
// 原先四处（张量过滤的输入态与已确认态、速查表的输入态与已确认态）
// 各写了一遍 `s[:len(s)-1]`，所以合并成这一个入口。
func backspaceOne(s string) string {
	if s == "" {
		return ""
	}
	_, size := utf8.DecodeLastRuneInString(s)
	return s[:len(s)-size]
}
