package tui

import "strings"

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
