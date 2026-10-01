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
	// **`?` 至今没有任何处理分支**：spec §8 的按键栏写着"`?` 速查表、
	// 随时可开"，而实现里速查表是从"速查表"栏目按 Enter 进的
	//（见 `ModelView` 的 sectionRef）。常量留着是给那一天用的，
	// **在那之前任何 Help() 都不许列它** —— 列了就是骗用户按，
	// 而这一条有两个测试盯着（ModelView / Library 的"帮助栏只列支持的键"）。
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
