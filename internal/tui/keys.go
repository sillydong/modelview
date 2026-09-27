package tui

import "strings"

// helpLine 是底部的按键提示。
//
// **只列当前视图真的支持的键**：列了不支持的等于骗用户按 ——
// 所以每个视图的 Help() 返回自己那一份，而不是从一张全局表里挑。
//
// 按键常量（keyUp / keyEnter / …）等到第一个用到它们的视图再加，
// 与样式同理：提前定义会让这一版 lint 报 unused。
func helpLine(bindings []string) string {
	return styleHelp.Render(strings.Join(bindings, "  "))
}
