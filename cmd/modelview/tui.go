package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/parser"
	"github.com/sillydong/modelview/internal/tui"
)

// runTUI 起交互界面。
//
// **只在真终端里跑**：重定向到管道或文件时 bubbletea 拿不到尺寸，
// 画出来的东西是给看不到它的人准备的 —— 而更糟的是，
// 那些 ANSI 转义会混进日志或 jq 的输入里，报错的地方离原因很远。
// 非交互场景用 --json。
func runTUI() error {
	// **stdin 与 stdout 都要是真终端**：只查 stdout 的话，
	// `modelview < /dev/null` 会照常启动并渲染，然后永远收不到按键 ——
	// q / Esc 全无反应，用户只能 Ctrl+C（实测过）。
	// 只看 stdout 不够的原因：/dev/null 也是字符设备。
	if !isTerminal(os.Stdin) || !isTerminal(os.Stdout) {
		return fmt.Errorf("交互界面需要终端；非交互请用 modelview --json scan")
	}
	// 解析入口在这里注入：tui 包不 import internal/parser（spec §4.0），
	// 依赖从包级 import 变成运行时注入。
	p := tea.NewProgram(tui.New(tui.NewLibrary().WithParse(parser.Parse)), tea.WithAltScreen())
	_, err := p.Run()
	return err
}

// isTerminal 判断是不是真终端。
//
// 用字符设备位而不是引入 golang.org/x/term：
// 判断"是不是终端"这一件事不值得一个依赖。
//
// **已知边界**：/dev/null 也是字符设备，所以 `modelview > /dev/null`
// 判不出区别 —— 界面会写进黑洞，用户看到屏幕一空、按 q 才回来。
// 后果很小（不会污染任何东西），但这条边界是已知的，不是没想到。
func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	if err != nil {
		return false
	}
	return st.Mode()&os.ModeCharDevice != 0
}
