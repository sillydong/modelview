package main

import (
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/sillydong/modelview/internal/tui"
)

// runTUI 起交互界面。
//
// **只在真终端里跑**：重定向到管道或文件时 bubbletea 拿不到尺寸，
// 画出来的东西是给看不到它的人准备的 —— 而更糟的是，
// 那些 ANSI 转义会混进日志或 jq 的输入里，报错的地方离原因很远。
// 非交互场景用 --json。
func runTUI() error {
	if !isTerminal(os.Stdout) {
		return fmt.Errorf("交互界面需要终端；非交互请用 modelview --json scan")
	}
	p := tea.NewProgram(tui.New(tui.NewLibrary(), 0), tea.WithAltScreen())
	_, err := p.Run()
	return err
}

// isTerminal 判断是不是真终端。
//
// 用字符设备位而不是引入 golang.org/x/term：
// 判断"是不是终端"这一件事不值得一个依赖。
func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	if err != nil {
		return false
	}
	return st.Mode()&os.ModeCharDevice != 0
}
