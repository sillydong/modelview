// Package testutil 是跨包共用的测试辅助。
package testutil

import "testing"

// SetHome 把"当前用户目录"指到 dir，**跨平台**。
//
// 只设 HOME 在 Windows 上不起作用：那边 `os.UserHomeDir()` 读的是
// USERPROFILE。实测：CI 的 windows runner 上，只设 HOME 的测试红了 ——
// 被测代码去找真实的家目录，临时造的那棵目录树根本没被看见。
//
// 两处都设，调用方就不用关心自己在哪个平台（USERPROFILE 在非 Windows
// 上没有副作用）。
func SetHome(t *testing.T, dir string) {
	t.Helper()
	t.Setenv("HOME", dir)
	t.Setenv("USERPROFILE", dir)
}
