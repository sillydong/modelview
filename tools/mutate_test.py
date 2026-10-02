#!/usr/bin/env python3
"""tools/mutate.py 的判定逻辑测试。

这个工具以前**零测试**，结果把"编译失败"判成了"被抓住" ——
本项目的变异验证体系全建在它上面，它一失效所有绿灯都失去意义。

跑法：
    python3 tools/mutate_test.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mutate import BUILD_FAILED, CAUGHT, ESCAPED, NO_TESTS, classify  # noqa: E402

# 真实捕获的 go test 输出（跑出来的，不是编的）
OUT_BUILD_FAILED = """# github.com/sillydong/modelview/internal/analyze [github.com/sillydong/modelview/internal/analyze.test]
internal/analyze/stats.go:135:2: declared and not used: step
FAIL	github.com/sillydong/modelview/internal/analyze [build failed]
FAIL
"""
OUT_NO_TESTS = """ok  	github.com/sillydong/modelview/internal/analyze	0.935s [no tests to run]
"""
OUT_FAIL = """--- FAIL: TestSample_等距且可复现 (0.00s)
    stats_test.go:184: 采样结果不可复现: [1] 5 vs 7
FAIL
FAIL	github.com/sillydong/modelview/internal/analyze	0.559s
FAIL
"""
OUT_PANIC = """--- FAIL: TestRunPickle_BINPUT空栈报错 (0.00s)
panic: runtime error: index out of range [0] with length 0 [recovered]
FAIL
"""
OUT_OK = """ok  	github.com/sillydong/modelview/internal/analyze	0.559s
"""


class TestClassify(unittest.TestCase):
    def test_编译失败不能算被抓住(self):
        # 退出码非 0，但一条测试都没跑 —— 这是本项目踩过的坑
        self.assertEqual(classify(1, OUT_BUILD_FAILED), BUILD_FAILED)

    def test_没有测试跑到要单独报(self):
        # "no tests to run" 的退出码也是 0，但语义完全不同
        self.assertEqual(classify(0, OUT_NO_TESTS), NO_TESTS)

    def test_断言失败算被抓住(self):
        self.assertEqual(classify(1, OUT_FAIL), CAUGHT)

    def test_panic_算被抓住(self):
        self.assertEqual(classify(2, OUT_PANIC), CAUGHT)

    def test_全绿算漏网(self):
        self.assertEqual(classify(0, OUT_OK), ESCAPED)

    def test_判不出来时算漏网而不是通过(self):
        # 非 0 但没有 FAIL 标记：宁可说"漏网"也不能说"抓住了"
        self.assertEqual(classify(1, "some unexpected output"), ESCAPED)

    def test_编译失败优先于断言失败(self):
        # 同时出现时以编译失败为准 —— 那种情况下断言根本不该被相信
        mixed = OUT_FAIL + OUT_BUILD_FAILED
        self.assertEqual(classify(1, mixed), BUILD_FAILED)


class Test工具自身行为(unittest.TestCase):
    """跑真实的 mutate.py，确认三种结果各自返回正确的退出码。"""

    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(HERE)

    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, os.path.join(self.HERE, "mutate.py"), *args],
            cwd=self.ROOT, capture_output=True, text=True)

    def test_锚点不存在直接拒绝(self):
        r = self.run_tool("--file", "internal/model/dtype.go",
                          "--old", "这段代码根本不存在", "--new", "x",
                          "--test", "./internal/model/", "--label", "不存在")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("出现 0 次", r.stdout)

    def test_锚点重复直接拒绝(self):
        r = self.run_tool("--file", "internal/model/dtype.go",
                          "--old", "BlockSize: 1", "--new", "BlockSize: 2",
                          "--test", "./internal/model/", "--label", "重复")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("必须恰好 1 次", r.stdout)

    def test_run_匹配不到测试时拒绝(self):
        # 这是修复前会报"✓ 被抓住"的场景之一
        r = self.run_tool("--file", "internal/model/dtype.go",
                          "--old", "BlockBytes: 144", "--new", "BlockBytes: 143",
                          "--test", "./internal/model/",
                          "--run", "TestThisDoesNotExist", "--label", "没测试")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("没匹配到任何测试", r.stdout)

    def test_超时的退出码与被抓住不同(self):
        # 超时只说明"执行到了"，不说明断言能区分对错 —— 按退出码判读的
        # 调用方必须能把它与"被抓住"分开（修复前两者都是 0）。
        # `select {}` 是必然卡死且不需要任何 import 的变异。
        r = self.run_tool("--file", "internal/model/blockbytes_test.go",
                          "--old", "func TestBlockElems(t *testing.T) {",
                          "--new", "func TestBlockElems(t *testing.T) {\n\tselect {}",
                          "--test", "./internal/model/",
                          "--run", "TestBlockElems", "--timeout", "5",
                          "--label", "超时")
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("超时", r.stdout)
        # 卡死的那次也要还原（否则整个仓库留在变异后的状态里）
        with open(os.path.join(self.ROOT, "internal/model/blockbytes_test.go"),
                  encoding="utf-8") as f:
            self.assertNotIn("select {}", f.read())

    def test_编译失败的变异被识别为无效(self):
        # 修复前这条会输出 "✓ 被抓住"
        r = self.run_tool("--file", "internal/analyze/stats.go",
                          "--old", "out[i] = int64(float64(i) * step)",
                          "--new", "out[i] = 0",
                          "--test", "./internal/analyze/",
                          "--run", "TestPickIndices", "--label", "破坏编译")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("编译不过", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
