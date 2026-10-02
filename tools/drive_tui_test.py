#!/usr/bin/env python3
"""tools/drive_tui.py 的屏幕模型测试。

这个脚本是"真终端里的行为"唯一的观察窗口（`internal/tui` 的测试直接调
`view.Update`，走不到 bubbletea 的运行时）。所以它自己失真时，代价是双向的：
要么去追不存在的 bug，要么把真的截断看成正常。

实测踩过前一种：一次 `os.read` 把 `\\x1b[3;38;5;244m` 切成两半，剩下那半
被当成可见字符画进屏幕，那一行因此右移 16 列、结尾被顶出屏幕 ——
**看起来就像应用截了一行**，而原始字节里根本没有那串字面量。

跑法：
    python3 tools/drive_tui_test.py
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive_tui import Screen, incomplete_escape  # noqa: E402


class TestScreen边界(unittest.TestCase):
    def test_第80列的字符不丢也不挪(self):
        """**写满一行不换行**：落在最后一列的字符留在那一行上。

        终端在写到最后一列时会置一个"待换行"标志，下一个字符才换行 ——
        仿真器对这一格的处理各不相同（丢了 / 挪到下一行）。真终端上
        "文件声明那行少了个 `]`"一度被怀疑是这里，实测不是。
        """
        s = Screen(80, 24)
        s.feed(b"A" * 79 + b"]" + b"\r\n" + b"NEXT")
        rows = s.text().split("\n")
        self.assertEqual(rows[0], "A" * 79 + "]")
        self.assertEqual(rows[1], "NEXT")

    def test_转义序列被切成两半时不落到屏幕上(self):
        """一次 read 的边界切在转义序列中间时，那半截不能被当成可见字符。

        切成两半喂进去的屏幕必须与整段喂进去的**逐字符相同**。
        """
        whole = "\x1b[1m标题\x1b[0m\r\n".encode("utf-8")
        for cut in (2, 4, 5):  # 分别切在 "\x1b[" / "m" 之后 / 汉字中间
            with self.subTest(cut=cut):
                one = Screen(80, 24)
                one.feed(whole)
                two = Screen(80, 24)
                two.feed(whole[:cut])
                two.feed(whole[cut:])
                self.assertEqual(two.text(), one.text())
                self.assertEqual(two.text().split("\n")[0], "标题")
                self.assertNotIn("[1m", two.text())

    def test_多字节字符被切成两半时不变成替换字符(self):
        """UTF-8 的三字节被切开时，增量解码器要把它们拼回来。

        `decode("utf-8", "replace")` 会切成两个 U+FFFD —— 屏幕上多两个
        占位符，而且每个都占一列，后面的内容整体右移。
        """
        whole = "标题".encode("utf-8")
        s = Screen(80, 24)
        s.feed(whole[:1])
        s.feed(whole[1:])
        self.assertEqual(s.text().split("\n")[0], "标题")

    def test_碎片判定(self):
        self.assertIsNone(incomplete_escape("abc"))
        self.assertIsNone(incomplete_escape("\x1b[0m"))
        self.assertIsNone(incomplete_escape("\x1b[0m abc"))
        self.assertEqual(incomplete_escape("\x1b["), 0)
        self.assertEqual(incomplete_escape("ab\x1b[3;38;5;"), 2)
        self.assertEqual(incomplete_escape("ab\x1b]0;title"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
