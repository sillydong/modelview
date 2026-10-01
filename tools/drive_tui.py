#!/usr/bin/env python3
"""在真 pty 里驱动 TUI，按"按键 → 期望屏幕内容"逐步断言。

    python3 tools/drive_tui.py --cmd '/tmp/mv' \\
        --send 'jjjj' --expect '▸ qwen2.5:3b' \\
        --send '\\r'   --expect '434 张量' \\
        --send '/'    --expect '过滤：▏'

## 为什么需要它（单元测试抓不到的那类）

`internal/tui` 的测试全都直接调 `view.Update`，绕过了根 Model 与
bubbletea 的运行时。实测被这条路径漏掉过的 bug：`Library.Init` 的命令
从没被执行过（屏幕永远停在"正在扫描…"而测试全绿）、过滤态敲 q
直接退出程序。**这两条都只有真终端能验。**

## 三个坑（不填的话要么一帧都画不出来，要么断言假红）

1. **termenv 启动时会问终端两件事并等回答**：`OSC 11`（背景色，
   `\\x1b]11;?\\x1b\\\\`）与 `CSI 6n`（光标位置，`\\x1b[6n`）。
   真终端会自动回，pty 不是终端、不会 —— 不替它回，程序就永远卡在
   启动那一帧（termenv 等约 1 秒后会放弃，于是表现为"偶发慢"）。
2. **pty 的 winsize 默认是 0×0**：不设尺寸的话内容区只有 1 行，
   屏幕上看不到任何列表，验了等于没验。
3. **bubbletea 只重写"变化的行"，不重画整屏**。所以
   "把输出末尾 N 行拼起来当屏幕"是错的：光标移动序列被剥掉之后，
   新旧行的文字会混在一起。实测踩过：列表里按一下 ↓（只重写两行），
   拼出来的"屏幕"里新旧选中项同时存在，断言直接假红。
   这个脚本因此带一个**最小 vt100 子集**（见 Screen），真正维护一屏
   字符网格；断言看的是那一屏。
4. **一次 write 多个字符 = 粘贴，不是多次按键**。bubbletea 把
   一次读到的多个可打印字符合成**一个** `KeyRunes{Paste: true}`，
   它的 `String()` 是 `"[jjjj]"` —— 列表导航的 `case "j"` 匹配不上，
   于是"按了 4 下 j 却没动"（实测：单写 `jj`、`jjjj`、`jkjj` 全部无反应，
   分 4 次写就正常）。**输入框相反**：粘贴的整段会被一次收下
   （过滤框里写 `att` → 过滤词直接是 `att`，这是对的行为）。
   所以 `--send` 会把按键**逐个写**（`\x1b[A` 这类序列算一次），
   这样 `--send 'jjjj'` 才是"按 4 下 j"。

## 屏幕模型的两条边界（判读屏幕时要知道）

1. **驱动不替应用补任何东西**：屏幕上没有的字，就是 pty 收到的字节里没有。
   实测过一次误判 —— "文件声明那行收尾的 `]` 不见了"看起来像驱动丢字符，
   实际是**应用自己**的 truncateLines 按终端宽度切掉的：原始字节里
   `（MOSTLY_Q4_K_M）` 后面直接就是 `\r\n`，把同一批字节重放一遍，
   得到的是同一个屏幕。判"应用是不是截了一行"要看**字节**，不是看屏幕。
2. **写到第 80 列本身不换行**：字符落在最后一列就留在那一行上，
   既不丢也不挪（有测试钉着）。超出宽度的字符会被丢弃、不折行 ——
   而 bubbletea 自己会把每行截到终端宽度，所以只有本来就超宽的帧会碰到。

## 退出码

0 = 全部断言通过；1 = 有断言不符（stderr 里写明第几步、期望什么、
实际屏幕是什么）；2 = 用法/启动错误（比如命令跑不起来）。
"""

from __future__ import annotations

import argparse
import codecs
import errno
import fcntl
import os
import pty
import re
import select
import struct
import sys
import termios
import time
import unicodedata

# CSI（含 SGR）/ OSC / 字符集选择 / 私有模式：一次匹配一个序列，
# 剩下的按"可见字符"逐个处理。
_TOKEN = re.compile(
    r"\x1b\[([0-9;?]*)([a-zA-Z])"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[()][AB0]|\x1b[>=]"
    r"|[\s\S]"
)
# termenv 问背景色 / 光标位置；替真终端回话：深色背景 + 光标在左上角
_QUERY = (b"\x1b]11;?", b"\x1b[6n")
_REPLY = b"\x1b]11;rgb:1e1e/1e1e/1e1e\x1b\\\x1b[1;1R"


def decode_keys(s: str) -> str:
    """把 `\\r`/`\\x7f` 这类写法还原成真正的按键字节。

    纯 ASCII 才走 unicode_escape —— 直接对含中文的字符串用它会把
    多字节字符按 latin-1 拆坏。
    """
    if s.isascii():
        return codecs.decode(s, "unicode_escape")
    return s


# 一次按键的边界：转义序列整体算一次（`\x1b[A` 是方向键，
# 拆开就变成裸 ESC + "[" + "A" 了），其余一个字符算一次。
_KEYSEQ = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z~]|\x1bO[\s\S]|[\s\S]")


def split_keys(s: str) -> list[str]:
    return _KEYSEQ.findall(s)


# 一个**完整**的转义序列（与 _TOKEN 的前几个分支同形，但没有"任意字符"兜底）。
# 用来判断末尾那个 \x1b 开头的序列是不是还没收全。
_SEQ_COMPLETE = re.compile(
    r"\x1b\[[0-9;?]*[a-zA-Z]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[()][AB0]|\x1b[>=]"
)


def incomplete_escape(text: str) -> int | None:
    """返回 text 末尾那个**没收全**的转义序列的起点；没有则 None。

    **一次 read 的边界可以把一个转义序列切成两半**：pty 的读长度由内核
    决定，不由对端的一次 write 决定。剩下那半若按可见字符处理，
    `\\x1b[3;38;5;244m` 就会在屏幕上留下字面量 `[3;38;5;244m` ——
    实测那一行因此右移 16 列、结尾被顶出屏幕，**凭空造出一次截断**。
    """
    i = text.rfind("\x1b")
    if i < 0:
        return None
    # 起点能匹配出一个完整序列的，就是完整的（后面还有可见字符也照常处理）
    if _SEQ_COMPLETE.match(text, i):
        return None
    return i


class Screen:
    """最小 vt100 子集：只实现 bubbletea 真的会发的那几个序列。

    维护一格一格的字符网格，所以"当前屏幕"是真的当前屏幕 ——
    不是从字节流尾部猜出来的。
    """

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.grid = [[" "] * width for _ in range(height)]
        self.row = self.col = 0
        # 跨 read 的碎片：转义序列与多字节 UTF-8 都可能被切成两半
        self._pending = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def feed(self, data: bytes) -> None:
        # **碎片必须留到下一轮再解析**（见 incomplete_escape 的说明）：
        # UTF-8 交给增量解码器（它自己会留住不完整的码点），
        # 转义序列自己留 —— 按可见字符处理的话，屏幕上会多出一串
        # 转义码的字面量，把那一行挤右、甚至把结尾顶出屏幕。
        text = self._pending + self._decoder.decode(data)
        self._pending = ""
        cut = incomplete_escape(text)
        if cut is not None:
            self._pending, text = text[cut:], text[:cut]
        for m in _TOKEN.finditer(text):
            if m.group(2):  # CSI：group(2) 是终止字母
                self._csi(m.group(1), m.group(2))
            elif m.group(0).startswith("\x1b"):
                continue  # OSC / 其它：与画面无关
            else:
                self._put(m.group(0))

    def _csi(self, params: str, final: str) -> None:
        nums = [int(p) for p in params.split(";") if p.isdigit()]

        def arg(i: int, default: int) -> int:
            return nums[i] if i < len(nums) and nums[i] else default

        if final in "Hf":  # 光标定位（\x1b[H 是归位）
            self.row, self.col = arg(0, 1) - 1, arg(1, 1) - 1
        elif final == "A":
            self.row -= arg(0, 1)
        elif final == "B":
            self.row += arg(0, 1)
        elif final == "C":
            self.col += arg(0, 1)
        elif final == "D":
            self.col -= arg(0, 1)
        elif final == "E":  # 移到下一行行首
            self.row += arg(0, 1)
            self.col = 0
        elif final == "F":
            self.row -= arg(0, 1)
            self.col = 0
        elif final == "K":  # 清行
            mode = arg(0, 0)
            if 0 <= self.row < self.h:
                if mode == 0:
                    for c in range(max(self.col, 0), self.w):
                        self.grid[self.row][c] = " "
                elif mode == 2:
                    self.grid[self.row] = [" "] * self.w
        elif final == "J":  # 清屏
            if arg(0, 0) == 2:
                self.grid = [[" "] * self.w for _ in range(self.h)]
        # SGR（m）与其余私有模式：与文字无关，忽略
        self._clamp()

    def _put(self, ch: str) -> None:
        if ch == "\r":
            self.col = 0
        elif ch == "\n":  # LF 只下移一行，列不动（bubbletea 依赖这一点）
            self.row += 1
        elif ch == "\b":
            self.col -= 1
        elif ch == "\t":
            self.col = (self.col // 8 + 1) * 8
        elif ord(ch) >= 32:
            wide = unicodedata.east_asian_width(ch) in ("W", "F")
            if 0 <= self.row < self.h and 0 <= self.col < self.w:
                self.grid[self.row][self.col] = ch
                if wide and self.col + 1 < self.w:
                    self.grid[self.row][self.col + 1] = ""  # 宽字符占两格
            self.col += 2 if wide else 1
        self._clamp()

    def _clamp(self) -> None:
        self.row = min(max(self.row, 0), self.h - 1)
        self.col = min(max(self.col, 0), self.w)

    def text(self) -> str:
        return "\n".join("".join(r).rstrip() for r in self.grid)


class Tui:
    """一个跑在 pty 里的 TUI 进程，外加它当前的屏幕。"""

    def __init__(self, cmd: list[str], width: int, height: int, startup_wait: float, key_gap: float):
        self.screen = Screen(width, height)
        self.key_gap = key_gap
        self.answered = 0
        self._tail = b""  # 查询可能被切成两次读到，留一点尾巴
        self.pid, self.fd = pty.fork()
        if self.pid == 0:  # 子进程：stdin/stdout/stderr 都换成 pty 从端
            os.environ["TERM"] = "xterm-256color"
            try:
                os.execvp(cmd[0], cmd)
            except OSError as e:
                os.write(2, f"无法执行 {cmd[0]}: {e}\n".encode())
                os._exit(127)
        # 坑 2：不给尺寸的话内容区只有 1 行
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", height, width, 0, 0))
        # **先等到真的看见输出，再往下走**：不能只等"安静 0.5 秒" ——
        # 命令可能是慢启动的（实测：一个刚构建出来、第一次被执行的二进制
        # 会被系统拖上几秒）。那期间 pty 会先把按键收下，而应用切到 raw
        # 模式时会把这条没结束的行丢掉，于是"第一步的按键凭空消失"。
        deadline = time.time() + startup_wait
        self.startup_bytes = 0
        while not self.startup_bytes and time.time() < deadline:
            self.startup_bytes = self._read_until_quiet(quiet=0.5, max_wait=0.5)
        if self.startup_bytes:  # 首帧之后再让它安静一会儿（异步填充会断续重绘）
            self._read_until_quiet(quiet=0.5, max_wait=1.0)

    # ── 读

    def _read_once(self, timeout: float) -> bytes:
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return b""
        try:
            data = os.read(self.fd, 1 << 20)
        except OSError as e:
            if e.errno == errno.EIO:  # 从端关闭 = 进程退出了
                return b""
            raise
        # 坑 1：替终端回答 termenv 的查询，否则它一直等（约 1s 后放弃）
        probe, self._tail = self._tail + data, data[-16:]
        if any(q in probe for q in _QUERY):
            os.write(self.fd, _REPLY)
            self.answered += 1
        return data

    def _read_until_quiet(self, quiet: float, max_wait: float) -> int:
        """读到"连续 quiet 秒没有新输出"为止（或 max_wait 到点），喂给屏幕。"""
        total, last, end = 0, time.time(), time.time() + max_wait
        while time.time() < end:
            data = self._read_once(0.1)
            if data:
                self.screen.feed(data)
                total += len(data)
                last = time.time()
            elif time.time() - last >= quiet:
                break
        return total

    # ── 状态

    def alive(self) -> bool:
        try:
            pid, _ = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            return False
        return pid == 0

    def send(self, keys: str) -> None:
        """逐键写（坑 4）：一次 write 多字符会被 bubbletea 当成粘贴。"""
        for k in split_keys(keys):
            os.write(self.fd, k.encode())
            time.sleep(self.key_gap)

    def wait_for(self, expect: str, reject: str, timeout: float) -> tuple[bool, str]:
        """轮询到 expect 出现（或 reject 出现 / 进程退出 / 超时）。

        **不做"睡一会儿再读一次"**：重绘可能还没到，那样会拿上一屏去断言，
        结果是偶发假红 —— 而一个偶尔假红的工具最后没人跑。
        """
        deadline = time.time() + timeout
        while True:
            self._read_until_quiet(quiet=0.35, max_wait=min(0.6, max(0.1, deadline - time.time())))
            scr = self.screen.text()
            if not self.alive():
                return False, f"进程已退出（断言时它应当还活着）\n{scr}"
            if reject and reject in scr:
                return False, f"屏幕上出现了不该出现的 {reject!r}\n{scr}"
            if expect in scr:
                return True, scr
            if time.time() >= deadline:
                return False, f"等不到 {expect!r}（超时 {timeout:g}s）\n{scr}"

    def close(self) -> None:
        if self.alive():
            os.kill(self.pid, 9)
        try:
            os.waitpid(self.pid, 0)
        except ChildProcessError:
            pass


def parse_size(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)[x×](\d+)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"尺寸要写成 80x24，收到 {s!r}")
    return int(m.group(1)), int(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="在真 pty 里驱动 TUI 并逐步断言（坑与用法见文件头注释）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cmd", required=True, help="要跑的命令（可含参数，按空格切分）")
    ap.add_argument("--size", type=parse_size, default=(80, 24), help="终端尺寸，默认 80x24")
    ap.add_argument("--timeout", type=float, default=5.0, help="每步断言超时秒数，默认 5")
    ap.add_argument("--startup-wait", type=float, default=4.0, help="启动后最多等首帧的秒数")
    ap.add_argument("--key-gap", type=float, default=0.05, help="逐键写之间的间隔秒数（见坑 4），默认 0.05")
    ap.add_argument("--verbose", action="store_true", help="每步打印当前屏幕")
    ap.add_argument("--send", action="append", default=[], help="这一步要发的按键（可重复；每个字符算一次按键）")
    ap.add_argument("--expect", action="append", default=[], help="本步后屏幕上应出现的文字（按下标与 --send 配对）")
    ap.add_argument("--reject", action="append", default=[], help="本步后屏幕上不应出现的文字（可选，按下标配对）")
    ap.add_argument("--expect-exit", action="store_true", help="最后一步之后要求进程自己退出")
    args = ap.parse_args()

    if len(args.expect) > len(args.send) or len(args.reject) > len(args.send):
        print("--expect/--reject 不能比 --send 多", file=sys.stderr)
        return 2

    width, height = args.size
    tui = Tui(args.cmd.split(), width, height, args.startup_wait, args.key_gap)
    try:
        print(f"启动 {args.cmd}（{width}x{height}）—— 替终端回应查询 {tui.answered} 次")
        if not tui.alive():
            print("命令启动后立刻退出了。最后一屏：", file=sys.stderr)
            print(tui.screen.text(), file=sys.stderr)
            return 2
        if not tui.startup_bytes:
            print(f"{args.startup_wait:g}s 内没有任何输出 —— 命令没起来？"
                  f"（加大 --startup-wait 再试）", file=sys.stderr)
            return 2
        if args.verbose:
            print("--- 首帧 ---\n" + tui.screen.text())

        failed = 0
        for i, keys in enumerate(args.send, 1):
            expect = args.expect[i - 1] if i - 1 < len(args.expect) else ""
            reject = args.reject[i - 1] if i - 1 < len(args.reject) else ""
            tui.send(decode_keys(keys))
            if not expect and not reject:
                tui._read_until_quiet(quiet=0.3, max_wait=0.5)
                print(f"✓ 第 {i} 步 {keys!r}（没断言）")
                continue
            ok, detail = tui.wait_for(expect, reject, args.timeout)
            if ok:
                what = f"看到 {expect!r}" if expect else f"没有 {reject!r}"
                print(f"✓ 第 {i} 步 {keys!r} → {what}")
                if args.verbose:
                    print(detail)
            else:
                failed += 1
                print(f"✗ 第 {i} 步 {keys!r} —— {detail}", file=sys.stderr)

        if args.expect_exit:
            deadline = time.time() + args.timeout
            while tui.alive() and time.time() < deadline:
                tui._read_until_quiet(quiet=0.3, max_wait=0.3)
            if tui.alive():
                failed += 1
                print("✗ 期望进程退出，但它还活着。最后一屏：\n" + tui.screen.text(), file=sys.stderr)
            else:
                print("✓ 进程按预期退出")

        if failed:
            print(f"\n{failed}/{len(args.send)} 步断言不符", file=sys.stderr)
            return 1
        print(f"\n{len(args.send)} 步全部通过")
        return 0
    finally:
        tui.close()


if __name__ == "__main__":
    sys.exit(main())
