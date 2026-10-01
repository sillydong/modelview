#!/usr/bin/env python3
"""变异验证：故意破坏被测的守卫，确认测试真的会失败。

不带这个工具时，变异通常写成一行 perl，转义写错就静默不匹配 ——
变异没发生，测试通过，看起来"验证过了"，实际什么都没验。

这个工具强制四件事：

  1. **先跑未变异的基线**。基线不绿就直接退出 —— 基线是红的，
     后面的"变异被抓住"分不清是变异起的作用还是本来就坏。
  2. 锚点必须在文件里**恰好出现一次**，否则直接报错退出。
  3. **按输出内容判定，不只看退出码**。退出码非 0 的原因有三种：
     测试失败（期望）、编译失败、`--run` 没匹配到测试。后两种情况下
     一条测试都没跑，判成"被抓住"是假绿。实测踩过：
     把 `out[i] = int64(float64(i) * step)` 改成 `out[i] = 0`
     会让 `step` 变成未使用、包编译不过，工具报"✓ 被抓住"。
  4. 无论结果如何都还原，并用 SHA-256 确认还原到了字节级。

超时用独立进程组，避免 go test 拉起的测试二进制变成孤儿进程继续跑
（实测：工具已退出、文件已还原，但一个 100% CPU 的 decode.test
还在跑变异后的二进制，直到 Go 自己的 10 分钟超时）。

用法：
    python3 tools/mutate.py --file internal/decode/quant32.go \\
        --old 'xh1 := ((qh >> uint(j+16)) << 4) & 0x10' \\
        --new 'xh1 := (qh >> uint(j+16)) & 0x10' \\
        --test './internal/decode/' --run 'TestDecode_量化类型与参考实现吻合/Q5_0'

退出码有**三种**，按退出码判读的调用方要分开：

  0 = 变异被抓住（期望结果）
  2 = 超时（测试卡死到 --timeout）—— **不能算被抓住**：它只证明执行到了
      这段代码，不证明断言能区分对错。原先这里也返回 0，于是对按退出码
      判读的调用方，超时与"被抓住"完全不可区分（打印的文字里才有区别）。
  1 = 其余一切（漏网、编译失败、没匹配到测试、锚点不唯一、还原失败、基线不红）

"""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import subprocess
import sys

# 判定结果
CAUGHT = "caught"          # 测试跑了且有断言失败 —— 期望结果
ESCAPED = "escaped"        # 测试跑了且全绿 —— 漏网
BUILD_FAILED = "build"     # 编译不过，一条测试都没跑
NO_TESTS = "no_tests"      # --run 没匹配到测试
TIMEOUT = "timeout"        # 超时（可能是死循环）

# go test 的输出特征。用内容判定而不是退出码：三者退出码都是非 0。
_BUILD_MARKERS = ("[build failed]", "setup failed", "cannot find package")
_PANIC_MARKERS = ("panic:", "fatal error:")
_FAIL_MARKERS = ("--- FAIL:", "FAIL\t")


def classify(returncode: int, output: str) -> str:
    """按 go test 的输出判定变异结果。

    独立成函数是为了能单测 —— 这个工具的判定逻辑以前没被任何测试看着，
    结果把"编译失败"判成了"被抓住"。
    """
    if any(m in output for m in _BUILD_MARKERS):
        return BUILD_FAILED
    if "[no tests to run]" in output:
        return NO_TESTS
    if any(m in output for m in _PANIC_MARKERS):
        return CAUGHT
    if any(m in output for m in _FAIL_MARKERS):
        return CAUGHT
    if returncode == 0:
        return ESCAPED
    # 非 0 但既没有编译错误也没有 FAIL 标记：判不出来就不能算通过
    return ESCAPED


def sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_tests(cmd: list[str], timeout: float) -> tuple[int, str]:
    """跑测试；超时则杀掉整个进程组，不留孤儿。"""
    # start_new_session 让子进程自成进程组，超时时能连同测试二进制一起杀
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        out, _ = p.communicate()
        return -1, (out or "") + "\n[mutate.py: 超时，已杀掉进程组]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="要变异的源文件")
    ap.add_argument("--old", required=True, help="被替换的原文（必须唯一匹配）")
    ap.add_argument("--new", required=True, help="替换后的内容")
    ap.add_argument("--test", required=True, help="go test 的包路径")
    ap.add_argument("--run", default="", help="go test -run 的模式，留空表示全部")
    ap.add_argument("--label", default="", help="报告里显示的名字")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="单个测试的超时秒数（默认 120）")
    args = ap.parse_args()

    label = args.label or args.file
    cmd = ["go", "test", args.test, "-count=1"]
    if args.run:
        cmd += ["-run", args.run]

    # --- 1. 基线：变异之前测试必须是绿的 ---
    rc, out = run_tests(cmd, args.timeout)
    base = classify(rc, out)
    if base == BUILD_FAILED:
        print(f"!!! {label}: 基线就编译不过，变异没有意义")
        print(out.strip()[:500])
        return 1
    if base == NO_TESTS:
        print(f"!!! {label}: --run {args.run!r} 没匹配到任何测试 —— "
              f"变异即使生效也没人看着")
        return 1
    if base == CAUGHT:
        print(f"!!! {label}: 基线就是红的，先修好再验证变异")
        print(out.strip()[:500])
        return 1

    # --- 2. 变异 ---
    before = sha(args.file)
    src = open(args.file, encoding="utf-8").read()
    n = src.count(args.old)
    if n != 1:
        print(f"✗ 锚点在 {args.file} 中出现 {n} 次，必须恰好 1 次 —— 变异未发生")
        return 1

    open(args.file, "w", encoding="utf-8").write(src.replace(args.old, args.new))
    try:
        rc, out = run_tests(cmd, args.timeout)
        result = classify(rc, out)
        if "[mutate.py: 超时" in out:
            result = TIMEOUT
    finally:
        open(args.file, "w", encoding="utf-8").write(src)
        restored = sha(args.file) == before

    if not restored:
        print(f"!!! {label}: 还原失败，文件已改变，必须人工处理")
        return 1

    # --- 3. 判定 ---
    if result == CAUGHT:
        print(f"✓ 被抓住  {label}（还原一致）")
        return 0
    if result == TIMEOUT:
        # 超时只证明"测试执行到了这段代码"，不证明断言能区分对错。
        # **退出码必须非 0**：调用方多半只看退出码，返回 0 的话
        # 超时与"被抓住"在它眼里一模一样 —— 而两者的结论正好相反。
        print(f"⚠ 超时    {label}（测试卡死 {args.timeout:g}s —— "
              f"只说明执行到了，不说明断言有效；还原一致）")
        return 2
    if result == BUILD_FAILED:
        print(f"✗✗ 无效    {label}: 变异后编译不过，一条测试都没跑 —— "
              f"这不等于被抓住，换一个不破坏编译的变异")
        print(out.strip()[:400])
        return 1
    if result == NO_TESTS:
        print(f"✗✗ 无效    {label}: 变异后没有测试跑到")
        return 1
    print(f"✗✗ 漏网    {label} —— 测试对这个守卫零敏感")
    return 1


if __name__ == "__main__":
    sys.exit(main())
