#!/usr/bin/env python3
"""核对 README 里**可机械验证**的具体值是否与代码一致。

    python3 tools/check_readme.py

## 为什么需要它

README 里写死了一批具体值：选项名、仓库路径、扫描根、脚本文件名、速查表组名。
它们的唯一真相源在代码里，而 README 改了代码不会红 —— 实测过：写文档时把
safetensors 的头部上限写成 100 MB（代码是 100 MiB）、把 TUI 的 `r` 键
写成"忽略缓存"（代码里 TUI 根本不碰缓存）。两个错都不会有任何检查发现。

## 它检查什么，不检查什么

**检查**：名字与路径这类可枚举的东西 —— 文档提到的必须存在。

**不检查**：任何数值。终端截图里的速查表条目数（`量化方案 (23)` 之类）、
内存实测值、位宽表里的数字，都**不在**本脚本的射程内：它们是"某次运行
的真实输出"，不是"当前的规格"。要让它们也可核对，得先让那些数从代码里
导出（例如给 ref 加一个打印条目数的入口），那是另一件事，别把本脚本
读成"README 全对"的保证。

## 两个踩过的坑（都体现在下面的写法里）

1. **扫描到空集必须是错误**。第一版没有这条守卫，于是把 README 清空、
   或者改掉文件名让正则匹配不上，脚本会输出"全部一致" —— 检查自己失效
   比不检查更危险。现在每一类提取到 0 项都会报错。
2. **按语法位置提取，不按子串包含**。脚本只认反引号里的内容（README 的
   写法约定），正文里顺嘴提一句 `--foo` 不算数。
   而且**字符类不能只写 `[a-z_]`**：那样 `verify_gguf_blocks_X.py` 这种
   坏名字压根不进集合，检查照样绿 —— 实测过这个绕过。大小写与数字全收进来，
   让"长得像脚本名"的一律进集合，再逐个证伪。
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

# Go 的 flag 包自动支持 -h/--help，它不经过 flag.X("help") 注册。
# 豁免必须显式列名，不能写成"找不到就放过"—— 那等于没有这条检查。
FLAG_EXEMPT = {"help"}

problems = []
counts = {}


def spans(pattern, text):
    """按语法位置提取：只取反引号包裹的 code span。"""
    return sorted(set(re.findall(pattern, text)))


def read(rel):
    return (ROOT / rel).read_text()


def main():
    if not README.exists():
        print("✗ 读不到 README.md —— 检查对象没了，不是通过", file=sys.stderr)
        return 1
    text = README.read_text()

    # --- 1. 选项名 ---
    # **取真正注册的 flag，不做全文子串搜索**：第一版写的是
    # `f'"{f}"' not in src`，于是文档里写 `--scan`（那是子命令，不是选项）
    # 照样通过 —— 源码里 `flag.Arg(0) == "scan"` 的那对引号满足了它。
    # 实测过这个绕过，这是"形式上满足"的典型：只问值出现过没有，
    # 字符串与注释都能满足它。按语法位置提取才问得出"它是不是一个选项"。
    src = "".join(p.read_text() for p in (ROOT / "cmd" / "modelview").glob("*.go"))
    registered = set(re.findall(
        r'flag\.(?:Bool|Int|Int64|String|Duration|Float64|Var)\(\s*"([A-Za-z0-9_-]+)"',
        src))
    if not registered:
        problems.append("【空集】没能从 cmd/modelview 里提取到任何 flag 注册 —— 检查失效")
    flags = [f for f in spans(r"`--([a-z][a-z-]*)", text) if f not in FLAG_EXEMPT]
    for f in flags:
        if f not in registered:
            problems.append(f"README 提到的 --{f} 不是已注册的选项")
    counts["选项名"] = len(flags)

    # --- 2. 仓库内路径 ---
    paths = spans(r"`((?:internal|docs|cmd|tools)/[A-Za-z0-9_./-]+)`", text)
    for p in paths:
        if not (ROOT / p).exists():
            problems.append(f"README 提到的路径不存在：{p}")
    counts["仓库内路径"] = len(paths)

    # --- 3. 扫描根（缓存退路在 analyze/cache.go，与扫描根不是一回事）---
    disc = read("internal/discover/discover.go")
    homepaths = spans(r"`(~/[^`]+)`", text)
    roots = 0
    for h in homepaths:
        if h.startswith("~/.cache/modelview"):
            continue
        roots += 1
        if f'"{h}"' not in disc:
            problems.append(f"README 列出的扫描路径不在 discover.Paths() 里：{h}")
    counts["扫描路径"] = roots

    # --- 4. 缓存的两条路径 ---
    cache = read("internal/analyze/cache.go")
    for lit in (".modelview-cache", "~/.cache/modelview"):
        if lit not in text:
            problems.append(f"README 没写缓存路径 {lit}")
    if "UserCacheDir" not in cache or "modelview" not in cache:
        problems.append("analyze/cache.go 里找不到退路缓存目录的构造，本条检查需要更新")
    counts["缓存路径"] = 2

    # --- 5. tools/ 下的脚本 ---
    scripts = spans(r"`([A-Za-z0-9_]+\.(?:py|go|sh))`", text)
    for s in scripts:
        if not (ROOT / "tools" / s).exists():
            problems.append(f"README 提到的脚本不存在：tools/{s}")
    counts["tools 脚本"] = len(scripts)

    # --- 6. 速查表组名（只看名字，不看条目数，见文件头）---
    ref = "".join(p.read_text() for p in (ROOT / "internal" / "ref").glob("*.go")
                  if not p.name.endswith("_test.go"))
    groups = spans(r"`(数值格式|量化方案|GGUF 元数据键|张量命名|GGML 类型码)`", text)
    for g in groups:
        if g not in ref:
            problems.append(f"README 提到的速查表组在 internal/ref 里不存在：{g}")
    counts["速查表组名"] = len(groups)

    # --- 空集守卫 ---
    for name, n in counts.items():
        if n == 0:
            problems.append(f"【空集】{name} 一类提取到 0 项 —— 检查本身失效了，不是通过")

    print("提取到的可核对项：" + "，".join(f"{k}={v}" for k, v in counts.items()))
    if problems:
        print(f"\n{len(problems)} 处不符：")
        for p in problems:
            print("  ✗", p)
        return 1
    print("\n✓ 名字与路径全部一致（数值不在本脚本射程内，见文件头）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
