#!/usr/bin/env python3
"""把 ref 的 general.file_type 取值表与 llama.cpp 上游逐条对照。

    python3 tools/verify_ftype_table.py

需要联网（取上游两个文件），退出码 0 表示完全一致。

## 为什么需要它

表里的码与名称是**手抄**自上游的，而手抄错了不会有任何东西发现：
速查表只被"有没有这一条"的测试守着，内容对不对没人管
（实测：把 22 号的显示名换成别的档位，全部测试照样绿）。

这是**唯一**能独立验证这张表的方式 —— 它不与 Go 实现共享任何东西，
从上游源码重新解析一遍。其余 verify_*.py 验的是"我们的解码与真实文件
一致"，这个验的是"我们的常量与上游一致"，是另一类。

## 对照的两处来源

  - include/llama.h                    enum llama_ftype 的**编号**
  - src/llama-model-loader.cpp         llama_ftype_name 的**显示名**

两边都要对：编号对而显示名错，界面上给用户看的还是错的。
"""

from __future__ import annotations

import os
import re
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/"
SOURCES = {
    "llama.h": RAW + "include/llama.h",
    "llama-model-loader.cpp": RAW + "src/llama-model-loader.cpp",
}
TABLE_GO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "internal", "ref", "ggufkeys.go")


def fetch(url: str) -> str:
    """取一个上游文件。

    网络问题要说人话：这个脚本是手动跑的，抛一屏 traceback
    会让人以为是脚本坏了，而实际只是网断了。
    """
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001 —— 网络错误种类多，一律转成一句人话
        raise SystemExit(f"取不到 {url}：{e}\n（这个脚本需要联网；"
                         f"网络不通时它无法验证任何东西，所以不能静默跳过）") from None


def parse_enum(header: str) -> dict[str, int]:
    """从 llama.h 里取出 enum llama_ftype 的 名称 → 编号。

    **只有没被注释掉的行才算数**：4/5/6 与 33/34/35 是上游注掉的保留编号。
    它们由 parse_deprecated 单独解析 —— 我们表里**照列但标明已废弃**
    （真实文件里会出现这些编号：实测 ollama 的 gpt-oss:20b 写的就是 4）。
    """
    m = re.search(r"enum llama_ftype\s*\{(.*?)\n\s*\};", header, re.S)
    if not m:
        raise SystemExit("在 llama.h 里找不到 enum llama_ftype —— 上游结构变了")
    out: dict[str, int] = {}
    for line in m.group(1).split("\n"):
        code = line.split("//")[0]          # 去掉行尾注释
        if code.strip().startswith("//"):   # 整行被注释掉
            continue
        mm = re.match(r"\s*LLAMA_FTYPE_(\w+)\s*=\s*(\d+)\s*,", code)
        if mm:
            out[mm.group(1)] = int(mm.group(2))
    if not out:
        raise SystemExit("enum llama_ftype 解析出 0 项 —— 解析器要跟着上游改")
    return out


def parse_deprecated(header: str) -> dict[int, str]:
    """从 llama.h 里取出**被注释掉**的废弃档：编号 → 枚举名。

    与 parse_enum 相反：这里要的正是那些 `//` 开头的行。
    每行的形状是 `// LLAMA_FTYPE_MOSTLY_Q4_2 = 5,  // support has been removed`。
    """
    out: dict[int, str] = {}
    for line in header.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("//"):
            continue
        m = re.match(r"//\s*LLAMA_FTYPE_(\w+)\s*=\s*(\d+)\s*,", stripped)
        if m:
            out[int(m.group(2))] = m.group(1)
    return out


def parse_go_deprecated(src: str) -> dict[int, str]:
    """从 ggufkeys.go 的 deprecatedFileTypes 里取 编号 → 枚举名。

    与 parse_go_table 分开：那是现役档（fileTypeValues），
    这是废弃档，两者在 Go 里就是两张表。
    """
    i = src.find("var deprecatedFileTypes")
    if i < 0:
        raise SystemExit("在 ggufkeys.go 里找不到 deprecatedFileTypes")
    j = src.find("}{", i)
    if j < 0:
        raise SystemExit("deprecatedFileTypes 的声明形状变了（找不到 `}{`）")
    j += 1
    depth, k = 0, j
    while k < len(src):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    out: dict[int, str] = {}
    for mm in re.finditer(r'\{(\d+),\s*"([^"]*)"', src[j:k]):
        out[int(mm.group(1))] = mm.group(2)
    if not out:
        raise SystemExit("deprecatedFileTypes 解析出 0 项 —— 解析器要跟着 Go 代码改")
    return out


def parse_display_names(cpp: str) -> dict[str, str]:
    """从 llama-model-loader.cpp 的 llama_ftype_name 里取 名称 → 显示名。

    形状是 `case LLAMA_FTYPE_MOSTLY_Q4_K_M: return "Q4_K - Medium";`，
    取值可能是三元表达式，那就只取第一个字符串字面量。
    """
    m = re.search(r"llama_ftype_name\s*\(.*?\)\s*\{(.*?)\n\s*\}", cpp, re.S)
    if not m:
        raise SystemExit("在 llama-model-loader.cpp 里找不到 llama_ftype_name")
    out: dict[str, str] = {}
    for mm in re.finditer(r"case\s+LLAMA_FTYPE_(\w+)\s*:(.*?)(?=case\s|\Z)",
                          m.group(1), re.S):
        name, body = mm.group(1), mm.group(2)
        s = re.search(r'"([^"]*)"', body)   # 三元表达式也只取首个字面量
        if s:
            out[name] = s.group(1)
    if not out:
        raise SystemExit("llama_ftype_name 解析出 0 项 —— 解析器要跟着上游改")
    return out


def parse_go_table(src: str) -> dict[int, tuple[str, str]]:
    """从 ggufkeys.go 里取 fileTypeValues 的 编号 → (枚举名, 显示名)。

    用花括号配对取字面量，**不用正则**：声明形状是
    `var fileTypeValues = []struct { 字段... }{ 值... }`，
    两个花括号挨在一起，正则很容易匹配到字段定义那一段而不是值那一段
    （第一版就是这么错的：解析出 0 项）。
    """
    i = src.find("var fileTypeValues")
    if i < 0:
        raise SystemExit("在 ggufkeys.go 里找不到 fileTypeValues")
    # 从 `}{` 的第二个花括号开始配对，跳过字段定义
    j = src.find("}{", i)
    if j < 0:
        raise SystemExit("fileTypeValues 的声明形状变了（找不到 `}{`）")
    j += 1
    depth, k = 0, j
    while k < len(src):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    body = src[j:k]

    out: dict[int, tuple[str, str]] = {}
    for mm in re.finditer(r'\{(\d+),\s*"([^"]*)",\s*"([^"]*)"', body):
        out[int(mm.group(1))] = (mm.group(2), mm.group(3))
    if not out:
        raise SystemExit("fileTypeValues 解析出 0 项 —— 解析器要跟着 Go 代码改")
    return out


def main() -> int:
    header = fetch(SOURCES["llama.h"])
    cpp = fetch(SOURCES["llama-model-loader.cpp"])
    enums = parse_enum(header)
    displays = parse_display_names(cpp)
    go_src = open(TABLE_GO, encoding="utf-8").read()
    ours = parse_go_table(go_src)

    # 上游把 LLAMA_FTYPE_ 前缀去掉后就是我们的枚举名
    upstream = {code: name for name, code in enums.items()}
    # GUESSED 是特殊档（1024），上游的显示名带 "(guessed) " 前缀
    bad = 0

    for code, (enum, display) in sorted(ours.items()):
        up_enum = upstream.get(code)
        if up_enum is None:
            print(f"✗ {code}: 我们的枚举名是 {enum}，上游这个编号不存在", file=sys.stderr)
            bad += 1
            continue
        if up_enum != enum:
            print(f"✗ {code}: 我们叫 {enum}，上游叫 {up_enum}", file=sys.stderr)
            bad += 1
        up_display = displays.get(up_enum)
        if up_display is None:
            print(f"✗ {code} {enum}: 上游 llama_ftype_name 里没有这一档", file=sys.stderr)
            bad += 1
            continue
        if up_display != display:
            print(f"✗ {code} {enum}: 显示名我们写 {display!r}，上游是 {up_display!r}",
                  file=sys.stderr)
            bad += 1

    # 反向：上游有而我们没列的应当**只有 GUESSED**。
    #
    # 它在上游是一个**标志位**而不是档位：与档位码按位或在一起
    #（读 1039 = 1024|15 就是"上游猜的 Q4_K_M"）。我们的表里也不把它
    # 当档位列，而是单独一个 ftypeGuessedFlag 常量 + FileTypeByCode 里剥掉它 ——
    # 列成档位的话，用户会以为 file_type=1024 是一种量化方案。
    expected_absent = {"GUESSED"}
    missing = [c for c in sorted(set(upstream) - set(ours))
               if upstream[c] not in expected_absent]
    if missing:
        print(f"✗ 上游有但我们没列：{[(c, upstream[c]) for c in missing]}", file=sys.stderr)
        bad += 1
    # 而 GUESSED 那个标志位的值必须与上游一致
    if upstream.get(1024) != "GUESSED":
        print(f"✗ 上游 1024 号不再是 GUESSED，而是 {upstream.get(1024)!r}", file=sys.stderr)
        bad += 1

    # 废弃档：与上游被注释掉的那几行双向比对。
    #
    # 这 6 条曾经完全没有对照 —— 唯一的守卫是同一次提交里手抄的测试，
    # 两边一起错就没人管；而注释里写着"复核用这个脚本"，
    # 跑一遍还会得到"✓ 一致"，误以为复核过了。
    up_dep = parse_deprecated(header)
    our_dep = parse_go_deprecated(go_src)
    for code, enum in sorted(our_dep.items()):
        if code not in up_dep:
            print(f"✗ 废弃档 {code}（我们叫 {enum}）在上游的注释里找不到",
                  file=sys.stderr)
            bad += 1
        elif up_dep[code] != enum:
            print(f"✗ 废弃档 {code}：我们叫 {enum}，上游注释里是 {up_dep[code]}",
                  file=sys.stderr)
            bad += 1
    for code, enum in sorted(up_dep.items()):
        if code not in our_dep:
            # **只提示，不判失败**：漏列一条废弃档由 Go 侧
            # TestFileType_废弃编号查得到 拦（它写死了那 6 个编号）。
            # 这里判失败的话，就成了"同一件事两处判"，
            # 而其中一处（这里）要联网才能跑。
            print(f"· 上游注释掉的 {code} {enum} 我们没列"
                  f"（漏列由 Go 侧 TestFileType_废弃编号查得到 拦）")
    print(f"  废弃档：我们 {len(our_dep)} 项，上游注释里 {len(up_dep)} 项")

    if bad:
        print(f"\n{bad} 处不一致（我们 {len(ours)} 项，上游 {len(upstream)} 项）",
              file=sys.stderr)
        return 1
    print(f"✓ file_type 表与上游一致：{len(ours)} 项（上游同编号 {len(upstream)} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
