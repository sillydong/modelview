#!/usr/bin/env python3
r"""把 model 的 GGML 类型码表与上游逐条对照。

    uv run --with gguf python3 tools/verify_ggml_types.py

退出码 0 表示完全一致。

## 为什么需要它

`internal/model/dtype.go` 里的 `GGMLCode` 是手抄自 ggml 的 `ggml_type`
枚举的，而**手抄错了不会有任何东西发现**：仓库里的双射测试只保证
"码与类型内部一致"（不重不漏、来回可逆）—— 一个**唯一但错误**的码
照样通过。实测：把 MXFP4 的 39 改成 38，双射测试全绿。

38 与 39 特别容易混：38 是 llama.cpp 的 **file_type** 编号
（LLAMA_FTYPE_MOSTLY_MXFP4_MOE），39 才是 ggml 的**类型码**（MXFP4）。
两个是不同的枚举，名字还撞在一起。

这是**唯一**能独立验证这张表的方式 —— 它不与 Go 实现共享任何东西，
直接从 gguf 包（= ggml 的常量表）读一遍。

## 我们没收录的

TQ1_0(34) / TQ2_0(35) / Q1_0(41)：目前没在野外见过用它们的模型
（2026-09 查过 HF 下载量前 100 与 mxfp4/nvfp4/fp8 标签下的模型）。
所以这里只**报告**，不算失败 —— 但它们一旦出现，解析会报"未知的块类型"。
"""

from __future__ import annotations

import os
import re
import sys

try:
    from gguf.constants import GGMLQuantizationType
except ImportError:
    raise SystemExit(
        "需要 gguf 包：uv run --with gguf python3 tools/verify_ggml_types.py")

DTYPE_GO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "internal", "model", "dtype.go")

# 有意不收录的类型码，以及原因（出现在报告里，不判失败）
KNOWN_MISSING = {
    34: ("TQ1_0", "三元量化，没在野外见过实例"),
    35: ("TQ2_0", "三元量化，没在野外见过实例"),
    41: ("Q1_0", "1.125 bit/权重，没在野外见过实例"),
}


def parse_go_table(src: str) -> dict[int, str]:
    r"""从 dtype.go 里取 GGML 码 → Dtype 字符串。

    用 `Dtype\w+: {…GGMLCode: N}` 的形状取；Dtype 的值（字符串字面量）
    才是与 ggml 对照的东西，不是常量名 —— 常量名是 `DtypeQ2K`，
    而值是 `"Q2_K"`。
    """
    # 先收常量名 → Dtype 字面量，再收常量名 → 类型码。
    # **不写成模块级的副作用**：那让 parse_go_table 的返回值依赖"之前调用过谁"。
    names: dict[str, str] = {}
    for m in re.finditer(r'^\t(Dtype\w+)\s+Dtype = "([^"]+)"', src, re.M):
        names[m.group(1)] = m.group(2)
    out: dict[int, str] = {}
    for m in re.finditer(r'^\t(Dtype\w+):\s*\{([^}]*)\}', src, re.M):
        body = m.group(2)
        c = re.search(r'GGMLCode:\s*(-?\d+)', body)
        if not c or int(c.group(1)) < 0:
            continue
        value = names.get(m.group(1))
        if value is None:
            raise SystemExit(f"{m.group(1)} 没有对应的 Dtype 字面量")
        out[int(c.group(1))] = value
    if not out:
        raise SystemExit("从 dtype.go 里没解析出任何类型码 —— 解析器要跟着改")
    return out


def main() -> int:
    ours = parse_go_table(open(DTYPE_GO, encoding="utf-8").read())
    upstream = {t.value: t.name for t in GGMLQuantizationType
                if t.name != "UNDEFINED"}

    bad = 0
    for code, name in sorted(ours.items()):
        up = upstream.get(code)
        if up is None:
            print(f"✗ {code}: 我们叫 {name}，上游这个编号不存在", file=sys.stderr)
            bad += 1
        elif up != name:
            print(f"✗ {code}: 我们叫 {name}，上游叫 {up}", file=sys.stderr)
            bad += 1

    # KNOWN_MISSING 里的名字也要与上游比对 —— 这个 dict 是手写的，
    # 写错名字的话报告会照抄错的，而 rc 仍是 0（实测过：
    # {34: ("这不是TQ1_0", "乱写的")} 照样通过）。
    for code, (nm, _why) in sorted(KNOWN_MISSING.items()):
        up = upstream.get(code)
        if up is None:
            print(f"✗ KNOWN_MISSING 里的 {code} 在上游不存在", file=sys.stderr)
            bad += 1
        elif up != nm:
            print(f"✗ KNOWN_MISSING[{code}] 写的是 {nm}，上游叫 {up}", file=sys.stderr)
            bad += 1

    missing = sorted(set(upstream) - set(ours))
    for code in missing:
        if code in KNOWN_MISSING:
            nm, why = KNOWN_MISSING[code]
            print(f"· 未收录 {code} {nm}（{why}）")
        else:
            print(f"✗ 上游有 {code} {upstream[code]}，我们没收录", file=sys.stderr)
            bad += 1

    if bad:
        print(f"\n{bad} 处不一致", file=sys.stderr)
        return 1
    print(f"✓ GGML 类型码与上游一致：{len(ours)} 项；"
          f"另有 {len(KNOWN_MISSING)} 项有意未收录")
    return 0


if __name__ == "__main__":
    sys.exit(main())
