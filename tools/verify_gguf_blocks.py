#!/usr/bin/env python3
"""用真实 GGUF 文件反推验证块大小表。

## 为什么需要这个脚本

`internal/parser/gguf/tensor.go` 的 `blockBytes` 表记录每种 GGML 量化类型
「一个块占多少字节」。这些数字**不能凭记忆写** —— 写错不会导致解析失败，
只会让所有该类型张量的 byte_size 静默算错，而文件大小、压缩比一路跟着错。

验证方法：GGUF 的数据区把张量紧排到文件末尾（末尾差值实测恰好为 0 字节）。
因此 `数据区起点 + Σ(每个张量的字节数) == 文件大小` 必须成立。
只要有一个类型的块大小写错，这个等式就会被打破。

Go 侧的 `TestBlockTable_与真实文件吻合` 做的是同一件事，但那条测试
依赖 `~/.ollama` 目录，在 CI 或别人机器上会跳过。本脚本是它的人工复现路径。

## 用法

    python3 tools/verify_gguf_blocks.py                      # 扫描 ~/.ollama 全部模型
    python3 tools/verify_gguf_blocks.py path/to/model.gguf   # 验证指定文件

退出码 0 表示全部吻合；1 表示有不吻合（说明块表或该文件有问题）。

## 覆盖边界

本脚本只能验证**给定文件里实际出现的类型**。本机五个 ollama 模型覆盖
F32/F16/BF16/Q4_K/Q5_0/Q6_K/Q8_0/MXFP4 共 8 项。其余 13 项
（Q4_0/Q4_1/Q5_1/Q8_1/Q2_K/Q3_K/Q5_K/Q8_K/I8/I16/I32/I64/F64，以及 NVFP4）
没有任何本地文件覆盖，只能靠 Go 侧 `TestTensorByteSize_覆盖全部类型码`
逐值钉住 —— 那张表的期望值来自 llama.cpp 的结构体定义，属于独立来源，
但如果那里也抄错了，两边一致地错是拦不住的。

**遇到未收录的类型码时本脚本判失败并说明"校验不完整"**，不跳过 ——
跳过会让那一段字节不算进跨度，凭空造出"张量间空隙"，把
"我表里缺一项"说成"文件有问题"。
"""

from __future__ import annotations

import glob
import os
import re
import struct
import sys

# GGML 值类型 → (struct 格式, 字节数)
SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}

# 与 Go 侧 blockBytes 对应的表（本脚本独立维护一份，用于交叉核对）
BLOCK_BYTES = {
    0: 4, 1: 2, 2: 18, 3: 20, 6: 22, 7: 24, 8: 34, 9: 36,
    10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 15: 292,
    24: 1, 25: 2, 26: 4, 27: 8, 28: 8, 30: 2,
    39: 17, 40: 36,   # MXFP4 / NVFP4
}
BLOCK_ELEMS = {
    2: 32, 3: 32, 6: 32, 7: 32, 8: 32, 9: 32,
    10: 256, 11: 256, 12: 256, 13: 256, 14: 256, 15: 256,
    39: 32, 40: 64,   # MXFP4 / NVFP4
}

TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 39: "MXFP4", 40: "NVFP4",
}

MAX_STRING = 64 << 20


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n):
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError(f"需要 {n} 字节，只读到 {len(b)}")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        n = self.u64()
        if n > MAX_STRING:
            raise ValueError(f"字符串长度 {n} 异常（指针可能已错位）")
        return self.raw(n).decode("utf-8", "replace")


def consume(r: Reader, t: int):
    """完整消费一个元数据值。数组必须全部读完，否则指针错位。"""
    if t == 8:
        r.string()
        return None
    if t == 9:
        et = r.u32()
        n = r.u64()
        if et == 8:
            for _ in range(n):
                r.string()
        elif et == 9:
            for _ in range(n):
                consume(r, 9)
        elif et in SCALAR:
            fmt, sz = SCALAR[et]
            r.raw(n * sz)
        else:
            raise ValueError(f"未知数组元素类型 {et}")
        return None
    if t in SCALAR:
        fmt, sz = SCALAR[t]
        return struct.unpack(fmt, r.raw(sz))[0]
    raise ValueError(f"未知元数据类型 {t}")


def analyze(path: str) -> tuple[bool, str]:
    """解析一个 GGUF，返回 (是否吻合, 说明)。"""
    with open(path, "rb") as fh:
        r = Reader(fh)
        if r.raw(4) != b"GGUF":
            return False, "不是 GGUF"

        r.u32()  # version
        tensor_count = r.u64()
        kv_count = r.u64()

        alignment = 32
        for _ in range(kv_count):
            key = r.string()
            vtype = r.u32()
            val = consume(r, vtype)
            if key == "general.alignment" and isinstance(val, int) and val > 0:
                alignment = val

        tensors = []
        for _ in range(tensor_count):
            r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ttype = r.u32()
            offset = r.u64()
            tensors.append((dims, ttype, offset))

        header_end = fh.tell()

    data_start = (header_end + alignment - 1) // alignment * alignment
    file_size = os.path.getsize(path)

    unknown = set()
    total = 0
    spans = []  # (起始偏移, 结束偏移, 类型名)
    for dims, ttype, offset in tensors:
        n = 1
        for d in dims:
            n *= d
        if ttype not in BLOCK_BYTES:
            unknown.add(ttype)
            continue
        elems = BLOCK_ELEMS.get(ttype, 1)
        if elems > 1 and n % elems != 0:
            return False, f"元素数 {n} 不是块大小 {elems} 的整数倍（类型 {ttype}）"
        size = (n // elems) * BLOCK_BYTES[ttype] if elems > 1 else n * BLOCK_BYTES[ttype]
        total += size
        start = data_start + offset
        spans.append((start, start + size, TYPE_NAME.get(ttype, str(ttype))))

    # 未收录的类型**必须让整次校验作废**，不能跳过它继续查空隙：
    # 跳过意味着那一段字节没被算进 spans，于是"空隙"必然出现 ——
    # 实测 MXFP4 没收录时，脚本在唯一含它的文件上报
    # "张量间空隙 141004800 字节"，把"我表里缺一项"说成了"文件有问题"。
    # 一个永远红、且指向错误原因的脚本，结果是没人再跑它。
    if unknown:
        names = sorted(unknown)
        return False, (f"**本次校验不完整**：文件里有未收录的类型码 {names}，"
                       f"它们的字节数无法计算，张量跨度检查已放弃。\n"
                       f"    先把这些类型补进 BLOCK_BYTES/BLOCK_ELEMS/TYPE_NAME 再跑。")

    # 关键不变式：按偏移排序后，相邻张量**不得重叠**.
    #
    # 不能只检查「最大的 offset+size 是否等于文件大小」——
    # 那样只有排在最末的那个类型会被验证到。把 Q4_K 从 144 改成 145，
    # 只要 Q4_K 张量不在末尾，max_end 就不变，检查照样通过。
    # 重叠检查对任何一处块大小偏大都敏感（偏小则表现为空隙）。
    spans.sort()
    for (s1, e1, n1), (s2, _e2, n2) in zip(spans, spans[1:]):
        if e1 > s2:
            return False, (
                f"张量重叠 {e1 - s2} 字节：{n1} 结束于 {e1}，{n2} 起始于 {s2}\n"
                f"    说明这两者之一的块大小算错了"
            )
        if s2 - e1 >= alignment:
            return False, (
                f"张量间空隙 {s2 - e1} 字节（≥ 对齐单位 {alignment}）："
                f"{n1} 结束于 {e1}，{n2} 起始于 {s2}"
            )

    diff = file_size - spans[-1][1] if spans else file_size - data_start
    covered = sorted({TYPE_NAME.get(t, str(t)) for _, t, _ in tensors if t in BLOCK_BYTES})
    note = f"数据区起点 {data_start:,}  张量总字节 {total:,}  末尾差值 {diff}"
    if unknown:
        note += f"  [未收录类型 {sorted(unknown)}]"
    ok = diff == 0
    return ok, f"{note}\n    覆盖类型: {', '.join(covered)}"


# 与 Go 侧 internal/ollamablob 同一判据。
#
# Python 不能 import Go，所以这份是**镜像**，判据要改就两处一起改。
# 用精确形状而不是 `"-partial" in name`：宽判据会把 `sha256-abc-partial`
# 这种短名字也当成下载中，于是正常 blob 被永远排除在外（静默漏报）。
_PARTIAL_RE = re.compile(r"^sha256-[0-9a-f]{64}-partial")


def is_in_progress(name: str) -> bool:
    """这个 blob 文件名是不是还没下完（ollama 下到一半的中间状态）。"""
    return _PARTIAL_RE.match(name) is not None

def find_blobs() -> list[str]:
    home = os.path.expanduser("~")
    d = os.path.join(home, ".ollama", "models", "blobs")
    if not os.path.isdir(d):
        return []
    out = []
    for p in sorted(glob.glob(os.path.join(d, "sha256-*"))):
        # 跳过没下完的（<digest>-partial）：读到一半的文件会给出错的结论
        if is_in_progress(os.path.basename(p)):
            continue
        if os.path.getsize(p) < 1 << 20:
            continue
        with open(p, "rb") as fh:
            if fh.read(4) == b"GGUF":
                out.append(p)
    return out


def main(argv: list[str]) -> int:
    paths = argv[1:] or find_blobs()
    if not paths:
        print("没有找到 GGUF 文件（~/.ollama/models/blobs 为空，且未指定路径）")
        return 0

    failed = 0
    for p in paths:
        name = os.path.basename(p)
        try:
            ok, note = analyze(p)
        except Exception as e:  # noqa: BLE001 - 顶层工具，任何解析失败都要报出来
            print(f"✗ {name}\n    解析失败: {e}")
            failed += 1
            continue
        print(f"{'✓' if ok else '✗'} {name}  ({os.path.getsize(p) / 2**30:.2f} GiB)")
        print(f"    {note}")
        if not ok:
            failed += 1

    print()
    if failed:
        print(f"{failed}/{len(paths)} 个文件不吻合 —— 块表或文件有问题")
        return 1
    print(f"{len(paths)}/{len(paths)} 全部吻合（末尾差值 0 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
