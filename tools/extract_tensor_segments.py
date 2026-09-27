#!/usr/bin/env python3
"""从本机模型文件里导出全部张量**段名**，作为速查表覆盖率的真值。

    python3 tools/extract_tensor_segments.py

输出 internal/ref/testdata/real_segments.txt（每行一个段，排序去重），入库。

## 为什么需要它

与元数据键同理（见 extract_metadata_keys.py）：`ref` 的价值是"用户在文件里
看到的每个段都能查到"，而**手写清单会按"我记得的"收敛** —— 第一版手列
24 个段，实测并集 73 个，漏掉的全是多模态塔（v./a.）、激活范围记录
（input_max/input_min）、Gemma 3n 的 per-layer 结构件那一批。

## 段的定义

按 `.` 切开，纯数字的段归一成 `#N` —— 层号有具体值但含义只有一种，
表里也是按 `#N` 收录的。这个归一**必须与 Go 侧 SplitTensorName 一致**：
两边不一致的话，清单里的段在表里查不到，测试会红（这正是我们要的 ——
它比"两边都错还互相印证"好）。

## 依赖

只读 GGUF 头部，不依赖本项目，也不依赖 gguf 包。
张量表紧跟在键区之后，所以得先把键区完整跳过去。
"""

from __future__ import annotations

import glob
import os
import struct
import sys

BLOBS = sorted(glob.glob(os.path.expanduser("~/.ollama/models/blobs/sha256-*")))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "internal", "ref", "testdata", "real_segments.txt")

# gguf_type 的定长标量宽度（GGUF 规范：0..7 与 10..12 是定长标量）
SCALAR_WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                10: 8, 11: 8, 12: 8}

# 头部最多读这么多。张量表在键区之后，词表（几十万项字符串）也在键区里，
# 64 MiB 够覆盖本机全部模型。
MAX_HEAD = 64 << 20


def read_string(buf: bytes, off: int) -> tuple[str, int]:
    """读一个 gguf_string（u64 长度 + UTF-8 字节），返回（串, 新偏移）。"""
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    return buf[off:off + n].decode("utf-8", "replace"), off + n


def skip_value(buf: bytes, off: int) -> int:
    """跳过一个 KV 值，返回新偏移。"""
    (t,) = struct.unpack_from("<I", buf, off)
    off += 4
    if t in SCALAR_WIDTH:
        return off + SCALAR_WIDTH[t]
    if t == 8:  # gguf_string
        _, off = read_string(buf, off)
        return off
    if t == 9:  # 数组：元素类型 + 个数 + 元素
        (et,) = struct.unpack_from("<I", buf, off)
        off += 4
        (n,) = struct.unpack_from("<Q", buf, off)
        off += 8
        if et in SCALAR_WIDTH:
            return off + n * SCALAR_WIDTH[et]
        if et == 8:
            for _ in range(n):  # 字符串数组（token 表就是它）
                _, off = read_string(buf, off)
            return off
        raise ValueError(f"数组元素类型 {et} 未处理")
    raise ValueError(f"未知的 gguf_type {t}")


class Skip(Exception):
    """这个文件不产出段 —— 原因要让人看得见，不能静默。"""


def segments_of(path: str) -> set[str]:
    """收集一个文件里张量名的全部段（层号归一成 #N）。

    读不了的文件抛 Skip 并**说明原因**：静默返回空集的话，
    上游一旦升 GGUF 版本（v3→v4 是真实会发生的事），这个脚本会
    悄悄产出一份残缺清单，而仓库里唯一的拦截是测试里那个总数断言 ——
    它会把"语料变了"和"解析器坏了"混成同一条红。
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(min(size, MAX_HEAD))
    if head[:4] != b"GGUF":
        raise Skip("不是 GGUF 文件")
    off = 4
    (version,) = struct.unpack_from("<I", head, off)
    off += 4
    if version not in (2, 3):
        # **与"不是 GGUF"分开报**：这条意味着上游变了，得改解析
        raise Skip(f"未支持的 GGUF 版本 v{version}（本脚本只认 v2/v3）")
    (tensor_count,) = struct.unpack_from("<Q", head, off)
    off += 8
    (kv_count,) = struct.unpack_from("<Q", head, off)
    off += 8

    # 先完整跳过键区 —— 张量表紧跟在它后面
    for _ in range(kv_count):
        _, off = read_string(head, off)
        off = skip_value(head, off)

    out: set[str] = set()
    for _ in range(tensor_count):
        name, off = read_string(head, off)
        (n_dims,) = struct.unpack_from("<I", head, off)
        off += 4
        off += 8 * n_dims   # 各维大小
        off += 4            # ggml_type
        off += 8            # 数据区偏移
        for seg in name.split("."):
            if not seg:
                continue
            out.add("#N" if seg.isdigit() else seg)
    return out


def main() -> int:
    found: set[str] = set()
    scanned = 0
    skipped: list[str] = []
    for b in BLOBS:
        try:
            segs = segments_of(b)
        except Skip as e:
            # 单个文件读不了不该让整件事失败，但**要说出来**
            skipped.append(f"{os.path.basename(b)}: {e}")
            continue
        except (ValueError, struct.error) as e:
            skipped.append(f"{os.path.basename(b)}: 解析失败 {e}")
            continue
        if not segs:
            skipped.append(f"{os.path.basename(b)}: 没有张量")
            continue
        scanned += 1
        found |= segs
    for line in skipped:
        print(f"跳过 {line}", file=sys.stderr)

    if not found:
        print("没有从任何文件里读到张量段 —— 先确认本机有 GGUF 模型", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# 由 tools/extract_tensor_segments.py 生成，勿手改\n")
        for s in sorted(found):
            f.write(s + "\n")
    print(f"扫了 {scanned} 个 GGUF 文件，已写入 {OUT}：{len(found)} 个段")
    return 0


if __name__ == "__main__":
    sys.exit(main())
