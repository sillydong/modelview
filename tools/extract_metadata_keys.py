#!/usr/bin/env python3
"""从本机模型文件里导出全部 GGUF 元数据键，作为速查表覆盖率的真值。

    python3 tools/extract_metadata_keys.py

输出 internal/ref/testdata/real_keys.txt（每行一个键，排序去重），入库。

## 为什么需要它

`ref` 的价值是"用户在文件里看到的每个键都能查到"。覆盖率只有拿真实文件
量才知道，而**手写清单会按"我记得的"收敛** —— 第一版手列 68 个，
实测并集 93 个，漏掉的正好是最少见的那批（gemma4 的 MoE/vision/rope 键、
若干 tokenizer 开关、general.parameter_count），
也就正好是最需要查释义的那批。

脚本直接读 GGUF 头部，不依赖本项目，也不依赖 gguf 包 ——
元数据是 `magic + version + tensor_count + kv_count + kv 数组`。
"""

from __future__ import annotations

import glob
import os
import re
import struct
import sys

# **跳过没下完的**（<digest>-partial 及其分片）：拿半个文件当语料，
# 生成出来的清单是假的，而它会被当成真值入库。
# 与 Go 侧 internal/ollamablob 同一判据。
#
# Python 不能 import Go，所以这份是**镜像**，判据要改就两处一起改。
# 用精确形状而不是 `"-partial" in name`：宽判据会把 `sha256-abc-partial`
# 这种短名字也当成下载中，于是正常 blob 被永远排除在外（静默漏报）。
_PARTIAL_RE = re.compile(r"^sha256-[0-9a-f]{64}-partial")


def is_in_progress(name: str) -> bool:
    """这个 blob 文件名是不是还没下完（ollama 下到一半的中间状态）。"""
    return _PARTIAL_RE.match(name) is not None

BLOBS = [p for p in sorted(glob.glob(
    os.path.expanduser("~/.ollama/models/blobs/sha256-*")))
    if not is_in_progress(os.path.basename(p))]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "internal", "ref", "testdata", "real_keys.txt")

# gguf_type 的定长标量宽度（GGUF 规范：0..7 与 10..12 是定长标量）
SCALAR_WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                10: 8, 11: 8, 12: 8}

# 头部最多读这么多。GGUF 的键区在数据区之前，词表（几十万项字符串）
# 也只占 1~2 MB，64 MiB 远远够。
#
# **不要"遇到词表就停"**：第一版这么做了，结果丢了词表之后的 4 个键
#（chat_template / cls_token_id / seperator_token_id / unknown_token_id）——
# 而"每个键都能查到"正是这个脚本要保证的事。省下的那点读盘时间
# 换来一份不完整的真值，是笔亏本买卖。
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


def keys_of(path: str) -> set[str]:
    """收集一个文件里的元数据键。"""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        # 头部读够了就停：GGUF 的键区在数据区之前
        head = f.read(min(size, MAX_HEAD))
    if head[:4] != b"GGUF":
        return set()
    off = 8  # magic + version
    off += 8  # tensor_count
    (kv_count,) = struct.unpack_from("<Q", head, off)
    off += 8

    out: set[str] = set()
    for _ in range(kv_count):
        k, off = read_string(head, off)
        out.add(k)
        off = skip_value(head, off)
    return out


def main() -> int:
    found: set[str] = set()
    scanned = 0
    for b in BLOBS:
        try:
            ks = keys_of(b)
        except (ValueError, struct.error) as e:
            # 单个文件读不了不该让整件事失败，但要说出来
            print(f"跳过 {os.path.basename(b)}: {e}", file=sys.stderr)
            continue
        if not ks:
            continue
        scanned += 1
        found |= ks

    if not found:
        print("没有从任何文件里读到键 —— 先确认本机有 GGUF 模型", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# 由 tools/extract_metadata_keys.py 生成，勿手改\n")
        for k in sorted(found):
            f.write(k + "\n")
    print(f"扫了 {scanned} 个 GGUF 文件，已写入 {OUT}：{len(found)} 个键")
    return 0


if __name__ == "__main__":
    sys.exit(main())
