#!/usr/bin/env python3
"""从本机 Ollama 的 GGUF 里抽出真实的量化块，存成 internal/decode 的测试样本。

## 为什么需要这个脚本

Q8_0 的编码器能对着 `gguf.quants` 逐位验，但 K 系列在那个包里是
`NotImplementedError` —— 没有参考实现可对。

真实文件本身可以当参照：一个真实块里**带着它自己的 d/dmin/子 scale**。
把块解码成数值，再喂回编码器的「码与打包」那一段（不重算 scale），
字节必须与原块完全相同。这条路绕开了 scale 搜索
（真实文件可能用了重要性矩阵加权，等权编码复现不了），
单独钉住码与打包这两段。

## 用法

    uv run --with gguf --with numpy python3 tools/extract_real_blocks.py

输出 internal/decode/testdata/real_blocks.json，入库。

## 为什么抽样本而不是读整个模型

- `internal/decode` 不能依赖 `internal/parser`（会形成反向依赖）
- 让编码器的测试依赖"本机装了某个 17 GB 的模型"太脆

样本一次抽出、入库，之后离线可跑。
"""

from __future__ import annotations

import json
import os
import sys

from gguf import GGUFReader, quants

BLOB = os.path.expanduser(
    "~/.ollama/models/blobs/"
    "sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6"
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "internal", "decode", "testdata", "real_blocks.json")
N_BLOCKS = 40


def blocks_of(reader: GGUFReader, want: str, n: int, start: int = 0):
    """找第一个 want 类型的张量，取它从 start 起的 n 个块的原始字节。

    张量名与起点一起写进产物 —— 只记 blob 文件名的话，这份样本的来历
    就不可考了（哪个张量、从第几块开始都无从知晓）。
    """
    q = getattr(quants, want)
    for t in reader.tensors:
        if t.tensor_type.name != want:
            continue
        raw = t.data.tobytes()
        size = q.type_size
        total = len(raw) // size
        if total < start + n:
            raise SystemExit(f"{t.name} 只有 {total} 个块，不足 {start + n}")
        return {
            "tensor": t.name,
            "start_block": start,
            "block_elems": q.block_size,
            "block_bytes": size,
            "blocks": [raw[i * size:(i + 1) * size].hex()
                       for i in range(start, start + n)],
        }
    raise SystemExit(f"这个文件里没有 {want} 张量")


def main() -> int:
    if not os.path.exists(BLOB):
        print(f"找不到 {BLOB}", file=sys.stderr)
        return 1
    reader = GGUFReader(BLOB)

    out: dict[str, object] = {"source": os.path.basename(BLOB)}
    for want in ("Q6_K", "Q4_K"):
        one = blocks_of(reader, want, N_BLOCKS)
        out[want] = one
        print(f"{want}: {one['tensor']} 的第 {one['start_block']} 起 "
              f"{len(one['blocks'])} 块 × {one['block_bytes']} 字节，"
              f"每块 {one['block_elems']} 个权重")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"已写入 {OUT}（{os.path.getsize(OUT)} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
