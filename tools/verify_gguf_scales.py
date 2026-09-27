#!/usr/bin/env python3
"""用**独立来源**验证 `decode.Scales` 的块头 scale 提取。

## 为什么需要它

`Scales` 从块头取出每个子块的 scale/min。已有的两类测试都不足以验它：

- 合成夹具是同一份理解写的第二遍，抄错布局时两边一起错
- 真实文件那条只检查"解码值落在 (scale, min) 允许的区间内"，
  而 `Scales` 与 `Decode` **共用** `getScaleMinK4` —— 它错了两者一起错，
  区间约束照样成立

这里换一条完全不同的路：**从解码出来的数值反推步长**。

子块内的值是等间隔的格点：
  - 对称量化（Q8_0/Q5_0/Q6_K）：value = scale × q
  - 带 min（Q4_K）：value = scale × q − min

所以「子块内相邻两个不同值的最小间隔」就是 |scale| 的整数倍 ——
这个量由 `gguf.quants`（llama.cpp 官方绑定）解码得到，
与 `Scales` 读块头的方式没有任何共同代码。

## 用法

    uv run --with gguf --with numpy python3 tools/verify_gguf_scales.py

输出 internal/decode/testdata/real_scales.json，入库。
Go 侧 internal/decode/scales_test.go 读它做断言。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from gguf import GGUFReader, quants

BLOB = os.path.expanduser(
    "~/.ollama/models/blobs/"
    "sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6"
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "internal", "decode", "testdata", "real_scales.json")
N_BLOCKS = 24

# 每种格式一个子块覆盖多少权重，以及它是否是带 min 的非对称量化
SUB = {
    "Q6_K": (16, False),
    "Q4_K": (32, True),
}


def min_gap(vals: np.ndarray) -> float | None:
    """子块内相邻两个**不同**值的最小间隔；不足两个不同值则返回 None。"""
    u = np.unique(vals)
    if len(u) < 2:
        return None
    return float(np.min(np.diff(u)))


def blocks_of(reader: GGUFReader, want: str, n: int):
    """取第一个 want 类型张量的前 n 块，连同独立解码出的数值。"""
    q = getattr(quants, want)
    for t in reader.tensors:
        if t.tensor_type.name != want:
            continue
        raw = t.data.tobytes()
        size, total = q.type_size, len(t.data.tobytes()) // q.type_size
        if total < n:
            raise SystemExit(f"{t.name} 只有 {total} 个块，不足 {n}")
        out = []
        for i in range(n):
            blk = np.frombuffer(raw[i * size:(i + 1) * size],
                                dtype=np.uint8).reshape(1, size)
            vals = np.asarray(q.dequantize(blk)).ravel()
            out.append(vals)
        return t.name, out
    raise SystemExit(f"这个文件里没有 {want} 张量")


def main() -> int:
    if not os.path.exists(BLOB):
        print(f"找不到 {BLOB}", file=sys.stderr)
        return 1
    reader = GGUFReader(BLOB)

    result: dict[str, object] = {"source": os.path.basename(BLOB)}
    for want, (sub_elems, asymmetric) in SUB.items():
        name, blocks = blocks_of(reader, want, N_BLOCKS)
        # 起点固定为 0：extract_real_blocks.py 也从第 0 块取，
        # 两边必须对齐，否则比的是不同批次的数据
        per_block = []
        for vals in blocks:
            gaps = []
            for j in range(0, len(vals), sub_elems):
                g = min_gap(vals[j:j + sub_elems])
                gaps.append(g)  # None = 该子块只有一个不同值（常量）
            per_block.append(gaps)
        result[want] = {
            "tensor": name,
            "start_block": 0,
            "sub_elems": sub_elems,
            "asymmetric": asymmetric,
            "gaps": per_block,
        }
        got = sum(1 for b in per_block for g in b if g is not None)
        print(f"{want}（{name}）：{len(blocks)} 块 × {len(per_block[0])} 个子块，"
              f"其中 {got} 个子块测到了最小间隔")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))
    print(f"已写入 {OUT}（{os.path.getsize(OUT)} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
