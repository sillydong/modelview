#!/usr/bin/env python3
"""用 llama.cpp 官方的 gguf 包生成编码器的真值。

## 为什么需要这个脚本

`internal/decode/quantize.go` 把 Q8_0/Q6_K/Q4_K 的真实编码算法移植了过来。
移植对不对，不能靠自己再写一遍来验 —— 那是同一份理解写两遍。

Q8_0 是这三档里唯一**参考实现也实现了编码器**的类型，所以它是
**编码器实现本身的验证锚点**：舍入模式（roundf，平局远离零）、
scale 存 fp16、码用的是未舍入的 1/d —— 这些细节验过之后，
Q6_K/Q4_K 只是换一套块结构与 scale 搜索，那部分由真实块样本
（tools/extract_real_blocks.py）钉住。

    uv run --with gguf --with numpy python3 tools/verify_gguf_quant_sim.py

输出 internal/decode/testdata/quant_sim.json，入库。

## 覆盖边界

`gguf.quants` 实现了 quantize 的只有 Q4_0/Q4_1/Q5_0/Q5_1/Q8_0，
K 系列（Q2_K/Q3_K/Q4_K/Q5_K/Q6_K）全是 NotImplementedError。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from gguf import quants

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "internal", "decode", "testdata", "quant_sim.json")
SEED = 20260926
CASES = 200
BLOCK = 32


def main() -> int:
    rng = np.random.default_rng(SEED)
    cases = []

    for _ in range(CASES):
        n_blocks = int(rng.integers(1, 8))
        scale = float(rng.choice([1e-3, 0.05, 1.0, 100.0]))
        x = (rng.standard_normal((n_blocks, BLOCK)) * scale).astype(np.float32)
        if rng.random() < 0.15:
            x[0] = 0.0          # 全零块：走 amax==0 那条分支
        raw = quants.Q8_0.quantize(x)
        cases.append({
            "input": [float(v) for v in x.ravel()],
            # 真值 = 参考实现编出来的**字节**，不是反量化的值 ——
            # 比对字节才能验证编码器，比对值只能验证编解码这一对自洽
            "bytes": raw.tobytes().hex(),
        })

    # 手工边界 1：**舍入平局**。
    #
    # 随机数据几乎不可能让 v*(127/amax) 正好落在 k+0.5 上，于是
    # roundf（平局远离零）与 round-to-even 在随机用例里给出完全相同的结果 ——
    # 变异验证时把 math.Round 换成 math.RoundToEven，随机用例 200/200 照样过。
    # 这两组是专门造出来的平局。
    #
    # amax = 127 → d = 1.0、id = 1.0（都是精确值），v 取 ±k.5 即为平局
    x = np.zeros((1, BLOCK), dtype=np.float32)
    x[0, 0] = np.float32(127.0)
    for k in range(1, 8):
        x[0, k] = np.float32(k + 0.5)
        x[0, 15 + k] = np.float32(-(k + 0.5))
    raw = quants.Q8_0.quantize(x)
    cases.append({"input": [float(v) for v in x.ravel()],
                  "bytes": raw.tobytes().hex()})

    # amax = 254 → d = 2.0、id = 0.5，奇数整数即平局
    x = np.zeros((1, BLOCK), dtype=np.float32)
    x[0, 0] = np.float32(254.0)
    for k in range(1, 16):
        x[0, k] = np.float32(2 * k - 1)
    raw = quants.Q8_0.quantize(x)
    cases.append({"input": [float(v) for v in x.ravel()],
                  "bytes": raw.tobytes().hex()})

    # 手工边界 2：全零块。参考实现里 d 记 0、码全 0，本实现必须一致 ——
    # 只比"解码出来的值"是不够的：d=0 时码取什么值解出来都是 0，
    # 断言数值的话，把码写坏也发现不了。
    x = np.zeros((1, BLOCK), dtype=np.float32)
    raw = quants.Q8_0.quantize(x)
    cases.append({"input": [float(v) for v in x.ravel()],
                  "bytes": raw.tobytes().hex()})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"block": BLOCK, "dtype": "Q8_0", "cases": cases},
                  f, separators=(",", ":"))

    total = sum(len(c["input"]) for c in cases)
    print(f"已写入 {OUT}")
    print(f"  {len(cases)} 组，共 {total} 个值")
    return 0


if __name__ == "__main__":
    sys.exit(main())
