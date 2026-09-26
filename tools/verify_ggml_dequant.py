#!/usr/bin/env python3
"""用 llama.cpp 官方的 gguf 包生成量化块的解码真值。

量化块的内部布局背不出来也推不出来 —— 这个脚本用**参考实现**算出真值，
再由 tools/compare_dequant.go 与 internal/decode 的回归测试拿去比对。

参考实现是 gguf.quants（llama.cpp 官方维护），本机通过 uv 取用：

    uv run --with gguf --with numpy python3 tools/verify_ggml_dequant.py

输出 internal/decode/testdata/vectors.json。**该文件入库**，
这样日常 `go test` 既不需要网络也不需要 Python。

为什么随机字节也能当真值：gguf.quants 是纯函数，喂什么字节都行。
所以本地语料里没有的类型（Q2_K / Q3_K / Q5_K / Q5_1）同样能被独立验证，
不需要为它们去找模型文件。

用法：
    uv run --with gguf --with numpy python3 tools/verify_ggml_dequant.py
    python3 tools/verify_ggml_dequant.py --check   # 只校验已入库的向量（不需要 gguf 包）
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 各类型里 fp16 scale 字段的偏移。随机字节会让 fp16 出现 inf/nan，
# 把这几处钉成 0.0625（位模式 0x2C00）才得到有限值。
SCALE_OFF = {
    "Q4_0": [0], "Q4_1": [0, 2], "Q5_0": [0], "Q5_1": [0, 2], "Q8_0": [0],
    "Q2_K": [80, 82], "Q3_K": [108], "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208],
}
HALF_0_0625 = (0x00, 0x2C)
BLOCKS_PER_TYPE = 3
SEED = 20260926

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "internal", "decode", "testdata", "vectors.json")


def check() -> int:
    """校验入库的向量文件自洽（不需要 gguf 包）。"""
    try:
        data = json.load(open(OUT, encoding="utf-8"))
    except FileNotFoundError:
        print(f"✗ {OUT} 不存在 —— 用 uv run --with gguf --with numpy 生成")
        return 1
    except json.JSONDecodeError as e:
        print(f"✗ {OUT} 不是合法 JSON: {e}")
        return 1

    if not data:
        # 空集不是成功：什么都没验证却报 0，是最危险的一种"通过"
        print(f"✗ {OUT} 里一个类型都没有 —— 什么都没验证")
        return 1

    bad = 0
    for name, v in sorted(data.items()):
        want_len = v["block_size"] * BLOCKS_PER_TYPE
        if len(v["expect"]) != want_len:
            print(f"✗ {name}: 真值 {len(v['expect'])} 个，应为 {want_len}")
            bad += 1
        if len(bytes.fromhex(v["bytes"])) != v["type_size"] * BLOCKS_PER_TYPE:
            print(f"✗ {name}: 字节数不对")
            bad += 1
    print(f"{len(data) - bad}/{len(data)} 种类型的向量自洽")
    return 1 if bad else 0


def generate() -> int:
    import numpy as np
    from gguf import quants

    rng = np.random.default_rng(SEED)
    out = {}
    for name, offs in SCALE_OFF.items():
        cls = getattr(quants, name, None)
        if cls is None:
            print(f"✗ 参考实现里没有 {name} —— gguf 包版本可能变了")
            return 1
        bs, ts = cls.block_size, cls.type_size
        arr = rng.integers(0, 256, size=BLOCKS_PER_TYPE * ts,
                           dtype=np.uint8).reshape(BLOCKS_PER_TYPE, ts).copy()
        for i in range(BLOCKS_PER_TYPE):
            for off in offs:
                arr[i, off:off + 2] = HALF_0_0625

        y = cls.dequantize(arr).astype(np.float32).ravel()
        if not np.isfinite(y).all():
            print(f"✗ {name}: 真值里出现 inf/nan，检查 SCALE_OFF")
            return 1
        out[name] = {
            "block_size": int(bs),
            "type_size": int(ts),
            "bytes": arr.tobytes().hex(),
            "expect": [float(v) for v in y],
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)

    print(f"已写入 {OUT}")
    for name, v in sorted(out.items()):
        derived = v["type_size"] * 8 / v["block_size"]
        print(f"  {name:6} {v['block_size']:>3} 元素 / {v['type_size']:>3} 字节 "
              f"（{derived:.4g} bit/权重）× {BLOCKS_PER_TYPE} 块")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只校验已入库的向量自洽，不需要 gguf 包")
    args = ap.parse_args()
    return check() if args.check else generate()


if __name__ == "__main__":
    sys.exit(main())
