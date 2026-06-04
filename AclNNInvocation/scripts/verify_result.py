#!/usr/bin/env python3
import sys
from typing import Tuple

import numpy as np


TOTAL_RANK_ENTRIES = 10
ROWS = 384
RANKS = 8


def check(name: str, actual_path: str, golden_path: str, shape: Tuple[int, int, int],
          atol: float, rtol: float) -> bool:
    actual = np.fromfile(actual_path, dtype=np.float32).reshape(shape)
    golden = np.fromfile(golden_path, dtype=np.float32).reshape(shape)

    diff = np.abs(actual - golden)
    max_abs = float(np.max(diff))
    max_rel = float(np.max(diff / np.maximum(np.abs(golden), 1.0)))
    ok = np.allclose(actual, golden, atol=atol, rtol=rtol)
    print(f"{name}: {'PASS' if ok else 'FAIL'}, max_abs={max_abs:.6g}, max_rel={max_rel:.6g}")
    if not ok:
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        print(f"  worst index={idx}, actual={actual[idx]}, golden={golden[idx]}")
    return ok


def main() -> int:
    shape = (TOTAL_RANK_ENTRIES, ROWS, RANKS)
    weightout_r_ok = check(
        "weightout_r",
        "output/output_weightout_r.bin",
        "output/golden_weightout_r.bin",
        shape,
        atol=5e-4,
        rtol=5e-4,
    )
    weightout_i_ok = check(
        "weightout_i",
        "output/output_weightout_i.bin",
        "output/golden_weightout_i.bin",
        shape,
        atol=5e-4,
        rtol=5e-4,
    )
    return 0 if weightout_r_ok and weightout_i_ok else 1


if __name__ == "__main__":
    sys.exit(main())
