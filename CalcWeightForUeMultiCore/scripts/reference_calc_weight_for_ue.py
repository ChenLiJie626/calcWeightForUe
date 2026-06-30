#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


FEATURES = 256


def calc_weight_for_ue(weight_r: np.ndarray,
                       weight_i: np.ndarray,
                       lens: np.ndarray,
                       flag: np.ndarray,
                       idx_count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if weight_r.ndim != 2 or weight_r.shape[1] != FEATURES:
        raise ValueError(f"weight_r shape must be [totalRankEntries, {FEATURES}]")
    if weight_i.shape != weight_r.shape:
        raise ValueError("weight_i shape must match weight_r")
    if lens.ndim != 1:
        raise ValueError("lens shape must be [indexCount]")
    if flag.shape != lens.shape:
        raise ValueError("flag shape must match lens")
    if idx_count.shape != (1,):
        raise ValueError("idx_count shape must be [1]")
    index_count = int(idx_count[0])
    if index_count > lens.shape[0]:
        raise ValueError("idx_count must be no larger than lens length")

    out_r = np.zeros_like(weight_r, dtype=np.float32)
    out_i = np.zeros_like(weight_i, dtype=np.float32)

    pos = 0
    for index in range(index_count):
        ranks = int(lens[index])
        if ranks == 0:
            continue
        next_pos = pos + ranks
        cur_r = weight_r[pos:next_pos]
        cur_i = weight_i[pos:next_pos]
        if flag[index] == 0:
            out_r[pos:next_pos] = cur_r
            out_i[pos:next_pos] = cur_i
            pos = next_pos
            continue

        norm = cur_r * cur_r + cur_i * cur_i
        weight_vec = norm.sum(axis=1)
        weight_vec = np.where(weight_vec > 0.0, weight_vec, np.float32(1.0)).astype(np.float32)
        scale = (np.float32(1.0) / np.sqrt(np.float32(ranks))) / np.sqrt(weight_vec)
        out_r[pos:next_pos] = cur_r * scale.reshape(ranks, 1)
        out_i[pos:next_pos] = cur_i * scale.reshape(ranks, 1)
        pos = next_pos

    return out_r.astype(np.float32), out_i.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU reference for CalcWeightForUe.")
    parser.add_argument("input_npz", type=Path)
    parser.add_argument("output_npz", type=Path)
    args = parser.parse_args()

    data = np.load(args.input_npz)
    out_r, out_i = calc_weight_for_ue(
        data["weight_r"].astype(np.float32),
        data["weight_i"].astype(np.float32),
        data["lens"].astype(np.uint32),
        data["flag"].astype(np.uint32),
        data["idxCount"].astype(np.uint32),
    )
    np.savez(args.output_npz, weightout_r=out_r, weightout_i=out_i)


if __name__ == "__main__":
    main()
