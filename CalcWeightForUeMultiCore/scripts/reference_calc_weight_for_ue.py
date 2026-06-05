#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


ROWS = 384
RANKS_PER_INDEX = 8


def shuffle_valid_columns(x: np.ndarray, shift: int, sum_rank: int) -> np.ndarray:
    out = x.copy()
    valid_cols = min(max(int(sum_rank), 0), RANKS_PER_INDEX)
    if valid_cols == 0:
        return out
    shift %= valid_cols
    if shift == 0:
        out[:, :valid_cols] = x[:, :valid_cols]
    else:
        out[:, :valid_cols] = np.concatenate([x[:, shift:valid_cols], x[:, :shift]], axis=1)
    return out


def calc_weight_for_ue(weight_r: np.ndarray,
                       weight_i: np.ndarray,
                       lens: np.ndarray,
                       getuser_id_rank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    index_count = weight_r.shape[0]
    if weight_r.shape != (index_count, ROWS, RANKS_PER_INDEX):
        raise ValueError(f"weight_r shape must be [indexCount, {ROWS}, {RANKS_PER_INDEX}]")
    if weight_i.shape != weight_r.shape:
        raise ValueError("weight_i shape must match weight_r")
    if lens.shape != (index_count,):
        raise ValueError("lens shape must be [indexCount]")

    total_entries = int(getuser_id_rank.shape[0])
    out_r = np.zeros((total_entries, ROWS, RANKS_PER_INDEX), dtype=np.float32)
    out_i = np.zeros_like(out_r)

    pos = 0
    for index in range(index_count):
        current_len = int(max(lens[index], 0))
        ranks = getuser_id_rank[pos:pos + current_len].astype(np.int64)
        if current_len == 0:
            continue

        norm = weight_r[index] * weight_r[index] + weight_i[index] * weight_i[index]
        weight_vec = norm.sum(axis=0)
        weight_vec = np.where(weight_vec > 0, weight_vec, 1.0)
        sum_rank = int(np.maximum(ranks, 0).sum())
        if sum_rank == 0:
            pos += current_len
            continue

        scale = (1.0 / np.sqrt(np.float32(sum_rank))) / np.sqrt(weight_vec)
        tmp_r = weight_r[index] * scale
        tmp_i = weight_i[index] * scale

        shuffle_pos = 0
        for k, rank in enumerate(ranks):
            out_r[pos + k] = shuffle_valid_columns(tmp_r, shuffle_pos, sum_rank)
            out_i[pos + k] = shuffle_valid_columns(tmp_i, shuffle_pos, sum_rank)
            shuffle_pos += int(max(rank, 0))
        pos += current_len
    return out_r, out_i


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU reference for CalcWeightForUe.")
    parser.add_argument("input_npz", type=Path)
    parser.add_argument("output_npz", type=Path)
    args = parser.parse_args()

    data = np.load(args.input_npz)
    out_r, out_i = calc_weight_for_ue(
        data["weight_r"].astype(np.float32),
        data["weight_i"].astype(np.float32),
        data["lens"].astype(np.int32),
        data["getuserIdRank"].astype(np.int32),
    )
    np.savez(args.output_npz, weightout_r=out_r, weightout_i=out_i)


if __name__ == "__main__":
    main()
