#!/usr/bin/env python3
import os

import numpy as np


ROWS = 384
RANKS = 8
DEFAULT_INDEX_COUNT = 4
DEFAULT_LENS = np.array([3, 2, 1, 4], dtype=np.int32)
DEFAULT_GET_USER_ID_RANK = np.array([1, 2, 3, 4, 2, 8, 1, 1, 2, 4], dtype=np.int32)


def build_lens(index_count: int) -> np.ndarray:
    scenario = os.environ.get("CALC_WEIGHT_SCENARIO", "default")
    if scenario == "avg2":
        base = np.array([1, 2, 3, 2], dtype=np.int32)
        return np.resize(base, index_count).astype(np.int32)
    if index_count == DEFAULT_INDEX_COUNT:
        return DEFAULT_LENS.copy()
    return np.full((index_count,), 2, dtype=np.int32)


def build_ranks(lens: np.ndarray) -> np.ndarray:
    scenario = os.environ.get("CALC_WEIGHT_SCENARIO", "default")
    if scenario == "default" and len(lens) == DEFAULT_INDEX_COUNT:
        return DEFAULT_GET_USER_ID_RANK.copy()

    ranks = []
    for index, cur_len in enumerate(lens):
        cur_len = int(cur_len)
        if cur_len == 1:
            current = [8]
        elif cur_len == 2:
            current = [1 + (index % 3), 3 + (index % 2)]
        elif cur_len == 3:
            current = [1, 2 + (index % 2), 3]
        else:
            current = [1] * cur_len
        if sum(current) > RANKS:
            raise ValueError(f"rank sum exceeds {RANKS}: index={index}, ranks={current}")
        ranks.extend(current)
    return np.array(ranks, dtype=np.int32)


INDEX_COUNT = int(os.environ.get("INDEX_COUNT", DEFAULT_INDEX_COUNT))
LENS = build_lens(INDEX_COUNT)
GET_USER_ID_RANK = build_ranks(LENS)
TOTAL_RANK_ENTRIES = int(np.sum(LENS))


def rotate_left_columns(weight: np.ndarray, shift: int) -> np.ndarray:
    shift %= RANKS
    if shift == 0:
        return weight.copy()
    return np.concatenate((weight[:, shift:], weight[:, :shift]), axis=1)


def column_norm_sum(weight_r: np.ndarray, weight_i: np.ndarray) -> np.ndarray:
    out = np.zeros((RANKS,), dtype=np.float32)
    norm = weight_r * weight_r + weight_i * weight_i
    for row in range(ROWS):
        out += norm[row]
    return np.where(out > 0.0, out, np.float32(1.0)).astype(np.float32)


def build_golden(weight_r: np.ndarray, weight_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out_r = np.zeros((TOTAL_RANK_ENTRIES, ROWS, RANKS), dtype=np.float32)
    out_i = np.zeros_like(out_r)
    pos = 0
    for index in range(INDEX_COUNT):
        cur_len = int(LENS[index])
        ranks = GET_USER_ID_RANK[pos:pos + cur_len]
        sum_rank = int(np.sum(np.maximum(ranks, 0)))
        if sum_rank == 0:
            pos += cur_len
            continue

        weight_vec = column_norm_sum(weight_r[index], weight_i[index])
        scale = (np.float32(1.0) / np.sqrt(np.float32(sum_rank))) / np.sqrt(weight_vec)
        weight_tmp_r = weight_r[index] * scale
        weight_tmp_i = weight_i[index] * scale

        shuffle_pos = 0
        for k, rank in enumerate(ranks):
            out_r[pos + k] = rotate_left_columns(weight_tmp_r, shuffle_pos)
            out_i[pos + k] = rotate_left_columns(weight_tmp_i, shuffle_pos)
            shuffle_pos += int(max(rank, 0))
        pos += cur_len
    return out_r, out_i


def main() -> None:
    rng = np.random.default_rng(20260604)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    shape = (INDEX_COUNT, ROWS, RANKS)
    weight_r = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    weight_i = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)

    weight_r.tofile("input/input_weight_r.bin")
    weight_i.tofile("input/input_weight_i.bin")
    LENS.tofile("input/input_lens.bin")
    GET_USER_ID_RANK.tofile("input/input_getuser_id_rank.bin")

    golden_r, golden_i = build_golden(weight_r, weight_i)
    golden_r.tofile("output/golden_weightout_r.bin")
    golden_i.tofile("output/golden_weightout_i.bin")

    rank_sums = []
    pos = 0
    for cur_len in LENS:
        ranks = GET_USER_ID_RANK[pos:pos + int(cur_len)]
        rank_sums.append(int(np.sum(ranks)))
        pos += int(cur_len)
    print(
        "Generate input and golden data success. "
        f"indexCount={INDEX_COUNT}, totalRankEntries={TOTAL_RANK_ENTRIES}, "
        f"lensAvg={float(np.mean(LENS)):.6g}, lensMin={int(np.min(LENS))}, lensMax={int(np.max(LENS))}, "
        f"rankSumMin={min(rank_sums)}, rankSumMax={max(rank_sums)}, firstRankSums={rank_sums[:8]}"
    )


if __name__ == "__main__":
    main()
