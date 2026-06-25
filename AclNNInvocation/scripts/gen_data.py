#!/usr/bin/env python3
import os

import numpy as np


FEATURES = 256
DEFAULT_INDEX_COUNT = 4
DEFAULT_LENS = np.array([3, 2, 1, 4], dtype=np.int32)
DEFAULT_FLAG = np.array([1, 0, 1, 1], dtype=np.int32)


def scenario_name() -> str:
    return os.environ.get("CALC_WEIGHT_SCENARIO", "default")


def default_index_count() -> int:
    if scenario_name() == "printable":
        return 2
    return DEFAULT_INDEX_COUNT


def build_lens(index_count: int) -> np.ndarray:
    scenario = scenario_name()
    if scenario == "printable":
        return np.resize(np.array([2, 3], dtype=np.int32), index_count).astype(np.int32)
    if scenario == "avg2":
        base = np.array([1, 2, 3, 2], dtype=np.int32)
        return np.resize(base, index_count).astype(np.int32)
    if index_count == DEFAULT_INDEX_COUNT:
        return DEFAULT_LENS.copy()
    return np.full((index_count,), 2, dtype=np.int32)


def build_flag(index_count: int) -> np.ndarray:
    scenario = scenario_name()
    if scenario == "printable":
        return np.resize(np.array([1, 0], dtype=np.int32), index_count).astype(np.int32)
    if scenario == "avg2":
        base = np.array([1, 1, 0, 1], dtype=np.int32)
        return np.resize(base, index_count).astype(np.int32)
    if index_count == DEFAULT_INDEX_COUNT:
        return DEFAULT_FLAG.copy()
    return np.ones((index_count,), dtype=np.int32)


INDEX_COUNT = int(os.environ.get("INDEX_COUNT", default_index_count()))
LENS = build_lens(INDEX_COUNT)
FLAG = build_flag(INDEX_COUNT)
TOTAL_RANK_ENTRIES = int(np.sum(np.maximum(LENS, 0)))


def build_golden(weight_r: np.ndarray, weight_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out_r = np.zeros_like(weight_r)
    out_i = np.zeros_like(weight_i)

    pos = 0
    for index in range(INDEX_COUNT):
        ranks = int(max(LENS[index], 0))
        if ranks == 0:
            continue

        next_pos = pos + ranks
        cur_r = weight_r[pos:next_pos]
        cur_i = weight_i[pos:next_pos]
        if FLAG[index] == 0:
            out_r[pos:next_pos] = cur_r
            out_i[pos:next_pos] = cur_i
            pos = next_pos
            continue

        norm = cur_r * cur_r + cur_i * cur_i
        weight_vec = np.sum(norm, axis=1)
        weight_vec = np.where(weight_vec > 0.0, weight_vec, np.float32(1.0)).astype(np.float32)
        scale = (np.float32(1.0) / np.sqrt(np.float32(ranks))) / np.sqrt(weight_vec)
        out_r[pos:next_pos] = cur_r * scale.reshape(ranks, 1)
        out_i[pos:next_pos] = cur_i * scale.reshape(ranks, 1)
        pos = next_pos

    return out_r.astype(np.float32), out_i.astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(20260604)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    shape = (TOTAL_RANK_ENTRIES, FEATURES)
    if scenario_name() == "printable":
        rank = np.arange(TOTAL_RANK_ENTRIES, dtype=np.float32).reshape(TOTAL_RANK_ENTRIES, 1)
        col = np.arange(FEATURES, dtype=np.float32).reshape(1, FEATURES)
        weight_r = (1.0 + rank * 0.2 + col * 0.001).astype(np.float32)
        weight_i = (-0.3 + rank * 0.03 - col * 0.0005).astype(np.float32)
    else:
        weight_r = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        weight_i = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)

    weight_r.tofile("input/input_weight_r.bin")
    weight_i.tofile("input/input_weight_i.bin")
    LENS.tofile("input/input_lens.bin")
    FLAG.tofile("input/input_flag.bin")

    golden_r, golden_i = build_golden(weight_r, weight_i)
    golden_r.tofile("output/golden_weightout_r.bin")
    golden_i.tofile("output/golden_weightout_i.bin")

    print(
        "Generate input and golden data success. "
        f"indexCount={INDEX_COUNT}, totalRankEntries={TOTAL_RANK_ENTRIES}, "
        f"features={FEATURES}, lensAvg={float(np.mean(LENS)):.6g}, "
        f"lensMin={int(np.min(LENS))}, lensMax={int(np.max(LENS))}, "
        f"flagOn={int(np.count_nonzero(FLAG))}, flagOff={int(FLAG.shape[0] - np.count_nonzero(FLAG))}"
    )


if __name__ == "__main__":
    main()
