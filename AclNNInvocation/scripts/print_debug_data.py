#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


ROWS = 384
RANKS = 8


def load_float(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def print_matrix(name: str, matrix: np.ndarray, rows: int) -> None:
    print(f"{name} first {rows} rows:")
    print(np.array2string(matrix[:rows], precision=6, suppress_small=False, max_line_width=180))


def main() -> int:
    parser = argparse.ArgumentParser(description="Print small CalcWeightForUe input/output data.")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--entries", type=int, default=8)
    args = parser.parse_args()

    base = Path(".")
    lens = np.fromfile(base / "input/input_lens.bin", dtype=np.int32)
    ranks = np.fromfile(base / "input/input_getuser_id_rank.bin", dtype=np.int32)
    index_count = int(lens.shape[0])
    total_entries = int(ranks.shape[0])
    if index_count == 0 or total_entries == 0:
        print("No input data found. Run scripts/gen_data.py first.")
        return 1

    weight_r = load_float(base / "input/input_weight_r.bin", (index_count, ROWS, RANKS))
    weight_i = load_float(base / "input/input_weight_i.bin", (index_count, ROWS, RANKS))
    golden_r = load_float(base / "output/golden_weightout_r.bin", (total_entries, ROWS, RANKS))
    golden_i = load_float(base / "output/golden_weightout_i.bin", (total_entries, ROWS, RANKS))
    actual_r = load_float(base / "output/output_weightout_r.bin", (total_entries, ROWS, RANKS))
    actual_i = load_float(base / "output/output_weightout_i.bin", (total_entries, ROWS, RANKS))

    np.set_printoptions(precision=6, suppress=False, linewidth=180)
    print(f"indexCount={index_count}, lens={lens.tolist()}, totalRankEntries={total_entries}")
    print(f"getuserIdRank={ranks.tolist()}")
    print()

    for index in range(min(index_count, 2)):
        print(f"=== input index {index} ===")
        print_matrix("weight_r", weight_r[index], args.rows)
        print_matrix("weight_i", weight_i[index], args.rows)
        norm = weight_r[index] * weight_r[index] + weight_i[index] * weight_i[index]
        print(f"weightVec=sum(norm, axis=0): {np.array2string(norm.sum(axis=0), precision=6, max_line_width=180)}")
        print()

    pos = 0
    for index, cur_len in enumerate(lens):
        shuffle_pos = 0
        for k in range(int(cur_len)):
            entry = pos + k
            if entry >= args.entries:
                break
            rank = int(ranks[entry])
            print(f"=== output entry {entry} (index={index}, k={k}, rank={rank}, shufflePos={shuffle_pos}) ===")
            print_matrix("golden_r", golden_r[entry], args.rows)
            print_matrix("actual_r", actual_r[entry], args.rows)
            print_matrix("golden_i", golden_i[entry], args.rows)
            print_matrix("actual_i", actual_i[entry], args.rows)
            max_r = float(np.max(np.abs(actual_r[entry] - golden_r[entry])))
            max_i = float(np.max(np.abs(actual_i[entry] - golden_i[entry])))
            print(f"entry max_abs_diff: real={max_r:.8g}, imag={max_i:.8g}")
            print()
            shuffle_pos += rank
        pos += int(cur_len)
        if pos >= args.entries:
            break

    all_real = float(np.max(np.abs(actual_r - golden_r)))
    all_imag = float(np.max(np.abs(actual_i - golden_i)))
    print(f"overall max_abs_diff: real={all_real:.8g}, imag={all_imag:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
