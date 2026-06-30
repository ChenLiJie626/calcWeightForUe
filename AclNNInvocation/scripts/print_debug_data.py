#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


FEATURES = 256


def load_float(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def print_rows(name: str, matrix: np.ndarray, rows: int, cols: int) -> None:
    print(f"{name} first {rows} rows, first {cols} cols:")
    print(np.array2string(matrix[:rows, :cols], precision=6, suppress_small=False, max_line_width=180))


def main() -> int:
    parser = argparse.ArgumentParser(description="Print small CalcWeightForUe input/output data.")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--entries", type=int, default=4)
    args = parser.parse_args()

    base = Path(".")
    lens = np.fromfile(base / "input/input_lens.bin", dtype=np.uint32)
    flag = np.fromfile(base / "input/input_flag.bin", dtype=np.uint32)
    idx_count = np.fromfile(base / "input/input_idx_count.bin", dtype=np.uint32)
    if idx_count.shape != (1,):
        print("input/input_idx_count.bin must contain exactly one uint32.")
        return 1
    index_count = int(idx_count[0])
    if index_count > lens.shape[0] or index_count > flag.shape[0]:
        print(f"idxCount={index_count} exceeds lens/flag entries: lens={lens.shape[0]}, flag={flag.shape[0]}")
        return 1
    active_lens = lens[:index_count]
    active_flag = flag[:index_count]
    total_entries = int(np.sum(active_lens))
    if index_count == 0 or total_entries == 0:
        print("No input data found. Run scripts/gen_data.py first.")
        return 1

    weight_r = load_float(base / "input/input_weight_r.bin", (total_entries, FEATURES))
    weight_i = load_float(base / "input/input_weight_i.bin", (total_entries, FEATURES))
    golden_r = load_float(base / "output/golden_weightout_r.bin", (total_entries, FEATURES))
    golden_i = load_float(base / "output/golden_weightout_i.bin", (total_entries, FEATURES))
    actual_r = load_float(base / "output/output_weightout_r.bin", (total_entries, FEATURES))
    actual_i = load_float(base / "output/output_weightout_i.bin", (total_entries, FEATURES))

    np.set_printoptions(precision=6, suppress=False, linewidth=180)
    print(f"idxCount={index_count}, totalRankEntries={total_entries}, lens={active_lens.tolist()}")
    print(f"flag={active_flag.tolist()}")
    print()

    pos = 0
    printed = 0
    for index, cur_len in enumerate(active_lens):
        ranks = int(cur_len)
        if ranks == 0:
            continue
        next_pos = pos + ranks
        if printed < args.entries:
            print(f"=== index {index} (rankRows={ranks}, flag={int(active_flag[index])}, pos={pos}) ===")
            rows = min(args.rows, ranks)
            print_rows("input weight_r", weight_r[pos:next_pos], rows, args.cols)
            print_rows("input weight_i", weight_i[pos:next_pos], rows, args.cols)
            norm = weight_r[pos:next_pos] * weight_r[pos:next_pos] + weight_i[pos:next_pos] * weight_i[pos:next_pos]
            print(f"weightVec=sum(norm, axis=1): {np.array2string(norm.sum(axis=1), precision=6, max_line_width=180)}")
            print_rows("golden_r", golden_r[pos:next_pos], rows, args.cols)
            print_rows("actual_r", actual_r[pos:next_pos], rows, args.cols)
            print_rows("golden_i", golden_i[pos:next_pos], rows, args.cols)
            print_rows("actual_i", actual_i[pos:next_pos], rows, args.cols)
            max_r = float(np.max(np.abs(actual_r[pos:next_pos] - golden_r[pos:next_pos])))
            max_i = float(np.max(np.abs(actual_i[pos:next_pos] - golden_i[pos:next_pos])))
            print(f"index max_abs_diff: real={max_r:.8g}, imag={max_i:.8g}")
            print()
            printed += 1
        pos = next_pos

    all_real = float(np.max(np.abs(actual_r - golden_r)))
    all_imag = float(np.max(np.abs(actual_i - golden_i)))
    print(f"overall max_abs_diff: real={all_real:.8g}, imag={all_imag:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
