# CalcWeightForUe

AscendC AIV custom-operator project for `CalcWeightForUe`.

Inputs:

- `weight_r`, `weight_i`: `float`, shape `[totalRankEntries, 256]`.
- `lens`: `int32`, shape `[indexCount]`. `lens[i]` is the rank-row count for index `i`.
- `flag`: `int32`, shape `[indexCount]`. `flag[i] != 0` enables normalization for that `[lens[i], 256]` segment; `flag[i] == 0` copies it unchanged.

Outputs:

- `weightout_r`, `weightout_i`: `float`, shape `[totalRankEntries, 256]`.

For each index `i`:

```text
pos = sum(max(lens[:i], 0))
ranks = max(lens[i], 0)
weight = weight[pos:pos + ranks, :]

if flag[i] == 0:
    weightout[pos:pos + ranks, :] = weight
else:
    norm[row, col] = weight_r[row, col]^2 + weight_i[row, col]^2
    weightVec[row] = sum(norm[row, :])
    scale[row] = 1 / sqrt(ranks) / sqrt(weightVec[row])
    weightout_r[row, :] = weight_r[row, :] * scale[row]
    weightout_i[row, :] = weight_i[row, :] * scale[row]
```

Implementation notes:

- The kernel runs on vector cores (`AIV`); the `__DAV_CUBE__` path returns immediately.
- Each rank row is processed as one `[256]` vector in UB.
- Elementwise norm and scaling use AscendC vector APIs: `Mul`, `Add`, `Duplicate`, and `Rsqrt`.
- The row reduction to `weightVec[row]` is done in UB with `LocalTensor::GetValue`.
- `weightVec[row] <= 0` is clamped to `1` before reciprocal square root so zero rows produce zero output instead of NaN.

Build:

```bash
./build.sh
```

Install the generated `build_out/custom_opp_<os>_<arch>.run` package before invoking the operator.
