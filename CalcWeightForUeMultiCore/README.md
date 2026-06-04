# CalcWeightForUe

AscendC AIV custom-operator project for `CalcWeightForUe`.

Inputs:

- `weight_r`, `weight_i`: `float`, shape `[indexCount, 384, 8]`.
- `lens`: `int32`, shape `[indexCount]`. `lens[i]` is the number of output rank entries for index `i`.
- `getuserIdRank`: `int32`, shape `[sum(lens)]`. Each value advances the next shuffle position; each per-index rank sum is expected to be no more than `8`.

Outputs:

- `weightout_r`, `weightout_i`: `float`, shape `[sum(lens), 384, 8]`.

The logical output is `[indexCount, lens[i], 384, 8]`; because Ascend tensor storage is contiguous rather than ragged, it is flattened by prefix offset:

```text
pos = sum(lens[:i])
weightout[pos + k, :, :] == logical weightout[i, k, :, :]
```

For each index `i`:

```text
norm[row, col] = weight_r[i, row, col]^2 + weight_i[i, row, col]^2
weightVec[col] = sum(norm[:, col])
sumRank = sum(getuserIdRank[pos:pos + lens[i]])
scale[col] = 1 / sqrt(sumRank) / sqrt(weightVec[col])
weightTmp_r = weight_r[i] * scale
weightTmp_i = weight_i[i] * scale

shufflePos = 0
for k in range(lens[i]):
    weightout[pos + k] = rotate_left_columns(weightTmp, shufflePos)
    shufflePos += getuserIdRank[pos + k]
```

`rotate_left_columns(x, 2)` maps columns `a,b,c,d,e,f,g,h` to `c,d,e,f,g,h,a,b`.

Implementation notes:

- The kernel runs on vector cores (`AIV`); the `__DAV_CUBE__` path returns immediately.
- Elementwise norm and complex scaling use AscendC vector APIs: `Mul`, `Add`, `Rsqrt`, and `Muls`.
- Column reduction and final column rotation are done in UB with `LocalTensor::GetValue` / `SetValue`; only complete `[384, 8]` matrices are copied between GM and UB.
- `weightVec[col] <= 0` is clamped to `1` before reciprocal square root so zero columns produce zero output instead of NaN.

Build:

```bash
./build.sh
```

Install the generated `build_out/custom_opp_<os>_<arch>.run` package before invoking the operator.
