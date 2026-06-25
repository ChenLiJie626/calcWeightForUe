# CalcWeightForUe AscendC Operator

This workspace contains an AscendC vector-kernel custom operator named `CalcWeightForUe`.

Directory layout:

- `CalcWeightForUeMultiCore/`: AscendC custom operator source, host registration, tiling, and build scripts.
- `AclNNInvocation/`: deterministic ACLNN invocation sample and numpy verification.
- `CalcWeightForUe.json`: op metadata for the four-input, two-output operator.

Inputs:

- `weight_r`, `weight_i`: `float`, shape `[totalRankEntries, 256]`.
- `lens`: `int32`, shape `[indexCount]`. `sum(max(lens, 0)) == totalRankEntries`.
- `flag`: `int32`, shape `[indexCount]`. `flag[i] != 0` enables normalization for index `i`; `flag[i] == 0` copies that segment to output unchanged.

Outputs:

- `weightout_r`, `weightout_i`: `float`, shape `[totalRankEntries, 256]`.

Logical behavior:

```text
posmatrix = 0
for i in range(indexCount):
    ranks = lens[i]
    weight = weight[posmatrix:posmatrix + ranks, :]
    if flag[i] == 0:
        weightout[posmatrix:posmatrix + ranks, :] = weight
    else:
        weightNorm = weight_r^2 + weight_i^2
        weightVec = sum(weightNorm, axis=1)
        f = 1 / sqrt(ranks)
        r = f / sqrt(weightVec)
        weightout = weight * r[:, None]
    posmatrix += ranks
```

Build the operator:

```bash
cd CalcWeightForUeMultiCore
./build.sh
```

After installing the generated package, run the sample:

```bash
cd ../AclNNInvocation
./run.sh
```
