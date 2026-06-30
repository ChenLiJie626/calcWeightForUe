# CalcWeightForUe AscendC Operator

This workspace contains an AscendC vector-kernel custom operator named `CalcWeightForUe`.

Directory layout:

- `CalcWeightForUeMultiCore/`: AscendC custom operator source, host registration, tiling, and build scripts.
- `AclNNInvocation/`: deterministic ACLNN invocation sample and numpy verification.
- `CalcWeightForUe.json`: op metadata for the five-input, two-output operator.

Inputs:

- `weight_r`, `weight_i`: `float`, shape `[totalRankEntries, 256]`.
- `lens`: `uint32`, shape `[lensStorageCount]`.
- `flag`: `uint32`, shape `[lensStorageCount]`. `flag[i] != 0` enables normalization for index `i`; `flag[i] == 0` copies that segment to output unchanged.
- `idxCount`: `uint32`, shape `[1]`. Only `lens[:idxCount]` and `flag[:idxCount]` participate in computation, and `sum(lens[:idxCount]) == totalRankEntries`.

Outputs:

- `weightout_r`, `weightout_i`: `float`, shape `[totalRankEntries, 256]`.

Logical behavior:

```text
posmatrix = 0
for i in range(idxCount):
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
