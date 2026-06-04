# CalcWeightForUe AscendC Operator

This workspace contains an AscendC vector-kernel custom operator named `CalcWeightForUe`.

Directory layout:

- `CalcWeightForUeMultiCore/`: AscendC custom operator source, host registration, tiling, and build scripts.
- `AclNNInvocation/`: deterministic ACLNN invocation sample and numpy verification.
- `CalcWeightForUe.json`: op metadata for the four-input, two-output operator.

Inputs:

- `weight_r`, `weight_i`: `float`, shape `[indexCount, 384, 8]`.
- `lens`: `int32`, shape `[indexCount]`.
- `getuserIdRank`: `int32`, shape `[sum(lens)]`.

Outputs:

- `weightout_r`, `weightout_i`: `float`, shape `[sum(lens), 384, 8]`.

The logical ragged output is `[indexCount, lens[i], 384, 8]`; it is stored as a flattened first dimension using `pos = sum(lens[:i])`.

Logical behavior:

```text
pos = 0
for i in range(indexCount):
    norm = weight_r[i] * weight_r[i] + weight_i[i] * weight_i[i]
    weightVec = sum(norm, axis=0)
    ranks = getuserIdRank[pos:pos + lens[i]]
    sumRank = sum(ranks)
    scale = 1 / sqrt(sumRank) / sqrt(weightVec)
    weightTmp = weight[i] * scale

    shufflePos = 0
    for k in range(lens[i]):
        weightout[pos + k] = rotate_left_columns(weightTmp, shufflePos)
        shufflePos += ranks[k]
    pos += lens[i]
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
