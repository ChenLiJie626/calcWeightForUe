# AclNNInvocation

This directory calls the installed `CalcWeightForUe` custom operator through the generated two-stage aclnn API:

```cpp
aclnnCalcWeightForUeGetWorkspaceSize(..., &workspaceSize, &executor);
aclnnCalcWeightForUe(workspace, workspaceSize, executor, stream);
```

Run it after building and installing the operator package from `../CalcWeightForUeMultiCore`:

```bash
./run.sh
```

The sample generates deterministic `[totalRankEntries, 256]` complex input data, runs the operator, and compares `weightout_r` / `weightout_i` with numpy golden output of the same shape. Set `CALC_WEIGHT_SCENARIO=printable` to generate a small readable case, or `CALC_WEIGHT_SCENARIO=avg2 INDEX_COUNT=136` to test 136 indexes with average `lens` equal to 2.
