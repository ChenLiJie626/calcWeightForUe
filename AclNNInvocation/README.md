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

The sample generates deterministic `[4, 384, 8]` complex input data, runs the operator, and compares `weightout_r` / `weightout_i` with numpy golden output of shape `[10, 384, 8]`.
