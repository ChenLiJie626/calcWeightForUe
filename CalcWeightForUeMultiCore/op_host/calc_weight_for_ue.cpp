#include "calc_weight_for_ue_tiling.h"

#include <cstddef>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace CalcWeightForUeConst;

namespace {
bool CheckVectorShape(const gert::Shape &shape, int64_t len)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == len;
}

bool CheckWeightShape(const gert::Shape &shape)
{
    return shape.GetDimNum() == 3 &&
           shape.GetDim(0) > 0 &&
           shape.GetDim(1) == ROWS &&
           shape.GetDim(2) == RANKS_PER_INDEX;
}

bool CheckSameShape(const gert::Shape &lhs, const gert::Shape &rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

bool CheckOutputShape(const gert::Shape &shape, int64_t totalRankEntries)
{
    return shape.GetDimNum() == 3 &&
           shape.GetDim(0) == totalRankEntries &&
           shape.GetDim(1) == ROWS &&
           shape.GetDim(2) == RANKS_PER_INDEX;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto weightRShape = context->GetInputTensor(0)->GetOriginShape();
    auto weightIShape = context->GetInputTensor(1)->GetOriginShape();
    auto lensShape = context->GetInputTensor(2)->GetOriginShape();
    auto ranksShape = context->GetInputTensor(3)->GetOriginShape();
    auto outRStorageShape = context->GetOutputShape(0);
    auto outIStorageShape = context->GetOutputShape(1);
    if (outRStorageShape == nullptr || outIStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto outRShape = outRStorageShape->GetOriginShape();
    auto outIShape = outIStorageShape->GetOriginShape();

    if (!CheckWeightShape(weightRShape) || !CheckSameShape(weightRShape, weightIShape)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t indexCount = weightRShape.GetDim(0);
    if (!CheckVectorShape(lensShape, indexCount) || ranksShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }

    const int64_t totalRankEntries = ranksShape.GetDim(0);
    if (totalRankEntries < 0 ||
        static_cast<uint64_t>(totalRankEntries) > static_cast<uint64_t>(indexCount) * RANKS_PER_INDEX) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckOutputShape(outRShape, totalRankEntries) ||
        !CheckOutputShape(outIShape, totalRankEntries)) {
        return ge::GRAPH_FAILED;
    }

    CalcWeightForUeTilingData tiling;
    tiling.set_indexCount(static_cast<uint32_t>(indexCount));
    tiling.set_totalRankEntries(static_cast<uint32_t>(totalRankEntries));

    uint32_t blockDim = DEFAULT_BLOCK_DIM;
    if (indexCount > 0 && indexCount < DEFAULT_BLOCK_DIM) {
        blockDim = static_cast<uint32_t>(indexCount);
    }
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class CalcWeightForUe : public OpDef {
public:
    explicit CalcWeightForUe(const char *name) : OpDef(name)
    {
        this->Input("weight_r")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("weight_i")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Input("getuserIdRank")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Output("weightout_r")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("weightout_i")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(CalcWeightForUe);
} // namespace ops
