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
    return shape.GetDimNum() == 2 &&
           shape.GetDim(0) > 0 &&
           shape.GetDim(1) == FEATURES;
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
    return shape.GetDimNum() == 2 &&
           shape.GetDim(0) == totalRankEntries &&
           shape.GetDim(1) == FEATURES;
}

static UINT32 InferShapeFunc(gert::InferShapeContext *context)
{
    const gert::Shape *weightRShape = context->GetInputShape(0);
    gert::Shape *outRShape = context->GetOutputShape(0);
    gert::Shape *outIShape = context->GetOutputShape(1);
    if (weightRShape == nullptr || outRShape == nullptr || outIShape == nullptr ||
        !CheckWeightShape(*weightRShape)) {
        return ge::GRAPH_FAILED;
    }

    *outRShape = *weightRShape;
    *outIShape = *weightRShape;
    return ge::GRAPH_SUCCESS;
}

static UINT32 InferDataTypeFunc(gert::InferDataTypeContext *context)
{
    const ge::DataType weightRType = context->GetInputDataType(0);
    const ge::DataType weightIType = context->GetInputDataType(1);
    const ge::DataType lensType = context->GetInputDataType(2);
    const ge::DataType flagType = context->GetInputDataType(3);
    if (weightRType != ge::DT_FLOAT || weightIType != ge::DT_FLOAT ||
        lensType != ge::DT_UINT32 || flagType != ge::DT_UINT32) {
        return ge::GRAPH_FAILED;
    }
    if (context->SetOutputDataType(0, weightRType) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(1, weightRType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto weightRShape = context->GetInputTensor(0)->GetOriginShape();
    auto weightIShape = context->GetInputTensor(1)->GetOriginShape();
    auto lensShape = context->GetInputTensor(2)->GetOriginShape();
    auto flagShape = context->GetInputTensor(3)->GetOriginShape();
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
    if (lensShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t indexCount = lensShape.GetDim(0);
    if (indexCount <= 0 || !CheckVectorShape(flagShape, indexCount)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t totalRankEntries = weightRShape.GetDim(0);
    if (totalRankEntries <= 0) {
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
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND});
        this->Input("flag")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND});
        this->Output("weightout_r")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("weightout_i")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(InferShapeFunc)
            .SetInferDataType(InferDataTypeFunc);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(CalcWeightForUe);
} // namespace ops
