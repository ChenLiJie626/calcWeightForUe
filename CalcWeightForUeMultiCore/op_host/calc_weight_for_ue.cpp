#include "calc_weight_for_ue_tiling.h"

#include <cstddef>
#include <iostream>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace CalcWeightForUeConst;

namespace {
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

bool CheckOutputShape(const gert::Shape &shape)
{
    return shape.GetDimNum() == 2 &&
           shape.GetDim(0) > 0 &&
           shape.GetDim(1) == FEATURES;
}

bool CheckScalarVectorShape(const gert::Shape &shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == 1;
}

void PrintShape(const char *name, const gert::Shape &shape)
{
    std::cerr << name << "=[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i > 0) {
            std::cerr << ",";
        }
        std::cerr << shape.GetDim(i);
    }
    std::cerr << "]";
}

void PrintInferShapeError(const char *reason)
{
    std::cerr << "[CalcWeightForUe][InferShape] " << reason << std::endl;
}

static UINT32 InferShapeFunc(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        PrintInferShapeError("context is null");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *weightRShape = context->GetInputShape(0);
    const gert::Shape *weightIShape = context->GetInputShape(1);
    const gert::Shape *lensShape = context->GetInputShape(2);
    const gert::Shape *flagShape = context->GetInputShape(3);
    const gert::Shape *idxCountShape = context->GetInputShape(4);
    gert::Shape *outRShape = context->GetOutputShape(0);
    gert::Shape *outIShape = context->GetOutputShape(1);
    if (weightRShape == nullptr || weightIShape == nullptr || lensShape == nullptr ||
        flagShape == nullptr || idxCountShape == nullptr) {
        PrintInferShapeError("one or more input shape pointers are null");
        return ge::GRAPH_FAILED;
    }
    if (outRShape == nullptr || outIShape == nullptr) {
        PrintInferShapeError("one or more output shape pointers are null");
        return ge::GRAPH_FAILED;
    }
    if (!CheckWeightShape(*weightRShape)) {
        std::cerr << "[CalcWeightForUe][InferShape] invalid weight_r shape, expected [N,"
                  << FEATURES << "], actual ";
        PrintShape("weight_r", *weightRShape);
        std::cerr << std::endl;
        return ge::GRAPH_FAILED;
    }
    if (!CheckSameShape(*weightRShape, *weightIShape)) {
        std::cerr << "[CalcWeightForUe][InferShape] weight_i shape must match weight_r, actual ";
        PrintShape("weight_r", *weightRShape);
        std::cerr << ", ";
        PrintShape("weight_i", *weightIShape);
        std::cerr << std::endl;
        return ge::GRAPH_FAILED;
    }
    if (lensShape->GetDimNum() != 1) {
        std::cerr << "[CalcWeightForUe][InferShape] lens must be 1-D, actual ";
        PrintShape("lens", *lensShape);
        std::cerr << std::endl;
        return ge::GRAPH_FAILED;
    }
    if (!CheckSameShape(*lensShape, *flagShape)) {
        std::cerr << "[CalcWeightForUe][InferShape] flag shape must match lens, actual ";
        PrintShape("lens", *lensShape);
        std::cerr << ", ";
        PrintShape("flag", *flagShape);
        std::cerr << std::endl;
        return ge::GRAPH_FAILED;
    }
    if (!CheckScalarVectorShape(*idxCountShape)) {
        std::cerr << "[CalcWeightForUe][InferShape] idxCount shape must be [1], actual ";
        PrintShape("idxCount", *idxCountShape);
        std::cerr << std::endl;
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
    const ge::DataType idxCountType = context->GetInputDataType(4);
    if (weightRType != ge::DT_FLOAT || weightIType != ge::DT_FLOAT ||
        lensType != ge::DT_UINT32 || flagType != ge::DT_UINT32 || idxCountType != ge::DT_UINT32) {
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
    auto idxCountShape = context->GetInputTensor(4)->GetOriginShape();
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
    if (lensShape.GetDimNum() != 1 || !CheckSameShape(lensShape, flagShape) ||
        !CheckScalarVectorShape(idxCountShape)) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckOutputShape(outRShape) || !CheckSameShape(outRShape, weightRShape) ||
        !CheckSameShape(outIShape, weightRShape)) {
        return ge::GRAPH_FAILED;
    }

    CalcWeightForUeTilingData tiling;
    context->SetBlockDim(DEFAULT_BLOCK_DIM);
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
        this->Input("idxCount")
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
