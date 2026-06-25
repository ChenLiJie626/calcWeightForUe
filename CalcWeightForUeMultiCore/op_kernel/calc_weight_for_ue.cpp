#include "kernel_operator.h"

using namespace AscendC;

namespace CalcWeightForUeKernel {
constexpr uint32_t FEATURES = 256;
constexpr uint32_t FLOAT_BYTES = sizeof(float);
constexpr uint32_t ROW_BYTES = FEATURES * FLOAT_BYTES;

__aicore__ inline uint32_t PositiveLen(int32_t value)
{
    return value > 0 ? static_cast<uint32_t>(value) : 0;
}

__aicore__ inline uint32_t GetEntryOffset(__gm__ int32_t *lensGm, uint32_t index, uint32_t totalRankEntries)
{
    uint32_t offset = 0;
    for (uint32_t i = 0; i < index; ++i) {
        const uint32_t currentLen = PositiveLen(lensGm[i]);
        const uint32_t remaining = offset < totalRankEntries ? totalRankEntries - offset : 0;
        if (currentLen >= remaining) {
            return totalRankEntries;
        }
        offset += currentLen;
    }
    return offset;
}

__aicore__ inline uint32_t GetAvailableLen(__gm__ int32_t *lensGm, uint32_t index,
                                           uint32_t entryOffset, uint32_t totalRankEntries)
{
    const uint32_t currentLen = PositiveLen(lensGm[index]);
    const uint32_t remaining = entryOffset < totalRankEntries ? totalRankEntries - entryOffset : 0;
    return currentLen < remaining ? currentLen : remaining;
}

__aicore__ inline void CopyRankRow(GlobalTensor<float> &src, GlobalTensor<float> &dst,
                                   TBuf<> &copyBuf, uint64_t rowIndex)
{
    LocalTensor<float> copyLocal = copyBuf.Get<float>();
    const uint64_t offset = rowIndex * FEATURES;
    DataCopy(copyLocal, src[offset], FEATURES);
    PipeBarrier<PIPE_ALL>();
    DataCopy(dst[offset], copyLocal, FEATURES);
}

__aicore__ inline float SumFeatureNorm(LocalTensor<float> &normLocal)
{
    float sum = 0.0f;
    for (uint32_t col = 0; col < FEATURES; ++col) {
        sum += normLocal.GetValue(col);
    }
    return sum > 0.0f ? sum : 1.0f;
}

__aicore__ inline float RankCountToFloat(uint32_t ranks)
{
    float value = 0.0f;
    for (uint32_t i = 0; i < ranks; ++i) {
        value += 1.0f;
    }
    return value;
}

__aicore__ inline void ProcessRankRow(GlobalTensor<float> &weightR, GlobalTensor<float> &weightI,
                                      GlobalTensor<float> &outR, GlobalTensor<float> &outI,
                                      TBuf<> &srcRBuf, TBuf<> &srcIBuf, TBuf<> &normBuf,
                                      TBuf<> &workBuf, TBuf<> &scaleBuf, TBuf<> &rankFactorBuf,
                                      uint64_t rowIndex, uint32_t ranks)
{
    LocalTensor<float> srcRLocal = srcRBuf.Get<float>();
    LocalTensor<float> srcILocal = srcIBuf.Get<float>();
    LocalTensor<float> normLocal = normBuf.Get<float>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<float> rankFactorLocal = rankFactorBuf.Get<float>();

    const uint64_t offset = rowIndex * FEATURES;
    DataCopy(srcRLocal, weightR[offset], FEATURES);
    DataCopy(srcILocal, weightI[offset], FEATURES);
    PipeBarrier<PIPE_ALL>();

    Mul(normLocal, srcRLocal, srcRLocal, FEATURES);
    Mul(workLocal, srcILocal, srcILocal, FEATURES);
    Add(normLocal, normLocal, workLocal, FEATURES);
    PipeBarrier<PIPE_ALL>();

    const float weightVec = SumFeatureNorm(normLocal);
    Duplicate(scaleLocal, weightVec, FEATURES);
    Duplicate(rankFactorLocal, RankCountToFloat(ranks), FEATURES);
    PipeBarrier<PIPE_ALL>();

    Rsqrt(scaleLocal, scaleLocal, FEATURES);
    Rsqrt(rankFactorLocal, rankFactorLocal, FEATURES);
    PipeBarrier<PIPE_ALL>();
    Mul(scaleLocal, scaleLocal, rankFactorLocal, FEATURES);
    PipeBarrier<PIPE_ALL>();

    Mul(srcRLocal, srcRLocal, scaleLocal, FEATURES);
    Mul(srcILocal, srcILocal, scaleLocal, FEATURES);
    PipeBarrier<PIPE_ALL>();

    DataCopy(outR[offset], srcRLocal, FEATURES);
    DataCopy(outI[offset], srcILocal, FEATURES);
}

__aicore__ inline void ProcessIndex(GlobalTensor<float> &weightR, GlobalTensor<float> &weightI,
                                    GM_ADDR lens, GM_ADDR flag,
                                    GlobalTensor<float> &outR, GlobalTensor<float> &outI,
                                    TBuf<> &srcRBuf, TBuf<> &srcIBuf, TBuf<> &normBuf,
                                    TBuf<> &workBuf, TBuf<> &scaleBuf, TBuf<> &rankFactorBuf,
                                    uint32_t totalRankEntries, uint32_t index)
{
    auto lensGm = reinterpret_cast<__gm__ int32_t *>(lens);
    auto flagGm = reinterpret_cast<__gm__ int32_t *>(flag);

    const uint32_t entryOffset = GetEntryOffset(lensGm, index, totalRankEntries);
    const uint32_t ranks = GetAvailableLen(lensGm, index, entryOffset, totalRankEntries);
    if (ranks == 0) {
        return;
    }

    if (flagGm[index] == 0) {
        for (uint32_t row = 0; row < ranks; ++row) {
            const uint64_t rowIndex = static_cast<uint64_t>(entryOffset + row);
            CopyRankRow(weightR, outR, srcRBuf, rowIndex);
            CopyRankRow(weightI, outI, srcIBuf, rowIndex);
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    for (uint32_t row = 0; row < ranks; ++row) {
        const uint64_t rowIndex = static_cast<uint64_t>(entryOffset + row);
        ProcessRankRow(weightR, weightI, outR, outI,
                       srcRBuf, srcIBuf, normBuf, workBuf, scaleBuf, rankFactorBuf,
                       rowIndex, ranks);
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace CalcWeightForUeKernel

extern "C" __global__ __aicore__ void calc_weight_for_ue(GM_ADDR weight_r, GM_ADDR weight_i,
                                                          GM_ADDR lens, GM_ADDR flag,
                                                          GM_ADDR weightout_r, GM_ADDR weightout_i,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(__DAV_CUBE__)
    (void)weight_r;
    (void)weight_i;
    (void)lens;
    (void)flag;
    (void)weightout_r;
    (void)weightout_i;
    (void)workspace;
    (void)tiling;
    return;
#else
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    using namespace CalcWeightForUeKernel;

    const uint32_t indexCount = tilingData.indexCount;
    const uint32_t totalRankEntries = tilingData.totalRankEntries;
    if (indexCount == 0 || totalRankEntries == 0) {
        return;
    }

    GlobalTensor<float> weightRGm;
    GlobalTensor<float> weightIGm;
    GlobalTensor<float> outRGm;
    GlobalTensor<float> outIGm;
    const uint64_t totalElems = static_cast<uint64_t>(totalRankEntries) * FEATURES;
    weightRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight_r), totalElems);
    weightIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight_i), totalElems);
    outRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightout_r), totalElems);
    outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightout_i), totalElems);

    TPipe pipe;
    TBuf<> srcRBuf;
    TBuf<> srcIBuf;
    TBuf<> normBuf;
    TBuf<> workBuf;
    TBuf<> scaleBuf;
    TBuf<> rankFactorBuf;
    pipe.InitBuffer(srcRBuf, ROW_BYTES);
    pipe.InitBuffer(srcIBuf, ROW_BYTES);
    pipe.InitBuffer(normBuf, ROW_BYTES);
    pipe.InitBuffer(workBuf, ROW_BYTES);
    pipe.InitBuffer(scaleBuf, ROW_BYTES);
    pipe.InitBuffer(rankFactorBuf, ROW_BYTES);

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t index = blockIdx; index < indexCount; index += blockNum) {
        ProcessIndex(weightRGm, weightIGm, lens, flag,
                     outRGm, outIGm, srcRBuf, srcIBuf, normBuf, workBuf, scaleBuf, rankFactorBuf,
                     totalRankEntries, index);
    }
#endif
}
