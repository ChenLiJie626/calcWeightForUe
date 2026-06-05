#include "kernel_operator.h"

using namespace AscendC;

namespace CalcWeightForUeKernel {
constexpr uint32_t ROWS = 384;
constexpr uint32_t RANKS_PER_INDEX = 8;
constexpr uint32_t FLOAT_BYTES = sizeof(float);
constexpr uint32_t MATRIX_ELEMS = ROWS * RANKS_PER_INDEX;
constexpr uint32_t MATRIX_BYTES = MATRIX_ELEMS * FLOAT_BYTES;
constexpr uint32_t SCALE_VECTOR_BYTES = RANKS_PER_INDEX * FLOAT_BYTES;
constexpr uint32_t ZERO_CHUNK_ELEMS = 1024;

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t GetAivBlockNum()
{
    return static_cast<uint64_t>(GetBlockNum());
}

__aicore__ inline uint32_t PositiveLen(int32_t value)
{
    return value > 0 ? static_cast<uint32_t>(value) : 0;
}

__aicore__ inline void ClearLocalMatrix(LocalTensor<float> &dst)
{
    for (uint32_t offset = 0; offset < MATRIX_ELEMS; offset += ZERO_CHUNK_ELEMS) {
        const uint32_t count = static_cast<uint32_t>(MinU64(ZERO_CHUNK_ELEMS, MATRIX_ELEMS - offset));
        Duplicate(dst[offset], 0.0f, count);
    }
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

__aicore__ inline uint32_t CalcSumRank(__gm__ int32_t *ranksGm, uint32_t entryOffset, uint32_t len)
{
    uint32_t sumRank = 0;
    for (uint32_t k = 0; k < len; ++k) {
        sumRank += PositiveLen(ranksGm[entryOffset + k]);
    }
    return sumRank;
}

__aicore__ inline void CalcColumnNormSums(LocalTensor<float> &normLocal, LocalTensor<float> &scaleVecLocal)
{
    for (uint32_t row = 0; row < ROWS; ++row) {
        const uint32_t base = row * RANKS_PER_INDEX;
        for (uint32_t col = 0; col < RANKS_PER_INDEX; ++col) {
            const float value = row == 0 ? 0.0f : scaleVecLocal.GetValue(col);
            scaleVecLocal.SetValue(col, value + normLocal.GetValue(base + col));
        }
    }
    for (uint32_t col = 0; col < RANKS_PER_INDEX; ++col) {
        if (scaleVecLocal.GetValue(col) <= 0.0f) {
            scaleVecLocal.SetValue(col, 1.0f);
        }
    }
}

__aicore__ inline float GetRankFactor(uint32_t sumRank)
{
    switch (sumRank) {
        case 1:
            return 1.0f;
        case 2:
            return 0.7071067811865475f;
        case 3:
            return 0.5773502691896258f;
        case 4:
            return 0.5f;
        case 5:
            return 0.4472135954999579f;
        case 6:
            return 0.4082482904638630f;
        case 7:
            return 0.3779644730092272f;
        case 8:
            return 0.3535533905932738f;
        default:
            return 0.0f;
    }
}

__aicore__ inline void FillScaleMatrix(LocalTensor<float> &scaleLocal, LocalTensor<float> &scaleVecLocal,
                                       uint32_t sumRank)
{
    Rsqrt(scaleVecLocal, scaleVecLocal, RANKS_PER_INDEX);
    Muls(scaleVecLocal, scaleVecLocal, GetRankFactor(sumRank), RANKS_PER_INDEX);
    PipeBarrier<PIPE_ALL>();

    for (uint32_t row = 0; row < ROWS; ++row) {
        const uint32_t base = row * RANKS_PER_INDEX;
        for (uint32_t col = 0; col < RANKS_PER_INDEX; ++col) {
            scaleLocal.SetValue(base + col, scaleVecLocal.GetValue(col));
        }
    }
}

__aicore__ inline void BuildShuffledMatrix(LocalTensor<float> &dst, LocalTensor<float> &src,
                                           uint32_t shufflePos, uint32_t sumRank)
{
    const uint32_t validCols = sumRank < RANKS_PER_INDEX ? sumRank : RANKS_PER_INDEX;
    const uint32_t shift = validCols == 0 ? 0 : shufflePos % validCols;
    for (uint32_t row = 0; row < ROWS; ++row) {
        const uint32_t base = row * RANKS_PER_INDEX;
        for (uint32_t col = 0; col < RANKS_PER_INDEX; ++col) {
            if (col < validCols) {
                uint32_t srcCol = col + shift;
                if (srcCol >= validCols) {
                    srcCol -= validCols;
                }
                dst.SetValue(base + col, src.GetValue(base + srcCol));
            } else {
                dst.SetValue(base + col, src.GetValue(base + col));
            }
        }
    }
}

__aicore__ inline void WriteShuffledMatrix(GlobalTensor<float> &out, LocalTensor<float> &tmpLocal,
                                           LocalTensor<float> &outLocal, uint64_t outOffset,
                                           uint32_t shufflePos, uint32_t sumRank)
{
    const uint32_t validCols = sumRank < RANKS_PER_INDEX ? sumRank : RANKS_PER_INDEX;
    if (validCols > 0 && shufflePos % validCols == 0) {
        DataCopy(out[outOffset], tmpLocal, MATRIX_ELEMS);
        return;
    }
    BuildShuffledMatrix(outLocal, tmpLocal, shufflePos, sumRank);
    PipeBarrier<PIPE_ALL>();
    DataCopy(out[outOffset], outLocal, MATRIX_ELEMS);
}

__aicore__ inline void ProcessIndex(GlobalTensor<float> &weightR, GlobalTensor<float> &weightI,
                                    GM_ADDR lens, GM_ADDR getUserIdRank,
                                    GlobalTensor<float> &outR, GlobalTensor<float> &outI,
                                    TBuf<> &srcRBuf, TBuf<> &srcIBuf, TBuf<> &normBuf,
                                    TBuf<> &workBuf, TBuf<> &scaleVecBuf, TBuf<> &scaleBuf,
                                    TBuf<> &tmpRBuf, TBuf<> &tmpIBuf,
                                    TBuf<> &outRBuf, TBuf<> &outIBuf,
                                    uint32_t totalRankEntries, uint32_t index)
{
    auto lensGm = reinterpret_cast<__gm__ int32_t *>(lens);
    auto ranksGm = reinterpret_cast<__gm__ int32_t *>(getUserIdRank);

    const uint32_t entryOffset = GetEntryOffset(lensGm, index, totalRankEntries);
    const uint32_t len = GetAvailableLen(lensGm, index, entryOffset, totalRankEntries);
    if (len == 0) {
        return;
    }

    const uint32_t sumRank = CalcSumRank(ranksGm, entryOffset, len);
    if (sumRank == 0) {
        LocalTensor<float> outRLocal = outRBuf.Get<float>();
        LocalTensor<float> outILocal = outIBuf.Get<float>();
        ClearLocalMatrix(outRLocal);
        ClearLocalMatrix(outILocal);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t k = 0; k < len; ++k) {
            const uint64_t outOffset = static_cast<uint64_t>(entryOffset + k) * MATRIX_ELEMS;
            DataCopy(outR[outOffset], outRLocal, MATRIX_ELEMS);
            DataCopy(outI[outOffset], outILocal, MATRIX_ELEMS);
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    LocalTensor<float> srcRLocal = srcRBuf.Get<float>();
    LocalTensor<float> srcILocal = srcIBuf.Get<float>();
    LocalTensor<float> normLocal = normBuf.Get<float>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> scaleVecLocal = scaleVecBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<float> tmpRLocal = tmpRBuf.Get<float>();
    LocalTensor<float> tmpILocal = tmpIBuf.Get<float>();
    LocalTensor<float> outRLocal = outRBuf.Get<float>();
    LocalTensor<float> outILocal = outIBuf.Get<float>();

    const uint64_t inputOffset = static_cast<uint64_t>(index) * MATRIX_ELEMS;
    DataCopy(srcRLocal, weightR[inputOffset], MATRIX_ELEMS);
    DataCopy(srcILocal, weightI[inputOffset], MATRIX_ELEMS);
    PipeBarrier<PIPE_ALL>();

    Mul(normLocal, srcRLocal, srcRLocal, MATRIX_ELEMS);
    Mul(workLocal, srcILocal, srcILocal, MATRIX_ELEMS);
    Add(normLocal, normLocal, workLocal, MATRIX_ELEMS);
    PipeBarrier<PIPE_ALL>();

    CalcColumnNormSums(normLocal, scaleVecLocal);
    FillScaleMatrix(scaleLocal, scaleVecLocal, sumRank);
    PipeBarrier<PIPE_ALL>();

    Mul(tmpRLocal, srcRLocal, scaleLocal, MATRIX_ELEMS);
    Mul(tmpILocal, srcILocal, scaleLocal, MATRIX_ELEMS);
    PipeBarrier<PIPE_ALL>();

    uint32_t shufflePos = 0;
    for (uint32_t k = 0; k < len; ++k) {
        const uint64_t outOffset = static_cast<uint64_t>(entryOffset + k) * MATRIX_ELEMS;
        WriteShuffledMatrix(outR, tmpRLocal, outRLocal, outOffset, shufflePos, sumRank);
        WriteShuffledMatrix(outI, tmpILocal, outILocal, outOffset, shufflePos, sumRank);
        PipeBarrier<PIPE_ALL>();
        shufflePos += PositiveLen(ranksGm[entryOffset + k]);
    }
}
} // namespace CalcWeightForUeKernel

extern "C" __global__ __aicore__ void calc_weight_for_ue(GM_ADDR weight_r, GM_ADDR weight_i,
                                                          GM_ADDR lens, GM_ADDR getuserIdRank,
                                                          GM_ADDR weightout_r, GM_ADDR weightout_i,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(__DAV_CUBE__)
    (void)weight_r;
    (void)weight_i;
    (void)lens;
    (void)getuserIdRank;
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
    const uint64_t weightElems = static_cast<uint64_t>(indexCount) * MATRIX_ELEMS;
    const uint64_t outElems = static_cast<uint64_t>(totalRankEntries) * MATRIX_ELEMS;
    weightRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight_r), weightElems);
    weightIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight_i), weightElems);
    outRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightout_r), outElems);
    outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weightout_i), outElems);

    TPipe pipe;
    TBuf<> srcRBuf;
    TBuf<> srcIBuf;
    TBuf<> normBuf;
    TBuf<> workBuf;
    TBuf<> scaleVecBuf;
    TBuf<> scaleBuf;
    TBuf<> tmpRBuf;
    TBuf<> tmpIBuf;
    TBuf<> outRBuf;
    TBuf<> outIBuf;
    pipe.InitBuffer(srcRBuf, MATRIX_BYTES);
    pipe.InitBuffer(srcIBuf, MATRIX_BYTES);
    pipe.InitBuffer(normBuf, MATRIX_BYTES);
    pipe.InitBuffer(workBuf, MATRIX_BYTES);
    pipe.InitBuffer(scaleVecBuf, SCALE_VECTOR_BYTES);
    pipe.InitBuffer(scaleBuf, MATRIX_BYTES);
    pipe.InitBuffer(tmpRBuf, MATRIX_BYTES);
    pipe.InitBuffer(tmpIBuf, MATRIX_BYTES);
    pipe.InitBuffer(outRBuf, MATRIX_BYTES);
    pipe.InitBuffer(outIBuf, MATRIX_BYTES);

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(GetAivBlockNum());
    for (uint32_t index = blockIdx; index < indexCount; index += blockNum) {
        ProcessIndex(weightRGm, weightIGm, lens, getuserIdRank,
                     outRGm, outIGm, srcRBuf, srcIBuf, normBuf, workBuf,
                     scaleVecBuf, scaleBuf, tmpRBuf, tmpIBuf, outRBuf, outIBuf,
                     totalRankEntries, index);
    }
#endif
}
