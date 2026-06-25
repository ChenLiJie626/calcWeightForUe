#ifndef CALC_WEIGHT_FOR_UE_TILING_H
#define CALC_WEIGHT_FOR_UE_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace CalcWeightForUeConst {
constexpr int32_t FEATURES = 256;
constexpr int32_t DEFAULT_BLOCK_DIM = 16;
} // namespace CalcWeightForUeConst

namespace optiling {
BEGIN_TILING_DATA_DEF(CalcWeightForUeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, indexCount);
TILING_DATA_FIELD_DEF(uint32_t, totalRankEntries);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CalcWeightForUe, CalcWeightForUeTilingData)
} // namespace optiling

#endif // CALC_WEIGHT_FOR_UE_TILING_H
