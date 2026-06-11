// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <cstdint>

// Shared scalar types used across the whole project.
namespace vf2prob {

using NodeId = int32_t;   // compact node identifier, 0..num_nodes-1
using LabelId = int32_t;  // categorical label identifier in a shared vocabulary

constexpr LabelId kNoLabel = -1;  // node/edge has no categorical label

}  // namespace vf2prob
