// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "vf2prob/types.hpp"

namespace vf2prob {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();
inline double quiet_nan() { return std::numeric_limits<double>::quiet_NaN(); }

// Implementation-independent search effort + result quality counters produced
// by every matcher. Wall-clock time and peak memory are measured by the
// harness (outside the solver) so the comparison across methods is fair.
struct Counters {
  uint64_t states = 0;     // expanded search states (the primary metric)
  uint64_t pruned = 0;     // branches cut by infeasibility or the bound
  uint64_t solutions = 0;  // complete embeddings found (>=1 means "matched")
  double best_loglik = kNegInf;  // best log-likelihood (VF2-Prob); -inf for exact
  double ub_gap_p50 = quiet_nan();  // bound-tightness samples (VF2-Prob only)
  double ub_gap_p90 = quiet_nan();
  bool timed_out = false;

  double prune_rate() const {
    const uint64_t denom = states + pruned;
    return denom ? static_cast<double>(pruned) / static_cast<double>(denom) : 0.0;
  }
};

struct MatchResult {
  // Best mapping q -> u (size |V_Q|), or nullopt if none / timed out empty.
  std::optional<std::vector<NodeId>> mapping;
  Counters counters;
};

}  // namespace vf2prob
