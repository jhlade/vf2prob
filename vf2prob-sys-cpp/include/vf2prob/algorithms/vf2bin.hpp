// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include "vf2prob/matcher.hpp"

namespace vf2prob {

// Keep only edges with existence probability >= tau, producing a deterministic
// graph (drops node/edge probabilities and labels otherwise unchanged).
LabeledGraph binarize(const LabeledGraph& G, double tau);

// VF2-Bin baseline: binarize uncertain edges at threshold tau (default 0.7),
// then run classic structural VF2 on the resulting deterministic graph. This
// emulates how a system without probabilistic support would handle uncertain
// edges. Metrics are exact VF2-style (no score, ub_gap = NaN).
class VF2BinMatcher : public IMatcher {
 public:
  std::string name() const override { return "vf2bin"; }
  MatchResult solve(const LabeledGraph& G, const LabeledGraph& Q,
                    const MatchOptions& opt) override;
};

}  // namespace vf2prob
