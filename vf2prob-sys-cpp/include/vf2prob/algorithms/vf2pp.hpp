// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include "vf2prob/matcher.hpp"

namespace vf2prob {

// Engineered VF2++-style exact baseline (Jüttner & Madarasi): connectivity-driven
// matching order, frontier-restricted candidate generation, and a label-wise
// neighborhood look-ahead cut. Returns the first structural embedding and prunes
// more than classic VF2.
class VF2ppMatcher : public IMatcher {
 public:
  std::string name() const override { return "vf2pp"; }
  MatchResult solve(const LabeledGraph& G, const LabeledGraph& Q,
                    const MatchOptions& opt) override;
};

}  // namespace vf2prob
