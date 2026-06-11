// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include "vf2prob/matcher.hpp"

namespace vf2prob {

// Most-Probable-Match: a representative uncertain-graph baseline (in the spirit
// of probabilistic / filter-verify matching on uncertain graphs). Keeps exact
// structure and HARD label predicates, but instead of returning the first
// embedding it maximizes the embedding's existence probability, i.e. the sum of
// log edge-existence probabilities sum log p(e) (edges independent Bernoulli).
// Branch-and-bound with the trivial admissible bound (remaining log p <= 0).
// Distinct from VF2-Bin (hard binarization) and VF2-Prob (calibrated logistic
// node+edge compatibilities).
class MpmMatcher : public IMatcher {
 public:
  std::string name() const override { return "mpm"; }
  MatchResult solve(const LabeledGraph& G, const LabeledGraph& Q,
                    const MatchOptions& opt) override;
};

}  // namespace vf2prob
