// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include "vf2prob/matcher.hpp"

namespace vf2prob {

// Classic VF2 exact subgraph isomorphism (Cordella et al., on Ullmann's
// backtracking). Hard label equality, degree filter, neighborhood consistency;
// returns the first structural embedding. best_loglik is -inf (exact, no score).
class VF2Matcher : public IMatcher {
 public:
  std::string name() const override { return "vf2"; }
  MatchResult solve(const LabeledGraph& G, const LabeledGraph& Q,
                    const MatchOptions& opt) override;
};

// Core reused by VF2-Bin: first structural embedding via classic VF2 on G.
MatchResult vf2_first_embedding(const LabeledGraph& G, const LabeledGraph& Q,
                                const MatchOptions& opt);

}  // namespace vf2prob
