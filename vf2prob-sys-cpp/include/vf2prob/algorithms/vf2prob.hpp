// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include "vf2prob/matcher.hpp"

namespace vf2prob {

// VF2-Prob: branch-and-bound maximum-likelihood subgraph isomorphism with
// calibrated nonlinear compatibilities and an admissible upper bound. The same
// objective and bound admit four configurations:
//   (DFS | A*) x (node-UB | assign-UB)
// All four return the same provable optimum when they terminate. The bound and
// search strategy are fixed per instance (constructor), so name() and behavior
// stay consistent regardless of the bound/search fields in MatchOptions.
class VF2ProbMatcher : public IMatcher {
 public:
  VF2ProbMatcher(BoundKind bound, SearchKind search)
      : bound_(bound), search_(search) {}

  std::string name() const override;
  MatchResult solve(const LabeledGraph& G, const LabeledGraph& Q,
                    const MatchOptions& opt) override;

 private:
  MatchResult solve_dfs(const LabeledGraph& G, const LabeledGraph& Q,
                        const MatchOptions& opt);
  MatchResult solve_astar(const LabeledGraph& G, const LabeledGraph& Q,
                          const MatchOptions& opt);

  BoundKind bound_;
  SearchKind search_;
};

}  // namespace vf2prob
