// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "vf2prob/compat.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/metrics.hpp"

namespace vf2prob {

enum class BoundKind { Node, Assign };   // admissible upper bound for VF2-Prob
enum class SearchKind { DFS, AStar };    // depth-first B&B (DFS) or best-first (A*)

struct MatchOptions {
  // Deterministic search-effort budget (0 = unlimited): the REPRODUCIBLE
  // completion criterion. A run is "complete" iff it finishes within max_states.
  uint64_t max_states = 0;
  // Wall-clock safety net only (timing-dependent, NOT reproducible); set high
  // enough not to trigger before max_states when reproducibility is required.
  double timeout_s = 60.0;
  // VF2-Prob configuration:
  BoundKind bound = BoundKind::Node;
  SearchKind search = SearchKind::DFS;
  bool use_node = true;  // include node compatibilities in the objective
  bool use_edge = true;  // include edge compatibilities in the objective
  bool collect_metrics = true;  // sample bound-tightness (off => ~0 overhead)
  // VF2-Bin configuration:
  double bin_tau = 0.7;  // keep edges with prob >= tau, then run structural VF2
  uint64_t seed = 324; // default seed
  // Compatibility weights used by the objective and both bounds.
  compat::Params weights{};
  // Optional learned expansion-order ranker (DFS only): kRankFeats weights that
  // reorder feasible candidates. Empty => default gain ordering. Ordering is
  // heuristic; it never changes feasibility or the optimum, so exactness holds.
  std::vector<double> rank_w{};
};

// Uniform interface for every algorithm, so the harness can dispatch and measure
// them identically.
class IMatcher {
 public:
  virtual ~IMatcher() = default;
  virtual std::string name() const = 0;
  // Find the best embedding of query Q into data graph G under `opt`.
  // Exact baselines return the first structural embedding; VF2-Prob returns the
  // maximum-likelihood embedding (same optimum across its four variants).
  virtual MatchResult solve(const LabeledGraph& G, const LabeledGraph& Q,
                            const MatchOptions& opt) = 0;
};

}  // namespace vf2prob
