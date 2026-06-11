// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <vector>

#include "vf2prob/graph.hpp"
#include "vf2prob/matcher.hpp"

namespace vf2prob {

// Per-query-node maximum node log-compat over all u with deg_G(u) >= deg_Q(q)
// (over-approximating the candidate set keeps the bound admissible and cheap).
// Computed once per (Q, G) pair.
std::vector<double> precompute_max_node_log(const LabeledGraph& G,
                                            const LabeledGraph& Q,
                                            const MatchOptions& opt);

// Cheap admissible node bound: sum over unmapped q of max_node_log[q].
double node_upper_bound(const LabeledGraph& Q,
                        const std::vector<NodeId>& mapping,
                        const std::vector<double>& max_node_log);

// Tighter admissible assignment bound: maximum-weight injective assignment of
// unmapped query nodes to distinct structurally-feasible candidates, with edge
// look-ahead to already-mapped neighbors. Falls back to the node bound when a
// full injective assignment is impossible or the instance is too large; both
// fallbacks remain admissible.
double assign_upper_bound(const LabeledGraph& G, const LabeledGraph& Q,
                          const std::vector<NodeId>& mapping,
                          const std::vector<char>& used,
                          const std::vector<double>& max_node_log,
                          const MatchOptions& opt);

// Instrumentation: how often assign_upper_bound fell back to the node bound
// (empty pool, no injective assignment, or instance over the size cap).
struct AssignStats {
  uint64_t calls = 0;
  uint64_t fallbacks = 0;
};
void reset_assign_stats();
AssignStats get_assign_stats();

}  // namespace vf2prob
