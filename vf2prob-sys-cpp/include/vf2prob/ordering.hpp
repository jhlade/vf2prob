// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <utility>
#include <vector>

#include "vf2prob/graph.hpp"

namespace vf2prob {

// Deterministic fail-first order of query nodes: decreasing degree, ties broken
// by (label, id). Reproducible regardless of hash/iteration order.
std::vector<NodeId> failfirst_order(const LabeledGraph& Q);

// Mapped neighbors of q: pairs (qn, f(qn)) for query neighbors qn already mapped
// (mapping[qn] >= 0).
std::vector<std::pair<NodeId, NodeId>> mapped_neighbors(
    const LabeledGraph& Q, NodeId q, const std::vector<NodeId>& mapping);

// Structurally feasible candidates u for q under the current partial mapping:
// not yet used, deg_G(u) >= deg_Q(q), and adjacent to the image of every mapped
// neighbor of q (so every query edge to a mapped neighbor is preserved).
// Returned sorted ascending by id (deterministic). No attribute filtering —
// labels are soft and enter only through the score.
std::vector<NodeId> feasible_candidates(const LabeledGraph& G,
                                        const LabeledGraph& Q, NodeId q,
                                        const std::vector<NodeId>& mapping,
                                        const std::vector<char>& used);

}  // namespace vf2prob
