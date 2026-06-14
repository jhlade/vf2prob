// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/bounds.hpp"

#include <algorithm>
#include <unordered_map>

#include "vf2prob/assignment.hpp"
#include "vf2prob/compat.hpp"
#include "vf2prob/ordering.hpp"

namespace vf2prob {

std::vector<double> precompute_max_node_log(const LabeledGraph& G,
                                            const LabeledGraph& Q,
                                            const MatchOptions& opt) {
  std::vector<double> out(Q.num_nodes(), 0.0);
  for (NodeId q = 0; q < Q.num_nodes(); ++q) {
    const int deg_q = Q.degree(q);
    double best = 0.0;
    for (NodeId u = 0; u < G.num_nodes(); ++u) {
      if (G.degree(u) < deg_q) continue;
      const double s =
          opt.use_node ? compat::node_score(Q, q, G, u, opt.weights) : 1.0;
      if (s > best) best = s;
    }
    out[q] = compat::safe_log(best > 0.0 ? best : 1e-12);
  }
  return out;
}

double node_upper_bound(const LabeledGraph& Q,
                        const std::vector<NodeId>& mapping,
                        const std::vector<double>& max_node_log) {
  double ub = 0.0;
  for (NodeId q = 0; q < Q.num_nodes(); ++q)
    if (mapping[q] < 0) ub += max_node_log[q];
  return ub;
}

// Optimistic weight of mapping q -> u: node log-compat plus the edge log-compat
// to every already-mapped neighbor of q (these edges exist by candidate
// construction and are incurred exactly by any completion through this pair).
static double pair_weight(const LabeledGraph& G, const LabeledGraph& Q, NodeId q,
                          NodeId u, const std::vector<NodeId>& mapping,
                          const MatchOptions& opt) {
  double w = opt.use_node
                 ? compat::safe_log(compat::node_score(Q, q, G, u, opt.weights))
                 : 0.0;
  if (opt.use_edge) {
    for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q);
         ++it) {
      const NodeId qn = *it;
      const NodeId un = mapping[qn];
      if (un < 0) continue;  // edge to unmapped neighbor: scored later
      w += compat::safe_log(compat::edge_compat(
          Q.edge_label(q, qn), G.edge_label(u, un), G.edge_prob(u, un),
          opt.weights));
    }
  }
  return w;
}

namespace {
thread_local uint64_t s_assign_calls = 0;
thread_local uint64_t s_assign_fallbacks = 0;
}  // namespace
void reset_assign_stats() {
  s_assign_calls = 0;
  s_assign_fallbacks = 0;
}
AssignStats get_assign_stats() { return {s_assign_calls, s_assign_fallbacks}; }

double assign_upper_bound(const LabeledGraph& G, const LabeledGraph& Q,
                          const std::vector<NodeId>& mapping,
                          const std::vector<char>& used,
                          const std::vector<double>& max_node_log,
                          const MatchOptions& opt) {
  ++s_assign_calls;
  const double node_ub = node_upper_bound(Q, mapping, max_node_log);

  std::vector<NodeId> unmapped;
  for (NodeId q = 0; q < Q.num_nodes(); ++q)
    if (mapping[q] < 0) unmapped.push_back(q);
  const int n = static_cast<int>(unmapped.size());
  if (n == 0) return 0.0;

  // Guard: very large frontiers stay on the cheap node bound.
  if (n > 64) {
    ++s_assign_fallbacks;
    return node_ub;
  }

  // Reduce each row to its top-n candidates by weight. In a max-weight injective
  // assignment a row is outranked on at most n-1 columns, so one of its top-n is
  // always free; restricting to them leaves the optimum unchanged and caps the
  // shared column count at m <= n*n.
  std::vector<std::vector<NodeId>> pools(n);
  std::vector<std::vector<double>> wts(n);
  std::unordered_map<NodeId, int> col_of;
  std::vector<NodeId> cols;
  for (int i = 0; i < n; ++i) {
    const NodeId q = unmapped[i];
    std::vector<NodeId> pool = feasible_candidates(G, Q, q, mapping, used);
    if (pool.empty()) {
      ++s_assign_fallbacks;
      return node_ub;  // dead state -> node bound is safe
    }
    std::vector<double> w(pool.size());
    for (size_t k = 0; k < pool.size(); ++k)
      w[k] = pair_weight(G, Q, q, pool[k], mapping, opt);
    if (static_cast<int>(pool.size()) > n) {
      std::vector<int> idx(pool.size());
      for (size_t k = 0; k < idx.size(); ++k) idx[k] = static_cast<int>(k);
      std::nth_element(idx.begin(), idx.begin() + n, idx.end(),
                       [&](int a, int b) { return w[a] > w[b]; });
      idx.resize(n);
      std::vector<NodeId> rp(n);
      std::vector<double> rw(n);
      for (int k = 0; k < n; ++k) {
        rp[k] = pool[idx[k]];
        rw[k] = w[idx[k]];
      }
      pool.swap(rp);
      w.swap(rw);
    }
    for (NodeId u : pool)
      if (col_of.emplace(u, static_cast<int>(cols.size())).second)
        cols.push_back(u);
    pools[i] = std::move(pool);
    wts[i] = std::move(w);
  }
  const int m = static_cast<int>(cols.size());

  // No injective assignment possible (fewer distinct candidates than rows).
  if (n > m) {
    ++s_assign_fallbacks;
    return node_ub;
  }

  std::vector<std::vector<double>> cost(n, std::vector<double>(m, kBigCost));
  for (int i = 0; i < n; ++i) {
    for (size_t k = 0; k < pools[i].size(); ++k)
      cost[i][col_of[pools[i][k]]] = -wts[i][k];  // maximize w == minimize -w
  }

  std::vector<int> row_to_col;
  const double total = hungarian_min_cost(cost, row_to_col);

  // If any row had to take a forbidden (kBigCost) column, the injective
  // assignment was infeasible -> fall back to the node bound.
  for (int i = 0; i < n; ++i) {
    const int j = row_to_col[i];
    if (j < 0 || cost[i][j] >= kBigCost * 0.5) {
      ++s_assign_fallbacks;
      return node_ub;
    }
  }
  return -total;  // == sum of chosen weights == UB_assign
}

}  // namespace vf2prob
