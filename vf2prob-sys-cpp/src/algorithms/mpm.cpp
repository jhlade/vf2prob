// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/algorithms/mpm.hpp"

#include <algorithm>
#include <chrono>
#include <functional>

#include "vf2prob/compat.hpp"
#include "vf2prob/ordering.hpp"

namespace vf2prob {

using Clock = std::chrono::steady_clock;

MatchResult MpmMatcher::solve(const LabeledGraph& G, const LabeledGraph& Q,
                              const MatchOptions& opt) {
  MatchResult R;
  Counters& C = R.counters;
  const int nq = Q.num_nodes();
  if (nq == 0) {
    R.mapping = std::vector<NodeId>();
    C.best_loglik = 0.0;
    C.solutions = 1;
    return R;
  }

  const std::vector<NodeId> order = failfirst_order(Q);
  std::vector<NodeId> mapping(nq, -1);
  std::vector<char> used(G.num_nodes(), 0);
  double best = kNegInf;
  std::vector<NodeId> best_map;
  const auto t0 = Clock::now();
  bool stop = false;

  auto timed_out = [&]() {
    return opt.timeout_s > 0 &&
           std::chrono::duration<double>(Clock::now() - t0).count() >
               opt.timeout_s;
  };
  // Edge log-prob added by mapping q -> u: sum over already-mapped neighbors.
  auto gain = [&](NodeId q, NodeId u) {
    double g = 0.0;
    for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q);
         ++it) {
      const NodeId un = mapping[*it];
      if (un >= 0) g += compat::safe_log(G.edge_prob(u, un));
    }
    return g;
  };

  std::function<void(int, double)> dfs = [&](int i, double cur) {
    if (stop) return;
    if (timed_out() || (opt.max_states && C.states >= opt.max_states)) {
      C.timed_out = true;  // incomplete: time or state budget exhausted
      stop = true;
      return;
    }
    if (i == nq) {
      C.solutions += 1;
      if (cur > best) {
        best = cur;
        best_map = mapping;
      }
      return;
    }
    if (cur <= best) {  // bound: remaining edge log-probs are <= 0
      C.pruned += 1;
      return;
    }

    const NodeId q = order[i];
    const LabelId lab = Q.node_label(q);
    std::vector<std::pair<double, NodeId>> ranked;
    for (NodeId u : feasible_candidates(G, Q, q, mapping, used)) {
      if (G.node_label(u) == lab)  // hard label predicate
        ranked.emplace_back(gain(q, u), u);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
      if (a.first != b.first) return a.first > b.first;  // best prob first
      return a.second < b.second;
    });

    for (const auto& [g, u] : ranked) {
      C.states += 1;
      mapping[q] = u;
      used[u] = 1;
      dfs(i + 1, cur + g);
      mapping[q] = -1;
      used[u] = 0;
      if (stop) return;
    }
  };

  dfs(0, 0.0);
  C.best_loglik = best;
  if (best > kNegInf) R.mapping = best_map;
  return R;
}

}  // namespace vf2prob
