// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/algorithms/vf2.hpp"

#include <algorithm>
#include <chrono>
#include <functional>

namespace vf2prob {

using Clock = std::chrono::steady_clock;

MatchResult vf2_first_embedding(const LabeledGraph& G, const LabeledGraph& Q,
                                const MatchOptions& opt) {
  MatchResult R;
  Counters& C = R.counters;
  const int nq = Q.num_nodes();
  if (nq == 0) {
    R.mapping = std::vector<NodeId>();
    C.solutions = 1;
    return R;
  }

  // Fail-first order: decreasing degree, ties by (label, id).
  std::vector<NodeId> order(nq);
  for (NodeId i = 0; i < nq; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](NodeId a, NodeId b) {
    if (Q.degree(a) != Q.degree(b)) return Q.degree(a) > Q.degree(b);
    if (Q.node_label(a) != Q.node_label(b))
      return Q.node_label(a) < Q.node_label(b);
    return a < b;
  });

  std::vector<NodeId> mapping(nq, -1);
  std::vector<char> used(G.num_nodes(), 0);
  const auto t0 = Clock::now();
  bool stop = false;

  auto timed_out = [&]() {
    if (opt.timeout_s <= 0) return false;
    return std::chrono::duration<double>(Clock::now() - t0).count() >
           opt.timeout_s;
  };

  std::function<void(int)> dfs = [&](int i) {
    if (stop) return;
    if (timed_out() || (opt.max_states && C.states >= opt.max_states)) {
      C.timed_out = true;  // incomplete: time or state budget exhausted
      stop = true;
      return;
    }
    if (i == nq) {
      C.solutions += 1;
      if (!R.mapping) R.mapping = mapping;
      stop = true;  // first embedding is enough
      return;
    }
    const NodeId q = order[i];
    const int deg_q = Q.degree(q);
    const LabelId lab_q = Q.node_label(q);

    // Candidates: same label, sufficient degree, unused. Try lower degree
    // first (classic VF2 ordering heuristic).
    std::vector<NodeId> cand;
    for (NodeId u = 0; u < G.num_nodes(); ++u) {
      if (!used[u] && G.node_label(u) == lab_q && G.degree(u) >= deg_q)
        cand.push_back(u);
    }
    std::sort(cand.begin(), cand.end(), [&](NodeId a, NodeId b) {
      if (G.degree(a) != G.degree(b)) return G.degree(a) < G.degree(b);
      return a < b;
    });

    for (NodeId u : cand) {
      C.states += 1;
      bool feasible = true;
      for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q);
           ++it) {
        const NodeId qn = *it;
        if (mapping[qn] >= 0 && !G.has_edge(u, mapping[qn])) {
          feasible = false;
          break;
        }
      }
      if (!feasible) {
        C.pruned += 1;
        continue;
      }
      mapping[q] = u;
      used[u] = 1;
      dfs(i + 1);
      mapping[q] = -1;
      used[u] = 0;
      if (stop) return;
    }
  };

  dfs(0);
  return R;
}

MatchResult VF2Matcher::solve(const LabeledGraph& G, const LabeledGraph& Q,
                              const MatchOptions& opt) {
  return vf2_first_embedding(G, Q, opt);
}

}  // namespace vf2prob
