// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/algorithms/vf2pp.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <unordered_map>

namespace vf2prob {

using Clock = std::chrono::steady_clock;

// VF2++ matching order: start from the highest-degree node, then repeatedly add
// the unordered node maximizing (#neighbors already ordered, degree), ties by
// (label, id). Produces a connected, constraint-maximizing order.
static std::vector<NodeId> vf2pp_order(const LabeledGraph& Q) {
  const int nq = Q.num_nodes();
  std::vector<NodeId> order;
  order.reserve(nq);
  std::vector<char> placed(nq, 0);
  std::vector<int> conn(nq, 0);  // # neighbors already placed

  auto better = [&](NodeId a, NodeId b) {
    if (conn[a] != conn[b]) return conn[a] > conn[b];
    if (Q.degree(a) != Q.degree(b)) return Q.degree(a) > Q.degree(b);
    if (Q.node_label(a) != Q.node_label(b))
      return Q.node_label(a) < Q.node_label(b);
    return a < b;
  };

  for (int placed_count = 0; placed_count < nq; ++placed_count) {
    NodeId best = -1;
    for (NodeId q = 0; q < nq; ++q) {
      if (placed[q]) continue;
      if (best < 0 || better(q, best)) best = q;
    }
    placed[best] = 1;
    order.push_back(best);
    for (const NodeId* it = Q.neighbors_begin(best);
         it != Q.neighbors_end(best); ++it)
      conn[*it] += 1;
  }
  return order;
}

// Label-wise neighborhood look-ahead: every unmapped query neighbor of q must
// be coverable by a distinct unused data neighbor of u with the same label.
static bool passes_lookahead(const LabeledGraph& G, const LabeledGraph& Q,
                             NodeId q, NodeId u,
                             const std::vector<NodeId>& mapping,
                             const std::vector<char>& used) {
  std::unordered_map<LabelId, int> need;
  for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q); ++it) {
    if (mapping[*it] < 0) need[Q.node_label(*it)] += 1;
  }
  if (need.empty()) return true;
  std::unordered_map<LabelId, int> avail;
  for (const NodeId* it = G.neighbors_begin(u); it != G.neighbors_end(u); ++it) {
    if (!used[*it]) avail[G.node_label(*it)] += 1;
  }
  for (const auto& [lab, c] : need) {
    auto f = avail.find(lab);
    if (f == avail.end() || f->second < c) return false;
  }
  return true;
}

MatchResult VF2ppMatcher::solve(const LabeledGraph& G, const LabeledGraph& Q,
                                const MatchOptions& opt) {
  MatchResult R;
  Counters& C = R.counters;
  const int nq = Q.num_nodes();
  if (nq == 0) {
    R.mapping = std::vector<NodeId>();
    C.solutions = 1;
    return R;
  }

  const std::vector<NodeId> order = vf2pp_order(Q);
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
      stop = true;
      return;
    }
    const NodeId q = order[i];
    const int deg_q = Q.degree(q);
    const LabelId lab_q = Q.node_label(q);

    // Frontier-restricted candidates: if q has a mapped neighbor, candidates
    // come from the intersection of the mapped neighbors' images' neighborhoods.
    std::vector<NodeId> images;
    for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q);
         ++it)
      if (mapping[*it] >= 0) images.push_back(mapping[*it]);

    std::vector<NodeId> cand;
    auto admit = [&](NodeId u) {
      return !used[u] && G.node_label(u) == lab_q && G.degree(u) >= deg_q;
    };
    if (images.empty()) {
      for (NodeId u = 0; u < G.num_nodes(); ++u)
        if (admit(u)) cand.push_back(u);
    } else {
      NodeId base = images[0];
      for (NodeId im : images)
        if (G.degree(im) < G.degree(base)) base = im;
      for (const NodeId* it = G.neighbors_begin(base);
           it != G.neighbors_end(base); ++it) {
        const NodeId u = *it;
        if (!admit(u)) continue;
        bool ok = true;
        for (NodeId im : images) {
          if (im == base) continue;
          if (!G.has_edge(u, im)) {
            ok = false;
            break;
          }
        }
        if (ok) cand.push_back(u);
      }
    }

    for (NodeId u : cand) {
      C.states += 1;
      // Consistency to all mapped neighbors is guaranteed by candidate
      // construction; apply the VF2++ label-wise look-ahead cut.
      if (!passes_lookahead(G, Q, q, u, mapping, used)) {
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

}  // namespace vf2prob
