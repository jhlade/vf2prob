// VF2-Prob, Jan Hladěna, FIM UHK
// tests auto-generated
// MPM (most-probable-match uncertain-graph baseline): finds a valid embedding,
// and returns the one maximizing sum log p(edge) (checked against brute force).
#include <algorithm>
#include <functional>
#include <vector>

#include "test_framework.hpp"
#include "vf2prob/compat.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/ordering.hpp"
#include "vf2prob/registry.hpp"

using namespace vf2prob;

static bool valid_embedding(const LabeledGraph& G, const LabeledGraph& Q,
                            const std::vector<NodeId>& mp) {
  std::vector<char> seen(G.num_nodes(), 0);
  for (NodeId q = 0; q < Q.num_nodes(); ++q) {
    const NodeId u = mp[q];
    if (u < 0 || u >= G.num_nodes() || seen[u]) return false;
    seen[u] = 1;
    if (Q.node_label(q) != G.node_label(u)) return false;
  }
  for (NodeId q = 0; q < Q.num_nodes(); ++q)
    for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q); ++it)
      if (*it > q && !G.has_edge(mp[q], mp[*it])) return false;
  return true;
}

// Brute-force max sum log p(edge) over structural + hard-label embeddings.
static double brute_best(const LabeledGraph& G, const LabeledGraph& Q) {
  const std::vector<NodeId> order = failfirst_order(Q);
  std::vector<NodeId> mp(Q.num_nodes(), -1);
  std::vector<char> used(G.num_nodes(), 0);
  double best = kNegInf;
  std::function<void(int, double)> rec = [&](int i, double cur) {
    if (i == static_cast<int>(order.size())) {
      best = std::max(best, cur);
      return;
    }
    const NodeId q = order[i];
    for (NodeId u : feasible_candidates(G, Q, q, mp, used)) {
      if (G.node_label(u) != Q.node_label(q)) continue;
      double g = 0.0;
      for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q); ++it)
        if (mp[*it] >= 0) g += compat::safe_log(G.edge_prob(u, mp[*it]));
      mp[q] = u;
      used[u] = 1;
      rec(i + 1, cur + g);
      mp[q] = -1;
      used[u] = 0;
    }
  };
  rec(0, 0.0);
  return best;
}

TEST("mpm: finds a valid embedding and the most probable one") {
  for (uint64_t seed = 1; seed <= 20; ++seed) {
    data::SynthParams sp;
    sp.n = 14;
    sp.p = 0.35;
    sp.nlabels = 3;
    sp.pmin = 0.5;  // genuine edge uncertainty
    sp.seed = seed;
    const LabeledGraph G = data::make_synthetic(sp);
    const auto qs = data::sample_connected_queries(G, 4, 5, 1, seed + 400);
    if (qs.empty()) continue;
    const LabeledGraph Q = induced_subgraph(G, qs[0]);  // embeddable by design

    MatchOptions opt;
    opt.timeout_s = 10.0;
    auto m = make_matcher("mpm");
    const MatchResult R = m->solve(G, Q, opt);
    CHECK_MSG(R.mapping.has_value(), "mpm: no embedding found");
    if (R.mapping) CHECK(valid_embedding(G, Q, *R.mapping));
    CHECK_NEAR(R.counters.best_loglik, brute_best(G, Q), 1e-6);
  }
}
