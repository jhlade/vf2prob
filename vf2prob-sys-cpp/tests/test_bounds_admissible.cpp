// VF2-Prob, Jan Hladěna, FIM UHK
// tests auto-generated
// Both upper bounds must be admissible — never below the true best achievable
// remaining log-likelihood — and the assignment bound must be no looser than the
// node bound.
#include <functional>
#include <vector>

#include "test_framework.hpp"
#include "vf2prob/bounds.hpp"
#include "vf2prob/compat.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/matcher.hpp"
#include "vf2prob/ordering.hpp"

using namespace vf2prob;

static double score(const LabeledGraph& G, const LabeledGraph& Q,
                    const std::vector<NodeId>& mp) {
  double s = 0.0;
  for (NodeId q = 0; q < Q.num_nodes(); ++q)
    if (mp[q] >= 0)
      s += compat::safe_log(
          compat::node_compat(Q.node_label(q), G.node_label(mp[q])));
  for (NodeId q = 0; q < Q.num_nodes(); ++q) {
    if (mp[q] < 0) continue;
    for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q);
         ++it) {
      const NodeId qn = *it;
      if (qn <= q || mp[qn] < 0) continue;
      const bool lm = Q.edge_label(q, qn) == G.edge_label(mp[q], mp[qn]);
      s += compat::safe_log(compat::edge_compat(lm, G.edge_prob(mp[q], mp[qn])));
    }
  }
  return s;
}

// Best full-embedding log-likelihood extending the partial mapping (-inf none).
static double brute(const LabeledGraph& G, const LabeledGraph& Q,
                    std::vector<NodeId>& mp, std::vector<char>& used,
                    const std::vector<NodeId>& order) {
  NodeId qi = -1;
  for (NodeId q : order)
    if (mp[q] < 0) {
      qi = q;
      break;
    }
  if (qi < 0) return score(G, Q, mp);
  double best = kNegInf;
  for (NodeId u : feasible_candidates(G, Q, qi, mp, used)) {
    mp[qi] = u;
    used[u] = 1;
    best = std::max(best, brute(G, Q, mp, used, order));
    mp[qi] = -1;
    used[u] = 0;
  }
  return best;
}

TEST("bounds: node and assignment bounds are admissible on every state") {
  MatchOptions opt;
  opt.use_node = true;
  opt.use_edge = true;
  const double kEps = 1e-7;

  for (uint64_t seed = 1; seed <= 40; ++seed) {
    data::SynthParams sp;
    sp.n = 7;
    sp.p = 0.5;
    sp.nlabels = 3;
    sp.elabels = 2;
    sp.pmin = 0.4;
    sp.seed = seed;
    const LabeledGraph G = data::make_synthetic(sp);

    const auto qs = data::sample_connected_queries(G, 4, 4, 1, seed + 100);
    if (qs.empty()) continue;
    const LabeledGraph Q = induced_subgraph(G, qs[0]);

    const std::vector<NodeId> order = failfirst_order(Q);
    const std::vector<double> mnl = precompute_max_node_log(G, Q, opt);

    std::vector<NodeId> mp(Q.num_nodes(), -1);
    std::vector<char> used(G.num_nodes(), 0);

    std::function<void()> visit = [&]() {
      std::vector<NodeId> mp_copy = mp;
      std::vector<char> used_copy = used;
      const double true_best = brute(G, Q, mp_copy, used_copy, order);
      if (true_best > kNegInf) {
        const double cur = score(G, Q, mp);
        const double rem = true_best - cur;
        const double ub_n = node_upper_bound(Q, mp, mnl);
        const double ub_a = assign_upper_bound(G, Q, mp, used, mnl, opt);
        CHECK_MSG(ub_n + kEps >= rem, "node bound below true remaining");
        CHECK_MSG(ub_a + kEps >= rem, "ASSIGN bound below true remaining");
        CHECK_MSG(ub_a <= ub_n + kEps, "assign bound looser than node bound");
      }
      NodeId qi = -1;
      for (NodeId q : order)
        if (mp[q] < 0) {
          qi = q;
          break;
        }
      if (qi < 0) return;
      for (NodeId u : feasible_candidates(G, Q, qi, mp, used)) {
        mp[qi] = u;
        used[u] = 1;
        visit();
        mp[qi] = -1;
        used[u] = 0;
      }
    };
    visit();
  }
}
