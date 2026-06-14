// VF2-Prob, Jan Hladěna, FIM UHK
// tests auto-generated
// Continuous (geometric) node compatibility: the RBF kernel keeps every score in
// (eps,1], so the admissible bounds and the four-variant optimum guarantee carry
// over unchanged from the categorical case. This is the exactness gate for the
// IAM / pattern-recognition experiment.
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "test_framework.hpp"
#include "vf2prob/bounds.hpp"
#include "vf2prob/compat.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/matcher.hpp"
#include "vf2prob/ordering.hpp"
#include "vf2prob/registry.hpp"

using namespace vf2prob;

// Topology from make_synthetic, with seeded 2-D coordinates attached so the only
// discriminating attribute is geometric (labels are constant in these tests).
static LabeledGraph with_coords(LabeledGraph G, uint64_t seed, double scale) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(0.0, scale);
  std::vector<double> c(static_cast<size_t>(G.num_nodes()) * 2);
  for (double& v : c) v = u(rng);
  G.set_node_coords(2, std::move(c));
  return G;
}

TEST("continuous kernel: node_compat_coords in (eps,1] and monotone in distance") {
  compat::Params w;
  w.node_kernel_h = 1.0;
  const double a[2] = {0.0, 0.0};
  const double same[2] = {0.0, 0.0};
  const double mid[2] = {1.0, 0.0};
  const double far[2] = {5.0, 0.0};
  const double s0 = compat::node_compat_coords(a, same, 2, w);
  const double s1 = compat::node_compat_coords(a, mid, 2, w);
  const double s2 = compat::node_compat_coords(a, far, 2, w);
  CHECK_NEAR(s0, 1.0, 1e-9);                        // identical coords => 1
  CHECK_MSG(s1 > w.eps && s1 < s0, "closer scores higher, above the floor");
  CHECK_MSG(s2 >= w.eps && s2 <= s1, "farther scores lower, floored at eps");
  CHECK_MSG(s2 <= 1.0, "score never exceeds 1");
}

TEST("continuous kernel: four VF2-Prob variants agree on the optimum") {
  const std::vector<std::string> variants = {
      "vf2prob", "vf2prob-assign", "vf2prob-astar", "vf2prob-astar-assign"};

  for (uint64_t seed = 1; seed <= 25; ++seed) {
    data::SynthParams sp;
    sp.n = 16;
    sp.p = 0.30;
    sp.nlabels = 1;   // constant labels: geometry is the only node signal
    sp.elabels = 1;
    sp.pmin = 1.0;    // certain edges: isolate the node kernel
    sp.flip = 0.0;
    sp.seed = seed;
    const LabeledGraph G = with_coords(data::make_synthetic(sp), seed * 7 + 1, 5.0);
    const auto qs = data::sample_connected_queries(G, 5, 6, 1, seed + 50);
    if (qs.empty()) continue;
    const LabeledGraph Q = induced_subgraph(G, qs[0]);  // carries coordinates

    MatchOptions opt;
    opt.timeout_s = 10.0;
    opt.collect_metrics = false;
    opt.weights.node_kernel = compat::NodeKernel::RBF;
    opt.weights.node_kernel_h = 1.5;

    double ref = 0.0;
    bool have_ref = false;
    for (const auto& v : variants) {
      auto m = make_matcher(v);
      const MatchResult R = m->solve(G, Q, opt);
      CHECK_MSG(!R.counters.timed_out, "continuous variant timed out");
      CHECK_MSG(R.mapping.has_value(), "continuous variant found no embedding");
      if (!have_ref) {
        ref = R.counters.best_loglik;
        have_ref = true;
      } else {
        CHECK_NEAR(R.counters.best_loglik, ref, 1e-6);
      }
    }
  }
}

static double score_rbf(const LabeledGraph& G, const LabeledGraph& Q,
                        const std::vector<NodeId>& mp, const compat::Params& w) {
  double s = 0.0;
  for (NodeId q = 0; q < Q.num_nodes(); ++q)
    if (mp[q] >= 0) s += compat::safe_log(compat::node_score(Q, q, G, mp[q], w));
  for (NodeId q = 0; q < Q.num_nodes(); ++q) {
    if (mp[q] < 0) continue;
    for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q); ++it) {
      const NodeId qn = *it;
      if (qn <= q || mp[qn] < 0) continue;
      const bool lm = Q.edge_label(q, qn) == G.edge_label(mp[q], mp[qn]);
      s += compat::safe_log(compat::edge_compat(lm, G.edge_prob(mp[q], mp[qn]), w));
    }
  }
  return s;
}

static double brute(const LabeledGraph& G, const LabeledGraph& Q,
                    std::vector<NodeId>& mp, std::vector<char>& used,
                    const std::vector<NodeId>& order, const compat::Params& w) {
  NodeId qi = -1;
  for (NodeId q : order)
    if (mp[q] < 0) {
      qi = q;
      break;
    }
  if (qi < 0) return score_rbf(G, Q, mp, w);
  double best = kNegInf;
  for (NodeId u : feasible_candidates(G, Q, qi, mp, used)) {
    mp[qi] = u;
    used[u] = 1;
    best = std::max(best, brute(G, Q, mp, used, order, w));
    mp[qi] = -1;
    used[u] = 0;
  }
  return best;
}

TEST("continuous kernel: node and assignment bounds stay admissible") {
  MatchOptions opt;
  opt.use_node = true;
  opt.use_edge = true;
  opt.weights.node_kernel = compat::NodeKernel::RBF;
  opt.weights.node_kernel_h = 1.5;
  const double kEps = 1e-7;

  for (uint64_t seed = 1; seed <= 30; ++seed) {
    data::SynthParams sp;
    sp.n = 8;
    sp.p = 0.5;
    sp.nlabels = 1;
    sp.elabels = 1;
    sp.pmin = 1.0;
    sp.seed = seed;
    const LabeledGraph G = with_coords(data::make_synthetic(sp), seed * 13 + 3, 4.0);

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
      const double true_best = brute(G, Q, mp_copy, used_copy, order, opt.weights);
      if (true_best > kNegInf) {
        const double cur = score_rbf(G, Q, mp, opt.weights);
        const double rem = true_best - cur;
        const double ub_n = node_upper_bound(Q, mp, mnl);
        const double ub_a = assign_upper_bound(G, Q, mp, used, mnl, opt);
        CHECK_MSG(ub_n + kEps >= rem, "node bound below true remaining (RBF)");
        CHECK_MSG(ub_a + kEps >= rem, "assign bound below true remaining (RBF)");
        CHECK_MSG(ub_a <= ub_n + kEps, "assign bound looser than node bound (RBF)");
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
