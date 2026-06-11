// VF2-Prob, Jan Hladěna, FIM UHK
// tests auto-generated
// (1) bounds stay admissible under arbitrary weights; (2) training is
// noise-aware (more label noise => smaller learned label weight).
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "test_framework.hpp"
#include "vf2prob/algorithms/vf2prob.hpp"
#include "vf2prob/bounds.hpp"
#include "vf2prob/compat.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/matcher.hpp"
#include "vf2prob/ml/calibrate.hpp"
#include "vf2prob/ml/ranker.hpp"
#include "vf2prob/ml/train.hpp"
#include "vf2prob/ordering.hpp"

using namespace vf2prob;

static double score_w(const LabeledGraph& G, const LabeledGraph& Q,
                      const std::vector<NodeId>& mp, const compat::Params& P) {
  double s = 0.0;
  for (NodeId q = 0; q < Q.num_nodes(); ++q)
    if (mp[q] >= 0)
      s += compat::safe_log(
          compat::node_compat(Q.node_label(q), G.node_label(mp[q]), P));
  for (NodeId q = 0; q < Q.num_nodes(); ++q) {
    if (mp[q] < 0) continue;
    for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q);
         ++it) {
      const NodeId qn = *it;
      if (qn <= q || mp[qn] < 0) continue;
      s += compat::safe_log(
          compat::edge_compat(Q.edge_label(q, qn), G.edge_label(mp[q], mp[qn]),
                              G.edge_prob(mp[q], mp[qn]), P));
    }
  }
  return s;
}

static double brute_w(const LabeledGraph& G, const LabeledGraph& Q,
                      std::vector<NodeId>& mp, std::vector<char>& used,
                      const std::vector<NodeId>& order, const compat::Params& P) {
  NodeId qi = -1;
  for (NodeId q : order)
    if (mp[q] < 0) {
      qi = q;
      break;
    }
  if (qi < 0) return score_w(G, Q, mp, P);
  double best = kNegInf;
  for (NodeId u : feasible_candidates(G, Q, qi, mp, used)) {
    mp[qi] = u;
    used[u] = 1;
    best = std::max(best, brute_w(G, Q, mp, used, order, P));
    mp[qi] = -1;
    used[u] = 0;
  }
  return best;
}

TEST("ml: bounds remain admissible under arbitrary (learned-like) weights") {
  auto scalar = [](double w0, double wl, double e0, double el, double ep) {
    compat::Params p;
    p.node_w0 = w0;
    p.node_w_label = wl;
    p.edge_w0 = e0;
    p.edge_w_label = el;
    p.edge_w_prob = ep;
    return p;
  };
  const std::vector<compat::Params> wsets = {
      scalar(-1.0, 2.0, 0.0, 1.0, -0.5),      // negative edge-prob weight
      scalar(-2.5, 2.66, -3.0, 2.84, -0.12),  // resembles a learned set
      scalar(0.0, 0.0, 0.0, 0.0, 0.0),        // degenerate: all scores equal
  };
  const double kEps = 1e-7;
  for (const auto& P : wsets) {
    MatchOptions opt;
    opt.weights = P;
    for (uint64_t seed = 1; seed <= 12; ++seed) {
      data::SynthParams sp;
      sp.n = 7;
      sp.p = 0.5;
      sp.nlabels = 3;
      sp.elabels = 2;
      sp.pmin = 0.4;
      sp.seed = seed;
      const LabeledGraph G = data::make_synthetic(sp);
      const auto qs = data::sample_connected_queries(G, 4, 4, 1, seed + 300);
      if (qs.empty()) continue;
      const LabeledGraph Q = induced_subgraph(G, qs[0]);
      const std::vector<NodeId> order = failfirst_order(Q);
      const std::vector<double> mnl = precompute_max_node_log(G, Q, opt);

      std::vector<NodeId> mp(Q.num_nodes(), -1);
      std::vector<char> used(G.num_nodes(), 0);
      std::function<void()> visit = [&]() {
        std::vector<NodeId> mc = mp;
        std::vector<char> uc = used;
        const double tb = brute_w(G, Q, mc, uc, order, P);
        if (tb > kNegInf) {
          const double rem = tb - score_w(G, Q, mp, P);
          CHECK(node_upper_bound(Q, mp, mnl) + kEps >= rem);
          CHECK(assign_upper_bound(G, Q, mp, used, mnl, opt) + kEps >= rem);
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
}

TEST("ml: training is noise-aware (more noise => smaller label weight)") {
  ml::TrainConfig lo;
  lo.instances = 40;
  lo.qsize = 7;
  lo.seed = 2;
  lo.flip = 0.0;
  ml::TrainConfig hi = lo;
  hi.flip = 0.4;

  const compat::Params wl = ml::train_weights(lo);
  const compat::Params wh = ml::train_weights(hi);

  CHECK_MSG(wl.node_w_label > 0.0, "clean data should give positive label weight");
  CHECK_MSG(wl.node_w_label > wh.node_w_label,
            "more label noise should shrink the learned label weight");
}

TEST("ml: edge-probability weight is learned positive when uncertainty informs") {
  ml::TrainConfig cfg;
  cfg.instances = 50;
  cfg.qsize = 7;
  cfg.seed = 4;
  cfg.flip = 0.1;  // defaults give high-prob true edges + low-prob decoys
  const compat::Params w = ml::train_weights(cfg);
  CHECK_MSG(w.edge_w_prob > 0.2,
            "high edge probability should indicate true correspondences");
  CHECK_MSG(w.edge_w_label > 0.0, "matching edge labels should indicate true edges");
}

TEST("ml: a per-label node table keeps the bounds admissible") {
  compat::Params P;
  P.n_node_labels = 3;
  P.node_table = {0.90, 0.10, 0.05, 0.20, 0.80, 0.30, 0.05, 0.40, 0.70};
  MatchOptions opt;
  opt.weights = P;
  const double kEps = 1e-7;
  for (uint64_t seed = 1; seed <= 12; ++seed) {
    data::SynthParams sp;
    sp.n = 7;
    sp.p = 0.5;
    sp.nlabels = 3;
    sp.elabels = 2;
    sp.pmin = 0.4;
    sp.seed = seed;
    const LabeledGraph G = data::make_synthetic(sp);
    const auto qs = data::sample_connected_queries(G, 4, 4, 1, seed + 700);
    if (qs.empty()) continue;
    const LabeledGraph Q = induced_subgraph(G, qs[0]);
    const std::vector<NodeId> order = failfirst_order(Q);
    const std::vector<double> mnl = precompute_max_node_log(G, Q, opt);
    std::vector<NodeId> mp(Q.num_nodes(), -1);
    std::vector<char> used(G.num_nodes(), 0);
    std::function<void()> visit = [&]() {
      std::vector<NodeId> mc = mp;
      std::vector<char> uc = used;
      const double tb = brute_w(G, Q, mc, uc, order, P);
      if (tb > kNegInf) {
        const double rem = tb - score_w(G, Q, mp, P);
        CHECK(node_upper_bound(Q, mp, mnl) + kEps >= rem);
        CHECK(assign_upper_bound(G, Q, mp, used, mnl, opt) + kEps >= rem);
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

TEST("ml: per-label table captures structured label confusion") {
  ml::TrainConfig cfg;
  cfg.instances = 60;
  cfg.qsize = 7;
  cfg.seed = 3;
  cfg.nlabels = 4;
  cfg.flip = 0.3;
  cfg.per_label_node = true;
  cfg.structured_noise = true;  // label i tends to flip to (i+1) mod n
  const compat::Params w = ml::train_weights(cfg);
  CHECK(w.n_node_labels == 4);
  // The structured-confusion entry (i, i+1) should beat a generic mismatch
  // (i, i+2) for most query labels.
  int wins = 0;
  for (int i = 0; i < 4; ++i) {
    const double confuse = w.node_table[i * 4 + (i + 1) % 4];
    const double other = w.node_table[i * 4 + (i + 2) % 4];
    if (confuse > other) ++wins;
  }
  CHECK_MSG(wins >= 3, "structured-confusion entries should dominate");
}

TEST("ml: a per-label edge table keeps the bounds admissible") {
  compat::Params P;
  P.n_edge_labels = 2;
  P.edge_table = {1.0, -1.0, -0.5, 0.8};  // logit-space biases
  P.edge_w_prob = 1.5;
  MatchOptions opt;
  opt.weights = P;
  const double kEps = 1e-7;
  for (uint64_t seed = 1; seed <= 12; ++seed) {
    data::SynthParams sp;
    sp.n = 7;
    sp.p = 0.5;
    sp.nlabels = 3;
    sp.elabels = 2;
    sp.pmin = 0.4;
    sp.seed = seed;
    const LabeledGraph G = data::make_synthetic(sp);
    const auto qs = data::sample_connected_queries(G, 4, 4, 1, seed + 900);
    if (qs.empty()) continue;
    const LabeledGraph Q = induced_subgraph(G, qs[0]);
    const std::vector<NodeId> order = failfirst_order(Q);
    const std::vector<double> mnl = precompute_max_node_log(G, Q, opt);
    std::vector<NodeId> mp(Q.num_nodes(), -1);
    std::vector<char> used(G.num_nodes(), 0);
    std::function<void()> visit = [&]() {
      std::vector<NodeId> mc = mp;
      std::vector<char> uc = used;
      const double tb = brute_w(G, Q, mc, uc, order, P);
      if (tb > kNegInf) {
        const double rem = tb - score_w(G, Q, mp, P);
        CHECK(node_upper_bound(Q, mp, mnl) + kEps >= rem);
        CHECK(assign_upper_bound(G, Q, mp, used, mnl, opt) + kEps >= rem);
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

TEST("ml: per-label edge table captures structured edge-label confusion") {
  ml::TrainConfig cfg;
  cfg.instances = 60;
  cfg.qsize = 7;
  cfg.seed = 5;
  cfg.elabels = 3;
  cfg.flip = 0.3;
  cfg.per_label_edge = true;
  cfg.structured_noise = true;
  const compat::Params w = ml::train_weights(cfg);
  CHECK(w.n_edge_labels == 3);
  int wins = 0;
  for (int i = 0; i < 3; ++i) {
    const double confuse = w.edge_table[i * 3 + (i + 1) % 3];
    const double other = w.edge_table[i * 3 + (i + 2) % 3];
    if (confuse > other) ++wins;
  }
  CHECK_MSG(wins >= 2, "structured edge-confusion biases should dominate");
}

TEST("ml: Platt scaling reduces ECE on overconfident predictions") {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<double> pred;
  std::vector<int> y;
  for (int i = 0; i < 4000; ++i) {
    const double q = 0.02 + 0.96 * u(rng);             // true probability
    const int yi = u(rng) < q ? 1 : 0;
    const double lz = std::log(q / (1.0 - q));
    pred.push_back(1.0 / (1.0 + std::exp(-2.0 * lz)));  // overconfident (2x logit)
    y.push_back(yi);
  }
  const double ece_raw = ml::expected_calibration_error(pred, y);
  const ml::Platt c = ml::fit_platt(pred, y);
  std::vector<double> cal(pred.size());
  for (size_t i = 0; i < pred.size(); ++i) cal[i] = ml::apply_platt(c, pred[i]);
  const double ece_cal = ml::expected_calibration_error(cal, y);
  CHECK_MSG(ece_cal < ece_raw, "Platt should improve ECE on overconfident preds");
  CHECK_MSG(c.a > 0.2 && c.a < 0.9, "recovered slope should undo the 2x overconfidence");
}

TEST("ml: isotonic calibration reduces ECE and is monotone") {
  std::mt19937 rng(43);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<double> pred;
  std::vector<int> y;
  for (int i = 0; i < 4000; ++i) {
    const double q = 0.02 + 0.96 * u(rng);
    const int yi = u(rng) < q ? 1 : 0;
    const double lz = std::log(q / (1.0 - q));
    pred.push_back(1.0 / (1.0 + std::exp(-2.0 * lz)));  // overconfident
    y.push_back(yi);
  }
  const double ece_raw = ml::expected_calibration_error(pred, y);
  const ml::Isotonic iso = ml::fit_isotonic(pred, y);
  std::vector<double> cal(pred.size());
  for (size_t i = 0; i < pred.size(); ++i) cal[i] = ml::apply_isotonic(iso, pred[i]);
  CHECK_MSG(ml::expected_calibration_error(cal, y) < ece_raw,
            "isotonic should improve ECE");
  double prev = -1.0;
  bool mono = true;
  for (double p = 0.0; p <= 1.0; p += 0.05) {
    const double v = ml::apply_isotonic(iso, p);
    if (v + 1e-9 < prev) mono = false;
    prev = v;
  }
  CHECK_MSG(mono, "isotonic map must be non-decreasing");
}

TEST("ml: learned expansion-order ranker preserves the optimum (exactness)") {
  // Ordering is heuristic only: the learned ranker must never change the
  // returned maximum-likelihood optimum, for any weights.
  ml::TrainConfig cfg;
  cfg.instances = 20;
  cfg.n = 80;
  cfg.qsize = 8;
  cfg.nlabels = 3;
  cfg.elabels = 2;
  cfg.flip = 0.1;
  cfg.seed = 5;
  const std::vector<double> w = ml::train_ranker(cfg);
  CHECK_MSG(w.size() == static_cast<size_t>(ml::kRankFeats), "ranker should train");

  VF2ProbMatcher matcher(BoundKind::Node, SearchKind::DFS);
  int checked = 0;
  for (int inst = 0; inst < 8; ++inst) {
    data::SynthParams sp;
    sp.n = cfg.n;
    sp.nlabels = cfg.nlabels;
    sp.elabels = cfg.elabels;
    sp.pmin = 1.0;
    sp.flip = 0.0;
    sp.seed = cfg.seed + 50000 + inst;
    const LabeledGraph Gclean = data::make_synthetic(sp);
    const auto qs = data::sample_connected_queries(Gclean, cfg.qsize, cfg.qsize,
                                                   1, cfg.seed + 60000 + inst);
    if (qs.empty()) continue;
    const LabeledGraph Q = induced_subgraph(Gclean, qs[0]);
    const LabeledGraph Gobs = ml::make_observed(Gclean, cfg, cfg.seed + 70000 + inst);
    MatchOptions o;
    o.collect_metrics = false;
    o.timeout_s = 10.0;
    o.rank_w.clear();
    const MatchResult R0 = matcher.solve(Gobs, Q, o);
    o.rank_w = w;
    const MatchResult R1 = matcher.solve(Gobs, Q, o);
    if (R0.counters.timed_out || R1.counters.timed_out) continue;
    CHECK_MSG(std::fabs(R0.counters.best_loglik - R1.counters.best_loglik) < 1e-6,
              "learned ordering must return the same optimum as gain ordering");
    ++checked;
  }
  CHECK_MSG(checked > 0, "should verify at least one instance");
}
