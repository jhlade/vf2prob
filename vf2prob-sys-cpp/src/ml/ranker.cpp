// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/ml/ranker.hpp"

#include <cmath>

#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/ordering.hpp"

namespace vf2prob::ml {

namespace {
double sigmoid(double z) {
  if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
  const double e = std::exp(z);
  return e / (1.0 + e);
}
}  // namespace

RankFeatures rank_features(const LabeledGraph& G, const LabeledGraph& Q, NodeId q,
                           NodeId u, const std::vector<NodeId>& mapping,
                           const compat::Params& w, bool use_node, bool use_edge) {
  RankFeatures f;
  f.nc = compat::node_compat(Q.node_label(q), G.node_label(u), w);
  double g = use_node ? compat::safe_log(f.nc) : 0.0;
  if (use_edge) {
    for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q); ++it) {
      const NodeId qn = *it;
      const NodeId un = mapping[qn];
      if (un < 0) continue;
      g += compat::safe_log(compat::edge_compat(
          Q.edge_label(q, qn), G.edge_label(u, un), G.edge_prob(u, un), w));
    }
  }
  f.gain = g;
  const double du = static_cast<double>(G.degree(u));
  f.pressure = static_cast<double>(Q.degree(q)) / (du + 1.0);
  f.target = std::log1p(du);
  return f;
}

double rank_score(const std::vector<double>& w, const RankFeatures& f) {
  if (w.size() < static_cast<size_t>(kRankFeats)) return f.gain;  // inactive
  const auto x = feature_vector(f);
  double z = 0.0;
  for (int j = 0; j < kRankFeats; ++j) z += w[j] * x[j];
  return z;
}

std::vector<double> train_ranker(const TrainConfig& cfg, const LabeledGraph* base) {
  const compat::Params w0;  // default fixed scorer, as used by the search default
  std::vector<std::array<double, kRankFeats>> X;
  std::vector<int> Y;

  for (int inst = 0; inst < cfg.instances; ++inst) {
    data::SynthParams sp;
    sp.n = cfg.n;
    sp.p = cfg.p;
    sp.nlabels = cfg.nlabels;
    sp.elabels = cfg.elabels;
    sp.pmin = 1.0;
    sp.flip = 0.0;
    sp.seed = cfg.seed + 1 + inst;
    const LabeledGraph Gclean = base ? *base : data::make_synthetic(sp);
    const auto qs = data::sample_connected_queries(Gclean, cfg.qsize, cfg.qsize,
                                                   1, cfg.seed + 9001 + inst);
    if (qs.empty()) continue;
    const std::vector<NodeId>& S = qs[0];
    const LabeledGraph Q = induced_subgraph(Gclean, S);
    const LabeledGraph Gobs = make_observed(Gclean, cfg, cfg.seed + 11001 + inst);

    const std::vector<NodeId> order = failfirst_order(Q);
    std::vector<NodeId> mapping(Q.num_nodes(), -1);
    std::vector<char> used(Gobs.num_nodes(), 0);
    for (NodeId qi : order) {
      const auto cands = feasible_candidates(Gobs, Q, qi, mapping, used);
      const NodeId truth = S[qi];
      for (NodeId u : cands) {
        const RankFeatures f =
            rank_features(Gobs, Q, qi, u, mapping, w0, true, true);
        X.push_back(feature_vector(f));
        Y.push_back(u == truth ? 1 : 0);
      }
      mapping[qi] = truth;  // advance the ground-truth partial mapping
      if (truth >= 0 && static_cast<size_t>(truth) < used.size()) used[truth] = 1;
    }
  }
  if (X.empty()) return {};

  // Logistic regression (batch gradient descent + L2; bias unregularized).
  std::vector<double> w(kRankFeats, 0.0);
  const double lr = cfg.lr > 0.0 ? cfg.lr : 0.1;
  const double l2 = cfg.l2;
  const int epochs = cfg.epochs > 0 ? cfg.epochs : 400;
  const double N = static_cast<double>(X.size());
  for (int ep = 0; ep < epochs; ++ep) {
    std::vector<double> grad(kRankFeats, 0.0);
    for (size_t i = 0; i < X.size(); ++i) {
      double z = 0.0;
      for (int j = 0; j < kRankFeats; ++j) z += w[j] * X[i][j];
      const double err = sigmoid(z) - Y[i];
      for (int j = 0; j < kRankFeats; ++j) grad[j] += err * X[i][j];
    }
    for (int j = 0; j < kRankFeats; ++j)
      w[j] -= lr * (grad[j] / N + (j == 0 ? 0.0 : l2 * w[j]));
  }
  return w;
}

}  // namespace vf2prob::ml
