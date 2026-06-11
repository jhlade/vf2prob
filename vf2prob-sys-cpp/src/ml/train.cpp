// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/ml/train.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <unordered_set>
#include <vector>

#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"

namespace vf2prob::ml {

namespace {

double sigmoid(double z) {
  if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
  const double e = std::exp(z);
  return e / (1.0 + e);
}

double logit_clamped(double p) {
  const double pe = std::min(1.0 - 1e-6, std::max(1e-6, p));
  return std::log(pe / (1.0 - pe));
}

// One labeled correspondence sample: features and binary target.
struct NodeSample {
  LabelId qlab;        // query / data labels (for the per-label table)
  LabelId ulab;
  double label_match;  // 0/1
  double y;            // 1 = true correspondence, 0 = negative
};
struct EdgeSample {
  LabelId qel;  // query / data edge labels (for the per-label edge table)
  LabelId uel;
  double label_match;
  double logit_p;
  double y;
};

// Gradient-descent logistic regression. features[i] is the feature vector
// (including a leading 1 for the intercept); w has the same length. L2 skips
// the intercept (index 0).
std::vector<double> fit_logistic(const std::vector<std::vector<double>>& X,
                                 const std::vector<double>& y, int dim,
                                 int epochs, double lr, double l2) {
  std::vector<double> w(dim, 0.0);
  const size_t N = X.size();
  if (N == 0) return w;
  for (int ep = 0; ep < epochs; ++ep) {
    std::vector<double> g(dim, 0.0);
    for (size_t i = 0; i < N; ++i) {
      double z = 0.0;
      for (int d = 0; d < dim; ++d) z += w[d] * X[i][d];
      const double err = sigmoid(z) - y[i];
      for (int d = 0; d < dim; ++d) g[d] += err * X[i][d];
    }
    for (int d = 0; d < dim; ++d) {
      g[d] /= static_cast<double>(N);
      if (d > 0) g[d] += l2 * w[d];  // sdon't regularize the intercept
      w[d] -= lr * g[d];
    }
  }
  return w;
}

}  // namespace

LabeledGraph make_observed(const LabeledGraph& clean, const TrainConfig& cfg,
                           uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::uniform_real_distribution<double> hp(cfg.true_p_min, 1.0);
  std::uniform_real_distribution<double> lp(cfg.pmin, cfg.decoy_p_max);
  const NodeId n = clean.num_nodes();
  const int nl = std::max(1, cfg.nlabels);
  const int el = std::max(1, cfg.elabels);

  // Node labels with flip noise (uniform-other, or structured i -> (i+1) mod n).
  std::vector<LabelId> labels = clean.node_labels();
  if (cfg.flip > 0.0 && nl > 1) {
    for (NodeId u = 0; u < n; ++u) {
      if (labels[u] == kNoLabel || uni(rng) >= cfg.flip) continue;
      labels[u] = cfg.structured_noise
                      ? (labels[u] + 1) % nl
                      : (labels[u] + 1 + static_cast<int>(rng() % (nl - 1))) % nl;
    }
  }

  auto key = [&](NodeId a, NodeId b) {
    if (a > b) std::swap(a, b);
    return static_cast<long long>(a) * n + b;
  };
  std::vector<LabeledGraph::Edge> edges;
  std::unordered_set<long long> present;

  // True edges: high existence probability, flip-noised labels.
  long n_true = 0;
  for (NodeId u = 0; u < n; ++u) {
    for (const NodeId* it = clean.neighbors_begin(u); it != clean.neighbors_end(u);
         ++it) {
      const NodeId v = *it;
      if (v <= u) continue;
      LabelId lab = clean.edge_label(u, v);
      if (cfg.flip > 0.0 && el > 1 && lab != kNoLabel && uni(rng) < cfg.flip)
        lab = cfg.structured_noise
                  ? (lab + 1) % el
                  : (lab + 1 + static_cast<int>(rng() % (el - 1))) % el;
      edges.emplace_back(u, v, lab, hp(rng));
      present.insert(key(u, v));
      ++n_true;
    }
  }

  // Decoy edges: spurious, low existence probability, random labels.
  const long n_decoy = std::llround(cfg.decoy_frac * static_cast<double>(n_true));
  std::uniform_int_distribution<NodeId> pick(0, n > 0 ? n - 1 : 0);
  for (long added = 0, attempts = 0;
       added < n_decoy && attempts < n_decoy * 20 + 100; ++attempts) {
    const NodeId u = pick(rng), v = pick(rng);
    if (u == v || present.count(key(u, v))) continue;
    present.insert(key(u, v));
    const LabelId lab = el > 0 ? static_cast<LabelId>(rng() % el) : kNoLabel;
    edges.emplace_back(u, v, lab, lp(rng));
    ++added;
  }

  return LabeledGraph(n, std::move(labels), edges);
}

static compat::Params train_core(
    const TrainConfig& cfg,
    const std::function<LabeledGraph(int)>& clean_provider) {
  std::vector<NodeSample> node_data;
  std::vector<EdgeSample> edge_data;
  std::mt19937_64 rng(cfg.seed);

  for (int inst = 0; inst < cfg.instances; ++inst) {
    // Clean graph (clean labels, certain edges) -> the "intended" query.
    const LabeledGraph Gclean = clean_provider(inst);
    // Observed data graph: high-prob true edges + low-prob decoys, flip-noised.
    const LabeledGraph Gnoisy = make_observed(Gclean, cfg, cfg.seed + 9001 + inst);

    const auto qs = data::sample_connected_queries(Gclean, cfg.qsize, cfg.qsize,
                                                   1, cfg.seed + 4001 + inst);
    if (qs.empty()) continue;
    const std::vector<NodeId>& S = qs[0];      // ground-truth images
    const LabeledGraph Q = induced_subgraph(Gclean, S);  // Q node i <-> S[i]
    const NodeId k = Q.num_nodes();

    std::uniform_int_distribution<NodeId> pick_node(0, Gnoisy.num_nodes() - 1);

    // Node correspondences.
    for (NodeId i = 0; i < k; ++i) {
      const NodeId t = S[i];
      node_data.push_back({Q.node_label(i), Gnoisy.node_label(t),
                           Q.node_label(i) == Gnoisy.node_label(t) ? 1.0 : 0.0,
                           1.0});
      for (int neg = 0; neg < cfg.neg_per_pos; ++neg) {
        NodeId u = pick_node(rng);
        if (u == t || Gnoisy.degree(u) < Q.degree(i)) continue;
        node_data.push_back(
            {Q.node_label(i), Gnoisy.node_label(u),
             Q.node_label(i) == Gnoisy.node_label(u) ? 1.0 : 0.0, 0.0});
      }
    }

    // Edge correspondences (true edge -> wrong existing edges as negatives).
    // Collect existing data edges once for negative sampling.
    std::vector<std::pair<NodeId, NodeId>> data_edges;
    for (NodeId u = 0; u < Gnoisy.num_nodes(); ++u)
      for (const NodeId* it = Gnoisy.neighbors_begin(u);
           it != Gnoisy.neighbors_end(u); ++it)
        if (*it > u) data_edges.emplace_back(u, *it);
    std::uniform_int_distribution<size_t> pick_edge(
        0, data_edges.empty() ? 0 : data_edges.size() - 1);

    for (NodeId i = 0; i < k; ++i) {
      for (const NodeId* it = Q.neighbors_begin(i); it != Q.neighbors_end(i);
           ++it) {
        const NodeId j = *it;
        if (j <= i) continue;
        const NodeId a = S[i], b = S[j];
        edge_data.push_back(
            {Q.edge_label(i, j), Gnoisy.edge_label(a, b),
             Q.edge_label(i, j) == Gnoisy.edge_label(a, b) ? 1.0 : 0.0,
             logit_clamped(Gnoisy.edge_prob(a, b)), 1.0});
        for (int neg = 0; neg < cfg.neg_per_pos && !data_edges.empty(); ++neg) {
          const auto [x, yv] = data_edges[pick_edge(rng)];
          edge_data.push_back(
              {Q.edge_label(i, j), Gnoisy.edge_label(x, yv),
               Q.edge_label(i, j) == Gnoisy.edge_label(x, yv) ? 1.0 : 0.0,
               logit_clamped(Gnoisy.edge_prob(x, yv)), 0.0});
        }
      }
    }
  }

  // Fit node logistic: features [1, label_match].
  std::vector<std::vector<double>> Xn;
  std::vector<double> yn;
  Xn.reserve(node_data.size());
  for (const auto& s : node_data) {
    Xn.push_back({1.0, s.label_match});
    yn.push_back(s.y);
  }
  const auto wn = fit_logistic(Xn, yn, 2, cfg.epochs, cfg.lr, cfg.l2);

  // Fit edge logistic: features [1, label_match, logit_p].
  std::vector<std::vector<double>> Xe;
  std::vector<double> ye;
  Xe.reserve(edge_data.size());
  for (const auto& s : edge_data) {
    Xe.push_back({1.0, s.label_match, s.logit_p});
    ye.push_back(s.y);
  }
  const auto we = fit_logistic(Xe, ye, 3, cfg.epochs, cfg.lr, cfg.l2);

  compat::Params p;  // keeps default eps
  if (wn.size() == 2) {
    p.node_w0 = wn[0];
    p.node_w_label = wn[1];
  }
  if (we.size() == 3) {
    p.edge_w0 = we[0];
    p.edge_w_label = we[1];
    p.edge_w_prob = we[2];
  }

  // Optional per-label node table: P(true | query-label, data-label), shrunk
  // toward the scalar model so sparse pairs stay sensible.
  if (cfg.per_label_node && cfg.nlabels > 0) {
    const int nl = cfg.nlabels;
    std::vector<double> sum_y(static_cast<size_t>(nl) * nl, 0.0);
    std::vector<double> cnt(static_cast<size_t>(nl) * nl, 0.0);
    for (const auto& s : node_data) {
      if (s.qlab < 0 || s.ulab < 0 || s.qlab >= nl || s.ulab >= nl) continue;
      const size_t idx = static_cast<size_t>(s.qlab) * nl + s.ulab;
      sum_y[idx] += s.y;
      cnt[idx] += 1.0;
    }
    const double a = 5.0;  // shrinkage strength toward the scalar prior
    std::vector<double> table(static_cast<size_t>(nl) * nl, 0.0);
    for (int q = 0; q < nl; ++q)
      for (int u = 0; u < nl; ++u) {
        const size_t idx = static_cast<size_t>(q) * nl + u;
        const double prior = compat::node_compat(q, u, p);  // scalar (n=0 here)
        table[idx] = (sum_y[idx] + a * prior) / (cnt[idx] + a);
      }
    p.n_node_labels = nl;
    p.node_table = std::move(table);
  }

  // Optional per-label edge table: a logistic with one bias per (q-edge-label,
  // u-edge-label) pair plus the shared logit(p) slope. Recovers e_prob too.
  if (cfg.per_label_edge && cfg.elabels > 0) {
    const int nel = cfg.elabels;
    const int dim = nel * nel + 1;  // per-pair bias + shared logit_p slope
    std::vector<std::vector<double>> Xpe;
    std::vector<double> ype;
    Xpe.reserve(edge_data.size());
    for (const auto& s : edge_data) {
      if (s.qel < 0 || s.uel < 0 || s.qel >= nel || s.uel >= nel) continue;
      std::vector<double> x(dim, 0.0);
      x[s.qel * nel + s.uel] = 1.0;
      x[nel * nel] = s.logit_p;
      Xpe.push_back(std::move(x));
      ype.push_back(s.y);
    }
    const auto wpe = fit_logistic(Xpe, ype, dim, cfg.epochs, cfg.lr, cfg.l2);
    if (static_cast<int>(wpe.size()) == dim) {
      p.n_edge_labels = nel;
      p.edge_table.assign(wpe.begin(), wpe.begin() + nel * nel);
      p.edge_w_prob = wpe[nel * nel];
    }
  }
  return p;
}

compat::Params train_weights(const TrainConfig& cfg) {
  return train_core(cfg, [&](int inst) {
    data::SynthParams sp;
    sp.n = cfg.n;
    sp.p = cfg.p;
    sp.nlabels = cfg.nlabels;
    sp.elabels = cfg.elabels;
    sp.pmin = 1.0;
    sp.flip = 0.0;
    sp.seed = cfg.seed + 1 + inst;
    return data::make_synthetic(sp);
  });
}

compat::Params train_weights_from_graph(const LabeledGraph& clean_base,
                                        const TrainConfig& cfg_in) {
  TrainConfig cfg = cfg_in;
  int nl = 1, el = 1;
  for (NodeId u = 0; u < clean_base.num_nodes(); ++u)
    nl = std::max(nl, clean_base.node_label(u) + 1);
  for (NodeId u = 0; u < clean_base.num_nodes(); ++u)
    for (const NodeId* it = clean_base.neighbors_begin(u);
         it != clean_base.neighbors_end(u); ++it)
      if (*it > u) el = std::max(el, clean_base.edge_label(u, *it) + 1);
  cfg.nlabels = nl;
  cfg.elabels = el;
  return train_core(cfg, [&](int) { return clean_base; });
}

}  // namespace vf2prob::ml
