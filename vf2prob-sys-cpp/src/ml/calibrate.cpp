// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/ml/calibrate.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"

namespace vf2prob::ml {

namespace {
double sigmoid(double z) {
  if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
  const double e = std::exp(z);
  return e / (1.0 + e);
}
double logit(double p) {
  const double pe = std::min(1.0 - 1e-9, std::max(1e-9, p));
  return std::log(pe / (1.0 - pe));
}
}  // namespace

double expected_calibration_error(const std::vector<double>& pred,
                                  const std::vector<int>& y, int bins) {
  const size_t N = pred.size();
  if (N == 0 || bins <= 0) return 0.0;
  std::vector<double> conf(bins, 0.0), acc(bins, 0.0);
  std::vector<int> cnt(bins, 0);
  for (size_t i = 0; i < N; ++i) {
    int b = static_cast<int>(pred[i] * bins);
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    conf[b] += pred[i];
    acc[b] += y[i];
    cnt[b] += 1;
  }
  double e = 0.0;
  for (int b = 0; b < bins; ++b) {
    if (cnt[b] == 0) continue;
    const double c = conf[b] / cnt[b];
    const double a = acc[b] / cnt[b];
    e += (static_cast<double>(cnt[b]) / N) * std::fabs(c - a);
  }
  return e;
}

std::vector<ReliabilityBin> reliability_bins(const std::vector<double>& pred,
                                             const std::vector<int>& y, int bins) {
  std::vector<double> conf(bins, 0.0), acc(bins, 0.0);
  std::vector<int> cnt(bins, 0);
  for (size_t i = 0; i < pred.size(); ++i) {
    int b = static_cast<int>(pred[i] * bins);
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    conf[b] += pred[i];
    acc[b] += y[i];
    cnt[b] += 1;
  }
  std::vector<ReliabilityBin> out;
  for (int b = 0; b < bins; ++b)
    if (cnt[b] > 0) out.push_back({conf[b] / cnt[b], acc[b] / cnt[b], cnt[b]});
  return out;
}

Platt fit_platt(const std::vector<double>& pred, const std::vector<int>& y,
                int epochs, double lr) {
  const size_t N = pred.size();
  if (N == 0) return {};
  std::vector<double> z(N);
  for (size_t i = 0; i < N; ++i) z[i] = logit(pred[i]);
  double a = 1.0, b = 0.0;
  for (int ep = 0; ep < epochs; ++ep) {
    double ga = 0.0, gb = 0.0;
    for (size_t i = 0; i < N; ++i) {
      const double err = sigmoid(a * z[i] + b) - y[i];
      ga += err * z[i];
      gb += err;
    }
    a -= lr * ga / N;
    b -= lr * gb / N;
  }
  return {a, b};
}

double apply_platt(const Platt& c, double p) {
  return sigmoid(c.a * logit(p) + c.b);
}

Isotonic fit_isotonic(const std::vector<double>& pred, const std::vector<int>& y) {
  Isotonic c;
  const size_t N = pred.size();
  if (N == 0) return c;
  std::vector<size_t> idx(N);
  for (size_t i = 0; i < N; ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(),
            [&](size_t a, size_t b) { return pred[a] < pred[b]; });
  // Pool Adjacent Violators: merge while the previous block mean exceeds the
  // current one, enforcing a non-decreasing fit.
  std::vector<double> bsum, bw, bxhi;
  for (size_t k = 0; k < N; ++k) {
    const size_t i = idx[k];
    bsum.push_back(static_cast<double>(y[i]));
    bw.push_back(1.0);
    bxhi.push_back(pred[i]);
    while (bsum.size() >= 2) {
      const size_t m = bsum.size();
      if (bsum[m - 2] / bw[m - 2] <= bsum[m - 1] / bw[m - 1]) break;
      bsum[m - 2] += bsum[m - 1];
      bw[m - 2] += bw[m - 1];
      bxhi[m - 2] = bxhi[m - 1];
      bsum.pop_back();
      bw.pop_back();
      bxhi.pop_back();
    }
  }
  for (size_t b = 0; b < bsum.size(); ++b) {
    c.x.push_back(bxhi[b]);
    c.v.push_back(bsum[b] / bw[b]);
  }
  return c;
}

double apply_isotonic(const Isotonic& c, double p) {
  if (c.x.empty()) return p;
  const auto it = std::lower_bound(c.x.begin(), c.x.end(), p);
  if (it == c.x.end()) return c.v.back();
  return c.v[static_cast<size_t>(it - c.x.begin())];
}

void node_correspondence_samples(const TrainConfig& cfg, const compat::Params& w,
                                 uint64_t base_seed, int instances,
                                 std::vector<double>& pred, std::vector<int>& y,
                                 const LabeledGraph* base) {
  std::mt19937_64 rng(base_seed);
  for (int inst = 0; inst < instances; ++inst) {
    data::SynthParams sp;
    sp.n = cfg.n;
    sp.p = cfg.p;
    sp.nlabels = cfg.nlabels;
    sp.elabels = cfg.elabels;
    sp.pmin = 1.0;
    sp.flip = 0.0;
    sp.seed = base_seed + 1 + inst;
    const LabeledGraph Gclean = base ? *base : data::make_synthetic(sp);
    const auto qs = data::sample_connected_queries(Gclean, cfg.qsize, cfg.qsize,
                                                   1, base_seed + 3001 + inst);
    if (qs.empty()) continue;
    const std::vector<NodeId>& S = qs[0];
    const LabeledGraph Q = induced_subgraph(Gclean, S);
    const LabeledGraph Gobs = make_observed(Gclean, cfg, base_seed + 6001 + inst);
    std::uniform_int_distribution<NodeId> pick(0, Gobs.num_nodes() - 1);

    for (NodeId i = 0; i < Q.num_nodes(); ++i) {
      const NodeId t = S[i];
      pred.push_back(compat::node_compat(Q.node_label(i), Gobs.node_label(t), w));
      y.push_back(1);
      for (int neg = 0; neg < cfg.neg_per_pos; ++neg) {
        const NodeId u = pick(rng);
        if (u == t || Gobs.degree(u) < Q.degree(i)) continue;
        pred.push_back(
            compat::node_compat(Q.node_label(i), Gobs.node_label(u), w));
        y.push_back(0);
      }
    }
  }
}

}  // namespace vf2prob::ml
