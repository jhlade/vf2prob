// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <array>
#include <vector>

#include "vf2prob/compat.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/ml/train.hpp"

namespace vf2prob::ml {

// Learned expansion-order ranker for the DFS branch-and-bound. It scores each
// structurally-feasible candidate mapping q -> u and orders expansion by that
// score. Because the order of expansion affects ONLY efficiency (a better order
// finds a high-likelihood incumbent sooner, so the admissible bound prunes more)
// and never which embeddings are feasible or which optimum is returned, the
// exact maximum-likelihood guarantee is preserved for any ranker weights.
//
// This is the paper's "learning-guided yet still exact" variant: a representative
// of the learning-based family that we can train and run ourselves, in contrast
// to inexact GNN matchers.

constexpr int kRankFeats = 5;  // [bias, gain, degree-pressure, log-target-deg, node-compat]

struct RankFeatures {
  double gain = 0.0;      // incremental log-likelihood of q->u (node + mapped edges)
  double pressure = 0.0;  // deg_Q(q) / (deg_G(u)+1): structural tightness
  double target = 0.0;    // log1p(deg_G(u)): target-vertex degree
  double nc = 1.0;        // node compatibility s_v(q,u)
};

// Feature vector fed to the linear model (scaled to O(1) ranges so a single
// learning rate trains stably). Identical in training and inference.
inline std::array<double, kRankFeats> feature_vector(const RankFeatures& f) {
  return {1.0, 0.2 * f.gain, f.pressure, 0.2 * f.target, f.nc};
}

// Compute the candidate features under the current partial mapping. `gain` is
// the same incremental log-likelihood the search uses for its g-value.
RankFeatures rank_features(const LabeledGraph& G, const LabeledGraph& Q, NodeId q,
                           NodeId u, const std::vector<NodeId>& mapping,
                           const compat::Params& w, bool use_node, bool use_edge);

// Linear ranker score (a logit). Ordering by this equals ordering by the
// predicted P(u is the true image), since the sigmoid is monotone. `w` has
// kRankFeats entries; an empty `w` means "inactive" (fall back to gain order).
double rank_score(const std::vector<double>& w, const RankFeatures& f);

// Train the ranker by logistic regression on ground-truth correspondences: walk
// the fail-first order with the true partial mapping and, at each step, label the
// true image as positive and the other feasible candidates as negative. Returns
// the kRankFeats weights (empty if no training signal). If `base` is non-null it
// is used as the clean source graph (real data) instead of fresh synthetic ones.
std::vector<double> train_ranker(const TrainConfig& cfg,
                                 const LabeledGraph* base = nullptr);

}  // namespace vf2prob::ml
