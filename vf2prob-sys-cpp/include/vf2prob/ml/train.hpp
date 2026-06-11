// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <cstdint>

#include "vf2prob/compat.hpp"
#include "vf2prob/graph.hpp"

namespace vf2prob::ml {

// Configuration for learning the compatibility weights. Training instances are
// synthetic with a known ground-truth embedding (query = induced subgraph of a
// clean graph; data graph = same graph with flipped labels and uncertain edge
// probabilities). Node and edge logistic models are fit by gradient descent;
// the result goes into MatchOptions.weights.
struct TrainConfig {
  int instances = 60;
  int n = 60;
  double p = 0.06;
  int nlabels = 5;
  int elabels = 3;
  double pmin = 0.6;
  double flip = 0.15;     // label-noise regime we learn to be robust to
  int qsize = 8;
  int neg_per_pos = 4;
  int epochs = 400;
  double lr = 0.3;
  double l2 = 1e-3;
  uint64_t seed = 1;
  // Edge uncertainty: true edges keep a high existence probability, while
  // spurious "decoy" edges are added with a low one — so the probability is
  // informative for the edge weight e_prob.
  double decoy_frac = 0.5;    // # decoy edges as a fraction of true edges
  double true_p_min = 0.7;    // true edge prob ~ Uniform[true_p_min, 1]
  double decoy_p_max = 0.4;   // decoy edge prob ~ Uniform[pmin, decoy_p_max]
  bool per_label_node = false;   // learn a per-label-pair node compatibility table
  bool per_label_edge = false;   // learn a per-label-pair edge bias table
  bool structured_noise = false; // flip labels to (label+1) mod n, not uniformly
};

// Observed data graph from a clean graph: true edges with high existence
// probability and flip-noised labels, plus spurious low-probability decoy edges.
LabeledGraph make_observed(const LabeledGraph& clean, const TrainConfig& cfg,
                           uint64_t seed);

// Learn weights from synthetic instances (a fresh ER graph per instance).
compat::Params train_weights(const TrainConfig& cfg);

// Learn weights from REAL correspondences: every instance uses `clean_base`
// (e.g. a loaded SNAP graph) as the source, sampling a different query each
// time; nlabels/elabels are inferred from the graph. Same observation-noise
// model (make_observed) and fitting as the synthetic path.
compat::Params train_weights_from_graph(const LabeledGraph& clean_base,
                                        const TrainConfig& cfg);

}  // namespace vf2prob::ml
