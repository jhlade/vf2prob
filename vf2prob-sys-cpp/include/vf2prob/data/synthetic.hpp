// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <cstdint>
#include <vector>

#include "vf2prob/graph.hpp"

namespace vf2prob::data {

struct SynthParams {
  int n = 120;
  double p = 0.03;     // Erdős–Rényi edge probability
  int nlabels = 5;     // node label alphabet size
  int elabels = 3;     // edge label alphabet size
  double pmin = 0.8;   // edge-existence prob ~ Uniform[pmin, 1] (1.0 => certain)
  double flip = 0.0;   // fraction of node labels flipped (attribute noise)
  uint64_t seed = 324; // defaulkt seed
};

// Erdős–Rényi property graph with categorical node/edge labels, uncertain edge
// probabilities and optional label noise.
LabeledGraph make_synthetic(const SynthParams& p);

// Sample `num` connected induced subgraphs of size in [qmin, qmax] (BFS growth).
// Returns lists of data-graph node ids.
std::vector<std::vector<NodeId>> sample_connected_queries(const LabeledGraph& G,
                                                          int qmin, int qmax,
                                                          int num,
                                                          uint64_t seed);

// Rebuild G with injected uncertainty: edge probs resampled to Uniform[pmin,1]
// (when pmin<1) and node labels flipped with probability `flip`.
LabeledGraph with_noise(const LabeledGraph& G, double pmin, double flip,
                        uint64_t seed);

}  // namespace vf2prob::data
