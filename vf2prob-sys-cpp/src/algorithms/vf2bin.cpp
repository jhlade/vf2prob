// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/algorithms/vf2bin.hpp"

#include "vf2prob/algorithms/vf2.hpp"

namespace vf2prob {

LabeledGraph binarize(const LabeledGraph& G, double tau) {
  std::vector<LabeledGraph::Edge> edges;
  for (NodeId u = 0; u < G.num_nodes(); ++u) {
    for (const NodeId* it = G.neighbors_begin(u); it != G.neighbors_end(u); ++it) {
      const NodeId v = *it;
      if (v <= u) continue;  // each undirected edge once
      if (G.edge_prob(u, v) >= tau)
        edges.emplace_back(u, v, G.edge_label(u, v), G.edge_prob(u, v));
    }
  }
  return LabeledGraph(G.num_nodes(), G.node_labels(), edges);
}

MatchResult VF2BinMatcher::solve(const LabeledGraph& G, const LabeledGraph& Q,
                                 const MatchOptions& opt) {
  const LabeledGraph H = binarize(G, opt.bin_tau);
  return vf2_first_embedding(H, Q, opt);
}

}  // namespace vf2prob
