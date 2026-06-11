// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/ordering.hpp"

#include <algorithm>

namespace vf2prob {

std::vector<NodeId> failfirst_order(const LabeledGraph& Q) {
  std::vector<NodeId> order(Q.num_nodes());
  for (NodeId i = 0; i < Q.num_nodes(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](NodeId a, NodeId b) {
    const int da = Q.degree(a), db = Q.degree(b);
    if (da != db) return da > db;  // decreasing degree
    const LabelId la = Q.node_label(a), lb = Q.node_label(b);
    if (la != lb) return la < lb;
    return a < b;
  });
  return order;
}

std::vector<std::pair<NodeId, NodeId>> mapped_neighbors(
    const LabeledGraph& Q, NodeId q, const std::vector<NodeId>& mapping) {
  std::vector<std::pair<NodeId, NodeId>> out;
  for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q); ++it) {
    const NodeId qn = *it;
    if (mapping[qn] >= 0) out.emplace_back(qn, mapping[qn]);
  }
  return out;
}

std::vector<NodeId> feasible_candidates(const LabeledGraph& G,
                                        const LabeledGraph& Q, NodeId q,
                                        const std::vector<NodeId>& mapping,
                                        const std::vector<char>& used) {
  const int deg_q = Q.degree(q);
  std::vector<NodeId> images;
  for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q); ++it) {
    if (mapping[*it] >= 0) images.push_back(mapping[*it]);
  }

  std::vector<NodeId> out;
  if (images.empty()) {
    for (NodeId u = 0; u < G.num_nodes(); ++u) {
      if (!used[u] && G.degree(u) >= deg_q) out.push_back(u);
    }
    return out;  // already ascending
  }

  // Base = image with the smallest degree, to minimize the scan.
  NodeId base = images[0];
  for (NodeId im : images)
    if (G.degree(im) < G.degree(base)) base = im;

  for (const NodeId* it = G.neighbors_begin(base); it != G.neighbors_end(base);
       ++it) {
    const NodeId u = *it;
    if (used[u] || G.degree(u) < deg_q) continue;
    bool ok = true;
    for (NodeId im : images) {
      if (im == base) continue;
      if (!G.has_edge(u, im)) {
        ok = false;
        break;
      }
    }
    if (ok) out.push_back(u);
  }
  std::sort(out.begin(), out.end());  // base neighbors are sorted, but be safe
  return out;
}

}  // namespace vf2prob
