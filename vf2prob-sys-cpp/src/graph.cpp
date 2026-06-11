// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/graph.hpp"

#include <algorithm>
#include <numeric>

namespace vf2prob {

LabeledGraph::LabeledGraph(NodeId num_nodes, std::vector<LabelId> node_labels,
                           const std::vector<Edge>& edges)
    : n_(num_nodes), node_label_(std::move(node_labels)) {
  if (static_cast<NodeId>(node_label_.size()) != n_)
    node_label_.assign(n_, kNoLabel);

  // Keep only valid, non-self-loop edges; count degrees.
  std::vector<Edge> es;
  es.reserve(edges.size());
  std::vector<int64_t> deg(n_ + 1, 0);
  for (const auto& [u, v, lab, prob] : edges) {
    if (u == v || u < 0 || v < 0 || u >= n_ || v >= n_) continue;
    es.emplace_back(u, v, lab, prob);
    ++deg[u];
    ++deg[v];
  }

  row_ptr_.assign(n_ + 1, 0);
  for (NodeId i = 0; i < n_; ++i) row_ptr_[i + 1] = row_ptr_[i] + deg[i];
  const int64_t total = row_ptr_[n_];
  col_idx_.assign(total, 0);
  edge_label_.assign(total, kNoLabel);
  edge_prob_.assign(total, 1.0);

  std::vector<int64_t> cur(row_ptr_.begin(), row_ptr_.begin() + n_);
  auto emit = [&](NodeId a, NodeId b, LabelId lab, double prob) {
    const int64_t p = cur[a]++;
    col_idx_[p] = b;
    edge_label_[p] = lab;
    edge_prob_[p] = prob;
  };
  for (const auto& [u, v, lab, prob] : es) {
    emit(u, v, lab, prob);
    emit(v, u, lab, prob);
  }

  // Sort each adjacency row by neighbor id, carrying labels and probabilities.
  std::vector<int64_t> order;
  for (NodeId u = 0; u < n_; ++u) {
    const int64_t b = row_ptr_[u], e = row_ptr_[u + 1];
    const int64_t len = e - b;
    if (len <= 1) continue;
    order.resize(len);
    std::iota(order.begin(), order.end(), b);
    std::sort(order.begin(), order.end(),
              [&](int64_t a, int64_t c) { return col_idx_[a] < col_idx_[c]; });
    std::vector<NodeId> tc(len);
    std::vector<LabelId> tl(len);
    std::vector<double> tp(len);
    for (int64_t k = 0; k < len; ++k) {
      tc[k] = col_idx_[order[k]];
      tl[k] = edge_label_[order[k]];
      tp[k] = edge_prob_[order[k]];
    }
    for (int64_t k = 0; k < len; ++k) {
      col_idx_[b + k] = tc[k];
      edge_label_[b + k] = tl[k];
      edge_prob_[b + k] = tp[k];
    }
  }

  m_ = static_cast<int64_t>(es.size());
}

int64_t LabeledGraph::find_pos(NodeId u, NodeId v) const {
  if (u < 0 || u >= n_) return -1;
  int64_t lo = row_ptr_[u], hi = row_ptr_[u + 1];
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo) / 2;
    if (col_idx_[mid] < v)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo < row_ptr_[u + 1] && col_idx_[lo] == v) return lo;
  return -1;
}

LabelId LabeledGraph::edge_label(NodeId u, NodeId v) const {
  const int64_t p = find_pos(u, v);
  return p < 0 ? kNoLabel : edge_label_[p];
}

double LabeledGraph::edge_prob(NodeId u, NodeId v) const {
  const int64_t p = find_pos(u, v);
  return p < 0 ? 1.0 : edge_prob_[p];
}

LabeledGraph induced_subgraph(const LabeledGraph& G,
                              const std::vector<NodeId>& nodes) {
  const NodeId k = static_cast<NodeId>(nodes.size());
  std::vector<NodeId> remap(G.num_nodes(), -1);
  std::vector<LabelId> labels(k, kNoLabel);
  for (NodeId i = 0; i < k; ++i) {
    remap[nodes[i]] = i;
    labels[i] = G.node_label(nodes[i]);
  }
  std::vector<LabeledGraph::Edge> edges;
  for (NodeId i = 0; i < k; ++i) {
    const NodeId u = nodes[i];
    for (const NodeId* it = G.neighbors_begin(u); it != G.neighbors_end(u);
         ++it) {
      const NodeId w = *it;
      const NodeId j = remap[w];
      if (j > i)  // emit each undirected edge once
        edges.emplace_back(i, j, G.edge_label(u, w), G.edge_prob(u, w));
    }
  }
  return LabeledGraph(k, std::move(labels), edges);
}

}  // namespace vf2prob
