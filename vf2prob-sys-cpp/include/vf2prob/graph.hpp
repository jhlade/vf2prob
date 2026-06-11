// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <cstdint>
#include <tuple>
#include <vector>

#include "vf2prob/types.hpp"

namespace vf2prob {

// Undirected labeled property graph with optional categorical edge labels and
// edge-existence probabilities. Stored in CSR form with neighbor lists sorted
// ascending, so degree / neighbor iteration are O(1)/O(deg) and has_edge /
// edge lookups are O(log deg). Node ids are compact (0..n-1); labels live in a
// shared vocabulary (see io/ and induced_subgraph) so that equal LabelId means
// equal label across a query/data pair.
class LabeledGraph {
 public:
  // One undirected edge: (u, v, edge_label, existence_probability).
  using Edge = std::tuple<NodeId, NodeId, LabelId, double>;

  LabeledGraph() = default;
  LabeledGraph(NodeId num_nodes, std::vector<LabelId> node_labels,
               const std::vector<Edge>& edges);

  NodeId num_nodes() const { return n_; }
  int64_t num_edges() const { return m_; }

  LabelId node_label(NodeId u) const { return node_label_[u]; }
  const std::vector<LabelId>& node_labels() const { return node_label_; }

  int degree(NodeId u) const {
    return static_cast<int>(row_ptr_[u + 1] - row_ptr_[u]);
  }
  // Neighbor range [begin, end) into the CSR column array (sorted ascending).
  const NodeId* neighbors_begin(NodeId u) const {
    return col_idx_.data() + row_ptr_[u];
  }
  const NodeId* neighbors_end(NodeId u) const {
    return col_idx_.data() + row_ptr_[u + 1];
  }

  bool has_edge(NodeId u, NodeId v) const { return find_pos(u, v) >= 0; }
  // kNoLabel if the edge is absent or unlabeled.
  LabelId edge_label(NodeId u, NodeId v) const;
  // 1.0 if the edge is absent or carries no probability.
  double edge_prob(NodeId u, NodeId v) const;

 private:
  int64_t find_pos(NodeId u, NodeId v) const;  // index into col_idx_, or -1

  NodeId n_ = 0;
  int64_t m_ = 0;
  std::vector<LabelId> node_label_;  // size n
  std::vector<int64_t> row_ptr_;     // size n+1
  std::vector<NodeId> col_idx_;      // size 2m, sorted per row
  std::vector<LabelId> edge_label_;  // parallel to col_idx_
  std::vector<double> edge_prob_;    // parallel to col_idx_
};

// Induced subgraph on `nodes` (kept in the given order, relabeled to 0..k-1).
// Node and edge labels and edge probabilities are copied from G, so the result
// shares G's label vocabulary — exactly what the matchers assume for a (Q, G)
// pair where Q is sampled from G.
LabeledGraph induced_subgraph(const LabeledGraph& G,
                              const std::vector<NodeId>& nodes);

}  // namespace vf2prob
