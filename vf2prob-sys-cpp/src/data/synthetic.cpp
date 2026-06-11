// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/data/synthetic.hpp"

#include <algorithm>
#include <random>
#include <unordered_set>

namespace vf2prob::data {

LabeledGraph make_synthetic(const SynthParams& pr) {
  std::mt19937_64 rng(pr.seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::uniform_int_distribution<int> nlab(0, std::max(0, pr.nlabels - 1));
  std::uniform_int_distribution<int> elab(0, std::max(0, pr.elabels - 1));

  std::vector<LabelId> node_labels(pr.n);
  for (int i = 0; i < pr.n; ++i) node_labels[i] = nlab(rng);

  // Label noise: flip a fraction of node labels to a different label.
  if (pr.flip > 0.0 && pr.nlabels > 1) {
    for (int i = 0; i < pr.n; ++i) {
      if (uni(rng) < pr.flip) {
        const int add = 1 + (rng() % (pr.nlabels - 1));
        node_labels[i] = (node_labels[i] + add) % pr.nlabels;
      }
    }
  }

  std::vector<LabeledGraph::Edge> edges;
  for (int i = 0; i < pr.n; ++i) {
    for (int j = i + 1; j < pr.n; ++j) {
      if (uni(rng) < pr.p) {
        const LabelId el = pr.elabels > 0 ? elab(rng) : kNoLabel;
        const double prob =
            pr.pmin < 1.0 ? pr.pmin + (1.0 - pr.pmin) * uni(rng) : 1.0;
        edges.emplace_back(i, j, el, prob);
      }
    }
  }
  return LabeledGraph(pr.n, std::move(node_labels), edges);
}

std::vector<std::vector<NodeId>> sample_connected_queries(const LabeledGraph& G,
                                                          int qmin, int qmax,
                                                          int num,
                                                          uint64_t seed) {
  std::mt19937_64 rng(seed);
  if (qmin > qmax) std::swap(qmin, qmax);
  const NodeId n = G.num_nodes();
  std::vector<std::vector<NodeId>> out;
  if (n == 0) return out;
  std::uniform_int_distribution<int> ksize(qmin, qmax);
  std::uniform_int_distribution<NodeId> pick(0, n - 1);

  for (int qi = 0; qi < num; ++qi) {
    const int k = ksize(rng);
    std::vector<NodeId> chosen;
    for (int attempt = 0; attempt < 64 && chosen.empty(); ++attempt) {
      const NodeId start = pick(rng);
      std::unordered_set<NodeId> seen{start};
      std::vector<NodeId> frontier{start};
      size_t head = 0;
      while (static_cast<int>(seen.size()) < k && head < frontier.size()) {
        const NodeId u = frontier[head++];
        for (const NodeId* it = G.neighbors_begin(u); it != G.neighbors_end(u);
             ++it) {
          if (seen.insert(*it).second) frontier.push_back(*it);
          if (static_cast<int>(seen.size()) >= k) break;
        }
      }
      if (static_cast<int>(seen.size()) >= k) {
        chosen.assign(seen.begin(), seen.end());
        std::sort(chosen.begin(), chosen.end());
      }
    }
    if (!chosen.empty()) out.push_back(std::move(chosen));
  }
  return out;
}

LabeledGraph with_noise(const LabeledGraph& G, double pmin, double flip,
                        uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  int nlabels = 1;
  for (NodeId u = 0; u < G.num_nodes(); ++u)
    nlabels = std::max(nlabels, G.node_label(u) + 1);

  std::vector<LabelId> labels = G.node_labels();
  if (flip > 0.0 && nlabels > 1) {
    for (NodeId u = 0; u < G.num_nodes(); ++u) {
      if (labels[u] != kNoLabel && uni(rng) < flip) {
        const int add = 1 + (rng() % (nlabels - 1));
        labels[u] = (labels[u] + add) % nlabels;
      }
    }
  }

  std::vector<LabeledGraph::Edge> edges;
  for (NodeId u = 0; u < G.num_nodes(); ++u) {
    for (const NodeId* it = G.neighbors_begin(u); it != G.neighbors_end(u);
         ++it) {
      const NodeId v = *it;
      if (v <= u) continue;
      double prob = G.edge_prob(u, v);
      if (pmin < 1.0) prob = pmin + (1.0 - pmin) * uni(rng);
      edges.emplace_back(u, v, G.edge_label(u, v), prob);
    }
  }
  return LabeledGraph(G.num_nodes(), std::move(labels), edges);
}

}  // namespace vf2prob::data
