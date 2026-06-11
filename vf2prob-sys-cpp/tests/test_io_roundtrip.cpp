// VF2-Prob, Jan Hladěna, FIM UHK
// tests auto-generated
#include <filesystem>
#include <unordered_map>
#include <vector>

#include "test_framework.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/io/graphml.hpp"
#include "vf2prob/io/queries.hpp"

using namespace vf2prob;

TEST("io: GraphML write/read round-trips structure, labels and probs") {
  data::SynthParams sp;
  sp.n = 40;
  sp.p = 0.15;
  sp.nlabels = 4;
  sp.elabels = 3;
  sp.pmin = 0.6;
  sp.seed = 7;
  const LabeledGraph G = data::make_synthetic(sp);

  const auto tmp = std::filesystem::temp_directory_path() / "vf2prob_rt.graphml";
  io::write_graphml(tmp.string(), G);
  const auto loaded = io::read_graphml(tmp.string());
  const LabeledGraph& H = loaded.graph;

  CHECK(H.num_nodes() == G.num_nodes());
  CHECK(H.num_edges() == G.num_edges());
  for (NodeId u = 0; u < G.num_nodes(); ++u)
    CHECK(H.node_label(u) == G.node_label(u));

  // Spot-check edges: same adjacency, labels and probabilities.
  for (NodeId u = 0; u < G.num_nodes(); ++u) {
    for (const NodeId* it = G.neighbors_begin(u); it != G.neighbors_end(u);
         ++it) {
      const NodeId v = *it;
      if (v <= u) continue;
      CHECK(H.has_edge(u, v));
      CHECK(H.edge_label(u, v) == G.edge_label(u, v));
      CHECK_NEAR(H.edge_prob(u, v), G.edge_prob(u, v), 1e-5);
    }
  }
  std::filesystem::remove(tmp);
}

TEST("io: queries write/read round-trip") {
  const auto tmp = std::filesystem::temp_directory_path() / "vf2prob_q.json";
  const std::vector<std::vector<NodeId>> queries = {{0, 1, 2}, {3, 4, 5, 6}};
  io::write_queries(tmp.string(), "S", queries, 42);

  std::unordered_map<std::string, NodeId> id_to_index;
  for (NodeId i = 0; i < 10; ++i) id_to_index[std::to_string(i)] = i;
  const auto specs = io::read_queries(tmp.string(), id_to_index);

  CHECK(specs.size() == 2);
  CHECK(specs[0].nodes == std::vector<NodeId>({0, 1, 2}));
  CHECK(specs[1].nodes == std::vector<NodeId>({3, 4, 5, 6}));
  std::filesystem::remove(tmp);
}
