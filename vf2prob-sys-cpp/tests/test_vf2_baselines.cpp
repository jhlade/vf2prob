// VF2-Prob, Jan Hladěna, FIM UHK
// tests auto-generated
#include <vector>

#include "test_framework.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/registry.hpp"

using namespace vf2prob;

// A returned mapping must be injective, label-consistent and edge-preserving.
static bool valid_embedding(const LabeledGraph& G, const LabeledGraph& Q,
                            const std::vector<NodeId>& mp) {
  std::vector<char> seen(G.num_nodes(), 0);
  for (NodeId q = 0; q < Q.num_nodes(); ++q) {
    const NodeId u = mp[q];
    if (u < 0 || u >= G.num_nodes()) return false;
    if (seen[u]) return false;
    seen[u] = 1;
    if (Q.node_label(q) != G.node_label(u)) return false;
  }
  for (NodeId q = 0; q < Q.num_nodes(); ++q)
    for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q); ++it)
      if (*it > q && !G.has_edge(mp[q], mp[*it])) return false;
  return true;
}

TEST("baselines: VF2 and VF2++ find a valid embedding when one exists") {
  for (uint64_t seed = 1; seed <= 20; ++seed) {
    data::SynthParams sp;
    sp.n = 30;
    sp.p = 0.2;
    sp.nlabels = 4;
    sp.pmin = 1.0;  // deterministic graph
    sp.seed = seed;
    const LabeledGraph G = data::make_synthetic(sp);
    const auto qs = data::sample_connected_queries(G, 4, 6, 1, seed + 200);
    if (qs.empty()) continue;
    const LabeledGraph Q = induced_subgraph(G, qs[0]);  // embeddable by design

    MatchOptions opt;
    opt.timeout_s = 10.0;

    for (const std::string& name : {"vf2", "vf2pp"}) {
      auto m = make_matcher(name);
      const MatchResult R = m->solve(G, Q, opt);
      CHECK_MSG(R.mapping.has_value(), name + ": no embedding found");
      if (R.mapping) CHECK_MSG(valid_embedding(G, Q, *R.mapping),
                               name + ": invalid embedding");
      CHECK(R.counters.solutions >= 1);
    }
  }
}

TEST("baselines: VF2-Bin runs and respects the threshold on clean graphs") {
  data::SynthParams sp;
  sp.n = 25;
  sp.p = 0.25;
  sp.nlabels = 3;
  sp.pmin = 1.0;  // all edge probs = 1 -> binarization keeps every edge
  sp.seed = 3;
  const LabeledGraph G = data::make_synthetic(sp);
  const auto qs = data::sample_connected_queries(G, 4, 5, 1, 999);
  CHECK(!qs.empty());
  const LabeledGraph Q = induced_subgraph(G, qs[0]);

  MatchOptions opt;
  opt.timeout_s = 10.0;
  opt.bin_tau = 0.7;
  auto m = make_matcher("vf2bin");
  const MatchResult R = m->solve(G, Q, opt);
  CHECK_MSG(R.mapping.has_value(), "vf2bin: no embedding on a clean graph");
  if (R.mapping) CHECK(valid_embedding(G, Q, *R.mapping));
}
