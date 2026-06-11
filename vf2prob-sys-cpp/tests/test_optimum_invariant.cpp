// VF2-Prob, Jan Hladěna, FIM UHK
// tests auto-generated
// All four VF2-Prob variants (DFS|A* x node|assign) return the same provable
// optimum when they terminate.
#include <vector>

#include "test_framework.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/registry.hpp"

using namespace vf2prob;

TEST("vf2prob: four variants agree on the optimum") {
  const std::vector<std::string> variants = {
      "vf2prob", "vf2prob-assign", "vf2prob-astar", "vf2prob-astar-assign"};

  for (uint64_t seed = 1; seed <= 25; ++seed) {
    data::SynthParams sp;
    sp.n = 18;
    sp.p = 0.35;
    sp.nlabels = 3;
    sp.elabels = 2;
    sp.pmin = 0.5;   // genuine uncertainty so the bound does real work
    sp.flip = 0.1;
    sp.seed = seed;
    const LabeledGraph G = data::make_synthetic(sp);
    const auto qs = data::sample_connected_queries(G, 5, 6, 1, seed + 50);
    if (qs.empty()) continue;
    const LabeledGraph Q = induced_subgraph(G, qs[0]);

    MatchOptions opt;
    opt.timeout_s = 10.0;
    opt.collect_metrics = false;

    double ref = 0.0;
    bool have_ref = false;
    for (const auto& v : variants) {
      auto m = make_matcher(v);
      const MatchResult R = m->solve(G, Q, opt);
      CHECK_MSG(!R.counters.timed_out, "variant timed out on a tiny instance");
      CHECK_MSG(R.mapping.has_value(), "variant found no embedding");
      if (!have_ref) {
        ref = R.counters.best_loglik;
        have_ref = true;
      } else {
        CHECK_NEAR(R.counters.best_loglik, ref, 1e-6);
      }
    }
  }
}
