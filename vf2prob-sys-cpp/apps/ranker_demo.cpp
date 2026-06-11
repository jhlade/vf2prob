// VF2-Prob, Jan Hladěna, FIM UHK
// Learned expansion-order ranker demo. Trains the ranker (Section: ML extension)
// and compares DFS branch-and-bound search effort (explored states) under the
// default gain ordering vs. the learned ordering on held-out instances, while
// verifying that BOTH orderings return the same maximum-likelihood optimum --
// i.e. the learned ordering speeds up the search WITHOUT breaking exactness.
#include <cmath>
#include <iostream>
#include <vector>

#include "app_common.hpp"
#include "vf2prob/algorithms/vf2prob.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/ml/ranker.hpp"
#include "vf2prob/ml/train.hpp"

using namespace vf2prob;

int main(int argc, char** argv) {
  const auto a = appcli::parse(argc, argv);
  ml::TrainConfig cfg;
  cfg.instances = appcli::geti(a, "instances", cfg.instances);
  cfg.n = appcli::geti(a, "n", cfg.n);
  cfg.p = appcli::getd(a, "p", cfg.p);
  cfg.nlabels = appcli::geti(a, "nlabels", cfg.nlabels);
  cfg.elabels = appcli::geti(a, "elabels", cfg.elabels);
  cfg.flip = appcli::getd(a, "flip", cfg.flip);
  cfg.qsize = appcli::geti(a, "qsize", cfg.qsize);
  cfg.epochs = appcli::geti(a, "epochs", cfg.epochs);
  cfg.lr = appcli::getd(a, "lr", cfg.lr);
  cfg.l2 = appcli::getd(a, "l2", cfg.l2);
  cfg.seed = static_cast<uint64_t>(appcli::geti(a, "seed", 1));
  cfg.structured_noise = appcli::get(a, "structured", "0") == "1";
  const int eval_n = appcli::geti(a, "eval", 40);
  const BoundKind bound =
      appcli::get(a, "bound", "assign") == "node" ? BoundKind::Node : BoundKind::Assign;

  std::cout << "training ranker: instances=" << cfg.instances << " n=" << cfg.n
            << " qsize=" << cfg.qsize << " flip=" << cfg.flip
            << " nlabels=" << cfg.nlabels << "\n";
  const std::vector<double> w = ml::train_ranker(cfg);
  if (w.empty()) {
    std::cerr << "no training signal\n";
    return 1;
  }
  std::cout << "ranker weights [bias,gain,pressure,log-deg,nc]:";
  for (double v : w) std::cout << " " << v;
  std::cout << "\n";

  VF2ProbMatcher matcher(bound, SearchKind::DFS);
  std::vector<double> sg, sl, ratio;  // states: gain order / learned order
  int mismatches = 0, solved = 0;
  const uint64_t hold = cfg.seed + 200000;  // disjoint from training seeds
  for (int inst = 0; inst < eval_n; ++inst) {
    data::SynthParams sp;
    sp.n = cfg.n;
    sp.p = cfg.p;
    sp.nlabels = cfg.nlabels;
    sp.elabels = cfg.elabels;
    sp.pmin = 1.0;
    sp.flip = 0.0;
    sp.seed = hold + 1 + inst;
    const LabeledGraph Gclean = data::make_synthetic(sp);
    const auto qs = data::sample_connected_queries(Gclean, cfg.qsize, cfg.qsize,
                                                   1, hold + 7001 + inst);
    if (qs.empty()) continue;
    const LabeledGraph Q = induced_subgraph(Gclean, qs[0]);
    const LabeledGraph Gobs = ml::make_observed(Gclean, cfg, hold + 5001 + inst);

    MatchOptions o;
    o.collect_metrics = false;
    o.timeout_s = 30.0;
    o.rank_w.clear();  // default gain ordering
    const MatchResult R0 = matcher.solve(Gobs, Q, o);
    o.rank_w = w;  // learned ordering
    const MatchResult R1 = matcher.solve(Gobs, Q, o);
    if (R0.counters.timed_out || R1.counters.timed_out) continue;
    ++solved;
    sg.push_back(static_cast<double>(R0.counters.states));
    sl.push_back(static_cast<double>(R1.counters.states));
    if (R1.counters.states > 0)
      ratio.push_back(static_cast<double>(R0.counters.states) /
                      static_cast<double>(R1.counters.states));
    if (std::fabs(R0.counters.best_loglik - R1.counters.best_loglik) > 1e-6)
      ++mismatches;
  }

  std::cout << "\nheld-out instances solved by both: " << solved << "\n";
  std::cout << "median states, gain ordering    = " << appcli::median(sg) << "\n";
  std::cout << "median states, learned ordering = " << appcli::median(sl) << "\n";
  std::cout << "REI (median per-instance gain/learned) = " << appcli::median(ratio)
            << "\n";
  std::cout << "optimum mismatches (must be 0)  = " << mismatches << " / " << solved
            << "\n";
  return 0;
}
