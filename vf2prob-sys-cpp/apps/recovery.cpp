// VF2-Prob, Jan Hladěna, FIM UHK
// Sensitivity analysis: ground-truth recovery accuracy vs attribute-noise rate,
// comparing the uncertain-graph competitor (mpm), VF2-Bin, and VF2-Prob (with
// fixed expert weights and with weights LEARNED at each noise level) on
// identical, seeded instances. Shows (i) the soft probabilistic objective
// degrades far more gracefully than the hard-label methods, and (ii) the ML
// extension's advantage grows with the noise rate.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "app_common.hpp"
#include "vf2prob/compat.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/matcher.hpp"
#include "vf2prob/ml/train.hpp"
#include "vf2prob/registry.hpp"

using namespace vf2prob;

// Recovery accuracy of `method` at noise `flip` over n seeded instances. If `w`
// is non-null the matcher uses those (learned) compatibility weights.
static double recovery(const std::string& method, ml::TrainConfig cfg, double flip,
                       int n, uint64_t base, const compat::Params* w = nullptr) {
  cfg.flip = flip;
  auto m = make_matcher(method);
  if (!m) return -1.0;
  int correct = 0, total = 0;
  for (int inst = 0; inst < n; ++inst) {
    data::SynthParams sp;
    sp.n = cfg.n;
    sp.p = cfg.p;
    sp.nlabels = cfg.nlabels;
    sp.elabels = cfg.elabels;
    sp.pmin = 1.0;
    sp.flip = 0.0;
    sp.seed = base + 1 + inst;
    const LabeledGraph Gclean = data::make_synthetic(sp);
    const auto qs = data::sample_connected_queries(Gclean, cfg.qsize, cfg.qsize,
                                                   1, base + 7001 + inst);
    if (qs.empty()) continue;
    const std::vector<NodeId>& S = qs[0];
    const LabeledGraph Q = induced_subgraph(Gclean, S);
    const LabeledGraph Gobs = ml::make_observed(Gclean, cfg, base + 5001 + inst);
    MatchOptions opt;
    opt.timeout_s = 10.0;
    opt.collect_metrics = false;
    if (w) opt.weights = *w;
    const MatchResult R = m->solve(Gobs, Q, opt);
    ++total;
    if (R.mapping && *R.mapping == S) ++correct;
  }
  return total ? static_cast<double>(correct) / total : 0.0;
}

int main(int argc, char** argv) {
  const auto a = appcli::parse(argc, argv);
  ml::TrainConfig cfg;
  cfg.n = appcli::geti(a, "n", 60);
  cfg.p = appcli::getd(a, "p", 0.06);
  cfg.qsize = appcli::geti(a, "qsize", 8);
  cfg.nlabels = appcli::geti(a, "nlabels", 5);
  cfg.elabels = appcli::geti(a, "elabels", 3);
  cfg.structured_noise = appcli::get(a, "structured", "0") == "1";
  const int n = appcli::geti(a, "instances", 50);
  const uint64_t seed = static_cast<uint64_t>(appcli::geti(a, "seed", 1));
  const std::string out = appcli::get(a, "out", "results/recovery.csv");
  const std::vector<std::string> methods = appcli::split_csv(
      appcli::get(a, "methods", "mpm,vf2bin,vf2prob-astar-assign"));
  const bool learned = appcli::get(a, "learned", "1") == "1";
  // Fine attribute-noise grid (fraction of node labels flipped).
  std::vector<double> flips;
  for (const auto& s : appcli::split_csv(
           appcli::get(a, "flips", "0,0.01,0.03,0.05,0.1,0.15,0.2,0.3,0.4,0.5")))
    flips.push_back(std::stod(s));

  const std::filesystem::path outp(out);
  if (outp.has_parent_path())
    std::filesystem::create_directories(outp.parent_path());
  std::ofstream f(out);
  f << "flip,method,accuracy,n\n";
  std::cout << "recovery sensitivity (n=" << n << " per cell, qsize=" << cfg.qsize
            << ", nlabels=" << cfg.nlabels
            << (cfg.structured_noise ? ", structured" : "")
            << ", learned=" << learned << ")\n";
  for (size_t fi = 0; fi < flips.size(); ++fi) {
    const double flip = flips[fi];
    const uint64_t base = seed + 100000ull * (fi + 1);  // same instances across methods
    std::cout << "  flip=" << flip << ": ";
    for (const auto& mth : methods) {
      const double acc = recovery(mth, cfg, flip, n, base);
      f << flip << ',' << mth << ',' << acc << ',' << n << "\n";
      std::cout << mth << "=" << acc << "  ";
    }
    if (learned) {
      // Learn the compatibilities at THIS noise level (disjoint training seed),
      // then run the assignment-bound A* with them.
      ml::TrainConfig tcfg = cfg;
      tcfg.flip = flip;
      tcfg.seed = seed + 900000ull + fi;
      const compat::Params w = ml::train_weights(tcfg);
      const double acc =
          recovery("vf2prob-astar-assign", cfg, flip, n, base, &w);
      f << flip << ",vf2prob-learned," << acc << ',' << n << "\n";
      std::cout << "vf2prob-learned=" << acc << "  ";
    }
    std::cout << "\n";
  }
  std::cout << "wrote " << out << "\n";
  return 0;
}
