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
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "app_common.hpp"
#include "vf2prob/compat.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/io/graphml.hpp"
#include "vf2prob/matcher.hpp"
#include "vf2prob/ml/train.hpp"
#include "vf2prob/registry.hpp"

using namespace vf2prob;

// Copy of G with node labels redrawn uniformly from `nlabels` classes (seeded).
// Used for the real-topology experiment: keep the SNAP topology and Jaccard
// edge probabilities, but control label cardinality so that exact recovery is
// measurable (the native degree-quantile labels are too coarse for that).
static LabeledGraph relabel_uniform(const LabeledGraph& G, int nlabels,
                                    uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<LabelId> labels(G.num_nodes());
  for (NodeId u = 0; u < G.num_nodes(); ++u)
    labels[u] = static_cast<LabelId>(rng() % nlabels);
  std::vector<LabeledGraph::Edge> edges;
  edges.reserve(static_cast<size_t>(G.num_edges()));
  for (NodeId u = 0; u < G.num_nodes(); ++u)
    for (const NodeId* it = G.neighbors_begin(u); it != G.neighbors_end(u); ++it)
      if (*it > u)
        edges.emplace_back(u, *it, G.edge_label(u, *it), G.edge_prob(u, *it));
  return LabeledGraph(G.num_nodes(), std::move(labels), edges);
}

// Recovery accuracy of `method` at noise `flip` over n seeded instances. If
// `base` is non-null it is the clean source graph (real topology); otherwise a
// fresh synthetic graph per instance. If `w` is non-null the matcher uses those
// (learned) compatibility weights.
static double recovery(const std::string& method, ml::TrainConfig cfg, double noise,
                       bool geometric, compat::NodeKernel kernel, double kernel_h,
                       int n, uint64_t base_seed, const LabeledGraph* base = nullptr,
                       const compat::Params* w = nullptr) {
  if (geometric) {
    cfg.coord_noise = noise;  // sweep coordinate-jitter stddev
    cfg.flip = 0.0;
  } else {
    cfg.flip = noise;
    cfg.coord_noise = 0.0;
  }
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
    sp.seed = base_seed + 1 + inst;
    const LabeledGraph Gclean = base ? *base : data::make_synthetic(sp);
    const auto qs = data::sample_connected_queries(Gclean, cfg.qsize, cfg.qsize,
                                                   1, base_seed + 7001 + inst);
    if (qs.empty()) continue;
    const std::vector<NodeId>& S = qs[0];
    const LabeledGraph Q = induced_subgraph(Gclean, S);
    const LabeledGraph Gobs = ml::make_observed(Gclean, cfg, base_seed + 5001 + inst);
    MatchOptions opt;
    opt.timeout_s = 10.0;
    opt.collect_metrics = false;
    if (w) opt.weights = *w;
    opt.weights.node_kernel = kernel;
    opt.weights.node_kernel_h = kernel_h;
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
  // Continuous-attribute (geometric) mode for the IAM/pattern-recognition
  // experiment: score nodes by an RBF kernel on coordinates and sweep the
  // coordinate-jitter stddev instead of the label-flip rate.
  const std::string node_kernel_s = appcli::get(a, "node-kernel", "categorical");
  const bool geometric = node_kernel_s == "rbf";
  const compat::NodeKernel kernel =
      geometric ? compat::NodeKernel::RBF : compat::NodeKernel::Categorical;
  const double kernel_h = appcli::getd(a, "kernel-h", 1.0);
  cfg.decoy_frac = appcli::getd(a, "decoy-frac", cfg.decoy_frac);
  cfg.true_p_min = appcli::getd(a, "true-pmin", cfg.true_p_min);
  // Optional override of the fixed weights: a file with the five weights
  // (vf2prob_run --weights format).
  std::optional<compat::Params> fixedw;
  if (const std::string fwp = appcli::get(a, "fixed-weights", ""); !fwp.empty()) {
    std::ifstream fw(fwp);
    std::string s, line;
    while (std::getline(fw, line)) s += line + "\n";
    if (!s.empty()) {
      fixedw = compat::parse_params(s);
      std::cout << "fixed weights overridden from " << fwp << "\n";
    } else {
      std::cerr << "warning: cannot read --fixed-weights " << fwp << "\n";
    }
  }
  // Real-topology mode: load a GraphML graph (e.g. SNAP) as the clean base and
  // optionally redraw its node labels with `--relabel N` classes (seeded).
  const std::string data_path = appcli::get(a, "data-path", "");
  const int relabel = appcli::geti(a, "relabel", 0);
  std::optional<LabeledGraph> base_holder;
  const LabeledGraph* base = nullptr;
  if (!data_path.empty()) {
    base_holder = io::read_graphml(data_path).graph;
    if (relabel > 1)
      base_holder = relabel_uniform(*base_holder, relabel, seed + 424242);
    base = &*base_holder;
    int nl = 1, el = 1;
    for (NodeId u = 0; u < base->num_nodes(); ++u)
      nl = std::max(nl, base->node_label(u) + 1);
    for (NodeId u = 0; u < base->num_nodes(); ++u)
      for (const NodeId* it = base->neighbors_begin(u);
           it != base->neighbors_end(u); ++it)
        if (*it > u) el = std::max(el, base->edge_label(u, *it) + 1);
    cfg.nlabels = nl;
    cfg.elabels = el;
    std::cout << "real topology: " << data_path << " (nodes=" << base->num_nodes()
              << " edges=" << base->num_edges() << " nlabels=" << nl
              << " elabels=" << el
              << (relabel > 1 ? " [relabelled uniformly]" : "") << ")\n";
  }
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
            << (geometric ? ", geometric/RBF" : "")
            << ", learned=" << learned << ")\n";
  for (size_t fi = 0; fi < flips.size(); ++fi) {
    const double flip = flips[fi];
    const uint64_t inst_seed = seed + 100000ull * (fi + 1);  // same instances across methods
    std::cout << "  flip=" << flip << ": ";
    for (const auto& mth : methods) {
      const double acc = recovery(mth, cfg, flip, geometric, kernel, kernel_h, n,
                                  inst_seed, base, fixedw ? &*fixedw : nullptr);
      f << flip << ',' << mth << ',' << acc << ',' << n << "\n";
      std::cout << mth << "=" << acc << "  ";
    }
    if (learned) {
      // Learn the compatibilities at THIS noise level (disjoint training seed),
      // then run the assignment-bound A* with them. With a real base graph the
      // weights are trained from correspondences on that same topology.
      ml::TrainConfig tcfg = cfg;
      tcfg.flip = flip;
      tcfg.seed = seed + 900000ull + fi;
      const compat::Params w = base ? ml::train_weights_from_graph(*base, tcfg)
                                    : ml::train_weights(tcfg);
      const double acc = recovery("vf2prob-astar-assign", cfg, flip, geometric,
                                  kernel, kernel_h, n, inst_seed, base, &w);
      f << flip << ",vf2prob-learned," << acc << ',' << n << "\n";
      std::cout << "vf2prob-learned=" << acc << "  ";
    }
    std::cout << "\n";
  }
  std::cout << "wrote " << out << "\n";
  return 0;
}
