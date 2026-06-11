// VF2-Prob, Jan Hladěna, FIM UHK
// Learn the compatibility weights from data and report held-out match accuracy
// with learned vs fixed weights. Writes the weights for `vf2prob_run --weights`.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <vector>

#include "app_common.hpp"
#include "vf2prob/algorithms/vf2prob.hpp"
#include "vf2prob/compat.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/io/graphml.hpp"
#include "vf2prob/ml/calibrate.hpp"
#include "vf2prob/ml/train.hpp"

using namespace vf2prob;

// Fraction of held-out noisy instances whose maximum-likelihood embedding
// exactly equals the ground truth, under the given weights. If `base` is set it
// is the clean source graph (real data); otherwise a fresh synthetic one.
static double match_accuracy(const compat::Params& w, const ml::TrainConfig& cfg,
                             int n_instances, uint64_t base_seed,
                             const LabeledGraph* base = nullptr) {
  int correct = 0, total = 0;
  for (int inst = 0; inst < n_instances; ++inst) {
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
    const LabeledGraph Gnoisy = ml::make_observed(Gclean, cfg, base_seed + 5001 + inst);

    VF2ProbMatcher matcher(BoundKind::Assign, SearchKind::DFS);
    MatchOptions opt;
    opt.timeout_s = 10.0;
    opt.collect_metrics = false;
    opt.weights = w;
    const MatchResult R = matcher.solve(Gnoisy, Q, opt);
    ++total;
    if (R.mapping && *R.mapping == S) ++correct;
  }
  return total ? static_cast<double>(correct) / total : 0.0;
}

int main(int argc, char** argv) {
  const auto a = appcli::parse(argc, argv);
  ml::TrainConfig cfg;
  cfg.instances = appcli::geti(a, "instances", cfg.instances);
  cfg.n = appcli::geti(a, "n", cfg.n);
  cfg.p = appcli::getd(a, "p", cfg.p);
  cfg.nlabels = appcli::geti(a, "nlabels", cfg.nlabels);
  cfg.elabels = appcli::geti(a, "elabels", cfg.elabels);
  cfg.pmin = appcli::getd(a, "pmin", cfg.pmin);
  cfg.flip = appcli::getd(a, "flip", cfg.flip);
  cfg.qsize = appcli::geti(a, "qsize", cfg.qsize);
  cfg.epochs = appcli::geti(a, "epochs", cfg.epochs);
  cfg.lr = appcli::getd(a, "lr", cfg.lr);
  cfg.l2 = appcli::getd(a, "l2", cfg.l2);
  cfg.seed = static_cast<uint64_t>(appcli::geti(a, "seed", 1));
  cfg.per_label_node = appcli::get(a, "per-label", "0") == "1";
  cfg.per_label_edge = appcli::get(a, "per-label-edge", "0") == "1";
  cfg.structured_noise = appcli::get(a, "structured", "0") == "1";
  const std::string out = appcli::get(a, "out", "results/weights.txt");
  const int eval_n = appcli::geti(a, "eval", 30);

  // Optional: learn from a real graph (correspondences = induced subgraphs of
  // a loaded GraphML, e.g. SNAP) instead of synthetic ER graphs.
  const std::string data_path = appcli::get(a, "data-path", "");
  std::optional<LabeledGraph> base_holder;
  const LabeledGraph* base = nullptr;
  if (!data_path.empty()) {
    base_holder = io::read_graphml(data_path).graph;
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
    std::cout << "training from real graph: " << data_path
              << " (nodes=" << base->num_nodes() << " edges=" << base->num_edges()
              << " nlabels=" << nl << " elabels=" << el << ")\n";
  }

  std::cout << "training: instances=" << cfg.instances << " n=" << cfg.n
            << " flip=" << cfg.flip << " pmin=" << cfg.pmin
            << " qsize=" << cfg.qsize << "\n";

  const compat::Params learned =
      base ? ml::train_weights_from_graph(*base, cfg) : ml::train_weights(cfg);
  const compat::Params fixed;  // default calibrated

  std::cout << "\nlearned weights:\n";
  std::cout << "  node: w0=" << learned.node_w0
            << " w_label=" << learned.node_w_label << "\n";
  std::cout << "  edge: e0=" << learned.edge_w0
            << " e_label=" << learned.edge_w_label
            << " e_prob=" << learned.edge_w_prob << "\n";
  if (cfg.per_label_node)
    std::cout << "  per-label node table: " << learned.n_node_labels << "x"
              << learned.n_node_labels << "\n";
  if (cfg.per_label_edge)
    std::cout << "  per-label edge table: " << learned.n_edge_labels << "x"
              << learned.n_edge_labels << "\n";

  const bool per_label = cfg.per_label_node || cfg.per_label_edge;
  if (eval_n > 0) {
    const uint64_t hold = cfg.seed + 100000;
    std::cout << "\nheld-out match accuracy (" << eval_n << " instances, flip="
              << cfg.flip << (cfg.structured_noise ? ", structured" : "")
              << (base ? ", real graph" : "") << "):\n";
    std::cout << "  fixed         = "
              << match_accuracy(fixed, cfg, eval_n, hold, base) << "\n";
    if (per_label) {
      ml::TrainConfig scfg = cfg;
      scfg.per_label_node = false;
      scfg.per_label_edge = false;
      const compat::Params scalar = base ? ml::train_weights_from_graph(*base, scfg)
                                          : ml::train_weights(scfg);
      std::cout << "  learned scalar= "
                << match_accuracy(scalar, cfg, eval_n, hold, base) << "\n";
      std::cout << "  learned table = "
                << match_accuracy(learned, cfg, eval_n, hold, base) << "\n";
    } else {
      std::cout << "  learned       = "
                << match_accuracy(learned, cfg, eval_n, hold, base) << "\n";
    }
  }

  if (appcli::get(a, "calibrate", "0") == "1") {
    const uint64_t cseed = cfg.seed + 200000;  // disjoint from train/eval seeds
    // Calibrate a scorer: fit Platt + isotonic on one held-out split, report
    // raw / +Platt / +isotonic ECE on a disjoint test split.
    auto report = [&](const char* name, const compat::Params& w) {
      std::vector<double> cp, tp;
      std::vector<int> cy, ty;
      ml::node_correspondence_samples(cfg, w, cseed, 30, cp, cy, base);
      ml::node_correspondence_samples(cfg, w, cseed + 50000, 30, tp, ty, base);
      const ml::Platt pl = ml::fit_platt(cp, cy);
      const ml::Isotonic iso = ml::fit_isotonic(cp, cy);
      std::vector<double> tpp(tp.size()), tpi(tp.size());
      for (size_t i = 0; i < tp.size(); ++i) {
        tpp[i] = ml::apply_platt(pl, tp[i]);
        tpi[i] = ml::apply_isotonic(iso, tp[i]);
      }
      std::cout << "  " << name
                << ": raw=" << ml::expected_calibration_error(tp, ty)
                << "  +Platt=" << ml::expected_calibration_error(tpp, ty)
                << "  +isotonic=" << ml::expected_calibration_error(tpi, ty)
                << "\n";
    };
    std::cout << "\ncalibration of node compatibilities (ECE, lower is better):\n";
    report("fixed  ", fixed);
    report("learned", learned);  // per-label when --per-label is set

    // Reliability-diagram data for the fixed scorer: raw vs Platt-calibrated.
    std::vector<double> cp, tp;
    std::vector<int> cy, ty;
    ml::node_correspondence_samples(cfg, fixed, cseed, 30, cp, cy, base);
    ml::node_correspondence_samples(cfg, fixed, cseed + 50000, 30, tp, ty, base);
    const ml::Platt pl = ml::fit_platt(cp, cy);
    std::vector<double> tpp(tp.size());
    for (size_t i = 0; i < tp.size(); ++i) tpp[i] = ml::apply_platt(pl, tp[i]);
    std::filesystem::create_directories("results");
    std::ofstream rf("results/reliability.csv");
    rf << "which,conf,acc,count\n";
    for (const auto& b : ml::reliability_bins(tp, ty))
      rf << "raw," << b.conf << "," << b.acc << "," << b.count << "\n";
    for (const auto& b : ml::reliability_bins(tpp, ty))
      rf << "platt," << b.conf << "," << b.acc << "," << b.count << "\n";
    std::cout << "wrote results/reliability.csv\n";
  }

  const std::filesystem::path outp(out);
  if (outp.has_parent_path())
    std::filesystem::create_directories(outp.parent_path());
  std::ofstream f(out);
  f << compat::to_string(learned) << "\n";
  std::cout << "\nwrote " << out
            << "  (use: vf2prob_run --weights " << out << ")\n";
  return 0;
}
