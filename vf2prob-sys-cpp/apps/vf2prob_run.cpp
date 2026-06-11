// VF2-Prob, Jan Hladěna, FIM UHK
// Unified experiment runner: dispatches every method through the IMatcher
// interface and writes a per-(query, method, run) CSV with wall-clock (ns) and
// peak RSS measured by the harness.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "app_common.hpp"
#include "vf2prob/data/synthetic.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/io/csv.hpp"
#include "vf2prob/io/graphml.hpp"
#include "vf2prob/io/queries.hpp"
#include "vf2prob/bounds.hpp"
#include "vf2prob/registry.hpp"

using namespace vf2prob;
using Clock = std::chrono::steady_clock;

// Export a labeled graph in Glasgow's vertex-labelled LAD format:
//   line 1: <n>;  then per vertex i: <label> <degree> <neighbor ids...>.
// Used to feed identical instances to the Glasgow Subgraph Solver (a modern
// exact structural baseline) for a fair external comparison.
static void write_lad(const std::string& path, const LabeledGraph& G) {
  std::ofstream f(path);
  f << G.num_nodes() << "\n";
  for (NodeId u = 0; u < G.num_nodes(); ++u) {
    const NodeId* b = G.neighbors_begin(u);
    const NodeId* e = G.neighbors_end(u);
    f << G.node_label(u) << " " << (e - b);
    for (const NodeId* it = b; it != e; ++it) f << " " << *it;
    f << "\n";
  }
}

int main(int argc, char** argv) {
  const auto a = appcli::parse(argc, argv);
  const std::string dataset = appcli::get(a, "dataset", "synth");
  const double timeout = appcli::getd(a, "timeout", 30.0);
  // Deterministic state budget for reproducible completion (0 = unlimited).
  const uint64_t max_states =
      static_cast<uint64_t>(appcli::geti(a, "max-states", 0));
  const int runs = appcli::geti(a, "runs", 3);
  const uint64_t seed = static_cast<uint64_t>(appcli::geti(a, "seed", 324));
  const double bin_tau = appcli::getd(a, "bin-threshold", 0.7);
  const int qmin = appcli::geti(a, "qmin", 6);
  const int qmax = appcli::geti(a, "qmax", 10);
  const int nq = appcli::geti(a, "nqueries", 8);
  const std::string out = appcli::get(a, "out", "results/out.csv");
  // Optional: export the target graph + query patterns in vertex-labelled LAD
  // (for the Glasgow Subgraph Solver) instead of running the benchmark.
  const std::string export_lad = appcli::get(a, "export-lad", "");
  const bool collect = appcli::get(a, "no-metrics", "0") != "1";
  // Optional learned compatibility weights.
  compat::Params weights;
  const std::string wpath = appcli::get(a, "weights", "");
  if (!wpath.empty()) {
    std::ifstream wf(wpath);
    if (wf) {
      std::stringstream ss;
      ss << wf.rdbuf();
      weights = compat::parse_params(ss.str());
      std::cout << "using learned weights from " << wpath << "\n";
    } else {
      std::cerr << "warning: cannot read --weights " << wpath << "\n";
    }
  }
  const std::vector<std::string> methods = appcli::split_csv(appcli::get(
      a, "methods",
      "vf2,vf2pp,vf2bin,mpm,vf2prob,vf2prob-assign,vf2prob-astar,"
      "vf2prob-astar-assign"));

  LabeledGraph G;
  std::vector<io::QuerySpec> queries;
  std::string dsname = dataset;

  try {
    if (dataset == "synth") {
      data::SynthParams sp;
      sp.seed = seed;
      sp.pmin = appcli::getd(a, "pmin", 0.8);
      sp.flip = appcli::getd(a, "flip", 0.0);
      G = data::make_synthetic(sp);
      const auto qs = data::sample_connected_queries(G, qmin, qmax, nq, seed + 1);
      for (size_t i = 0; i < qs.size(); ++i)
        queries.push_back({static_cast<int>(i), qs[i]});
      dsname = "synth";
    } else {  // graphml
      const std::string dp = appcli::get(a, "data-path", "");
      if (dp.empty()) {
        std::cerr << "error: --data-path required for --dataset graphml\n";
        return 2;
      }
      auto lg = io::read_graphml(dp);
      G = lg.graph;
      dsname = appcli::path_stem(dp);
      const double pmin = appcli::getd(a, "pmin", 1.0);
      const double flip = appcli::getd(a, "flip", 0.0);
      if (pmin < 1.0 || flip > 0.0)
        G = data::with_noise(G, pmin, flip, seed + 7);
      const std::string qp = appcli::get(a, "queries-path", "");
      if (!qp.empty()) queries = io::read_queries(qp, lg.id_to_index);
      if (queries.empty()) {
        const auto qs =
            data::sample_connected_queries(G, qmin, qmax, nq, seed + 1);
        for (size_t i = 0; i < qs.size(); ++i)
          queries.push_back({static_cast<int>(i), qs[i]});
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  std::cout << "dataset=" << dsname << " nodes=" << G.num_nodes()
            << " edges=" << G.num_edges() << " queries=" << queries.size()
            << " runs=" << runs << " timeout=" << timeout << "s\n";

  if (!export_lad.empty()) {
    std::filesystem::create_directories(export_lad);
    write_lad(export_lad + "/target.lad", G);
    for (const auto& qspec : queries)
      write_lad(export_lad + "/q" + std::to_string(qspec.id) + ".lad",
                induced_subgraph(G, qspec.nodes));
    std::cout << "exported target + " << queries.size()
              << " query patterns to " << export_lad
              << "/ (vertex-labelled LAD for Glasgow)\n";
    return 0;
  }

  const std::filesystem::path outp(out);
  if (outp.has_parent_path())
    std::filesystem::create_directories(outp.parent_path());
  io::CsvWriter csv(out);

  for (const std::string& method : methods) {
    auto matcher = make_matcher(method);
    if (!matcher) {
      std::cerr << "warning: unknown method '" << method << "' (skipped)\n";
      continue;
    }
    std::vector<double> states_v, ms_v;
    bool any_timeout = false;
    reset_assign_stats();  // assignment-bound fallback rate for this method
    for (const auto& qspec : queries) {
      const LabeledGraph Q = induced_subgraph(G, qspec.nodes);
      for (int r = 0; r < runs; ++r) {
        MatchOptions opt;
        opt.timeout_s = timeout;
        opt.max_states = max_states;
        opt.bin_tau = bin_tau;
        opt.collect_metrics = collect;
        opt.seed = seed;
        opt.weights = weights;

        const auto t0 = Clock::now();
        const MatchResult R = matcher->solve(G, Q, opt);
        const auto t1 = Clock::now();
        const int64_t wall =
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                .count();

        io::ResultRow row;
        row.dataset = dsname;
        row.query_id = qspec.id;
        row.query_size = Q.num_nodes();
        row.method = matcher->name();
        row.run_idx = r;
        row.wall_ns = wall;
        row.peak_rss_kb = appcli::peak_rss_kb();
        row.states = R.counters.states;
        row.pruned = R.counters.pruned;
        row.prune_rate = R.counters.prune_rate();
        row.best_loglik = R.counters.best_loglik;
        row.solutions = R.counters.solutions;
        row.timed_out = R.counters.timed_out ? 1 : 0;
        row.ub_gap_p50 = R.counters.ub_gap_p50;
        row.ub_gap_p90 = R.counters.ub_gap_p90;
        csv.write(row);

        states_v.push_back(static_cast<double>(R.counters.states));
        ms_v.push_back(wall / 1e6);
        any_timeout |= R.counters.timed_out;
      }
    }
    const auto as = get_assign_stats();
    std::cout << "  " << matcher->name()
              << ": median_states=" << appcli::median(states_v)
              << " median_ms=" << appcli::median(ms_v);
    if (as.calls)
      std::cout << " assign_fallback="
                << (100.0 * static_cast<double>(as.fallbacks) / as.calls) << "%";
    std::cout << (any_timeout ? " [some timed out]" : "") << "\n";
  }

  std::cout << "wrote " << out << "\n";
  return 0;
}
