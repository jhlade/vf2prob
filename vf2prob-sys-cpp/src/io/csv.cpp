// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/io/csv.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace vf2prob::io {

const char* csv_header() {
  return "dataset,query_id,query_size,method,run_idx,wall_ns,peak_rss_kb,"
         "states,pruned,prune_rate,best_loglik,solutions,timed_out,"
         "ub_gap_p50,ub_gap_p90";
}

namespace {
std::string fmt_double(double v, const char* spec) {
  if (std::isnan(v) || std::isinf(v)) return std::string();  // empty cell
  char buf[64];
  std::snprintf(buf, sizeof(buf), spec, v);
  return buf;
}
}  // namespace

CsvWriter::CsvWriter(const std::string& path) : path_(path), need_header_(true) {
  // Truncate any existing file so each run starts fresh. Appending to stale
  // results from a previous invocation would silently mix runs (e.g. different
  // state budgets) and break reproducibility of the aggregated numbers.
  std::ofstream f(path_, std::ios::trunc);
  if (!f) throw std::runtime_error("cannot open CSV for writing: " + path_);
}

void CsvWriter::write(const ResultRow& r) {
  std::ofstream f(path_, std::ios::app);
  if (!f) throw std::runtime_error("cannot append to CSV: " + path_);
  if (need_header_) {
    f << csv_header() << "\n";
    need_header_ = false;
  }
  f << r.dataset << ',' << r.query_id << ',' << r.query_size << ',' << r.method
    << ',' << r.run_idx << ',' << r.wall_ns << ',' << r.peak_rss_kb << ','
    << r.states << ',' << r.pruned << ',' << fmt_double(r.prune_rate, "%.4f")
    << ',' << fmt_double(r.best_loglik, "%.6f") << ',' << r.solutions << ','
    << r.timed_out << ',' << fmt_double(r.ub_gap_p50, "%.6f") << ','
    << fmt_double(r.ub_gap_p90, "%.6f") << "\n";
}

}  // namespace vf2prob::io
