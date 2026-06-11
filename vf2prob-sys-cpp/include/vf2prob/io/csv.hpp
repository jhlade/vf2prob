// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <cstdint>
#include <string>

namespace vf2prob::io {

// One per-(dataset, query, method, run) measurement row (long-format).
// Aggregation to median/IQR is done downstream.
struct ResultRow {
  std::string dataset;
  int query_id = 0;
  int query_size = 0;
  std::string method;
  int run_idx = 0;
  int64_t wall_ns = 0;       // wall-clock for the solve only
  int64_t peak_rss_kb = 0;   // process peak RSS (getrusage)
  uint64_t states = 0;
  uint64_t pruned = 0;
  double prune_rate = 0.0;
  double best_loglik = 0.0;  // empty cell when -inf
  uint64_t solutions = 0;
  int timed_out = 0;
  double ub_gap_p50 = 0.0;   // empty cell when NaN
  double ub_gap_p90 = 0.0;
};

// Appends rows to `path`, writing the header first if the file is new/empty.
class CsvWriter {
 public:
  explicit CsvWriter(const std::string& path);
  void write(const ResultRow& r);

 private:
  std::string path_;
  bool need_header_;
};

const char* csv_header();

}  // namespace vf2prob::io
