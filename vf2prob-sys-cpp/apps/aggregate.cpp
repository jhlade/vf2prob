// VF2-Prob, Jan Hladěna, FIM UHK
// Aggregate the per-query CSV from vf2prob_run into per-(dataset, method)
// medians/IQRs, and optionally run a paired Wilcoxon signed-rank test and a
// relative-efficiency index (REI) on explored states between two methods.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "app_common.hpp"

namespace {

std::vector<std::string> split(const std::string& s, char d) {
  std::vector<std::string> out;
  std::string cur;
  std::istringstream ss(s);
  while (std::getline(ss, cur, d)) out.push_back(cur);
  if (!s.empty() && s.back() == d) out.push_back("");
  return out;
}

double pctl(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  if (p <= 0) return v.front();
  if (p >= 100) return v.back();
  const double k = (v.size() - 1) * (p / 100.0);
  const size_t f = static_cast<size_t>(std::floor(k));
  const size_t c = static_cast<size_t>(std::ceil(k));
  if (f == c) return v[f];
  return v[f] * (c - k) + v[c] * (k - f);
}

struct PerQuery {
  std::vector<double> states, ms;
  bool timed = false;
  double solutions = 0;
};
using Key = std::pair<std::string, std::string>;  // (dataset, method)

}  // namespace

int main(int argc, char** argv) {
  const auto a = appcli::parse(argc, argv);
  const std::string in = appcli::get(a, "in", "results/out.csv");
  const std::string out = appcli::get(a, "out", "");
  const std::string compare = appcli::get(a, "compare", "");

  std::ifstream f(in);
  if (!f) {
    std::cerr << "cannot open " << in << "\n";
    return 1;
  }
  std::string line;
  if (!std::getline(f, line)) return 1;
  const auto header = split(line, ',');
  std::map<std::string, int> col;
  for (size_t i = 0; i < header.size(); ++i) col[header[i]] = static_cast<int>(i);
  for (const char* req : {"dataset", "method", "query_id", "states", "wall_ns",
                          "solutions", "timed_out"}) {
    if (!col.count(req)) {
      std::cerr << "missing column: " << req << "\n";
      return 1;
    }
  }

  // (dataset,method) -> query_id -> PerQuery
  std::map<Key, std::map<int, PerQuery>> data;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const auto fld = split(line, ',');
    if (fld.size() < header.size()) continue;
    const std::string ds = fld[col["dataset"]];
    const std::string me = fld[col["method"]];
    const int qid = std::stoi(fld[col["query_id"]]);
    PerQuery& pq = data[{ds, me}][qid];
    pq.states.push_back(std::stod(fld[col["states"]]));
    pq.ms.push_back(std::stod(fld[col["wall_ns"]]) / 1e6);
    if (std::stoi(fld[col["timed_out"]])) pq.timed = true;
    pq.solutions = std::max(pq.solutions, std::stod(fld[col["solutions"]]));
  }

  // Aggregate.
  std::ofstream of;
  std::ostream* o = &std::cout;
  if (!out.empty()) {
    of.open(out);
    o = &of;
  }
  *o << "dataset,method,n_queries,states_med,states_iqr,ms_med,ms_iqr,"
        "success_rate,any_timeout\n";
  for (const auto& [key, byq] : data) {
    std::vector<double> qstates, qms;
    int solved = 0, any_to = 0;
    for (const auto& [qid, pq] : byq) {
      qstates.push_back(pctl(pq.states, 50));
      qms.push_back(pctl(pq.ms, 50));
      const bool ok = pq.solutions >= 1 && !pq.timed;
      if (ok) ++solved;
      if (pq.timed) any_to = 1;
    }
    const double sr = byq.empty() ? 0.0 : double(solved) / byq.size();
    *o << key.first << ',' << key.second << ',' << byq.size() << ','
       << pctl(qstates, 50) << ',' << (pctl(qstates, 75) - pctl(qstates, 25))
       << ',' << pctl(qms, 50) << ',' << (pctl(qms, 75) - pctl(qms, 25)) << ','
       << sr << ',' << any_to << "\n";
  }
  if (!out.empty()) std::cerr << "wrote " << out << "\n";

  // Paired Wilcoxon signed-rank + REI on states between two methods.
  if (!compare.empty()) {
    const auto parts = appcli::split_csv(
        std::string(compare).replace(compare.find(':'), 1, ","));
    if (parts.size() != 2) {
      std::cerr << "--compare expects methodA:methodB\n";
      return 1;
    }
    const std::string& A = parts[0];
    const std::string& B = parts[1];
    std::vector<double> xs, ys;  // per shared (dataset,query) median states
    for (const auto& [key, byq] : data) {
      if (key.second != A) continue;
      const Key kb{key.first, B};
      auto itb = data.find(kb);
      if (itb == data.end()) continue;
      for (const auto& [qid, pqa] : byq) {
        auto jb = itb->second.find(qid);
        if (jb == itb->second.end()) continue;
        xs.push_back(pctl(pqa.states, 50));
        ys.push_back(pctl(jb->second.states, 50));
      }
    }
    const size_t np = xs.size();
    std::cerr << "\n[compare] " << A << " vs " << B << " on states (" << np
              << " paired queries)\n";
    if (np == 0) return 0;

    // Signed-rank with average ties.
    std::vector<double> d, ad;
    for (size_t i = 0; i < np; ++i) {
      const double diff = xs[i] - ys[i];
      if (diff != 0.0) {
        d.push_back(diff);
        ad.push_back(std::fabs(diff));
      }
    }
    const size_t n = d.size();
    double wp = 0, wm = 0, z = 0, p = 1.0;
    if (n > 0) {
      std::vector<size_t> idx(n);
      for (size_t i = 0; i < n; ++i) idx[i] = i;
      std::sort(idx.begin(), idx.end(),
                [&](size_t i, size_t j) { return ad[i] < ad[j]; });
      std::vector<double> rank(n);
      size_t i = 0;
      while (i < n) {
        size_t j = i;
        while (j < n && ad[idx[j]] == ad[idx[i]]) ++j;
        const double avg = ((i + 1) + j) / 2.0;
        for (size_t k = i; k < j; ++k) rank[idx[k]] = avg;
        i = j;
      }
      for (size_t k = 0; k < n; ++k) (d[k] > 0 ? wp : wm) += rank[k];
      const double mean = n * (n + 1) / 4.0;
      const double var = n * (n + 1) * (2.0 * n + 1) / 24.0;
      z = var > 0 ? (wp - mean) / std::sqrt(var) : 0.0;
      p = std::erfc(std::fabs(z) / std::sqrt(2.0));  // two-sided normal approx
    }
    std::vector<double> ratio_ba;  // B has fewer? states_A / states_B
    for (size_t i = 0; i < np; ++i)
      if (ys[i] > 0) ratio_ba.push_back(xs[i] / ys[i]);
    std::cerr << "  median states " << A << "=" << pctl(xs, 50) << "  " << B
              << "=" << pctl(ys, 50) << "\n";
    std::cerr << "  REI median(" << A << "/" << B << ")=" << pctl(ratio_ba, 50)
              << "  (>1 => " << B << " explores fewer)\n";
    std::cerr << "  Wilcoxon signed-rank: W+=" << wp << " W-=" << wm
              << " z=" << z << " p=" << p << " (n=" << n << " nonzero)\n";
  }
  return 0;
}
