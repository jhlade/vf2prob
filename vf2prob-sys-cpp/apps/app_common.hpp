// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <sys/resource.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

// Small CLI helpers shared by the apps (header-only; no extra link unit).
namespace appcli {

// Parse "--key value" pairs and "--flag" toggles from argv into a map.
inline std::unordered_map<std::string, std::string> parse(int argc,
                                                          char** argv) {
  std::unordered_map<std::string, std::string> m;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) != 0) continue;
    const std::string key = a.substr(2);
    if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      m[key] = argv[++i];
    } else {
      m[key] = "1";  // flag
    }
  }
  return m;
}

inline std::string get(const std::unordered_map<std::string, std::string>& m,
                       const std::string& k, const std::string& def) {
  auto it = m.find(k);
  return it == m.end() ? def : it->second;
}
inline double getd(const std::unordered_map<std::string, std::string>& m,
                   const std::string& k, double def) {
  auto it = m.find(k);
  return it == m.end() ? def : std::stod(it->second);
}
inline int geti(const std::unordered_map<std::string, std::string>& m,
                const std::string& k, int def) {
  auto it = m.find(k);
  return it == m.end() ? def : std::stoi(it->second);
}

inline std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      if (!cur.empty()) out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

// Process peak resident set size in KiB (getrusage; macOS reports bytes).
inline long peak_rss_kb() {
  struct rusage ru {};
  getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
  return ru.ru_maxrss / 1024;  // bytes -> KiB
#else
  return ru.ru_maxrss;  // already KiB
#endif
}

// Filename stem (no directory, no extension) — used as the dataset name.
inline std::string path_stem(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  const std::string base =
      slash == std::string::npos ? path : path.substr(slash + 1);
  const size_t dot = base.find_last_of('.');
  return dot == std::string::npos ? base : base.substr(0, dot);
}

inline double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace appcli
