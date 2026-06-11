// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/io/queries.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vf2prob::io {

namespace {
std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open queries: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}
}  // namespace

std::vector<QuerySpec> read_queries(
    const std::string& path,
    const std::unordered_map<std::string, NodeId>& id_to_index) {
  const std::string s = slurp(path);
  std::vector<QuerySpec> out;

  // Lightweight scan: each query is the "nodes" array following an (optional)
  // "id". We assign sequential ids in order of appearance, which is all the
  // harness needs for labeling.
  size_t p = 0;
  int seq = 0;
  while ((p = s.find("\"nodes\"", p)) != std::string::npos) {
    const size_t lb = s.find('[', p);
    if (lb == std::string::npos) break;
    const size_t rb = s.find(']', lb);
    if (rb == std::string::npos) break;

    QuerySpec q;
    q.id = seq++;
    size_t i = lb + 1;
    while (i < rb) {
      while (i < rb && !(std::isdigit(static_cast<unsigned char>(s[i])) ||
                         s[i] == '-'))
        ++i;
      if (i >= rb) break;
      size_t j = i;
      while (j < rb && (std::isdigit(static_cast<unsigned char>(s[j])) ||
                        s[j] == '-'))
        ++j;
      const std::string tok = s.substr(i, j - i);
      auto it = id_to_index.find(tok);
      if (it != id_to_index.end()) q.nodes.push_back(it->second);
      i = j;
    }
    if (q.nodes.size() >= 2) out.push_back(std::move(q));
    p = rb + 1;
  }
  return out;
}

void write_queries(const std::string& path, const std::string& regime,
                   const std::vector<std::vector<NodeId>>& queries, int seed) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write queries: " + path);
  f << "{\n  \"meta\": {\"regime\": \"" << regime << "\", \"seed\": " << seed
    << ", \"nqueries\": " << queries.size() << "},\n";
  f << "  \"queries\": [\n";
  for (size_t i = 0; i < queries.size(); ++i) {
    f << "    {\"id\": " << i << ", \"nodes\": [";
    for (size_t j = 0; j < queries[i].size(); ++j) {
      if (j) f << ", ";
      f << queries[i][j];
    }
    f << "]}";
    f << (i + 1 < queries.size() ? ",\n" : "\n");
  }
  f << "  ]\n}\n";
}

}  // namespace vf2prob::io
