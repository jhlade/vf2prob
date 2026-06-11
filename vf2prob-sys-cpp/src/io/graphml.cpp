// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/io/graphml.hpp"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace vf2prob::io {

namespace {

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open GraphML: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

// Extract the value of attribute `name` from a tag's attribute string. Requires
// `name` to start at a token boundary, so "name" does not match "attr.name".
std::optional<std::string> attr(const std::string& tag, const std::string& name) {
  size_t p = 0;
  while ((p = tag.find(name, p)) != std::string::npos) {
    const bool boundary =
        (p == 0) || std::isspace(static_cast<unsigned char>(tag[p - 1]));
    size_t q = p + name.size();
    while (q < tag.size() && std::isspace(static_cast<unsigned char>(tag[q])))
      ++q;
    if (boundary && q < tag.size() && tag[q] == '=') {
      ++q;
      while (q < tag.size() && std::isspace(static_cast<unsigned char>(tag[q])))
        ++q;
      if (q < tag.size() && (tag[q] == '"' || tag[q] == '\'')) {
        const char quote = tag[q];
        const size_t s = q + 1;
        const size_t f = tag.find(quote, s);
        if (f != std::string::npos) return tag.substr(s, f - s);
      }
    }
    p += name.size();
  }
  return std::nullopt;
}

// A label vocabulary: integer-looking labels map to their integer value;
// everything else is interned above a high base to avoid collisions.
struct LabelVocab {
  std::unordered_map<std::string, LabelId> interned;
  LabelId next = 1'000'000;
  LabelId get(const std::string& raw) {
    const std::string v = trim(raw);
    if (v.empty()) return kNoLabel;
    try {
      size_t pos = 0;
      const long val = std::stol(v, &pos);
      if (pos == v.size()) return static_cast<LabelId>(val);
    } catch (...) {
    }
    auto it = interned.find(v);
    if (it != interned.end()) return it->second;
    return interned[v] = next++;
  }
};

// Collect (key-id, value) pairs from <data ...>value</data> within a block.
std::vector<std::pair<std::string, std::string>> parse_datas(
    const std::string& block) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t p = 0;
  while ((p = block.find("<data", p)) != std::string::npos) {
    const size_t gt = block.find('>', p);
    if (gt == std::string::npos) break;
    const std::string tag = block.substr(p, gt - p);
    const auto key = attr(tag, "key");
    const size_t end = block.find("</data>", gt);
    if (end == std::string::npos) break;
    const std::string val = block.substr(gt + 1, end - gt - 1);
    if (key) out.emplace_back(*key, val);
    p = end + 7;
  }
  return out;
}

}  // namespace

LoadedGraph read_graphml(const std::string& path) {
  const std::string s = slurp(path);

  // 1) key id -> attr.name (so we can interpret <data key=...> regardless of
  //    whether ids are "d0" (networkx) or "nl"/"el"/"ep" (our writer)).
  std::unordered_map<std::string, std::string> key_name;
  {
    size_t p = 0;
    while ((p = s.find("<key", p)) != std::string::npos) {
      const size_t gt = s.find('>', p);
      if (gt == std::string::npos) break;
      const std::string tag = s.substr(p, gt - p);
      const auto id = attr(tag, "id");
      const auto name = attr(tag, "attr.name");
      if (id && name) key_name[*id] = *name;
      p = gt + 1;
    }
  }

  LabelVocab node_vocab, edge_vocab;
  std::vector<LabelId> node_labels;
  std::unordered_map<std::string, NodeId> id_to_index;

  // 2) nodes (in order of appearance => compact index).
  {
    size_t p = 0;
    while ((p = s.find("<node", p)) != std::string::npos) {
      // guard: must be the element <node, not a longer name
      const char after = s[p + 5];
      if (!(std::isspace(static_cast<unsigned char>(after)) || after == '>' ||
            after == '/')) {
        p += 5;
        continue;
      }
      const size_t gt = s.find('>', p);
      if (gt == std::string::npos) break;
      const std::string open = s.substr(p, gt - p);
      const auto id = attr(open, "id");
      std::string inner;
      size_t advance = gt + 1;
      if (gt > 0 && s[gt - 1] != '/') {  // not self-closing -> has body
        const size_t end = s.find("</node>", gt);
        if (end != std::string::npos) {
          inner = s.substr(gt + 1, end - gt - 1);
          advance = end + 7;
        }
      }
      const NodeId idx = static_cast<NodeId>(node_labels.size());
      LabelId lab = kNoLabel;
      for (const auto& [k, v] : parse_datas(inner)) {
        auto kn = key_name.find(k);
        if (kn != key_name.end() && kn->second == "label") lab = node_vocab.get(v);
      }
      node_labels.push_back(lab);
      if (id) id_to_index[*id] = idx;
      p = advance;
    }
  }

  // 3) edges.
  std::vector<LabeledGraph::Edge> edges;
  {
    size_t p = 0;
    while ((p = s.find("<edge", p)) != std::string::npos) {
      const char after = s[p + 5];
      if (!(std::isspace(static_cast<unsigned char>(after)) || after == '>' ||
            after == '/')) {
        p += 5;
        continue;
      }
      const size_t gt = s.find('>', p);
      if (gt == std::string::npos) break;
      const std::string open = s.substr(p, gt - p);
      const auto src = attr(open, "source");
      const auto dst = attr(open, "target");
      std::string inner;
      size_t advance = gt + 1;
      if (gt > 0 && s[gt - 1] != '/') {
        const size_t end = s.find("</edge>", gt);
        if (end != std::string::npos) {
          inner = s.substr(gt + 1, end - gt - 1);
          advance = end + 7;
        }
      }
      LabelId elab = kNoLabel;
      double prob = 1.0;
      for (const auto& [k, v] : parse_datas(inner)) {
        auto kn = key_name.find(k);
        if (kn == key_name.end()) continue;
        if (kn->second == "label")
          elab = edge_vocab.get(v);
        else if (kn->second == "prob") {
          try {
            prob = std::stod(trim(v));
          } catch (...) {
            prob = 1.0;
          }
        }
      }
      if (src && dst) {
        auto a = id_to_index.find(*src);
        auto b = id_to_index.find(*dst);
        if (a != id_to_index.end() && b != id_to_index.end())
          edges.emplace_back(a->second, b->second, elab, prob);
      }
      p = advance;
    }
  }

  LoadedGraph out;
  out.graph = LabeledGraph(static_cast<NodeId>(node_labels.size()),
                           std::move(node_labels), edges);
  out.id_to_index = std::move(id_to_index);
  return out;
}

void write_graphml(const std::string& path, const LabeledGraph& G) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write GraphML: " + path);
  f << "<?xml version='1.0' encoding='utf-8'?>\n";
  f << "<graphml>\n";
  f << "  <key id=\"nl\" for=\"node\" attr.name=\"label\" attr.type=\"long\"/>\n";
  f << "  <key id=\"el\" for=\"edge\" attr.name=\"label\" attr.type=\"long\"/>\n";
  f << "  <key id=\"ep\" for=\"edge\" attr.name=\"prob\" attr.type=\"double\"/>\n";
  f << "  <graph edgedefault=\"undirected\">\n";
  for (NodeId u = 0; u < G.num_nodes(); ++u) {
    f << "    <node id=\"" << u << "\">";
    if (G.node_label(u) != kNoLabel)
      f << "<data key=\"nl\">" << G.node_label(u) << "</data>";
    f << "</node>\n";
  }
  char buf[64];
  for (NodeId u = 0; u < G.num_nodes(); ++u) {
    for (const NodeId* it = G.neighbors_begin(u); it != G.neighbors_end(u);
         ++it) {
      const NodeId v = *it;
      if (v <= u) continue;
      f << "    <edge source=\"" << u << "\" target=\"" << v << "\">";
      if (G.edge_label(u, v) != kNoLabel)
        f << "<data key=\"el\">" << G.edge_label(u, v) << "</data>";
      std::snprintf(buf, sizeof(buf), "%.6g", G.edge_prob(u, v));
      f << "<data key=\"ep\">" << buf << "</data>";
      f << "</edge>\n";
    }
  }
  f << "  </graph>\n</graphml>\n";
}

}  // namespace vf2prob::io
