// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <string>
#include <unordered_map>

#include "vf2prob/graph.hpp"

namespace vf2prob::io {

struct LoadedGraph {
  LabeledGraph graph;
  // Original node-id string -> compact index (queries reference original ids).
  std::unordered_map<std::string, NodeId> id_to_index;
};

// Read a GraphML file. Attribute-driven (maps <key> ids to attr.name/for), so
// it handles both our writer's output and networkx's. Categorical `label`
// values are parsed as integers (the LabelId); non-integer labels fall back to
// an interned vocabulary. Edge `prob` defaults to 1.0.
LoadedGraph read_graphml(const std::string& path);

// Write a GraphML file with integer node/edge labels and edge probabilities.
void write_graphml(const std::string& path, const LabeledGraph& G);

}  // namespace vf2prob::io
