// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "vf2prob/graph.hpp"

namespace vf2prob::io {

// One query: an id and the data-graph node ids (compact indices) that induce it.
struct QuerySpec {
  int id = 0;
  std::vector<NodeId> nodes;
};

// Read queries from a JSON file of the form
//   {"meta": {...}, "queries": [{"id": 0, "nodes": [..]}, ...]}
// Node ids are looked up through id_to_index; ids not present are dropped.
// Queries with fewer than 2 surviving nodes are skipped.
std::vector<QuerySpec> read_queries(
    const std::string& path,
    const std::unordered_map<std::string, NodeId>& id_to_index);

// Write queries (node lists are original/compact ids) in the same JSON format.
void write_queries(const std::string& path, const std::string& regime,
                   const std::vector<std::vector<NodeId>>& queries, int seed);

}  // namespace vf2prob::io
