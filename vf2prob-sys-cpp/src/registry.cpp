// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/registry.hpp"

#include "vf2prob/algorithms/mpm.hpp"
#include "vf2prob/algorithms/vf2.hpp"
#include "vf2prob/algorithms/vf2bin.hpp"
#include "vf2prob/algorithms/vf2pp.hpp"
#include "vf2prob/algorithms/vf2prob.hpp"

namespace vf2prob {

std::unique_ptr<IMatcher> make_matcher(const std::string& name) {
  if (name == "vf2") return std::make_unique<VF2Matcher>();
  if (name == "vf2pp") return std::make_unique<VF2ppMatcher>();
  if (name == "vf2bin") return std::make_unique<VF2BinMatcher>();
  if (name == "mpm") return std::make_unique<MpmMatcher>();
  if (name == "vf2prob")
    return std::make_unique<VF2ProbMatcher>(BoundKind::Node, SearchKind::DFS);
  if (name == "vf2prob-assign")
    return std::make_unique<VF2ProbMatcher>(BoundKind::Assign, SearchKind::DFS);
  if (name == "vf2prob-astar")
    return std::make_unique<VF2ProbMatcher>(BoundKind::Node, SearchKind::AStar);
  if (name == "vf2prob-astar-assign")
    return std::make_unique<VF2ProbMatcher>(BoundKind::Assign,
                                            SearchKind::AStar);
  return nullptr;
}

std::vector<std::string> all_method_names() {
  return {"vf2", "vf2pp", "vf2bin", "mpm",
          "vf2prob", "vf2prob-assign", "vf2prob-astar", "vf2prob-astar-assign"};
}

}  // namespace vf2prob
