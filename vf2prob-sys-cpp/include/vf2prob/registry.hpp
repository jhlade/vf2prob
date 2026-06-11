// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "vf2prob/matcher.hpp"

namespace vf2prob {

// Construct a matcher by method name. Recognized names:

//   vf2, vf2pp, vf2bin,
//   variants: vf2prob, vf2prob-assign, vf2prob-astar, vf2prob-astar-assign

// Returns nullptr for an unknown name.
std::unique_ptr<IMatcher> make_matcher(const std::string& name);

// All known method names, in a canonical order.
std::vector<std::string> all_method_names();

}  // namespace vf2prob
