// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <vector>

namespace vf2prob {

// Hungarian (Kuhn–Munkres) minimum-cost assignment of n rows to m columns with
// n <= m. All cost entries must be finite; forbidden pairs should be encoded
// with a large finite sentinel (kBigCost) by the caller. Returns the minimum
// total cost and fills row_to_col[i] with the column assigned to row i (or -1
// if the row was left unassigned, which only happens for n > m).
//
// Used by the assignment upper bound (maximize total weight == minimize total
// negated weight). Complexity O(n^2 * m).
constexpr double kBigCost = 1e18;

double hungarian_min_cost(const std::vector<std::vector<double>>& cost,
                          std::vector<int>& row_to_col);

}  // namespace vf2prob
