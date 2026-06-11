// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/assignment.hpp"

#include <limits>

namespace vf2prob {

// Classic O(n^2 m) Hungarian with potentials (the standard "cp-algorithms"
// formulation), adapted from 1-indexed to our 0-indexed input. Solves the
// rectangular case n <= m by leaving (m - n) columns unmatched.
double hungarian_min_cost(const std::vector<std::vector<double>>& a,
                          std::vector<int>& row_to_col) {
  const int n = static_cast<int>(a.size());
  row_to_col.assign(n, -1);
  if (n == 0) return 0.0;
  const int m = static_cast<int>(a[0].size());

  const double INF = std::numeric_limits<double>::infinity();
  std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
  std::vector<int> p(m + 1, 0), way(m + 1, 0);

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(m + 1, INF);
    std::vector<char> used(m + 1, 0);
    do {
      used[j0] = 1;
      const int i0 = p[j0];
      int j1 = -1;
      double delta = INF;
      for (int j = 1; j <= m; ++j) {
        if (used[j]) continue;
        const double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= m; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  double cost = 0.0;
  for (int j = 1; j <= m; ++j) {
    if (p[j] != 0) {
      row_to_col[p[j] - 1] = j - 1;
      cost += a[p[j] - 1][j - 1];
    }
  }
  return cost;
}

}  // namespace vf2prob
