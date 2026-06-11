// VF2-Prob, Jan Hladěna, FIM UHK
// tests auto-generated
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "test_framework.hpp"
#include "vf2prob/assignment.hpp"

using namespace vf2prob;

// Brute-force minimum-cost assignment of n rows to m>=n columns.
static double brute_min(const std::vector<std::vector<double>>& c) {
  const int n = static_cast<int>(c.size());
  const int m = static_cast<int>(c[0].size());
  std::vector<int> cols(m);
  std::iota(cols.begin(), cols.end(), 0);
  double best = 1e300;
  std::sort(cols.begin(), cols.end());
  do {
    double s = 0;
    for (int i = 0; i < n; ++i) s += c[i][cols[i]];
    best = std::min(best, s);
  } while (std::next_permutation(cols.begin(), cols.end()));
  return best;
}

TEST("hungarian: matches brute force on random square matrices") {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> u(-5.0, 5.0);
  for (int trial = 0; trial < 50; ++trial) {
    const int n = 1 + static_cast<int>(rng() % 5);  // 1..5
    std::vector<std::vector<double>> c(n, std::vector<double>(n));
    for (auto& row : c)
      for (auto& x : row) x = u(rng);
    std::vector<int> assign;
    const double got = hungarian_min_cost(c, assign);
    CHECK_NEAR(got, brute_min(c), 1e-6);
  }
}

TEST("hungarian: handles rectangular n < m") {
  std::mt19937 rng(777);
  std::uniform_real_distribution<double> u(0.0, 10.0);
  for (int trial = 0; trial < 30; ++trial) {
    const int n = 1 + static_cast<int>(rng() % 4);
    const int m = n + static_cast<int>(rng() % 3);
    std::vector<std::vector<double>> c(n, std::vector<double>(m));
    for (auto& row : c)
      for (auto& x : row) x = u(rng);
    std::vector<int> assign;
    const double got = hungarian_min_cost(c, assign);
    CHECK_NEAR(got, brute_min(c), 1e-6);
    // assignment is injective
    std::vector<int> seen;
    for (int i = 0; i < n; ++i) {
      CHECK(assign[i] >= 0 && assign[i] < m);
      seen.push_back(assign[i]);
    }
    std::sort(seen.begin(), seen.end());
    CHECK(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
  }
}
