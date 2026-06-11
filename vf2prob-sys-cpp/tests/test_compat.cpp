// VF2-Prob, Jan Hladěna, FIM UHK
// tests auto-generated
#include "test_framework.hpp"
#include "vf2prob/compat.hpp"

using namespace vf2prob;

TEST("compat: node logistic matches calibrated constants") {
  // sigma(+3) for a match, sigma(-3) for a mismatch.
  CHECK_NEAR(compat::node_compat(1, 1), 0.9525741, 1e-5);
  CHECK_NEAR(compat::node_compat(1, 2), 0.0474259, 1e-5);
  CHECK(compat::node_compat(1, 1) > compat::node_compat(1, 2));
}

TEST("compat: edge logistic is nonlinear in the probability and label") {
  // Higher existence probability and a label match both raise s_e.
  CHECK(compat::edge_compat(true, 0.9) > compat::edge_compat(true, 0.6));
  CHECK(compat::edge_compat(true, 0.8) > compat::edge_compat(false, 0.8));
  // z = -1 + 3 + 2*logit(0.8) = 4.7726 -> sigma ~ 0.99161.
  CHECK_NEAR(compat::edge_compat(true, 0.8), 0.991614, 1e-4);
  // All values stay in (0, 1].
  CHECK(compat::edge_compat(false, 1e-6) > 0.0);
  CHECK(compat::edge_compat(true, 1.0) <= 1.0);
}

TEST("compat: safe_log clamps to (0,1]") {
  CHECK_NEAR(compat::safe_log(1.0), 0.0, 1e-12);
  CHECK(compat::safe_log(0.5) < 0.0);
  CHECK(std::isfinite(compat::safe_log(0.0)));  // floored, not -inf
}
