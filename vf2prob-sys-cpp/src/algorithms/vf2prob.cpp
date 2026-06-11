// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/algorithms/vf2prob.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <queue>
#include <random>
#include <vector>

#include "vf2prob/bounds.hpp"
#include "vf2prob/compat.hpp"
#include "vf2prob/ml/ranker.hpp"
#include "vf2prob/ordering.hpp"

namespace vf2prob {

using Clock = std::chrono::steady_clock;

std::string VF2ProbMatcher::name() const {
  std::string n = "vf2prob";
  if (search_ == SearchKind::AStar) n += "-astar";
  if (bound_ == BoundKind::Assign) n += "-assign";
  return n;
}

// Reservoir of bound-tightness samples (deficit between bound-based estimate
// and the incumbent), capped to keep overhead bounded.
namespace {
struct GapReservoir {
  std::vector<double> buf;
  uint64_t seen = 0;
  std::mt19937_64 rng;
  static constexpr size_t kCap = 4096;
  explicit GapReservoir(uint64_t seed) : rng(seed) {}
  void add(double x) {
    ++seen;
    if (buf.size() < kCap) {
      buf.push_back(x);
    } else {
      std::uniform_int_distribution<uint64_t> d(0, seen - 1);
      const uint64_t j = d(rng);
      if (j < kCap) buf[j] = x;
    }
  }
  void finalize(Counters& c) const {
    if (buf.empty()) return;
    std::vector<double> s = buf;
    std::sort(s.begin(), s.end());
    const size_t k = s.size();
    c.ub_gap_p50 = s[static_cast<size_t>(0.5 * (k - 1))];
    c.ub_gap_p90 = s[static_cast<size_t>(0.9 * (k - 1))];
  }
};
}  // namespace

// Incremental gain of mapping q -> u under the current partial mapping:
// node log-compat plus edge log-compat to already-mapped neighbors.
static double delta_gain(const LabeledGraph& G, const LabeledGraph& Q, NodeId q,
                         NodeId u, const std::vector<NodeId>& mapping,
                         const MatchOptions& opt) {
  double g = opt.use_node
                 ? compat::safe_log(compat::node_compat(
                       Q.node_label(q), G.node_label(u), opt.weights))
                 : 0.0;
  if (opt.use_edge) {
    for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q);
         ++it) {
      const NodeId qn = *it;
      const NodeId un = mapping[qn];
      if (un < 0) continue;
      g += compat::safe_log(compat::edge_compat(
          Q.edge_label(q, qn), G.edge_label(u, un), G.edge_prob(u, un),
          opt.weights));
    }
  }
  return g;
}

MatchResult VF2ProbMatcher::solve(const LabeledGraph& G, const LabeledGraph& Q,
                                  const MatchOptions& opt) {
  MatchOptions o = opt;
  o.bound = bound_;
  o.search = search_;
  return search_ == SearchKind::AStar ? solve_astar(G, Q, o)
                                      : solve_dfs(G, Q, o);
}

MatchResult VF2ProbMatcher::solve_dfs(const LabeledGraph& G,
                                      const LabeledGraph& Q,
                                      const MatchOptions& opt) {
  MatchResult R;
  Counters& C = R.counters;
  const int nq = Q.num_nodes();
  if (nq == 0) {
    R.mapping = std::vector<NodeId>();
    C.best_loglik = 0.0;
    C.solutions = 1;
    return R;
  }

  const std::vector<NodeId> order = failfirst_order(Q);
  const std::vector<double> max_node_log = precompute_max_node_log(G, Q, opt);
  std::vector<NodeId> mapping(nq, -1);
  std::vector<char> used(G.num_nodes(), 0);

  double best_score = kNegInf;
  std::vector<NodeId> best_map;
  GapReservoir gaps(opt.seed);
  const auto t0 = Clock::now();
  bool stop = false;

  auto timed_out = [&]() {
    if (opt.timeout_s <= 0) return false;
    return std::chrono::duration<double>(Clock::now() - t0).count() >
           opt.timeout_s;
  };
  auto UB = [&](const std::vector<NodeId>& mp,
                const std::vector<char>& us) -> double {
    return opt.bound == BoundKind::Assign
               ? assign_upper_bound(G, Q, mp, us, max_node_log, opt)
               : node_upper_bound(Q, mp, max_node_log);
  };

  // dfs(i, cur): order[0..i-1] are mapped; explore order[i].
  std::function<void(int, double)> dfs = [&](int i, double cur) {
    if (stop) return;
    if (timed_out() || (opt.max_states && C.states >= opt.max_states)) {
      C.timed_out = true;  // incomplete: time or state budget exhausted
      stop = true;
      return;
    }
    if (i == nq) {
      C.solutions += 1;
      if (cur > best_score) {
        best_score = cur;
        best_map = mapping;
      }
      return;
    }

    const double ub = UB(mapping, used);
    if (opt.collect_metrics && best_score > kNegInf)
      gaps.add(std::max(0.0, (cur + ub) - best_score));
    if (cur + ub <= best_score) {
      C.pruned += 1;
      return;
    }

    const NodeId q = order[i];
    std::vector<NodeId> cands = feasible_candidates(G, Q, q, mapping, used);

    // Order candidates by descending score: the learned ranker if one is
    // supplied, else the immediate gain (id-ascending tie-break). Ordering is
    // heuristic only -- it never affects feasibility or the optimum -- so this
    // leaves the exact maximum-likelihood guarantee intact.
    struct Cand {
      double key;
      double gain;
      NodeId u;
    };
    const bool use_ranker = opt.rank_w.size() >= static_cast<size_t>(ml::kRankFeats);
    std::vector<Cand> ranked;
    ranked.reserve(cands.size());
    for (NodeId u : cands) {
      const double gain = delta_gain(G, Q, q, u, mapping, opt);
      double key = gain;
      if (use_ranker) {
        ml::RankFeatures f = ml::rank_features(G, Q, q, u, mapping, opt.weights,
                                               opt.use_node, opt.use_edge);
        f.gain = gain;
        key = ml::rank_score(opt.rank_w, f);
      }
      ranked.push_back({key, gain, u});
    }
    std::sort(ranked.begin(), ranked.end(), [](const Cand& a, const Cand& b) {
      if (a.key != b.key) return a.key > b.key;
      return a.u < b.u;
    });

    for (const Cand& c : ranked) {
      C.states += 1;
      mapping[q] = c.u;
      used[c.u] = 1;
      dfs(i + 1, cur + c.gain);
      mapping[q] = -1;
      used[c.u] = 0;
      if (stop) return;
    }
  };

  dfs(0, 0.0);

  C.best_loglik = best_score;
  if (best_score > kNegInf) R.mapping = best_map;
  gaps.finalize(C);
  return R;
}

namespace {
// Frontier nodes store only the partial mapping (size |V_Q|, small). The used[]
// marker (size |V_G|) is reconstructed into a single scratch buffer on
// expansion, so memory per queued node is O(|V_Q|) instead of O(|V_G|).
struct AStarNode {
  double f;
  double g;
  int depth;
  uint64_t seq;
  std::vector<NodeId> mapping;
};
struct AStarCmp {
  // priority_queue is a max-heap under this "less" comparator: top = max f,
  // then max g, then min seq (deterministic).
  bool operator()(const AStarNode& a, const AStarNode& b) const {
    if (a.f != b.f) return a.f < b.f;
    if (a.g != b.g) return a.g < b.g;
    return a.seq > b.seq;
  }
};
}  // namespace

MatchResult VF2ProbMatcher::solve_astar(const LabeledGraph& G,
                                        const LabeledGraph& Q,
                                        const MatchOptions& opt) {
  MatchResult R;
  Counters& C = R.counters;
  const int nq = Q.num_nodes();
  if (nq == 0) {
    R.mapping = std::vector<NodeId>();
    C.best_loglik = 0.0;
    C.solutions = 1;
    return R;
  }

  const std::vector<double> max_node_log = precompute_max_node_log(G, Q, opt);
  GapReservoir gaps(opt.seed);
  const auto t0 = Clock::now();

  auto timed_out = [&]() {
    if (opt.timeout_s <= 0) return false;
    return std::chrono::duration<double>(Clock::now() - t0).count() >
           opt.timeout_s;
  };
  auto UB = [&](const std::vector<NodeId>& mp,
                const std::vector<char>& us) -> double {
    return opt.bound == BoundKind::Assign
               ? assign_upper_bound(G, Q, mp, us, max_node_log, opt)
               : node_upper_bound(Q, mp, max_node_log);
  };

  auto has_mapped_neighbor = [&](NodeId q, const std::vector<NodeId>& mp) {
    for (const NodeId* it = Q.neighbors_begin(q); it != Q.neighbors_end(q); ++it)
      if (mp[*it] >= 0) return true;
    return false;
  };
  // Most-constrained query node, restricted to the frontier (nodes adjacent to a
  // mapped node) to keep candidate generation neighborhood-bounded; for a
  // connected query this still reaches every embedding. Ties: fewest candidates,
  // then higher degree, then id; falls back to max-degree unmapped.
  auto select_q = [&](const std::vector<NodeId>& mp,
                      const std::vector<char>& us) -> NodeId {
    NodeId best = -1;
    long best_k = 0;
    for (NodeId q = 0; q < nq; ++q) {
      if (mp[q] >= 0 || !has_mapped_neighbor(q, mp)) continue;
      const long k =
          static_cast<long>(feasible_candidates(G, Q, q, mp, us).size());
      if (best < 0 || k < best_k ||
          (k == best_k && Q.degree(q) > Q.degree(best))) {
        best = q;
        best_k = k;
      }
    }
    if (best >= 0) return best;
    for (NodeId q = 0; q < nq; ++q)
      if (mp[q] < 0 && (best < 0 || Q.degree(q) > Q.degree(best))) best = q;
    return best;
  };

  // Single reusable used[] buffer, set to a node's mapped images on expansion
  // and cleared afterwards (so queued nodes need not each carry it).
  std::vector<char> used(G.num_nodes(), 0);
  auto set_used = [&](const std::vector<NodeId>& mp, char v) {
    for (NodeId q = 0; q < nq; ++q)
      if (mp[q] >= 0) used[mp[q]] = v;
  };

  std::priority_queue<AStarNode, std::vector<AStarNode>, AStarCmp> pq;
  uint64_t seq = 0;
  {
    std::vector<NodeId> mp(nq, -1);
    const double ub0 = UB(mp, used);  // used is all-zero here
    pq.push({ub0, 0.0, 0, seq++, std::move(mp)});
  }

  double best_score = kNegInf;
  std::vector<NodeId> best_map;
  bool timed = false;

  while (!pq.empty()) {
    if (timed_out() || (opt.max_states && C.states >= opt.max_states)) {
      timed = true;  // incomplete: time or state budget exhausted
      break;
    }
    AStarNode cur = pq.top();
    pq.pop();

    if (cur.depth == nq) {
      best_score = cur.g;
      best_map = cur.mapping;
      break;  // first complete node popped is optimal
    }

    set_used(cur.mapping, 1);  // reconstruct used[] for this node
    const NodeId q = select_q(cur.mapping, used);
    const std::vector<NodeId> cands =
        feasible_candidates(G, Q, q, cur.mapping, used);

    for (NodeId u : cands) {
      const double gnew = cur.g + delta_gain(G, Q, q, u, cur.mapping, opt);
      std::vector<NodeId> mp = cur.mapping;
      mp[q] = u;
      used[u] = 1;  // temporarily mark for the child's bound
      const double ub = UB(mp, used);
      used[u] = 0;
      const double f = gnew + ub;

      if (opt.collect_metrics && best_score > kNegInf)
        gaps.add(std::max(0.0, f - best_score));
      if (f <= best_score) {
        C.pruned += 1;
        continue;
      }
      pq.push({f, gnew, cur.depth + 1, seq++, std::move(mp)});
      C.states += 1;
    }
    set_used(cur.mapping, 0);  // clear used[] before the next pop
  }

  C.timed_out = timed;
  C.best_loglik = best_score;
  if (best_score > kNegInf) {
    R.mapping = best_map;
    C.solutions = 1;
  }
  gaps.finalize(C);
  return R;
}

}  // namespace vf2prob
