// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <string>
#include <vector>

#include "vf2prob/graph.hpp"
#include "vf2prob/types.hpp"

// Calibrated logistic node/edge compatibilities (the probabilistic core):
//   s_v = sigma(w0 + w_l * 1[label match])                       in (0,1]
//   s_e = sigma(e0 + e_l * 1[edge label match] + e_p * logit(p)) in (0,1]
// A small floor eps keeps log-scores finite; the objective is additive in log
// space: l(f) = sum log s_v + sum log s_e (each term <= 0).
namespace vf2prob::compat {

// Node attribute model: categorical label compatibility (default) or an RBF
// kernel on continuous coordinates (e.g. geometric pattern-recognition graphs).
enum class NodeKernel { Categorical, RBF };

// Fixed, calibrated weights (kept public so tests can assert numeric parity).
struct Params {
  double node_w0 = -3.0;     // mismatch prior   -> s_v ~ 0.047
  double node_w_label = 6.0; // match            -> s_v ~ 0.953
  double edge_w0 = -1.0;
  double edge_w_label = 3.0;
  double edge_w_prob = 2.0;
  double eps = 1e-3;
  // Optional per-label node compatibility. If n_node_labels > 0, s_v for a
  // (query-label, data-label) pair is read from node_table[q*n + u] (values in
  // (0,1]); otherwise the scalar (node_w0, node_w_label) model is used. Captures
  // structured label confusion that a single match/mismatch bonus cannot.
  int n_node_labels = 0;
  std::vector<double> node_table;
  // Optional per-label edge compatibility: a logit-space bias per (query-edge-
  // label, data-edge-label) pair, combined with the shared e_prob*logit(p) term.
  int n_edge_labels = 0;
  std::vector<double> edge_table;
  // Continuous node attributes: when node_kernel==RBF and both graphs carry
  // coordinates, s_v = exp(-||x_q - x_u||^2 / (2 h^2)), floored at eps so the
  // bounds stay admissible exactly as in the categorical case. Default
  // Categorical reproduces the labels-only behaviour unchanged.
  NodeKernel node_kernel = NodeKernel::Categorical;
  double node_kernel_h = 1.0;  // RBF bandwidth, in coordinate units
};
// Default calibrated parameters.
const Params& params();

// Scored with explicit weights. The result is always clamped to (0,1], so the
// bounds stay admissible for any Params.
double node_compat(LabelId q_lab, LabelId u_lab, const Params& p);
double edge_compat(bool label_match, double p_exist, const Params& p);
// Continuous (geometric) node compatibility: an RBF on the L2 distance between
// coordinate vectors, clamped to (eps,1] so the bounds stay admissible.
double node_compat_coords(const double* xq, const double* xu, int dim,
                          const Params& p);
// Single node-score entry point used by the objective and both bounds: the RBF
// kernel on coordinates when selected and available, else the categorical label
// compatibility. Exactness is invariant to the branch taken (both land in (0,1]).
double node_score(const LabeledGraph& Q, NodeId q, const LabeledGraph& G,
                  NodeId u, const Params& p);
// Edge compatibility from label ids: uses the per-label edge table if present,
// otherwise the scalar match/mismatch model.
double edge_compat(LabelId q_elabel, LabelId u_elabel, double p_exist,
                   const Params& p);

// Convenience overloads using the default calibrated parameters.
double node_compat(LabelId q_lab, LabelId u_lab);   // s_v
double edge_compat(bool label_match, double p_exist);  // s_e
double safe_log(double x);  // log(clamp(x, 1e-12, 1.0))

// Serialize/parse the five learnable weights (space/newline separated:
// node_w0 node_w_label edge_w0 edge_w_label edge_w_prob).
std::string to_string(const Params& p);
Params parse_params(const std::string& s);

}  // namespace vf2prob::compat
