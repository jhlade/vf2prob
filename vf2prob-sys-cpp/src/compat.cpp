// VF2-Prob, Jan Hladěna, FIM UHK
#include "vf2prob/compat.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace vf2prob::compat {

const Params& params() {
  static const Params p{};
  return p;
}

static double sigmoid(double z) {
  if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
  const double e = std::exp(z);
  return e / (1.0 + e);
}

double node_compat(LabelId q_lab, LabelId u_lab, const Params& p) {
  if (p.n_node_labels > 0 && q_lab >= 0 && u_lab >= 0 &&
      q_lab < p.n_node_labels && u_lab < p.n_node_labels) {
    const size_t idx = static_cast<size_t>(q_lab) * p.n_node_labels + u_lab;
    if (idx < p.node_table.size())
      return std::max(p.eps, std::min(1.0, p.node_table[idx]));
  }
  const double lab = (q_lab == u_lab) ? 1.0 : 0.0;
  const double z = p.node_w0 + p.node_w_label * lab;
  return std::max(p.eps, std::min(1.0, sigmoid(z)));
}

double edge_compat(bool label_match, double p_exist, const Params& p) {
  const double lab = label_match ? 1.0 : 0.0;
  const double pe = std::min(1.0 - 1e-6, std::max(1e-6, p_exist));
  const double logit_p = std::log(pe / (1.0 - pe));
  const double z = p.edge_w0 + p.edge_w_label * lab + p.edge_w_prob * logit_p;
  return std::max(p.eps, std::min(1.0, sigmoid(z)));
}

double edge_compat(LabelId q_elabel, LabelId u_elabel, double p_exist,
                   const Params& p) {
  if (p.n_edge_labels > 0 && q_elabel >= 0 && u_elabel >= 0 &&
      q_elabel < p.n_edge_labels && u_elabel < p.n_edge_labels) {
    const size_t idx = static_cast<size_t>(q_elabel) * p.n_edge_labels + u_elabel;
    if (idx < p.edge_table.size()) {
      const double pe = std::min(1.0 - 1e-6, std::max(1e-6, p_exist));
      const double z = p.edge_table[idx] + p.edge_w_prob * std::log(pe / (1.0 - pe));
      return std::max(p.eps, std::min(1.0, sigmoid(z)));
    }
  }
  return edge_compat(q_elabel == u_elabel, p_exist, p);
}

double node_compat(LabelId q_lab, LabelId u_lab) {
  return node_compat(q_lab, u_lab, params());
}
double edge_compat(bool label_match, double p_exist) {
  return edge_compat(label_match, p_exist, params());
}

double safe_log(double x) { return std::log(std::max(1e-12, std::min(1.0, x))); }

std::string to_string(const Params& p) {
  std::ostringstream ss;
  ss.precision(10);
  ss << p.node_w0 << ' ' << p.node_w_label << ' ' << p.edge_w0 << ' '
     << p.edge_w_label << ' ' << p.edge_w_prob;
  if (p.n_node_labels > 0) {
    ss << "\nNODE_TABLE " << p.n_node_labels;
    for (double v : p.node_table) ss << ' ' << v;
  }
  if (p.n_edge_labels > 0) {
    ss << "\nEDGE_TABLE " << p.n_edge_labels;
    for (double v : p.edge_table) ss << ' ' << v;
  }
  return ss.str();
}

Params parse_params(const std::string& s) {
  Params p;  // defaults; overwritten by whatever parses
  std::istringstream ss(s);
  ss >> p.node_w0 >> p.node_w_label >> p.edge_w0 >> p.edge_w_label >>
      p.edge_w_prob;
  auto read_table = [&](int& n_out, std::vector<double>& tbl_out) {
    int n = 0;
    if (ss >> n && n > 0) {
      std::vector<double> tbl;
      double v;
      while (static_cast<int>(tbl.size()) < n * n && (ss >> v)) tbl.push_back(v);
      if (static_cast<int>(tbl.size()) == n * n) {
        n_out = n;
        tbl_out = std::move(tbl);
      }
    }
  };
  std::string tok;
  while (ss >> tok) {
    if (tok == "NODE_TABLE")
      read_table(p.n_node_labels, p.node_table);
    else if (tok == "EDGE_TABLE")
      read_table(p.n_edge_labels, p.edge_table);
  }
  return p;
}

}  // namespace vf2prob::compat
