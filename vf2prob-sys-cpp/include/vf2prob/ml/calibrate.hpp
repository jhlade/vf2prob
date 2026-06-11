// VF2-Prob, Jan Hladěna, FIM UHK
#pragma once
#include <cstdint>
#include <vector>

#include "vf2prob/compat.hpp"
#include "vf2prob/graph.hpp"
#include "vf2prob/ml/train.hpp"

namespace vf2prob::ml {

// Platt scaling: calibrated = sigmoid(a * logit(p) + b).
struct Platt {
  double a = 1.0;
  double b = 0.0;
};

// Isotonic calibration: a monotone non-decreasing step function fit by Pool
// Adjacent Violators. x are block upper-bound scores (ascending), v the
// (non-decreasing) calibrated values.
struct Isotonic {
  std::vector<double> x;
  std::vector<double> v;
};
Isotonic fit_isotonic(const std::vector<double>& pred, const std::vector<int>& y);
double apply_isotonic(const Isotonic& c, double p);

// Expected calibration error: weighted mean |confidence - accuracy| over bins.
double expected_calibration_error(const std::vector<double>& pred,
                                  const std::vector<int>& y, int bins = 10);

// Per-bin reliability data (for a reliability diagram): mean confidence vs
// empirical accuracy, for non-empty bins.
struct ReliabilityBin {
  double conf;
  double acc;
  int count;
};
std::vector<ReliabilityBin> reliability_bins(const std::vector<double>& pred,
                                             const std::vector<int>& y,
                                             int bins = 10);

// Fit Platt parameters on (predicted prob, label) by gradient descent.
Platt fit_platt(const std::vector<double>& pred, const std::vector<int>& y,
                int epochs = 600, double lr = 0.3);
double apply_platt(const Platt& c, double p);

// Held-out node correspondences scored under `w`: pred[i] = s_v(q-label,d-label),
// y[i] = 1 for the true correspondence, 0 for negatives. Same generation as
// training, with disjoint seeds. If `base` is non-null it is used as the clean
// source graph (real data) instead of a fresh synthetic graph per instance.
void node_correspondence_samples(const TrainConfig& cfg, const compat::Params& w,
                                 uint64_t base_seed, int instances,
                                 std::vector<double>& pred, std::vector<int>& y,
                                 const LabeledGraph* base = nullptr);

}  // namespace vf2prob::ml
