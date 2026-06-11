// VF2-Prob, Jan Hladěna, FIM UHK
// tests auto-generated
#pragma once
// Minimal self-contained test framework (no external dependency, so the build
// works offline). TEST(name){...} registers a case; CHECK*/ macros record
// failures; test_main.cpp runs everything and returns non-zero on any failure.
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace tf {
struct Test {
  std::string name;
  std::function<void()> fn;
};
inline std::vector<Test>& registry() {
  static std::vector<Test> r;
  return r;
}
inline int& failures() {
  static int f = 0;
  return f;
}
struct Reg {
  Reg(const std::string& n, std::function<void()> fn) {
    registry().push_back({n, std::move(fn)});
  }
};
inline void fail(const std::string& msg) {
  ++failures();
  std::cerr << "    FAIL: " << msg << "\n";
}
}  // namespace tf

#define TF_CONCAT_(a, b) a##b
#define TF_CONCAT(a, b) TF_CONCAT_(a, b)
#define TEST(name)                                                       \
  static void TF_CONCAT(tf_test_, __LINE__)();                           \
  static tf::Reg TF_CONCAT(tf_reg_, __LINE__)(name,                      \
                                              TF_CONCAT(tf_test_, __LINE__)); \
  static void TF_CONCAT(tf_test_, __LINE__)()

#define CHECK(cond)                                                         \
  do {                                                                      \
    if (!(cond))                                                            \
      tf::fail(std::string(__FILE__) + ":" + std::to_string(__LINE__) +     \
               " CHECK(" #cond ")");                                        \
  } while (0)

#define CHECK_MSG(cond, msg)                                            \
  do {                                                                  \
    if (!(cond))                                                        \
      tf::fail(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
               " " + (msg));                                            \
  } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
  do {                                                                         \
    const double tf_a = (a), tf_b = (b);                                       \
    if (std::fabs(tf_a - tf_b) > (eps))                                        \
      tf::fail(std::string(__FILE__) + ":" + std::to_string(__LINE__) +        \
               " CHECK_NEAR " + std::to_string(tf_a) + " vs " +               \
               std::to_string(tf_b));                                          \
  } while (0)
