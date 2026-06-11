// VF2-Prob, Jan Hladěna, FIM UHK
// tests auto-generated
#include "test_framework.hpp"

int main() {
  int run = 0;
  for (auto& t : tf::registry()) {
    std::cerr << "[ RUN  ] " << t.name << "\n";
    t.fn();
    ++run;
  }
  if (tf::failures() == 0) {
    std::cerr << "[ PASS ] all " << run << " test cases\n";
    return 0;
  }
  std::cerr << "[ FAIL ] " << tf::failures() << " check(s) failed across " << run
            << " test cases\n";
  return 1;
}
