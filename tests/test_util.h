// Minimal assertion helpers so the C++ tests need no test framework.
#pragma once

#include <cmath>
#include <iostream>
#include <string>

namespace test {

inline int &failures() {
  static int count = 0;
  return count;
}

inline void Fail(const char *file, int line, const std::string &message) {
  ++failures();
  std::cerr << file << ":" << line << ": FAILED: " << message << "\n";
}

inline int Finish(const char *name) {
  if (failures() == 0) {
    std::cout << name << ": all checks passed\n";
    return 0;
  }
  std::cerr << name << ": " << failures() << " check(s) failed\n";
  return 1;
}

}  // namespace test

#define EXPECT_TRUE(condition)                                     \
  do {                                                             \
    if (!(condition)) test::Fail(__FILE__, __LINE__, #condition);  \
  } while (0)

#define EXPECT_NEAR(actual, expected, tolerance)                              \
  do {                                                                        \
    const double test_actual_ = (actual), test_expected_ = (expected);        \
    if (!(std::abs(test_actual_ - test_expected_) <= (tolerance))) {          \
      test::Fail(__FILE__, __LINE__,                                          \
                 std::string(#actual) + " = " + std::to_string(test_actual_) + \
                     ", expected " + std::to_string(test_expected_) +         \
                     " +/- " + std::to_string(tolerance));                    \
    }                                                                         \
  } while (0)
