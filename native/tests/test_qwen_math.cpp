#include "qwen_math.hpp"
#include <iostream>
static void require(bool ok) {
  if (!ok)
    throw std::runtime_error("test failed");
}
int main() {
  std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8};
  auto original = a;
  dli_research::neox_rope(a, 2, 4, 0, 10000);
  require(a == original);
  dli_research::neox_rope(a, 2, 4, 1, 10000);
  require(std::abs(a[0] - (std::cos(1.0f) - 3 * std::sin(1.0f))) < 1e-6);
  require(std::abs(a[1] - (2 * std::cos(.01f) - 4 * std::sin(.01f))) < 1e-6);
  for (int h = 0; h < 2; ++h) {
    float x = 0, y = 0;
    for (int i = 0; i < 4; ++i) {
      x += a[h * 4 + i] * a[h * 4 + i];
      y += original[h * 4 + i] * original[h * 4 + i];
    }
    require(std::abs(x - y) < 1e-4);
  }
  std::vector<float> b{1, 2};
  dli_research::add_bias(b, {3, 4});
  require(b == std::vector<float>({4, 6}));
  bool rejected = false;
  try {
    dli_research::add_bias(b, {1});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected);
  rejected = false;
  try {
    dli_research::neox_rope(b, 1, 3, 1, 10000);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected);
  std::cout << "PASS: NeoX layout, identity, norms, bias and shape rejection\n";
}
