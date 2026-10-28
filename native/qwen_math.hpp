#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

namespace dli_research {
inline void neox_rope(std::vector<float> &values, int heads, int dim,
                      int position, float theta) {
  if (heads <= 0 || dim <= 0 || dim % 2 || position < 0 || !(theta > 0) ||
      !std::isfinite(theta) ||
      values.size() != static_cast<size_t>(heads * dim))
    throw std::invalid_argument("invalid NeoX RoPE shape/parameters");
  for (int h = 0; h < heads; ++h) {
    for (int i = 0; i < dim / 2; ++i) {
      const float angle = position * std::pow(theta, -2.0f * i / dim);
      const float c = std::cos(angle), s = std::sin(angle);
      const size_t a = h * dim + i, b = a + dim / 2;
      const float x = values[a], y = values[b];
      values[a] = x * c - y * s;
      values[b] = x * s + y * c;
    }
  }
}
inline void add_bias(std::vector<float> &values,
                     const std::vector<float> &bias) {
  if (values.size() != bias.size())
    throw std::invalid_argument("bias shape mismatch");
  for (size_t i = 0; i < values.size(); ++i)
    values[i] += bias[i];
}
}
