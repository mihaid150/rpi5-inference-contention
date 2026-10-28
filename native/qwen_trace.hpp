#pragma once
#include <map>
#include <string>
#include <vector>
namespace dli_research {
inline thread_local bool trace_enabled = false;
inline thread_local std::map<std::string, std::vector<float>> trace_values;
inline void trace(const std::string &name, int layer,
                  const std::vector<float> &values) {
  if (trace_enabled)
    trace_values[name + "-" + std::to_string(layer)] = values;
}
}
