#include "dli_stage/runtimes/llama_cpu_executor.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
int main() {
  dli_stage::LlamaExecutorHyperparams hp;
  hp.architecture = "qwen2";
  hp.n_head = 12;
  hp.n_head_kv = 2;
  hp.head_dim = 128;
  hp.hidden_size = 1536;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1, 1);
  for (int seq : {1, 2, 8, 31, 32, 255, 256, 257}) {
    int kv = hp.n_head_kv * hp.head_dim, cap = seq + 7;
    std::vector<float> q(hp.hidden_size), k(seq * kv), v(seq * kv),
        vt(cap * kv, 987.0f);
    for (auto &x : q)
      x = dist(rng);
    for (auto &x : k)
      x = dist(rng);
    for (auto &x : v)
      x = dist(rng);
    for (int p = 0; p < seq; ++p)
      for (int d = 0; d < kv; ++d)
        vt[d * cap + p] = v[p * kv + d];
    auto a =
        dli_stage::LlamaCpuExecutor::compute_attention_scalar(hp, q, k, v, seq);
    auto b = dli_stage::LlamaCpuExecutor::compute_attention_ggml_views(
        hp, q, k.data(), vt.data(), cap, seq);
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      if (!std::isfinite(b[i]))
        return 1;
      worst = std::max(worst, double(std::abs(a[i] - b[i])));
    }
    std::cout << "seq=" << seq << " max_abs=" << worst << '\n';
    if (worst > 2e-3)
      return 1;
  }
}
