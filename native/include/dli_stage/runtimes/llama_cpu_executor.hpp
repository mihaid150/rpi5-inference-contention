#pragma once

#include "dli/common/tensor.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace dli_stage {

struct LlamaExecutorHyperparams {
  std::string architecture = "llama";

  int hidden_size = 0;
  int ffn_size = 0;

  int n_head = 0;
  int n_head_kv = 0;
  int head_dim = 0;

  int vocab_size = 0;

  float rms_eps = 1.0e-5f;
  float rope_theta = 10000.0f;
};

struct LlamaSamplingParams {

  std::vector<float> *capture_logits = nullptr;
  bool enabled = false;
  float temperature = 1.0f;
  int top_k = 0;
  float top_p = 1.0f;

  bool has_seed = false;
  long long seed = 0;
};

struct LlamaExecutorResult {
  dli::common::TensorBuffer hidden_states;
  int next_token_id = -1;
};

struct LayerKvCache {
  int seq_len = 0;
  std::vector<float> k;
  std::vector<float> v;

  std::vector<float> v_t;
  int v_t_capacity = 0;
};

class LlamaCpuExecutor {
public:
  LlamaCpuExecutor(ggml_context *tensor_context,
                   LlamaExecutorHyperparams hyperparams);

  dli::common::TensorBuffer
  execute_source(const dli::common::TensorBuffer &token_ids,
                 const std::vector<int> &layers,
                 const std::string &generation_mode, int kv_seq_before,
                 const std::string &output_activation_precision);

  dli::common::TensorBuffer
  execute_intermediate(const dli::common::TensorBuffer &hidden_states,
                       const std::vector<int> &layers,
                       const std::string &generation_mode, int kv_seq_before,
                       const std::string &output_activation_precision);

  int execute_terminal(const dli::common::TensorBuffer &hidden_states,
                       const std::vector<int> &layers,
                       const std::string &generation_mode, int kv_seq_before,
                       bool owns_norm, bool owns_lm_head,
                       const LlamaSamplingParams &sampling);

  std::uint64_t kv_cache_bytes() const;

  std::uint64_t kv_allocated_bytes() const;
  std::uint64_t non_kv_persistent_cache_bytes() const;

  struct KvExport {
    std::string metadata_json;
    std::vector<std::uint8_t> bytes;
  };

  KvExport export_kv(const std::vector<int> &layers) const;
  void import_kv(const std::string &metadata_json,
                 const std::vector<std::uint8_t> &bytes);

  void reset_profile();
  double profile_matmul_ms() const;
  double profile_attention_ms() const;

  double profile_elementwise_ms() const;
  double profile_overhead_ms() const;
  double profile_execute_ms() const;

  static int configured_threads();

  static std::vector<float>
  compute_attention_scalar(const LlamaExecutorHyperparams &hp,
                           const std::vector<float> &q,
                           const std::vector<float> &k_cache,
                           const std::vector<float> &v_cache, int required_seq);
  static std::vector<float>
  compute_attention_ggml(const LlamaExecutorHyperparams &hp,
                         const std::vector<float> &q,
                         const std::vector<float> &k_cache,
                         const std::vector<float> &v_cache, int required_seq);

  static std::vector<float>
  compute_attention_ggml_views(const LlamaExecutorHyperparams &hp,
                               const std::vector<float> &q, const float *k_src,
                               const float *v_t_src, int v_capacity,
                               int required_seq);

private:
  ggml_context *tensor_context_ = nullptr;
  LlamaExecutorHyperparams hp_;

  std::unordered_map<int, LayerKvCache> layer_cache_;

  std::mt19937 rng_;

  mutable double matmul_ms_ = 0.0;
  mutable double attention_ms_ = 0.0;
  mutable double elementwise_ms_ = 0.0;
  mutable double execute_ms_ = 0.0;

  mutable std::vector<float> rope_inv_freq_;

  mutable std::vector<float> rope_cos_;
  mutable std::vector<float> rope_sin_;
  mutable int rope_cache_position_ = -1;

  mutable std::unordered_map<std::string, std::vector<float>>
      vector_tensor_cache_;

  ggml_tensor *require_tensor(const std::string &name) const;

  std::vector<float> tensor_row_to_float(const ggml_tensor *tensor,
                                         int64_t row) const;

  const std::vector<float> &tensor_vector_to_float(const std::string &name,
                                                   int64_t expected_size) const;

  std::vector<float> matvec(const std::string &tensor_name,
                            const std::vector<float> &x) const;

  std::vector<float> rms_norm(const std::vector<float> &x,
                              const std::vector<float> &weight) const;

  void apply_rope(std::vector<float> &q, std::vector<float> &k,
                  int position) const;

  std::vector<float> attention(int layer_id, const std::vector<float> &q,
                               const std::vector<float> &k,
                               const std::vector<float> &v, int position);

  std::vector<float> run_layer(int layer_id, const std::vector<float> &input,
                               int seq_len, int kv_seq_before);

  dli::common::TensorBuffer
  float_hidden_to_tensor(const std::vector<float> &hidden, int batch,
                         int seq_len,
                         const std::string &activation_precision) const;

  std::vector<float>
  tensor_to_hidden_f32(const dli::common::TensorBuffer &tensor) const;

  std::vector<std::int64_t>
  tensor_to_token_ids(const dli::common::TensorBuffer &tensor) const;

  int sample_lm_head(const std::vector<float> &last_hidden,
                     const LlamaSamplingParams &sampling);

  int sample_from_logits(std::vector<float> logits,
                         const LlamaSamplingParams &sampling);
};

}
