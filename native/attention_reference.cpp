

#include "ggml-backend.h"
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>

struct Profile {
  std::map<int, std::vector<float>> inputs;
  std::map<int, std::pair<double, uint64_t>> influence;
};
static bool profile_tensor(ggml_tensor *t, bool ask, void *data) {
  std::string name = t->name;
  bool input = name.rfind("poc_attn_in-", 0) == 0;
  bool output = name.rfind("ffn_inp-", 0) == 0;
  if (ask)
    return input || output;
  if (!input && !output)
    return true;
  if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t))
    throw std::runtime_error("Unsupported profile tensor");
  int layer = std::stoi(name.substr(name.rfind('-') + 1));
  auto &p = *static_cast<Profile *>(data);
  std::vector<float> values(ggml_nelements(t));
  ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
  if (input) {
    p.inputs[layer] = std::move(values);
    return true;
  }
  auto found = p.inputs.find(layer);
  if (found == p.inputs.end())
    throw std::runtime_error("Missing attention input");
  auto &x = found->second;
  size_t dim = t->ne[0], rows = values.size() / dim;
  if (x.size() < values.size())
    throw std::runtime_error("Profile shape mismatch");

  size_t offset = x.size() - values.size();
  for (size_t row = 0; row < rows; ++row) {
    double dot = 0, xx = 0, yy = 0;
    for (size_t j = 0; j < dim; ++j) {
      double a = x[offset + row * dim + j], b = values[row * dim + j];
      dot += a * b;
      xx += a * a;
      yy += b * b;
    }
    if (xx <= 0 || yy <= 0)
      throw std::runtime_error("Invalid profile norm");
    double score = 1 - std::clamp(dot / std::sqrt(xx * yy), -1.0, 1.0);
    p.influence[layer].first += score;
    p.influence[layer].second++;
  }
  p.inputs.erase(found);
  return true;
}
static double cpu_seconds() {
  rusage r{};
  getrusage(RUSAGE_SELF, &r);
  return r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 + r.ru_stime.tv_sec +
         r.ru_stime.tv_usec * 1e-6;
}
static std::set<int> mask_from_env() {
  std::set<int> mask;
  const char *raw = std::getenv("DLI_ATTN_SKIP");
  if (!raw || !*raw)
    return mask;
  std::string text(raw);
  if (text.back() == ',')
    throw std::runtime_error("Trailing mask delimiter");
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty() ||
        item.find_first_not_of("0123456789") != std::string::npos)
      throw std::runtime_error("Invalid mask syntax");
    int layer = std::stoi(item);
    if (!mask.insert(layer).second)
      throw std::runtime_error("Duplicate mask layer");
  }
  return mask;
}
int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 5)
      throw std::runtime_error(
          "model threads [microbatch] [all_logits] required");
    int threads = std::stoi(argv[2]);
    if (threads < 1 || threads > 4)
      throw std::runtime_error("threads 1..4");
    int microbatch = argc >= 4 ? std::stoi(argv[3]) : 1;
    if (microbatch < 1 || microbatch > 64)
      throw std::runtime_error("microbatch 1..64");
    bool all_logits = argc == 5 && std::string(argv[4]) == "all";
    const auto mask = mask_from_env();
    Profile profile;
    const bool profiling = std::getenv("DLI_ATTN_PROFILE") != nullptr;
    if (profiling && !mask.empty())
      throw std::runtime_error("Profile only with zero mask");
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model)
      throw std::runtime_error("load failed");
    for (int layer : mask)
      if (layer >= llama_model_n_layer(model))
        throw std::runtime_error("Mask outside layer range");
    if (llama_model_n_layer(model) != 48)
      throw std::runtime_error(
          "POC only supports the qualified 48-layer Qwen14B");

    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 64;
    cp.n_ubatch = microbatch;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (profiling) {
      cp.cb_eval = profile_tensor;
      cp.cb_eval_user_data = &profile;
    }
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx)
      throw std::runtime_error("context failed");
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    uint32_t expected = 0;
    while (true) {
      uint32_t header[2];
      std::cin.read(reinterpret_cast<char *>(header), 8);
      if (std::cin.eof() && std::cin.gcount() == 0)
        break;
      if (!std::cin)
        throw std::runtime_error("truncated header");
      uint32_t pos = header[0], bytes = header[1];
      if (!bytes || bytes % 8 || bytes > 64 * 8 || pos >= 2048 ||
          pos + bytes / 8 > 2048)
        throw std::runtime_error("invalid batch");
      if (pos == 0) {
        llama_memory_clear(llama_get_memory(ctx), true);
        expected = 0;
      }
      if (pos != expected)
        throw std::runtime_error("nonsequential position");
      std::vector<int64_t> ids(bytes / 8);
      std::cin.read(reinterpret_cast<char *>(ids.data()), bytes);
      if (!std::cin)
        throw std::runtime_error("truncated payload");
      for (auto id : ids)
        if (id < 0 || id >= vocab)
          throw std::runtime_error("token outside vocabulary");
      auto batch = llama_batch_init(ids.size(), 0, 1);
      batch.n_tokens = ids.size();
      for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = all_logits || (i + 1 == ids.size());
      }

      int phase_threads = threads;
      if (const char *raw = std::getenv("DLI_DECODE_THREADS")) {
        int decode_threads = std::stoi(raw);
        if (decode_threads < 1 || decode_threads > 4)
          throw std::runtime_error("decode threads 1..4");
        const char *boundary = std::getenv("DLI_PREFILL_TOKENS");
        if (!boundary)
          throw std::runtime_error("Missing prefill boundary");
        if (pos >= std::stoul(boundary))
          phase_threads = decode_threads;
      }
      llama_set_n_threads(ctx, phase_threads, phase_threads);
      auto step_start = std::chrono::steady_clock::now();
      double cpu0 = cpu_seconds();
      int status = llama_decode(ctx, batch);
      llama_batch_free(batch);
      double elapsed = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - step_start)
                           .count();
      std::cerr << std::setprecision(12) << "POC_STEP {\"position\":" << pos
                << ",\"tokens\":" << ids.size() << ",\"wall_s\":" << elapsed
                << ",\"cpu_s\":" << cpu_seconds() - cpu0 << "}\n";
      if (status)
        throw std::runtime_error("decode failed");
      uint32_t length = vocab * 4 * (all_logits ? ids.size() : 1);
      std::cout.write(reinterpret_cast<char *>(&length), 4);
      for (int i = all_logits ? 0 : ids.size() - 1; i < int(ids.size()); ++i) {
        auto *logits = llama_get_logits_ith(ctx, i);
        if (!logits)
          throw std::runtime_error("missing logits");
        std::cout.write(reinterpret_cast<char *>(logits), vocab * 4);
      }
      std::cout.flush();
      expected += ids.size();
    }
    for (auto &entry : profile.influence) {
      auto &metric = entry.second;
      std::cerr << std::setprecision(12)
                << "POC_INFLUENCE {\"layer\":" << entry.first
                << ",\"mean\":" << metric.first / metric.second
                << ",\"tokens\":" << metric.second << "}\n";
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
