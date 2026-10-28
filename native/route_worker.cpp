
#include "dli_stage/runtimes/llama_cpu_executor.hpp"
#include "llama.h"
#include "mapped_shard.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#ifdef DLI_RELEASE_WEIGHTS
#include <malloc.h>
#endif

static std::string memory_snapshot(size_t mapped, int target) {
  auto read = [](const char *path) {
    std::ifstream f(path);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
  };
  auto quote = [](const std::string &s) {
    std::string r = "\"";
    for (char c : s) {
      if (c == '\n')
        r += "\\n";
      else if (c == '\"' || c == '\\') {
        r += '\\';
        r += c;
      } else if (c == '\t')
        r += "\\t";
      else
        r += c;
    }
    return r + "\"";
  };
  std::ostringstream out;
  out << "{\"pid\":" << getpid() << ",\"mapped_weight_bytes\":" << mapped
      << ",\"target\":" << target;
  for (auto field : {std::pair<const char *, const char *>{
                         "smaps_rollup", "/proc/self/smaps_rollup"},
                     {"process_status", "/proc/self/status"},
                     {"maps", "/proc/self/maps"},
                     {"meminfo", "/proc/meminfo"},
                     {"memory_pressure", "/proc/pressure/memory"},
                     {"cgroup_current", "/sys/fs/cgroup/memory.current"},
                     {"cgroup_peak", "/sys/fs/cgroup/memory.peak"}})
    out << ",\"" << field.first << "\":" << quote(read(field.second));
  out << "}";
  return out.str();
}

int main(int argc, char **argv) {
  try {
    if (argc != 3 && argc != 4)
      throw std::runtime_error(
          "path stage(0..3 or ref) [alternate-shard] required");
    llama_backend_init();
    bool reference = std::string(argv[2]) == "ref";
    int stage = reference ? -1 : std::stoi(argv[2]);
    if (!reference && (stage < 0 || stage > 3))
      throw std::runtime_error("invalid stage");
    std::unique_ptr<MappedShard> shard;
    std::unique_ptr<MappedShard> alternate;
    bool use_alternate = false;
    std::unique_ptr<dli_stage::LlamaCpuExecutor> exec;
    dli_stage::LlamaExecutorHyperparams hp;
    std::vector<int> layers;
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    if (reference) {
      auto mp = llama_model_default_params();
      mp.n_gpu_layers = 0;
      mp.use_extra_bufts = false;
      model = llama_model_load_from_file(argv[1], mp);
      if (!model)
        throw std::runtime_error("reference model");
      auto cp = llama_context_default_params();
      cp.n_ctx = 2048;
      cp.n_batch = 1;
      cp.n_ubatch = 1;
      cp.n_threads = 1;
      cp.n_threads_batch = 1;
      cp.type_k = GGML_TYPE_F32;
      cp.type_v = GGML_TYPE_F32;
      cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
      ctx = llama_init_from_model(model, cp);
      if (!ctx)
        throw std::runtime_error("reference context");
      hp.vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
    } else {
      shard = std::make_unique<MappedShard>(argv[1]);
      auto *g = shard->metadata;
      auto integer = [&](const char *k) {
        int i = gguf_find_key(g, k);
        if (i < 0)
          throw std::runtime_error(k);
        return int(gguf_get_val_u32(g, i));
      };
      auto real = [&](const char *k) {
        int i = gguf_find_key(g, k);
        if (i < 0)
          throw std::runtime_error(k);
        return gguf_get_val_f32(g, i);
      };
      hp.architecture = "qwen2";
      hp.hidden_size = integer("qwen2.embedding_length");
      hp.ffn_size = integer("qwen2.feed_forward_length");
      hp.n_head = integer("qwen2.attention.head_count");
      hp.n_head_kv = integer("qwen2.attention.head_count_kv");
      hp.head_dim = hp.hidden_size / hp.n_head;
      hp.rope_theta = real("qwen2.rope.freq_base");
      hp.rms_eps = real("qwen2.attention.layer_norm_rms_epsilon");
      hp.vocab_size =
          gguf_get_arr_n(g, gguf_find_key(g, "tokenizer.ggml.tokens"));
      int total = integer("qwen2.block_count");
      if (total % 4)
        throw std::runtime_error("partition mismatch");
      for (int i = stage * total / 4; i < (stage + 1) * total / 4; ++i)
        layers.push_back(i);
#ifndef DLI_RELEASE_WEIGHTS
      if (argc == 4)
        alternate = std::make_unique<MappedShard>(argv[3]);
#endif
    }
    uint32_t expected_position = 0;
    while (true) {
      uint32_t header[2];
      std::cin.read(reinterpret_cast<char *>(header), sizeof(header));
      if (std::cin.eof() && std::cin.gcount() == 0)
        break;
      if (!std::cin)
        throw std::runtime_error("truncated header");
      uint32_t position = header[0], bytes = header[1];
      if (position == UINT32_MAX - 1) {
        if (reference || bytes != 0)
          throw std::runtime_error("invalid telemetry command");
        auto report =
            memory_snapshot(shard->size + (alternate ? alternate->size : 0),
                            use_alternate ? 1 : 0);
        uint32_t length = report.size();
        std::cout.write(reinterpret_cast<char *>(&length), 4);
        std::cout.write(report.data(), length);
        std::cout.flush();
        continue;
      }

      if (position == UINT32_MAX) {
        if (reference || argc != 4 || bytes > 1 || !exec)
          throw std::runtime_error("invalid transition command");
        auto before = exec->export_kv(layers);
#ifdef DLI_RELEASE_WEIGHTS

        exec.reset();
        shard.reset();
        malloc_trim(0);
        shard = std::make_unique<MappedShard>(bytes ? argv[3] : argv[1]);
        auto replacement =
            std::make_unique<dli_stage::LlamaCpuExecutor>(shard->tensors, hp);
#else
        auto replacement = std::make_unique<dli_stage::LlamaCpuExecutor>(
            bytes ? alternate->tensors : shard->tensors, hp);
#endif
        replacement->import_kv(before.metadata_json, before.bytes);
        auto after = replacement->export_kv(layers);
        if (before.metadata_json != after.metadata_json ||
            before.bytes != after.bytes)
          throw std::runtime_error("KV roundtrip mismatch");
        exec = std::move(replacement);
        use_alternate = bytes != 0;
        uint32_t length = 8;
        uint64_t count = after.bytes.size();
        std::cout.write(reinterpret_cast<char *>(&length), 4);
        std::cout.write(reinterpret_cast<char *>(&count), 8);
        std::cout.flush();
        continue;
      }
      if (position >= 2048 ||
          bytes != uint32_t(reference || stage == 0 ? 8 : hp.hidden_size * 4))
        throw std::runtime_error("invalid request");
      if (position == 0) {
        expected_position = 0;
        if (reference)
          llama_memory_clear(llama_get_memory(ctx), true);
        else {
#ifdef DLI_RELEASE_WEIGHTS
          exec =
              std::make_unique<dli_stage::LlamaCpuExecutor>(shard->tensors, hp);
#else
          exec = std::make_unique<dli_stage::LlamaCpuExecutor>(
              use_alternate ? alternate->tensors : shard->tensors, hp);
#endif
        }
      }
      if (position != expected_position || (!reference && !exec))
        throw std::runtime_error("nonsequential KV position");
      dli::common::TensorBuffer input;
      input.bytes.resize(bytes);
      std::cin.read(reinterpret_cast<char *>(input.bytes.data()), bytes);
      if (!std::cin)
        throw std::runtime_error("truncated payload");
      std::vector<float> result;
      if (reference) {
        int64_t token;
        std::memcpy(&token, input.bytes.data(), 8);
        if (token < 0 || token >= hp.vocab_size)
          throw std::runtime_error("token range");
        llama_token id = token;
        auto batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, batch))
          throw std::runtime_error("reference decode");
        auto *logits = llama_get_logits_ith(ctx, -1);
        if (!logits)
          throw std::runtime_error("reference logits");
        result.assign(logits, logits + hp.vocab_size);
      } else {
        input.metadata.dtype = stage == 0 ? "int64" : "float32";
        input.metadata.shape = stage == 0
                                   ? std::vector<int64_t>{1, 1}
                                   : std::vector<int64_t>{1, 1, hp.hidden_size};
        if (stage == 3) {
          dli_stage::LlamaSamplingParams sp;
          sp.capture_logits = &result;
          exec->execute_terminal(input, layers, "decode", position, true, true,
                                 sp);
        } else {
          auto output =
              stage == 0 ? exec->execute_source(input, layers, "decode",
                                                position, "float32")
                         : exec->execute_intermediate(input, layers, "decode",
                                                      position, "float32");
          result.resize(output.bytes.size() / 4);
          std::memcpy(result.data(), output.bytes.data(), output.bytes.size());
        }
      }
      uint32_t length = result.size() * 4;
      std::cout.write(reinterpret_cast<char *>(&length), 4);
      std::cout.write(reinterpret_cast<char *>(result.data()), length);
      std::cout.flush();
      ++expected_position;
    }
    if (ctx)
      llama_free(ctx);
    if (model)
      llama_model_free(model);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
