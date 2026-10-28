

#include "llama.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

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
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model)
      throw std::runtime_error("load failed");

    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 64;
    cp.n_ubatch = microbatch;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
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
      int status = llama_decode(ctx, batch);
      llama_batch_free(batch);
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
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
