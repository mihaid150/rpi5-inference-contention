#pragma once
#include "ggml.h"
#include "gguf.h"
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
struct MappedShard {
  ggml_context *tensors = nullptr;
  gguf_context *metadata = nullptr;
  void *mapping = MAP_FAILED;
  size_t size = 0;
  explicit MappedShard(const std::string &path) {
    gguf_init_params p{};
    p.no_alloc = true;
    p.ctx = &tensors;
    metadata = gguf_init_from_file(path.c_str(), p);
    if (!metadata || !tensors)
      throw std::runtime_error("shard metadata");
    int fd = open(path.c_str(), O_RDONLY);
    struct stat st{};
    if (fd < 0)
      throw std::runtime_error("shard open");
    if (fstat(fd, &st)) {
      close(fd);
      throw std::runtime_error("shard stat");
    }
    size = st.st_size;
    mapping = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED)
      throw std::runtime_error("shard mmap");
    for (int64_t i = 0; i < gguf_get_n_tensors(metadata); ++i) {
      auto *t = ggml_get_tensor(tensors, gguf_get_tensor_name(metadata, i));
      size_t offset =
          gguf_get_data_offset(metadata) + gguf_get_tensor_offset(metadata, i);
      if (!t || offset > size || ggml_nbytes(t) > size - offset)
        throw std::runtime_error("shard bounds");
      t->data = static_cast<char *>(mapping) + offset;
    }
  }
  ~MappedShard() {
    if (tensors)
      ggml_free(tensors);
    if (metadata)
      gguf_free(metadata);
    if (mapping != MAP_FAILED)
      munmap(mapping, size);
  }
  MappedShard(const MappedShard &) = delete;
  MappedShard &operator=(const MappedShard &) = delete;
};
