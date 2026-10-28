# Preparing the measured model

The measured model is Qwen2.5-14B-Instruct quantized from the original
high-precision weights with an importance matrix. Token embeddings and the
output tensor use Q4_K. The GGUF SHA-256 is:

```text
8123a526b3c9b9eeda6cb1dd3af78bd1de0da9af2cef8259f52c3e0a8cd3cebd
```

Its size is 8,787,386,464 bytes. Model identity is checked before timing.
Do not substitute another publisher's Q4_K_M file and describe it as the
same artifact. The original imatrix and F16 hashes are in `assets/versions.json`.

If the original GGUF is available, no conversion or quantization is needed.
Otherwise prepare it on an ARM64 machine with sufficient disk and memory.
The following steps can take many hours; they are not part of E1–E3 timing.

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/hf download Qwen/Qwen2.5-14B-Instruct --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 --local-dir workspace/qwen-source --include '*.safetensors' --include '*.json' --include merges.txt --include vocab.json --include LICENSE
git clone https://github.com/ggml-org/llama.cpp.git workspace/llama.cpp
git -C workspace/llama.cpp checkout 838374375cb8c467c33c263be4801c19d1089763
.venv/bin/pip install -r workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
cmake -S workspace/llama.cpp -B workspace/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod -DGGML_CPU_ALL_VARIANTS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
cmake --build workspace/llama.cpp/build --target llama-imatrix llama-quantize -j2
.venv/bin/python scripts/prepare_model.py --source workspace/qwen-source --llama workspace/llama.cpp --output workspace/quantized
```

Install the host build prerequisites `build-essential`, `cmake`, `git` and
`pkg-config` before these commands. The script converts to F16, computes the
importance matrix from `assets/calibration.txt`, then produces Q4_K_M with
one quantization thread. The calibration text is frozen from the original
19 CoQA training conversations and is separate from the six E2/E3 questions.
Its original whitespace is retained because it is part of the calibrated byte
sequence; `.gitattributes` prevents line-ending conversion for this fixture.

Conversion and quantization can differ across toolchains and platforms.
The release has not yet rebuilt the entire model from scratch to establish
byte-for-byte reconstruction. The script retains conversion logs and all
intermediate hashes and fails if the result differs from the measured GGUF.
Do not bypass this check. A mismatching artifact requires an explicitly
reported replication with a different model, not a claim of exact reproduction.
