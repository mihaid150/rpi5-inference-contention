FROM python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd AS build
ARG LLAMA_REVISION=838374375cb8c467c33c263be4801c19d1089763
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git init /llama && git -C /llama remote add origin https://github.com/ggml-org/llama.cpp.git && git -C /llama fetch --depth=1 origin ${LLAMA_REVISION} && git -C /llama checkout --detach FETCH_HEAD
RUN cmake -S /llama -B /build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod -DGGML_CPU_ALL_VARIANTS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_WEBUI=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF && cmake --build /build --target llama-server -j2
RUN mkdir /standard && cp /build/bin/llama-server /standard/ && cp -L /build/bin/*.so* /standard/
FROM python:3.11-slim-bookworm@sha256:94a8fee35252d06ab75856f15f12c7bae67b10e8a6da93c7fded82b2610926cd
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY --from=build /standard /standard
COPY scripts /app/scripts
COPY scripts/shared_budget_14b /poc-input
ENV PYTHONPATH=/app/scripts:/poc-input PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false XDG_CACHE_HOME=/tmp/cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
USER 1000:1000
ENTRYPOINT ["python", "/app/scripts/campaign.py"]
