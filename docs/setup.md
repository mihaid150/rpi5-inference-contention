# Configuration and infrastructure

## Host

Use a Raspberry Pi 5 with 16 GB RAM, a 64-bit Linux OS, cgroup v2, four enabled
CPU cores numbered 0–3, NVMe storage, stable power and active cooling. Preserve
CPU governor settings across paired experiments. Record throttling and temperature
if available. Do not run unrelated inference, indexing or load generators.
The package does not change the host governor or disable services automatically.

Install Git, Python 3.11 with venv support, and Docker Engine using the host
OS installation instructions. The user running the launcher needs Docker access.
GPU hardware and CUDA are not used. Allow at least 20 GB free for prepared
inference assets, image layers and results. Rebuilding the model from original
weights needs substantially more storage, at least 70 GB free in addition to
the downloaded source. Preparation and timed execution are separate steps.

The host requires internet access while preparing assets and building the image.
Timed Docker execution has networking disabled and uses only local model files.
The HTTP backend binds to loopback inside its container; no host port is published.

## Assets

`prepare_assets.py` downloads the pinned MiniLM model and Qwen tokenizer,
downloads and checks the official SQuAD evaluator, copies fixed inputs, and
creates `assets-manifest.json`. It refuses to reuse an existing destination.
It never downloads the 14B weights implicitly. Supply the verified GGUF via
`--gguf` after obtaining it or rebuilding it.

The resulting layout is:

```text
workspace/assets/
  assets-manifest.json
  campaign.json
  corpus.json
  prior.json
  versions.json
  official_evaluator.py
  minilm/
  q4-tokenizer/
  models/family/Q4_K_M.gguf
```

Keep this directory outside the tracked repository tree or under the ignored
`workspace/` directory. The input and model mounts are read-only. The frozen corpus preserves historical
private-network addresses as document text for scientific consistency. They are
not used as deployment targets or contacted by the runner. Each attempt
gets its own writable result directory. No SSH key, Hugging Face token or kubeconfig
belongs in the image or this repository. Public models need no bundled token.

## Runtime

The image builds the unmodified llama.cpp server at
`838374375cb8c467c33c263be4801c19d1089763`, with ARMv8.2 FP16 and dot-product
support. It uses Python 3.11, the CPU Torch 2.11.0 wheel with a pinned hash,
Transformers 5.7.0 and NumPy 2.2.6, matching the original direct dependencies.

The selected server configuration is context 2048, one inference slot, logical
batch 512, physical microbatch 512, F16 KV, automatic flash attention, CPU
repacking and zero GPU layers. Requests are greedy, with a 64-token cap and
prompt reuse disabled. Worker checks reject prompt truncation or reused prompts.
MiniLM uses masked mean pooling, L2 normalization, up to 256-token chunks,
batch size four and one inter-op thread. Intra-op threads depend on the policy.

A 12 GiB limit is a safety cap, not a measured minimum deployment requirement.
The runner requires a four-core quota and checks the allowed CPU set. Workers
record cgroup values and verify thread affinity throughout execution.

## Kubernetes alternative

Use an existing ARM64 Kubernetes or K3s node. The package does not install or
modify a cluster. The target node must have at least four allocatable CPU cores
and 12 GiB allocatable memory. Place the prepared assets and result parent
directory on that node, and run the launcher there with a working kubeconfig.
The generated Job uses node placement and hostPath volumes. Paths are paths on
the selected node, not on a remote laptop.

Build the image there and make it available to the cluster runtime, either via
an accessible registry or, for a single-node K3s installation:

```bash
docker save shared-budget-evaluation:local -o workspace/evaluation-image.tar
sudo k3s ctr images import workspace/evaluation-image.tar
```

Use the same image name in the launcher. A private registry may require a
cluster-managed pull secret; no secret is included in this package.
The Job runs without a service-account token, with a read-only root filesystem
and dropped capabilities. It does not need cluster privileges inside the pod.
