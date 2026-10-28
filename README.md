# Shared CPU inference and document updates

Reproduction package for the E1–E3 evaluation of Qwen2.5-14B-Instruct and
MiniLM on a Raspberry Pi 5. The framework measures how document publication,
CPU allocation and inference queuing affect answers to independently arriving
questions. It implements fixed allocation policies, not a dynamic scheduler.

The measured configuration uses a Raspberry Pi 5 with 16 GB RAM, NVMe storage,
four Cortex-A76 cores and a 2.4 GHz maximum clock. Inference is CPU-only.
The experiment container has a four-core CPU quota and a 12 GiB memory cap.
Physical memory and the container cap are different quantities.

## What is included

- The tested E1, E2 and E3 workers, index publisher, retrieval code and checks.
- The frozen 75-cell schedule and the minimal fixtures used by these stages.
- Model, tokenizer and encoder identities, calibration input and SHA-256 checks.
- A portable sequential runner, Docker image definition and Kubernetes deployment generator.
- Unit tests and instructions for building, running and interpreting outputs.

This branch has an independent root commit. It contains no earlier repository
history, credentials, personal deployment configuration, model weights, binaries, raw experimental
runs, paper build files, IDE configuration or local backups.

The earlier partitioned four-node pilot, private prefill/attention prototype and
24-question backend-selection study are outside this package. E1–E3 use the
complete 14B model on one node and the already selected standard llama.cpp
configuration. This package does not rerun that earlier selection process.

## Quick start

Read [configuration and infrastructure](docs/setup.md) first. Run the following
on the ARM64 Raspberry Pi, as a non-root user who can access Docker. Keep the
node free of other application workloads.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-prepare.txt
.venv/bin/python scripts/prepare_assets.py --output workspace/assets --gguf /path/to/Q4_K_M.gguf
docker build --platform linux/arm64 -f deploy/Dockerfile -t shared-budget-evaluation:local .
python3 scripts/launch.py --assets workspace/assets --output results/attempt-01
```

The supplied GGUF must match the recorded hash. An arbitrary Q4_K_M download
will be rejected. If you do not have the measured model, use the
[model preparation instructions](docs/model.md), then pass its verified GGUF
path to `prepare_assets.py`. Weights are not bundled with Git.

The runner executes E1, then E2, derives and freezes predictions from E2, then
runs E3. It uses a new output directory and stops at the first failure.
It never retries failed cells silently. A complete campaign can take hours.
Use a new attempt directory after a failure; preserve the failed attempt.

To monitor progress and completion:

```bash
watch -n 10 'cat results/attempt-01/status.json; tail -n 5 results/attempt-01/e*/status.json 2>/dev/null'
```

A successful campaign creates `results/attempt-01/DONE` and `summary.json`.
A stopped or failed container does not count as a successful campaign.
`status.json` identifies failures; stage `worker.log` and backend logs retain details.
Docker removes the container after exit; bind-mounted results remain.

For Kubernetes deployment, collection, stop commands and interpretation see
[execution](docs/experiments.md). No experiment starts during image building,
asset preparation, tests or deployment dry runs.

## Tests

Python 3.11 is required. The local tests use the standard library and do not
load a model, install Torch, contact a cluster or start timed experiments.

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
```

## Reproducibility and validation

The original workers ran on ARM64. This release adapts their orchestration and
paths for a fresh deployment; its local verification is documented in
[validation](docs/validation.md). Publishing instructions are in
[the branch publication guide](docs/publishing.md). Do not interpret a passing unit test as a
newly completed 14B hardware campaign. Published timing values are observations,
not acceptance thresholds for a different machine.

Direct Python dependencies and llama.cpp source are pinned. The base image digest is pinned.
Debian packages and transitive Python dependencies can change. Record the built
image digest and installed dependency inventory with each reproduction. Use an
immutable image digest when sharing a deployment between machines.

Model and dataset attribution appears in [third-party notices](THIRD_PARTY.md).
