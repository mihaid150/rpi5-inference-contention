# Raspberry Pi 5 inference contention: paper artifact

Code, fixed inputs and recorded evidence for **From Resource Savings to Useful
Answers: Characterizing Shared ARM Inference and Document Updates**.
The artifact covers every experimental family reported in the main paper:

- Four-node Qwen2.5-14B precision release, KV retention and full-prefix replay.
- MiniLM updates overlapping autoregressive decode, with model-page accounting.
- Native prefill batching, CPU repacking and fixed attention omission.
- A tuned standard llama.cpp backend, separate-document QA and application episodes.
- E1–E3 CPU allocation, independent arrivals and frozen timing predictions.

The framework is a research evaluation, not a dynamic scheduler or a production
assistant. It includes C++ inference workers and Python experiment control.

## Verify the published results without running models

Python 3.11 and NumPy are sufficient. No model download, GPU, cluster or deployment
is required for this path:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-test.txt
.venv/bin/python scripts/audit_artifact.py
.venv/bin/python -m unittest discover -s tests -v
```

The audit verifies frozen evidence hashes, all generated main-paper tables and
figure data, twelve full-vocabulary KL jobs, the process-versus-node memory
observations, the earlier 81-chunk update, 37,475 saved affinity samples and the
original seventeen-row E2 prefix preserved after its failure.
It does not change an automatic quality failure into a pass.

## Reproduce the experiments

Start with the [artifact guide](ARTIFACT.md). It links the complete
[claim-to-code-and-evidence map](docs/coverage.md),
[native build instructions](docs/native.md),
[full experiment procedure](docs/paper-experiments.md), and
[validation record](docs/validation.md).

A prepared deployment can schedule every family sequentially:

```bash
.venv/bin/python scripts/run_paper.py --assets workspace/assets --precision-config workspace/precision.json --output results/paper-attempt-01 --execute
```

This requires four prepared ARM64 stage nodes, one Raspberry Pi 5 with 16 GB RAM
for the single-node campaigns, accessible container images, the exact Q4/Q3 model
family, and permissions to create the explicitly configured experiment pods.
The command is not a substitute for setup. Without `--execute`, it only prints
the schedule. Failed stages stop the sequence and preserve their outputs.

Monitor the whole sequence with:

```bash
watch -n 10 cat results/paper-attempt-01/status.json
```

Success creates `results/paper-attempt-01/DONE`. Semantic review remains separate.
The model preparation and full hardware campaign can take many hours or days.

For E1–E3 alone, use the smaller image and [existing instructions](docs/experiments.md):

```bash
python3 scripts/launch.py --assets workspace/assets --output results/e1-e3-attempt-01
```

## Repository contents

| Path | Purpose |
|---|---|
| `native/` | C++ partition, state-transition, decode and prefill workers and native tests |
| `scripts/` | Preparation, deployment, experiment control, prediction and audit code |
| `assets/` | Fixed prompts, article excerpts, calibration text, cohort and model identities |
| `configs/` | Frozen E1–E3 schedule and configurable four-node deployment example |
| `protocols/` | Original experimental design records, retained as historical evidence |
| `evidence/` | Required frozen results, compressed logits and affinity samples |
| `analysis/` | Frozen reports, derived verification inputs and evidence audit scripts |
| `docs/` | Configuration, infrastructure, deployment, execution and interpretation |

The branch has an independent root commit. It excludes unrelated repository
history, credentials, personal environment files, model weights, compiled
binaries, index databases, installation caches and local backups. Recorded
reports and document snapshots retain historical paths and private-network
addresses as evidence; these are not deployment targets used by the runner.

The artifact is verified at the levels listed in `docs/validation.md`.
A fresh full ARM64 campaign and byte-for-byte model regeneration have not been
completed for this release. New portable orchestration must not be confused
with a new hardware result or an independently reviewed artifact badge.

[Publishing this branch](docs/publishing.md) · [Third-party attribution](THIRD_PARTY.md)
