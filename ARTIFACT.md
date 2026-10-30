# Artifact guide

The main manuscript and the recorded reports distinguish process memory,
node availability, isolated runtime performance, answer quality, document
publication and application completion. The artifact keeps those evidence
lineages separate.

## Three levels of use

1. **Audit recorded findings:** run `scripts/audit_artifact.py`. This is local,
   read-only and uses no models. It verifies derivations from archived evidence.
2. **Qualify software:** run the Python tests, compile native workers and run
   CTest. An optional small-shard integration test checks KV transfer and actual
   unmapping. This is not an ARM14B performance experiment.
3. **Rerun hardware experiments:** prepare the exact model family, images and
   nodes, then use `scripts/run_paper.py` or run each family separately.
   New measurements belong to a new output directory and must be reported
   with their hardware, image digest, model identity and failure history.

See [coverage](docs/coverage.md) for each finding's command and evidence.
No command on the audit path creates a pod, starts inference or modifies the cluster.

## Models and data

Weights are intentionally external. Q4_K_M is not a generic interchangeable
file name: the measured model has a fixed SHA-256 and importance calibration.
The distributed experiment also needs the matched Q3_K_M variant with identical
non-block tensors. The repository supplies calibration text, source pins,
quantization commands and tensor-preserving shard construction.
Model regeneration still needs verification against the recorded hashes.

Small selected inputs and the evidence needed to recompute reported findings
are bundled. Large discarded campaigns, arbitrary historical logs and databases
are not. The selected SQuAD and CoQA cases retain their identifiers and attribution.
The original exact-ID exposure audit is archived. Its presence does not establish
absence of model pretraining exposure or permit rescanning unbundled private history.

## Runtime expectations

Single-node runs use Raspberry Pi 5, 16 GB physical RAM, NVMe, four Cortex-A76
cores, CPU-only inference, a four-core quota and a 12 GiB safety cap. Four-node
precision runs use twelve transformer layers per stage, one native compute
thread per stage, and coordinator-mediated `kubectl exec` activation relay.
That relay is the studied functional pilot transport, not production networking.

Original failure accounting is preserved: E2 recovered seventeen committed
rows and continued the remaining nineteen. The portable runner instead starts
a new complete attempt and stops at a failure; it does not silently repeat a
subset of completed cells.

## Scope of historical context

The current main paper reports the Qwen14B study. Earlier TinyLlama A/P0/C
protocols were narrative background in an older appendix, not data pooled into
these results. They are not silently relabeled as reproducible Qwen experiments.
The evidence guide at `docs/evidence.md` describes run accounting and
measurement scope without distributing the manuscript or its submission appendix. The pre-release history is not bundled.

## Status

This is a reproducibility artifact with recorded evidence and portable execution
paths. Independent semantic adjudication is pending. No artifact badge,
production service-level guarantee, universal scheduler result, causal bandwidth
attribution or new kernel is claimed.
