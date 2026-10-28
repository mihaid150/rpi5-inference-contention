# Validation record — 28 October 2026

The release separates recorded measurements, software checks and a new hardware
reproduction. Their completion states are not interchangeable.

## Recorded evidence

`audit_artifact.py` passes. It checks source digests, final report counts,
main-paper tables and diagram coordinates, the three release cycles, twelve
quality jobs from full-vocabulary logit matrices, the 81-chunk decode-overlap
control, 37,475 thread-affinity samples, the frozen E3 prediction identity and
the seventeen original E2 rows preserved after recovery. QA cohort and original
protocol identities are also checked. The audit does not adjudicate meaning.

The new E1–E3 summarizer was checked against the original reports and reproduces
fresh-completion MAE 22.216038003120975 seconds. Old failures remain in the evidence.

## Python and workflow checks

There are 38 tests. The default model-free run passes 37 and explicitly skips
one optional real-shard integration test. The same tests and complete evidence
audit pass in a clean Python 3.11/NumPy 2.2.6 container, with the repository
read-only and networking disabled. The optional integration test also passes
locally when compatible small Qwen shards are supplied, so all 38 pass in that
configuration. Those shards are test prerequisites, not bundled model weights.

Tests cover inherited affinity, disjoint sets, independent arrivals, deferred
publication, atomic index visibility, preservation of v1, sampler thread exit,
fixed schedules, causal direction/alignment of KL, complete-prefix replay,
framing failures, pod ownership checks and stopping later stages after failure.
Both the E1–E3 controller and full-paper schedule have simulated success/failure
checks. Python modules compile. Project Python/C++ sources have no comments;
unchanged official styles and compressed third-party evaluators retain their notices.

## Native checks

All seven native worker/test targets compile on x86 using the pinned runtime
headers and existing matching libraries. Both native CTest checks pass. The
real-shard integration test verifies continuation parity between mapped-control
and release workers, switches Q4/Q3 and checks disappearance/reappearance of
actual process mappings. It uses a small local model as software qualification,
not as a new 14B result.

The ARM64 worker artifact image cross-compiles successfully from the pinned
upstream checkout. It builds the unmodified standard server, precision workers,
restricted decode worker, private attention runtime, prefill worker and numerical
reference worker. Cross-compilation requires no ARM emulation; execution of
those ARM binaries has not been repeated on the study nodes for this release.

## Manuscript and deployment

The supplied main paper builds to twelve pages including references, with main
content within ten pages. The separate artifact appendix builds to three pages.
There are no unresolved references or overfull boxes. Official style files
remain byte-identical to the active manuscript's original files.
Docker/Kubernetes launch plans and the whole-paper schedule have non-executing
preview paths. Building the worker artifact creates no research pods.

## Outstanding hardware validation

The newly adapted complete pipeline has not completed a fresh ARM64 campaign
from a clean clone. The Python/Torch runtime and final paper image still require
a native ARM64 build, or a separately provisioned emulation environment. The
local x86 host has no ARM emulation; no host binfmt settings or cluster services
were changed for these checks.

The model family has not been regenerated end to end for this release. Its
recorded hashes are mandatory. Preparation retains logs and fails if regenerated
bytes differ. The Python base image is pinned; native builder OS packages and
transitive Python dependencies are not a complete bit-for-bit environment lock.
Retain built image digests, installed package inventories and execution provenance.

The artifact covers all families reported in the current main paper. Historical
TinyLlama A/P0/C narrative has been separated from the current artifact appendix
rather than bundled as if it were part of this Qwen14B experiment. Independent
answer adjudication, a production SLA and an artifact badge remain unclaimed.
