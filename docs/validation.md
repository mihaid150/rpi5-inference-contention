# Release validation

Validation performed on 28 October 2026:

- 20 standard-library tests pass both locally and in a clean Python 3.11
  container with read-only sources and no network, including inherited affinity, disjoint CPU
  sets, independent arrivals, deferred publication, atomic index visibility,
  preservation of v1, sampler thread exit, fixed schedules and the predictor.
- Tests simulate the full sequential controller and confirm that E2 failure
  prevents E3, preserves E1, and produces no successful completion marker.
- A fresh single-branch clone passes all 20 tests and has a clean Git status.
  The branch contains 44 tracked files and an independent root commit.
- All Python modules compile. Project Python code has no comments or docstrings.
- Docker and Kubernetes deployment dry runs generate bounded, read-only-input
  deployments without starting experiments.
- `docker build --check -f deploy/Dockerfile .` passes without warnings.
- The official SQuAD evaluator download matches the original hash.
- The new summarizer was run against copies of the original complete reports
  outside the release tree. It reads 27 E1, 36 E2 and 12 E3 cells, validates
  the frozen prediction and E2 report hashes, and reproduces fresh completion
  MAE 22.216038003120975 seconds. No original raw reports are bundled.

The adapted orchestration has not yet completed a fresh full ARM64 campaign.
A local ARM64 image build was attempted on x86, but failed with `exec format
error` because ARM emulation is unavailable. The experiment node has neither
Docker nor nerdctl installed, so native build validation was not performed
there. The instructions use a native ARM64 Docker host. No host services,
binfmt configuration or cluster infrastructure were changed for this check.

The 14B model has not been regenerated from source for this release. Its
recorded hash is mandatory; model preparation deliberately fails on a different
artifact. The base image digest is pinned, but OS packages and transitive dependency
versions are not a complete bit-for-bit environment lock. Retain the built image digest,
`environment.json`, versions, backend logs and model identity with a run.

This is an organized reproduction release of a research evaluation. Passing
these checks does not establish production application readiness, a latency
service-level guarantee, independently adjudicated answer quality or identical
performance on other Raspberry Pi configurations.
