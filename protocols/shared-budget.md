# Shared-budget Qwen2.5-14B campaign — prospective design

Prepared 26 October 2026, after examining the 24 October results. This is a new
exploratory protocol; it does not rewrite prior protocols or qualify as a
preregistration made before all relevant data. No cluster run has been launched.
Machine-readable order: configs/shared-budget-14b/campaign-2026-10-26.json.

## Problem and scope

Under a fixed four-CPU allocation, how should an assistant accommodate document
maintenance while serving current and new-evidence questions? Test whether a
measured inference-side change alters contention, evidence availability and
answer completion, with quality assessed separately. The intended output is
an explanatory comparison across regimes, not a guaranteed positive speedup.

Use the existing Qwen2.5-14B Q4_K_M (SHA256 in manifest), one ARM node09,
unmodified llama-server at 838374375cb8c467c33c263be4801c19d1089763.
Keep selected batch512/F16-KV/flash-auto/repack, ctx2048, slot1, greedy,
64 generated-token cap, no prefix reuse and MiniLM batch4/chunks<=256.
The 12-GiB cap is the existing safety cap, not a demonstrated application need.
No distribution, quantization transition, mask selection or new kernel in this
campaign. It cannot establish benefits for those different mechanisms.

## E1 — What can CPU allocation explain? 27 cells

Three cyclic blocks: update-only once, and each old 184/565-token prompt under
inference-only, shared concurrent, fixed 2+2 CPU partition, fixed 3+1 partition.
Use the original real incremental 81-chunk update, no duplicated/padded data.
Each cell starts with an identical warmed backend and fresh equivalent v1;
setup excluded from request time but retained as readiness cost. Clear request KV.

Shared: LLM four prefill/two decode threads and encoder two threads on the same
four allowed CPUs. Partition2+2: LLM two/two on two CPUs, encoder two on the other
two. Partition3+1: LLM three/two on three, encoder one on the remaining CPU.
Inter-op=1. No increase in aggregate CPU or memory entitlement. Affinity must
apply to every existing/new thread of the relevant subprocess; verify thread
allowed lists, effective cpuset and actual CPU quota. A four-CPU quota alone is
not proof of four physical assigned CPUs. If the pod cannot obtain a controlled
four-CPU set without touching unrelated workloads, stop this campaign.

Compare shared slowdown against isolated inference/update. Partitions are joint
allocation policies, changing both thread counts and placement; they do NOT
isolate cache or bandwidth effects. Decompose prefill/decode/embedding and collect
per-process CPU time, per-thread scheduling, runnable/wait time where available,
cgroup throttling, context switches, memory and temperature/frequency snapshots.
CPU-pressure evidence narrows hypotheses; do not declare cache/BW causality
without further controlled experiments. Hardware counters optional, no automatic
privilege change; missing counters explicitly recorded.

## E2 — Does the intervention affect the requested outcome? 36 episodes

Factorial design: 3 fixed source-order documents × 2 maintenance modes ×
2 independent fresh-question arrival offsets (0,90 seconds) × 3 policies.
Policies: shared concurrent, partition2+2 concurrent, standard deferred.
No E1 winner selection: these policies are fixed now. Rotate their order by block.
The three documents are the first three answerable cases in the existing cohort;
prior exposure is disclosed. They are exploratory fixtures, not unseen semantic
test data. Same old supported prompt of 565 tokens in every matched episode.

Maintenance modes:
- Incremental: original Git v1→v2 delta plus the fixed added article, preserving
  embeddings for unchanged chunks.
- Full rebuild: build the SAME v2 corpus into a separate empty staging index,
  encoding each actual chunk once, then atomically publish. v1 remains readable.
  This represents an explicit rebuild operation, not ordinary incremental cost.
  Record corpus/chunk/token counts and compare policies within mode. Never
  increase the corpus to force a bottleneck; no synthetic busy loop or sleeps.

Use fresh equivalent v1 and identical documents per matched cell. Assert the new
source is absent from v1. At t=0 submit the old pinned-context request and mark
the update arrival. In deferred policy start update after current completion.
An external event records the fresh question at its prescribed offset regardless
of publication or current-answer status. Arrival is NOT postponed by the runner.
The fresh question requires the added evidence. It is held by an admission gate
until v2 exists. Then retrieve real top3 v2 passages and submit to the standard
single-slot FIFO service, possibly while the old request is active. Record
admission, retrieval and queue/processing events separately. Do not submit a stale
v1 request and later score its answer as a fresh-version response.

Offsets are controlled sensitivity probes informed by earlier response durations,
not measured production traffic. Run both offsets even if results are unfavorable.
A remaining single-slot constraint may still make slot occupancy dominant; that
is an allowed outcome. We are testing whether this changes, not assuming it will.

## E3 — Test explanation on separate timing cases: 12 episodes

Next three source-order answerable articles (06,08,10), incremental updates,
arrival offsets30/60, shared versus partition2+2. No policy/mask tuning.
These articles have previous model outputs: only timing cases are held out of E2,
not prior semantic exposure. Before E3 starts, archive predictions for EACH
policy/case from E1/E2: bottleneck classification, current completion, publication,
retrieval duration and fresh service duration. Preserve model parameters/hash.
No refit using E3. Do not omit mispredicted cells or retune afterward.

Provisional timing predictor:
Tfinish = max(Tarrival,Tpublication,Tslot-free) + Tretrieval + Tfresh-service.
This approximation assumes retrieval is serialized at admission. If runner retrieval
occurs before slot release, instead use max(Tarrival,Tpublication)+Tretrieval as
request submission and max(Tsubmission,Tslot-free)+Tfresh-service as completion.
Use the implemented event ordering, not whichever formula fits better. Phase
durations use policy-specific E1/E2 medians; residual interference can invalidate
them. Report absolute prediction error, relative error and observed-versus-predicted
ordering. No arbitrary accuracy threshold selected from E3. The max identity alone
is not a research result; held-out predictive validity and regime coverage matter.

## Quality and measurement contract (all stages)

Capture monotonic timestamps for update arrival/start, each request arrival,
publication, retrieval, queue entry, first token, completion and old slot release.
Report fresh latency FROM ITS OWN ARRIVAL as well as time from update arrival.
Keep event logs, actual prompt/generation counts, cache counters and raw outputs.
Record peak process/pod memory and startup separately. No timestamp of an incorrect
answer counts as successful useful-answer completion.

Retain prior frozen lexical/citation checks as one layer; do not silently rewrite
unknown or exact-match rules. Add a separately labeled blinded semantic review
packet for current/fresh answers, evidence, reference alternatives and retrieved
source/version. Human reviewer marks factual correctness, evidence support,
locator validity and abstention independently; show raw automatic and reviewed
labels side by side. Reviewer must be independent of arm selection. Until completed,
report timings and automatic eligibility only, not semantic preservation.
No new speed benefit qualifies if it breaks the baseline-passing answer contract.
Prior skip4 remains excluded; no mask is rescued by relabeling old failures.

For each matched regime report current completion, publication, fresh latency,
quality counts and paired differences. Do not pool maintenance types or documents
as interchangeable repeated measurements. Three blocks are descriptive; no
noninferiority/equivalence claim. Report dominance and trade-offs, without inventing
an SLA or a weighted utility score. All failures stay in denominators.

## Sequential execution, runtime and failure policy

Order E1→E2→freeze predictions→E3→collect/review/analyze. Planned 75 cells;
request lengths and corpus rebuild cost can make runtime exceed one working day.
After E1 estimate remaining runtime from measured times and actual corpus token
count. Report estimate with setup margin; no promise that every cell fits24h.
Use a 24-hour hard campaign deadline and preserve partial status as incomplete.
Do not shrink corpus/repetitions or drop slow regimes after observing their results.
Infrastructure failure stops; no automatic scientific retries or changed parameters.
A follow-up explicit amendment records any rerun and original failure.

One dedicated pod, loopback-only service; inspect node inventory and protect all
unrelated cluster workloads. Use the explicit project KUBECONFIG from CLAUDE.md.
Cleanup only campaign-owned pod/processes. DONE requires collection and verified
cleanup, not merely that the last inference returned. Never reuse the old runner
without adaptation: it delays fresh arrival until BOTH old completion and publication.

## Implementation readiness

The manifest is a planned schedule, NOT a cluster-executable runner. Required
new adapter work before launch:
1. Independent event arrivals and asynchronous streaming requests to one server.
2. Real full-rebuild staging/publication path, preserving v1 availability.
3. Verified subprocess/thread CPU affinity plus matching thread configuration.
4. Phase/resource event sampler and stable failure/cleanup ownership.
5. Prediction freeze between E2/E3; separate generated/quality-qualified endpoints.
6. Small functional tests: arrival while busy, v1 absence/v2 visibility, no premature
   retrieval, output retention on quality failure, and prediction freeze enforcement.

The existing runner/backend/encoder paths are implementation references; their
old frozen behavior must remain intact. Implement in a new module, then qualify
on tiny fixtures before spending 14B time. Preparation of this experiment set
has not performed a cluster launch or changed any existing runtime.
