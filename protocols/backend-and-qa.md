# Sequential validation protocol, 24 October 2026

User-authorized sequence: competitive standard backend, separate documents,
one decisive joint-application experiment. This protocol is frozen before the
new model evaluations. Original experiments/thresholds remain unchanged.

## Scope and baseline

Single node09, same Qwen2.5-14B calibrated Q4_K_M, CPU-only, 4 CPU / 12 GiB safety
cap. No production deployment or additional node is modified. No distributed
run is scheduled: the tested question is whether a stronger static backend
already removes the apparent benefit. A partitioned experiment is justified
only by a new, explicit capacity/placement question, not by these timings alone.

Build unmodified llama-server from local qualified llama.cpp revision
838374375cb8c467c33c263be4801c19d1089763, separate from the private masked graph.
ARMv8.2-a+FP16+dotprod; repacking enabled, slots 1, ctx 2048, logical batch 512,
4 prefill / 2 decode threads, no GPU. Compare physical batches 64/256/512 with
F32 KV and flash attention off, plus a standard-default-like batch512/F16-KV/
flash-auto candidate. This is a bounded tuning search in the frozen library,
not a claim about the best possible current backend. The F32 arms isolate
batch size; the default-like arm is a competitive configuration, not a causal
single-variable ablation. HTTP listens only on pod loopback; no Service exposed.

For every request supply identical explicit token IDs, greedy temperature 0,
repeat penalty 1, maximum 64 generated tokens, prompt reuse disabled, no silent
context truncation. Check server timings.cache_n=0 and processed prompt length.
In this frozen server, tokens_cached is final slot occupancy, not reused-token
count: it is preserved but is not used to assert cache reuse. Readiness/load
cost, warm streaming TTFT, answer wall time, internal timers, generated tokens,
full responses and process resource snapshots are distinct evidence.

## Stage 1: finish prior campaign and select standard configuration

Require the prior 112-row campaign's completed status and collected outputs;
never rerun its cells automatically. Select the standard configuration using
only the old supported short and long fixtures (184/565 tokens). Three cyclic
repetitions of each standard arm, strict factual/unit/citation/EOS checks.
Choose minimum median answer time among complete passing standard arms.
Also measure the fixed prototype-full (repack64, F32 KV, logical frame64) and
prototype-skip4 arms on the same old fixtures, three repeats. Configurations
are not changed after seeing any new-document answer.

## Stage 2: document-disjoint answer evaluation

Use the official SQuAD 2.0 development JSON and official evaluator, downloaded
from the project dataset endpoint and upstream repository; both SHA256 digests
are pinned in prepare.py. CoQA selection preparation was abandoned before any
model run because the conservative audit did not leave 24 eligible mctest
records. No earlier outcome or historical gate was changed.

Scan exact quoted SQuAD question IDs in prior results/research-v2, results/C,
results/research-u3 and isolated attention/prefill results JSON/JSONL. Exclude an
entire SQuAD article if any of its question IDs is found. Also reject paragraphs
exactly duplicating normalized story text in the complete earlier CoQA input.
This does not detect unlogged/other-format use, paraphrase overlap or model
pretraining contamination. Preserve the audit and source hashes.

Deterministically take one paragraph/question from each of the first 24 eligible
article titles, alternating answerable and unanswerable cases (12 each), with
prompt <=1200 tokens. Choose the first qualifying paragraph/question in source
order. No choice uses model predictions. Preserve context, original reference
spans and all annotated answer alternatives. This is a source/article-disjoint
cohort for this tuning sequence, not a general confirmatory or contamination-free
set. SQuAD is not a domain-specific household utility corpus.

Evaluate selected standard, fixed prototype-full and fixed prototype-skip4 once
per document, with independently cleared KV. Store official SQuAD max-reference
EM/F1, exact agreement with an annotation, expected source citation and EOS,
complete output and token IDs. Canonical bare unknown maps to the official
empty-answer representation for unanswerable cases; no confidence threshold is
tuned. An automatic task pass requires exact supported answer plus citation
for answerable cases, or bare unknown with no citation for unanswerable cases.
These are lexical/citation proxies, not semantic adjudication. Admission requires
no proxy loss on a document where standard passes. Report absolute scores and
all paired regressions, without discarding difficult documents or selecting a
new mask here.

Create answer-review.jsonl with source passage, gold span/alternatives, every
arm's answer, automatic checks, and empty reviewer/semantic-support labels.
Independent human review remains pending. No statistical noninferiority or
independent semantic quality claim follows from the automatic gate.

## Stage 3: one versioned-update experiment

Four fixed policies: selected standard with concurrent update, standard with
update deferred until current answer completes, prototype-full concurrent,
prototype-skip4 concurrent. Three cyclic repetitions; repetition i always adds
the i-th answerable cohort document to the real existing 81-new-chunk Git update. No padding,
reembedding a full corpus, invented arrival rate or artificial memory pressure.
All policies receive the same v1/v2 documents for each repetition. A fresh v1
index is prepared per episode, outside timing. The new source must be absent
from v1. MiniLM intra-op2/inter-op1, batch4, stable <=256-token chunks.

At time zero, an existing supported prompt is submitted and the update arrives.
Concurrent policies start actual incremental embedding/publication immediately,
including overlap with prefill. Deferred starts after that answer. The current
answer uses the same pinned old oracle context across arms; it is not claimed
as an independently retrieved RAG question. After both current answer and
publication, perform real dense top3 retrieval against the published v2 index
for the newly introduced question, then generate its answer. No gold passage
is substituted into retrieval and no truncated prompt is silently accepted.

Measure current-answer completion, atomic publication and fresh-answer completion
from the same episode origin. Record update embedding/corpus counts, retrieved
version/passages and all answers. Fresh correctness requires the automatic
answer proxy, v2, retrieval containing the new source, and every generated
citation locating a passage actually retrieved. Failed retrieval/answer remains
a failed episode, not an omitted latency sample. Candidate timings still run
for diagnosis after a stage2 regression, but cannot be declared a quality-admitted
benefit. Compare medians only after all three matched episodes complete.

A positive automatic joint result requires valid baseline/candidate episodes,
no stage2 paired regression, no worse median current-answer/publication/fresh-answer
time and at least one strictly better median versus standard-concurrent.
No favorable SLA, weighted score or percentile threshold is chosen after results.
Three episodes are descriptive, not confidence bounds. A lack of dominance is
an explicit negative/null result; no new tuning, rerun or distributed expansion
is triggered automatically. Even dominance does not establish novelty or
semantic quality until the broader evidence gaps are closed.

## Execution and failure handling

One dedicated pod, stages strictly sequential, 24-hour safety deadline. Local
systemd controller survives terminal closure, snapshots all scripts/configs/
protocol/inputs/binaries, mirrors status and collects final or partial output.
Infrastructure/malformed-output failure stops and preserves logs. Answer failures
are scored and preserved; they do not silently disappear. No auto-retry with
changed settings. Delete only the dedicated pod on exit. Report DONE only after
result collection and deletion. Host and cluster must remain available.

Primary implementation references: frozen tools/server/README.md and server-task.cpp,
server-context.cpp; official SQuAD2 evaluator and dataset pinned in the preparation manifest.
The prior prompt-length ladder replaces record_progress.sh, which targets a
different distributed deployment. No new performance change is made to that service.

The reviewer package additionally includes a deterministic shuffled file without
arm labels or automatic scores, with a separate review key. No reviewer is
assigned automatically and no pending label is treated as human validation.
