# Prefill first: ARM/Qwen14B execution design, 2026-10-24

## Decision and evidence

Optimize the complete Qwen2.5-14B Q4_K_M prefill on node09 before evaluating
fixed four-attention skipping as an additional, approximate configuration.
This is a private prototype, not a new algorithm or an integrated DLI controller.
The original branch, model weights and qualified vendor tree stay untouched.

The completed 75-cell diagnostic campaign measured 297.7 s prefill compute and
338.2 s whole request for the 565-token case using 4 prefill / 2 decode threads;
short-case request median was 133.0 s versus 145.1 s with 4/4. CPU partitioning
hurt publication or prefill. Microbatch 4/8 changed final logits by up to 1.79,
while keeping the probed first argmax. Those candidates were excluded by the
old **exact** gate, so their answer quality and performance were not established.
The new protocol explicitly evaluates approximate execution rather than
silently relaxing that completed campaign's gate.

Hardware read on 2026-10-24: four Cortex-A76 cores, 2.4 GHz maximum, 512 KiB L2
per core and 2 MiB shared L3; FP16 and integer dot-product present, no I8MM/SVE
in the reported features. Governor ondemand remains unchanged. Frequency and
thermal variation remain possible; collect temperatures and configuration.

Qwen14B has 48 blocks, hidden width 5120, FFN width 13824, 40 attention heads
and 8 KV heads. Each block's FFN has 212,336,640 projection parameters versus
62,914,560 attention projection parameters. FFN is 77.1% of this projection work;
attention softmax and other costs are excluded. Four skipped attentions remove
only 1.9% of all block projection parameters. This explains the limited scope
for attention-only savings; it is not a prediction of measured latency.

## Mechanism and concrete implementation

1. **Reuse quantized weights across more prompt tokens.** Existing execution
   uses logical input frames of 64 tokens but physical microbatch 2. These are
   different quantities: only n_ubatch controls graph splitting. Evaluate raw
   Q4_K_M with microbatch 2, 8, 32, 64; logical frame stays 64 throughout.
   Larger batches amortize graph scheduling and may improve weight reuse.
   A76's small cache cannot hold 8.79 GB of weights; no claim of measured
   bandwidth saturation follows from that size comparison alone.
2. **Use the CPU kernel layout supported by this frozen library.** The old
   worker hard-disables mp.use_extra_bufts. In the frozen source,
   ggml-cpu/repack.cpp selects Q4_K/Q6_K 8x4 repacked kernels when NEON +
   dot-product are available and output rows are divisible by 8. Those conditions
   fit the major Qwen projection dimensions. Test repack at microbatch 2 and 64
   separately, retaining a raw-mb64 comparator. This isolates layout from batch
   changes. Runtime layout conversion does not change the GGUF quantization
   format, but changed arithmetic and execution may change logits. Startup may
   become slower and anonymous RSS may rise; measure both. Twelve GiB is a
   safety cap, not evidence of natural deployment feasibility or page ownership.
3. **Retain 4/2 phase threads and persist the loaded model within each arm.**
   Add a zero-byte binary control frame declaring the next prompt length;
   position-zero token input always clears KV. Thus singleton tail-prefill
   batches use prefill threads, each request has independent KV, and model load
   is measured separately from warm requests. Default remains unmasked/raw;
   larger microbatch, repack and masks require explicit approximate opt-in.
4. **Only then compose fixed attention skipping.** Compare the admitted full
   winner with the same winner plus attention mask 39–42, retaining all FFNs.
   Mask remains fixed for the entire worker session. Never reactivate skipped
   layers during a request: missing KV entries and changed downstream states
   would invalidate that operation. Do not retune the mask on the new cases.

No BLAS build, flash-attention change, KV quantization, new model quantization,
retraining, or prompt shortening is bundled into this increment. Those would
confound the identified batch/layout mechanisms. Exact prefix-cache reuse may
be useful for recurring documents but requires a version/token/model/mask key
and an actual repeated-prefix workload; this increment measures fresh prefill.

## Frozen experiment order and gates

The protocol and fixture JSON are hashed before deployment. Every inspected
case is development data. The previous six cases are retained, plus four new
synthetic document controls (two selection, two transfer); these are not an
independent or representative utility dataset.

0. New raw-mb2 worker must reproduce reference last logits bit-for-bit after
   64 and 565 tokens; nonfinite values/protocol failure abort the run. Compare
   both shorter-prefix probes and full greedy answers for execution variants.
1. Six arms above, four fixed selection cases each, one deterministic greedy
   pass per cell. Store full generated tokens, text, last-prefix logits,
   max-absolute/RMS/KL/top-argmax differences, compute and wall times, startup
   and memory. Changed logits are reported, not treated as quality success or
   failure. Select lowest summed prefill compute among paired factual passes
   with no regression on any baseline strict-output pass. No incomplete arm
   can qualify. Selection-time speed is exploratory n=1.
2. Compare baseline and selected arm on six reserved **development transfer**
   cases, including original long prompt, version correction, negation and
   unsupported questions. They were already inspected in earlier campaigns
   except the two new controls: no held-out claim. If the quality gate fails,
   retain evidence and stop composition/benefit claims; do not try another
   selected arm on the same transfer cases automatically.
3. Evaluate fixed skip4 on all ten cases against the full selected arm. Only a
   passing paired gate allows skip4 into performance/co-tenancy comparisons.
4. Three repetitions, cyclic arm order: baseline, selected and optional skip4
   on short and long answers, alone and with real MiniLM incremental updates
   starting at decode. Three update-alone controls. Capture warm TTFT, prefill
   compute, decode, answer-completion time, publication, joint-completion time,
   ITLs, CPU and memory. Include
   identical-token teacher-forced prefix/decode probes in numerical records;
   greedy timing is an actual answer comparison and may have different lengths.
5. Baseline and admitted selected full arm, fresh KV prefix ladder
   128/256/512/1024, three cyclic repetitions, no decode or co-tenant. Prefixes
   longer than the original tokenized prompt are constructed from repeated
   original document text with one chat template, explicitly labeled synthetic
   length probes. Numerical/performance evidence only, not answer quality.

Strict unknown remains failed when a citation is added. A separate, anchored
abstention check accepts only unknown optionally followed by the expected
citation, never an invented fact. Admission requires factual passes plus no
strict regression on baseline strict passes; it does **not** declare all strict
checks passed. Report these two outcomes separately. Required values/citations
and forbidden obsolete values are mechanical checks, not human adjudication.

No practical response/update deadline has been approved. Faster prefill alone
cannot establish service success, adaptive utility, energy benefit or MLSys
novelty. General attention dropping is already published; larger batches and
CPU repacking are existing runtime mechanisms. The research value would need a
reproducible deployment trade-off and stronger independent task evaluation.

## Sources inspected

- Frozen llama.cpp 838374375cb8c467c33c263be4801c19d1089763:
  ggml/src/ggml-cpu/repack.cpp (Q4_K/Q6_K selection),
  ggml/src/ggml-cpu/ggml-cpu.c (matmul scheduling),
  include/llama.h (logical/physical batch distinction), local source snapshot.
- [Official llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md):
  CPU build/backend options; the experiment uses the frozen local implementation.
- [Qwen2.5-14B model card](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/main/README.md):
  model architecture and GQA dimensions.
- [Attention Drop](https://arxiv.org/html/2406.15786v6): prior work on sublayer
  redundancy; this experiment does not establish a new dropping method.

The old record_progress.sh targets the historical distributed deployment.
For this monolithic private backend, use this campaign's ladder and append
its resulting medians to runs/PROGRESS.md; do not benchmark the other system.
