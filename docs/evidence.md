# Recorded evidence and audit

The evidence bundle is the subset needed to verify reported findings. It is
not the full local workspace or a dump of every historical experiment.

- `analysis/evidence/2026-10-24-final/` preserves the final 112-row prefill report
  and 108-row/twelve-episode backend, QA and sequential application report.
- `analysis/evidence/2026-10-27/` preserves E1/E2/E3, the prelaunch predictions
  and the recorded numerical audit. These are separate evidence lineages.
- `evidence/distributed/` preserves three memory cycles, twelve quality results
  and their compressed full-vocabulary teacher-forced matrices. The audit
  recomputes KL from the matrices, not from a selected top-k approximation.
- `evidence/decode/` preserves the earlier decode-overlap report and provenance.
- `evidence/affinity/` contains one deterministically compressed sample file per
  E1/E2/E3 cell, including recovered E2 prefix samples. Uncompressed hashes and
  row counts are recorded. The audit checks 37,475 samples against each policy.
- `evidence/e2-original/` preserves the original interrupted E2 report/failure.
  The audit verifies that its seventeen committed rows survive unchanged.
- `evidence/selection/` records the selected QA cohort, original exposure audit,
  references and blinded review packet. The separate key permits later mapping;
  it does not mean that independent review has happened.

`evidence/manifest.json` pins the frozen files. The original paper audit scripts
also check their source digests and every generated table/figure input. The
archive has not been edited to remove quality failures or improve the narrative.
Historical paths, source hashes and private-network addresses in these reports
are evidence, not active deployment configuration. No kubeconfig, access token,
SSH key, weight file, binary or index database is bundled.

Run the audit with regular Python, not `python -O`:

```bash
.venv/bin/python scripts/audit_artifact.py
```

## Manuscript availability

The manuscript, its separate submission appendix and conference style files
are not distributed in this artifact branch. The repository provides code,
fixed inputs and recorded evidence independently of the manuscript.
The `analysis/generated/` files and the numerical memory plot source are
verification inputs, not a copy of the manuscript.

Do not use generated or synthetic results to fill a missing hardware observation.
Saved full-logit matrices support recomputation of the reported KL. Evidence
integrity does not replace model regeneration, hardware reproduction or human
adjudication of the answers.

## Run accounting and measurement scope

The execution campaign contains 112 runs. Subsequent backend and answer
validation contains 108 runs: 36 on existing fixtures and 72 on 24 selected
articles across three configurations. Twelve sequential application episodes
are counted separately from those 108 runs. The two complete reports are
`analysis/evidence/2026-10-24-final/prefill.json` and `validation.json`.

E1–E3 contain 75 runs: 27 diagnostic E1 runs, 36 E2 application episodes and
12 E3 timing-evaluation episodes. The 48 application episodes cover six distinct
questions and are included in the 75 runs, not additional to them. Reports
and frozen predictions are under `analysis/evidence/2026-10-27/`.

The distributed precision-transition experiment remains separate under
`evidence/distributed/`: three memory cycles and twelve state/quality jobs
(three conversations with four treatments). Teacher-forced and generated
passes do not represent the latency of one application request. Process PSS,
node memory availability and application completion have different scopes.
Neither distinct documents nor different workload families are independent
replications of a common request.

## Sequential request-submission control

Four policies compare standard concurrent, standard deferred, full-prototype
concurrent and skip4-prototype concurrent execution over three matched blocks.
Each block adds a different answerable article to the same real Git update
with 81 new chunks. The new request waits for current completion and index
publication before retrieving the top three passages from v2. This differs
from independently scheduled arrivals in E2/E3.

Fresh v1 preparation is outside timing and checks that the new source is absent.
The MiniLM encoder uses intra-op 2/inter-op 1, batch 4 and chunks of at most
256 encoder tokens. Current requests use supplied context, while new requests
perform actual retrieval. Answer, version and citation failures remain in the
records. Complete episodes are in the validation report's `episodes` array;
`analysis/generated/application-rows.tex` retains the derived timing medians.

All twelve episodes complete and retrieve the new source, but no configuration
passes the original all-episode answer-quality gate. The full prototype's
publication median is 10.3 s versus 14.7 s for standard concurrent execution,
while new-answer completion is 198.9 s versus 189.8 s. Faster publication
therefore does not establish faster valid answers. Independent semantic
review has not been completed.
