# Recorded evidence and audit

The evidence bundle is the subset needed to verify reported findings. It is
not the full local workspace or a dump of every historical experiment.

- `paper/evidence/2026-10-24-final/` preserves the final 112-row prefill report
  and 108-row/twelve-episode backend, QA and sequential application report.
- `paper/evidence/2026-10-27/` preserves E1/E2/E3, the prelaunch predictions
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

For a paper build, install LaTeX with PGFPlots/TikZ and run from `paper/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error artifact-appendix.tex
```

The official style files are unchanged. Figure and table data are supplied.
Do not use generated or synthetic results to fill a missing hardware observation.
Saved full-logit matrices support recomputation of the reported KL. Evidence
integrity does not replace model regeneration, hardware reproduction or human
adjudication of the answers.
