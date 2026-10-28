# Main-paper claim coverage

| Finding or mechanism | Implemented code | Recorded evidence | Rerun |
|---|---|---|---|
| Stage-0 process PSS falls about 382.56 MiB without rising node availability | `native/route_worker.cpp`, `mapped_shard.hpp`, `scripts/precision.py` | `evidence/distributed/b1/cycle-*.json` | `precision.py --execute`: three warmed cycles and same-precision continuation gate |
| Retention differs from Q4; replay recovers measured target logits | Native executor, KV export/import, `research_v2_conversation_schedule.py`, `research_v2_full_kl.py` | Twelve `evidence/distributed/b2/*.json` and compressed full-logit matrices | `precision.py --execute`: three documents, four treatments, separate teacher-forced and generation passes |
| An 81-new-chunk update slows during decode overlap | `native/legacy_decode.cpp`, `scripts/legacy/decode/` | `evidence/decode/report.json` | `launch.py --stage decode` |
| Larger native microbatch and repacking accelerate the restricted prefill baseline | `native/prefill_worker.cpp`, private graph preparation, `scripts/legacy/prefill/` | `paper/evidence/2026-10-24-final/prefill.json` | `launch.py --stage prefill`: 112 rows when all frozen gates permit completion |
| Standard server is competitive with the private prototype | Unmodified llama.cpp, `scripts/legacy/validation/backends.py` and selection policy | `paper/evidence/2026-10-24-final/validation.json`, baseline rows | `launch.py --stage validation`: all four server candidates before new-document evaluation |
| Aggregate F1 hides paired answer/citation regressions under skip4 | Fixed attention mask 39–42, full QA policy and official scorer | Same validation report, selected 24 articles, archived selection/exposure records and review packet | Same validation stage: 72 new-document rows, with frozen failures retained |
| Earlier sequential application control | `scripts/legacy/validation/run.py` | Twelve episodes in the validation report | Same validation stage; three distinct articles and four policies |
| E1 fixed reservations trade update time for prefill | `scripts/shared_budget_14b/worker.py` | `paper/evidence/2026-10-27/e1.json`, compressed affinity samples | `launch.py --stage e1-e3`: 27 E1 cells |
| E2 independent arrivals, incremental/rebuild and admission/slot interaction | `worker_e2.py`, `timeline.py`, versioned dense index | `e2.json`, original failure and prefix, recovered samples | Same runner: 36 E2 episodes |
| E3 frozen phase predictions and held-out timing cases | `predictions.py`, campaign freeze, E3 worker | `e3.json`, prediction freeze, twelve E3 sample files | Same runner: fit on E2 incremental only, freeze, then twelve E3 episodes |
| Main figures and tables | `paper/build_research_evidence.py`, `paper/build_budget_evidence.py` | Frozen source reports and `paper/generated/` | `audit_artifact.py` or direct paper audit commands |

The original full-pipeline quality time includes worker setup and separate
teacher-forced/generated passes. It must not be interpreted as one request's
latency. The portable precision coordinator preserves the treatment schedule
and invariants but does not reproduce the old recovery supervisor or require
unbundled predecessor logit files. New timings are replication measurements.

The artifact does not turn the original single-node experiments into distributed
inference. Only the precision/state pilot is partitioned; E1–E3 run the full
14B model on one Raspberry Pi 5.
