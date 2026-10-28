# Sources and attribution

The runtime is built from [llama.cpp](https://github.com/ggml-org/llama.cpp)
at the revision in `assets/versions.json`. Its upstream license applies.
No upstream source tree or compiled runtime is committed here.

The inference model and tokenizer are
[Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct).
The document encoder is
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
They are downloaded separately and retain their publishers' licenses.

The six question/document fixtures in `assets/prior.json` derive from
[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/), by Rajpurkar et al.,
with document and question identifiers retained. SQuAD is distributed under
CC BY-SA 4.0. The official evaluator is archived unchanged in compressed form and verified
against the recorded SHA-256 before extraction. It is third-party evidence,
not modified project code.

`assets/calibration.txt` contains the frozen chat rendering of 19 training
conversations from [CoQA](https://stanfordnlp.github.io/coqa/), by Reddy et al.
Source conversation IDs are recorded in `assets/calibration-sources.json`.
The original CoQA source terms apply to these excerpts.

`assets/corpus.json` contains the repository-document snapshots actually used
as controlled ingestion inputs, including their historical provenance. These
are experimental fixtures, not deployment instructions or current claims.
Only the required snapshots are included, not the historical repository.

Redistribution and use of third-party assets remain subject to their source
terms. No new license over third-party content is implied by this package.

The distributed quality evidence includes selected CoQA development conversations
and derived full-vocabulary logit matrices. The full-paper QA evidence includes
the selected 24 SQuAD paragraphs, questions and reference alternatives. These
remain subject to the original dataset terms. The included research C++ workers
are project code; the upstream llama.cpp source is fetched separately at the
recorded revision. The official MLSys style files retain their original notices
and have not been reformatted or stripped of third-party attribution.
