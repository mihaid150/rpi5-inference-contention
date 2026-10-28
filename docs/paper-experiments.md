# Full-paper experiment procedure

## 1. Prepare the machines

Follow [setup](setup.md). Use one Raspberry Pi 5 with 16 GB RAM and NVMe for
the single-node campaigns. Use four distinct ARM64 Kubernetes stage nodes with
enough allocatable RAM for each shard and the configured 6 GiB safety cap.
A coordinator needs Python 3.11, Docker access and `kubectl` access to the chosen
namespace. The four-node precision stage and single-node campaigns run
sequentially; they are not a simultaneous distributed assistant workload.

Allow at least 40 GB free for both prepared models, image layers, the temporary
accounted model copy and results; source conversion needs the additional space
described in the model guide.

Record OS/kernel, CPU model, governor, thermal/throttling state and free storage.
Keep unrelated application workloads off the selected nodes. No package script
changes the governor, drops global caches or installs a cluster automatically.

## 2. Prepare the models and fixed inputs

Create the preparation environment and obtain the measured Q4/Q3 pair. See
[model preparation](model.md). For regeneration from the pinned high-precision
source, add `--include-q3` to `prepare_model.py`. This reuses the same F16 model
and importance matrix and quantizes both formats with Q4_K embedding/output
tensors. Both resulting GGUF hashes must match `assets/versions.json`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-prepare.txt
.venv/bin/python scripts/prepare_assets.py --output workspace/assets --gguf /path/to/Q4_K_M.gguf --q3 /path/to/Q3_K_M.gguf
```

The prepared directory includes the two models, tokenizer, MiniLM, the frozen
prefill and full 24-article QA configurations, decode prompt, three transition
conversations and verified official SQuAD/CoQA evaluators. Evaluators are extracted from immutable
archives; model/tokenizer downloads occur here,
not during timing. Do not replace the selected cases after seeing outputs.

## 3. Build runtime images

Cross-compilation of the C++ workers can run on x86 or ARM64:

```bash
docker build --platform linux/arm64 -f deploy/Dockerfile.native -t shared-budget-native:local .
```

Build the Python/Torch runtime and final image natively on an ARM64 Docker host:

```bash
docker build --platform linux/arm64 -f deploy/Dockerfile -t shared-budget-evaluation:local .
docker build --platform linux/arm64 -f deploy/Dockerfile.paper -t shared-paper-evaluation:local .
```

If the worker image was built on a separate x86 host, transfer its image with
`docker save`/`docker load` before the final build. The worker image is a binary
artifact carrier; it is not itself an executable service. Push the final image
to your registry or import it into each stage node's container runtime. Use
immutable image digests for a shared reproduction and retain the image inventory.

## 4. Construct and preposition the four shards

Install `gguf` from the same pinned llama.cpp checkout used for model preparation.
If you supplied existing GGUF files and have no checkout yet, first clone
`https://github.com/ggml-org/llama.cpp.git` into `workspace/llama.cpp` and check out
`838374375cb8c467c33c263be4801c19d1089763`. Then run:

```bash
.venv/bin/pip install ./workspace/llama.cpp/gguf-py
.venv/bin/python scripts/prepare_shards.py --assets workspace/assets --output workspace/shards
```

The script checks both model hashes and identical non-block tensors, then
writes twelve-layer Q4 shards and a Q3 stage-0 alternate. Tensor names, types,
shapes and bytes are checked after writing. It does not requantize a quantized
model. The manifest records every shard hash.

Preposition `stage-0.gguf` and `alternate-stage-0.gguf` on stage node 0,
`stage-1.gguf` on node 1, and similarly for nodes 2 and 3. Choose a directory
such as `/srv/paper-shards` on each node and grant read access to UID 1000.
Use your normal file-transfer mechanism; no SSH address or credential is embedded
in the package. Staging time is outside the measured precision pilot.

Copy `configs/precision.example.json` to `workspace/precision.json` and set the
actual namespace, image, four node names, node-local shard directories and
absolute coordinator path to `workspace/shards/manifest.json`. Add `context`
if your kubeconfig has multiple clusters. Create the namespace yourself if needed.

## 5. Run the precision/state family

First inspect the plan; then explicitly execute it:

```bash
.venv/bin/pip install -r requirements-test.txt
.venv/bin/pip install transformers==5.7.0
.venv/bin/python scripts/precision.py --config workspace/precision.json --assets workspace/assets --output results/precision-01
.venv/bin/python scripts/precision.py --config workspace/precision.json --assets workspace/assets --output results/precision-01 --execute
```

The coordinator needs CPU Torch too for the tokenizer library environment; on
ARM64 the pinned `requirements.txt` supplies the original dependencies. The
precision model compute itself runs only inside the four native worker pods.

The script creates four uniquely named pods, verifies remote shard hashes,
runs three warmed release cycles and same-precision continuation checks, then
three documents under Q4, static Q3, retained and rebuilt treatments. Each
quality treatment uses separate teacher-forced and greedy-generation passes.
It checks target replay parity and records complete logit vectors and telemetry.
Owned pods are deleted on completion or failure; cleanup verifies their UIDs.
A failed cleanup prevents a success marker.

The portable coordinator warms the same sixteen-token prefix but uses the
current run's control outputs rather than the old predecessor run's external
NumPy oracles. It does not reproduce the historical network-retry supervisor.
All measured repetitions are new observations, not promises of identical timing.

## 6. Run the single-node families

The following commands run detached Docker containers on the Pi. They are
listed in experimental order; wait for each `DONE` before starting the next.

```bash
python3 scripts/launch.py --stage decode --image shared-paper-evaluation:local --assets workspace/assets --output results/decode-01
python3 scripts/launch.py --stage prefill --image shared-paper-evaluation:local --assets workspace/assets --output results/prefill-01
python3 scripts/launch.py --stage validation --image shared-paper-evaluation:local --assets workspace/assets --output results/validation-01
python3 scripts/launch.py --stage e1-e3 --image shared-paper-evaluation:local --assets workspace/assets --output results/budget-01
```

These four lines are individual commands, not a sequential shell schedule.
Use the next section to automate waiting and failure handling.
Each stage also supports `--dry-run`. The existing Kubernetes launcher accepts
the same `--stage` argument when you supply the target node and image.

Decode reruns the accounted model-copy control with batches 1/4/8 and actual
decode overlap. Prefill reruns the numerical qualification, frozen selection,
transfer, skip gate and length ladder. If a gate fails, later work is stopped;
there is no automatic reselection. Validation compares all four server
configurations, then three arms on 24 different articles and twelve application
episodes. E1–E3 retain their separate 27/36/12 schedule and prospective prediction freeze.

## 7. Schedule everything sequentially

On the prepared ARM64 coordinator/Pi with Docker and cluster access:

```bash
.venv/bin/python scripts/run_paper.py --assets workspace/assets --precision-config workspace/precision.json --output results/paper-01
.venv/bin/python scripts/run_paper.py --assets workspace/assets --precision-config workspace/precision.json --output results/paper-01 --execute
```

The first invocation prints a plan only. The second runs precision, decode,
prefill, validation and E1–E3 one after another. The first failed stage stops
the schedule. Use a fresh top-level directory for a new attempt and retain
partial outputs. The runner does not quietly retry or overwrite failed cells.

```bash
watch -n 10 cat results/paper-01/status.json
```

The final `DONE` means the required experimental families completed, not that
all answers were judged correct. Human semantic review remains a separate task.
The full sequence can take days, particularly if model preparation is included.

E1–E3 retain the original frozen batch512/F16/flash-auto configuration. If a
new backend-selection run prefers a different candidate, preserve that result
and report the distinction. Do not silently substitute a new arm into the
frozen schedule or call the original arm optimal on the new hardware.
