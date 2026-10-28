# Running and reading the experiments

## Schedule

| Stage | Cells | Question |
|---|---:|---|
| E1 | 27 | How do shared and fixed CPU allocations affect inference and updates separately and together? |
| E2 | 36 | Does earlier publication improve the time to answer an independently arriving question? |
| E3 | 12 | Do phase predictions fitted on E2 transfer to different documents and arrival times? |

E1 uses two fixed selection prompts, three blocks, an update-only control and
shared, inference-only, 2+2 and 3+1 allocations. E2 crosses three different
documents with incremental/full rebuild, arrival offsets 0/90 seconds and
shared/2+2/deferred execution. E3 uses three different documents, offsets 30/60
seconds and shared/2+2 execution. Three repeated configurations do not imply
three independent questions. E2/E3 together cover six answerable questions.

The shared policy gives inference and the encoder the same four CPUs. The
2+2 and 3+1 policies use disjoint fixed sets. These reservations persist after
the update ends. Deferred execution starts the update after the current answer.
None of the policies implements dynamic core reclamation.

Untimed functional qualification runs before E2 and E3. It verifies version
publication, preservation of the previous snapshot during rebuild, and retrieval
of the added source. A functional failure stops the campaign before timing.

After E2 completes, the runner fits per-policy phase medians from the six
incremental observations per policy. Fresh service uses engine prompt and
generation times, excluding queue residence. It writes predictions and their
hash before E3 starts. E3 validates that hash and preserves prediction errors.
No E3 answer length or timing is used to fit predictions.

## Docker

Use the README commands. Preview the launch command without creating files or
starting workloads:

```bash
python3 scripts/launch.py --assets workspace/assets --output results/attempt-01 --dry-run
```

To stop an attempt, find its name in `docker ps` and run `docker stop NAME`.
Keep its partial results. There is no automatic resume of a failed attempt in
this portable runner. The original historical E2 recovery is not replayed.

## Kubernetes

Run these commands on the node holding the prepared assets, using your own
node name, image location and existing namespace:

```bash
python3 scripts/launch.py --engine kubernetes --node YOUR_NODE --namespace YOUR_NAMESPACE --image shared-budget-evaluation:local --assets workspace/assets --output results/attempt-01 --name evaluation-01 --dry-run > workspace/job-preview.json
python3 scripts/launch.py --engine kubernetes --node YOUR_NODE --namespace YOUR_NAMESPACE --image shared-budget-evaluation:local --assets workspace/assets --output results/attempt-01 --name evaluation-01
kubectl -n YOUR_NAMESPACE logs -f job/evaluation-01
kubectl -n YOUR_NAMESPACE wait --for=condition=complete job/evaluation-01 --timeout=72h
```

The Job has `backoffLimit: 0`, so failed stages are not silently retried.
Completed/failed Jobs have a one-hour TTL. Durable results stay in the hostPath.
Stop only the Job you launched:

```bash
kubectl -n YOUR_NAMESPACE delete job evaluation-01
```

The `DONE` file, not deletion of a pod, indicates experimental success.
Do not launch Docker and Kubernetes campaigns on the node simultaneously.

## Outputs

Each attempt contains `status.json`, `versions.json`, the frozen predictions,
`summary.json`, and `e1/`, `e2/`, `e3/`. Stage folders contain reports, per-cell
logs, thread samples, index state and answer review artifacts. Preserve these
outside Git. To collect them from a node without following index symlinks:

```bash
tar -czf attempt-01.tar.gz -C results attempt-01
```

Do not use `tar -h` or recursive collection that dereferences absolute index
symlinks. The report and answer artifacts remain readable after collection;
opening archived index pointers on another machine may require relocation.

`fresh_complete_s` is measured from episode origin. Fresh response latency is
measured from the actual fresh-question arrival. Publication, submission and
completion are separate events. A faster publication does not imply an earlier
answer when the single inference slot is occupied.

Quality fields are automatic lexical and citation checks. Preserve failures.
A lexical mismatch is not sufficient evidence of a hallucination. Independent
semantic review remains necessary. Neither `DONE` nor a timing improvement
certifies answer quality or production service readiness.
