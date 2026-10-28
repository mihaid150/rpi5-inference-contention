import contextlib
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import threading
import time
import traceback
from backends import Standard
from design import allocation, regime_rows, remaining_prefix, prediction_rows
from index_path import point
from timeline import run_timeline
from worker import atomic, environment, process_sample, receive

OUT = Path(os.environ.get("DLI_OUTPUT", "/output"))


def version(index):
    with sqlite3.connect(Path(index) / "index.sqlite") as db:
        return db.execute("SELECT version FROM current WHERE singleton=1").fetchone()[0]


def encoder_process(conn, cpus, root, template, mode):
    try:
        os.sched_setaffinity(0, set(cpus))
        import torch

        torch.set_num_interop_threads(1)
        from transformers import AutoTokenizer
        from research_u3 import publish, retrieve
        from research_u3_encoder import MiniLMEncoder
        from research_u3_chunking import StableChunker

        root = Path(root)
        root.mkdir()
        encoder = MiniLMEncoder("/input/minilm", threads=2)
        chunker = StableChunker(encoder.token_count, encoder.encoder_id)
        tok = AutoTokenizer.from_pretrained(
            "/input/q4-tokenizer", local_files_only=True
        )
        corpus = json.loads(Path("/input/corpus.json").read_text())
        old = root / "v1-index"
        active = root / "active-index"
        before = publish(
            old,
            "v1",
            corpus["snapshots"]["v1"],
            encoder,
            encoder.encoder_id,
            batch_size=4,
            chunker=chunker,
        )
        with sqlite3.connect(old / "index.sqlite") as db:
            if db.execute(
                "SELECT count(*) FROM members WHERE version=? AND source=?",
                ("v1", template["source"]),
            ).fetchone()[0]:
                raise RuntimeError("New source already present in v1")
        point(active, old)
        conn.send(
            dict(
                ready=True,
                identity=encoder.identity,
                v1=before,
                affinity=process_sample(os.getpid(), cpus),
            )
        )
        if conn.recv() != "publish":
            raise RuntimeError("Wrong publication command")
        docs = corpus["snapshots"]["v2"] + [
            dict(id=template["source"], text=template["document"])
        ]
        start = time.monotonic()
        target = old if mode == "incremental" else root / "rebuilt-index"
        result = publish(
            target,
            "v2",
            docs,
            encoder,
            encoder.encoder_id,
            batch_size=4,
            chunker=chunker,
        )
        if mode == "full-rebuild":
            if version(active) != "v1" or version(old) != "v1":
                raise RuntimeError("v1 lost during rebuild")
            point(active, target)
        end = time.monotonic()
        if version(active) != "v2":
            raise RuntimeError("v2 pointer not published")
        conn.send(
            dict(
                start=start,
                end=end,
                duration_s=end - start,
                result=result,
                mode=mode,
                old_index_version=version(old),
                affinity=process_sample(os.getpid(), cpus),
            )
        )
        if conn.recv() != "retrieve":
            raise RuntimeError("Wrong retrieval command")
        rs = time.monotonic()
        passages = retrieve(
            active,
            template["question"],
            [],
            encoder,
            encoder.encoder_id,
            version=None,
            k=3,
        )
        rendered = "\n\n".join(
            (
                f"[{p['source_id']}:{p['chunk']}]\n{p['text']}"
                for p in passages["passages"]
            )
        )
        messages = [
            dict(role="system", content=template["messages"][0]["content"]),
            dict(
                role="user",
                content=f"Document version: {passages['version']}\nDocuments:\n{rendered}\nQuestion: {template['question']}",
            ),
        ]
        tokens = tok.encode(
            tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            add_special_tokens=False,
        )
        if len(tokens) > 1984:
            raise RuntimeError("Retrieved prompt too long; no truncation permitted")
        case = dict(
            template,
            tokens=tokens,
            messages=messages,
            citation_chunks=[
                p["chunk"]
                for p in passages["passages"]
                if p["source_id"] == template["source"]
            ],
        )
        conn.send(dict(start=rs, end=time.monotonic(), retrieval=passages, case=case))
        conn.recv()
    except BaseException:
        with contextlib.suppress(Exception):
            conn.send(dict(error=traceback.format_exc()))
    finally:
        conn.close()


def main():
    plan = json.loads(Path("/input/campaign.json").read_text())
    prior = json.loads(Path("/input/prior.json").read_text())
    stage = os.environ.get("EXPERIMENT_STAGE", "E2")
    cells = prediction_rows(plan) if stage == "E3" else regime_rows(plan)
    total = len(cells)
    env = environment()
    cpus = env["cpus"]
    report = dict(
        status="preflight",
        stage=stage,
        rows=[],
        environment=env,
        plan=plan,
        semantic_review="pending",
    )
    predictions = {}
    if stage == "E3":
        source = Path(os.environ["DLI_FREEZE"]) / "predictions.json"
        expected = json.loads(
            (Path(os.environ["DLI_FREEZE"]) / "prediction-lock.json").read_text()
        )["sha256"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Prediction freeze changed")
        freeze = json.loads(source.read_text())
        predictions = {x["cell"]["id"]: x for x in freeze["predictions"]}
        if [x["cell"] for x in freeze["predictions"]] != cells:
            raise RuntimeError("Predictions differ from fixed E3 schedule")
        report["prediction_freeze"] = dict(sha256=expected, content=freeze)
    resume_path = Path("/poc-input/resume.json")
    if resume_path.exists():
        resume = json.loads(resume_path.read_text())
        previous_path = Path("/poc-input/resume-report.json")
        if (
            hashlib.sha256(previous_path.read_bytes()).hexdigest()
            != resume["report_sha256"]
        ):
            raise RuntimeError("Resume report digest mismatch")
        previous = json.loads(previous_path.read_text())
        done = [row["cell"]["id"] for row in previous["rows"]]
        remaining = remaining_prefix(cells, previous["rows"], resume["completed_ids"])
        report["rows"] = [
            dict(row, artifact_run=resume["source_run"]) for row in previous["rows"]
        ]
        report["resume"] = resume
        cells = remaining

    def save(**state):
        atomic(OUT / "report.json", report)
        atomic(
            OUT / "status.json",
            dict(
                updated=time.time(), completed=len(report["rows"]), total=total, **state
            ),
        )
        print(json.dumps(state), flush=True)

    save(phase="preflight", message="Verifying model bytes")
    with Path("/models/family/Q4_K_M.gguf").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != plan["model_sha256"]:
        raise RuntimeError("Model digest mismatch")
    report["model_sha256"] = digest
    report["status"] = "running"
    from plan import quality
    from policy import score

    spec = importlib.util.spec_from_file_location(
        "official_squad", "/input/official_evaluator.py"
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)

    class Evaluator:
        compute_exact = staticmethod(evaluator.compute_exact)

        def compute_turn_score(self, document_id, turn_id, answer):
            case = next(
                (c for c in prior["evaluation"] if c["document_id"] == document_id)
            )
            gold = case["gold_answers"]
            return dict(
                em=max((evaluator.compute_exact(g, answer) for g in gold)),
                f1=max((evaluator.compute_f1(g, answer) for g in gold)),
            )

    scoring = Evaluator()
    templates = {c["id"]: c for c in prior["evaluation"]}
    oldcase = next((c for c in prior["selection"] if c["id"] == "eval-original-u3"))
    review = []
    review_key = []
    if resume_path.exists():
        review = [
            json.loads(line)
            for line in Path("/poc-input/resume-review.jsonl").read_text().splitlines()
        ]
        review_key = json.loads(Path("/poc-input/resume-review-key.json").read_text())
    for cell in cells:
        backend = None
        child = None
        conn = None
        sampler = None
        stop = threading.Event()
        sampling_errors = []
        record = dict(
            cell=cell,
            artifact_run=os.environ.get(
                "RUN_ID", os.environ.get("E2_RUN_ID", "shared-budget-14b-e2-20261027")
            ),
            setup_start=time.monotonic(),
        )
        save(phase="setup", cell=cell)
        try:
            arm, encoder_cpus, _ = allocation(
                "shared" if cell["policy"] == "deferred" else cell["policy"], cpus
            )
            arm.update(
                name="server-" + stage.lower() + "-" + cell["policy"],
                microbatch=512,
                cache="f16",
                flash="auto",
            )
            backend = Standard(arm, cell["id"], OUT)
            record.update(startup_s=backend.load_s, argv=backend.argv)
            record["warmup"] = backend.complete(oldcase, "warmup", cell["block"])
            ctx = mp.get_context("spawn")
            conn, remote = ctx.Pipe()
            child = ctx.Process(
                target=encoder_process,
                args=(
                    remote,
                    encoder_cpus,
                    str(OUT / cell["id"]),
                    templates[cell["fixture"]],
                    cell["update"],
                ),
            )
            child.start()
            remote.close()
            record["encoder_ready"] = receive(conn, 900)
            samples = OUT / (cell["id"] + "-samples.jsonl")

            def sample():
                try:
                    with samples.open("w") as f:
                        while not stop.is_set():
                            item = dict(
                                t=time.monotonic(),
                                llm=process_sample(backend.proc.pid, arm["cpus"]),
                                encoder=process_sample(child.pid, encoder_cpus),
                                cgroup_cpu=Path("/sys/fs/cgroup/cpu.stat").read_text(),
                                cgroup_memory=Path(
                                    "/sys/fs/cgroup/memory.current"
                                ).read_text(),
                            )
                            for label, path in [
                                (
                                    "temperature",
                                    "/sys/class/thermal/thermal_zone0/temp",
                                ),
                                (
                                    "frequency",
                                    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
                                ),
                            ]:
                                with contextlib.suppress(OSError):
                                    item[label] = Path(path).read_text().strip()
                            f.write(json.dumps(item) + "\n")
                            f.flush()
                            stop.wait(0.5)
                except BaseException:
                    sampling_errors.append(traceback.format_exc())

            events = OUT / (cell["id"] + "-events.jsonl")

            def event(item):
                with events.open("a") as f:
                    f.write(json.dumps(item) + "\n")

            def current():
                return backend.complete(oldcase, "current", cell["block"])

            def update():
                conn.send("publish")
                return receive(conn, 900)

            def fresh(mark):
                mark("retrieval-start")
                conn.send("retrieve")
                retrieved = receive(conn, 900)
                mark("retrieval-complete")
                record["retrieval"] = retrieved
                mark("fresh-submit")
                result = backend.complete(retrieved["case"], "fresh", cell["block"])
                return result

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            save(phase="running", cell=cell)
            timeline = run_timeline(
                cell["fresh_arrival_s"],
                cell["policy"] == "deferred",
                current,
                update,
                fresh,
                event=event,
                timeout=1800,
            )
            record["timeline"] = timeline
            if sampling_errors:
                raise RuntimeError(sampling_errors)
            tc = timeline["current"]
            fresh_result = timeline["fresh"]
            retrieved = record["retrieval"]
            record["current_quality"] = quality(oldcase, tc["text"], tc["stop"])
            record["fresh_quality"] = score(
                retrieved["case"], fresh_result["text"], fresh_result["stop"], scoring
            )
            passages = retrieved["retrieval"]
            present = {(p["source_id"], p["chunk"]) for p in passages["passages"]}
            cited = {(s, int(k)) for s, k in record["fresh_quality"]["citations"]}
            record["retrieval_passed"] = passages["version"] == "v2" and any(
                (
                    p["source_id"] == templates[cell["fixture"]]["source"]
                    for p in passages["passages"]
                )
            )
            record["automatic_fresh_passed"] = (
                record["fresh_quality"]["automatic_supported_correct"]
                and bool(cited)
                and cited.issubset(present)
                and record["retrieval_passed"]
            )
            origin = timeline["origin"]
            arrived = next(
                (e["t"] for e in timeline["events"] if e["name"] == "fresh-arrival")
            )
            record["metrics"] = dict(
                current_complete_s=tc["completion_monotonic"] - origin,
                publication_s=timeline["update"]["end"] - origin,
                fresh_complete_s=fresh_result["completion_monotonic"] - origin,
                fresh_latency_from_actual_arrival_s=fresh_result["completion_monotonic"]
                - arrived,
                arrival_lateness_s=arrived - (origin + cell["fresh_arrival_s"]),
                fresh_client_ttft_s=fresh_result["ttft_s"],
                retrieval_s=retrieved["end"] - retrieved["start"],
            )
            record["queue_measurement"] = (
                "client submission and first token observed; exact queue-dequeue timestamp unavailable in unmodified backend"
            )
            if stage == "E3":
                if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Prediction freeze changed during E3")
                pred = predictions[cell["id"]]["prediction"]
                record["prediction_check"] = dict(
                    predicted=pred,
                    absolute_error_s={
                        key: abs(pred[key] - record["metrics"][key])
                        for key in (
                            "current_complete_s",
                            "publication_s",
                            "fresh_complete_s",
                            "fresh_latency_from_actual_arrival_s",
                        )
                    },
                    relative_error={
                        key: abs(pred[key] - record["metrics"][key])
                        / max(abs(record["metrics"][key]), 1e-09)
                        for key in (
                            "current_complete_s",
                            "publication_s",
                            "fresh_complete_s",
                            "fresh_latency_from_actual_arrival_s",
                        )
                    },
                )
            for kind, output in [("current", tc), ("fresh", fresh_result)]:
                code = hashlib.sha256(
                    (cell["id"] + "|" + kind + "|20261027").encode()
                ).hexdigest()[:16]
                case = oldcase if kind == "current" else retrieved["case"]
                review.append(
                    dict(
                        review_id=code,
                        messages=case["messages"],
                        references=case.get("gold_answers"),
                        answer=output["text"],
                        human_correct=None,
                        human_supported=None,
                        human_locator_valid=None,
                        reviewer=None,
                    )
                )
                review_key.append(dict(review_id=code, cell=cell, kind=kind))
            with (OUT / "answer-review-blinded.jsonl").open("w") as f:
                for row in sorted(review, key=lambda x: x["review_id"]):
                    f.write(json.dumps(row) + "\n")
            atomic(OUT / "answer-review-key.json", review_key)
            record["end"] = time.monotonic()
            record["sample_file"] = samples.name
            report["rows"].append(record)
            save(phase="cell-completed", cell=cell)
        finally:
            stop.set()
            if sampler:
                sampler.join(timeout=10)
            if conn:
                with contextlib.suppress(Exception):
                    conn.send("close")
                conn.close()
            if child:
                child.join(timeout=10)
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=10)
            if backend:
                backend.close()
    report["status"] = "completed"
    save(phase="completed")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        atomic(
            OUT / "failure.json", dict(error=traceback.format_exc(), time=time.time())
        )
        atomic(
            OUT / "status.json",
            dict(phase="failed", error=traceback.format_exc(), time=time.time()),
        )
        raise
