import os
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import statistics
import threading
import time
from transformers import AutoTokenizer
from research_u3 import publish, retrieve
from research_u3_encoder import MiniLMEncoder
from research_u3_chunking import StableChunker
from plan import quality as old_quality
from policy import (
    SERVER_ARMS,
    score,
    select_standard,
    paired_regressions,
    application_verdict,
)
from backends import Standard, Prototype, memory
import prefill_worker

OUT = Path(os.environ.get("DLI_OUTPUT", "/output"))
CONFIG = json.loads(Path("/input/validation-config.json").read_text())
REPORT = dict(
    status="running",
    started=time.time(),
    config=CONFIG,
    rows=[],
    sessions=[],
    stages=[],
    episodes=[],
    gates={},
    human_review_complete=False,
    distribution_scheduled=False,
    limits=[
        "Document disjointness is bounded by the saved local exposure audit",
        "Automatic CoQA scores and citations do not replace independent semantic review",
        "One model, one node, no approved SLA or novelty claim",
    ],
)
PROTO = dict(name="prototype-full", microbatch=64, repack=True, mask=[])
SKIP = dict(PROTO, name="prototype-skip4", mask=[39, 40, 41, 42])
CURRENT = None


def atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def save():
    atomic(OUT / "report.json", REPORT)


def state(**fields):
    atomic(
        OUT / "status.json",
        dict(
            updated=time.time(),
            completed_rows=len(REPORT["rows"]),
            completed_episodes=len(REPORT["episodes"]),
            **fields,
        ),
    )
    print(json.dumps(fields), flush=True)


def engine(arm, tag):
    global CURRENT
    value = (
        Standard(arm, tag, OUT)
        if arm["name"].startswith("server-")
        else Prototype(arm, tag, OUT)
    )
    CURRENT = value
    REPORT["sessions"].append(
        dict(
            tag=tag,
            arm=arm,
            startup_s=value.load_s,
            argv=value.argv,
            memory_after_load=memory(value.proc.pid),
        )
    )
    save()
    return value


def run_case(backend, case, stage, repeat=1):
    state(
        phase="running",
        stage=stage,
        arm=backend.arm["name"],
        fixture=case["id"],
        repeat=repeat,
    )
    row = backend.complete(case, stage, repeat)
    if stage == "baseline":
        row["assessment"] = old_quality(case, row["text"], row["stop"])
    else:
        row["assessment"] = score(case, row["text"], row["stop"], EVAL)
    REPORT["rows"].append(row)
    save()
    return row


def review_packet():
    cases = {c["id"]: c for c in CONFIG["evaluation"]}
    with (OUT / "answer-review.jsonl").open("w") as f:
        for row in REPORT["rows"]:
            if row["stage"] != "new-documents":
                continue
            case = cases[row["fixture"]]
            f.write(
                json.dumps(
                    dict(
                        document=case["document"],
                        question=case["question"],
                        references=case["gold_answers"],
                        gold_spans=case["gold_spans"],
                        arm=row["arm"],
                        answer=row["text"],
                        assessment=row["assessment"],
                        human_correct=None,
                        human_supported=None,
                        reviewer=None,
                    )
                )
                + "\n"
            )
    blinded = []
    key = []
    for row in REPORT["rows"]:
        if row["stage"] != "new-documents":
            continue
        case = cases[row["fixture"]]
        code = hashlib.sha256(
            (case["id"] + "|" + row["arm"] + "|20261024").encode()
        ).hexdigest()[:16]
        blinded.append(
            dict(
                review_id=code,
                document=case["document"],
                question=case["question"],
                references=case["gold_answers"],
                gold_spans=case["gold_spans"],
                answer=row["text"],
                human_correct=None,
                human_supported=None,
                reviewer=None,
            )
        )
        key.append(
            dict(
                review_id=code,
                fixture=row["fixture"],
                arm=row["arm"],
                automatic=row["assessment"],
            )
        )
    with (OUT / "answer-review-blinded.jsonl").open("w") as stream:
        for item in sorted(blinded, key=lambda x: x["review_id"]):
            stream.write(json.dumps(item) + "\n")
    atomic(OUT / "answer-review-key.json", key)


def fresh_case(template, passages, tok):
    rendered = "\n\n".join(
        (f"[{p['source_id']}:{p['chunk']}]\n{p['text']}" for p in passages["passages"])
    )
    messages = [
        dict(role="system", content=template["messages"][0]["content"]),
        dict(
            role="user",
            content=f"Document version: {passages['version']}\nDocuments:\n{rendered}\nQuestion: {template['question']}",
        ),
    ]
    tokens = tok.encode(
        tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        add_special_tokens=False,
    )
    if len(tokens) > 1984:
        raise ValueError(
            "Fresh retrieval prompt exceeds frozen context; no silent truncation"
        )
    return dict(
        template,
        tokens=tokens,
        messages=messages,
        citation_chunks=[
            p["chunk"]
            for p in passages["passages"]
            if p["source_id"] == template["source"]
        ],
    )


def episode(backend, policy, rep, encoder, chunker, corpus, tok):
    state(phase="running", stage="application", policy=policy, repeat=rep + 1)
    template = [c for c in CONFIG["evaluation"] if not c["unsupported"]][rep]
    index = OUT / f"episode-{policy}-{rep}"
    publish(
        index,
        "v1",
        corpus["snapshots"]["v1"],
        encoder,
        encoder.encoder_id,
        batch_size=4,
        chunker=chunker,
    )
    docs = corpus["snapshots"]["v2"] + [
        dict(id=template["source"], text=template["document"])
    ]
    with sqlite3.connect(index / "index.sqlite") as db:
        assert (
            db.execute(
                "SELECT count(*) FROM members WHERE version=? AND source=?",
                ("v1", template["source"]),
            ).fetchone()[0]
            == 0
        )
    start = time.monotonic()
    row = dict(
        policy=policy,
        repeat=rep + 1,
        new_document=template["document_id"],
        update_arrival=start,
    )
    errors = []

    def update():
        try:
            row["update"] = publish(
                index,
                "v2",
                docs,
                encoder,
                encoder.encoder_id,
                batch_size=4,
                chunker=chunker,
            )
            row["publication_s"] = time.monotonic() - start
        except BaseException as exc:
            errors.append(repr(exc))

    publisher = None
    if policy != "standard-deferred":
        publisher = threading.Thread(target=update)
        publisher.start()
    current = backend.complete(
        CONFIG["selection"][0], "application-current", repeat=rep + 1
    )
    row["current_answer_s"] = time.monotonic() - start
    row["current_output"] = current
    row["current_passed"] = old_quality(
        CONFIG["selection"][0], current["text"], current["stop"]
    )["strict_passed"]
    if policy == "standard-deferred":
        update()
    else:
        publisher.join(timeout=900)
        if publisher.is_alive():
            raise TimeoutError("Application publisher deadline")
    if errors:
        raise RuntimeError(errors)
    retrieved = retrieve(
        index, template["question"], [], encoder, encoder.encoder_id, version=None, k=3
    )
    row["retrieval"] = retrieved
    row["retrieval_passed"] = retrieved["version"] == "v2" and any(
        (p["source_id"] == template["source"] for p in retrieved["passages"])
    )
    fresh = fresh_case(template, retrieved, tok)
    result = backend.complete(fresh, "application-fresh", repeat=rep + 1)
    row["fresh_answer_s"] = time.monotonic() - start
    row["fresh_output"] = result
    row["fresh_assessment"] = score(fresh, result["text"], result["stop"], EVAL)
    cited = set(((s, int(k)) for s, k in row["fresh_assessment"]["citations"]))
    present = {(p["source_id"], p["chunk"]) for p in retrieved["passages"]}
    row["fresh_passed"] = (
        row["fresh_assessment"]["automatic_supported_correct"]
        and bool(cited)
        and cited.issubset(present)
    )
    REPORT["episodes"].append(row)
    save()


def main():
    global EVAL
    try:
        import torch

        torch.set_num_interop_threads(1)
        spec = importlib.util.spec_from_file_location(
            "official_squad", "/input/official_evaluator.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Evaluator:
            compute_exact = staticmethod(module.compute_exact)

            def compute_turn_score(self, document_id, turn_id, answer):
                case = next(
                    (c for c in CONFIG["evaluation"] if c["document_id"] == document_id)
                )
                gold = [""] if case["unsupported"] else case["gold_answers"]
                prediction = (
                    ""
                    if case["unsupported"] and answer.lower().strip() == "unknown"
                    else answer
                )
                return dict(
                    em=max((module.compute_exact(g, prediction) for g in gold)),
                    f1=max((module.compute_f1(g, prediction) for g in gold)),
                )

        EVAL = Evaluator()
        tok = AutoTokenizer.from_pretrained(
            "/input/q4-tokenizer", local_files_only=True
        )
        prefill_worker.TOK = tok
        prefill_worker.OUT = OUT / "prototype"
        prefill_worker.OUT.mkdir()
        prefill_worker.REPORT = dict(
            status="running", rows=[], sessions=[], config=CONFIG
        )
        for rep in range(3):
            arms = SERVER_ARMS[rep:] + SERVER_ARMS[:rep]
            for arm in arms:
                with engine(arm, f"baseline-{rep}-{arm['name']}") as backend:
                    for case in CONFIG["selection"]:
                        run_case(backend, case, "baseline", rep + 1)
        winner, gates = select_standard(
            REPORT["rows"],
            [c["id"] for c in CONFIG["selection"]],
            lambda r: r["assessment"],
        )
        REPORT["gates"]["standard_selection"] = gates
        REPORT["standard_selected"] = winner
        save()
        if winner is None:
            raise RuntimeError(
                "No complete standard configuration passes the old supported controls"
            )
        standard = next((a for a in SERVER_ARMS if a["name"] == winner))
        for rep in range(3):
            for arm in (PROTO, SKIP):
                with engine(arm, f"prototype-control-{rep}-{arm['name']}") as backend:
                    for case in CONFIG["selection"]:
                        run_case(backend, case, "baseline", rep + 1)
        REPORT["stages"].append("standard-baseline")
        save()
        independent = {}
        for arm in (standard, PROTO, SKIP):
            with engine(arm, "new-documents-" + arm["name"]) as backend:
                independent[arm["name"]] = [
                    run_case(backend, c, "new-documents") for c in CONFIG["evaluation"]
                ]
        admitted = {}
        for arm in (PROTO, SKIP):
            gate = paired_regressions(independent[winner], independent[arm["name"]])
            REPORT["gates"][arm["name"]] = gate
            admitted[arm["name"]] = gate["passed"]
        review_packet()
        REPORT["stages"].append("new-documents")
        save()
        encoder = MiniLMEncoder("/input/minilm", threads=2)
        chunker = StableChunker(encoder.token_count, encoder.encoder_id)
        corpus = json.loads(Path("/input/corpus.json").read_text())
        REPORT["encoder"] = encoder.identity
        policies = [
            ("standard-concurrent", standard),
            ("standard-deferred", standard),
            ("prototype-full", PROTO),
            ("prototype-skip4", SKIP),
        ]
        for rep in range(3):
            for policy, arm in policies[rep:] + policies[:rep]:
                with engine(arm, f"application-{rep}-{policy}") as backend:
                    episode(backend, policy, rep, encoder, chunker, corpus, tok)
        REPORT["application"] = application_verdict(REPORT["episodes"], admitted)
        REPORT["stages"].append("application")
        REPORT.update(status="completed", finished=time.time())
        save()
        state(phase="completed", stages=REPORT["stages"])
        (OUT / "DONE").write_text(
            "All three stages completed. Independent human answer review remains pending.\n"
        )
    except BaseException as exc:
        REPORT.update(status="failed", error=repr(exc))
        save()
        state(phase="failed", error=repr(exc))
        raise
    finally:
        if CURRENT:
            proc = getattr(CURRENT, "proc", None)
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


if __name__ == "__main__":
    main()
