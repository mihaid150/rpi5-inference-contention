import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import resource
import sqlite3
import tempfile
import time


def retrieval_query(question, history):
    parts = []
    for message in history:
        if message.get("role") in ("user", "assistant"):
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError("History content must be plain text")
            parts.append(content)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Empty question")
    return "\n".join(parts + [question])


def _version(value):
    if not isinstance(value, str) or not re.fullmatch("[A-Za-z0-9_-]+", value):
        raise ValueError("Unsafe version")
    return value


def _vector(values):
    values = [float(v) for v in values]
    if not values or not all((math.isfinite(v) for v in values)):
        raise ValueError("Invalid embedding")
    norm = math.sqrt(sum((v * v for v in values)))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Invalid embedding norm")
    return [v / norm for v in values]


def publish(
    root,
    version,
    documents,
    encoder,
    encoder_id,
    batch_size=4,
    chunk_words=200,
    stride_words=160,
    chunker=None,
):
    started = time.monotonic()
    _version(version)
    documents = list(documents)
    if not encoder_id or batch_size < 1 or (not 0 < stride_words <= chunk_words):
        raise ValueError("Invalid encoder or chunk/batch configuration")
    rows, ids = ([], set())
    for doc in documents:
        source = doc["id"]
        if not isinstance(source, str) or not source or source in ids:
            raise ValueError("Duplicate or empty source id")
        ids.add(source)
        if not isinstance(doc["text"], str):
            raise ValueError("Document text must be plain text")
        if chunker is None:
            words = doc["text"].split()
            fragments = (
                (offset, " ".join(words[offset : offset + chunk_words]))
                for offset in range(0, len(words), stride_words)
            )
        else:
            fragments = enumerate(chunker(doc["text"]))
        for offset, text in fragments:
            digest = hashlib.sha256(text.encode()).hexdigest()
            rows.append((source, offset, digest, text))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "index.sqlite", timeout=30)
    try:
        db.executescript(
            "\n            PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;\n            CREATE TABLE IF NOT EXISTS vectors (\n                encoder TEXT, digest TEXT, text TEXT, vector TEXT,\n                PRIMARY KEY(encoder,digest));\n            CREATE TABLE IF NOT EXISTS versions (version TEXT PRIMARY KEY, meta TEXT);\n            CREATE TABLE IF NOT EXISTS members (\n                version TEXT, source TEXT, offset INTEGER, digest TEXT,\n                PRIMARY KEY(version,source,offset));\n            CREATE TABLE IF NOT EXISTS current (singleton INTEGER PRIMARY KEY CHECK(singleton=1), version TEXT);\n        "
        )
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM versions WHERE version=?", (version,)).fetchone():
            raise ValueError("Version already published")
        missing = {}
        for _, _, digest, text in rows:
            if not db.execute(
                "SELECT 1 FROM vectors WHERE encoder=? AND digest=?",
                (encoder_id, digest),
            ).fetchone():
                missing[digest] = text
        prior = db.execute(
            "SELECT vector FROM vectors WHERE encoder=? LIMIT 1", (encoder_id,)
        ).fetchone()
        dimension = len(json.loads(prior[0])) if prior else None
        items = list(missing.items())
        embedding_start = time.monotonic()
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            vectors = list(encoder([text for _, text in batch]))
            if len(vectors) != len(batch):
                raise ValueError("Encoder output count mismatch")
            for (digest, text), values in zip(batch, vectors):
                vector = _vector(values)
                if dimension is None:
                    dimension = len(vector)
                if len(vector) != dimension:
                    raise ValueError("Encoder dimension changed")
                db.execute(
                    "INSERT INTO vectors VALUES (?,?,?,?)",
                    (encoder_id, digest, text, json.dumps(vector)),
                )
        embedding_s = time.monotonic() - embedding_start
        db.executemany(
            "INSERT INTO members VALUES (?,?,?,?)",
            [(version, source, offset, digest) for source, offset, digest, _ in rows],
        )
        meta = dict(
            version=version,
            encoder_id=encoder_id,
            documents=len(ids),
            chunks=len(rows),
            embedded_unique_chunks=len(items),
            chunk_words=chunk_words,
            stride_words=stride_words,
            chunking=chunker.identity if chunker else dict(method="legacy-word-window"),
            chunk_locator="ordinal" if chunker else "word-offset",
            embedding_s=embedding_s,
            corpus_sha256=hashlib.sha256(
                json.dumps(documents, sort_keys=True).encode()
            ).hexdigest(),
        )
        db.execute("INSERT INTO versions VALUES (?,?)", (version, json.dumps(meta)))
        db.execute("INSERT OR REPLACE INTO current VALUES (1,?)", (version,))
        db.commit()
        return dict(meta, publish_total_s=time.monotonic() - started)
    finally:
        db.close()


def retrieve(
    root,
    question,
    history,
    encoder,
    encoder_id,
    version=None,
    k=3,
    section_selection=False,
):
    if k < 1:
        raise ValueError("k must be positive")
    path = (Path(root) / "index.sqlite").resolve()
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        db.execute("BEGIN")
        if version is None:
            row = db.execute("SELECT version FROM current WHERE singleton=1").fetchone()
            if not row:
                raise ValueError("No published version")
            version = row[0]
        _version(version)
        row = db.execute(
            "SELECT meta FROM versions WHERE version=?", (version,)
        ).fetchone()
        if not row:
            raise ValueError("Version not published")
        if json.loads(row[0])["encoder_id"] != encoder_id:
            raise ValueError("Query encoder differs from index")
        scope = None
        focus = question
        if section_selection:
            from research_u3_sections import (
                scope_from_question,
                answer_focus,
                diversify,
            )

            scope = scope_from_question(question)
            focus = answer_focus(question)
        outputs = list(encoder([retrieval_query(focus, history)]))
        if len(outputs) != 1:
            raise ValueError("Query encoder output count mismatch")
        query = _vector(outputs[0])
        scope_vector = None
        if scope:
            scoped = list(encoder([scope]))
            if len(scoped) != 1:
                raise ValueError("Scope encoder output count mismatch")
            scope_vector = _vector(scoped[0])
            if len(scope_vector) != len(query):
                raise ValueError("Scope dimension differs from query")

        def candidates():
            for source, offset, text, raw in db.execute(
                "\n                SELECT m.source,m.offset,v.text,v.vector FROM members m JOIN vectors v\n                ON m.digest=v.digest AND v.encoder=? WHERE m.version=?",
                (encoder_id, version),
            ):
                vector = json.loads(raw)
                if len(query) != len(vector):
                    raise ValueError("Query dimension differs from index")
                yield dict(
                    source_id=source,
                    chunk=offset,
                    text=text,
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    scope_score=(
                        sum((a * b for a, b in zip(scope_vector, vector)))
                        if scope_vector
                        else None
                    ),
                    score=sum((a * b for a, b in zip(query, vector))),
                )

        exact_title = (
            scope
            if scope and re.search('\\bsection\\s+["\\u201c]', question, re.I)
            else None
        )
        ranked = (
            diversify(candidates(), k, exact_title=exact_title)
            if scope
            else heapq.nsmallest(
                k, candidates(), key=lambda p: (-p["score"], p["source_id"], p["chunk"])
            )
        )
        return dict(
            version=version,
            passages=ranked,
            section_scope=scope,
            selection="section-diverse-v1" if scope else "dense-top-k",
            section_resolved=bool(ranked) if exact_title else None,
        )
    finally:
        db.close()


def pilot_gate(evidence):
    requirements = (
        "contract_approved",
        "retrieval_qualified",
        "backend_qualified",
        "baseline_latency_qualified",
        "shared_memory_domain_verified",
        "memory_conflict_observed",
        "q4_small_batch_profiled",
        "encoder_identity_pinned",
        "independent_development_episodes",
        "infrastructure_stable",
        "wall_time_budget_approved",
    )
    missing = [name for name in requirements if evidence.get(name) is not True]
    return dict(
        status="blocked" if missing else "ready_for_pilot_review",
        missing=missing,
        launch_authorized=False,
    )


def profile_update(root, version, documents, encoder, encoder_id, batch_size=4):
    before = resource.getrusage(resource.RUSAGE_SELF)
    result = publish(root, version, documents, encoder, encoder_id, batch_size)
    after = resource.getrusage(resource.RUSAGE_SELF)
    return dict(
        result,
        batch_size=batch_size,
        cpu_user_s=after.ru_utime - before.ru_utime,
        cpu_system_s=after.ru_stime - before.ru_stime,
        minor_faults=after.ru_minflt - before.ru_minflt,
        major_faults=after.ru_majflt - before.ru_majflt,
        process_lifetime_maxrss_platform_units=after.ru_maxrss,
        shared_memory_qualified=False,
    )


def smoke():

    def encoder(texts):
        return [
            [1 + text.lower().count("network"), 1 + text.lower().count("storage")]
            for text in texts
        ]

    with tempfile.TemporaryDirectory(prefix="dli-u3-smoke-") as root:
        docs = [
            dict(id="network", text="network endpoint old"),
            dict(id="storage", text="storage procedure"),
        ]
        first = publish(root, "v1", docs, encoder, "synthetic-test-only")
        docs[0]["text"] = "network endpoint new"
        second = publish(root, "v2", docs, encoder, "synthetic-test-only")
        old = retrieve(
            root, "network", [], encoder, "synthetic-test-only", version="v1"
        )
        new = retrieve(root, "network", [], encoder, "synthetic-test-only")
        assert first["embedded_unique_chunks"] == 2
        assert second["embedded_unique_chunks"] == 1
        assert (
            "old" in old["passages"][0]["text"] and "new" in new["passages"][0]["text"]
        )
        return dict(
            status="smoke_passed", qualification=False, first=first, second=second
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["smoke", "gate"])
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.action == "smoke":
        print(json.dumps(smoke(), indent=2))
    else:
        result = pilot_gate(
            json.loads(args.evidence.read_text()) if args.evidence else {}
        )
        print(json.dumps(result, indent=2))
        raise SystemExit(2 if result["status"] == "blocked" else 0)
