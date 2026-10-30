import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit():
    import numpy as np

    sys.path.insert(0, str(ROOT / "scripts"))
    from research_v2_full_kl import metrics

    manifest = json.loads((ROOT / "evidence/manifest.json").read_text())
    for name, expected in manifest.items():
        if sha((ROOT / name).read_bytes()) != expected:
            raise ValueError("Frozen evidence changed: " + name)
    lineage = json.loads((ROOT / "assets/source-lineage.json").read_text())
    for item in lineage:
        assert sha((ROOT / item["release_file"]).read_bytes()) == item["release_sha256"]
    original = {item["release_file"]: item["original_sha256"] for item in lineage}
    pconfig = json.loads((ROOT / "assets/legacy/prefill.json").read_text())
    vconfig = json.loads((ROOT / "assets/legacy/validation.json").read_text())
    dconfig = json.loads((ROOT / "evidence/distributed/protocol.json").read_text())
    assert (
        original["native/route_worker.cpp"]
        == dconfig["hashes"]["experiments/research-v2/route_worker.cpp"]
    )
    assert original["native/prefill_worker.cpp"] == pconfig["sources"]["worker.cpp"]
    assert (
        original["scripts/legacy/prefill/worker.py"] == pconfig["sources"]["worker.py"]
    )
    assert original["scripts/legacy/validation/run.py"] == vconfig["sources"]["run.py"]
    subprocess.run(
        [sys.executable, str(ROOT / "analysis/build_research_evidence.py")], check=True
    )
    subprocess.run(
        [sys.executable, str(ROOT / "analysis/build_budget_evidence.py")], check=True
    )
    validation = json.loads((ROOT / "assets/legacy/validation.json").read_text())
    for key, name in [
        ("evaluation_sha256", "evaluation.json"),
        ("exposure_audit_sha256", "exposure-audit.json"),
    ]:
        assert validation[key] == sha((ROOT / "evidence/selection" / name).read_bytes())
    selected = json.loads(
        (ROOT / "evidence/selection/evaluation-gold.json").read_text()
    )["data"]
    lookup = {
        q["id"]: (a["title"], p["context"], q)
        for a in selected
        for p in a["paragraphs"]
        for q in p["qas"]
    }
    assert (
        len(lookup) == 24
        and len({c["document_key"] for c in validation["evaluation"]}) == 24
    )
    for case in validation["evaluation"]:
        title, document, question = lookup[case["document_id"]]
        assert (
            title == case["document_key"]
            and document == case["document"]
            and (question["question"] == case["question"])
        )
        assert bool(question["is_impossible"]) == case["unsupported"]
    for asset, key, doc in [
        ("prefill", "design_sha256", "prefill.md"),
        ("validation", "protocol_sha256", "backend-and-qa.md"),
    ]:
        config = json.loads((ROOT / f"assets/legacy/{asset}.json").read_text())
        assert config[key] == sha((ROOT / "protocols" / doc).read_bytes())
    versions = json.loads((ROOT / "assets/versions.json").read_text())
    for name, key in [
        ("coqa-evaluator.py.gz", "coqa_evaluator_sha256"),
        ("squad-evaluator.py.gz", "official_evaluator_sha256"),
    ]:
        assert (
            sha(gzip.decompress((ROOT / "assets/official" / name).read_bytes()))
            == versions[key]
        )
    cycles = [
        json.loads((ROOT / f"evidence/distributed/b1/cycle-{i}.json").read_text())
        for i in range(3)
    ]
    assert [round(c["pss_saving_bytes"] / 2**20, 2) for c in cycles] == [
        382.56,
        382.56,
        382.55,
    ]
    assert [round(c["node_available_delta"] / 2**20, 2) for c in cycles] == [
        -5.82,
        -1.12,
        -11.4,
    ]
    logits = {}
    for entry in json.loads(
        (ROOT / "evidence/distributed/logits-manifest.json").read_text()
    ):
        raw = gzip.decompress((ROOT / entry["path"]).read_bytes())
        assert sha(raw) == entry["raw_sha256"]
        logits[Path(entry["path"]).name.removesuffix(".npy.gz")] = np.load(
            io.BytesIO(raw), allow_pickle=False
        )
    cohort = json.loads((ROOT / "assets/legacy/transition-cohort.json").read_text())
    for i, document in enumerate(cohort):
        for mode in ("q4", "q3", "retained", "rebuild"):
            row = json.loads(
                (ROOT / f"evidence/distributed/b2/d{i}-{mode}.json").read_text()
            )
            values = logits[f"d{i}-{mode}"]
            baseline = logits[f"d{i}-q4"]
            assert (
                sha(
                    gzip.decompress(
                        (
                            ROOT / f"evidence/distributed/b2/d{i}-{mode}.npy.gz"
                        ).read_bytes()
                    )
                )
                == row["logits_sha256"]
            )
            assert len(values) == len(document["answer"]) == len(row["tokens"])
            for a, b, target, saved in zip(
                baseline, values, document["answer"], row["tokens"]
            ):
                result = metrics(a, b, target)
                assert (
                    abs(
                        result["kl_reference_to_candidate"]
                        - saved["kl_reference_to_candidate"]
                    )
                    < 1e-10
                )
                assert result["top1_agreement"] == saved["top1_agreement"]
    summary = json.loads((ROOT / "evidence/distributed/b2/summary.json").read_text())[
        "scores"
    ]
    for mode in ("q4", "q3", "retained", "rebuild"):
        rows = [
            json.loads((ROOT / f"evidence/distributed/b2/d{i}-{mode}.json").read_text())
            for i in range(3)
        ]
        for metric, key in [
            ("elapsed_s", "mean_elapsed_s"),
            ("f1", "f1"),
            ("em", "em"),
        ]:
            assert (
                abs(statistics.mean(row[metric] for row in rows) - summary[mode][key])
                < 1e-9
            )
    memory_figure = (ROOT / "analysis/figures/memory.tex").read_text()
    for i, cycle in enumerate(cycles, 1):
        assert f"(Cycle {i},{cycle['pss_saving_bytes']/2**20:.2f})" in memory_figure
        assert f"(Cycle {i},{cycle['node_available_delta']/2**20:.2f})" in memory_figure
    assert abs(summary["retained"]["mean_document_kl"] - 0.08349959458876481) < 1e-12
    decode = json.loads((ROOT / "evidence/decode/report.json").read_text())
    rows = [r for r in decode["cases"] if r["batch"] == 4]
    assert len(rows) == 2 and all(
        (r["update"]["embedded_unique_chunks"] == 81 for r in rows)
    )
    assert [round(r["update"]["publish_total_s"], 2) for r in rows] == [3.64, 11.77]
    reports = {
        name: json.loads((ROOT / f"analysis/evidence/2026-10-27/{name}.json").read_text())
        for name in ("e1", "e2", "e3")
    }
    policies = {
        (stage, row["cell"]["id"]): row["cell"]["policy"]
        for stage, report in reports.items()
        for row in report["rows"]
    }
    sample_count = 0
    for entry in json.loads((ROOT / "evidence/affinity/manifest.json").read_text()):
        raw = gzip.decompress((ROOT / entry["path"]).read_bytes())
        assert sha(raw) == entry["raw_sha256"]
        lines = raw.splitlines()
        assert len(lines) == entry["rows"]
        sample_count += len(lines)
        policy = policies[entry["stage"], entry["cell"]]
        llm = (
            [0, 1]
            if policy == "partition-2-2"
            else [0, 1, 2] if policy == "partition-3-1" else [0, 1, 2, 3]
        )
        encoder = (
            [2, 3]
            if policy == "partition-2-2"
            else [3] if policy == "partition-3-1" else [0, 1, 2, 3]
        )
        for line in lines:
            sample = json.loads(line)
            for key, expected in [("llm", llm), ("encoder", encoder)]:
                for thread in sample.get(key, {}).get("threads", []):
                    assert thread["cpus"] == expected
    assert sample_count == 37475
    original = json.loads((ROOT / "evidence/e2-original/report.json").read_text())
    e2 = json.loads((ROOT / "analysis/evidence/2026-10-27/e2.json").read_text())
    assert len(original["rows"]) == 17
    for saved, continued in zip(original["rows"], e2["rows"]):
        assert saved == {k: v for k, v in continued.items() if k != "artifact_run"}
    print(
        json.dumps(
            dict(
                status="verified",
                frozen_files=len(manifest),
                affinity_sample_rows=sample_count,
                quality_jobs=12,
                semantic_review="pending",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    audit()
