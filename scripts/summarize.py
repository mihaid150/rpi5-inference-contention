import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(root):
    reports = {
        stage: json.loads((root / stage / "report.json").read_text())
        for stage in ("e1", "e2", "e3")
    }
    summary = {}
    for stage, report in reports.items():
        if report["status"] != "completed":
            raise ValueError(f"Incomplete {stage}")
        groups = {}
        for row in report["rows"]:
            cell = row["cell"]
            factors = {
                k: cell.get(k)
                for k in ("policy", "fixture", "update", "fresh_arrival_s")
            }
            key = json.dumps(factors, sort_keys=True)
            group = groups.setdefault(key, dict(factors=factors, observations=[]))
            metrics = dict(row.get("metrics", {}))
            if stage == "e1":
                if "update_complete_s" in row:
                    metrics["update_complete_s"] = row["update_complete_s"]
                if "inference" in row:
                    inf = row["inference"]
                    metrics["answer_complete_s"] = row["answer_end"] - row["origin"]
                    metrics["prefill_s"] = inf["server_timings"]["prompt_ms"] / 1000
                    metrics["decode_s"] = inf["server_timings"]["predicted_ms"] / 1000
            group["observations"].append(metrics)
        results = []
        for group in groups.values():
            observations = group["observations"]
            keys = sorted(
                {
                    k
                    for row in observations
                    for k, v in row.items()
                    if isinstance(v, (int, float)) and (not isinstance(v, bool))
                }
            )
            medians = {
                k: statistics.median((row[k] for row in observations if k in row))
                for k in keys
            }
            results.append(
                dict(factors=group["factors"], count=len(observations), medians=medians)
            )
        summary[stage] = dict(cells=len(report["rows"]), groups=results)
    frozen = (root / "predictions.json").read_bytes()
    expected = hashlib.sha256(frozen).hexdigest()
    if reports["e3"]["prediction_freeze"]["sha256"] != expected:
        raise ValueError("E3 prediction digest mismatch")
    e2_bytes = (root / "e2/report.json").read_bytes()
    if json.loads(frozen)["source_sha256"] != hashlib.sha256(e2_bytes).hexdigest():
        raise ValueError("E2 changed after prediction freeze")
    summary["e3"]["prediction_mae_s"] = {
        key: statistics.mean(
            (
                row["prediction_check"]["absolute_error_s"][key]
                for row in reports["e3"]["rows"]
            )
        )
        for key in reports["e3"]["rows"][0]["prediction_check"]["absolute_error_s"]
    }
    summary["semantic_review"] = (
        "pending; automatic lexical checks are not semantic adjudication"
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    print(json.dumps(summarize(parser.parse_args().output), indent=2))
