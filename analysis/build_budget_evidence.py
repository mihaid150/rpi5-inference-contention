import hashlib, json, statistics
from pathlib import Path

P = Path(__file__).resolve().parent
E = P / "evidence/2026-10-27"
reports = [
    json.loads((E / (name + ".json")).read_text()) for name in ("e1", "e2", "e3")
]
audit = json.loads((E / "audit.json").read_text())
for name, report, count, run in zip(
    ("e1", "e2", "e3"), reports, (27, 36, 12), audit["source_sha256"]
):
    assert report["status"] == "completed" and len(report["rows"]) == count
    assert (
        hashlib.sha256((E / (name + ".json")).read_bytes()).hexdigest()
        == audit["source_sha256"][run]
    )
assert (
    hashlib.sha256((E / "predictions.json").read_bytes()).hexdigest()
    == audit["prediction_sha256"]
)
assert reports[2]["prediction_freeze"]["sha256"] == audit["prediction_sha256"]
data = ["phase shared partition"]
for field, label in [("prompt_ms", "Prefill"), ("predicted_ms", "Decode")]:
    vals = []
    for policy in ("shared", "partition-2-2"):
        rs = [
            r
            for r in reports[0]["rows"]
            if r["cell"]["policy"] == policy
            and r["cell"]["fixture"] == "eval-original-u3"
        ]
        vals.append(
            statistics.median((r["inference"]["server_timings"][field] for r in rs))
            / 1000
        )
    data.append(f"{label} {vals[0]:.6f} {vals[1]:.6f}")
vals = []
for policy in ("shared", "partition-2-2"):
    rs = [
        r
        for r in reports[0]["rows"]
        if r["cell"]["policy"] == policy and r["cell"]["fixture"] == "eval-original-u3"
    ]
    vals.append(statistics.median((r["update"]["duration_s"] for r in rs)))
data.append(f"Update {vals[0]:.6f} {vals[1]:.6f}")
expected = "\n".join(data) + "\n"
target = P / "generated/budget-phases.dat"
import sys

if "--write" in sys.argv:
    target.write_text(expected)
else:
    assert target.read_text() == expected, "Figure data differs from frozen reports"
print("Verified E1/E2/E3 counts, source/prediction digests and phase figure data.")
