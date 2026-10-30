import argparse
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "evidence/2026-10-24-final"


def derive():
    p = json.loads((SOURCE / "prefill.json").read_text())
    v = json.loads((SOURCE / "validation.json").read_text())
    assert p["status"] == v["status"] == "completed"
    assert (
        len(p["rows"]) == 112 and len(v["rows"]) == 108 and (len(v["episodes"]) == 12)
    )
    outputs = {}
    manifest = dict(
        sources={
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in SOURCE.glob("*.json")
            if f.name in ("prefill.json", "validation.json")
        }
    )
    baseline = []
    names = [
        ("server-512-default", "Standard"),
        ("prototype-full", "Full prototype"),
        ("prototype-skip4", "Skip4 prototype"),
    ]
    lines = []
    for arm, label in names:
        values = []
        for fixture in ("dev-shards", "eval-original-u3"):
            rows = [
                r
                for r in v["rows"]
                if r["stage"] == "baseline"
                and r["arm"] == arm
                and (r["fixture"] == fixture)
            ]
            assert len(rows) == 3 and all(
                (r["assessment"]["strict_passed"] for r in rows)
            )
            m = st.median((r["total_request_s"] for r in rows))
            values.append(m)
            baseline.append(
                dict(
                    arm=arm,
                    fixture=fixture,
                    n=3,
                    median=m,
                    minimum=min((r["total_request_s"] for r in rows)),
                    maximum=max((r["total_request_s"] for r in rows)),
                )
            )
        lines.append(f"{label} & {values[0]:.1f} & {values[1]:.1f} " + chr(92) * 2)
    outputs["baseline-rows.tex"] = "\n".join(lines + ["\\bottomrule"]) + "\n"
    ladder = []
    for n in (128, 256, 512, 1024):
        row = [str(n)]
        for arm in ("raw-mb2", "repack-mb64"):
            rows = [
                r
                for r in p["rows"]
                if r["stage"] == "ladder"
                and r["arm"] == arm
                and (r["prompt_tokens"] == n)
            ]
            assert len(rows) == 3
            row.append(f"{st.median((r['prefill_compute_s'] for r in rows)):.9f}")
        ladder.append(" ".join(row))
    outputs["ladder.dat"] = "tokens raw packed\n" + "\n".join(ladder) + "\n"
    bars = []
    for arm, label in names:
        s, l = [r for r in baseline if r["arm"] == arm]
        bars.append(f"{dict(names).get(arm)}")
    outputs["latency.dat"] = (
        "id short long\n"
        + "\n".join(
            (
                f"{i} {baseline[2 * i]['median']:.9f} {baseline[2 * i + 1]['median']:.9f}"
                for i in range(3)
            )
        )
        + "\n"
    )
    cases = {c["id"]: c for c in v["config"]["evaluation"]}
    quality = []
    matrix = []
    order = list(range(0, 24, 2)) + list(range(1, 24, 2))
    for y, (arm, label) in enumerate(names):
        rows = [
            r for r in v["rows"] if r["stage"] == "new-documents" and r["arm"] == arm
        ]
        assert len(rows) == 24 and len({r["fixture"] for r in rows}) == 24
        lookup = {r["fixture"]: r for r in rows}
        ans = [r for r in rows if not cases[r["fixture"]]["unsupported"]]
        noans = [r for r in rows if cases[r["fixture"]]["unsupported"]]
        stats = dict(
            arm=arm,
            f1=st.mean((r["assessment"]["f1"] for r in rows)),
            em=st.mean((r["assessment"]["em"] for r in rows)),
            answerable_pass=sum(
                (r["assessment"]["automatic_supported_correct"] for r in ans)
            ),
            unanswerable_pass=sum(
                (r["assessment"]["automatic_supported_correct"] for r in noans)
            ),
        )
        quality.append(stats)
        for x, i in enumerate(order):
            r = lookup[f"newdoc-{i:02d}"]
            case = cases[r["fixture"]]
            color = (
                "passgreen"
                if r["assessment"]["automatic_supported_correct"]
                else (
                    "formatamber"
                    if case["unsupported"]
                    and r["text"].strip().lower().startswith("unknown")
                    else "failgray"
                )
            )
            matrix.append(
                f"\\fill[{color}] ({x * 0.235:.3f},{-y * 0.4:.3f}) rectangle ++(.208,.27);"
            )
    outputs["quality-cells.tex"] = "\n".join(matrix) + "\n"
    outputs["quality-rows.tex"] = (
        "\n".join(
            (
                f"{label} & {q['f1']:.3f} & {q['answerable_pass']}/12 & {q['unanswerable_pass']}/12 "
                + chr(92) * 2
                for (_, label), q in zip(names, quality)
            )
        )
        + "\n\\bottomrule\n"
    )
    timeline = []
    app = []
    policies = [
        ("standard-concurrent", "Standard concurrent"),
        ("standard-deferred", "Standard deferred"),
        ("prototype-full", "Full prototype"),
        ("prototype-skip4", "Skip4 prototype"),
    ]
    for i, (name, label) in enumerate(policies):
        rows = [r for r in v["episodes"] if r["policy"] == name]
        assert len(rows) == 3
        metrics = {
            k: st.median((r[k] for r in rows))
            for k in ("current_answer_s", "publication_s", "fresh_answer_s")
        }
        app.append(
            dict(
                policy=name,
                **metrics,
                valid=sum(
                    (
                        r["current_passed"]
                        and r["retrieval_passed"]
                        and r["fresh_passed"]
                        for r in rows
                    )
                ),
            )
        )
        r = next((r for r in rows if r["repeat"] == 1))
        end = r["fresh_answer_s"]
        begin = end - r["fresh_output"]["total_request_s"]
        current = r["current_answer_s"]
        pub = r["publication_s"]
        u0 = current if name == "standard-deferred" else 0
        y = -i * 1.12
        timeline += [
            f"\\node[anchor=east,font=\\scriptsize] at (-.18,{y - 0.12:.3f}) {{{label}}};",
            f"\\fill[answerblue] (0,{y:.3f}) rectangle ({current * 0.054:.4f},{y + 0.2:.3f});",
            f"\\fill[updatepurple] ({u0 * 0.054:.4f},{y - 0.25:.3f}) rectangle ({pub * 0.054:.4f},{y - 0.05:.3f});",
            f"\\draw[black!45,densely dotted] ({max(current, pub) * 0.054:.4f},{y + 0.1:.3f}) -- ({begin * 0.054:.4f},{y + 0.1:.3f});",
            f"\\fill[freshorange] ({begin * 0.054:.4f},{y:.3f}) rectangle ({end * 0.054:.4f},{y + 0.2:.3f});",
            f"\\node[anchor=west,font=\\scriptsize] at ({end * 0.054 + 0.1:.4f},{y + 0.1:.3f}) {{{end:.1f}s}};",
            f"\\draw[updatepurple] ({pub * 0.054:.4f},{y - 0.3:.3f}) -- ({pub * 0.054:.4f},{y - 0.46:.3f});",
            f"\\node[anchor=north,font=\\scriptsize,text=updatepurple] at ({pub * 0.054:.4f},{y - 0.46:.3f}) {{{pub:.1f}s}};",
        ]
        if not r["fresh_passed"]:
            timeline.append(
                f"\\node[anchor=west,font=\\scriptsize,text=red!70!black] at ({begin * 0.054 + 0.1:.4f},{y - 0.29:.3f}) {{correct fact; citation-format failure}};"
            )
    outputs["episode-timeline.tex"] = "\n".join(timeline) + "\n"
    outputs["application-rows.tex"] = (
        "\n".join(
            (
                f"{label} & {r['current_answer_s']:.1f} & {r['publication_s']:.1f} & {r['fresh_answer_s']:.1f} "
                + chr(92) * 2
                for (_, label), r in zip(policies, app)
            )
        )
        + "\n\\bottomrule\n"
    )
    manifest.update(
        baseline=baseline,
        quality=quality,
        application=app,
        application_design="Three matched episode blocks, each with a different newly added article; no within-document replications",
        limitations="Automatic frozen verdicts retained. Format diagnostics are post hoc and do not reclassify passes.",
    )
    outputs["derived.json"] = json.dumps(manifest, indent=2) + "\n"
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for name, content in derive().items():
        path = ROOT / "generated" / name
        if args.write:
            path.write_text(content)
        else:
            assert path.read_text() == content, f"Manuscript evidence mismatch: {name}"
    print(
        "Verified final report counts, completed cell sizes, tables, diagram coordinates and source digests."
    )
