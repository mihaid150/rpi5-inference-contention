import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "shared_budget_14b"))
from design import diagnosis_rows, prediction_rows, regime_rows
from predictions import estimate, fit


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def freeze(e2, plan):
    phases = fit(e2["rows"])
    return dict(
        created_unix=time.time(),
        phases=phases,
        training="E2 incremental only; no E3 timings or answers",
        predictions=[
            dict(cell=c, prediction=estimate(c["fresh_arrival_s"], phases[c["policy"]]))
            for c in prediction_rows(plan)
        ],
    )


def preflight():
    if platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("The measured deployment requires Linux ARM64")
    cpus = sorted(os.sched_getaffinity(0))
    quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    if len(cpus) != 4 or quota == "max" or int(quota) != 4 * int(period):
        raise RuntimeError("Exactly four allowed CPUs and a four-core quota required")
    if Path("/sys/fs/cgroup/memory.max").read_text().strip() != str(12 * 1024**3):
        raise RuntimeError("Memory cap must be 12 GiB")
    versions = json.loads(Path("/input/versions.json").read_text())
    required = {
        Path("/models/family/Q4_K_M.gguf"): versions["model_sha256"],
        Path("/input/official_evaluator.py"): versions["official_evaluator_sha256"],
    }
    required.update(
        {
            Path("/input/minilm") / name: value
            for name, value in versions["minilm_files"].items()
        }
    )
    manifest = json.loads(Path("/input/assets-manifest.json").read_text())
    required.update({Path("/input") / name: value for name, value in manifest.items()})
    for path, expected in required.items():
        if digest(path) != expected:
            raise RuntimeError(f"Asset digest mismatch: {path}")
    return versions


def main():
    output = Path("/output")
    if any(output.iterdir()):
        raise RuntimeError(
            "Output must be empty; existing attempts are never overwritten"
        )
    state = dict(phase="preflight", started_unix=time.time())
    atomic(output / "status.json", state)
    try:
        versions = preflight()
        atomic(output / "versions.json", versions)
        inventory = dict(
            python=sys.version,
            platform=platform.platform(),
            uname=list(platform.uname()),
            backend_sha256=digest("/standard/llama-server"),
            code_sha256={
                str(p.relative_to("/app")): digest(p)
                for p in Path("/app/scripts").rglob("*.py")
            },
            dependencies=subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True
            ),
        )
        atomic(output / "environment.json", inventory)
        plan = json.loads(Path("/input/campaign.json").read_text())
        stages = [
            ("E1", "worker.py", diagnosis_rows(plan)),
            ("E2", "worker_e2.py", regime_rows(plan)),
            ("E3", "worker_e2.py", prediction_rows(plan)),
        ]
        for stage, script, cells in stages:
            target = output / stage.lower()
            target.mkdir()
            state.update(phase="running", stage=stage, expected_cells=len(cells))
            atomic(output / "status.json", state)
            env = dict(
                os.environ,
                EXPERIMENT_STAGE=stage,
                DLI_OUTPUT=str(target),
                DLI_FREEZE=str(output),
            )
            if stage != "E1":
                subprocess.run(
                    [sys.executable, "/poc-input/qualify_e2.py"],
                    env=env,
                    check=True,
                    timeout=3600,
                )
            with (target / "worker.log").open("w") as log:
                subprocess.run(
                    [sys.executable, "-u", "/poc-input/" + script],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=86400,
                )
            report_path = target / "report.json"
            report = json.loads(report_path.read_text())
            if (
                report["status"] != "completed"
                or [r["cell"] for r in report["rows"]] != cells
            ):
                raise RuntimeError(f"Incomplete stage: {stage}")
            if stage == "E2":
                prediction = freeze(report, plan)
                prediction.update(
                    source_sha256=digest(report_path),
                    plan_sha256=digest("/input/campaign.json"),
                )
                atomic(output / "predictions.json", prediction)
                lock = dict(sha256=digest(output / "predictions.json"))
                atomic(output / "prediction-lock.json", lock)
            (target / "DONE").write_text(
                "All scheduled cells completed. Semantic review pending.\n"
            )
        subprocess.run(
            [sys.executable, "/app/scripts/summarize.py", str(output)], check=True
        )
        state.update(
            phase="completed", finished_unix=time.time(), semantic_review="pending"
        )
        atomic(output / "status.json", state)
        (output / "DONE").write_text(
            "E1, E2, E3 completed and summarized. Semantic review pending.\n"
        )
    except BaseException as error:
        state.update(phase="failed", error=repr(error), finished_unix=time.time())
        atomic(output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
