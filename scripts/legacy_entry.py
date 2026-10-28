import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, "/app/scripts")
from campaign import atomic, preflight, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["decode", "prefill", "validation"])
    args = parser.parse_args()
    output = Path("/output")
    if any(output.iterdir()):
        raise RuntimeError("Choose a fresh output directory")
    state = dict(stage=args.stage, phase="preflight", started_unix=time.time())
    atomic(output / "status.json", state)
    try:
        versions = preflight()
        atomic(output / "versions.json", versions)
        binaries = [
            Path("/standard/llama-server"),
            Path("/native/qwen-release-worker"),
            Path("/native/u3-microbatch2-worker"),
            Path("/poc/attention-worker"),
            Path("/reference/attention-worker"),
        ]
        atomic(
            output / "execution-provenance.json",
            dict(
                python=sys.version,
                platform=platform.platform(),
                binaries={str(p): digest(p) for p in binaries},
                code={
                    str(p.relative_to("/app")): digest(p)
                    for p in Path("/app/scripts").rglob("*.py")
                },
                dependencies=subprocess.check_output(
                    [sys.executable, "-m", "pip", "freeze"], text=True
                ),
                configuration_note="Input source/binary manifests describe the historical frozen protocol; these execution hashes identify this reproduction.",
            ),
        )
        if args.stage == "decode":
            os.environ["U3_ACCOUNTED_MODEL"] = "1"
            os.environ["U3_BATCHES"] = "1,4,8"
            script = "/app/scripts/legacy/decode/research_u3_shared_profile.py"
        else:
            script = (
                "/app/scripts/legacy/"
                + args.stage
                + ("/run.py" if args.stage == "validation" else "/worker.py")
            )
        sys.path.insert(0, str(Path(script).parent))
        state.update(phase="running")
        atomic(output / "status.json", state)
        runpy.run_path(script, run_name="__main__")
        report_path = output / (
            "shared-profile/report.json" if args.stage == "decode" else "report.json"
        )
        report = json.loads(report_path.read_text())
        if report["status"] not in (
            "completed",
            "completed_screening_not_causal_proof",
        ):
            raise RuntimeError("Worker did not complete")
        if args.stage == "prefill" and len(report["rows"]) != 112:
            raise RuntimeError("Prefill campaign stopped at a gate; retain all outputs")
        if args.stage == "validation" and (
            len(report["rows"]) != 108 or len(report["episodes"]) != 12
        ):
            raise RuntimeError("Validation campaign incomplete")
        state.update(
            phase="completed", finished_unix=time.time(), semantic_review="pending"
        )
        atomic(output / "completion.json", state)
        atomic(output / "status.json", state)
        (output / "DONE").write_text(
            "Stage completed; semantic adjudication remains separate.\n"
        )
    except BaseException as error:
        state.update(phase="failed", error=repr(error), finished_unix=time.time())
        atomic(output / "completion.json", state)
        atomic(output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
