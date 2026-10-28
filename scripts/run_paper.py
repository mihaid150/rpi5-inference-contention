import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision-config", type=Path, required=True)
    parser.add_argument("--image", default="shared-paper-evaluation:local")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    stages = ["precision", "decode", "prefill", "validation", "e1-e3"]
    if not args.execute:
        print(
            json.dumps(
                dict(
                    execution=False,
                    stages=stages,
                    assets=str(args.assets.resolve()),
                    output=str(args.output.resolve()),
                ),
                indent=2,
            )
        )
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def state(**value):
        temp = output / "status.tmp"
        temp.write_text(
            json.dumps(dict(updated_unix=time.time(), **value), indent=2) + "\n"
        )
        temp.replace(output / "status.json")

    active = None
    try:
        for stage in stages:
            state(phase="running", stage=stage)
            target = output / stage
            if stage == "precision":
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/precision.py"),
                        "--config",
                        str(args.precision_config.resolve()),
                        "--assets",
                        str(args.assets.resolve()),
                        "--output",
                        str(target),
                        "--execute",
                    ],
                    check=True,
                )
            else:
                name = "paper-" + stage + "-" + str(time.time_ns())
                active = name
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/launch.py"),
                        "--stage",
                        stage,
                        "--assets",
                        str(args.assets.resolve()),
                        "--output",
                        str(target),
                        "--image",
                        args.image,
                        "--name",
                        name,
                    ],
                    check=True,
                )
                deadline = time.monotonic() + 72 * 3600
                while not (target / "DONE").is_file():
                    probe = subprocess.run(
                        ["docker", "inspect", "--format", "{{.State.Running}}", name],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if probe.returncode != 0 or probe.stdout.strip() != "true":
                        if not (target / "DONE").is_file():
                            raise RuntimeError(
                                stage
                                + " stopped without a successful marker; retain partial outputs"
                            )
                    if time.monotonic() > deadline:
                        subprocess.run(["docker", "stop", name], check=True, timeout=60)
                        raise TimeoutError(stage + " exceeded 72 hours")
                    time.sleep(5)
            if not (target / "DONE").is_file():
                raise RuntimeError(stage + " did not complete")
            active = None
        state(phase="completed", semantic_review="pending")
        (output / "DONE").write_text(
            "All paper experiment families completed; semantic review remains separate.\n"
        )
    except BaseException as error:
        if active:
            subprocess.run(
                ["docker", "stop", active], capture_output=True, text=True, timeout=60
            )
        state(phase="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
