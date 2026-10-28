import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source, llama, output):
    versions = json.loads((ROOT / "assets/versions.json").read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(llama), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != versions["runtime_revision"]:
        raise ValueError("llama.cpp revision differs from the frozen runtime")
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < 70_000_000_000:
        raise RuntimeError("At least 70 GB free disk required")
    corpus = ROOT / "assets/calibration.txt"
    if digest(corpus) != versions["calibration_sha256"]:
        raise ValueError("Calibration corpus changed")
    commands = [
        [
            sys.executable,
            str(llama / "convert_hf_to_gguf.py"),
            str(source),
            "--outfile",
            str(output / "model-f16.gguf"),
            "--outtype",
            "f16",
            "--use-temp-file",
        ],
        [
            str(llama / "build/bin/llama-imatrix"),
            "-m",
            str(output / "model-f16.gguf"),
            "-f",
            str(corpus),
            "-c",
            "512",
            "-b",
            "512",
            "-ub",
            "512",
            "-t",
            "1",
            "-tb",
            "1",
            "-ngl",
            "0",
            "--no-repack",
            "--parse-special",
            "--no-ppl",
            "-o",
            str(output / "imatrix.gguf"),
        ],
        [
            str(llama / "build/bin/llama-quantize"),
            "--imatrix",
            str(output / "imatrix.gguf"),
            "--token-embedding-type",
            "q4_K",
            "--output-tensor-type",
            "q4_K",
            str(output / "model-f16.gguf"),
            str(output / "Q4_K_M.gguf"),
            "Q4_K_M",
            "1",
        ],
    ]
    for index, command in enumerate(commands):
        with (output / f"step-{index}.log").open("w") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=48 * 3600,
            )
    hashes = {
        name: digest(output / name)
        for name in ("model-f16.gguf", "imatrix.gguf", "Q4_K_M.gguf")
    }
    (output / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    if hashes["Q4_K_M.gguf"] != versions["model_sha256"]:
        raise RuntimeError(
            "Rebuilt model differs from the recorded artifact. Retain logs and hashes; do not relabel as an exact reproduction."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.llama.resolve(), args.output.resolve())
