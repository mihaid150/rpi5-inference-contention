import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(target, gguf=None):
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=False)
    versions = json.loads((ROOT / "assets/versions.json").read_text())
    for name in ("prior.json", "corpus.json", "versions.json"):
        shutil.copy2(ROOT / "assets" / name, target / name)
    shutil.copy2(
        ROOT / "configs/shared-budget-14b/campaign-2026-10-26.json",
        target / "campaign.json",
    )
    snapshot_download(
        "sentence-transformers/all-MiniLM-L6-v2",
        revision=versions["minilm_revision"],
        allow_patterns=list(versions["minilm_files"]),
        local_dir=target / "minilm",
    )
    snapshot_download(
        "Qwen/Qwen2.5-14B-Instruct",
        revision=versions["qwen_revision"],
        allow_patterns=["*.json", "merges.txt", "vocab.json", "LICENSE"],
        local_dir=target / "q4-tokenizer",
    )
    with urllib.request.urlopen(
        "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/evaluate-v2.0.py",
        timeout=60,
    ) as response:
        (target / "official_evaluator.py").write_bytes(response.read())
    if (
        digest(target / "official_evaluator.py")
        != versions["official_evaluator_sha256"]
    ):
        raise ValueError(
            "Official evaluator changed; refusing to use different scoring"
        )
    for name, expected in versions["minilm_files"].items():
        if digest(target / "minilm" / name) != expected:
            raise ValueError(f"MiniLM digest mismatch: {name}")
    if gguf:
        if digest(gguf) != versions["model_sha256"]:
            raise ValueError("This GGUF differs from the measured artifact")
        family = target / "models/family"
        family.mkdir(parents=True)
        shutil.copy2(gguf, family / "Q4_K_M.gguf")
    manifest = {
        str(p.relative_to(target)): digest(p)
        for p in sorted(target.rglob("*"))
        if p.is_file()
        and ".cache" not in p.parts
        and "models" not in p.relative_to(target).parts
    }
    (target / "assets-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gguf", type=Path)
    args = parser.parse_args()
    prepare(args.output, args.gguf)
