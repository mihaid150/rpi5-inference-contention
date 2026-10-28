import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(target, gguf=None, q3=None):
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
    for archive, name, key in [
        ("squad-evaluator.py.gz", "official_evaluator.py", "official_evaluator_sha256"),
        ("coqa-evaluator.py.gz", "coqa_evaluator.py", "coqa_evaluator_sha256"),
    ]:
        raw = gzip.decompress((ROOT / "assets/official" / archive).read_bytes())
        if hashlib.sha256(raw).hexdigest() != versions[key]:
            raise ValueError("Official evaluator archive changed: " + archive)
        (target / name).write_bytes(raw)
    for name, expected in versions["minilm_files"].items():
        if digest(target / "minilm" / name) != expected:
            raise ValueError("MiniLM digest mismatch: " + name)
    for source, name in [
        ("prefill.json", "prefill-config.json"),
        ("validation.json", "validation-config.json"),
        ("decode-request.json", "request.json"),
        ("transition-cohort.json", "transition-cohort.json"),
        ("coqa-documents.json", "coqa-documents.json"),
    ]:
        shutil.copy2(ROOT / "assets/legacy" / source, target / name)
    for model, name, key in [
        (gguf, "Q4_K_M.gguf", "model_sha256"),
        (q3, "Q3_K_M.gguf", "q3_model_sha256"),
    ]:
        if model:
            if digest(model) != versions[key]:
                raise ValueError("GGUF differs from the measured artifact: " + name)
            family = target / "models/family"
            family.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model, family / name)
    manifest = {
        str(p.relative_to(target)): digest(p)
        for p in sorted(target.rglob("*"))
        if p.is_file()
        and ".cache" not in p.parts
        and ("models" not in p.relative_to(target).parts)
    }
    (target / "assets-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--q3", type=Path)
    args = parser.parse_args()
    prepare(args.output, args.gguf, args.q3)
