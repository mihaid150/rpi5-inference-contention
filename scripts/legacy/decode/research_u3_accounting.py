import hashlib
import os
from pathlib import Path

MODEL_SHA = "8123a526b3c9b9eeda6cb1dd3af78bd1de0da9af2cef8259f52c3e0a8cd3cebd"


def counters(raw):
    return {k: int(v) for k, v in (line.split() for line in raw.splitlines())}


def charge_gate(before, after, size):
    delta = (
        counters(after["memory.stat"])["file"] - counters(before["memory.stat"])["file"]
    )
    return dict(
        file_charge_delta_bytes=delta,
        model_bytes=size,
        minimum_fraction=0.9,
        passed=delta >= size * 0.9,
    )


def residency_gate(snapshot, model_rss, size):
    charged = counters(snapshot["memory.stat"])["file"]
    return dict(
        model_rss_bytes=model_rss,
        model_bytes=size,
        file_charge_bytes=charged,
        minimum_fraction=0.9,
        measurement_phase="after_prefill",
        passed=model_rss >= size * 0.9 and charged >= model_rss * 0.9,
    )


def copy_model(source, target):
    source = Path(source)
    target = Path(target)
    digest = hashlib.sha256()
    with source.open("rb") as src, target.open("xb") as dst:
        while block := src.read(4 * 1024 * 1024):
            digest.update(block)
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    if digest.hexdigest() != MODEL_SHA:
        raise ValueError("Model copy hash mismatch")
    return dict(
        sha256=digest.hexdigest(),
        bytes=target.stat().st_size,
        inode=target.stat().st_ino,
        source_inode=source.stat().st_ino,
    )
