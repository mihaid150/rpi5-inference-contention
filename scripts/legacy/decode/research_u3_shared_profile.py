import json
import os
from pathlib import Path
import subprocess
import threading
import time
from research_u3 import publish
from research_u3_encoder import MiniLMEncoder
from research_u3_chunking import StableChunker
from research_u3_binary import logits
from research_u3_accounting import copy_model, charge_gate, residency_gate


def snapshot():
    fields = {}
    for name in (
        "memory.current",
        "memory.peak",
        "memory.max",
        "memory.events",
        "memory.stat",
        "memory.pressure",
        "cpu.stat",
        "cpu.pressure",
        "io.stat",
    ):
        path = Path("/sys/fs/cgroup") / name
        fields[name] = path.read_text() if path.exists() else None
    fields["node_meminfo"] = Path("/proc/meminfo").read_text()
    fields["at"] = time.monotonic()
    return fields


def process_memory(pid):
    base = Path("/proc") / str(pid)
    result = {"pid": pid}
    for name in ("smaps_rollup", "status", "cgroup"):
        result[name] = (base / name).read_text()
    raw = (base / "smaps").read_text()
    blocks = []
    current = []
    import re

    for line in raw.splitlines():
        if re.match("^[0-9a-f]+-[0-9a-f]+ ", line):
            if current and ".gguf" in current[0]:
                blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current and ".gguf" in current[0]:
        blocks.append("\n".join(current))
    result["model_mappings"] = blocks
    return result


def main():
    root = Path("/output/shared-profile")
    root.mkdir(exist_ok=False)
    corpus = json.loads(Path("/input/corpus.json").read_text())
    tokens = json.loads(Path("/input/request.json").read_text())["tokens"]
    stop = threading.Event()

    def sample():
        with (root / "telemetry.jsonl").open("w") as stream:
            while not stop.is_set():
                stream.write(json.dumps(snapshot()) + "\n")
                stream.flush()
                stop.wait(0.25)

    sampler = threading.Thread(target=sample)
    sampler.start()
    proc = None
    owned_model = None
    model_path = "/models/family/Q4_K_M.gguf"
    report = {
        "scope": "single real revision development screening",
        "shared_container": True,
        "memory_limit_is_safety_cap_not_natural_device_budget": True,
        "file_cache_ownership_not_reassigned": True,
        "memory_conflict_demonstrated": False,
        "adaptive_qualified": False,
    }
    try:
        if os.environ.get("U3_ACCOUNTED_MODEL") == "1":
            model_path = "/output/accounted-Q4.gguf"
            if Path(model_path).exists():
                raise FileExistsError(model_path)
            owned_model = Path(model_path)
            before_copy = snapshot()
            copy_info = copy_model("/models/family/Q4_K_M.gguf", model_path)
            after_copy = snapshot()
            gate = charge_gate(before_copy, after_copy, copy_info["bytes"])
            report["accounting_preflight"] = dict(
                copy=copy_info,
                before=before_copy,
                after=after_copy,
                copy_cache_diagnostic=gate,
                gate={
                    "passed": False,
                    "reason": "awaiting resident model after prefill",
                },
            )
            report["file_cache_ownership_not_reassigned"] = False
            report["accounting_method"] = (
                "new inode written and read inside same container; original untouched"
            )
        encoder = MiniLMEncoder("/input/minilm")
        chunker = StableChunker(encoder.token_count, encoder.encoder_id)
        report["encoder_identity"] = encoder.identity
        report["cases"] = []
        batches = [int(x) for x in os.environ.get("U3_BATCHES", "1,4,8").split(",")]
        if not batches or any((x not in (1, 4, 8) for x in batches)):
            raise ValueError("Invalid batches")
        for batch in batches:
            for concurrent in (False, True):
                index = root / f"index-{batch}-{int(concurrent)}"
                publish(
                    index,
                    "v1",
                    corpus["snapshots"]["v1"],
                    encoder,
                    encoder.encoder_id,
                    batch_size=batch,
                    chunker=chunker,
                )
                decoder = None
                finished = threading.Event()
                decode = {"tokens": 0, "error": None}
                if concurrent:
                    log = (root / f"worker-{batch}.log").open("w")
                    proc = subprocess.Popen(
                        ["/native/u3-microbatch2-worker", model_path, "4", "2"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=log,
                        bufsize=0,
                        env=dict(os.environ, LD_LIBRARY_PATH="/native"),
                    )
                    deadline = time.monotonic() + 600
                    for i in range(0, len(tokens), 64):
                        values = logits(proc, i, tokens[i : i + 64], 152064, deadline)
                    if owned_model:
                        import re

                        warm_memory = process_memory(proc.pid)
                        warm_snapshot = snapshot()
                        rss = sum(
                            (
                                int(v) * 1024
                                for b in warm_memory["model_mappings"]
                                for v in re.findall("^Rss:\\s+(\\d+)", b, re.M)
                            )
                        )
                        warm_gate = residency_gate(
                            warm_snapshot, rss, owned_model.stat().st_size
                        )
                        print(
                            json.dumps(
                                dict(
                                    phase="resident_accounting_gate",
                                    batch=batch,
                                    **warm_gate,
                                )
                            ),
                            flush=True,
                        )
                        report["accounting_preflight"]["gate"] = warm_gate
                        report.setdefault("accounting_checks", []).append(
                            dict(
                                batch=batch,
                                gate=warm_gate,
                                snapshot=warm_snapshot,
                                process=warm_memory,
                            )
                        )
                        if not warm_gate["passed"]:
                            raise RuntimeError(
                                "Resident model accounting qualification failed"
                            )

                    def generate():
                        nonlocal values
                        try:
                            for i in range(48):
                                t = int(values.argmax())
                                if t in (151643, 151645):
                                    break
                                values = logits(
                                    proc, len(tokens) + i, [t], 152064, deadline
                                )
                                decode["tokens"] += 1
                        except Exception as exc:
                            decode["error"] = repr(exc)
                        finally:
                            finished.set()

                    decoder = threading.Thread(target=generate)
                    decoder.start()
                before = snapshot()
                processes_before = {"encoder": process_memory(os.getpid())}
                if proc:
                    processes_before["q4"] = process_memory(proc.pid)
                if proc and owned_model:
                    import re

                    resident = sum(
                        (
                            int(v) * 1024
                            for b in processes_before["q4"]["model_mappings"]
                            for v in re.findall("^Rss:\\s+(\\d+)", b, re.M)
                        )
                    )
                    if not residency_gate(before, resident, owned_model.stat().st_size)[
                        "passed"
                    ]:
                        raise RuntimeError(
                            "Resident model not sufficiently covered by cgroup file accounting"
                        )
                began = time.monotonic()
                result = publish(
                    index,
                    "v2",
                    corpus["snapshots"]["v2"],
                    encoder,
                    encoder.encoder_id,
                    batch_size=batch,
                    chunker=chunker,
                )
                ended = time.monotonic()
                after = snapshot()
                processes_after = {"encoder": process_memory(os.getpid())}
                if proc:
                    processes_after["q4"] = process_memory(proc.pid)
                overlapped = concurrent and (not finished.is_set())
                if decoder:
                    decoder.join(timeout=600)
                    if decoder.is_alive():
                        raise TimeoutError("Decode did not finish")
                    proc.stdin.close()
                    proc.wait(timeout=20)
                    proc = None
                    log.close()
                    if decode["error"]:
                        raise RuntimeError(decode["error"])
                report["cases"].append(
                    dict(
                        batch=batch,
                        concurrent=concurrent,
                        update=result,
                        update_start=began,
                        update_end=ended,
                        decode=decode,
                        before=before,
                        after=after,
                        processes_before=processes_before,
                        processes_after=processes_after,
                        decode_active_at_update_end=overlapped,
                    )
                )
                (root / "partial.json").write_text(json.dumps(report, indent=2))
        report["status"] = "completed_screening_not_causal_proof"
        report["reason"] = (
            "Raw shared cgroup counters and paired timings available. No causal precision-release comparison or validated deployment budget; do not interpret CPU contention or PSS alone as memory conflict."
        )
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        stop.set()
        sampler.join(timeout=5)
        if owned_model and owned_model.exists():
            owned_model.unlink()
            report["temporary_model_copy_removed"] = True
        (root / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
