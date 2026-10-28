import contextlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import threading
import time
import numpy as np
from transformers import AutoTokenizer
from research_u3 import publish
from research_u3_encoder import MiniLMEncoder
from research_u3_chunking import StableChunker
from channel import declare, exact, logits
from plan import ARMS, quality, paired_gate, select

OUT = Path(os.environ.get("DLI_OUTPUT", "/output"))
VOCAB = 152064
CONFIG = json.loads(Path("/input/prefill-config.json").read_text())
REPORT = dict(
    status="running",
    started=time.time(),
    config=CONFIG,
    rows=[],
    numerical=[],
    gates={},
    stages=[],
)
CURRENT = None


def atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def save():
    atomic(OUT / "report.json", REPORT)


def state(**fields):
    atomic(
        OUT / "status.json",
        dict(updated=time.time(), completed_rows=len(REPORT["rows"]), **fields),
    )
    print(json.dumps(fields), flush=True)


def memory(pid):
    result = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM", "RssAnon", "RssFile"):
                result[key + "_kib"] = int(value.split()[0])
    except FileNotFoundError:
        pass
    for name in ("memory.current", "memory.peak"):
        try:
            result["cgroup_" + name] = int(Path("/sys/fs/cgroup", name).read_text())
        except (FileNotFoundError, ValueError):
            pass
    try:
        result["temperature_c"] = (
            int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
        )
    except OSError:
        pass
    return result


def cpu(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def delta(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    la = a - np.max(a)
    lb = b - np.max(b)
    la -= np.log(np.exp(la).sum())
    lb -= np.log(np.exp(lb).sum())
    return dict(
        bit_identical=bool(a.tobytes() == b.tobytes()),
        max_abs=float(np.abs(a - b).max()),
        rms=float(np.sqrt(np.mean((a - b) ** 2))),
        kl_reference_candidate=float(np.sum(np.exp(la) * (la - lb))),
        same_argmax=bool(a.argmax() == b.argmax()),
    )


class Session:

    def __init__(self, arm, tag, reference=False):
        global CURRENT
        self.arm = arm
        self.tag = tag
        self.reference = reference
        self.path = OUT / (tag + ".log")
        self.log = self.path.open("w")
        self.offset = 0
        env = dict(
            os.environ,
            LD_LIBRARY_PATH="/reference" if reference else "/poc",
            DLI_ATTN_SKIP=",".join(map(str, arm["mask"])),
            DLI_CPU_REPACK="1" if arm["repack"] else "0",
            DLI_ALLOW_APPROX_PREFILL="1",
            DLI_DECODE_THREADS="2",
            DLI_WORKER_READY="1",
        )
        if reference:
            for key in (
                "DLI_CPU_REPACK",
                "DLI_ALLOW_APPROX_PREFILL",
                "DLI_WORKER_READY",
                "DLI_DECODE_THREADS",
            ):
                env.pop(key, None)
        start = time.monotonic()
        self.proc = subprocess.Popen(
            [
                "/reference/attention-worker" if reference else "/poc/attention-worker",
                "/models/family/Q4_K_M.gguf",
                "4",
                str(arm["microbatch"]),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            bufsize=0,
            env=env,
        )
        CURRENT = self
        if not reference:
            marker = struct.unpack(
                "<I", exact(self.proc.stdout, 4, time.monotonic() + 300)
            )[0]
            if marker:
                raise RuntimeError("Wrong worker readiness marker")
        self.load_s = time.monotonic() - start
        self.after_load_memory = memory(self.proc.pid)
        self.log.flush()
        if arm["repack"] and "CPU_REPACK" not in self.path.read_text():
            raise RuntimeError("Requested repack buffer was not actually selected")
        REPORT.setdefault("sessions", []).append(
            dict(
                tag=tag,
                arm=arm,
                startup_s=self.load_s,
                memory=self.after_load_memory,
                reference=reference,
            )
        )
        save()

    def prefix(self, case):
        if not self.reference:
            declare(self.proc, len(case["tokens"]))
        for pos in range(0, len(case["tokens"]), 64):
            values = logits(
                self.proc,
                pos,
                case["tokens"][pos : pos + 64],
                VOCAB,
                time.monotonic() + 900,
            )
        return values

    def run(
        self,
        case,
        stage,
        repeat=1,
        encoder=None,
        chunker=None,
        corpus=None,
        forced=None,
        prefix_only=False,
    ):
        state(
            phase="running",
            stage=stage,
            arm=self.arm["name"],
            fixture=case["id"],
            repeat=repeat,
        )
        idx = None
        if encoder:
            idx = OUT / f"index-{len(REPORT['rows']):04d}"
            publish(
                idx,
                "v1",
                corpus["snapshots"]["v1"],
                encoder,
                encoder.encoder_id,
                batch_size=4,
                chunker=chunker,
            )
        before = memory(self.proc.pid)
        sampled = []
        stop_sample = threading.Event()

        def sample():
            while not stop_sample.wait(0.25):
                sampled.append(memory(self.proc.pid))

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        start = time.monotonic()
        q0 = cpu(self.proc.pid)
        row = dict(
            stage=stage,
            arm=self.arm["name"],
            fixture=case["id"],
            repeat=repeat,
            configuration=self.arm,
            concurrent=bool(encoder),
            prompt_tokens=len(case["tokens"]),
            memory_before=before,
            session=self.tag,
        )
        try:
            values = self.prefix(case)
            row["ttft_s"] = time.monotonic() - start
            stamp = f"{len(REPORT['rows']):04d}-{stage}-{self.arm['name']}-{case['id']}"
            np.save(OUT / (stamp + "-prefix.npy"), values)
            row["prefix_logits"] = stamp + "-prefix.npy"
            if not prefix_only:
                generated = []
                times = []
                ds = time.monotonic()
                dq0 = cpu(self.proc.pid)
                e0 = cpu(os.getpid())

                def generate():
                    nonlocal values
                    try:
                        limit = 64 if forced is None else len(forced)
                        for i in range(limit):
                            token = (
                                int(values.argmax())
                                if forced is None
                                else int(forced[i])
                            )
                            if forced is None and token in (
                                151643,
                                151645,
                                TOK.eos_token_id,
                            ):
                                row["stop"] = "eos"
                                break
                            generated.append(token)
                            times.append(time.monotonic())
                            values = logits(
                                self.proc,
                                len(case["tokens"]) + i,
                                [token],
                                VOCAB,
                                time.monotonic() + 120,
                            )
                        else:
                            row["stop"] = (
                                "length" if forced is None else "teacher-forced"
                            )
                        row["decode_end"] = time.monotonic()
                    except BaseException as exc:
                        row["decode_error"] = repr(exc)

                decode = threading.Thread(target=generate)
                decode.start()
                if encoder:
                    us = time.monotonic()
                    row["update"] = publish(
                        idx,
                        "v2",
                        corpus["snapshots"]["v2"],
                        encoder,
                        encoder.encoder_id,
                        batch_size=4,
                        chunker=chunker,
                    )
                    row["update_s"] = time.monotonic() - us
                    row["update_model_cpu_s"] = cpu(self.proc.pid) - dq0
                    row["update_encoder_process_cpu_s"] = cpu(os.getpid()) - e0
                decode.join(timeout=900)
                if decode.is_alive() or row.get("decode_error"):
                    raise RuntimeError(f"Decode failed {row.get('decode_error')}")
                row.update(
                    decode_s=row["decode_end"] - ds,
                    generated_tokens=generated,
                    token_times=times,
                    text=TOK.decode(generated, skip_special_tokens=True),
                    decode_cpu_s=cpu(self.proc.pid) - dq0,
                )
                if forced is None:
                    row["quality"] = quality(case, row["text"], row["stop"])
                else:
                    np.save(OUT / (stamp + "-forced.npy"), values)
                    row["forced_logits"] = stamp + "-forced.npy"
            row.update(
                total_request_s=(time.monotonic() if prefix_only else row["decode_end"])
                - start,
                joint_completion_s=time.monotonic() - start,
                model_cpu_s=cpu(self.proc.pid) - q0,
                memory_after=memory(self.proc.pid),
            )
            stop_sample.set()
            sampler.join(timeout=2)
            row["sampled_memory_max"] = (
                {
                    key: max((s[key] for s in sampled if key in s))
                    for key in set().union(*(s.keys() for s in sampled))
                }
                if sampled
                else {}
            )
            self.log.flush()
            with self.path.open() as reader:
                reader.seek(self.offset)
                lines = reader.readlines()
                self.offset = reader.tell()
            steps = [
                json.loads(line[9:]) for line in lines if line.startswith("POC_STEP ")
            ]
            row["prefill_compute_s"] = sum(
                (s["wall_s"] for s in steps if s["position"] < len(case["tokens"]))
            )
            row["steps"] = steps
            REPORT["rows"].append(row)
            save()
            return row
        finally:
            stop_sample.set()
            sampler.join(timeout=2)

    def close(self):
        global CURRENT
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.log.close()
        CURRENT = None
        if self.proc.returncode:
            raise RuntimeError(f"Worker exit {self.proc.returncode}; see {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, typ, value, tb):
        if typ:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=10)
            self.log.close()
        else:
            self.close()


def compare_rows(reference, candidate, stage):
    by = {r["fixture"]: r for r in reference}
    for row in candidate:
        a = np.load(OUT / by[row["fixture"]]["prefix_logits"])
        b = np.load(OUT / row["prefix_logits"])
        REPORT["numerical"].append(
            dict(stage=stage, arm=row["arm"], fixture=row["fixture"], **delta(a, b))
        )
    save()


def complete(stage):
    REPORT["stages"].append(stage)
    save()


def summary():
    groups = {}
    for row in REPORT["rows"]:
        if row["stage"] not in ("timing", "ladder"):
            continue
        key = (row["stage"], row["arm"], row["fixture"], row["concurrent"])
        groups.setdefault(key, []).append(row)
    rows = []
    for (stage, arm, fixture, concurrent), values in groups.items():
        item = dict(
            stage=stage, arm=arm, fixture=fixture, concurrent=concurrent, n=len(values)
        )
        for key in (
            "ttft_s",
            "prefill_compute_s",
            "decode_s",
            "total_request_s",
            "update_s",
        ):
            observations = [v[key] for v in values if key in v]
            if observations:
                item[key + "_median"] = statistics.median(observations)
        if "quality" in values[0]:
            item["strict_passes"] = sum((v["quality"]["strict_passed"] for v in values))
            item["factual_passes"] = sum(
                (v["quality"]["factual_passed"] for v in values)
            )
        rows.append(item)
    REPORT["summary"] = rows
    save()


def main():
    global TOK
    try:
        import torch

        torch.set_num_interop_threads(1)
        TOK = AutoTokenizer.from_pretrained(
            "/input/q4-tokenizer", local_files_only=True
        )
        cases = CONFIG["cases"]
        selection = [c for c in cases if c["partition"] == "selection"]
        transfer = [c for c in cases if c["partition"] == "transfer"]
        by = {c["id"]: c for c in cases}
        base = ARMS[0]
        baseline = []
        state(phase="qualifying", stage="reference")
        probes = [
            dict(
                by["dev-shards"], id="prefix64", tokens=by["dev-shards"]["tokens"][:64]
            ),
            by["eval-original-u3"],
        ]
        old = []
        with Session(base, "qualification-reference", reference=True) as session:
            for case in probes:
                old.append(session.prefix(case).copy())
        with Session(base, "qualification-new") as session:
            for case, reference in zip(probes, old):
                values = session.prefix(case)
                result = dict(
                    stage="qualification",
                    fixture=case["id"],
                    **delta(reference, values),
                )
                REPORT["numerical"].append(result)
                save()
                if not result["bit_identical"]:
                    raise RuntimeError("Reference qualification failed")
        complete("qualification")
        candidates = {}
        for arm in ARMS:
            rows = []
            with Session(arm, "screen-" + arm["name"]) as session:
                for case in selection:
                    rows.append(session.run(case, "selection"))
                if arm == base:
                    baseline = rows
                teacher = dict(
                    by["dev-shards"],
                    id="teacher64",
                    tokens=by["dev-shards"]["tokens"][:64],
                )
                row = session.run(
                    teacher, "numerical", forced=baseline[0]["generated_tokens"][:8]
                )
                if arm == base:
                    teacher_ref = row
                else:
                    q = delta(
                        np.load(OUT / teacher_ref["forced_logits"]),
                        np.load(OUT / row["forced_logits"]),
                    )
                    REPORT["numerical"].append(
                        dict(stage="teacher-forced", arm=arm["name"], **q)
                    )
                    save()
            candidates[arm["name"]] = rows
            compare_rows(baseline, rows, "selection")
        winner, gates = select(baseline, candidates)
        REPORT["gates"]["selection"] = gates
        REPORT["selected"] = winner
        save()
        complete("selection")
        if winner is None:
            REPORT["outcome"] = "no_selection_candidate_passed"
            return
        selected = next((a for a in ARMS if a["name"] == winner))
        transfer_rows = {}
        for arm in [base] if selected == base else [base, selected]:
            with Session(arm, "transfer-" + arm["name"]) as session:
                transfer_rows[arm["name"]] = [
                    session.run(case, "transfer") for case in transfer
                ]
        gate = paired_gate(transfer_rows[base["name"]], transfer_rows[selected["name"]])
        REPORT["gates"]["transfer"] = gate
        save()
        complete("transfer")
        if not gate["passed"]:
            REPORT["outcome"] = "selected_candidate_failed_transfer"
            return
        full_rows = candidates[winner] + transfer_rows[winner]
        masked = dict(selected, name=winner + "-skip4", mask=[39, 40, 41, 42])
        with Session(masked, "skip-quality") as session:
            masked_rows = [session.run(case, "skip-quality") for case in cases]
        gate = paired_gate(full_rows, masked_rows)
        REPORT["gates"]["skip4"] = gate
        save()
        complete("skipping")
        bench = (
            [base]
            + ([] if selected == base else [selected])
            + ([masked] if gate["passed"] else [])
        )
        encoder = MiniLMEncoder("/input/minilm", threads=2)
        chunker = StableChunker(encoder.token_count, encoder.encoder_id)
        corpus = json.loads(Path("/input/corpus.json").read_text())
        REPORT["encoder"] = encoder.identity
        REPORT["update_alone"] = []
        for rep in range(3):
            index = OUT / f"update-alone-{rep}"
            publish(
                index,
                "v1",
                corpus["snapshots"]["v1"],
                encoder,
                encoder.encoder_id,
                batch_size=4,
                chunker=chunker,
            )
            start = time.monotonic()
            u = publish(
                index,
                "v2",
                corpus["snapshots"]["v2"],
                encoder,
                encoder.encoder_id,
                batch_size=4,
                chunker=chunker,
            )
            REPORT["update_alone"].append(
                dict(elapsed_s=time.monotonic() - start, update=u)
            )
            save()
        for rep in range(3):
            for arm in bench[rep % len(bench) :] + bench[: rep % len(bench)]:
                with Session(arm, f"timing-{rep}-{arm['name']}") as session:
                    for concurrent in (False, True):
                        for case in (by["dev-shards"], by["eval-original-u3"]):
                            session.run(
                                case,
                                "timing",
                                repeat=rep + 1,
                                encoder=encoder if concurrent else None,
                                chunker=chunker,
                                corpus=corpus,
                            )
            summary()
        complete("timing")
        ladders = CONFIG["ladder"]
        for rep in range(3):
            arms = [base] if selected == base else [base, selected]
            if rep % 2:
                arms.reverse()
            for arm in arms:
                with Session(arm, f"ladder-{rep}-{arm['name']}") as session:
                    for case in ladders:
                        session.run(case, "ladder", repeat=rep + 1, prefix_only=True)
            summary()
        complete("ladder")
        REPORT["outcome"] = "completed_development_comparison"
    except BaseException as exc:
        REPORT.update(status="failed", error=repr(exc))
        save()
        state(phase="failed", error=repr(exc))
        raise
    finally:
        if CURRENT and CURRENT.proc.poll() is None:
            CURRENT.proc.kill()
            CURRENT.proc.wait(timeout=10)
        if REPORT["status"] != "failed":
            REPORT.update(status="completed", finished=time.time())
            summary()
            save()
            state(phase="completed", outcome=REPORT.get("outcome"))
            (OUT / "DONE").write_text(REPORT.get("outcome", "completed") + "\n")


if __name__ == "__main__":
    main()
