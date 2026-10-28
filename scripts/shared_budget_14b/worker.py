import contextlib
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
import traceback
from backends import Standard
from design import allocation, diagnosis_rows

OUT = Path(os.environ.get("DLI_OUTPUT", "/output"))


def atomic(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(path)


def process_sample(pid, expected=None):
    root = Path(f"/proc/{pid}")
    threads = []
    for task in sorted((root / "task").glob("*")):
        try:
            status = task.joinpath("status").read_text()
            allowed = next(
                (
                    x.split(":", 1)[1].strip()
                    for x in status.splitlines()
                    if x.startswith("Cpus_allowed_list:")
                )
            )
            affinity = sorted(os.sched_getaffinity(int(task.name)))
            if expected is not None and affinity != sorted(expected):
                raise RuntimeError(
                    f"Affinity violation pid={pid} tid={task.name}: {affinity}"
                )
            stat = task.joinpath("stat").read_text().rsplit(")", 1)[1].split()
            threads.append(
                dict(
                    tid=int(task.name),
                    allowed=allowed,
                    cpus=affinity,
                    cpu_ticks=int(stat[11]) + int(stat[12]),
                    schedstat=task.joinpath("schedstat").read_text().strip(),
                    context_switches=[
                        x for x in status.splitlines() if "ctxt_switches" in x
                    ],
                )
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
    return dict(pid=pid, threads=threads, status=root.joinpath("status").read_text())


def encoder_process(conn, cpus, threads, index):
    try:
        os.sched_setaffinity(0, set(cpus))
        import torch

        torch.set_num_interop_threads(1)
        from research_u3 import publish
        from research_u3_encoder import MiniLMEncoder
        from research_u3_chunking import StableChunker

        encoder = MiniLMEncoder("/input/minilm", threads=threads)
        chunker = StableChunker(encoder.token_count, encoder.encoder_id)
        corpus = json.loads(Path("/input/corpus.json").read_text())
        publish(
            Path(index),
            "v1",
            corpus["snapshots"]["v1"],
            encoder,
            encoder.encoder_id,
            batch_size=4,
            chunker=chunker,
        )
        conn.send(
            dict(
                ready=True,
                identity=encoder.identity,
                affinity=process_sample(os.getpid(), cpus),
            )
        )
        command = conn.recv()
        if command != "run":
            raise RuntimeError("Wrong encoder command")
        start = time.monotonic()
        result = publish(
            Path(index),
            "v2",
            corpus["snapshots"]["v2"],
            encoder,
            encoder.encoder_id,
            batch_size=4,
            chunker=chunker,
        )
        end = time.monotonic()
        conn.send(
            dict(
                start=start,
                end=end,
                duration_s=end - start,
                result=result,
                affinity=process_sample(os.getpid(), cpus),
            )
        )
        conn.recv()
    except BaseException:
        with contextlib.suppress(Exception):
            conn.send(dict(error=traceback.format_exc()))
    finally:
        conn.close()


def receive(conn, timeout):
    if not conn.poll(timeout):
        raise TimeoutError("Encoder message deadline")
    value = conn.recv()
    if "error" in value:
        raise RuntimeError(value["error"])
    return value


def environment():
    value = dict(cpus=sorted(os.sched_getaffinity(0)))
    for name, path in {
        "cpu_max": "/sys/fs/cgroup/cpu.max",
        "cpuset": "/sys/fs/cgroup/cpuset.cpus.effective",
        "mem_max": "/sys/fs/cgroup/memory.max",
        "cpu_stat": "/sys/fs/cgroup/cpu.stat",
    }.items():
        value[name] = Path(path).read_text().strip()
    quota, period = value["cpu_max"].split()
    if quota == "max" or int(quota) != 4 * int(period):
        raise RuntimeError("CPU quota is not four cores")
    if len(value["cpus"]) != 4:
        raise RuntimeError("CPU set must contain exactly four cores")
    return value


def main():
    plan = json.loads(Path("/input/campaign.json").read_text())
    prior = json.loads(Path("/input/prior.json").read_text())
    rows = diagnosis_rows(plan)
    env = environment()
    cpus = env["cpus"]
    report = dict(
        status="preflight",
        stage="E1",
        rows=[],
        environment=env,
        plan=plan,
        semantic_review="pending",
    )

    def save(**state):
        atomic(OUT / "report.json", report)
        atomic(
            OUT / "status.json",
            dict(updated=time.time(), completed=len(report["rows"]), total=27, **state),
        )
        print(json.dumps(state), flush=True)

    save(phase="preflight", message="Verifying model bytes")
    with Path("/models/family/Q4_K_M.gguf").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != plan["model_sha256"]:
        raise RuntimeError("14B artifact digest mismatch")
    report["model_sha256"] = digest
    report["status"] = "running"
    from plan import quality

    cases = {x["id"]: x for x in prior["selection"]}
    for cell in rows:
        backend = None
        child = None
        conn = None
        sampler = None
        stop = threading.Event()
        sampling_errors = []
        record = dict(cell=cell, setup_start=time.monotonic())
        save(phase="setup", cell=cell)
        try:
            arm, encoder_cpus, nthreads = allocation(cell["policy"], cpus)
            arm.update(
                name="server-e1-" + cell["policy"],
                microbatch=512,
                cache="f16",
                flash="auto",
            )
            if cell["policy"] != "update-only":
                backend = Standard(arm, cell["id"], OUT)
                record.update(startup_s=backend.load_s, argv=backend.argv)
                process_sample(backend.proc.pid, arm["cpus"])
                record["warmup"] = backend.complete(
                    cases[cell["fixture"]], "warmup", cell["block"]
                )
            if cell["policy"] != "inference-only":
                ctx = mp.get_context("spawn")
                conn, remote = ctx.Pipe()
                child = ctx.Process(
                    target=encoder_process,
                    args=(remote, encoder_cpus, nthreads, str(OUT / cell["id"])),
                )
                child.start()
                remote.close()
                record["encoder_ready"] = receive(conn, 900)
            sample_path = OUT / (cell["id"] + "-samples.jsonl")

            def sample():
                try:
                    with sample_path.open("w") as stream:
                        while not stop.is_set():
                            item = dict(
                                t=time.monotonic(),
                                cgroup_cpu=Path("/sys/fs/cgroup/cpu.stat").read_text(),
                                cgroup_memory=Path(
                                    "/sys/fs/cgroup/memory.current"
                                ).read_text(),
                            )
                            if backend:
                                item["llm"] = process_sample(
                                    backend.proc.pid, arm["cpus"]
                                )
                            if child:
                                item["encoder"] = process_sample(
                                    child.pid, encoder_cpus
                                )
                            for label, path in [
                                (
                                    "temperature",
                                    "/sys/class/thermal/thermal_zone0/temp",
                                ),
                                (
                                    "frequency",
                                    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
                                ),
                            ]:
                                with contextlib.suppress(OSError):
                                    item[label] = Path(path).read_text().strip()
                            stream.write(json.dumps(item) + "\n")
                            stream.flush()
                            stop.wait(0.5)
                except BaseException:
                    sampling_errors.append(traceback.format_exc())

            sampler = threading.Thread(target=sample, daemon=True)
            record["origin"] = time.monotonic()
            sampler.start()
            save(phase="running", cell=cell)
            if conn:
                conn.send("run")
            if backend:
                result = backend.complete(
                    cases[cell["fixture"]], "diagnosis", cell["block"]
                )
                record["answer_end"] = time.monotonic()
                record["inference"] = result
                record["automatic_quality"] = quality(
                    cases[cell["fixture"]], result["text"], result["stop"]
                )
            if conn:
                record["update"] = receive(conn, 1800)
                record["update_complete_s"] = record["update"]["end"] - record["origin"]
            stop.set()
            if sampler:
                sampler.join(timeout=10)
            if sampling_errors:
                raise RuntimeError(sampling_errors)
            record["end"] = time.monotonic()
            record["sample_file"] = sample_path.name
            report["rows"].append(record)
            save(phase="cell-completed", cell=cell)
        finally:
            stop.set()
            if sampler:
                sampler.join(timeout=10)
            if conn:
                with contextlib.suppress(Exception):
                    conn.send("close")
                conn.close()
            if child:
                child.join(timeout=10)
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=10)
            if backend:
                backend.close()
    report["status"] = "completed"
    save(phase="completed")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        atomic(
            OUT / "failure.json", dict(error=traceback.format_exc(), time=time.time())
        )
        atomic(
            OUT / "status.json",
            dict(phase="failed", error=traceback.format_exc(), time=time.time()),
        )
        raise
