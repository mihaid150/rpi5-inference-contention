import contextlib
import json
import multiprocessing as mp
from pathlib import Path
from worker import OUT, atomic, environment, receive
from worker_e2 import encoder_process


def main():
    cpus = environment()["cpus"]
    prior = json.loads(Path("/input/prior.json").read_text())
    template = next((c for c in prior["evaluation"] if not c["unsupported"]))
    results = []
    for mode in ("incremental", "full-rebuild"):
        ctx = mp.get_context("spawn")
        conn, remote = ctx.Pipe()
        proc = ctx.Process(
            target=encoder_process,
            args=(
                remote,
                cpus[2:],
                str(OUT / ("qualification-" + mode)),
                template,
                mode,
            ),
        )
        proc.start()
        remote.close()
        try:
            ready = receive(conn, 900)
            conn.send("publish")
            update = receive(conn, 900)
            conn.send("retrieve")
            retrieved = receive(conn, 900)
            if retrieved["retrieval"]["version"] != "v2":
                raise RuntimeError("Qualification retrieved wrong version")
            if not any(
                (
                    p["source_id"] == template["source"]
                    for p in retrieved["retrieval"]["passages"]
                )
            ):
                raise RuntimeError("Qualification cannot retrieve added source")
            if mode == "full-rebuild" and update["old_index_version"] != "v1":
                raise RuntimeError("Rebuild destroyed old snapshot")
            results.append(
                dict(mode=mode, ready=ready, update=update, retrieved=retrieved)
            )
        finally:
            with contextlib.suppress(Exception):
                conn.send("close")
            conn.close()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
    atomic(
        OUT / "qualification.json",
        dict(
            passed=True,
            results=results,
            scope="untimed functional qualification, no application inference",
        ),
    )


if __name__ == "__main__":
    main()
