import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import statistics
import struct
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_v2_conversation_schedule import prefill
from research_v2_full_kl import metrics


def exact(stream, count, timeout=180):
    data = bytearray()
    deadline = time.monotonic() + timeout
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        while len(data) < count:
            if not selector.select(max(0, deadline - time.monotonic())):
                raise TimeoutError("Native worker response timeout")
            chunk = os.read(stream.fileno(), count - len(data))
            if not chunk:
                raise RuntimeError("Native worker exited during frame")
            data.extend(chunk)
    return bytes(data)


def exchange(process, position, payload, expected):
    process.stdin.write(struct.pack("<II", position, len(payload)) + payload)
    process.stdin.flush()
    count = struct.unpack("<I", exact(process.stdout, 4))[0]
    if count != expected:
        raise ValueError("Native response shape mismatch")
    return exact(process.stdout, count)


def parity(a, b):
    import numpy as np

    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite logits")
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if (
        np.max(np.abs(delta)) > 0.01
        or np.sqrt(np.mean(delta**2)) > 0.001
        or np.argmax(a) != np.argmax(b)
    ):
        raise ValueError("Numerical parity gate failed")


def field(raw, key):
    for line in raw.splitlines():
        parts = line.split()
        if parts and parts[0].rstrip(":") == key:
            return int(parts[1]) * 1024
    raise ValueError("Missing memory telemetry: " + key)


def pod(config, stage, name):
    node = config["nodes"][stage]
    return dict(
        apiVersion="v1",
        kind="Pod",
        metadata=dict(name=name),
        spec=dict(
            nodeSelector={"kubernetes.io/hostname": node["name"]},
            restartPolicy="Never",
            activeDeadlineSeconds=172800,
            automountServiceAccountToken=False,
            securityContext=dict(
                runAsUser=1000,
                runAsGroup=1000,
                runAsNonRoot=True,
                seccompProfile=dict(type="RuntimeDefault"),
            ),
            containers=[
                dict(
                    name="worker",
                    image=config["image"],
                    command=["sleep", "172800"],
                    resources=dict(
                        requests=dict(cpu="1", memory="6Gi"),
                        limits=dict(cpu="4", memory="6Gi"),
                    ),
                    env=[
                        dict(name="LD_LIBRARY_PATH", value="/native"),
                        dict(name="DLI_GGML_THREADS", value="1"),
                    ],
                    securityContext=dict(
                        allowPrivilegeEscalation=False,
                        readOnlyRootFilesystem=True,
                        capabilities=dict(drop=["ALL"]),
                    ),
                    volumeMounts=[
                        dict(name="weights", mountPath="/weights", readOnly=True)
                    ],
                )
            ],
            volumes=[
                dict(
                    name="weights",
                    hostPath=dict(path=node["shard_dir"], type="Directory"),
                )
            ],
        ),
    )


class Deployment:

    def __init__(self, config, output):
        self.config, self.output = (config, output)
        self.names = [
            "paper-precision-" + uuid.uuid4().hex[:12] + "-" + str(i) for i in range(4)
        ]
        self.owned = {}

    def kub(self, *args, **kwargs):
        prefix = ["kubectl"]
        if self.config.get("context"):
            prefix += ["--context", self.config["context"]]
        return subprocess.run(
            prefix + ["-n", self.config["namespace"], *map(str, args)],
            check=True,
            **kwargs,
        )

    def open(self):
        manifest = json.loads(Path(self.config["shard_manifest"]).read_text())
        try:
            for i, name in enumerate(self.names):
                spec = pod(self.config, i, name)
                (self.output / f"pod-{i}.json").write_text(
                    json.dumps(spec, indent=2) + "\n"
                )
                result = self.kub(
                    "create",
                    "-f",
                    "-",
                    "-o",
                    "json",
                    input=json.dumps(spec),
                    text=True,
                    capture_output=True,
                    timeout=60,
                )
                self.owned[name] = json.loads(result.stdout)["metadata"]["uid"]
                self.kub(
                    "wait",
                    "--for=condition=Ready",
                    "pod/" + name,
                    "--timeout=180s",
                    timeout=190,
                )
                live = self.kub(
                    "get",
                    "pod",
                    name,
                    "-o",
                    "json",
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                (self.output / f"pod-{i}-ready.json").write_text(live.stdout)
                files = [f"stage-{i}.gguf"] + (
                    ["alternate-stage-0.gguf"] if i == 0 else []
                )
                for file in files:
                    result = self.kub(
                        "exec",
                        name,
                        "--",
                        "sha256sum",
                        "/weights/" + file,
                        text=True,
                        capture_output=True,
                        timeout=180,
                    )
                    if result.stdout.split()[0] != manifest["hashes"][file]:
                        raise ValueError("Remote shard digest mismatch")
        except BaseException:
            self.close()
            raise

    def close(self):
        errors = []
        for name, uid in self.owned.items():
            try:
                found = self.kub(
                    "get",
                    "pod",
                    name,
                    "-o",
                    "json",
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                if json.loads(found.stdout)["metadata"]["uid"] != uid:
                    raise RuntimeError("Pod ownership changed; refusing deletion")
                self.kub(
                    "delete", "pod", name, "--wait=true", "--timeout=60s", timeout=70
                )
            except Exception as error:
                errors.append(dict(pod=name, error=repr(error)))
        if errors:
            (self.output / "cleanup-failures.json").write_text(
                json.dumps(errors, indent=2) + "\n"
            )
            raise RuntimeError(
                "Owned-pod cleanup incomplete; see cleanup-failures.json"
            )
        self.owned.clear()


class Route:

    def __init__(self, deployment, directory, tag):
        self.deployment, self.directory, self.tag = (deployment, directory, tag)
        self.processes = []
        self.logs = []

    def open(self):
        try:
            for i, name in enumerate(self.deployment.names):
                log = (self.directory / f"{self.tag}-stage-{i}.log").open("wb")
                self.logs.append(log)
                config = self.deployment.config
                argv = ["kubectl"]
                if config.get("context"):
                    argv += ["--context", config["context"]]
                argv += [
                    "-n",
                    config["namespace"],
                    "exec",
                    "-i",
                    name,
                    "--",
                    "/native/qwen-release-worker",
                    f"/weights/stage-{i}.gguf",
                    str(i),
                ]
                if i == 0:
                    argv += ["/weights/alternate-stage-0.gguf"]
                self.processes.append(
                    subprocess.Popen(
                        argv,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=log,
                        bufsize=0,
                    )
                )
            for position in range(16):
                self.token(position, 100 + position * 7919 % 100000)
            self.event("warm_prefix", positions=16)
        except BaseException:
            self.close()
            raise

    def event(self, name, **kwargs):
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(event=name, tag=self.tag, monotonic=time.monotonic(), **kwargs)
                )
                + "\n"
            )

    def token(self, position, token):
        import numpy as np

        manifest = json.loads(
            Path(self.deployment.config["shard_manifest"]).read_text()
        )
        payload = struct.pack("<q", int(token))
        for i, process in enumerate(self.processes):
            payload = exchange(
                process,
                position,
                payload,
                (manifest["vocab"] if i == 3 else manifest["hidden"]) * 4,
            )
        value = np.frombuffer(payload, dtype="<f4").copy()
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite response")
        return value

    def snapshot(self):
        process = self.processes[0]
        process.stdin.write(struct.pack("<II", 4294967294, 0))
        process.stdin.flush()
        count = struct.unpack("<I", exact(process.stdout, 4))[0]
        if not 0 < count < 100000:
            raise ValueError("Telemetry shape")
        return json.loads(exact(process.stdout, count))

    def switch(self, target, label):
        process = self.processes[0]
        process.stdin.write(struct.pack("<II", 4294967295, target))
        process.stdin.flush()
        if struct.unpack("<I", exact(process.stdout, 4))[0] != 8:
            raise ValueError("KV acknowledgement")
        count = struct.unpack("<Q", exact(process.stdout, 8))[0]
        if not count:
            raise ValueError("Empty KV state")
        self.event("switch", target=target, label=label, kv_bytes=count)

    def close(self):
        for process in self.processes:
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout:
                process.stdout.close()
        for log in self.logs:
            log.close()


def cycle(deployment, output, index):
    route = Route(deployment, output, "cycle-" + str(index))
    route.open()
    try:
        tokens = [100 + i * 7919 % 100000 for i in range(64)]
        control = [route.token(i, t) for i, t in enumerate(tokens)]
        before = route.snapshot()
        route.switch(1, "down")
        for i, t in enumerate(tokens):
            route.token(i, t)
        after = route.snapshot()
        route.switch(0, "up")
        for i, t in enumerate(tokens):
            parity(control[i], route.token(i, t))
        expected = route.token(64, 231)
        for i, t in enumerate(tokens):
            route.token(i, t)
        route.switch(0, "same_precision")
        parity(expected, route.token(64, 231))
        return dict(
            cycle=index,
            before=before,
            after=after,
            pss_saving_bytes=field(before["smaps_rollup"], "Pss")
            - field(after["smaps_rollup"], "Pss"),
            node_available_delta=field(after["meminfo"], "MemAvailable")
            - field(before["meminfo"], "MemAvailable"),
            mapping_decreased=after["mapped_weight_bytes"]
            < before["mapped_weight_bytes"],
        )
    finally:
        route.close()


def quality(deployment, output, assets):
    import numpy as np
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location("coqa", assets / "coqa_evaluator.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    tokenizer = AutoTokenizer.from_pretrained(
        assets / "q4-tokenizer", local_files_only=True
    )
    documents = json.loads((assets / "transition-cohort.json").read_text())
    rows = []
    for index, document in enumerate(documents):
        reference = None
        for mode in ("q4", "q3", "retained", "rebuild"):
            route = Route(deployment, output, f"d{index}-{mode}")
            start = time.monotonic()
            route.open()
            try:
                value = prefill(
                    route, document["prompt"], document["down"], document["up"], mode
                )
                values = []
                for j, target in enumerate(document["answer"]):
                    values.append(value.copy())
                    if j + 1 < len(document["answer"]):
                        value = route.token(len(document["prompt"]) + j, target)
                values = np.stack(values)
                np.save(output / f"d{index}-{mode}.npy", values)
                if mode == "q4":
                    reference = values
                if mode == "rebuild":
                    for a, b in zip(reference, values):
                        parity(a, b)
                scores = [
                    metrics(a, b, t)
                    for a, b, t in zip(reference, values, document["answer"])
                ]
                value = prefill(
                    route, document["prompt"], document["down"], document["up"], mode
                )
                generated = []
                stop = "length"
                for j in range(64):
                    token = int(value.argmax())
                    if token in (tokenizer.eos_token_id, 151643, 151645):
                        stop = "eos"
                        break
                    generated.append(token)
                    if j < 63:
                        value = route.token(len(document["prompt"]) + j, token)
                answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
                row = dict(
                    source_id=document["source_id"],
                    mode=mode,
                    text=answer,
                    stop=stop,
                    tokens=scores,
                    elapsed_s=time.monotonic() - start,
                    **evaluator.CoQAEvaluator._compute_turn_score(
                        document["aliases"], answer
                    ),
                )
                (output / f"d{index}-{mode}.json").write_text(
                    json.dumps(row, indent=2) + "\n"
                )
                rows.append(row)
            finally:
                route.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if len(config["nodes"]) != 4 or len({n["name"] for n in config["nodes"]}) != 4:
        parser.error("Four distinct worker nodes required")
    if not args.execute:
        print(
            json.dumps(
                dict(
                    execution=False, config=config, cycles=3, documents=3, treatments=4
                ),
                indent=2,
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    assets = args.assets.resolve()
    manifest = json.loads((assets / "assets-manifest.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((assets / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Prepared input changed: " + name)
    versions = json.loads((assets / "versions.json").read_text())
    if (
        hashlib.sha256((assets / "coqa_evaluator.py").read_bytes()).hexdigest()
        != versions["coqa_evaluator_sha256"]
    ):
        raise ValueError("Official CoQA evaluator changed")
    protocol = dict(
        created_unix=time.time(),
        config=config,
        input_manifest=manifest,
        versions=versions,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    deployment = Deployment(config, args.output)
    try:
        deployment.open()
        rows = []
        for i in range(3):
            row = cycle(deployment, args.output, i)
            rows.append(row)
            (args.output / f"cycle-{i}.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
        if (
            not all((r["mapping_decreased"] for r in rows))
            or statistics.median((r["pss_saving_bytes"] for r in rows)) <= 0
        ):
            raise RuntimeError("Release gate inconclusive; quality stage stopped")
        quality(deployment, args.output, args.assets.resolve())
    finally:
        deployment.close()
    (args.output / "DONE").write_text(
        "Precision cycles and twelve quality treatments completed. Semantic review pending.\n"
    )


if __name__ == "__main__":
    main()
