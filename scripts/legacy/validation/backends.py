import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error


def memory(pid):
    result = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM", "RssAnon", "RssFile"):
            result[key + "_kib"] = int(value.split()[0])
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    result["cpu_s"] = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    for name in ("memory.current", "memory.peak"):
        try:
            result["cgroup_" + name] = int(Path("/sys/fs/cgroup", name).read_text())
        except OSError:
            pass
    return result


def request(path, payload=None, timeout=10):
    req = urllib.request.Request(
        "http://127.0.0.1:8091" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


class Standard:

    def __init__(self, arm, tag, out):
        self.arm = arm
        self.tag = tag
        self.log = (out / (tag + ".log")).open("w")
        self.argv = [
            "/standard/llama-server",
            "--model",
            "/models/family/Q4_K_M.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "8091",
            "--threads",
            "2",
            "--threads-batch",
            "4",
            "--ctx-size",
            "2048",
            "--parallel",
            "1",
            "--batch-size",
            "512",
            "--ubatch-size",
            str(arm["microbatch"]),
            "--cache-type-k",
            arm["cache"],
            "--cache-type-v",
            arm["cache"],
            "--flash-attn",
            arm["flash"],
            "--cache-ram",
            "0",
            "--no-warmup",
            "--repack",
            "--n-gpu-layers",
            "0",
        ]
        start = time.monotonic()
        self.proc = subprocess.Popen(
            self.argv,
            stdout=self.log,
            stderr=self.log,
            env=dict(os.environ, LD_LIBRARY_PATH="/standard"),
        )
        try:
            deadline = start + 300
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError("Standard backend exited; see " + tag)
                try:
                    with request("/health") as response:
                        if json.load(response).get("status") == "ok":
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Standard backend load deadline")
            self.load_s = time.monotonic() - start
            with request("/props") as response:
                self.props = json.load(response)
            (out / (tag + "-props.json")).write_text(json.dumps(self.props, indent=2))
        except BaseException:
            self.close()
            raise

    def complete(self, case, stage, repeat=1):
        payload = dict(
            prompt=case["tokens"],
            n_predict=64,
            temperature=0,
            seed=0,
            repeat_penalty=1.0,
            cache_prompt=False,
            return_tokens=True,
            stream=True,
            top_p=1.0,
            min_p=0.0,
            n_probs=0,
            timings_per_token=True,
        )
        before = memory(self.proc.pid)
        start = time.monotonic()
        first = None
        content = []
        tokens = []
        events = []
        final = None
        with request("/completion", payload, timeout=900) as response:
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                events.append(event)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("stop"):
                    final = event
                    break
                if event.get("tokens") or event.get("content"):
                    if first is None:
                        first = time.monotonic()
                content.append(event.get("content", ""))
                tokens.extend(event.get("tokens", []))
        if final is None:
            raise RuntimeError("No final standard response")
        if final.get("truncated") or final.get("timings", {}).get("cache_n", 0):
            raise RuntimeError("Unexpected truncation or reused prompt")
        if final.get("timings", {}).get("prompt_n") != len(case["tokens"]):
            raise RuntimeError(
                "Standard backend did not evaluate the full fresh prompt"
            )
        evaluated = final.get(
            "tokens_evaluated", final.get("timings", {}).get("prompt_n")
        )
        if evaluated is not None and evaluated != len(case["tokens"]):
            raise RuntimeError(
                f"Prompt count mismatch: {evaluated} != {len(case['tokens'])}"
            )
        content.append(final.get("content", ""))
        return dict(
            stage=stage,
            arm=self.arm["name"],
            fixture=case["id"],
            repeat=repeat,
            text="".join(content),
            stop=final.get("stop_type", "unknown"),
            generated_tokens=tokens,
            ttft_s=first - start if first is not None else time.monotonic() - start,
            total_request_s=time.monotonic() - start,
            server_timings=final.get("timings"),
            prompt_tokens=len(case["tokens"]),
            response_events=events,
            resource_before=before,
            resource_after=memory(self.proc.pid),
        )

    def close(self):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Prototype:

    def __init__(self, arm, tag, out):
        import prefill_worker

        self.impl = prefill_worker
        self.arm = arm
        self.tag = tag
        self.session = prefill_worker.Session(arm, tag)
        self.proc = self.session.proc
        self.load_s = self.session.load_s
        self.argv = ["private-worker", str(arm)]

    def complete(self, case, stage, repeat=1):
        return self.session.run(case, stage, repeat=repeat)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.session.__exit__(*args)
