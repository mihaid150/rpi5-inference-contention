import threading
import time


def run_timeline(
    offset, deferred, current, publish, fresh, event=lambda *a: None, timeout=1800
):
    origin = time.monotonic()
    arrived = threading.Event()
    published = threading.Event()
    current_done = threading.Event()
    cancel = threading.Event()
    errors = []
    result = {"origin": origin}
    lock = threading.Lock()

    def mark(name, **kw):
        with lock:
            result.setdefault("events", []).append(
                dict(name=name, t=time.monotonic(), **kw)
            )
            event(result["events"][-1])

    def wait(flag):
        deadline = time.monotonic() + timeout
        while not flag.wait(0.02):
            if cancel.is_set():
                raise RuntimeError("Episode cancelled by another task")
            if time.monotonic() > deadline:
                raise TimeoutError("Episode dependency deadline")
        if cancel.is_set():
            raise RuntimeError("Episode cancelled by another task")

    def task(fn):
        try:
            fn()
        except BaseException as exc:
            errors.append(exc)
            cancel.set()

    def arrival():
        if cancel.wait(max(0, origin + offset - time.monotonic())):
            return
        mark("fresh-arrival", scheduled=origin + offset)
        arrived.set()

    def old():
        mark("current-submit")
        result["current"] = current()
        mark("current-complete")
        current_done.set()

    def update():
        if deferred:
            wait(current_done)
        mark("update-start")
        result["update"] = publish()
        mark("publication-observed")
        published.set()

    def new():
        wait(arrived)
        wait(published)
        mark("fresh-admission")
        result["fresh"] = fresh(mark)
        mark("fresh-complete")

    threads = [
        threading.Thread(target=lambda f=f: task(f), daemon=True)
        for f in (old, arrival, update, new)
    ]
    mark("episode-start")
    for t in threads:
        t.start()
    deadline = origin + timeout
    for t in threads:
        t.join(max(0, deadline - time.monotonic()))
    if any((t.is_alive() for t in threads)):
        cancel.set()
        raise TimeoutError("Episode tasks exceeded deadline")
    if errors:
        raise errors[0]
    return result
