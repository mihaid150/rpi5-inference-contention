import os
import select
import struct
import time
import numpy as np


def logits(proc, position, tokens, vocab, deadline):
    if not 1 <= len(tokens) <= 64 or position < 0 or position + len(tokens) > 2048:
        raise ValueError("Invalid position/batch")
    if any((not 0 <= int(t) < vocab for t in tokens)):
        raise ValueError("Invalid token")
    if time.monotonic() >= deadline:
        raise TimeoutError("Request deadline exceeded")
    payload = struct.pack("<" + "q" * len(tokens), *map(int, tokens))
    proc.stdin.write(struct.pack("<II", position, len(payload)) + payload)
    proc.stdin.flush()

    def exact(n):
        result = bytearray()
        while len(result) < n:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([proc.stdout], [], [], remaining)[0]:
                raise TimeoutError("Request deadline exceeded")
            part = os.read(proc.stdout.fileno(), n - len(result))
            if not part:
                raise RuntimeError("Worker EOF")
            result.extend(part)
        return result

    size = struct.unpack("<I", exact(4))[0]
    if size != vocab * 4:
        raise ValueError("Unexpected logits dimension")
    values = np.frombuffer(exact(size), dtype="<f4").copy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite logits")
    return values
