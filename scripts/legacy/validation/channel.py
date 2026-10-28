import os
import select
import struct
import time
import numpy as np


def exact(stream, n, deadline):
    result = bytearray()
    while len(result) < n:
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([stream], [], [], left)[0]:
            raise TimeoutError("Worker deadline exceeded")
        part = os.read(stream.fileno(), n - len(result))
        if not part:
            raise RuntimeError("Worker EOF")
        result.extend(part)
    return result


def declare(proc, length):
    if not 1 <= length <= 1984:
        raise ValueError("Prompt must leave 64 output slots")
    proc.stdin.write(struct.pack("<II", length, 0))
    proc.stdin.flush()


def logits(proc, position, tokens, vocab, deadline):
    if not 1 <= len(tokens) <= 64 or not 0 <= position <= 2048 - len(tokens):
        raise ValueError("Invalid frame")
    if any((not 0 <= int(t) < vocab for t in tokens)):
        raise ValueError("Invalid token")
    if time.monotonic() >= deadline:
        raise TimeoutError("Worker deadline exceeded")
    payload = struct.pack("<" + "q" * len(tokens), *map(int, tokens))
    proc.stdin.write(struct.pack("<II", position, len(payload)) + payload)
    proc.stdin.flush()
    size = struct.unpack("<I", exact(proc.stdout, 4, deadline))[0]
    if size != vocab * 4:
        raise ValueError("Wrong logits dimension")
    value = np.frombuffer(exact(proc.stdout, size, deadline), dtype="<f4").copy()
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite logits")
    return value
