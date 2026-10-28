import json
import os
from pathlib import Path
import struct
import subprocess
import unittest
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from precision import exact, exchange as request

ROOT = Path(__file__).resolve().parents[1]


class NativeReleaseTests(unittest.TestCase):

    def test_release_matches_mapped_control_and_unmaps(self):
        bin = Path(os.environ.get("DLI_NATIVE_BIN", ROOT / "workspace/native-build"))
        q4 = Path(
            os.environ.get("DLI_TEST_Q4_SHARD", ROOT / "workspace/test-shards/q4.gguf")
        )
        q3 = Path(
            os.environ.get("DLI_TEST_Q3_SHARD", ROOT / "workspace/test-shards/q3.gguf")
        )
        if not all(
            (
                p.exists()
                for p in [
                    bin / "qwen-release-worker",
                    bin / "qwen-route-worker",
                    q4,
                    q3,
                ]
            )
        ):
            self.skipTest("Optional local artifacts unavailable")
        procs = []
        try:
            for executable in ("qwen-route-worker", "qwen-release-worker"):
                procs.append(
                    subprocess.Popen(
                        [str(bin / executable), str(q4), "0", str(q3)],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        bufsize=0,
                        env=dict(os.environ, DLI_GGML_THREADS="1"),
                    )
                )

            def token(pos, t):
                values = [
                    np.frombuffer(
                        request(p, pos, struct.pack("<q", t), 1536 * 4), dtype="<f4"
                    )
                    for p in procs
                ]
                np.testing.assert_array_equal(*values)

            def snapshot():
                p = procs[1]
                p.stdin.write(struct.pack("<II", 4294967294, 0))
                p.stdin.flush()
                n = struct.unpack("<I", exact(p.stdout, 4))[0]
                return json.loads(exact(p.stdout, n))

            def switch(target):
                counts = []
                for p in procs:
                    p.stdin.write(struct.pack("<II", 4294967295, target))
                    p.stdin.flush()
                    self.assertEqual(struct.unpack("<I", exact(p.stdout, 4))[0], 8)
                    counts.append(struct.unpack("<Q", exact(p.stdout, 8))[0])
                self.assertEqual(counts[0], counts[1])
                self.assertGreater(counts[0], 0)

            for pos, t in enumerate([17, 488, 220, 17]):
                token(pos, t)
            before = snapshot()
            self.assertIn(str(q4), before["maps"])
            self.assertNotIn(str(q3), before["maps"])
            switch(1)
            token(4, 26)
            after = snapshot()
            self.assertNotIn(str(q4), after["maps"])
            self.assertIn(str(q3), after["maps"])
            self.assertEqual(after["mapped_weight_bytes"], q3.stat().st_size)
            switch(0)
            token(5, 31)
            restored = snapshot()
            self.assertIn(str(q4), restored["maps"])
            self.assertNotIn(str(q3), restored["maps"])
            switch(0)
            token(6, 52)
        finally:
            for p in procs:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                if p.stdin:
                    p.stdin.close()
                if p.stdout:
                    p.stdout.close()


if __name__ == "__main__":
    unittest.main()
