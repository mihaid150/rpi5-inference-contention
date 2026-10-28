import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from precision import Deployment, exact, exchange, field, pod
from launch import command, manifest


class PrecisionTests(unittest.TestCase):

    def test_bounded_exchange_rejects_wrong_shape(self):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                'import sys,struct;sys.stdin.buffer.read(8);sys.stdout.buffer.write(struct.pack("<I",8)+b"12345678");sys.stdout.buffer.flush()',
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        try:
            with self.assertRaises(ValueError):
                exchange(process, 0, b"", 4)
        finally:
            process.stdin.close()
            process.wait()
            process.stdout.close()

    def test_short_frame_fails(self):
        reader, writer = os.pipe()
        os.write(writer, b"xx")
        os.close(writer)
        with os.fdopen(reader, "rb") as stream:
            with self.assertRaises(RuntimeError):
                exact(stream, 4)

    def test_pod_uses_one_compute_thread_and_readonly_weights(self):
        cfg = dict(image="image", nodes=[dict(name="pi", shard_dir="/srv/shards")] * 4)
        spec = pod(cfg, 0, "test")["spec"]
        container = spec["containers"][0]
        self.assertFalse(spec["automountServiceAccountToken"])
        self.assertEqual(container["env"][1]["value"], "1")
        self.assertTrue(container["volumeMounts"][0]["readOnly"])
        self.assertEqual(field("Pss: 100 kB\n", "Pss"), 102400)

    def test_cleanup_refuses_changed_owner(self):
        with tempfile.TemporaryDirectory() as td:
            deployment = Deployment(dict(namespace="test"), Path(td))
            deployment.owned = {"owned": "expected"}
            deployment.kub = Mock(
                return_value=Mock(
                    stdout=json.dumps(dict(metadata=dict(uid="different")))
                )
            )
            with self.assertRaises(RuntimeError):
                deployment.close()
            self.assertEqual(deployment.kub.call_count, 1)

    def test_legacy_stage_explicit_entrypoints(self):
        args = command("image", Path("/assets"), Path("/results"), "trial", "prefill")
        self.assertEqual(args[-2:], ["/app/scripts/legacy_entry.py", "prefill"])
        job = manifest(
            "image", Path("/assets"), Path("/results"), "pi", "trial", "validation"
        )
        self.assertEqual(
            job["spec"]["template"]["spec"]["containers"][0]["command"],
            ["python", "/app/scripts/legacy_entry.py", "validation"],
        )


if __name__ == "__main__":
    unittest.main()
