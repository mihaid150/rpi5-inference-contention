import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_paper


class FullScheduleTests(unittest.TestCase):

    def test_official_evaluator_archives_match_measured_versions(self):
        versions = json.loads((ROOT / "assets/versions.json").read_text())
        for name, key in [
            ("squad-evaluator.py.gz", "official_evaluator_sha256"),
            ("coqa-evaluator.py.gz", "coqa_evaluator_sha256"),
        ]:
            raw = gzip.decompress((ROOT / "assets/official" / name).read_bytes())
            self.assertEqual(hashlib.sha256(raw).hexdigest(), versions[key])

    def simulate(self, output, fail=None):
        calls = []

        def run(argv, **kwargs):
            if argv[0] == "docker":
                return subprocess.CompletedProcess(argv, 0)
            if argv[1].endswith("precision.py"):
                stage = "precision"
            else:
                stage = argv[argv.index("--stage") + 1]
            calls.append(stage)
            if stage == fail:
                raise subprocess.CalledProcessError(1, argv)
            target = Path(argv[argv.index("--output") + 1])
            target.mkdir(parents=True)
            (target / "DONE").write_text("simulated completion")
            return subprocess.CompletedProcess(argv, 0)

        argv = [
            "run_paper.py",
            "--assets",
            "/unused/assets",
            "--precision-config",
            "/unused/config",
            "--output",
            str(output),
            "--execute",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            run_paper.subprocess, "run", side_effect=run
        ):
            if fail:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_paper.main()
            else:
                run_paper.main()
        return calls

    def test_all_families_run_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "attempt"
            self.assertEqual(
                self.simulate(output),
                ["precision", "decode", "prefill", "validation", "e1-e3"],
            )
            self.assertTrue((output / "DONE").exists())

    def test_failure_stops_later_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "attempt"
            self.assertEqual(
                self.simulate(output, "prefill"), ["precision", "decode", "prefill"]
            )
            self.assertFalse((output / "DONE").exists())
            self.assertEqual(
                json.loads((output / "status.json").read_text())["phase"], "failed"
            )


if __name__ == "__main__":
    unittest.main()
