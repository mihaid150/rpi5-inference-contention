import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from campaign import freeze
from launch import command, manifest
from summarize import summarize


class ReleaseTests(unittest.TestCase):
    def test_deployment_has_bounded_resources_and_no_retry(self):
        job = manifest(
            "example@sha256:abc",
            Path("/srv/assets"),
            Path("/srv/results/new"),
            "pi",
            "trial",
        )
        self.assertEqual(job["spec"]["backoffLimit"], 0)
        spec = job["spec"]["template"]["spec"]
        self.assertFalse(spec["automountServiceAccountToken"])
        container = spec["containers"][0]
        self.assertEqual(container["resources"]["limits"], dict(cpu="4", memory="12Gi"))
        writable = [
            m["name"] for m in container["volumeMounts"] if not m.get("readOnly", False)
        ]
        self.assertEqual(writable, ["output", "tmp"])
        self.assertTrue(container["securityContext"]["readOnlyRootFilesystem"])

    def test_docker_is_offline_and_does_not_mount_repository(self):
        args = command("image", Path("/srv/assets"), Path("/srv/output"), "trial")
        self.assertEqual(args[args.index("--network") + 1], "none")
        self.assertIn("--read-only", args)
        self.assertEqual(sum(a == "--mount" for a in args), 3)
        self.assertNotIn(str(ROOT), " ".join(args))

    def test_freeze_uses_only_training_rows_and_fixed_holdout(self):
        plan = json.loads(
            (ROOT / "configs/shared-budget-14b/campaign-2026-10-26.json").read_text()
        )
        rows = [
            dict(
                cell=dict(policy=p, update="incremental"),
                metrics=dict(current_complete_s=100, publication_s=10, retrieval_s=2),
                timeline=dict(
                    fresh=dict(server_timings=dict(prompt_ms=20000, predicted_ms=10000))
                ),
            )
            for p in ("shared", "partition-2-2")
            for _ in range(6)
        ]
        prediction = freeze(dict(rows=rows), plan)
        self.assertEqual(len(prediction["predictions"]), 12)
        self.assertEqual(
            prediction["predictions"][0]["prediction"]["fresh_complete_s"], 130
        )

    def test_assets_are_the_required_fixed_subset(self):
        prior = json.loads((ROOT / "assets/prior.json").read_text())
        self.assertEqual(
            {c["id"] for c in prior["evaluation"]},
            {f"newdoc-{n:02}" for n in range(0, 12, 2)},
        )
        versions = json.loads((ROOT / "assets/versions.json").read_text())
        self.assertEqual(
            hashlib.sha256((ROOT / "assets/calibration.txt").read_bytes()).hexdigest(),
            versions["calibration_sha256"],
        )

    def test_summary_rejects_modified_training_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stage in ("e1", "e2", "e3"):
                (root / stage).mkdir()
                report = dict(status="completed", rows=[])
                if stage == "e3":
                    report["prediction_freeze"] = dict(
                        sha256=hashlib.sha256(b"{}").hexdigest()
                    )
                (root / stage / "report.json").write_text(json.dumps(report))
            (root / "predictions.json").write_text("{}")
            with self.assertRaises((ValueError, KeyError)):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
