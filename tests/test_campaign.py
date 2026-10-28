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
import campaign
from design import diagnosis_rows, regime_rows, prediction_rows
from summarize import summarize


class CampaignTests(unittest.TestCase):

    def simulate(self, root, fail_stage=None):
        output = root / "output"
        output.mkdir()
        inputs = root / "input"
        inputs.mkdir()
        plan = json.loads(
            (ROOT / "configs/shared-budget-14b/campaign-2026-10-26.json").read_text()
        )
        (inputs / "campaign.json").write_text(json.dumps(plan))
        (root / "backend").write_bytes(b"test backend")
        calls = []

        def path(value):
            value = str(value)
            if value == "/standard/llama-server":
                return root / "backend"
            if value == "/app/scripts":
                return root / "app/scripts"
            if value == "/output":
                return output
            if value.startswith("/input/"):
                return inputs / value.removeprefix("/input/")
            return Path(value)

        def run(argv, **kwargs):
            if "summarize.py" in str(argv[1]):
                summarize(output)
                return
            env = kwargs["env"]
            stage = env["EXPERIMENT_STAGE"]
            if "qualify" in str(argv[1]):
                calls.append("qualify-" + stage)
                return
            calls.append(stage)
            if stage == fail_stage:
                raise subprocess.CalledProcessError(1, argv)
            cells = {"E1": diagnosis_rows, "E2": regime_rows, "E3": prediction_rows}[
                stage
            ](plan)
            rows = [
                dict(
                    cell=c,
                    metrics=dict(
                        current_complete_s=100, publication_s=10, retrieval_s=2
                    ),
                    timeline=dict(
                        fresh=dict(
                            server_timings=dict(prompt_ms=20000, predicted_ms=10000)
                        )
                    ),
                )
                for c in cells
            ]
            report = dict(status="completed", rows=rows)
            if stage == "E3":
                frozen = (output / "predictions.json").read_bytes()
                self.assertTrue((output / "e2/DONE").is_file())
                report["prediction_freeze"] = dict(
                    sha256=hashlib.sha256(frozen).hexdigest()
                )
                for row in rows:
                    row["prediction_check"] = dict(
                        absolute_error_s=dict(fresh_complete_s=0)
                    )
            (Path(env["DLI_OUTPUT"]) / "report.json").write_text(json.dumps(report))

        with patch.object(campaign, "Path", side_effect=path), patch.object(
            campaign, "preflight", return_value={}
        ), patch.object(campaign.subprocess, "run", side_effect=run), patch.object(
            campaign.subprocess, "check_output", return_value="test inventory"
        ):
            if fail_stage:
                with self.assertRaises(subprocess.CalledProcessError):
                    campaign.main()
            else:
                campaign.main()
        return (output, calls)

    def test_sequential_execution_freezes_before_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, calls = self.simulate(Path(tmp))
            self.assertEqual(calls, ["E1", "qualify-E2", "E2", "qualify-E3", "E3"])
            self.assertTrue((output / "DONE").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertEqual(
                json.loads((output / "status.json").read_text())["phase"], "completed"
            )

    def test_failure_stops_pipeline_and_preserves_failed_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, calls = self.simulate(Path(tmp), "E2")
            self.assertNotIn("E3", calls)
            self.assertFalse((output / "DONE").exists())
            self.assertFalse((output / "predictions.json").exists())
            self.assertTrue((output / "e1/DONE").is_file())
            self.assertEqual(
                json.loads((output / "status.json").read_text())["phase"], "failed"
            )


if __name__ == "__main__":
    unittest.main()
