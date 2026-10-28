import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/shared_budget_14b"))
from design import prediction_rows
from predictions import estimate, fit


class Tests(unittest.TestCase):

    def test_predictor_accounts_for_retrieval_before_slot(self):
        p = dict(
            current_complete_s=100, publication_s=10, retrieval_s=2, fresh_service_s=30
        )
        self.assertEqual(estimate(30, p)["fresh_complete_s"], 130)
        self.assertEqual(estimate(120, p)["fresh_complete_s"], 152)
        self.assertEqual(estimate(30, p)["bottleneck"], "current-slot")

    def test_fixed_holdout_and_original_schedule(self):
        plan = json.loads(
            (ROOT / "configs/shared-budget-14b/campaign-2026-10-26.json").read_text()
        )
        rows = prediction_rows(plan)
        self.assertEqual(
            {x["fixture"] for x in rows}, {"newdoc-06", "newdoc-08", "newdoc-10"}
        )
        plan["rows"].remove(rows[0])
        with self.assertRaises(ValueError):
            prediction_rows(plan)

    def test_fit_uses_engine_service_not_queued_duration(self):
        rows = []
        for p in ("shared", "partition-2-2"):
            for i in range(6):
                rows.append(
                    dict(
                        cell=dict(policy=p, update="incremental"),
                        metrics=dict(
                            current_complete_s=100, publication_s=10, retrieval_s=2
                        ),
                        timeline=dict(
                            fresh=dict(
                                total_request_s=999,
                                server_timings=dict(
                                    prompt_ms=20000, predicted_ms=10000
                                ),
                            )
                        ),
                    )
                )
        self.assertEqual(fit(rows)["shared"]["fresh_service_s"], 30)
        with self.assertRaises(ValueError):
            fit(rows[:-1])


if __name__ == "__main__":
    unittest.main()
