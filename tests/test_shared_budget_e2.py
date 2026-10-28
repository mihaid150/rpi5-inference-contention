import json
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/shared_budget_14b"))
sys.path.insert(0, str(ROOT / "scripts"))
from timeline import run_timeline
from design import regime_rows, remaining_prefix
from index_path import point
from research_u3 import publish, retrieve


class Tests(unittest.TestCase):

    def test_arrival_and_submission_do_not_wait_for_current(self):
        fresh_seen = threading.Event()

        def current():
            if not fresh_seen.wait(1):
                raise AssertionError("fresh incorrectly serialized after current")
            return {"text": "old"}

        def fresh(mark):
            mark("fresh-submit")
            fresh_seen.set()
            return {"text": "failed quality preserved", "passed": False}

        r = run_timeline(0, False, current, lambda: {"version": "v2"}, fresh, timeout=2)
        names = [e["name"] for e in r["events"]]
        self.assertLess(names.index("fresh-submit"), names.index("current-complete"))
        self.assertFalse(r["fresh"]["passed"])

    def test_deferred_and_publication_gate(self):
        r = run_timeline(
            0.01,
            True,
            lambda: time.sleep(0.04),
            lambda: "v2",
            lambda mark: mark("retrieval"),
            timeout=1,
        )
        n = [e["name"] for e in r["events"]]
        self.assertLess(n.index("fresh-arrival"), n.index("current-complete"))
        self.assertLess(n.index("current-complete"), n.index("update-start"))
        self.assertLess(n.index("publication-observed"), n.index("retrieval"))

    def test_empty_staging_rebuild_preserves_v1(self):
        encoder = lambda texts: [[1 + len(t), 1 + t.count("new")] for t in texts]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old = root / "v1"
            new = root / "rebuild"
            active = root / "active"
            publish(old, "v1", [{"id": "old", "text": "old evidence"}], encoder, "test")
            point(active, old)
            self.assertEqual(
                retrieve(active, "new", [], encoder, "test")["version"], "v1"
            )
            publish(new, "v2", [{"id": "new", "text": "new evidence"}], encoder, "test")
            self.assertEqual(
                retrieve(active, "new", [], encoder, "test")["version"], "v1"
            )
            point(active, new)
            self.assertEqual(
                retrieve(active, "new", [], encoder, "test")["version"], "v2"
            )
            self.assertEqual(retrieve(old, "new", [], encoder, "test")["version"], "v1")

    def test_sampler_thread_exit_is_not_affinity_failure(self):
        from worker import process_sample

        with patch("worker.os.sched_getaffinity", side_effect=ProcessLookupError()):
            self.assertEqual(process_sample(os.getpid())["threads"], [])
        with self.assertRaises(RuntimeError):
            process_sample(os.getpid(), [-1])

    def test_resume_skips_only_exact_completed_prefix(self):
        cells = [dict(id=str(n)) for n in range(36)]
        previous = [dict(cell=cell) for cell in cells[:17]]
        self.assertEqual(
            remaining_prefix(cells, previous, [str(n) for n in range(17)]), cells[17:]
        )
        with self.assertRaises(ValueError):
            remaining_prefix(cells, previous[::-1], [str(n) for n in range(17)])
        with self.assertRaises(ValueError):
            remaining_prefix(cells, previous, ["unexpected"])

    def test_fixed_factorial_and_no_prediction_stage(self):
        p = json.loads(
            (ROOT / "configs/shared-budget-14b/campaign-2026-10-26.json").read_text()
        )
        rs = regime_rows(p)
        self.assertEqual(len(rs), 36)
        self.assertTrue(all((r["stage"] == "regimes" for r in rs)))
        p["rows"] = [r for r in p["rows"] if r["id"] != rs[0]["id"]]
        with self.assertRaises(ValueError):
            regime_rows(p)


if __name__ == "__main__":
    unittest.main()
