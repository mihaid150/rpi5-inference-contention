import importlib.util
import json
from pathlib import Path
import unittest
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "design", ROOT / "scripts/shared_budget_14b/design.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DesignTests(unittest.TestCase):

    def test_partition_disjoint_and_fixed_budget(self):
        for policy in ["partition-2-2", "partition-3-1"]:
            llm, enc, n = m.allocation(policy, [4, 6, 8, 10])
            self.assertFalse(set(llm["cpus"]) & set(enc))
            self.assertEqual(set(llm["cpus"]) | set(enc), {4, 6, 8, 10})
            self.assertEqual(n, len(enc))

    def test_exec_affinity_inherited_by_new_threads(self):
        cpu = min(os.sched_getaffinity(0))
        code = (
            "import os,threading; expected={"
            + str(cpu)
            + "}; assert os.sched_getaffinity(0)==expected; t=threading.Thread(target=lambda: os._exit(5) if os.sched_getaffinity(0)!=expected else None); t.start(); t.join()"
        )
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/shared_budget_14b/affinity_exec.py"),
                str(cpu),
                sys.executable,
                "-c",
                code,
            ]
        )
        self.assertEqual(r.returncode, 0)

    def test_bad_allowance_stops(self):
        for cpus in ([0, 1], [0, 1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                m.allocation("shared", cpus)

    def test_schedule_complete_and_missing_cell_rejected(self):
        p = json.loads(
            (ROOT / "configs/shared-budget-14b/campaign-2026-10-26.json").read_text()
        )
        self.assertEqual(len(m.diagnosis_rows(p)), 27)
        p["rows"].pop(0)
        with self.assertRaises(ValueError):
            m.diagnosis_rows(p)


if __name__ == "__main__":
    unittest.main()
