from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unittest
import numpy as np
from research_v2_full_kl import metrics, teacher_forced


class MetricsTests(unittest.TestCase):

    def test_identity(self):
        self.assertAlmostEqual(
            metrics([1, 2], [1, 2], 1)["kl_reference_to_candidate"], 0
        )

    def test_known_direction(self):
        p = np.array([0.8, 0.2])
        q = np.array([0.3, 0.7])
        self.assertAlmostEqual(
            metrics(np.log(p), np.log(q), 0)["kl_reference_to_candidate"],
            float(np.sum(p * np.log(p / q))),
        )

    def test_stability(self):
        self.assertTrue(
            np.isfinite(
                metrics([10000, -10000], [-10000, 10000], 0)[
                    "kl_reference_to_candidate"
                ]
            )
        )

    def test_reject(self):
        for p, q, t in [([float("nan")], [0], 0), ([0, 1], [0], 0), ([0], [0], 1)]:
            with self.assertRaises(ValueError):
                metrics(p, q, t)

    def test_alignment(self):
        calls = []

        def step(pos, t):
            calls.append((pos, t))
            return [pos, t]

        rows = list(teacher_forced(step, [10, 11], [20, 21]))
        self.assertEqual(calls, [(0, 10), (1, 11), (2, 20)])
        self.assertEqual([r[1] for r in rows], [20, 21])
        np.testing.assert_array_equal(rows[0][2], [1, 11])
        np.testing.assert_array_equal(rows[1][2], [2, 20])


if __name__ == "__main__":
    unittest.main()
