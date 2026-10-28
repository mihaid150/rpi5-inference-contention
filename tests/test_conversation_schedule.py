from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unittest
from research_v2_conversation_schedule import prefill


class Fake:

    def __init__(self):
        self.calls = []

    def switch(self, target, label):
        self.calls.append(("switch", target, label))

    def token(self, pos, t):
        self.calls.append(("token", pos, t))
        return t

    def event(self, *a, **kw):
        pass


class ScheduleTests(unittest.TestCase):

    def test_retained(self):
        r = Fake()
        prefill(r, list(range(8)), 2, 6, "retained")
        self.assertEqual([x[1] for x in r.calls if x[0] == "token"], list(range(8)))
        self.assertEqual([x[1] for x in r.calls if x[0] == "switch"], [0, 1, 0])

    def test_rebuild_entire_prefix(self):
        r = Fake()
        prefill(r, list(range(8)), 2, 6, "rebuild")
        self.assertEqual(
            [x[1] for x in r.calls if x[0] == "token"],
            [0, 1, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7],
        )

    def test_static(self):
        for mode, target in [("q4", 0), ("q3", 1)]:
            r = Fake()
            prefill(r, list(range(8)), 2, 6, mode)
            self.assertEqual([x[1] for x in r.calls if x[0] == "switch"], [target])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            prefill(Fake(), [1, 2], 1, 1, "retained")


if __name__ == "__main__":
    unittest.main()
