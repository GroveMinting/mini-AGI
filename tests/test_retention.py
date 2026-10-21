"""M7: the longitudinal gate log and its summary."""

import json
import os
import tempfile
import types
import unittest

import torch

from minagi.retention import RetentionLog, load, summary
from minagi.watchdog import HeldOutWatchdog


class _FakeModel:
    def __init__(self):
        self.cfg = types.SimpleNamespace(block=16)
        self.p = torch.zeros(1)

    def parameters(self):
        return iter([self.p])

    def train(self, mode=True):
        return self


class _FakeLearner:
    def __init__(self, weights_dir):
        self.weights_dir = weights_dir
        self.steps = 0
        self.unsaved = 0
        self.scale = 1.0
        self.manifest = {"val": 1.0, "read_chars": 12345}

    def save(self):
        self.unsaved = 0
        return True

    def set_scale(self, scale):
        self.scale = scale

    def scale_lr(self, factor):
        self.scale *= factor

    def reset_stream(self):
        pass


class RetentionLogTest(unittest.TestCase):
    def test_append_load_and_skip_a_torn_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "retention.jsonl")
            log = RetentionLog(path)
            log.append({"action": "baseline", "val": 1.0,
                        "domains": {"chat": 1.0, "code": 2.0}})
            log.append({"action": "improved", "val": 0.9,
                        "domains": {"chat": 0.8, "code": 2.0}})
            with open(path, "a") as f:
                f.write('{"torn":')
            rows = load(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["action"], "improved")

            s = summary(rows)
            self.assertEqual(s["rows"], 2)
            self.assertEqual(s["with_domains"], 2)
            self.assertAlmostEqual(s["domains"]["chat"]["delta"], -0.2)
            self.assertEqual(s["domains"]["code"]["delta"], 0.0)
            self.assertAlmostEqual(s["domains"]["chat"]["best"], 0.8)

    def test_summary_of_nothing_is_empty(self):
        self.assertEqual(summary([])["with_domains"], 0)


class WatchdogRetentionTest(unittest.TestCase):
    def test_gate_events_are_written_with_domains_and_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "weights")
            os.makedirs(weights)
            with open(os.path.join(weights, "core.npz"), "w") as f:
                f.write("state")
            log = os.path.join(tmp, "runs", "retention.jsonl")
            learner = _FakeLearner(weights)
            wd = HeldOutWatchdog(
                _FakeModel(), learner, tmp,
                best_dir=os.path.join(tmp, "best"), chunk=16, context=16,
                chunks=1, factor=1.05, lr_control=False,
                retention_log=log, verbose=False)
            scores = [1.0, 0.9, 0.91]
            wd._score = lambda: (scores.pop(0), 0.0,
                                 {"chat": 0.5, "code": 0.6})
            wd.start()
            learner.steps = 7
            wd.check()
            learner.steps = 14
            wd.check()

            rows = load(log)
            self.assertEqual([r["action"] for r in rows],
                             ["baseline", "improved", "ok"])
            self.assertEqual(rows[1]["live_steps"], 7)
            self.assertEqual(rows[2]["live_steps"], 14)
            self.assertEqual(rows[0]["domains"], {"chat": 0.5, "code": 0.6})
            self.assertEqual(rows[0]["chars"], 12345)
            self.assertEqual(wd.state()["retention_log"], log)


if __name__ == "__main__":
    unittest.main()
