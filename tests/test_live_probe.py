"""M1 probe: drive the real LiveLearner over a tiny model and score held-out.

The probe is a measurement tool, not production code, but it must not silently
break: this runs one configuration on a CPU micro-model and checks the report
is complete and finite.
"""

import importlib.util
import math
import os
import tempfile
import unittest
from pathlib import Path

from minagi.create import create

ROOT = Path(__file__).resolve().parents[1]


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "live_probe", ROOT / "replication" / "live_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_weights(root):
    path = os.path.join(root, "weights")
    create(path, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
           d_model=8, trunk_d_ff=16, n_head=1, block=16, max_steps=1,
           top_k=1)
    return path


def _held_out(root):
    path = os.path.join(root, "val")
    text = "held out text for a tiny model. " * 40
    for domain in ("chat", "code"):
        d = os.path.join(path, domain)
        os.makedirs(d)
        with open(os.path.join(d, "sample.txt"), "w") as f:
            f.write(text)
    return path


class LiveProbeTest(unittest.TestCase):
    def test_reports_before_after_and_delta(self):
        probe = _load_probe()
        with tempfile.TemporaryDirectory() as tmp:
            weights = _tiny_weights(tmp)
            held = _held_out(tmp)
            conv = probe.DEFAULT_CONVERSATIONS[:8]
            r = probe.run_probe(
                weights, held, conv, device="cpu", chunk=16, mode="both",
                eval_chunks=1, passes=1, resident=1, ram_cache=2, context=16,
                precision="fp32")

            self.assertEqual(set(r["domains"]), {"chat", "code"})
            self.assertGreaterEqual(r["steps"], 1)
            self.assertGreater(r["chars"], 0)
            self.assertIsNotNone(r["train_loss"])
            self.assertTrue(math.isfinite(r["train_loss"]))
            for name in r["domains"]:
                for key in ("before", "after", "delta"):
                    self.assertIn(name, r[key])
                    self.assertTrue(math.isfinite(r[key][name]),
                                    f"{key}/{name} not finite")

    def test_user_only_mode_runs(self):
        probe = _load_probe()
        with tempfile.TemporaryDirectory() as tmp:
            weights = _tiny_weights(tmp)
            held = _held_out(tmp)
            conv = probe.DEFAULT_CONVERSATIONS[:6]
            r = probe.run_probe(
                weights, held, conv, device="cpu", chunk=16, mode="user",
                eval_chunks=1, passes=1, resident=1, ram_cache=2, context=16,
                precision="fp32")
            self.assertEqual(r["mode"], "user")
            self.assertEqual(set(r["domains"]), {"chat", "code"})
            self.assertGreater(r["chars"], 0)


if __name__ == "__main__":
    unittest.main()
