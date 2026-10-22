"""The held-out gate: snapshot the best, improve, and revert on a regression."""

import json
import math
import os
import tempfile
import types
import unittest

import torch

from minagi.create import create
from minagi.live import LiveLearner
from minagi.recur import load_any
from minagi.watchdog import HeldOutWatchdog
import train as T


class _FakeModel:
    def __init__(self):
        self.cfg = types.SimpleNamespace(block=16)
        self.training = True
        self.p = __import__("torch").zeros(1)

    def parameters(self):
        return iter([self.p])

    def train(self, mode=True):
        self.training = mode
        return self


class _FakeLearner:
    def __init__(self, weights_dir, scale=1.0):
        self.weights_dir = weights_dir
        self.steps = 0
        self.unsaved = 0
        self.manifest = {"val": 1.0}
        self.scale = scale
        self.saved = 0
        self.resets = 0

    def save(self):
        self.saved += 1
        self.unsaved = 0
        return True

    def set_scale(self, scale):
        self.scale = scale

    def scale_lr(self, factor):
        self.scale *= factor

    def reset_stream(self):
        self.resets += 1


class WatchdogFlowTest(unittest.TestCase):
    def test_rebind_keeps_the_best_and_follows_new_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "weights")
            os.makedirs(weights)
            with open(os.path.join(weights, "core.npz"), "w") as f:
                f.write("best-state")
            wd = HeldOutWatchdog(
                _FakeModel(), _FakeLearner(weights), tmp,
                best_dir=os.path.join(tmp, "best"), chunk=16, context=16,
                chunks=1, lr_control=True, verbose=False)
            wd._score = lambda: (1.0, 0.0, {"chat": 1.0})
            wd.start()
            best = wd.best
            fresh_model = _FakeModel()
            fresh_learner = _FakeLearner(weights)
            fresh_learner.steps = 7
            wd.rebind(fresh_model, fresh_learner)
            self.assertIs(wd.model, fresh_model)
            self.assertIs(wd.learner, fresh_learner)
            self.assertEqual(wd.best, best)
            self.assertEqual(wd.last_step, 7)
            self.assertIsNone(wd.ev)

    def _setup(self, tmp, scores, ses=None, lr_control=False):
        weights = os.path.join(tmp, "weights")
        os.makedirs(weights)
        with open(os.path.join(weights, "core.npz"), "w") as f:
            f.write("best-state")
        learner = _FakeLearner(weights)
        rebuilt = []

        def rebuild(path):
            new_learner = _FakeLearner(path)
            rebuilt.append(path)
            return _FakeModel(), new_learner

        wd = HeldOutWatchdog(
            _FakeModel(), learner, tmp, best_dir=os.path.join(tmp, "best"),
            chunk=16, context=16, chunks=1, factor=1.05,
            lr_control=lr_control, rebuild=rebuild, verbose=False)
        ses = list(ses or [0.02] * len(scores))
        wd._score = lambda: (scores.pop(0), ses.pop(0), {"chat": 0.5})
        return wd, weights, rebuilt

    def test_baseline_improve_and_revert(self):
        with tempfile.TemporaryDirectory() as tmp:
            wd, weights, rebuilt = self._setup(tmp, [1.0, 0.9, 1.2])
            self.assertEqual(wd.start()["action"], "baseline")
            self.assertEqual(wd.best, 1.0)
            self.assertTrue(os.path.isdir(wd.best_dir))
            with open(os.path.join(wd.best_dir, "gate.json")) as f:
                self.assertEqual(json.load(f)["val"], 1.0)

            self.assertEqual(wd.check()["action"], "improved")
            self.assertEqual(wd.best, 0.9)

            # simulate a save of the diverged state, replacing (not editing)
            # the inode so the hardlinked snapshot keeps the old bytes
            with open(os.path.join(weights, "core.npz.tmp"), "w") as f:
                f.write("diverged")
            os.replace(os.path.join(weights, "core.npz.tmp"),
                       os.path.join(weights, "core.npz"))
            wd.learner.unsaved = 1

            self.assertEqual(wd.check()["action"], "reverted")
            self.assertEqual(rebuilt, [weights])
            self.assertEqual(wd.best, 0.9)
            self.assertAlmostEqual(wd.learner.scale, 0.5)
            with open(os.path.join(weights, "core.npz")) as f:
                self.assertEqual(f.read(), "best-state")

    def test_check_runs_once_per_cadence(self):
        with tempfile.TemporaryDirectory() as tmp:
            wd, _weights, _rebuilt = self._setup(tmp, [1.0, 0.9, 0.91])
            wd.start()
            wd.learner.steps = 32
            self.assertEqual(wd.maybe_check(32)["action"], "improved")
            self.assertIsNone(wd.maybe_check(32))
            wd.learner.steps = 64
            self.assertEqual(wd.maybe_check(32)["action"], "ok")

    def test_within_factor_is_not_a_revert(self):
        with tempfile.TemporaryDirectory() as tmp:
            wd, _weights, rebuilt = self._setup(tmp, [1.0, 1.02])
            wd.start()
            self.assertEqual(wd.check()["action"], "ok")
            self.assertEqual(rebuilt, [])
            self.assertEqual(wd.best, 1.0)

    def test_noise_floor_cannot_widen_the_material_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 3 * se would allow a 12% regression, but the 5% factor is the
            # material cap. A noisy instrument must not turn 8% into "ok".
            wd, _weights, rebuilt = self._setup(
                tmp, [1.0, 1.08], ses=[0.04, 0.04])
            wd.start()
            self.assertEqual(wd.check()["action"], "reverted")
            self.assertEqual(rebuilt, [wd.weights_dir])

    def test_plasticity_scale_halves_on_revert(self):
        with tempfile.TemporaryDirectory() as tmp:
            wd, _weights, rebuilt = self._setup(tmp, [1.0, 1.2],
                                                lr_control=True)
            wd.start()
            wd.check()
            self.assertEqual(rebuilt, [wd.weights_dir])
            self.assertAlmostEqual(wd.plasticity.scale, 0.5)
            self.assertAlmostEqual(wd.learner.scale, 0.5)


def _tiny_weights(root):
    path = os.path.join(root, "weights")
    create(path, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
           d_model=8, trunk_d_ff=16, n_head=1, block=16, max_steps=1,
           top_k=1)
    model, _cfg, _pool, man = T.build_paged(path, "cpu", resident=1,
                                            ram_capacity=2)
    model.eval()
    return path, model, man


def _held_out(root):
    path = os.path.join(root, "val")
    text = "held out text for a tiny model. " * 40
    for domain in ("chat", "code"):
        d = os.path.join(path, domain)
        os.makedirs(d)
        with open(os.path.join(d, "sample.txt"), "w") as f:
            f.write(text)
    return path


class WatchdogPagedRebuildTest(unittest.TestCase):
    def test_revert_restores_a_paged_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights, model, man = _tiny_weights(tmp)
            held = _held_out(tmp)
            learner = LiveLearner(model, chunk=16, context=16,
                                  weights_dir=weights, manifest=man,
                                  save_every=0)

            def rebuild(path):
                fresh, _ = load_any(path, "cpu", read_only=False)
                with open(os.path.join(path, "manifest.json")) as f:
                    manifest = json.load(f)
                live = LiveLearner(fresh, chunk=16, context=16,
                                   weights_dir=path, manifest=manifest,
                                   save_every=0)
                return fresh, live

            wd = HeldOutWatchdog(
                model, learner, held, best_dir=os.path.join(tmp, "best"),
                chunk=16, context=16, chunks=1, factor=1.0, lr_control=False,
                rebuild=rebuild, verbose=False)
            wd._score = lambda: (1.0, 0.0, {"chat": 1.0, "code": 1.0})
            wd.start()

            named = dict(model.named_parameters())
            name = next(n for n in named if not n.startswith("pool."))
            before = named[name].detach().clone()
            with torch.no_grad():
                named[name].add_(1.0)
            learner.unsaved = 1
            learner.save()

            old = wd.model
            wd._revert()
            self.assertIsNot(wd.model, old)
            self.assertEqual(wd.learner.weights_dir, os.path.abspath(weights))
            self.assertAlmostEqual(wd.learner.scale, 0.5)
            after = dict(wd.model.named_parameters())[name]
            self.assertTrue(torch.allclose(after, before))


class WatchdogIntegrationTest(unittest.TestCase):
    def test_scores_a_real_tiny_held_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights, model, man = _tiny_weights(tmp)
            held = _held_out(tmp)
            learner = LiveLearner(model, chunk=16, context=16,
                                  weights_dir=weights, manifest=man,
                                  save_every=0)
            wd = HeldOutWatchdog(
                model, learner, held, best_dir=os.path.join(tmp, "best"),
                chunk=16, context=16, chunks=1, factor=0, lr_control=False,
                verbose=False)
            ev = wd.start()
            self.assertEqual(ev["action"], "baseline")
            self.assertTrue(math.isfinite(ev["val"]))
            self.assertEqual(set(ev["domains"]), {"chat", "code"})
            self.assertTrue(os.path.isdir(wd.best_dir))


if __name__ == "__main__":
    unittest.main()
