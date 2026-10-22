"""The live stream's mechanics: steps, window, save cadence, LR scale."""

import json
import os
import tempfile
import unittest

from minagi.create import create
from minagi.live import LiveLearner
from minagi.pool import PooledMLP, split_trunk_pool
from minagi.tokenizer import ByteTokenizer
import train as T

TEXT = "the model reads a stream of characters. " * 4


def _tiny(root):
    path = os.path.join(root, "weights")
    create(path, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
           d_model=8, trunk_d_ff=16, n_head=1, block=32, max_steps=1,
           top_k=1)
    model, _cfg, _pool, man = T.build_paged(path, "cpu", resident=1,
                                            ram_capacity=2)
    model.eval()
    return path, model, man


class LiveLearnerTest(unittest.TestCase):
    def test_split_trunk_pool_keeps_routing_in_the_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            _path, model, _man = _tiny(tmp)
            trunk, pool = split_trunk_pool(model)
            trunk_ids = {id(q) for q in trunk}
            pool_ids = {id(q) for q in pool}
            self.assertFalse(trunk_ids & pool_ids)
            self.assertIn(id(model.pool.gate), pool_ids)
            for site in model.modules():
                if isinstance(site, PooledMLP):
                    self.assertIn(id(site.router.weight), pool_ids)
                    self.assertIn(id(site.depth_emb), pool_ids)

    def test_short_message_takes_no_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, man = _tiny(tmp)
            learner = LiveLearner(model, chunk=16, context=32,
                                  weights_dir=path, manifest=man,
                                  save_every=0)
            recs = learner.feed("hi", ByteTokenizer())
            self.assertEqual(recs, [])
            self.assertEqual(learner.steps, 0)
            self.assertEqual(learner.pending, 2)

    def test_steps_and_window_stay_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, man = _tiny(tmp)
            learner = LiveLearner(model, chunk=16, context=32,
                                  weights_dir=path, manifest=man,
                                  save_every=0)
            recs = learner.feed(TEXT, ByteTokenizer())
            self.assertEqual(learner.steps, len(TEXT) // 16)
            self.assertEqual(len(recs), learner.steps)
            self.assertLess(len(learner.buf), 32 + 1)
            self.assertLess(learner.pending, 16)

    def test_save_cadence_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, man = _tiny(tmp)
            learner = LiveLearner(model, chunk=16, context=32,
                                  weights_dir=path, manifest=man,
                                  save_every=2)
            learner.feed("x" * 16, ByteTokenizer())
            self.assertFalse(learner.due_to_save())
            learner.feed("y" * 16, ByteTokenizer())
            self.assertTrue(learner.due_to_save())
            self.assertTrue(learner.save())
            self.assertEqual(learner.unsaved, 0)
            self.assertFalse(learner.due_to_save())
            self.assertTrue(os.path.exists(os.path.join(path, "manifest.json")))

    def test_lr_scale_and_stream_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, man = _tiny(tmp)
            learner = LiveLearner(model, chunk=16, context=32,
                                  weights_dir=path, manifest=man,
                                  save_every=0)
            base = [g["base_lr"] for g in learner.opt.param_groups]
            learner.set_scale(0.5)
            for group, rate in zip(learner.opt.param_groups, base):
                self.assertAlmostEqual(group["lr"], rate * 0.5)
            learner.scale_lr(0.5)
            self.assertAlmostEqual(learner.scale, 0.25)
            learner.feed("x" * 20, ByteTokenizer())
            learner.reset_stream()
            self.assertEqual(learner.buf, [])
            self.assertEqual(learner.pending, 0)

    def test_for_run_uses_training_rate_and_live_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, man = _tiny(tmp)
            man.update({"lr": 0.001,
                        "plasticity": {"scale": 0.5, "step": 100}})
            learner = LiveLearner.for_run(
                model, weights_dir=path, manifest=man, chunk=16, context=32,
                save_every=0, config={})
            self.assertAlmostEqual(learner.factor(), 0.5)
            self.assertAlmostEqual(
                learner.opt.param_groups[-1]["base_lr"], 0.001)
            self.assertAlmostEqual(
                learner.opt.param_groups[-1]["lr"], 0.0005)
            learner.set_scale(0.5)
            self.assertAlmostEqual(
                learner.opt.param_groups[-1]["lr"], 0.00025)
            self.assertAlmostEqual(learner.factor(), 0.5)
            self.assertAlmostEqual(learner.effective_scale(), 0.25)

    def test_live_steps_continue_across_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, man = _tiny(tmp)
            man.update({"step": 100, "live_steps": 5, "lr": 0.001})
            learner = LiveLearner.for_run(
                model, weights_dir=path, manifest=man, chunk=16, context=32,
                save_every=0, config={})
            self.assertEqual(learner.start_step, 105)
            self.assertEqual(learner.model_step, 105)
            learner.feed("x" * 16, ByteTokenizer())
            self.assertEqual(learner.model_step, 106)
            self.assertTrue(learner.save())
            with open(os.path.join(path, "manifest.json")) as f:
                saved = json.load(f)
            self.assertEqual(saved["step"], 100)
            self.assertEqual(saved["live_steps"], 6)
            again = LiveLearner.for_run(
                model, weights_dir=path, manifest=saved, chunk=16, context=32,
                save_every=0, config={})
            self.assertEqual(again.start_step, 106)

    def test_fold_steps_moves_live_steps_into_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, man = _tiny(tmp)
            man.update({"step": 100, "live_steps": 5, "lr": 0.001})
            learner = LiveLearner.for_run(
                model, weights_dir=path, manifest=man, chunk=16, context=32,
                save_every=0, config={})
            learner.fold_steps(106)
            self.assertEqual(learner.start_step, 106)
            self.assertEqual(learner.steps, 0)
            self.assertTrue(learner.save())
            with open(os.path.join(path, "manifest.json")) as f:
                saved = json.load(f)
            self.assertEqual(saved["step"], 106)
            self.assertEqual(saved["live_steps"], 0)

    def test_save_records_base_rate_when_manifest_lacks_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, man = _tiny(tmp)
            man.pop("lr", None)
            learner = LiveLearner(model, chunk=16, context=32,
                                  weights_dir=path, manifest=man,
                                  save_every=0)
            learner.feed("x" * 16, ByteTokenizer())
            self.assertTrue(learner.save())
            with open(os.path.join(path, "manifest.json")) as f:
                saved = json.load(f)
            self.assertAlmostEqual(saved["lr"], 3e-4)

    def test_save_preserves_rate_and_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, model, man = _tiny(tmp)
            man.update({"lr": 0.001,
                        "plasticity": {"scale": 0.5, "step": 100},
                        "read_chars": 100})
            learner = LiveLearner.for_run(
                model, weights_dir=path, manifest=man, chunk=16, context=32,
                save_every=0, config={})
            learner.feed("x" * 16, ByteTokenizer())
            self.assertTrue(learner.save())
            with open(os.path.join(path, "manifest.json")) as f:
                saved = json.load(f)
            self.assertAlmostEqual(saved["lr"], 0.001)
            self.assertAlmostEqual(saved["plasticity"]["scale"], 0.5)
            self.assertGreater(saved["plasticity"]["step"], 100)
            self.assertGreater(saved["read_chars"], 100)
            self.assertEqual(saved["step"], man.get("step"))


if __name__ == "__main__":
    unittest.main()
