"""The live stream's mechanics: steps, window, save cadence, LR scale."""

import os
import tempfile
import unittest

from minagi.create import create
from minagi.live import LiveLearner
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


if __name__ == "__main__":
    unittest.main()
