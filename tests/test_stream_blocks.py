"""Whole-block and spread held-out sampling."""

import os
import tempfile
import unittest

from minagi.create import create
from minagi.stream import FolderEvaluator
import train as T


def _write(root, name, text):
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


class BlockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        _write(root, "chat/a.txt", "a" * 80)
        _write(root, "chat/b.txt", "b" * 40)
        _write(root, "code/c.txt", "c" * 65)
        _write(root, "chat/short.txt", "s" * 5)
        self.ev = FolderEvaluator(None, root, 8, 32, "cpu")

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_whole_blocks_are_selected(self):
        self.assertEqual(self.ev.block, 32)
        chat = self.ev.blocks("chat")
        self.assertEqual(len(chat), 3)       # 80 -> 0,32; 40 -> 0
        self.assertEqual([s for _f, s in chat], [0, 32, 0])
        code = self.ev.blocks("code")
        self.assertEqual([s for _f, s in code], [0, 32])

    def test_whole_false_keeps_the_leftover_part(self):
        old = FolderEvaluator(None, self.tmp.name, 8, 32, "cpu",
                              whole=False)
        chat = old.blocks("chat")
        self.assertEqual([s for _f, s in chat], [0, 32, 64, 0, 32])

    def test_spread_is_even_and_stable(self):
        picked = self.ev.spread("chat", 1)
        self.assertEqual(len(picked), 1)
        self.assertEqual(picked, self.ev.spread("chat", 1))
        self.assertEqual(self.ev.spread("chat", 9), self.ev.blocks("chat"))
        text = self.ev._texts("chat", 1)
        self.assertEqual(len(text[0]), self.ev.block + 1)
        self.assertIs(text, self.ev._texts("chat", 1))

    def test_run_reads_spread_blocks(self):
        with tempfile.TemporaryDirectory() as wtmp:
            path = os.path.join(wtmp, "weights")
            create(path, verbose=False, experts=3, resident=1, d_ff=8,
                   depth=1, d_model=8, trunk_d_ff=16, n_head=1, block=32,
                   max_steps=1, top_k=1)
            model, _cfg, _pool, _man = T.build_paged(
                path, "cpu", resident=1, ram_capacity=2)
            ev = FolderEvaluator(model, self.tmp.name, 8, 32, "cpu")
            top = ev.run(2)
            spread = ev.run(2, spread=True)
        self.assertIn("chat", top)
        self.assertIn("code", top)
        self.assertIn("chat", spread)
        self.assertIn("code", spread)
        self.assertIn("stderr", spread)


if __name__ == "__main__":
    unittest.main()
