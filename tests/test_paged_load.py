"""A weights directory that was created but never paged-saved still has the
on-disk expert layout. It must load through the paging path, not in memory.

Reported as issue #17 by eletec: serve.py's startup banner crashed calling
`pool.vram_params()` on the in-memory SharedPool such a directory loaded into.
"""

import os
import tempfile
import unittest

import torch

from minagi import recur
from minagi.create import create
from minagi.paged import PagedPool


def _create_tiny(path):
    create(path, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
           d_model=8, trunk_d_ff=16, n_head=1, block=16, max_steps=1,
           top_k=1)


class PagedLoadTest(unittest.TestCase):
    def test_never_saved_dir_loads_through_the_paging_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = os.path.join(tmp, "weights")
            _create_tiny(w)
            self.assertTrue(os.path.isdir(os.path.join(w, "experts")),
                            "create() must leave the expert layout on disk")

            model, _meta = recur._load_dir(w, torch.device("cpu"),
                                           read_only=True)

            self.assertIsInstance(model.pool, PagedPool)
            # the banner asks for these; a SharedPool has n_experts/n_params
            # but no vram_params, which is what used to crash serve.py
            self.assertTrue(hasattr(model.pool, "vram_params"))
            self.assertEqual(model.pool.n_experts(), 3)


if __name__ == "__main__":
    unittest.main()
