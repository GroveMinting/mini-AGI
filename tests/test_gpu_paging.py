"""The GPU half of the paging path, which the CPU suite cannot see.

Every test here is skipped unless a CUDA device is present, so `make test` on
the CI runner runs them as no-ops while `make test-gpu` on a card exercises the
paths the carried patches touch under real VRAM paging:

    0001  a dry read's overlay must not write into the source checkpoint
    0006  a created-but-never-saved directory must load through paging
"""

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest

import torch

from minagi import recur
from minagi.create import create
from minagi.paged import PagedPool
from minagi.precision import amp
from train import build_paged

CUDA = unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
DEV = torch.device("cuda")


def _create_tiny(path, block=64):
    create(path, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
           d_model=8, trunk_d_ff=16, n_head=1, block=block, max_steps=1,
           top_k=1)


def _digest_tree(path):
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            file_path = os.path.join(root, name)
            digest.update(os.path.relpath(file_path, path).encode())
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    digest.update(chunk)
    return digest.hexdigest()


@CUDA
class CudaPagingTest(unittest.TestCase):
    def test_never_saved_dir_loads_paged_on_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = os.path.join(tmp, "weights")
            _create_tiny(w)
            model, _meta = recur._load_dir(w, DEV, read_only=True)
            self.assertIsInstance(model.pool, PagedPool)
            self.assertTrue(hasattr(model.pool, "vram_params"),
                            "serve.py's banner calls vram_params")
            self.assertEqual(model.pool.n_experts(), 3)

    def test_cuda_forward_backward_reaches_resident_experts(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = os.path.join(tmp, "weights")
            _create_tiny(w)
            model, _cfg, pool, _ = build_paged(w, DEV, resident=1,
                                               ram_capacity=1)
            model.train()
            x = torch.randint(0, 32, (1, 64), device=DEV)
            y = torch.randint(0, 32, (1, 64), device=DEV)
            with amp(DEV):
                _, loss = model(x, y)
            self.assertTrue(torch.isfinite(loss).item(),
                            "the tiny forward produced a non-finite loss")
            loss.backward()
            grads = [p.grad for p in (pool.w1, pool.w3, pool.w2)
                     if p.grad is not None]
            self.assertTrue(grads, "no resident expert received a gradient")

    def test_dry_read_on_cuda_leaves_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = os.path.join(tmp, "weights")
            text = os.path.join(tmp, "input.txt")
            _create_tiny(w)
            with open(text, "w") as f:
                f.write("abcdefgh" * 12)
            before = _digest_tree(w)
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            proc = subprocess.run(
                [sys.executable, "train.py", "--device", "cuda", "read", text,
                 "--weights-dir", w,
                 "--chunk", "8", "--context-start", "64", "--context-end",
                 "64", "--resident", "1", "--ram-capacity", "1",
                 "--passage", "64", "--grow-k", "0", "--held-out", "",
                 "--history", "", "--precision", "bf16",
                 "--train-steps-mean", "0", "--save-every", "0", "--no-plots"],
                cwd=root, text=True, capture_output=True, timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("temporary overlay", proc.stdout)
            self.assertEqual(_digest_tree(w), before,
                             "a save-less read changed the source weights")


if __name__ == "__main__":
    unittest.main()
