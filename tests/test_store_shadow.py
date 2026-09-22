import contextlib
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

import train
from minagi import store
from minagi.create import create
from train import build_paged


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


def _metadata_tree(path):
    out = {}
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(dirs + files):
            item = os.path.join(root, name)
            stat = os.stat(item)
            out[os.path.relpath(item, path)] = (
                stat.st_mode, stat.st_nlink, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns)
    return out


def _make_writable(path):
    for root, _, files in os.walk(path):
        os.chmod(root, 0o700)
        for name in files:
            os.chmod(os.path.join(root, name), 0o600)


def _make_read_only(path):
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            os.chmod(os.path.join(root, name), 0o400)
        os.chmod(root, 0o500)


def _create_tiny(path, block=16):
    create(path, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
           d_model=8, trunk_d_ff=16, n_head=1, block=block, max_steps=1,
           top_k=1)


def _admit(pool, idx):
    """Force one expert onto the card through the current admission API.

    begin_text starts a fresh text, so the vote that admits experts is only
    this call's mass. select_temperature is 0 for the tiny models these tests
    build, so admit takes the top-mass experts deterministically; with
    resident=1 that is the single expert named here, which evicts whatever the
    one slot held before.
    """
    mass = torch.zeros(pool.n_experts())
    mass[idx] = 1.0
    pool.begin_text()
    pool.admit(mass)


def _read_args(text, source):
    return [
        "--device", "cpu", "read", text,
        "--weights-dir", source,
        "--chunk", "8",
        "--context-start", "64",
        "--context-end", "64",
        "--resident", "1",
        "--ram-capacity", "1",
        "--passage", "64",
        "--grow-k", "0",
        "--held-out", "",
        "--history", "",
        "--precision", "fp32",
        "--train-steps-mean", "0",
        "--save-every", "0",
        "--no-plots",
    ]


class WeightShadowTest(unittest.TestCase):
    def test_dirty_expert_survives_eviction_without_changing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "weights")
            _create_tiny(source)
            _make_read_only(source)
            before = _digest_tree(source)
            metadata = _metadata_tree(source)

            shadow = store.shadow(source)
            try:
                model, _, pool, _ = build_paged(
                    source, torch.device("cpu"), resident=1,
                    ram_capacity=1, ceiling=16,
                    expert_write_path=os.path.join(shadow.path, "experts"))
                # expert 0 onto the card, so the one slot holds it
                _admit(pool, 0)
                opt = torch.optim.AdamW([pool.w1, pool.w3, pool.w2])
                expected_moments = []
                for i, param in enumerate((pool.w1, pool.w3, pool.w2), 1):
                    state = opt.state[param]
                    state["step"] = torch.tensor(1.0)
                    # Binary fractions survive the intentional bf16 moment
                    # packing exactly, so equality tests paging rather than
                    # storage quantisation.
                    state["exp_avg"] = torch.full_like(param, i / 4)
                    state["exp_avg_sq"] = torch.full_like(param, i / 8)
                    expected_moments.append((state["exp_avg"].clone(),
                                             state["exp_avg_sq"].clone()))
                pool.attach_optimiser(opt)
                # The step marker makes the slot's moments the expert's own, so
                # eviction writes them beside its weights. (Moments otherwise
                # travel only at a step, by design: a load while a reply is
                # written carries weights alone.)
                pool._count_step()
                with torch.no_grad():
                    pool.w1.add_(1.0)
                expected = pool.w1.detach().clone()

                # A RAM cache of one forces expert 0 through a shadow writeback
                # before it is requested again.
                _admit(pool, 1)
                _admit(pool, 2)
                _admit(pool, 0)
                # What a step does before using the moments: an arriving expert
                # brings weights alone, so its own are restored here from the
                # overlay.
                pool._own_moments()

                self.assertTrue(torch.equal(pool.w1, expected))
                for param, (avg, avg_sq) in zip(
                        (pool.w1, pool.w3, pool.w2), expected_moments):
                    self.assertTrue(torch.equal(opt.state[param]["exp_avg"], avg))
                    self.assertTrue(torch.equal(
                        opt.state[param]["exp_avg_sq"], avg_sq))
                self.assertTrue(os.path.exists(os.path.join(
                    shadow.path, "experts", "e00000.npz")))
                pool.tiers.delete(1)
                self.assertTrue(os.path.exists(os.path.join(
                    source, "experts", "e00001.npz")))
                with self.assertRaisesRegex(RuntimeError, "temporary expert overlay"):
                    store.save(model, source)
                self.assertEqual(_digest_tree(source), before)
                self.assertEqual(_metadata_tree(source), metadata)
                del model, pool
            finally:
                shadow.cleanup()
                _make_writable(source)

    def test_fresh_dry_model_exists_only_in_shadow(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "weights")
            shadow = store.shadow(source)
            shadow_root = os.path.dirname(shadow.path)
            try:
                _create_tiny(shadow.path)
                self.assertTrue(os.path.exists(
                    os.path.join(shadow.path, "core.npz")))
                self.assertFalse(os.path.exists(source))
            finally:
                shadow.cleanup()
            self.assertFalse(os.path.exists(shadow_root))

    def test_read_command_leaves_existing_weights_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "weights")
            text = os.path.join(tmp, "input.txt")
            _create_tiny(source, block=64)
            with open(text, "w") as f:
                f.write("abcdefgh" * 12)
            before = _digest_tree(source)
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            proc = subprocess.run(
                [sys.executable, "train.py"] + _read_args(text, source),
                cwd=root, text=True, capture_output=True, timeout=60)

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("temporary overlay", proc.stdout)
            self.assertIn("not saved", proc.stdout)
            self.assertEqual(_digest_tree(source), before)

    def test_read_command_creates_fresh_model_only_in_shadow(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "weights")
            text = os.path.join(tmp, "input.txt")
            with open(text, "w") as f:
                f.write("abcdefgh" * 12)

            def tiny_create(path, **_):
                _create_tiny(path, block=64)

            stdout = io.StringIO()
            with mock.patch("minagi.create.create", side_effect=tiny_create), \
                    mock.patch.object(sys, "argv",
                                      ["train.py"] + _read_args(text, source)), \
                    contextlib.redirect_stdout(stdout):
                with self.assertRaises(SystemExit) as stopped:
                    train.main()

            self.assertEqual(stopped.exception.code, 0)
            self.assertIn("temporary overlay", stdout.getvalue())
            self.assertFalse(os.path.exists(source))


if __name__ == "__main__":
    unittest.main()
