"""The default --device must prefer CUDA, then MPS, then CPU.

Carried from upstream PR #3 (MetalstormHS): on Apple Silicon the old inline
default fell back to CPU even though MPS works, with no signal to the user.
The ordering is asserted here by stubbing the two availability probes, so it
holds on a machine that has neither backend.
"""

import unittest
from unittest import mock

import torch

import train


class AutoDeviceTest(unittest.TestCase):
    def _pick(self, cuda, mps):
        with mock.patch.object(torch.cuda, "is_available", return_value=cuda), \
             mock.patch.object(torch.backends.mps, "is_available",
                               return_value=mps):
            return train._auto_device()

    def test_cuda_wins_over_mps(self):
        self.assertEqual(self._pick(True, True), "cuda")

    def test_cuda_when_only_cuda(self):
        self.assertEqual(self._pick(True, False), "cuda")

    def test_mps_when_no_cuda(self):
        self.assertEqual(self._pick(False, True), "mps")

    def test_cpu_is_the_fallback(self):
        self.assertEqual(self._pick(False, False), "cpu")


if __name__ == "__main__":
    unittest.main()
