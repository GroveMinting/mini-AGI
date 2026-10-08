"""GradSNR.ratio: no bias correction on averages that start at the first reading.

Ported from 10xCrazyHorse's pytest suite on PR #28 to the repo's unittest runner.
"""

import unittest

import torch

from minagi.optim import GradSNR


class GradSNRRatioTest(unittest.TestCase):
    def test_constant_gradient_reads_one(self):
        a = torch.nn.Parameter(torch.zeros(4))
        snr = GradSNR()
        for n in range(1, 401):
            a.grad = torch.full((4,), 0.5)
            r = snr.observe([a])
            if n >= 8:                    # from the first reading, not only late
                self.assertLess(abs(r - 1.0), 1e-5, (n, r))

    def test_noise_reads_below_one_early(self):
        torch.manual_seed(0)
        a = torch.nn.Parameter(torch.zeros(256))
        snr = GradSNR()
        for _ in range(8):
            a.grad = torch.randn(256)
            r = snr.observe([a])
        self.assertGreaterEqual(r, 0.0)
        self.assertLess(r, 1.0)


if __name__ == "__main__":
    unittest.main()
