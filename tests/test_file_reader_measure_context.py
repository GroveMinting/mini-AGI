"""FileReader._measure must not feed more tokens than the rotary tables allow.

Ported from kevin9327's pytest suite on PR #27 to the repo's unittest runner.
"""

import unittest

import numpy as np
import torch

from minagi.precision import amp
from minagi.recur import RecurCoder, RecurConfig
from minagi.stream import FileReader


def _tiny_model(block=16):
    cfg = RecurConfig(
        vocab_size=64,
        d_model=32,
        n_head=4,
        d_ff=64,
        n_prelude=1,
        n_recur=1,
        n_coda=1,
        max_steps=1,
        block=block,
        use_pool=False,
        tie_embeddings=True,
    )
    return RecurCoder(cfg).eval()


class FileReaderMeasureContextTest(unittest.TestCase):
    def test_measure_caps_chunk_to_context(self):
        # File longer than chunk; chunk larger than the model's rotary table.
        model = _tiny_model(block=16)
        data = np.arange(80, dtype=np.uint8) % 60
        reader = FileReader(model, data, "heldout", chunk=64, context=16,
                            device=torch.device("cpu"))
        loss = reader.step(learn=False)
        self.assertIsNotNone(loss)
        self.assertLessEqual(reader.pos, 16)
        self.assertLessEqual(reader.seen, 16)

    def test_measure_raises_on_main_without_cap(self):
        """Document the old failure: an uncapped read past rope length raises."""
        model = _tiny_model(block=16)
        data = np.arange(80, dtype=np.uint8) % 60
        reader = FileReader(model, data, "heldout", chunk=64, context=16,
                            device=torch.device("cpu"))

        def uncapped_measure(n):
            if reader.caches is None or reader.seen + n > reader.context:
                reader.caches = reader.model.empty_caches()
                reader.seen = 0
            a = reader.data[reader.pos:reader.pos + n + 1].astype(np.int64)
            x = torch.from_numpy(a[:-1]).to(reader.device).unsqueeze(0)
            y = torch.from_numpy(a[1:]).to(reader.device).unsqueeze(0)
            with amp(reader.device):
                return reader.model(x, y, caches=reader.caches,
                                    pos_offset=reader.seen)

        with self.assertRaisesRegex(ValueError, "rotary tables"):
            uncapped_measure(64)


if __name__ == "__main__":
    unittest.main()
