"""Held-out log banners and the one-time block migration."""

import unittest

from minagi import heldout


class HeldoutTest(unittest.TestCase):
    def test_banner_levels_and_shift(self):
        text = (
            f"{heldout.PREFIX}{heldout.SPREAD}: 0.6000   chat 0.6000\n"
            f"{heldout.PREFIX}{heldout.CHECKS}: 0.6200   chat 0.6200\n"
            f"{heldout.PREFIX}{heldout.TOP}: 0.7000   chat 0.7000\n"
        )
        lines = heldout.banner(
            {"chat": 0.6}, {"chat": 0.62}, {"chat": 0.7})
        self.assertIn(heldout.SPREAD, lines)
        groups = heldout.levels(text)
        self.assertEqual(len(groups), 1)
        self.assertAlmostEqual(groups[0][1][heldout.SPREAD]["chat"], 0.6)
        at, delta = heldout.first_shift(text)
        self.assertEqual(at, 0)
        self.assertAlmostEqual(delta["chat"], 0.1)
        moved_val, moved_dom = heldout.moved(
            0.7, {"chat": 0.7}, delta)
        self.assertAlmostEqual(moved_val, 0.6)
        self.assertAlmostEqual(moved_dom["chat"], 0.6)

    def test_part_shift_is_separate_from_top_shift(self):
        text = (
            f"{heldout.PREFIX}{heldout.SPREAD}: 0.6000   chat 0.6000\n"
            f"{heldout.PREFIX}{heldout.PARTS}: 0.6500   chat 0.6500\n"
        )
        at, delta = heldout.part_shift(text)
        self.assertEqual(at, 0)
        self.assertAlmostEqual(delta["chat"], 0.05)
        self.assertIsNone(heldout.first_shift(text))

    def test_parse_log_moves_old_entries(self):
        text = (
            "step 1,000   0.1M characters   1 min\n"
            "held-out loss 0.7000 +/-0.0100 nats\n"
            "  chat 0.700\n"
            f"{heldout.PREFIX}{heldout.SPREAD}: 0.6000   chat 0.6000\n"
            f"{heldout.PREFIX}{heldout.TOP}: 0.7000   chat 0.7000\n"
            "step 2,000   0.2M characters   2 min\n"
            "held-out loss 0.6100 +/-0.0100 nats\n"
            "  chat 0.610\n"
        )
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            rows = heldout.parse_log(path)
        finally:
            import os
            os.unlink(path)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["shifted"])
        self.assertAlmostEqual(rows[0]["val"], 0.6)
        self.assertFalse(rows[1].get("shifted", False))
        self.assertAlmostEqual(rows[1]["val"], 0.61)


if __name__ == "__main__":
    unittest.main()
