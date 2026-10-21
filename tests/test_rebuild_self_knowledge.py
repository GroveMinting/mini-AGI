"""Rebuild the self-knowledge lane from a checkpoint manifest."""

import contextlib
import io
import json
import os
import tempfile
import unittest

from tools import rebuild_self_knowledge as R


def _manifest(tmp, n_experts=175):
    weights = os.path.join(tmp, "weights")
    os.makedirs(weights, exist_ok=True)
    with open(os.path.join(weights, "manifest.json"), "w") as f:
        json.dump({"n_experts": n_experts, "context_now": 512,
                   "step": 1, "cfg": {}}, f)
    return weights


class RebuildSelfKnowledgeTest(unittest.TestCase):
    def test_missing_manifest_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = R.main(["--weights", os.path.join(tmp, "weights"),
                         "--source", os.path.join(tmp, "source"),
                         "--out", os.path.join(tmp, "data")])
            self.assertEqual(rc, 2)

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = _manifest(tmp)
            source = os.path.join(tmp, "source")
            out = os.path.join(tmp, "data")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = R.main(["--weights", weights, "--source", source,
                             "--out", out, "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(source))
            self.assertIn("corpora chat", buf.getvalue())

    def test_rebuild_uses_manifest_and_writes_both_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = _manifest(tmp, n_experts=175)
            source = os.path.join(tmp, "source")
            out = os.path.join(tmp, "data")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = R.main(["--weights", weights, "--source", source,
                             "--out", out, "--conversations", "8",
                             "--val", "1", "--seed", "1",
                             "--shard-chars", "1000"])
            self.assertEqual(rc, 0)
            train = os.path.join(out, "train", "self-knowledge")
            val = os.path.join(out, "val", "self-knowledge")
            self.assertTrue(os.listdir(train))
            self.assertTrue(os.listdir(val))
            text = ""
            for name in sorted(os.listdir(train)):
                if name.endswith(".txt"):
                    with open(os.path.join(train, name)) as f:
                        text += f.read()
            self.assertIn("175", text)
            self.assertIn("<user>", text)
            self.assertIn("<bot>", text)

    def test_yaml_points_at_an_existing_tool(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tool = os.path.join(root, "tools", "rebuild_self_knowledge.py")
        self.assertTrue(os.path.exists(tool))
        with open(os.path.join(root, "corpora", "self_knowledge.yaml")) as f:
            self.assertIn("tools/rebuild_self_knowledge.py", f.read())

    def test_read_command_is_non_growing_and_saves(self):
        cmd = R._read_command("/w", "/lane", "/val", 2048, 1, 5e-5)
        self.assertIn("--save", cmd)
        self.assertIn("--grow-k", cmd)
        self.assertEqual(cmd[cmd.index("--grow-k") + 1], "0")
        self.assertEqual(cmd[cmd.index("--grow-every") + 1], "0")


if __name__ == "__main__":
    unittest.main()
