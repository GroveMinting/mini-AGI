"""Regression test for issue #30: the text probe must not split a character.

Reported by Dshamir: `looks_like_text` decoded the first 4,096 bytes strictly,
so a file whose 4,096th byte fell inside a multi-byte character raised
UnicodeDecodeError and was dropped - from training and from the held-out set -
with no message. This was carried as patch 0007 until upstream `ce31731`
rewrote the probe (`why_not_text`, decoding incrementally with `final=not cut`
and bumping the verdict-cache version). The patch was dropped at that sync and
this test kept, because it passes against upstream's implementation and pins
the behaviour either way.
"""

import os
import tempfile
import unittest

from minagi.ingest import looks_like_text


def _write(root, name, data):
    path = os.path.join(root, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


class TextProbeTest(unittest.TestCase):
    def test_multibyte_char_straddling_the_probe_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 4,095 ASCII bytes then a 3-byte character: byte 4,096 (0-indexed)
            # lands inside it.
            data = b"a" * 4095 + "\u25cf".encode("utf-8") + b"\n"
            path = _write(tmp, "f.txt", data)
            with open(path, "rb") as f:
                head = f.read(4096)
            with self.assertRaises(UnicodeDecodeError):
                head.decode("utf-8")           # the old behaviour's trigger
            self.assertTrue(looks_like_text(path))

    def test_binary_sample_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "blob", b"\x00\x01\x02\xff\xfe" + b"x" * 4096)
            self.assertFalse(looks_like_text(path))

    def test_truncated_multibyte_file_is_rejected(self):
        # A 1-2 byte file that is nothing but a UTF-8 lead sequence decodes to
        # nothing under the incremental decoder. It must be rejected, not
        # divided by zero (the regression the first cut of this patch had).
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in (("one", b"\xe2"), ("two", b"\xe2\x97"),
                               ("emoji", b"\xf0\x9f")):
                path = _write(tmp, name, data)
                self.assertFalse(looks_like_text(path))

    def test_invalid_utf8_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "bad.txt", b"hello " + b"\xff\xfe" + b"world")
            self.assertFalse(looks_like_text(path))


if __name__ == "__main__":
    unittest.main()
