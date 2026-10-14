"""looks_like_text must not reject a text file whose probe boundary cuts a
multi-byte UTF-8 character in half.

Reported as issue #30 by Dshamir: a strict decode of the first 4,096 bytes
raised whenever byte 4,096 fell inside a character, and the file was silently
dropped from training and held-out evaluation. Authored here; upstream has no
PR for #30.
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

    def test_invalid_utf8_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "bad.txt", b"hello " + b"\xff\xfe" + b"world")
            self.assertFalse(looks_like_text(path))


if __name__ == "__main__":
    unittest.main()
