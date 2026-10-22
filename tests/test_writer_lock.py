"""One writer per weights directory, not one writer per file."""

import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from minagi.writer_lock import WriterBusy, WriterLock, lock_path, writer_lock


class TrainReadLockTest(unittest.TestCase):
    def test_read_save_holds_the_directory_lock(self):
        from train import cmd_read
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "weights")
            os.makedirs(weights)
            args = types.SimpleNamespace(save=True, weights_dir=weights)

            def fake_read(_args):
                with self.assertRaises(WriterBusy):
                    WriterLock(weights, "another writer").acquire()
                return 0

            with mock.patch("train._cmd_read", side_effect=fake_read):
                self.assertEqual(cmd_read(args), 0)
            with writer_lock(weights, "after"):
                pass


class WriterLockTest(unittest.TestCase):
    def test_second_writer_is_refused_until_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "weights")
            os.makedirs(weights)
            lock = writer_lock(weights, "first")
            try:
                with self.assertRaises(WriterBusy):
                    WriterLock(weights, "second").acquire()
                with open(lock_path(weights), "rb") as f:
                    holder = f.read()
                self.assertIn(b"first", holder)
            finally:
                lock.release()
            with writer_lock(weights, "third"):
                pass

    def test_lock_is_beside_the_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "weights")
            os.makedirs(weights)
            with writer_lock(weights, "test"):
                self.assertEqual(os.path.dirname(lock_path(weights)), tmp)
                self.assertNotEqual(lock_path(weights),
                                    os.path.join(weights, "manifest.json"))

    def test_different_directories_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = os.path.join(tmp, "one")
            two = os.path.join(tmp, "two")
            os.makedirs(one)
            os.makedirs(two)
            with writer_lock(one, "one"), writer_lock(two, "two"):
                pass

    def test_another_process_holds_the_same_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "weights")
            os.makedirs(weights)
            code = (
                "import sys, time\n"
                "from minagi.writer_lock import writer_lock\n"
                "lock = writer_lock(sys.argv[1], 'child')\n"
                "print('locked', flush=True)\n"
                "time.sleep(20)\n")
            child = subprocess.Popen(
                [sys.executable, "-c", code, weights],
                stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                with self.assertRaises(WriterBusy):
                    WriterLock(weights, "parent").acquire()
            finally:
                child.terminate()
                child.wait(timeout=10)
                child.stdout.close()
            with writer_lock(weights, "after"):
                pass

    def test_release_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "weights")
            lock = WriterLock(weights, "test")
            lock.acquire()
            lock.release()
            lock.release()


if __name__ == "__main__":
    unittest.main()
