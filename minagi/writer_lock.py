"""One writer per weights directory.

The store publishes files atomically, but two live learners can still take
turns overwriting each other's manifest, optimiser moments, and expert ages.
That is a corruption the atomic file writes cannot see. This advisory lock is
shared by every process that learns into a weights directory: serve.py,
train.py read --save, and ingestd. Read-only serving does not take it.

The lock lives beside the directory (``.weights.writer.lock``) rather than
inside it, so it is not copied into snapshots or mistaken for a model file.
POSIX uses flock; Windows uses the first byte of the same file. The lock file
is deliberately not deleted: unlinking it on release would let a third process
create a new inode and lock that while the old one is still held.
"""

import errno
import json
import os
import socket
import threading
import time


class WriterBusy(RuntimeError):
    """Another process (or this one) already holds the directory's lock."""


_HELD = set()
_HELD_GUARD = threading.Lock()
_BUSY_ERRNOS = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
_BUSY_WINERRORS = {32, 33, 36}


def lock_path(weights):
    """Where the directory's writer lock lives."""
    path = os.path.realpath(os.path.abspath(
        os.path.expanduser(os.fspath(weights))))
    path = path.rstrip(os.sep) or os.sep
    parent, base = os.path.split(path)
    if not parent:
        parent = os.curdir
    return os.path.join(parent, f".{base or 'weights'}.writer.lock")


def _is_busy(exc):
    return (getattr(exc, "errno", None) in _BUSY_ERRNOS
            or getattr(exc, "winerror", None) in _BUSY_WINERRORS)


def _holder(path):
    """Best-effort metadata from the last holder, for the error message."""
    try:
        with open(path, "rb") as f:
            raw = f.readline(4096)
        rec = json.loads(raw.decode("utf-8", "replace"))
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None
    return rec


def _describe(holder):
    if not holder:
        return "another writer"
    bits = []
    if holder.get("pid"):
        bits.append(f"pid {holder['pid']}")
    if holder.get("host"):
        bits.append(str(holder["host"]))
    if holder.get("purpose"):
        bits.append(str(holder["purpose"]))
    if not bits:
        return "another writer"
    if len(bits) == 1:
        return bits[0]
    return f"{bits[0]} ({', '.join(bits[1:])})"


class WriterLock:
    """An exclusive, advisory lock on one weights directory."""

    def __init__(self, weights, purpose="writer", wait=0.0):
        self.weights = os.path.abspath(os.path.expanduser(os.fspath(weights)))
        self.path = lock_path(self.weights)
        self.purpose = str(purpose)
        self.wait = max(0.0, float(wait))
        self.fd = None
        self._path_key = os.path.normcase(os.path.realpath(self.path))

    def _open(self):
        parent = os.path.dirname(self.path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        return os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)

    @staticmethod
    def _lock(fd):
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(fd):
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)

    def _write_holder(self):
        rec = {"pid": os.getpid(), "purpose": self.purpose,
               "since": time.time(), "host": socket.gethostname(),
               "weights": self.weights}
        data = (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8")
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.write(self.fd, data)
        os.ftruncate(self.fd, len(data))

    def acquire(self):
        """Take the lock or raise WriterBusy. Returns self for chaining."""
        with _HELD_GUARD:
            if self._path_key in _HELD:
                raise WriterBusy(
                    f"{self.weights} is already held by this process")
        deadline = time.monotonic() + self.wait
        while True:
            fd = self._open()
            try:
                self._lock(fd)
            except OSError as exc:
                holder = _holder(self.path)
                os.close(fd)
                if not _is_busy(exc):
                    raise
                if time.monotonic() < deadline:
                    time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
                    continue
                raise WriterBusy(
                    f"{self.weights} is already being written by "
                    f"{_describe(holder)}") from None
            self.fd = fd
            with _HELD_GUARD:
                _HELD.add(self._path_key)
            try:
                self._write_holder()
            except Exception:
                self.release()
                raise
            return self

    def release(self):
        """Release the lock. Safe to call more than once."""
        fd, self.fd = self.fd, None
        if fd is None:
            return
        try:
            self._unlock(fd)
        finally:
            os.close(fd)
            with _HELD_GUARD:
                _HELD.discard(self._path_key)

    def __enter__(self):
        if self.fd is not None:
            return self
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def writer_lock(weights, purpose="writer", wait=0.0):
    """Acquire and return a WriterLock; use it as a context manager."""
    return WriterLock(weights, purpose=purpose, wait=wait).acquire()
