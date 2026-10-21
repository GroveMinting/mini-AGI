"""Scheduled file ingestion: offsets, opt-in sources, limits, and gate rebind."""

import json
import os
import tempfile
import unittest

from minagi.ingestd import IngestDaemon, Ledger, RateLimiter, Source, read_slice


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Learner:
    def __init__(self):
        self.fed = []
        self.notes = []
        self.steps = 0
        self.unsaved = 0

    def feed(self, text, _tok, note=""):
        self.fed.append(text)
        self.notes.append(note)
        self.steps += 1
        return [{"step": self.steps}]

    def due_to_save(self):
        return False

    def save(self):
        return True


class _Watch:
    def __init__(self, learner):
        self.learner = learner
        self.calls = 0

    def maybe_check(self, _every):
        self.calls += 1
        self.learner = _Learner()
        return {"action": "ok", "val": 1.0}


class RateLimiterTest(unittest.TestCase):
    def test_refills_and_caps(self):
        clock = _Clock()
        limiter = RateLimiter(60, clock=clock)
        self.assertEqual(limiter.available(), 60)
        self.assertEqual(limiter.take(30), 30)
        self.assertEqual(limiter.take(100), 30)
        self.assertEqual(limiter.available(), 0)
        clock.advance(30)
        self.assertEqual(limiter.available(), 30)
        clock.advance(600)
        self.assertEqual(limiter.available(), 60)


class LedgerTest(unittest.TestCase):
    def test_new_append_rewrite_and_shrink(self):
        ledger = Ledger()
        self.assertEqual(ledger.plan("f", 5, 10), (0, "new"))
        ledger.commit("f", 5, 10, 5)
        self.assertEqual(ledger.plan("f", 5, 10), (5, "unchanged"))
        self.assertEqual(ledger.plan("f", 7, 11), (5, "appended"))
        self.assertEqual(ledger.plan("f", 5, 11), (0, "rewritten"))
        self.assertEqual(ledger.plan("f", 4, 12), (0, "shrunk"))

    def test_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            ledger = Ledger(path)
            ledger.commit("f", 5, 10, 5)
            ledger.save()
            fresh = Ledger(path)
            self.assertEqual(fresh.files["f"], [5, 10, 5])
            with open(path) as f:
                self.assertEqual(json.load(f)["version"], 1)


class SliceTest(unittest.TestCase):
    def test_reads_chars_and_reports_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.txt")
            with open(path, "w") as f:
                f.write("éx")
            text, end, _before, _after = read_slice(path, 0, 1)
            self.assertEqual(text, "é")
            text2, end2, _before2, _after2 = read_slice(path, end, 4)
            self.assertEqual(text2, "x")
            self.assertGreater(end2, end)


class IngestDaemonTest(unittest.TestCase):
    def _daemon(self, tmp, sources, learner=None, **kw):
        cache = os.path.join(tmp, "index.json")
        state = os.path.join(tmp, "state.json")
        ledger = Ledger(state)
        return IngestDaemon(
            sources, learner or _Learner(), tokenizer=object(),
            ledger=ledger, cache=cache, quiet=True, **kw)

    def test_only_opted_in_sources_and_binary_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = os.path.join(tmp, "one")
            two = os.path.join(tmp, "two")
            os.makedirs(one)
            os.makedirs(two)
            with open(os.path.join(one, "a.txt"), "w") as f:
                f.write("hello\n")
            with open(os.path.join(one, "image.png"), "wb") as f:
                f.write(b"\x00\x01\x02")
            with open(os.path.join(two, "b.txt"), "w") as f:
                f.write("not this source\n")
            learner = _Learner()
            daemon = self._daemon(tmp, [Source(one)])
            daemon.learner = learner
            stats = daemon.tick()
            self.assertEqual(learner.fed, ["hello\n"])
            self.assertEqual(stats["files"], 1)
            self.assertEqual(stats["chars"], 6)

    def test_append_reads_only_new_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.makedirs(src)
            path = os.path.join(src, "log.txt")
            with open(path, "w") as f:
                f.write("abc\n")
            daemon = self._daemon(tmp, [Source(src)])
            learner = daemon.learner
            daemon.tick()
            self.assertEqual(learner.fed, ["abc\n"])
            with open(path, "a") as f:
                f.write("def\n")
            os.utime(path, ns=(2_000_000_000, 2_000_000_000))
            daemon.tick()
            self.assertEqual(learner.fed, ["abc\n", "def\n"])

    def test_rewrite_resets_to_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.makedirs(src)
            path = os.path.join(src, "note.txt")
            with open(path, "w") as f:
                f.write("abc")
            daemon = self._daemon(tmp, [Source(src)])
            daemon.tick()
            with open(path, "w") as f:
                f.write("xyz")
            os.utime(path, ns=(3_000_000_000, 3_000_000_000))
            daemon.tick()
            self.assertEqual(daemon.learner.fed, ["abc", "xyz"])

    def test_rate_limit_per_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.makedirs(src)
            with open(os.path.join(src, "a.txt"), "w") as f:
                f.write("abcdefghij")
            clock = _Clock()
            source = Source(src, rate=4, clock=clock)
            daemon = self._daemon(tmp, [source])
            daemon.tick()
            self.assertEqual(daemon.learner.fed, ["abcd"])
            daemon.tick()
            self.assertEqual(daemon.learner.fed, ["abcd"])
            clock.advance(60)
            daemon.tick()
            self.assertEqual(daemon.learner.fed, ["abcd", "efgh"])
            clock.advance(60)
            daemon.tick()
            self.assertEqual(daemon.learner.fed, ["abcd", "efgh", "ij"])

    def test_file_slice_limit_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.makedirs(src)
            with open(os.path.join(src, "a.txt"), "w") as f:
                f.write("abcdefg")
            daemon = self._daemon(tmp, [Source(src, rate=100)],
                                  max_file_chars=3)
            daemon.tick()
            daemon.tick()
            daemon.tick()
            self.assertEqual(daemon.learner.fed, ["abc", "def", "g"])

    def test_persisted_state_prevents_reread(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.makedirs(src)
            with open(os.path.join(src, "a.txt"), "w") as f:
                f.write("abc")
            first = self._daemon(tmp, [Source(src)])
            first.tick()
            second = self._daemon(tmp, [Source(src)])
            second.tick()
            self.assertEqual(second.learner.fed, [])

    def test_small_budget_rotates_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a")
            b = os.path.join(tmp, "b")
            os.makedirs(a)
            os.makedirs(b)
            with open(os.path.join(a, "a.txt"), "w") as f:
                f.write("aaa")
            with open(os.path.join(b, "b.txt"), "w") as f:
                f.write("bbb")
            daemon = self._daemon(tmp, [Source(a, rate=100),
                                        Source(b, rate=100)], max_files=1)
            daemon.tick()
            daemon.tick()
            self.assertEqual(sorted(daemon.learner.fed), ["aaa", "bbb"])

    def test_gate_rebinds_learner(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.makedirs(src)
            with open(os.path.join(src, "a.txt"), "w") as f:
                f.write("abc")
            learner = _Learner()
            watch = _Watch(learner)
            daemon = self._daemon(tmp, [Source(src)], learner=learner,
                                  watchdog=watch, gate_every=1)
            daemon.tick()
            self.assertIs(daemon.learner, watch.learner)
            self.assertEqual(watch.calls, 1)


if __name__ == "__main__":
    unittest.main()
