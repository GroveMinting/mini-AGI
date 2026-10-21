"""Cumulative experience: the JSONL log, corpus sampling, and replay cadence."""

import json
import os
import tempfile
import unittest

from minagi.experience import CorpusSampler, ExchangeLog, Replay


class ExchangeLogTest(unittest.TestCase):
    def test_append_and_sample_excludes_the_last(self):
        log = ExchangeLog()
        for i in range(3):
            log.append(f"user {i}", f"bot {i}")
        self.assertEqual(len(log), 3)
        for _ in range(20):
            rec = log.sample()
            self.assertNotEqual(rec["user"], "user 2")

    def test_jsonl_round_trip_and_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "exchanges.jsonl")
            log = ExchangeLog(path)
            log.append("hello", "hi", turn=1, steps=0)
            log.append("again", "yes", turn=2, steps=3)
            with open(path) as f:
                lines = f.read().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["user"], "hello")
            with open(path, "a") as f:
                f.write("not json\n")
            loaded = ExchangeLog(path)
            self.assertEqual(loaded.loaded, 2)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.recent[-1]["bot"], "yes")

    def test_single_record_is_not_replayed(self):
        log = ExchangeLog()
        log.append("only", "one")
        self.assertIsNone(log.sample())


class CorpusSamplerTest(unittest.TestCase):
    def test_samples_a_bounded_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(2):
                with open(os.path.join(tmp, f"{i}.txt"), "w") as f:
                    f.write(("word " * 400) + f" file {i}")
            a = CorpusSampler(tmp, characters=64, seed=7)
            b = CorpusSampler(tmp, characters=64, seed=7)
            self.assertEqual(len(a.files), 2)
            first = a.sample()
            self.assertTrue(first)
            self.assertLessEqual(len(first), 64)
            self.assertEqual(first, b.sample())


class ReplayTest(unittest.TestCase):
    def _log(self):
        log = ExchangeLog()
        for i in range(3):
            log.append(f"past user {i}", f"past bot {i}")
        return log

    def test_cadence_and_excludes_the_newest(self):
        replay = Replay(self._log(), every=2, seed=0)
        self.assertIsNone(replay.maybe_next())
        text, note = replay.maybe_next()
        self.assertEqual(note, "replay")
        self.assertNotIn("past user 2", text)
        self.assertIn("past user", text)

    def test_user_only_drops_the_model_reply(self):
        replay = Replay(self._log(), every=1, user_only=True, seed=0)
        text, note = replay.maybe_next()
        self.assertEqual(note, "replay")
        self.assertIn("past user", text)
        self.assertNotIn("past bot", text)

    def test_corpus_alternates_with_exchanges(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "doc.txt"), "w") as f:
                f.write("corpus body text " * 40)
            corpus = CorpusSampler(tmp, characters=32, seed=0)
            replay = Replay(self._log(), every=1, corpus=corpus, seed=0)
            notes = [replay.maybe_next()[1] for _ in range(3)]
            self.assertEqual(notes, ["replay", "replay-corpus", "replay"])

    def test_paused_replay_does_not_advance(self):
        replay = Replay(self._log(), every=1, seed=0)
        self.assertIsNone(replay.maybe_next(allow=False))
        self.assertEqual(replay.turn, 0)
        self.assertEqual(replay.maybe_next()[1], "replay")


if __name__ == "__main__":
    unittest.main()
