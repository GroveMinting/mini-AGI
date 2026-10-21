"""
Cumulative experience for the live stream.

The live learner reads the conversation once and lets it leave the window.
That is what keeps the diet light, but it also means a fact mentioned an hour
ago is gone. This module keeps a bounded recent history and, on a cadence,
puts a past exchange (or a slice of corpus text) back into the stream.

Everything here is a proposal to the same live learner, not a second path:
replay text goes through `LiveLearner.feed`, takes steps at the same density,
and is scored by the same held-out gate. If replay damages held-out, the gate
reverts it like any other live step.

PRIVACY: `--experience-log` writes the user's and the model's text to a JSONL
file. It is off by default; the in-memory history still works without it.

The sample is from the last `recent` exchanges, not the whole file. A replay
of a very old exchange is more likely to be a distribution the model has left
behind; the recent window is the part that still has context.
"""

import json
import os
import random
import sys
import time
from collections import deque

from .live import exchange_text


class ExchangeLog:
    """Bounded recent exchanges, optionally appended to a JSONL file."""

    def __init__(self, path=None, recent=512, seed=0):
        self.path = os.path.abspath(path) if path else None
        self.recent = deque(maxlen=int(recent))
        self.rng = random.Random(seed)
        self.appended = 0
        self.loaded = 0
        if self.path and os.path.exists(self.path):
            self._load()

    def _load(self):
        try:
            with open(self.path, errors="replace") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict) and "user" in rec and "bot" in rec:
                        self.recent.append(rec)
                        self.loaded += 1
        except OSError as e:
            print(f"[experience] log not loaded: {e}", file=sys.stderr)

    def append(self, user, bot, **meta):
        rec = {"user": user or "", "bot": bot or "", "at": time.time()}
        rec.update(meta)
        self.recent.append(rec)
        self.appended += 1
        if self.path:
            try:
                parent = os.path.dirname(self.path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as e:
                print(f"[experience] log append failed: {e}", file=sys.stderr)
        return rec

    def sample(self, exclude_last=True):
        if not self.recent:
            return None
        items = list(self.recent)
        if exclude_last:
            # never replay the exchange that just finished
            items = items[:-1]
        if not items:
            return None
        return self.rng.choice(items)

    def __len__(self):
        return len(self.recent)


class CorpusSampler:
    """A random slice of a text folder, for replaying documents too."""

    def __init__(self, root, characters=512, seed=0):
        self.root = os.path.abspath(root)
        self.characters = max(8, int(characters))
        self.rng = random.Random(seed)
        self.files = []
        if os.path.isdir(self.root):
            for dirpath, _dirs, names in os.walk(self.root):
                for name in names:
                    if name.endswith(".txt"):
                        self.files.append(os.path.join(dirpath, name))
        self.files.sort()

    def sample(self):
        if not self.files:
            return None
        for _ in range(min(4, len(self.files))):
            path = self.rng.choice(self.files)
            try:
                with open(path, errors="replace") as f:
                    text = f.read(self.characters * 8)
            except OSError:
                continue
            text = text.strip()
            if len(text) < 8:
                continue
            if len(text) > self.characters:
                start = self.rng.randrange(0, len(text) - self.characters + 1)
                text = text[start:start + self.characters]
            return text
        return None


class Replay:
    """Every `every` turns, hand the learner one past item."""

    def __init__(self, log, every=0, corpus=None, user_only=False, seed=0):
        self.log = log
        self.every = max(0, int(every))
        self.corpus = corpus
        self.user_only = bool(user_only)
        self.turn = 0

    def maybe_next(self, allow=True):
        """
        The next replay as `(text, note)`, or None.

        `allow=False` pauses the cadence without consuming it: while the gate
        is recovering from a revert, a replay would be one more variable.
        """
        if not self.every or not allow:
            return None
        self.turn += 1
        if self.turn % self.every:
            return None
        # alternate: exchange first, corpus second, and so on
        use_corpus = (self.corpus is not None
                      and (self.turn // self.every) % 2 == 0)
        if use_corpus:
            text = self.corpus.sample()
            if text:
                return text, "replay-corpus"
        rec = self.log.sample()
        if rec is not None:
            text = exchange_text(rec.get("user", ""),
                                 "" if self.user_only else rec.get("bot", ""))
            if text:
                return text, "replay"
        if self.corpus is not None:
            text = self.corpus.sample()
            if text:
                return text, "replay-corpus"
        return None
