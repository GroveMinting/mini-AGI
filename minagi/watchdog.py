"""
A held-out watchdog for the live stream.

M1 measured what live learning costs: at the serving density (chunk 2048) the
held-out change was inside the instrument, but at four times the density it was
material, and feeding the model's own reply roughly tripled it. The live loop
therefore needs more than a learning rate: it needs a signal that does not
come from the conversation it is learning from, and a way back.

This keeps the last state that scored well, scores the held-out directory on a
step cadence, and:

  * snapshots the weights when held-out improves (hardlinks, so it is cheap
    and does not double the disk);
  * restores that snapshot when held-out is worse than the best by `factor`
    or by three evaluation standard errors, whichever is smaller - the factor
    is a hard material cap, while the standard error can only tighten it when
    the evaluation is precise enough;
  * moves the learning rate with `Plasticity` when it is given, and halves it
    on a revert.

The held-out score is the unweighted mean over its sub-folders, matching the
probe's `others` aggregate. By default it reads whole blocks spread over all of
each subject - the same kind of sample training scores - so a reading means the
weights changed, not that a different part of the file was reached.

REVERTING A PAGED MODEL. The pool pages experts from the weights directory,
so restoring the directory and then re-loading the model is the reliable path.
The caller may pass `rebuild(path)` returning `(model, learner)`; serve.py does
this, because a paged model's resident set and optimiser are tied together. A
non-paged caller can omit it and let `store.load` reload the existing model.
"""

import json
import os
import shutil
import sys
import time

import torch

from . import store
from .plasticity import Plasticity
from .retention import RetentionLog
from .stream import FolderEvaluator


def _link(src, dst):
    """Hardlink where the filesystem allows it, copy where it does not."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _replace_dir(source, target):
    """
    Make `target` a copy of `source` without moving the bytes twice.

    Files are hardlinked, which is what makes a per-improvement snapshot cheap
    on a multi-gigabyte pool. `store.save` writes each file to a `.tmp` and
    renames it, so a later save in `target` replaces the link rather than
    changing the snapshot's bytes.
    """
    source, target = os.path.abspath(source), os.path.abspath(target)
    if source == target:
        raise ValueError("snapshot source and target are the same directory")
    if target.startswith(source + os.sep):
        raise ValueError("snapshot target cannot live inside its source")
    tmp, old = target + ".tmp", target + ".old"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    shutil.copytree(source, tmp, copy_function=_link)
    had_target = os.path.exists(target)
    if had_target:
        os.rename(target, old)
    try:
        os.rename(tmp, target)
    except Exception:
        if had_target and os.path.exists(old) and not os.path.exists(target):
            os.rename(old, target)
        raise
    shutil.rmtree(old, ignore_errors=True)


class HeldOutWatchdog:
    """Score held-out on a cadence and keep the best state recoverable."""

    def __init__(self, model, learner, held_out, best_dir=None, chunk=2048,
                 context=None, chunks=4, factor=1.10, device=None,
                 lr_control=True, rebuild=None, retention_log=None,
                 verbose=True, spread=True):
        if not learner.weights_dir:
            raise ValueError("the watchdog needs a live weights directory")
        self.model = model
        self.learner = learner
        self.held_out = os.path.abspath(held_out)
        self.weights_dir = os.path.abspath(learner.weights_dir)
        self.best_dir = os.path.abspath(
            best_dir or (self.weights_dir.rstrip(os.sep) + ".best"))
        if self.best_dir == self.weights_dir:
            raise ValueError("the best snapshot must not be the live directory")
        self.device = device or next(model.parameters()).device
        self.context = int(context or getattr(model.cfg, "block", 4096))
        self.chunk = max(8, min(int(chunk), self.context))
        self.chunks = max(1, int(chunks))
        self.spread = bool(spread)
        self.factor = float(factor)
        self.rebuild = rebuild
        self.verbose = verbose
        pool = getattr(model, "pool", None)
        self.paged = pool is not None and hasattr(pool, "tiers")
        self.plasticity = Plasticity() if lr_control else None
        self.retention = None
        if retention_log:
            try:
                self.retention = RetentionLog(retention_log)
            except OSError as e:
                print(f"  [retention] disabled: {e}", file=sys.stderr)
        self.ev = None
        self.best = None
        self.best_se = 0.0
        self.last = None
        self.last_step = int(learner.steps)
        self.events = []

    def _say(self, text):
        if self.verbose:
            print(f"  [gate] {text}", flush=True)

    def _score(self):
        """(mean held-out loss, its stderr, per-domain losses)."""
        if self.ev is None:
            self.ev = FolderEvaluator(
                self.model, self.held_out, self.chunk, self.context,
                self.device)
        was_training = self.model.training
        try:
            d = self.ev.run(self.chunks, spread=self.spread)
        finally:
            # FolderEvaluator leaves the model in train mode; generation must
            # not inherit dropout from a background check.
            self.model.train(was_training)
        se = float(d.pop("stderr", 0.0) or 0.0)
        vals = {k: float(v) for k, v in d.items()}
        if not vals:
            raise RuntimeError(f"no held-out domains under {self.held_out}")
        mean = sum(vals.values()) / len(vals)
        return mean, se, vals

    def _read_meta(self):
        """The best this directory already holds, if it describes the same
        instrument. A restart must not overwrite a better snapshot with its
        own baseline."""
        try:
            with open(os.path.join(self.best_dir, "gate.json")) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return None
        try:
            same = (os.path.abspath(meta["held_out"]) == self.held_out
                    and int(meta["chunk"]) == self.chunk
                    and int(meta["context"]) == self.context
                    and int(meta["chunks"]) == self.chunks
                    and bool(meta.get("spread", False)) == self.spread)
        except (KeyError, TypeError, ValueError):
            return None
        if not same or meta.get("val") is None:
            return None
        return meta

    def _write_meta(self, path):
        meta = {"val": self.best, "se": self.best_se,
                "held_out": self.held_out, "chunk": self.chunk,
                "context": self.context, "chunks": self.chunks,
                "spread": self.spread, "at": time.time()}
        tmp = os.path.join(path, "gate.json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, os.path.join(path, "gate.json"))

    def _snapshot(self):
        _replace_dir(self.weights_dir, self.best_dir)
        self._write_meta(self.best_dir)

    def _apply_lr(self):
        if self.plasticity is not None:
            self.learner.set_scale(self.plasticity.scale)

    def _record(self, ev):
        """Persist the reading; a dashboard failure must not break the gate."""
        if self.retention is None:
            return
        rec = dict(ev)
        rec["model_step"] = int(self.learner.manifest.get("step") or 0)
        rec["live_steps"] = int(self.learner.steps)
        rec["chars"] = self.learner.manifest.get("read_chars")
        try:
            self.retention.append(rec)
        except OSError as e:
            print(f"  [retention] write failed: {e}", file=sys.stderr)

    def rebind(self, model, learner):
        """Follow a model/learner that was rebuilt (revert or pool change).

        The best score and snapshot stay: they describe the weights, not the
        Python objects holding them.
        """
        self.model, self.learner = model, learner
        self.device = next(model.parameters()).device
        self.ev = None
        self.last_step = int(learner.steps)
        if self.plasticity is not None:
            self.plasticity.step = int(learner.steps)
        self._apply_lr()

    def start(self):
        """Score the loaded state and make it the best one to return to."""
        if self.learner.unsaved:
            self.learner.save()
        val, se, doms = self._score()
        stored = self._read_meta()
        if stored is not None and float(stored["val"]) <= val:
            # The snapshot is from an earlier session and is still better
            # than the weights this session opened on. Keep it.
            self.best = float(stored["val"])
            self.best_se = float(stored.get("se") or 0.0)
            kept = True
        else:
            self.best, self.best_se = val, se
            self._snapshot()
            kept = False
        if self.plasticity is not None:
            self.plasticity.step = int(self.learner.steps)
            self.plasticity.observe(val, se=se)
            self._apply_lr()
        ev = {"action": "baseline", "at": time.time(), "val": val,
              "se": se, "best": self.best, "domains": doms,
              "lr_scale": self.learner.scale, "kept_best": kept,
              "best_dir": self.best_dir}
        self.last, self.last_step = ev, int(self.learner.steps)
        self.events.append(ev)
        self._record(ev)
        self._say(f"baseline {val:.4f} over {len(doms)} domains; "
                  f"best {self.best:.4f}"
                  + (" (kept the earlier snapshot)" if kept
                     else f"; best snapshot in {self.best_dir}"))
        return ev

    def maybe_check(self, every=32):
        """Check if `every` live steps have passed since the last one."""
        if self.learner.steps - self.last_step < max(1, int(every)):
            return None
        # Advance before the check: one reading, not one on every subsequent
        # exchange until the next threshold.
        self.last_step = int(self.learner.steps)
        return self.check()

    def check(self):
        if self.best is None:
            return self.start()
        val, se, doms = self._score()
        if self.plasticity is not None:
            self.plasticity.step = int(self.learner.steps)
            note = self.plasticity.observe(val, se=se)
        else:
            note = None
        # The factor is the material cap. A noisy instrument must not widen
        # it: at four 512-character chunks the standard error is ~0.037, so
        # three of them (~0.11) would otherwise let an 11% drift look "ok".
        # A precise reading still tightens the limit; the smaller wins.
        limit = min(self.best * self.factor,
                    self.best + 3.0 * max(se, self.best_se, 1e-9))
        ev = {"action": "ok", "at": time.time(), "val": val, "se": se,
              "best": self.best, "limit": limit, "domains": doms,
              "note": note}

        if val < self.best:
            if self.learner.unsaved:
                self.learner.manifest["val"] = val
                self.learner.save()
            self.best, self.best_se = val, se
            self._snapshot()
            ev.update(action="improved", best=val)
            self._say(f"held-out {val:.4f} - improved, snapshot advanced")
        elif self.factor > 0 and val > limit:
            ev["action"] = "reverted"
            try:
                self._revert()
                ev["best"] = self.best
                self._say(f"held-out {val:.4f} is past the limit "
                          f"{limit:.4f} (best {self.best:.4f}) - reverted, "
                          f"rate x{self.learner.scale:.3f}")
            except Exception as e:                            # noqa: BLE001
                ev.update(action="revert-failed", error=str(e))
                self._say(f"revert failed: {e}")
        else:
            self._say(f"held-out {val:.4f} (best {self.best:.4f}, "
                      f"limit {limit:.4f})")

        self._apply_lr()
        ev["lr_scale"] = self.learner.scale
        self.last = ev
        self.events.append(ev)
        self._record(ev)
        return ev

    def _revert(self):
        if self.rebuild is None and self.paged:
            raise RuntimeError("a paged model needs a rebuild hook to revert")
        if self.plasticity is not None:
            self.plasticity.scale = max(self.plasticity.FLOOR,
                                        self.plasticity.scale * 0.5)
        # The pool pages experts from the live directory, so the best state
        # must become that directory, not merely be loaded once: otherwise the
        # next forward pages the diverged experts back in.
        _replace_dir(self.best_dir, self.weights_dir)
        if self.rebuild is not None:
            model, learner = self.rebuild(self.weights_dir)
            self.model, self.learner = model, learner
            self.device = next(model.parameters()).device
            self.ev = None
        else:
            man, _, _ = store.load(self.model, self.weights_dir,
                                   opt=self.learner.opt, device=self.device)
            pool = getattr(self.model, "pool", None)
            if pool is not None and hasattr(pool, "invalidate"):
                pool.invalidate()
            self.learner.manifest.clear()
            self.learner.manifest.update(man)
            self.learner.reset_stream()
        if self.plasticity is None:
            self.learner.scale_lr(0.5)
        self.learner.unsaved = 0
        self.last_step = int(self.learner.steps)
        if self.plasticity is not None:
            # The step counter restarts with the rebuilt learner; without
            # clearing this the controller would ignore the next readings
            # until it reached the old count.
            self.plasticity.seen_at = None
            self._apply_lr()

    def state(self):
        return {"best": self.best, "best_se": self.best_se,
                "best_dir": self.best_dir, "last": self.last,
                "factor": self.factor, "chunks": self.chunks,
                "chunk": self.chunk, "context": self.context,
                "retention_log": (self.retention.path
                                  if self.retention is not None else None),
                "lr_scale": getattr(self.learner, "scale", None),
                "lr": (self.plasticity.state()["scale"]
                       if self.plasticity is not None else None)}
