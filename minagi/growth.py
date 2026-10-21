"""Grow and prune the expert pool while serving, on held-out evidence.

The pool is allowed to change size during a live session.  That is a model
change, not a learning step, so it rides the same watchdog as the optimizer:
a check that grows the pool is followed by the next held-out score, and a pool
that made held-out worse is reverted with everything else.

Nothing here decides on its own.  `AutoGrow` (minagi/pool.py) owns the growth
brakes and `PagedPool.prune` owns the deletion rule; this class only sequences
them at the cadence the server asks for, translates the live step counter into
the checkpoint's model step, and reports what changed so the caller can save and
rebuild.
"""


class PoolKeeper:
    """Run prune/grow after a held-out check, at a chosen cadence.

    `base_step` is the checkpoint's step when the session opened.  Live steps
    are counted from zero in every learner, but `born`, `last_seen`, in-flight
    trials and the growth log are all in model steps, so the two counters are
    added here and nowhere else.

    `every` is in held-out checks, not live steps: growth is expensive and the
    gate is the only thing that can judge it.
    """

    def __init__(self, model, learner, grower, survival=3000, base_step=0,
                 every=1, prune=True):
        self.model = model
        self.learner = learner
        self.pool = model.pool
        self.grower = grower
        self.survival = int(survival)
        self.base_step = int(base_step)
        self.every = max(0, int(every))
        self.prune = bool(prune)
        self.checks = 0
        self.events = []

    def model_step(self):
        return self.base_step + int(self.learner.steps)

    def after_check(self, val):
        """Called with a held-out mean; returns what changed or None."""
        self.checks += 1
        if not self.every or self.checks % self.every:
            return None
        step = self.model_step()
        out = {"step": step, "pruned": 0, "grew": 0}
        if self.prune:
            out["pruned"] = int(self.pool.prune(step,
                                                survival=self.survival) or 0)
        rec = self.grower.step(float(val), self.pool, step)
        out["grew"] = int(rec.get("grew", 0))
        out["reason"] = rec.get("reason", "")
        if not out["pruned"] and not out["grew"]:
            return None
        out["experts"] = int(self.pool.n_experts())
        self.events.append(out)
        return out

    def rebind(self, model, learner, base_step=None):
        """Point at the model and learner a save/reload just produced."""
        self.model, self.learner = model, learner
        self.pool = model.pool
        if base_step is not None:
            self.base_step = int(base_step)
