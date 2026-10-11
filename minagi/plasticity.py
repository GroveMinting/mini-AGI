"""
The learning rate, governed by held-out loss rather than by a horizon.

A cosine schedule asserts that the run ends. For a model that reads
continually that assertion is false, and it fails in a specific way: when the
regime changes - a new corpus, a change of shape - a count of characters
already read says the model is finished when it has in fact just started
again.

TWO RULES, AND THEY POINT IN OPPOSITE DIRECTIONS. That symmetry is the whole
point; a cosine can only ever go down.

  EASING    the rate is nudged at EVERY evaluation by an amount that varies
            smoothly with the evidence, rather than stepping every eighth one.
            The evidence is an exponentially weighted least-squares fit of
            held-out against evaluation index - no window, so nothing ever
            falls off an edge and produces a jump in the signal that becomes a
            jump in the rate.

            THE EVIDENCE IS AN EFFECT SIZE, NOT A t-STATISTIC. A t-statistic
            is the wrong variable here: se(slope) shrinks as the fit
            accumulates weight, so given enough observations ANY downward drift
            becomes significant. It measures how long the controller has been
            watching, not how much has improved, and pins the rate at the
            ceiling.

            So the deciding number is
                e = -slope / sigma          improvement per evaluation, in
                                            units of the residual scatter
            which is scale-free and does NOT grow as evidence accumulates. The
            significance test is kept as a CAP rather than as the signal, since
            acting on a slope that could be noise is still wrong, so the rate
            moves on min(t, EFFECT * e).

            A second fit at LAM_FAST caps it again. The slow fit remembers ~66
            evaluations, so when improvement stops it goes on reporting the old
            decline for tens of evaluations. The fast fit notices in about 12,
            and min() takes whichever is less impressed.

            The nudge itself is Plasticity.nudge: about x0.9986 per evaluation
            where there is no evidence, x0.996 at the most where held-out is
            getting worse, up to x1.0008 where improvement is clear, and
            continuous everywhere between. So the resting posture is a gentle
            decay - on a flat run about 0.2% an evaluation, a tenth or less in
            six hours of ten-minute rounds - and only clear improvement holds
            the rate up. No nudge can move it more than MAX_MOVE, 2%.

  REGIME    held-out jumped by several standard errors and stayed there ->
            step the rate back up. New material, or a change of shape. It has
            to persist to count, so a single noisy evaluation cannot trigger
            it. This is the same detector `tools/plot_progress.py` uses to
            decide where to fit its trend.

THREE THINGS ARE NOT EVIDENCE, and each once moved the rate by itself:

  A READING THAT DOES NOT HOLD. One further from the slow fit's line than
            SPIKE times the fit's own scatter is held back until the next
            evaluation. If that one lands nearer the held reading than the
            line - the level really moved - both count. If not, it was a bad
            measurement and never counts. The run's spikes are single readings
            0.01 to 0.02 above their neighbours and gone by the next check:
            many times the scatter, and far below anything the regime rule
            sees. Once, a single broken evaluation (1.126 among 0.73s) cut the
            rate from x0.38 to x0.21.
  NO READING. An evaluation with fewer than MIN_STEPS optimiser steps since the
            last one measures the same model again. The x axis counts
            evaluations, so thirty back-to-back rounds with nothing read read
            as thirty rounds of no progress, and each eased the rate.
  A RESTART. A restart is where settings change, and capacity_factor 0 moved
            held-out by 0.006 within one evaluation. The first RELEVEL
            readings after a restore are held back; the step is how far they
            sit, by their median, from where the fast fit's line was heading;
            both fits' history is shifted by it, and only then are they added.
            A changed setting is then neither a gain nor a harm: the trend
            carries on, and the level is the new one. The median, because the
            first readings of a session are where a spike is likeliest. A step
            the size of a regime change is left for the regime rule.

Nothing here is a hyperparameter the user has to set. The thresholds are read
off the measured standard error of the evaluation and the scatter of the run
itself, so the controller tightens automatically as either gets less noisy.
"""

import math
import re
from collections import deque


def _sums():
    return dict(w=0.0, w2=0.0, x=0.0, y=0.0, xx=0.0, xy=0.0, yy=0.0)


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class Plasticity:
    FLOOR = 0.01        # the rate is never allowed to reach zero
    CEIL = 1.0
    LAM = 0.97          # decay per evaluation; n_eff -> (1+L)/(1-L) = 65.7
    LAM_FAST = 0.85     # the second, shorter fit; n_eff -> 12.3. Not lower:
                        # below MIN_EFF the fast fit never engages at all and
                        # the cap silently stops existing.
    EFFECT = 45.0       # converts e into the same units as t, so the two can
                        # be compared by min(). Calibrated so that typical
                        # real-run evidence sits just BELOW T_MID, making
                        # gentle decay the resting posture.
    MIN_EFF = 12.0      # effective observations before it will act at all
    NUDGE = 0.0008      # log-scale gain per evaluation, going UP
    NUDGE_DOWN = 0.004  # ...and coming down. Deliberately larger - see nudge().
                        # Both 0.16 of the 19 September values, which on a flat
                        # run eased the rate ~1.7% an evaluation and halved it
                        # in six hours of ten-minute rounds. These ease it
                        # ~0.2%, a tenth or less in six hours
    T_W_DOWN = 6.0      # the width on the way down, over the range the verdict
                        # actually reaches (it runs to about -12)
    T_MID = 2.2         # the verdict at which it neither eases up nor down.
                        # DELIBERATELY ABOVE THE MIDDLE, for two reasons.
                        #
                        # The errors are not symmetric. Overshooting the rate
                        # costs a fraction of a nat and tens of millions of
                        # characters to repair; undershooting only costs time.
                        # So slow-but-real progress reads as slight decay, and
                        # only clearly better progress buys more rate.
                        #
                        # And `e` has a bias to correct. It is slope over
                        # RESIDUAL SCATTER, and that scatter is the model's own
                        # checkpoint-to-checkpoint wobble rather than the
                        # evaluation's error. The wobble falls over a run as
                        # the pool stops churning, while the evaluation error
                        # barely moves - so a run that merely gets QUIETER
                        # starts scoring as a run that is improving. Sitting
                        # the neutral point high is what absorbs that.
                        #
                        # This does not disable raising: real evidence still
                        # clears it, a few evaluations in ten rather than a
                        # third of them.
    T_W = 0.75          # how sharply it responds ABOVE that
    MAX_MOVE = 0.02     # no single nudge moves the rate by more than this,
                        # whatever the gains are set to. The gains above stay
                        # far inside it: 0.4% down, 0.08% up at the most
    JUMP_SE = 4.0       # standard errors that count as a regime change
    HOLD_SE = 2.0       # ...and it has to still be up here next time
    FLOOR_JUMP = 0.25   # a jump this large is a regime change whatever the noise
    UP = 2.0            # what a confirmed regime change restores
    WARMUP = 100        # optimiser steps, in case the moments are not restored
    MIN_STEPS = 32      # optimiser steps since the last evaluation for a new
                        # one to count as evidence (a round reads ~400)
    SPIKE = 5.0         # residual sigmas off the slow fit's line at which a
                        # reading is held back until the next one
    RELEVEL = 5         # readings after a restart whose median sets its step
    SEED = 256          # readings a state without fits is rebuilt from; the
                        # slow fit weighs the oldest of them at 0.0004

    def __init__(self, scale=1.0, best=None):
        self.scale = float(scale)
        # Two exponentially weighted least-squares fits of held-out against
        # evaluation index, kept as running sums. There is no window: every
        # past reading still counts, weighted by LAM ** age, so nothing ever
        # falls off an edge. A hard window of length N steps whenever its
        # oldest point drops out, which is a jump in the SIGNAL that then
        # becomes a jump in the rate - the staircase this replaced.
        #
        # The slow fit is the evidence. The fast one exists only to notice
        # sooner when improvement has stopped, and can only ever lower the
        # verdict - see observe().
        self.S = _sums()
        self.F = _sums()
        self.i = 0.0                   # evaluation counter, the x axis
        self.se_hist = deque(maxlen=64)
        self.prev = None               # the last reading the fits took
        self.held = None               # a reading waiting on the next one
        self.pending = []              # readings since a restart, not yet in
        self.events = []
        self.step = 0
        self.seen_at = None            # self.step at the last evaluation taken
        self._resumed = False          # restored: re-level before adding more
        self.last_t = 0.0
        self.last_e = 0.0

    # ---------------------------------------------------------------- lr
    def factor(self):
        """What to multiply every group's base rate by, right now."""
        w = min(1.0, (self.step + 1) / self.WARMUP) if self.step < self.WARMUP \
            else 1.0
        return self.scale * w

    def tick(self):
        self.step += 1

    # ------------------------------------------------------------ evidence
    def _se(self):
        return _median(self.se_hist) if self.se_hist else 0.0

    def _accumulate(self, y):
        """Add one reading to BOTH fits. They share the x axis."""
        self.i += 1.0
        x = self.i
        for S, lam in ((self.S, self.LAM), (self.F, self.LAM_FAST)):
            for k in ("w", "x", "y", "xx", "xy", "yy"):
                S[k] *= lam
            S["w2"] *= lam * lam
            S["w"] += 1.0
            S["w2"] += 1.0
            S["x"] += x
            S["y"] += y
            S["xx"] += x * x
            S["xy"] += x * y
            S["yy"] += y * y
        self.prev = y

    @staticmethod
    def _fit(S):
        """
        (t, n_eff, e) for one set of running sums.

        t is the slope over its own standard error - the significance. e is
        the slope over the RESIDUAL SCATTER - the effect size, improvement per
        evaluation in units of the noise it has to be seen through. Both are
        signed so that positive means held-out is falling.

        The difference between them is the whole point of this file. t carries
        a sqrt(sxx) that grows until the window fills and then stays large, so
        t rewards patience rather than progress. e carries no such factor.
        """
        if S["w"] <= 2:
            return 0.0, 0.0, 0.0
        n_eff = S["w"] ** 2 / max(S["w2"], 1e-12)
        sxx = S["xx"] - S["x"] ** 2 / S["w"]
        sxy = S["xy"] - S["x"] * S["y"] / S["w"]
        syy = S["yy"] - S["y"] ** 2 / S["w"]
        if sxx <= 0 or n_eff <= 2:
            return 0.0, n_eff, 0.0
        slope = sxy / sxx
        resid = max(syy - slope * sxy, 0.0)
        sigma = math.sqrt(resid / (n_eff - 2)) if n_eff > 2 else 0.0
        if sigma <= 0:
            # A perfectly straight run has no residual, so neither statistic
            # is defined. Report nothing rather than infinity; real held-out
            # is never this clean, and a run that is has no noise to see
            # through and needs no help from here.
            return 0.0, n_eff, 0.0
        t = -slope / (sigma / math.sqrt(sxx))
        e = -slope / sigma
        return max(-50.0, min(50.0, t)), n_eff, e

    @staticmethod
    def _line(S, x):
        """
        (where one fit's line puts evaluation x, the fit's residual scatter,
        its n_eff), or None while it has no line.
        """
        if S["w"] <= 2:
            return None
        n_eff = S["w"] ** 2 / max(S["w2"], 1e-12)
        sxx = S["xx"] - S["x"] ** 2 / S["w"]
        if sxx <= 0 or n_eff <= 2:
            return None
        sxy = S["xy"] - S["x"] * S["y"] / S["w"]
        syy = S["yy"] - S["y"] ** 2 / S["w"]
        slope = sxy / sxx
        sigma = math.sqrt(max(syy - slope * sxy, 0.0) / (n_eff - 2))
        return S["y"] / S["w"] + slope * (x - S["x"] / S["w"]), sigma, n_eff

    def _trend(self):
        """(t, effective observations) for the slow fit. Kept for callers."""
        t, n_eff, _ = self._fit(self.S)
        return t, n_eff

    def _verdict(self):
        """
        The number the rate actually moves on, and the two it is built from.

        min() of three readings, because each can only ever say "less than you
        think" and none may say "more":

          t_slow          is it distinguishable from noise at all - the
                          significance guard, taken from the better-powered
                          fit because that is what it is for
          EFFECT * e_slow is it big enough to be worth a higher rate
          EFFECT * e_fast ...and is it still happening now

        Only the SLOW t appears. t depends on how much window it was measured
        over - se(slope) shrinks as sqrt(sxx) - so a deliberately short fit has
        a mechanically small t and would veto everything regardless of what the
        loss is doing. The fast fit contributes its EFFECT SIZE instead, which
        does not know how long the window was.
        """
        t_s, n_s, e_s = self._fit(self.S)
        _, n_f, e_f = self._fit(self.F)
        t = min(t_s, self.EFFECT * e_s)
        if n_f >= self.MIN_EFF:
            t = min(t, self.EFFECT * e_f)
        return max(-50.0, min(50.0, t)), n_s, e_s

    @classmethod
    def nudge(cls, t):
        """
        The factor one verdict moves the rate by.

        ASYMMETRIC, AND THAT IS THE POINT. Going up and coming down are not
        the same question. A rate that is too high SHOWS you - held-out turns
        and keeps turning - so deterioration is direct evidence of overshoot
        and should be corrected in proportion to how bad it is. A rate that is
        merely working tells you nothing about whether a higher one would work
        better, so upward is a slow probe and stays at NUDGE.

        The two halves therefore need different WIDTHS, not just different
        gains. tanh saturates by |arg| = 2, so a single narrow width would make
        every verdict below about -0.3 produce an identical step: the
        controller would see a large fall and a small one and answer both at
        the same speed. T_W_DOWN spreads the response across the range the
        verdict actually reaches (down to roughly -12), so how bad the
        deterioration is reaches the rate.
        """
        up = t >= cls.T_MID
        gain = cls.NUDGE if up else cls.NUDGE_DOWN
        width = cls.T_W if up else cls.T_W_DOWN
        f = math.exp(gain * math.tanh((t - cls.T_MID) / width))
        return max(1.0 - cls.MAX_MOVE, min(1.0 + cls.MAX_MOVE, f))

    # -------------------------------------------------------------- guards
    def _relevel(self, ys):
        """
        A RESTART: shift both fits' history by the step the restart made, so
        the readings after it carry on from the readings before it.

        The step is the median distance of the readings after the restart from
        the FAST fit's line - where the last dozen evaluations were heading -
        so one bad reading among them cannot set it. Not the slow fit's line: a
        learning curve that is flattening, or a run that turned worse just
        before it was stopped, sits off the slow line at the end, and taking
        that for a step would erase it at every restart. Measured on the
        verdict: restarting a flattening run every 25 evaluations, with no step
        at all, moves it by +0.03 (by +1.44 read off the slow line), and a
        +0.006 step at a restart leaves it at +2.4, where it is without the
        step - carried as a trend, the same step reads -7.0.

        A step as large as a regime change is not a setting, and is left for
        the regime rule to judge.
        """
        gaps = []
        for k, y in enumerate(ys):
            line = (self._line(self.F, self.i + 1.0 + k)
                    or self._line(self.S, self.i + 1.0 + k))
            if line is not None:
                gaps.append(y - line[0])
        if not gaps:
            return
        d = _median(gaps)
        if d > max(self.JUMP_SE * self._se(), self.FLOOR_JUMP):
            return
        for S in (self.S, self.F):
            S["yy"] += 2.0 * d * S["y"] + d * d * S["w"]
            S["xy"] += d * S["x"]
            S["y"] += d * S["w"]
        if self.prev is not None:
            self.prev += d

    def _take(self, y):
        """
        One reading through the spike and regime guards and into the fits.
        Returns (readings the fits took, a note or None, whether the regime
        rule fired).
        """
        s = self._se()
        h, self.held = self.held, None
        note = None
        if h is not None:
            if h["jump"]:
                # ---- REGIME: a jump, confirmed on the following evaluation
                if y > h["from"] + self.HOLD_SE * max(s, 1e-9):
                    before = self.scale
                    self.scale = min(self.CEIL, self.scale * self.UP)
                    self.events.append({"at": self.step, "kind": "regime",
                                        "val": y, "scale": self.scale})
                    self.S, self.F, self.i = _sums(), _sums(), 0.0
                    self._accumulate(y)
                    return 1, (f"held-out moved to {y:.4f} from "
                               f"{h['from']:.4f} and stayed - new regime, "
                               f"rate {before:.3f} -> {self.scale:.3f}"), True
            else:
                # Did the level move, or was it one bad reading? Whichever of
                # the two this reading lands nearer to.
                line = self._line(self.S, self.i + 2.0)
                if line is not None and (y - line[0]) / h["off"] > 0.5:
                    self._accumulate(h["val"])
                    self._accumulate(y)
                    return 2, None, False
            note = (f"held-out {h['val']:.4f} did not hold ({y:.4f} next) - "
                    f"not evidence")
        # ---- is this reading one to hold back?
        if self.prev is not None and \
                y - self.prev > max(self.JUMP_SE * s, self.FLOOR_JUMP):
            self.held = {"val": y, "from": self.prev, "jump": True}
            return 0, note, False
        line = self._line(self.S, self.i + 1.0)
        if line is not None and line[1] > 0 and line[2] >= self.MIN_EFF \
                and abs(y - line[0]) > self.SPIKE * line[1]:
            self.held = {"val": y, "from": self.prev, "jump": False,
                         "off": y - line[0]}
            return 0, note, False
        self._accumulate(y)
        return 1, note, False

    def _feed(self, val):
        """
        Everything one reading does except move the rate: a restart's
        re-level, then the guards. Returns (readings the fits took, notes,
        whether the regime rule fired).
        """
        if self._resumed and self._line(self.S, self.i + 1.0) is not None:
            self.pending.append(val)
            if len(self.pending) < self.RELEVEL:
                return 0, [], False
            self._relevel(self.pending)
            ys, self.pending = self.pending, []
        else:
            ys, self.pending = [val], []
        self._resumed = False
        took, notes, regime = 0, [], False
        for y in ys:
            n, note, r = self._take(y)
            took += n
            regime = regime or r
            if note:
                notes.append(note)
        return took, notes, regime

    def observe(self, val, se=None):
        """
        One held-out evaluation. Returns a note if the rate moved notably, the
        regime rule fired, or a held-back reading did not hold.

        The rate is nudged once at every evaluation the fits take, by an
        amount that varies smoothly with the evidence - see nudge(). Deep in
        "no evidence" that is about x0.9986 per evaluation, compounding to
        x0.989 over eight: a drift rather than a staircase, so nothing the rate
        does is ever a shock to the run.
        """
        if val is None or not math.isfinite(float(val)):
            return None
        val = float(val)
        # NO READING, NO NEWS. Only counted where the caller counts steps
        # (tick()); a controller that is never ticked takes every evaluation.
        if (self.step > 0 and self.seen_at is not None
                and self.step - self.seen_at < self.MIN_STEPS):
            return None
        self.seen_at = self.step
        if se is not None and math.isfinite(float(se)) and float(se) > 0:
            self.se_hist.append(float(se))

        took, notes, regime = self._feed(val)
        t, n_eff, e = self._verdict()
        self.last_t = t
        self.last_e = e
        if took and not regime and n_eff >= self.MIN_EFF:
            before = self.scale
            self.scale = max(self.FLOOR, min(self.CEIL,
                                             self.scale * self.nudge(t)))
            # One line per evaluation would be noise. Record only when the
            # rate has drifted a full 2% since the last thing recorded.
            last = self.events[-1]["scale"] if self.events else 1.0
            if abs(math.log(self.scale / max(last, 1e-9))) > 0.02:
                kind = ("improving" if t > self.T_MID else
                        "deteriorating" if t < -self.T_MID else "settling")
                notes.append(f"{kind}: t={t:+.2f} (effect {e:+.3f} per "
                             f"evaluation) over {n_eff:.0f} effective "
                             f"evaluations, rate {before:.3f} -> "
                             f"{self.scale:.3f}")
                self.events.append({"at": self.step, "kind": kind,
                                    "val": val, "scale": self.scale,
                                    "t": round(t, 2)})
        return "; ".join(notes) or None

    def seed(self, readings):
        """
        Fits for a state that has none, from readings this controller did not
        take itself: oldest first, None where a session started. They pass
        the same guards and re-levels live readings do. Nothing is decided on
        them, the rate does not move, and the next live reading is treated as
        the restart it is.
        """
        scale, events = self.scale, list(self.events)
        if not self.has_fits():
            self.prev = None
        for v in list(readings)[-self.SEED:]:
            if v is None:
                self.held, self.pending = None, []
                self._resumed = True
            elif math.isfinite(float(v)):
                self._feed(float(v))
        self.scale, self.events = scale, events
        self.held, self.pending = None, []
        self._resumed = True

    @staticmethod
    def log_readings(text):
        """
        The readings in a sample log, oldest first, None where a session
        started. Each round's held-out check, and not the full evaluation a
        session ends on - the one block written without the controller's
        evidence line, a different measurement about 0.01 below the rounds.
        """
        return [None if m.group(1) else float(m.group(2)) for m in re.finditer(
            r"^(# session started)|^train loss [^\n]*evidence[^\n]*\n"
            r"held-out loss ([\d.]+)", text, re.M)]

    def has_fits(self):
        """Whether there is any evidence to go on, restored or seeded."""
        return self.S["w"] > 0

    # ------------------------------------------------------------- restart
    def state(self):
        t, n_eff, e = self._verdict()
        return {"scale": self.scale, "prev": self.prev,
                "S": dict(self.S), "F": dict(self.F), "i": self.i,
                "se": list(self.se_hist), "step": self.step,
                "seen_at": self.seen_at,
                "n": round(n_eff, 1), "t": round(t, 3), "e": round(e, 4),
                "events": self.events[-40:]}

    @classmethod
    def restore(cls, d):
        p = cls()
        if not d:
            return p
        p.scale = float(d.get("scale", 1.0))
        p.prev = d.get("prev")
        if isinstance(d.get("S"), dict):
            p.S.update({k: float(v) for k, v in d["S"].items() if k in p.S})
            p.i = float(d.get("i", 0.0) or 0.0)
            if isinstance(d.get("F"), dict):
                p.F.update({k: float(v) for k, v in d["F"].items()
                            if k in p.F})
            # A checkpoint that carries no F leaves the fast fit empty, and
            # that is correct. Seeding it from the slow fit would copy the slow
            # fit's n_eff with it, so the "fast" reading would just be the slow
            # one again until it decayed and the cap would never bind. Empty
            # costs nothing: MIN_EFF gates only this cap, not the controller,
            # so the slow fit keeps steering throughout.
        elif d.get("hist"):
            # an older checkpoint kept a plain window; replay it so the fit
            # starts from the evidence that was already gathered
            for v in d["hist"]:
                p._accumulate(float(v))
        # A state with neither - the controller that kept rotation levels
        # instead - starts with empty fits, and train.py seeds them from the
        # sample log.
        #
        # The warmup is there in case the moments were lost. They ARE restored
        # here, so re-running it on every resume only puts a notch in the rate
        # that nothing asked for.
        p.step = int(d.get("step", 0) or 0)
        p.seen_at = d.get("seen_at")
        for v in d.get("se") or []:
            p.se_hist.append(float(v))
        p.events = list(d.get("events") or [])
        p._resumed = True              # the first readings re-level the fits
        return p

    def describe(self):
        s = self._se()
        head = (f"learning rate is governed by held-out, not by a horizon - "
                f"eased at every check unless held-out is clearly improving, "
                f"by at most {1 - math.exp(-self.NUDGE_DOWN):.1%} a check: "
                f"rate x{self.scale:.3f}, floor x{self.FLOOR}")
        return head + (f", noise {s:.4f}" if s else "")
