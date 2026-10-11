#!/usr/bin/env python3
"""
One window to watch the run in.

training_recent.png and experts.png each answered half the question, so
keeping an eye on a run meant switching between them - and they disagreed
about what step it was whenever one had been redrawn and the other had not.
This is both, on one canvas, nine panels in three rows:

    is it learning        loss, by domain, and what the rate controller thinks
    what the pool is      who does the work, who is near deletion, who is new
    how it is running      pool over time, window and speed, concentration

It reuses the parsers from the two tools it replaces rather than re-deriving
them, so there is one definition of how the log is read and one of how
staleness is computed. Reads the sample log and the checkpoint; no torch, no
GPU, so it runs while training does.

    python3 tools/plot_dashboard.py --log runs/samples.txt --weights weights
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import plot_progress as PP                                   # noqa: E402
import plot_experts as PE                                    # noqa: E402

INK, MUTED, GRID = "#1b1b1f", "#6b6b76", "#e4e4ea"
TRAIN, VAL, WORK = "#b4b4bc", "#2b7bba", "#2ca25f"
WARM, COOL, DEAD = "#d9a326", "#2d6ec2", "#c2492d"
DOMS = ["arithmetic", "chat", "chat_hermes", "chess", "code", "reasoning",
        "stories", "wikipedia", "self_knowledge", "self-knowledge"]


def frame(ax, title, sub=None):
    ax.set_title(title, fontsize=10, loc="left", color=INK, fontweight="bold",
                 pad=7)
    if sub:
        ax.set_xlabel(sub, fontsize=7.6, color=MUTED, labelpad=5, loc="left")
    ax.grid(alpha=.3, color=GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def segments(x):
    """Split where characters-read goes backwards, so restarts do not join up."""
    cut = [0] + [i for i in range(1, len(x)) if x[i] < x[i - 1]] + [len(x)]
    return [slice(cut[i], cut[i + 1]) for i in range(len(cut) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/samples.txt")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--history", default="runs/expert_history.jsonl")
    ap.add_argument("--last", type=float, default=100.0,
                    help="million characters to show, counted back from now")
    ap.add_argument("--out", default="runs/dashboard.png")
    a = ap.parse_args()

    rows = PP.parse(a.log) if os.path.exists(a.log) else []
    if not rows:
        print(f"  {a.log} has no progress blocks yet")
        return 1
    allx = np.array([r["chars"] for r in rows], float)
    keep = allx >= (allx.max() - a.last) if a.last else np.ones(len(rows), bool)
    rows = [r for r, k in zip(rows, keep) if k]
    x = np.array([r["chars"] for r in rows], float)
    segs = segments(x)

    C = PE.cfg(os.path.dirname(HERE))
    hist = PE.history(a.history)
    man = {}
    try:
        man = json.load(open(os.path.join(a.weights, "manifest.json")))
    except OSError:
        pass
    tel = man.get("telemetry") or {}

    fig = plt.figure(figsize=(17.5, 12.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.30,
                          left=0.05, right=0.975, top=0.905, bottom=0.055)

    def col(name):
        return np.array([r.get(name, np.nan) if r.get(name) is not None
                         else np.nan for r in rows], float)

    v, se, tr = col("val"), col("se"), col("train")

    # ------------------------------------------------ 1. is it learning
    ax = fig.add_subplot(gs[0, 0])
    for g in segs:
        ax.plot(x[g], tr[g], lw=1, color=TRAIN, label="train" if g == segs[0] else None)
        ax.fill_between(x[g], (v - se)[g], (v + se)[g], color=VAL, alpha=.18)
        ax.plot(x[g], v[g], lw=1.9, color=VAL,
                label="held-out" if g == segs[0] else None)
    if np.isfinite(v).any():
        i = int(np.nanargmin(v))
        ax.plot(x[i], v[i], "o", ms=6, color=DEAD, zorder=5)
        ax.annotate(f"best {v[i]:.4f}", (x[i], v[i]), fontsize=8, color=DEAD,
                    textcoords="offset points", xytext=(6, 6))
    frame(ax, "Is it learning", "nats per character")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8, frameon=False)

    # ------------------------------------------------ 2. by domain
    ax = fig.add_subplot(gs[0, 1])
    names = [d for d in DOMS if any((r.get("dom") or {}).get(d) for r in rows)]
    cm = plt.get_cmap("tab10")
    for j, d in enumerate(names):
        y = np.array([(r.get("dom") or {}).get(d, np.nan) for r in rows], float)
        for g in segs:
            ax.plot(x[g], y[g], lw=1.2, color=cm(j % 10),
                    label=d if g == segs[0] else None)
    frame(ax, "Which domains are moving", "held-out per domain, nats")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if names:
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="best")

    # ----------------------------------- 3. what the rate controller thinks
    ax = fig.add_subplot(gs[0, 2])
    lr = col("lr")
    for g in segs:
        ax.plot(x[g], lr[g], lw=1.8, color=VAL)
    ax.set_ylabel("lr", color=VAL, fontsize=8.5)
    ax.set_yscale("log")
    q = ax.twinx()
    ev = col("evidence")
    for g in segs:
        q.plot(x[g], ev[g], lw=1, color="#9467bd")
    # the controller's verdict against T_MID in minagi/plasticity.py: above
    # it the rate eases up, below it down (the rows the plateau controller
    # wrote on 10 October carry its improvement in sd here instead)
    at = 2.2
    q.axhline(at, color="#9467bd", ls="--", lw=.9, alpha=.7)
    q.text(x.max(), at, " raises above here", fontsize=7, color="#9467bd",
           ha="right", va="bottom")
    q.set_ylabel("evidence t", color="#9467bd", fontsize=8)
    q.tick_params(labelsize=7)
    q.spines["top"].set_visible(False)
    frame(ax, "Learning rate, and why", "eased at every check unless held-out is clearly improving")

    # -------------------------------------------- 4. who does the work
    g_ = np.abs(np.asarray(tel.get("gate") or [], float))
    n = len(g_)
    if n:
        # THE SAME WINDOW AS EVERY OTHER PANEL. This used to window on
        # prune.survival_chars, which is a property of the pruning rule and
        # has nothing to do with how much of the run the dashboard is
        # showing. The two agreed at 100M only because both happened to be
        # 100M, so a dashboard drawn with --last 20 would have put a 20M
        # curve beside a 100M bar chart and said nothing about it.
        w, span = PE.window_use(hist, {**tel, "chars": man.get("read_chars", 0),
                                       "step": man.get("step", 0)},
                                a.last * 1e6)
        share = w / max(w.sum(), 1e-9)
        order = np.argsort(-share)
        ax = fig.add_subplot(gs[1, 0])
        ax.bar(np.arange(n), 100 * share[order], width=1.0,
               color=plt.get_cmap("viridis")(g_[order] / max(g_.max(), 1e-9)))
        top = int(np.searchsorted(np.cumsum(share[order]), 0.99)) + 1
        ax.axvline(top - .5, color=DEAD, ls="--", lw=1.1)
        ax.text(top + 1, ax.get_ylim()[1] * .82, f" {top} carry\n 99%",
                fontsize=8, color=DEAD)
        frame(ax, f"Who does the work  ({n} experts)",
              f"routing share over the last {span:.0f}M characters, colour is the gate")
        ax.set_ylabel("% of routing", fontsize=8.5)

        # ------------------------------------- 5. who is near deletion
        # Unmasked. The rule reads 0 for anything inside its first survival
        # window, and for the whole first window of a run that is EVERYTHING -
        # so the masked version draws an empty panel for a day while the
        # staleness accumulates underneath it. Protected experts are drawn in
        # the trial colour, so the shape is visible and it is still obvious
        # that prune cannot act on it yet.
        d_, young = PE.dying({**tel, "chars": man.get("read_chars", 0),
                              "step": man.get("step", 0)}, C, mask_young=False)
        ax = fig.add_subplot(gs[1, 1])
        o2 = np.argsort(-d_)
        cols = [COOL if y else (DEAD if t >= 1 else WARM if t >= C["dying_at"]
                                else WORK)
                for t, y in zip(d_[o2], young[o2])]
        ax.bar(np.arange(n), d_[o2], width=1.0, color=cols)
        ax.axhline(C["dying_at"], color=WARM, ls="--", lw=1.1)
        ax.axhline(1.0, color=DEAD, ls="--", lw=1.1)
        ax.text(n * .98, C["dying_at"] + .02, "dying", fontsize=7.5,
                color=WARM, ha="right")
        ax.text(n * .98, 1.02, "deleted", fontsize=7.5, color=DEAD, ha="right")
        ax.set_ylim(0, max(1.2, float(d_.max()) * 1.1))
        at_risk = int(((d_ >= C["dying_at"]) & ~young).sum())
        prot = int(young.sum())
        ax.set_ylabel("share of the survival window", fontsize=8.5)
        frame(ax, f"Near deletion  ({at_risk} dying"
                  + (f", {prot} still protected" if prot else "") + ")",
              # the caveat only when it is one: with the two windows equal,
              # "against the 100M survival window - not the last 100M" is a
              # sentence that gives the reader two identical numbers and
              # tells them they differ
              ("time unasked-for, as a share of the survival window"
               if abs(C["survival_chars"] / 1e6 - a.last) < 1 else
               f"time unasked-for, against the "
               f"{C['survival_chars']/1e6:.0f}M survival window - not the "
               f"last {a.last:.0f}M"))

        # -------------------------------- 6. are the newcomers earning
        born = np.asarray((tel.get("born") or [0] * n)[:n], float)
        rank = np.argsort(np.argsort(born))
        ax = fig.add_subplot(gs[1, 2])
        ax.scatter(rank[~young], 100 * share[~young], s=22, color=WORK,
                   edgecolor="none", alpha=.9, label="past its trial")
        if young.any():
            ax.scatter(rank[young], 100 * share[young], s=30, color=COOL,
                       edgecolor="none", label="on trial")
        founders = int((born <= born.min()).sum())
        if 0 < founders < n:
            ax.axvline(founders - .5, color=MUTED, ls=":", lw=1)
            ax.text(founders + 1, ax.get_ylim()[1] * .9, f" {founders} founders",
                    fontsize=7.5, color=MUTED)
        frame(ax, "Are the newcomers earning their place",
              "a newcomer with no share and no trial left is what prune removes")
        ax.set_ylabel("% of routing", fontsize=8.5)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")

        # ------------------------------------------- 9. concentration
        ax = fig.add_subplot(gs[2, 2])
        cum = 100 * np.cumsum(share[order])
        ax.plot(np.arange(1, n + 1), cum, lw=2, color=WORK)
        ax.plot([1, n], [100 / n, 100], color=MUTED, ls="--", lw=1)
        for qq, cc in ((50, COOL), (90, WARM), (99, DEAD)):
            k = int(np.searchsorted(cum, qq)) + 1
            ax.plot([k], [qq], "o", ms=5, color=cc)
            ax.annotate(f"{qq}% from {k}", (k, qq), fontsize=7.5, color=cc,
                        textcoords="offset points", xytext=(7, -8))
        frame(ax, "How concentrated the routing is",
              f"{C['top_k']} chosen per character, {C['resident']} on the card; "
              f"dashed is perfectly even")
        ax.set_xlabel("experts, busiest first", fontsize=8.5)

    # ------------------------------------------- 7. the pool over time
    ax = fig.add_subplot(gs[2, 0])
    ex = col("experts")
    for g in segs:
        ax.plot(x[g], ex[g], lw=1.8, color=WORK)
    ax.set_ylabel("experts", color=WORK, fontsize=8.5, labelpad=1)
    if hist:
        q = ax.twinx()
        hx = np.array([r["chars"] / 1e6 for r in hist], float)
        hy = np.array([PP._dead_pct(r) for r in hist], float)
        m = (hx >= x.min()) & (hx <= x.max())
        if m.any():
            q.plot(hx[m], hy[m], lw=1, color=WARM)
        q.set_ylim(0, max(5.0, float(np.nanmax(hy[m])) * 1.3 if m.any() else 5.0))
        q.set_ylabel("% dying", color=WARM, fontsize=8, labelpad=1)
        q.tick_params(labelsize=7)
        q.spines["top"].set_visible(False)
    frame(ax, "The pool over time")
    ax.set_xlabel("million characters read", fontsize=8.5)

    # -------------------------------------- 8. the window, and the speed
    ax = fig.add_subplot(gs[2, 1])
    ctx = col("context")
    for g in segs:
        ax.plot(x[g], ctx[g], lw=1.8, color="#8c564b")
    ax.set_ylabel("window", color="#8c564b", fontsize=8.5)
    q = ax.twinx()
    sp = col("rate")
    for g in segs:
        q.plot(x[g], sp[g], lw=.9, color=MUTED, alpha=.8)
    q.set_ylabel("char/s", color=MUTED, fontsize=8)
    q.tick_params(labelsize=7)
    q.spines["top"].set_visible(False)
    now_ctx = rows[-1].get("context")
    frame(ax, f"Window and reading speed"
              + (f"  (now {now_ctx:,})" if now_ctx else ""),
          f"ceiling {C.get('context_end', 0) or man.get('cfg', {}).get('block', 0):,}")
    ax.set_xlabel("million characters read", fontsize=8.5)

    last = rows[-1]
    fig.suptitle(
        f"mini-AGI  ·  step {last['step']:,}  ·  {last['chars']:.1f}M characters"
        + (f" of {last['corpus']:,.0f}M ({100*last['chars']/last['corpus']:.2f}%)"
           if last.get("corpus") else "")
        + (f"  ·  {n} experts" if n else "")
        + f"  ·  showing the last {x.max() - x.min():.0f}M"
        + (f" of {a.last:.0f}M asked for" if x.max() - x.min() < a.last * 0.9
           else ""),
        x=0.05, y=0.965, ha="left", fontsize=15, color=INK, fontweight="bold")
    bits = []
    if np.isfinite(v).any():
        bits.append(f"held-out {v[-1]:.4f} nats ({v[-1] / np.log(2):.3f} "
                    f"bits/char), best {np.nanmin(v):.4f}")
    if last.get("train") is not None and np.isfinite(v).any():
        bits.append(f"gap {v[-1] - last['train']:+.3f}")
    if last.get("lr"):
        bits.append(f"lr {last['lr']:.2e}")
    fig.text(0.05, 0.932, "   ·   ".join(bits), fontsize=10, color=MUTED,
             ha="left")
    fig.savefig(a.out, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {a.out}")
    if np.isfinite(v).any():
        print(f"  held-out {v[-1]:.4f}  best {np.nanmin(v):.4f}"
              + (f"  {n} experts" if n else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
