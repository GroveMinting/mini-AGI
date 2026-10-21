#!/usr/bin/env python3
"""Longitudinal held-out retention from runs/retention.jsonl.

The gate already scores every domain on each check. This draws those readings
over the life of a run: what improved, what the gate reverted, and how far each
domain has moved from its first baseline. No torch, no GPU, no conversation
text.

    python3 serve.py                    # writes runs/retention.jsonl by default
    python3 tools/plot_retention.py --log runs/retention.jsonl \
        --out runs/retention.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from minagi.retention import load, summary                     # noqa: E402

INK, MUTED, GRID = "#1b1b1f", "#6b6b76", "#e4e4ea"
GOOD, BAD, WORK = "#2ca25f", "#c2492d", "#2b7bba"
BASELINE, MARK = "#8c564b", "#d9a326"


def segments(xs):
    cut = [0] + [i for i in range(1, len(xs)) if xs[i] < xs[i - 1]] + [len(xs)]
    return [slice(cut[i], cut[i + 1]) for i in range(len(cut) - 1)]


def frame(ax, title, sub=None):
    ax.set_title(title, fontsize=10, loc="left", color=INK, fontweight="bold",
                 pad=7)
    if sub:
        ax.set_xlabel(sub, fontsize=7.6, color=MUTED, labelpad=5, loc="left")
    ax.grid(alpha=.3, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def axis(rows):
    """Use characters read when they move, else the model step."""
    if all(r.get("chars") is not None for r in rows):
        xs = np.array([float(r["chars"]) for r in rows])
        if xs.max() > xs.min():
            return xs / 1e6, "million characters read"
    if all(r.get("model_step") is not None for r in rows):
        return np.array([float(r["model_step"]) for r in rows]), "model step"
    return np.arange(len(rows), dtype=float), "check"


def domain_names(rows):
    names = set()
    for r in rows:
        if isinstance(r.get("domains"), dict):
            names.update(r["domains"])
    return sorted(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default="runs/retention.jsonl")
    ap.add_argument("--out", default="runs/retention.png")
    ap.add_argument("--last", type=float, default=0,
                    help="show only this many million characters (0 = all)")
    a = ap.parse_args()

    rows = [r for r in load(a.log)
            if isinstance(r.get("domains"), dict) and r["domains"]]
    if not rows:
        print(f"  {a.log} has no gate readings yet")
        return 1
    x, xlabel = axis(rows)
    if a.last and xlabel.startswith("million"):
        keep = x >= x.max() - a.last
        rows = [r for r, k in zip(rows, keep) if k]
        x = x[keep]
        if not rows:
            print(f"  nothing in the last {a.last:.1f}M characters")
            return 1
    segs = segments(x)
    names = domain_names(rows)
    cmap = plt.get_cmap("tab10")
    val = np.array([float(r.get("val", np.nan)) if r.get("val") is not None
                    else np.nan for r in rows])
    se = np.array([float(r.get("se", 0) or 0) for r in rows])
    actions = [r.get("action") for r in rows]

    fig = plt.figure(figsize=(14.5, 10.5))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25,
                          left=0.055, right=0.975, top=0.90, bottom=0.08)

    # ------------------------------------------------ 1. the whole score
    ax = fig.add_subplot(gs[0, 0])
    for g in segs:
        ax.fill_between(x[g], (val - se)[g], (val + se)[g],
                        color=WORK, alpha=.15)
        ax.plot(x[g], val[g], lw=2, color=WORK, label="mean held-out")
    for action, color, marker in (("improved", GOOD, "o"),
                                  ("reverted", BAD, "X")):
        at = [i for i, q in enumerate(actions) if q == action]
        if at:
            ax.scatter(x[at], val[at], s=32, color=color, marker=marker,
                       zorder=5, label=action)
    if np.isfinite(val).any():
        i = int(np.nanargmin(val))
        ax.annotate(f"best {val[i]:.4f}", (x[i], val[i]), fontsize=8,
                    color=GOOD, textcoords="offset points", xytext=(6, 6))
    frame(ax, "Held-out over the run", xlabel)
    ax.set_ylabel("nats per character", fontsize=8.5)
    ax.legend(fontsize=8, frameon=False)

    # ------------------------------------------------ 2. every domain
    ax = fig.add_subplot(gs[0, 1])
    for j, name in enumerate(names):
        y = np.array([(r["domains"].get(name, np.nan))
                      if isinstance(r.get("domains"), dict) else np.nan
                      for r in rows], float)
        for g in segs:
            ax.plot(x[g], y[g], lw=1.3, color=cmap(j % 10), label=name)
    frame(ax, "Per-domain held-out", xlabel)
    ax.set_ylabel("nats per character", fontsize=8.5)
    if names:
        ax.legend(fontsize=7.5, frameon=False, ncol=2)

    # ------------------------------------- 3. change from the baseline
    ax = fig.add_subplot(gs[1, 0])
    for j, name in enumerate(names):
        for g in segs:
            base = None
            ys = []
            for r in rows[g]:
                if base is None and name in r["domains"]:
                    base = float(r["domains"][name])
                ys.append(float(r["domains"][name]) - base
                          if base is not None and name in r["domains"]
                          else np.nan)
            ax.plot(x[g], ys, lw=1.3, color=cmap(j % 10), label=name)
    ax.axhline(0, color=MUTED, lw=1)
    frame(ax, "Retention, against the first reading",
          f"{xlabel}; above zero means the domain got worse")
    ax.set_ylabel("change in nats", fontsize=8.5)

    # ------------------------------------- 4. last change per domain
    s = summary(rows)
    ax = fig.add_subplot(gs[1, 1])
    if s["domains"]:
        order = sorted(s["domains"], key=lambda n: s["domains"][n]["delta"])
        delta = [s["domains"][n]["delta"] for n in order]
        colors = [GOOD if d <= 0 else BAD for d in delta]
        ax.barh(np.arange(len(order)), delta, color=colors, alpha=.85)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.axvline(0, color=MUTED, lw=1)
        for i, d in enumerate(delta):
            ax.text(d, i, f" {d:+.4f}", va="center", fontsize=7.5,
                    color=INK)
        ax.set_xlabel("last minus first, nats", fontsize=8.5)
    frame(ax, "What moved, and which way",
          f"{len(rows)} readings; gate actions "
          f"{actions.count('improved')} improved / "
          f"{actions.count('reverted')} reverted")

    last = rows[-1]
    fig.suptitle(
        f"mini-AGI retention  ·  step {int(last.get('model_step') or 0):,}"
        f"  ·  {len(rows)} readings"
        + (f"  ·  best {s['domains'] and min(v['best'] for v in s['domains'].values()):.4f}"
           if s["domains"] else ""),
        x=0.055, y=0.96, ha="left", fontsize=14, color=INK,
        fontweight="bold")
    fig.text(0.055, 0.925,
             " ".join(f"{n} {v['first']:.4f}->{v['last']:.4f} "
                      f"({v['delta']:+.4f})"
                      for n, v in sorted(s["domains"].items())),
             fontsize=8.5, color=MUTED)
    fig.savefig(a.out, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"  wrote {a.out}")
    print(f"  {len(rows)} readings, {len(names)} domains")
    for name in names:
        v = s["domains"].get(name)
        if v:
            print(f"  {name:16s} {v['first']:.4f} -> {v['last']:.4f}  "
                  f"{v['delta']:+.4f}  best {v['best']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
