"""Longitudinal held-out retention, one JSONL row per gate decision.

The gate already scores every held-out domain. This module is the durable
version of those readings: domain names, losses, the best and the revert limit,
the action taken, and the step it happened at. It stores no conversation text.

The log is append-only by convention (not a security boundary). A restarted
run appends to the same file; readers keep file order and split at a backwards
character/step count.
"""

import json
import os
import time


class RetentionLog:
    def __init__(self, path):
        self.path = os.path.abspath(os.fspath(path))
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def append(self, event):
        rec = dict(event)
        rec.setdefault("at", time.time())
        line = json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n"
        with open(self.path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        return rec


def load(path):
    """Read the log, skipping a torn final line rather than losing the rest."""
    rows = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    rows.append(rec)
    except OSError:
        pass
    return rows


def summary(rows):
    """
    First/last held-out loss per domain, and its change.

    Positive delta means the domain got worse: this is the retention question
    the dashboard asks. `points` is how many readings that domain appears in.
    """
    with_domains = [r for r in rows if isinstance(r.get("domains"), dict)
                    and r["domains"]]
    out = {"rows": len(rows), "with_domains": len(with_domains),
           "first_at": None, "last_at": None, "domains": {}}
    if not with_domains:
        return out
    first, last = with_domains[0], with_domains[-1]
    out["first_at"] = first.get("at")
    out["last_at"] = last.get("at")
    names = sorted(set().union(*(r["domains"] for r in with_domains)))
    for name in names:
        vals = [float(r["domains"][name]) for r in with_domains
                if name in r["domains"]]
        if not vals:
            continue
        first_v = vals[0]
        last_v = vals[-1]
        out["domains"][name] = {
            "points": len(vals),
            "first": first_v,
            "last": last_v,
            "delta": last_v - first_v,
            "best": min(vals),
            "worst": max(vals),
        }
    return out
