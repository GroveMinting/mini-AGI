"""
The level a held-out check reports at, and how the sample log records it.

A CHECK IS A SAMPLE. Every round scores a few blocks of each subject - a
fraction of a percent of what the subject holds - and a few blocks sit where
they sit: on the whole set they read a little above or below it, by an amount
that stays put for as long as the blocks do. So a session starts with a
larger evaluation spread over the whole set, reads the checks' own blocks
beside it, and reports every check that session at the larger evaluation's
level: what the check reads, plus the difference the two had at the start.
A check then moves with the model, at the level of an evaluation four times
its size.

THE LOG SAYS SO. The session banner carries one line for each evaluation the
start ran - the larger one, the checks' blocks, and the check as it was read
before checks were spread, from the top of each subject's files:

    # held-out at the start, spread over the whole set: 0.5975   arithmetic 0.6090   ...
    # held-out at the start, the checks' blocks: 0.6043   arithmetic 0.6120   ...
    # held-out at the start, from the top as checks read before: 0.6110   ...

and whatever draws or reports held-out reads them: a check logged before the
first of these banners was read from the top, and is moved onto the new level
by the difference that banner measured, subject by subject.

The same again for a later change. Held-out once counted the leftover at the
end of each file as a block of its own; the first session to count whole
blocks only also read the old blocks, once, and wrote

    # held-out at the start, with each file's leftover part-block, as read before: 0.6012   ...

so everything logged before it moves by that difference too.

Pure Python, no torch, so the plotting tools can import it.
"""
import re

PREFIX = "# held-out at the start, "
SPREAD = "spread over the whole set"
CHECKS = "the checks' blocks"
TOP = "from the top as checks read before"
PARTS = "with each file's leftover part-block, as read before"

_LINE = re.compile(r"^# held-out at the start, ([^:\n]+): ([\d.]+)([^\n]*)$",
                   re.M)
_PAIR = re.compile(r"([a-z_]+) ([\d.]+)")


def banner(spread, checks=None, top=None, parts=None):
    """The banner's lines for one session start, from dicts of subject ->
    loss. A dict left out or empty writes no line."""
    out = []
    for label, d in ((SPREAD, spread), (CHECKS, checks), (TOP, top),
                     (PARTS, parts)):
        if d:
            out.append(PREFIX + f"{label}: {sum(d.values()) / len(d):.4f}"
                       + "".join(f"   {k} {v:.4f}"
                                 for k, v in sorted(d.items())))
    return "".join(line + "\n" for line in out)


def levels(text):
    """Every session start's held-out lines, oldest first:
    [(offset of its first line, {label: {subject: loss}})]."""
    groups = []
    for m in _LINE.finditer(text):
        label = m.group(1).strip()
        if label == SPREAD or not groups:
            groups.append((m.start(), {}))
        groups[-1][1][label] = {k: float(v)
                                for k, v in _PAIR.findall(m.group(3))}
    return groups


def first_shift(text):
    """Where checks first reported at the spread level, and how far a check
    read from the top sat above that level there, subject by subject:
    (offset, {subject: top - spread}). None if the log never says."""
    for at, g in levels(text):
        if g.get(SPREAD) and g.get(TOP):
            return at, {k: g[TOP][k] - g[SPREAD][k]
                        for k in g[SPREAD] if k in g[TOP]}
    return None


def part_shift(text):
    """Where held-out first counted whole blocks only, and how far the blocks
    read before - with each file's leftover part-block among them - sat from
    that level there, subject by subject: (offset, {subject: parts -
    spread}). None if the log never says. The session that made the change
    read both, on the same weights (minagi/stream.py, WHOLE BLOCKS ONLY)."""
    for at, g in levels(text):
        if g.get(SPREAD) and g.get(PARTS):
            return at, {k: g[PARTS][k] - g[SPREAD][k]
                        for k in g[SPREAD] if k in g[PARTS]}
    return None


def moved(val, dom, delta):
    """An old check (overall loss, {subject: loss} or None) on the spread
    level. The overall moves by the mean difference over the subjects the
    check itself has, so one from before a subject existed moves by its own
    subjects' share."""
    if not delta:
        return val, dom
    common = [k for k in (dom or {}) if k in delta] or list(delta)
    val = val - sum(delta[k] for k in common) / len(common)
    if dom:
        dom = {k: (v - delta[k] if k in delta else v) for k, v in dom.items()}
    return val, dom


def is_check(head):
    """Whether a log entry's header is a round's check. Only a check writes
    the rate controller's evidence line; the larger evaluation a session ends
    on does not."""
    return "evidence t " in head


# A log entry's header: step, characters read (M), corpus size, minutes into
# the session, experts.
LOG_HEAD = re.compile(
    r"^step ([\d,]+)\s+([\d.]+)M (?:of ([\d,]+)M )?characters"
    r"[^\n]*?(\d+) min(?:\s+(\d+) experts)?", re.M)


def parse_log(path):
    """Every entry in a sample log, oldest first, with held-out on one level.

    Checks moved to blocks spread over the whole held-out set partway through
    the run. An entry from before that read the top of each subject, and sits
    away from the level checks report at now by what the first spread session
    start measured: it is moved by that much, subject by subject, and marked
    `shifted`. The larger evaluation a session ended on before then was a
    third measurement, neither the checks' nor the new level, and is left out
    - only a check writes the controller's evidence line, so once a log has
    one, an entry before the change without it is one of those. After the
    change that evaluation is the level itself and stays.

    Here rather than in tools/plot_progress.py so the trainer can read the
    log without importing a plotting library.
    """
    with open(path, errors="ignore") as f:
        txt = f.read()
    first = first_shift(txt)
    whole = part_shift(txt)
    rows, had_check = [], False
    heads = list(LOG_HEAD.finditer(txt))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(txt)
        blk = txt[m.end():min(m.end() + 700, end)]
        r = {"step": int(m.group(1).replace(",", "")),
             "chars": float(m.group(2)),
             "corpus": (float(m.group(3).replace(",", ""))
                        if m.group(3) else None),
             "min": int(m.group(4)),
             "experts": int(m.group(5)) if m.group(5) else None}
        g = re.search(r"context ([\d,]+) characters", blk)
        r["context"] = int(g.group(1).replace(",", "")) if g else None
        g = re.search(r"reading ([\d,]+) char/s", blk)
        r["rate"] = int(g.group(1).replace(",", "")) if g else None
        g = re.search(r"train loss ([\d.]+)", blk)
        r["train"] = float(g.group(1)) if g else None
        g = re.search(r"lr ([\d.e+-]+)", blk)
        r["lr"] = float(g.group(1)) if g else None
        g = re.search(r"evidence t ([+-][\d.]+) over (\d+)", blk)
        r["evidence"] = float(g.group(1)) if g else None
        r["evidence_n"] = int(g.group(2)) if g else None
        g = re.search(r"\(effect ([+-][\d.]+)\)", blk)
        r["effect"] = float(g.group(1)) if g else None
        g = re.search(r"grad norm ([\d.]+)", blk)
        r["gnorm"] = float(g.group(1)) if g else None
        g = re.search(r"held-out loss ([\d.]+) \+/-([\d.]+)", blk)
        if g:
            r["val"], r["se"] = float(g.group(1)), float(g.group(2))
        g = re.search(r"^  ([a-z_]+ [\d.]+(?:   [a-z_]+ [\d.]+)*)\s*$", blk,
                      re.M)
        if g:
            parts = g.group(1).split()
            r["dom"] = {parts[i]: float(parts[i + 1])
                        for i in range(0, len(parts) - 1, 2)}
        if first is None or m.start() < first[0]:
            check = is_check(blk)
            if had_check and not check:
                continue
            had_check = had_check or check
            if first is not None and r.get("val") is not None:
                r["val"], dom = moved(r["val"], r.get("dom"), first[1])
                if dom is not None:
                    r["dom"] = dom
                r["shifted"] = True
        # and everything before held-out counted whole blocks only moves by
        # what that change measured, on top of any move above
        if (whole is not None and m.start() < whole[0]
                and r.get("val") is not None):
            r["val"], dom = moved(r["val"], r.get("dom"), whole[1])
            if dom is not None:
                r["dom"] = dom
            r["shifted"] = True
        rows.append(r)
    corpus = next((r["corpus"] for r in reversed(rows) if r["corpus"]), None)
    for r in rows:
        r["corpus"] = r["corpus"] or corpus
    return rows
