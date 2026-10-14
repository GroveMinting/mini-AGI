"""
Reading a pile of files.

The model's alphabet is the 256 byte values, so there is nothing to prepare: a
file is already written in the only vocabulary it has. Point it at a directory
and it reads what is there - source, prose, notes, logs, transcripts - in
whatever order they come, and is different afterwards.

That is the whole interface. No tokenizer to fit, no corpus to build, no
preprocessing step that has to be re-run when the data changes.

Binary files are skipped rather than read as noise: a file is taken as text if
a sample of it decodes as UTF-8 and is mostly printable. Everything else is
left alone, because a model that reads a JPEG learns what a JPEG header looks
like and nothing else useful - and what is left alone is said, with the reason.
"""

import codecs
import json
import os

import numpy as np

SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv",
             "venv", ".mypy_cache", ".pytest_cache", "build", "dist"}
SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip",
            ".gz", ".bz2", ".xz", ".tar", ".7z", ".mp3", ".mp4", ".wav",
            ".mov", ".avi", ".so", ".dylib", ".dll", ".exe", ".bin", ".npy",
            ".npz", ".pt", ".pth", ".onnx", ".pyc", ".woff", ".woff2", ".ttf"}


def why_not_text(path, probe=4096):
    """
    None if this is worth reading, otherwise why not. Decides on a sample, not
    on the extension alone.
    """
    if os.path.splitext(path)[1].lower() in SKIP_EXT:
        return "of a binary type"
    try:
        with open(path, "rb") as f:
            head = f.read(probe + 1)
    except OSError:
        return "unreadable"
    cut = len(head) > probe              # the file goes on past the sample
    head = head[:probe]
    if not head:
        return "empty"
    if b"\x00" in head:
        return "with a NUL byte"
    # The sample ends wherever `probe` bytes falls, which can be inside a
    # character - é, ●, a CJK ideograph take two to four bytes. Decoded
    # strictly, that cut alone made a text file look binary, and it was
    # dropped without a word: 1,663 files of this repository's own corpus,
    # and 9% of one reader's (GitHub issue #30). A character cut by the
    # sample is not evidence. One broken anywhere else still is, and so is a
    # file that really does end in the middle of one.
    try:
        s = codecs.getincrementaldecoder("utf-8")().decode(head, final=not cut)
    except UnicodeDecodeError:
        return "not UTF-8"
    if not s:
        return "not UTF-8"
    printable = sum(1 for c in s if c.isprintable() or c in "\n\r\t")
    if printable / len(s) <= 0.9:
        return "mostly unprintable"
    return None


def looks_like_text(path, probe=4096):
    """Is this worth reading? Decides on a sample, not on the extension alone."""
    return why_not_text(path, probe) is None


def _walk(paths, follow_symlinks=False):
    """Every candidate path, without opening anything."""
    out = []
    for p in paths:
        if os.path.isfile(p):
            out.append(p)
            continue
        for root, dirs, files in os.walk(p, followlinks=follow_symlinks):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                       and not d.startswith(".")]
            for f in files:
                out.append(os.path.join(root, f))
    return sorted(out)


# Raised whenever the rule above or the cache's layout changes. A cache written
# under an older rule is then surveyed afresh rather than trusted - otherwise
# the rule that dropped files cut mid-character would go on dropping them from
# every corpus it had already seen.
_CACHE_VERSION = 2

_HELD = {}               # cache path -> ((size, mtime) of the file read, verdicts)


def _load_verdicts(cache):
    """The cached verdicts, {path: [size, mtime_ns, why_not]}; {} if none."""
    try:
        st = os.stat(cache)
    except OSError:
        return {}
    stamp = (st.st_size, st.st_mtime_ns)
    hit = _HELD.get(cache)
    if hit and hit[0] == stamp:          # the held-out set reads it 8 times
        return hit[1]
    try:
        with open(cache) as f:
            got = json.load(f)
        held = got.get("files") if got.get("version") == _CACHE_VERSION else None
    except (OSError, ValueError, AttributeError):
        held = None
    held = held if isinstance(held, dict) else {}
    _HELD[cache] = (stamp, held)
    return held


def _save_verdicts(cache, held):
    tmp = f"{cache}.{os.getpid()}.tmp"       # two writers never share a file
    try:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump({"version": _CACHE_VERSION, "files": held}, f,
                      separators=(",", ":"))
        os.replace(tmp, cache)
        st = os.stat(cache)
        _HELD[cache] = ((st.st_size, st.st_mtime_ns), held)
    except OSError:                          # a full disk, say: no litter
        try:
            os.remove(tmp)
        except OSError:
            pass


def collect(paths, follow_symlinks=False, cache="runs/corpus_index.json",
            quiet=False):
    """
    Every readable text file under the given files and directories.

    Deciding whether a file is text means reading a piece of it, and at half a
    million files that is minutes of every startup; looking at each file's size
    and modification time is a second or two. So every verdict is remembered
    against those two: an unchanged file is never opened again, and one
    rewritten under the same name is checked afresh. The cache used to be keyed
    on the list of names alone, so a file fixed after being skipped stayed
    skipped.

    What is skipped is said, with the reason, unless `quiet`. The rule is a
    guess from a sample, and a guess that drops files without a word drops
    them for good.
    """
    names = _walk(paths, follow_symlinks)
    held = _load_verdicts(cache) if cache else {}
    out, skipped, seen, changed = [], {}, set(), 0
    for p in names:
        if p in seen:                    # paths that overlap list a file twice
            continue
        seen.add(p)
        try:
            st = os.stat(p)
        except OSError:
            skipped.setdefault("unreadable", []).append(p)
            continue
        stamp = [st.st_size, st.st_mtime_ns]
        v = held.get(p)
        if isinstance(v, list) and len(v) == 3 and v[:2] == stamp:
            why = v[2]
        else:
            why = why_not_text(p) or ""
            held[p] = stamp + [why]
            changed += 1
        if why:
            skipped.setdefault(why, []).append(p)
        else:
            out.append(p)
    if cache:
        # forget what was under these paths and is gone, so the cache does
        # not grow with every corpus ever rebuilt
        roots = tuple(os.path.join(p, "") for p in paths if os.path.isdir(p))
        gone = [k for k in held if k not in seen and
                (k in paths or (roots and k.startswith(roots)))]
        for k in gone:
            del held[k]
        if changed or gone:
            _save_verdicts(cache, held)
    if skipped and not quiet:
        n = sum(len(v) for v in skipped.values())
        why = "; ".join(f"{len(v):,} {r} (e.g. {v[0]})" for r, v in
                        sorted(skipped.items(), key=lambda kv: -len(kv[1])))
        print(f"  skipped {n:,} of {len(seen):,} files under "
              f"{', '.join(paths)} as not text: {why}", flush=True)
    return out


_TOK = None


def as_stream(path):
    """
    One file as the character stream the model reads.

    Encoded through the tokenizer rather than read as raw bytes. For ordinary
    text the two are the same - every byte is its own token. They differ only
    where a file contains one of the structural markers, `<user>` or `<g>` or
    `<think>`: those are single ids above 255, and reading the file as bytes
    would turn each into six or seven separate characters and lose the
    boundary it marks.
    """
    global _TOK
    if _TOK is None:
        from minagi.tokenizer import ByteTokenizer
        _TOK = ByteTokenizer()
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", "surrogateescape")
    return np.asarray(_TOK.encode(text).ids, dtype=np.uint16)


def summarise(files):
    total = sum(os.path.getsize(f) for f in files)
    return {"files": len(files), "characters": total}
