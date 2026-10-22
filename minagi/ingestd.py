"""Scheduled, opt-in ingestion of text files into the live learner."""

import atexit
import json
import os
import signal
import sys
import threading
import time

from .ingest import collect


class RateLimiter:
    def __init__(self, per_minute, capacity=None, clock=time.monotonic):
        self.rate = float(per_minute)
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        self.capacity = float(capacity if capacity is not None else self.rate)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.tokens = self.capacity
        self.clock = clock
        self.last = clock()

    def _refill(self):
        now = self.clock()
        elapsed = max(0.0, now - self.last)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate / 60.0)
        self.last = now

    def available(self):
        self._refill()
        return int(self.tokens)

    def take(self, count):
        self._refill()
        got = min(max(0, int(count)), int(self.tokens))
        self.tokens -= got
        return got


class Ledger:
    VERSION = 1

    def __init__(self, path=None, files=None):
        self.path = os.path.abspath(path) if path else None
        self.files = dict(files or {})
        if self.path:
            self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                got = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(got, dict) or got.get("version") != self.VERSION:
            return
        files = got.get("files")
        if isinstance(files, dict):
            self.files = files

    def save(self):
        if not self.path:
            return
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"version": self.VERSION, "files": self.files}, f,
                          separators=(",", ":"))
            os.replace(tmp, self.path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def plan(self, path, size, mtime_ns):
        rec = self.files.get(path)
        if not isinstance(rec, list) or len(rec) < 3:
            return 0, "new"
        try:
            old_size, old_mtime, offset = (int(rec[0]), int(rec[1]),
                                           int(rec[2]))
        except (TypeError, ValueError):
            return 0, "invalid"
        size, mtime_ns = int(size), int(mtime_ns)
        if offset < 0:
            return 0, "moved"
        if size < old_size:
            return 0, "shrunk"
        if offset > size:
            return 0, "moved"
        if size == old_size and mtime_ns != old_mtime:
            return 0, "rewritten"
        if size > old_size:
            return offset, "appended"
        return offset, "unchanged"

    def commit(self, path, size, mtime_ns, offset):
        self.files[path] = [int(size), int(mtime_ns), int(offset)]


def read_slice(path, offset, limit):
    try:
        before = os.stat(path)
        if offset > before.st_size:
            offset = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if offset:
                f.seek(offset)
            text = f.read(max(0, int(limit)))
            end = f.tell()
        after = os.stat(path)
        return text, int(end), before, after
    except OSError:
        return None, offset, None, None


class Source:
    def __init__(self, path, rate=4096, clock=time.monotonic, name=None):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.name = name or os.path.basename(os.path.normpath(self.path))
        self.limiter = RateLimiter(rate, clock=clock)
        self.cursor = 0


class IngestDaemon:
    def __init__(self, sources, learner, tokenizer=None, watchdog=None,
                 ledger=None, max_files=8, max_file_chars=32768,
                 gate_every=32, cache="runs/ingest_index.json",
                 follow_symlinks=False, quiet=True):
        if tokenizer is None:
            from .tokenizer import ByteTokenizer
            tokenizer = ByteTokenizer()
        self.sources = list(sources)
        self.learner = learner
        self.tokenizer = tokenizer
        self.watchdog = watchdog
        self.ledger = ledger or Ledger()
        self.max_files = max(1, int(max_files))
        self.max_file_chars = max(1, int(max_file_chars))
        self.gate_every = max(1, int(gate_every))
        self.cache = cache
        self.follow_symlinks = bool(follow_symlinks)
        self.quiet = bool(quiet)
        self.turn = 0
        self._skip = {os.path.abspath(p) for p in
                      (self.ledger.path, self.cache) if p}

    def _files(self, source):
        files = collect([source.path], follow_symlinks=self.follow_symlinks,
                        cache=self.cache, quiet=True)
        return [p for p in files if os.path.abspath(p) not in self._skip]

    def _feed(self, source, path, text, end, after):
        stats = {"files": 0, "chars": 0, "steps": 0,
                 "source": source.name, "path": path}
        try:
            recs = self.learner.feed(text, self.tokenizer,
                                     note=f"ingest:{source.name}")
        except Exception as e:                                # noqa: BLE001
            stats["error"] = str(e)
            return stats
        used = len(text)
        source.limiter.take(used)
        stats.update(files=1, chars=used, steps=len(recs))
        self.ledger.commit(path, after.st_size, after.st_mtime_ns, end)
        return stats

    def tick(self):
        stats = {"files": 0, "chars": 0, "steps": 0, "sources": {},
                 "errors": []}
        if not self.sources:
            return stats
        quota = max(1, self.max_files // len(self.sources))
        remaining = self.max_files
        start_source = self.turn % len(self.sources)
        self.turn += 1
        ordered = self.sources[start_source:] + self.sources[:start_source]
        for source in ordered:
            if remaining <= 0:
                break
            files = self._files(source)
            used = {"files": 0, "chars": 0, "steps": 0}
            if not files:
                stats["sources"][source.name] = used
                continue
            budget = source.limiter.available()
            if budget <= 0:
                stats["sources"][source.name] = used
                continue
            start = source.cursor % len(files)
            for j in range(len(files)):
                if remaining <= 0 or budget <= 0 or used["files"] >= quota:
                    break
                path = files[(start + j) % len(files)]
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                offset, _why = self.ledger.plan(path, stat.st_size,
                                                stat.st_mtime_ns)
                if offset >= stat.st_size:
                    continue
                limit = min(budget, self.max_file_chars)
                text, end, _before, after = read_slice(path, offset, limit)
                if text is None:
                    stats["errors"].append(f"{path}: unreadable")
                    continue
                if not text:
                    self.ledger.commit(path, stat.st_size,
                                       stat.st_mtime_ns, stat.st_size)
                    continue
                got = self._feed(source, path, text, end, after or stat)
                if got.get("error"):
                    stats["errors"].append(f"{path}: {got['error']}")
                    break
                used["files"] += got["files"]
                used["chars"] += got["chars"]
                used["steps"] += got["steps"]
                budget -= got["chars"]
                remaining -= got["files"]
                source.cursor = ((start + j + (0 if end < stat.st_size else 1))
                                 % len(files))
                if end < stat.st_size:
                    break
            stats["files"] += used["files"]
            stats["chars"] += used["chars"]
            stats["steps"] += used["steps"]
            stats["sources"][source.name] = used
        if self.watchdog is not None:
            try:
                ev = self.watchdog.maybe_check(self.gate_every)
                if ev is not None:
                    stats["gate"] = ev
                    self.learner = self.watchdog.learner
            except Exception as e:                            # noqa: BLE001
                stats["errors"].append(f"gate: {e}")
        if self.learner.due_to_save():
            try:
                self.learner.save()
                stats["saved"] = True
            except Exception as e:                            # noqa: BLE001
                stats["errors"].append(f"save: {e}")
        self.ledger.save()
        return stats

    def run(self, poll=30.0, stop=None):
        stop = stop or threading.Event()
        while not stop.is_set():
            stats = self.tick()
            if not self.quiet:
                self._say(stats)
            if stop.wait(max(0.1, float(poll))):
                break

    def _say(self, stats):
        if stats["errors"]:
            for err in stats["errors"]:
                print(f"[ingest] {err}", file=sys.stderr)
        if stats["files"]:
            print(f"[ingest] {stats['files']} files, {stats['chars']} chars, "
                  f"{stats['steps']} steps", file=sys.stderr, flush=True)
        gate = stats.get("gate")
        if gate:
            print(f"[ingest] gate {gate.get('action')} "
                  f"{gate.get('val')}", file=sys.stderr, flush=True)


def _manifest(path):
    try:
        with open(os.path.join(path, "manifest.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _build(sources, args):
    from minagi.config import get as _get, load as _load
    from minagi.live import LiveLearner
    from minagi.precision import set_compute_dtype
    from minagi.recur import load_any

    cfg = _load()
    set_compute_dtype(args.precision
                      or _get(cfg, "training.precision", "bf16"))
    import torch
    dev = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_any(args.weights, dev, read_only=False)
    model.eval()
    manifest = _manifest(args.weights)
    learner = LiveLearner(
        model, lr=args.learn_lr,
        trunk_lr_mult=_get(cfg, "training.trunk_lr_mult", 0.1),
        wd=_get(cfg, "training.weight_decay", 0.1),
        clip=_get(cfg, "training.clip", 1.0),
        chunk=args.learn_chunk or _get(cfg, "training.chunk", 2048),
        context=args.learn_context or None,
        save_every=args.save_every,
        weights_dir=args.weights,
        manifest=manifest)

    def rebuild(path):
        fresh, _ = load_any(path, dev, read_only=False)
        fresh.eval()
        old = learner
        live = LiveLearner(
            fresh, lr=args.learn_lr,
            trunk_lr_mult=_get(cfg, "training.trunk_lr_mult", 0.1),
            wd=_get(cfg, "training.weight_decay", 0.1),
            clip=_get(cfg, "training.clip", 1.0),
            chunk=old.chunk, context=old.context,
            save_every=args.save_every,
            weights_dir=args.weights,
            manifest=_manifest(path))
        return fresh, live

    return model, learner, rebuild, dev


def _install_signals(stop):
    def handler(_signum, _frame):
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="read opted-in files into a live weights directory on a "
                    "schedule")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--source", action="append", required=True,
                    metavar="PATH", help="explicit file or directory; repeat "
                                         "for more sources")
    ap.add_argument("--source-rate", type=int, default=4096,
                    help="characters per minute for each source")
    ap.add_argument("--poll", type=float, default=30.0,
                    help="seconds between scans")
    ap.add_argument("--once", action="store_true",
                    help="scan once, read what fits the budget, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="list readable files and characters; do not load or "
                         "write weights")
    ap.add_argument("--max-files", type=int, default=8,
                    help="files read per tick across all sources")
    ap.add_argument("--max-file-chars", type=int, default=32768,
                    help="characters read from one file in one tick")
    ap.add_argument("--state", default="runs/ingest_state.json",
                    help="file offsets; empty disables persistence")
    ap.add_argument("--cache", default="runs/ingest_index.json",
                    help="text/binary verdict cache used by the file walker")
    ap.add_argument("--follow-symlinks", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", default=None, choices=["bf16", "fp32"])
    ap.add_argument("--learn-lr", type=float, default=3e-4)
    ap.add_argument("--learn-chunk", type=int, default=None)
    ap.add_argument("--learn-context", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=8)
    ap.add_argument("--held-out", default="data/val")
    ap.add_argument("--gate", dest="gate", action="store_true", default=None)
    ap.add_argument("--no-gate", dest="gate", action="store_false")
    ap.add_argument("--gate-every", type=int, default=32)
    ap.add_argument("--gate-chunks", type=int, default=4)
    ap.add_argument("--gate-chunk", type=int, default=2048)
    ap.add_argument("--gate-context", type=int, default=0)
    ap.add_argument("--gate-factor", type=float, default=1.10)
    ap.add_argument("--best-weights", default=None)
    ap.add_argument("--allow-ungated-dense", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.source_rate < 1:
        ap.error("--source-rate must be at least 1")
    if args.poll < 1:
        ap.error("--poll must be at least 1 second")
    if args.max_files < 1:
        ap.error("--max-files must be at least 1")
    if args.max_file_chars < 1:
        ap.error("--max-file-chars must be at least 1")
    if args.learn_context < 0:
        ap.error("--learn-context must be 0 or more")
    if args.save_every < 0:
        ap.error("--save-every must be 0 or more")

    sources = []
    seen = set()
    for raw in args.source:
        path = os.path.abspath(os.path.expanduser(raw))
        if not os.path.exists(path):
            ap.error(f"--source does not exist: {raw}")
        if path in seen:
            continue
        seen.add(path)
        sources.append(Source(path, rate=args.source_rate))

    if args.dry_run:
        for source in sources:
            files = collect([source.path], follow_symlinks=args.follow_symlinks,
                            cache=None, quiet=True)
            chars = sum(os.path.getsize(p) for p in files)
            print(f"[dry-run] {source.name}: {len(files)} files, "
                  f"{chars} bytes")
        return 0

    from minagi.config import get as _get, load as _load
    from minagi.watchdog import HeldOutWatchdog

    cfg = _load()
    live_chunk = args.learn_chunk or _get(cfg, "training.chunk", 2048)
    gate_on = bool(args.gate if args.gate is not None
                   else os.path.isdir(args.held_out))
    if args.gate and not os.path.isdir(args.held_out):
        ap.error(f"--gate needs a held-out directory; {args.held_out} is "
                 f"missing")
    if live_chunk < 2048 and not gate_on and not args.allow_ungated_dense:
        ap.error(f"ingestion at {live_chunk}-character steps is denser than "
                 f"the 2048 measured safe in M1; use --gate or "
                 f"--allow-ungated-dense")

    if not os.path.exists(os.path.join(args.weights, "core.npz")):
        ap.error(f"no model in {args.weights}; train one first")
    from .writer_lock import WriterBusy, writer_lock
    try:
        lock = writer_lock(args.weights, "ingestd")
    except WriterBusy as e:
        ap.error(str(e))
    atexit.register(lock.release)
    model, learner, rebuild, dev = _build(sources, args)
    print(f"[ingest] {len(sources)} source(s), {args.source_rate} chars/min "
          f"each, poll {args.poll:g}s, step {learner.chunk}", file=sys.stderr)
    print(f"[ingest] {model.n_params() / 1e6:.1f}M parameters on {dev}",
          file=sys.stderr)

    watch = None
    if gate_on:
        try:
            watch = HeldOutWatchdog(
                model, learner, args.held_out, best_dir=args.best_weights,
                chunk=args.gate_chunk, context=args.gate_context or None,
                chunks=args.gate_chunks, factor=args.gate_factor,
                device=dev, rebuild=rebuild)
            ev = watch.start()
            print(f"[ingest] gate baseline {ev.get('val'):.4f}",
                  file=sys.stderr)
        except Exception as e:                                # noqa: BLE001
            print(f"[ingest] gate disabled: {e}", file=sys.stderr)
            if live_chunk < 2048 and not args.allow_ungated_dense:
                raise SystemExit("the gate could not start and the step is "
                                 "denser than the measured-safe 2048")
            watch = None
    else:
        print("[ingest] gate off", file=sys.stderr)

    ledger = Ledger(args.state or None)
    daemon = IngestDaemon(sources, learner, watchdog=watch, ledger=ledger,
                          max_files=args.max_files,
                          max_file_chars=args.max_file_chars,
                          gate_every=args.gate_every, cache=args.cache,
                          follow_symlinks=args.follow_symlinks,
                          quiet=args.quiet)
    stop = threading.Event()
    _install_signals(stop)
    try:
        if args.once:
            stats = daemon.tick()
            if not args.quiet:
                daemon._say(stats)
        else:
            daemon.run(args.poll, stop)
    except KeyboardInterrupt:
        pass
    finally:
        if daemon.learner.unsaved:
            try:
                daemon.learner.save()
                print("[ingest] saved on exit", file=sys.stderr)
            except Exception as e:                            # noqa: BLE001
                print(f"[ingest] save failed on exit: {e}", file=sys.stderr)
        daemon.ledger.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
