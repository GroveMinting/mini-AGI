#!/usr/bin/env python3
"""Fault-injection soak for a weights directory.

    python3 scripts/soak.py                # tiny model, all scenarios
    python3 scripts/soak.py --keep         # keep the temp directory
    python3 scripts/soak.py --scenario crash

The point is not a long conversation. It is the moments a save is interrupted:
after some bundles, before the manifest, and with the disk full. Each scenario
starts from a loadable checkpoint, injects the fault, then proves the old
checkpoint still loads and that a retry resumes. The prune scenario checks that
a stale expert file is not deleted before the new manifest is durable.
"""

import argparse
import errno
import json
import os
import shutil
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from minagi import store                                      # noqa: E402
from minagi.create import create                              # noqa: E402
from minagi.recur import RecurCoder, RecurConfig, load_any    # noqa: E402


def _cfg(d):
    return RecurConfig(**{k: v for k, v in d.items()
                          if k in RecurConfig.__dataclass_fields__})


def _nonpaged(path):
    d = create(path, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
               d_model=8, trunk_d_ff=16, n_head=1, block=16, max_steps=1,
               top_k=1)
    return RecurCoder(_cfg(d)), d


def _step(path):
    with open(os.path.join(path, "manifest.json")) as f:
        return json.load(f)["step"]


def _loads(path):
    model, man = load_any(path, "cpu", read_only=False)
    return model is not None and int(man.get("step", -99))


def _child(path, step, fault):
    """A save in a child that dies at the chosen point, not a caught error."""
    try:
        model, _ = load_any(path, "cpu", read_only=False)
        if fault == "manifest":
            real = store.os.replace

            def replace(src, dst):
                if os.path.basename(str(dst)) == "manifest.json":
                    os._exit(42)
                return real(src, dst)

            store.os.replace = replace
        elif fault == "bundle":
            real = store._savez
            calls = {"n": 0}

            def savez(p, arrays):
                calls["n"] += 1
                if calls["n"] == 2:
                    os._exit(42)
                return real(p, arrays)

            store._savez = savez
        else:
            raise ValueError(fault)
        store.save(model, path, step=step)
        os._exit(0)
    except BaseException:                                    # noqa: BLE001
        traceback.print_exc()
        os._exit(1)


def _fork_save(path, step, fault):
    if not hasattr(os, "fork"):
        raise RuntimeError("this soak needs os.fork (Linux/macOS)")
    pid = os.fork()
    if pid == 0:
        _child(path, step, fault)
    _pid, status = os.waitpid(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    if code != 42:
        raise AssertionError(f"child exited {code}, expected 42")


def _baseline(tmp):
    path = os.path.join(tmp, "weights")
    model, cfg = _nonpaged(path)
    store.save(model, path, step=1, cfg=cfg)
    assert _loads(path) == 1
    return path, model, cfg


def scenario_baseline(tmp):
    path, _model, _cfg_ = _baseline(tmp)
    assert _loads(path) == 1
    return "baseline save -> load"


def scenario_crash(tmp):
    path, _model, _cfg_ = _baseline(tmp)
    _fork_save(path, 2, "manifest")
    assert _step(path) == 1, "a crash before the manifest advanced the step"
    assert _loads(path) == 1
    _fork_save(path, 2, "bundle")
    assert _step(path) == 1, "a crash mid-bundle advanced the manifest"
    assert _loads(path) == 1
    return "crash mid-bundle and before manifest -> old checkpoint loads"


def scenario_disk_full(tmp):
    path, model, cfg = _baseline(tmp)
    real = store._savez
    calls = {"n": 0}

    def full(p, arrays):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(p, arrays)

    store._savez = full
    try:
        try:
            store.save(model, path, step=2, cfg=cfg)
        except OSError as e:
            assert e.errno == errno.ENOSPC
        else:
            raise AssertionError("disk-full save unexpectedly succeeded")
    finally:
        store._savez = real
    assert _step(path) == 1
    assert _loads(path) == 1
    store.save(model, path, step=2, cfg=cfg)
    assert _step(path) == 2
    return "ENOSPC keeps the old checkpoint; retry publishes step 2"


def scenario_prune(tmp):
    path, model, cfg = _baseline(tmp)
    extra = os.path.join(path, "experts", "e99999.npz")
    shutil.copy(os.path.join(path, "experts", "e00000.npz"), extra)
    real = store.os.replace

    def break_manifest(src, dst):
        if os.path.basename(str(dst)) == "manifest.json":
            raise OSError(errno.EIO, "interrupted manifest")
        return real(src, dst)

    store.os.replace = break_manifest
    try:
        try:
            store.save(model, path, step=2, cfg=cfg)
        except OSError:
            pass
        else:
            raise AssertionError("manifest failure unexpectedly succeeded")
    finally:
        store.os.replace = real
    assert _step(path) == 1
    assert os.path.exists(extra), "stale expert deleted before manifest"
    store.save(model, path, step=3, cfg=cfg)
    assert not os.path.exists(extra), "stale expert not removed after manifest"
    return "manifest first, then prune deletion"


SCENARIOS = {
    "baseline": scenario_baseline,
    "crash": scenario_crash,
    "disk-full": scenario_disk_full,
    "prune": scenario_prune,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=["all"] + list(SCENARIOS),
                    default="all")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--dir", default="")
    a = ap.parse_args(argv)

    tmp = a.dir or tempfile.mkdtemp(prefix="minagi-soak-")
    os.makedirs(tmp, exist_ok=True)
    choose = list(SCENARIOS) if a.scenario == "all" else [a.scenario]
    failed = 0
    try:
        for name in choose:
            try:
                note = SCENARIOS[name](os.path.join(tmp, name))
                print(f"  OK  {name:10s} {note}")
            except Exception as e:                            # noqa: BLE001
                failed += 1
                print(f"  FAIL {name:10s} {e}")
                traceback.print_exc()
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"  kept {tmp}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
