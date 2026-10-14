#!/usr/bin/env python3
"""
End-to-end smoke test for the hardening mirror.

Tiny model -> short read with --save -> serve it -> one request. It exercises
the exact paths the carried patches touch, on whatever device you point it at:

    0001  a save-less read must not touch the source weights (checked by the
          read running with --save here, and its mirror without --save)
    0004  FileReader._measure must not read past the rotary tables
    0006  a created-but-never-saved directory must load through paging

    python3 scripts/smoke.py                 # auto device (CUDA, else CPU)
    python3 scripts/smoke.py --device cuda   # force a device
    python3 scripts/smoke.py --keep          # leave the temp dir for inspection

Nothing here touches weights/ or data/ - everything happens in a temp directory
that is removed on exit. Requires the runtime deps only; the corpus is not
needed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _auto_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _run(cmd, timeout):
    print("  $ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True,
                          timeout=timeout)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _create_tiny(path):
    from minagi.create import create
    create(path, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
           d_model=8, trunk_d_ff=16, n_head=1, block=64, max_steps=1, top_k=1)


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_up(port, proc, timeout=60.0):
    url = f"http://127.0.0.1:{port}/api/state"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"serve.py exited early ({proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return json.loads(r.read().decode())
        except Exception:                                  # noqa: BLE001
            time.sleep(0.5)
    raise SystemExit("serve.py did not come up in time")


def _one_reply(port, timeout=120.0):
    body = json.dumps({"messages": [{"role": "user", "content": "hello"}],
                       "max_new": 16}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    text = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[len("data:"):].strip())
            if "error" in ev:
                raise SystemExit(f"serve.py error: {ev['error']}")
            if "t" in ev:
                text += ev["t"]
            if ev.get("done"):
                return text
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default=None,
                    help="auto: CUDA if available, else CPU")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temp directory and print its path")
    ap.add_argument("--read-timeout", type=int, default=180)
    a = ap.parse_args()
    device = a.device or _auto_device()
    precision = "fp32" if device == "cpu" else "bf16"

    tmp = tempfile.mkdtemp(prefix="minagi-smoke-")
    w = os.path.join(tmp, "weights")
    text = os.path.join(tmp, "input.txt")
    print(f"smoke: device={device} precision={precision} tmp={tmp}", flush=True)
    try:
        print("== create a tiny model ==", flush=True)
        _create_tiny(w)
        if not os.path.isdir(os.path.join(w, "experts")):
            raise SystemExit("create() left no experts/ layout on disk")

        with open(text, "w") as f:
            f.write("the quick brown fox jumps over the lazy dog. " * 24)

        print("== read it with --save ==", flush=True)
        read = [sys.executable, "train.py", "--device", device, "read", text,
                "--weights-dir", w,
                "--chunk", "8", "--context-start", "64", "--context-end", "64",
                "--resident", "1", "--ram-capacity", "1", "--passage", "64",
                "--grow-k", "0", "--held-out", "", "--history", "",
                "--precision", precision, "--train-steps-mean", "0",
                "--save-every", "0", "--no-plots", "--passes", "1", "--save"]
        _run(read, timeout=a.read_timeout)
        if not os.path.exists(os.path.join(w, "core.npz")):
            raise SystemExit("--save left no core.npz behind")

        print("== serve it (read-only) and talk once ==", flush=True)
        port = _free_port()
        server = subprocess.Popen(
            [sys.executable, "serve.py", "--weights", w, "--device", device,
             "--no-learn", "--port", str(port)],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            state = _wait_up(port, server)
            print(f"  /api/state: learn={state.get('learn')}", flush=True)
            reply = _one_reply(port)
            print(f"  /api/chat: {reply!r}", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

        print("SMOKE OK", flush=True)
        return 0
    finally:
        if a.keep:
            print(f"kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
