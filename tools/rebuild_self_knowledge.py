#!/usr/bin/env python3
"""Regenerate the self-knowledge lane from a weights manifest."""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LANE = "self-knowledge"


def _manifest(weights):
    path = os.path.join(weights, "manifest.json")
    with open(path) as f:
        man = json.load(f)
    if not isinstance(man, dict):
        raise ValueError("manifest is not an object")
    return man


def _run(cmd, dry_run=False):
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def _expand(source, out, shard_chars):
    import numpy as np
    from corpora.expand import decode, write_shards
    from minagi.tokenizer import ByteTokenizer

    tok = ByteTokenizer()
    counts = {}
    for split in ("train", "val"):
        binf = os.path.join(source, f"{split}.bin")
        if not os.path.exists(binf):
            raise FileNotFoundError(binf)
        raw = decode(np.asarray(np.memmap(binf, dtype=np.uint16, mode="r")),
                     tok)
        lane = os.path.join(out, split, LANE)
        if os.path.isdir(lane):
            for name in os.listdir(lane):
                if name.endswith(".txt"):
                    os.remove(os.path.join(lane, name))
        counts[split] = write_shards(raw, lane, LANE, ".txt", shard_chars)
        print(f"  {split}/{LANE}: {counts[split][0]} files, "
              f"{counts[split][1]:,} chars", flush=True)
    return counts


def _read_command(weights, lane, held_out, chunk, passes, lr):
    return [sys.executable, "train.py", "read", lane,
            "--save", "--weights-dir", weights, "--held-out", held_out,
            "--chunk", str(chunk), "--passes", str(passes), "--lr", str(lr),
            "--grow-every", "0", "--grow-k", "0"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="regenerate the self-knowledge lane from a checkpoint "
                    "manifest, optionally re-reading it")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--source", default="data_self_knowledge_char")
    ap.add_argument("--out", default="data")
    ap.add_argument("--conversations", type=int, default=400000)
    ap.add_argument("--val", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-chars", type=int, default=200_000)
    ap.add_argument("--read", action="store_true",
                    help="after rebuilding, read the lane into --weights with "
                         "train.py read --save")
    ap.add_argument("--read-held-out", default="data/val")
    ap.add_argument("--read-chunk", type=int, default=2048)
    ap.add_argument("--read-passes", type=int, default=1)
    ap.add_argument("--read-lr", type=float, default=5e-5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    weights = os.path.abspath(os.path.expanduser(args.weights))
    source = os.path.abspath(os.path.expanduser(args.source))
    out = os.path.abspath(os.path.expanduser(args.out))
    held_out = os.path.abspath(os.path.expanduser(args.read_held_out))

    try:
        man = _manifest(weights)
    except (OSError, ValueError) as e:
        print(f"cannot read {weights}/manifest.json: {e}", file=sys.stderr)
        return 2
    if not man.get("n_experts"):
        print(f"{weights}/manifest.json has no n_experts", file=sys.stderr)
        return 2
    if args.shard_chars < 1:
        ap.error("--shard-chars must be at least 1")
    if args.conversations < 1:
        ap.error("--conversations must be at least 1")
    if args.val < 0:
        ap.error("--val must be 0 or more")
    if args.read_chunk < 1 or args.read_passes < 1:
        ap.error("--read-chunk and --read-passes must be at least 1")

    env = dict(os.environ)
    env["MINAGI_FACTS_WEIGHTS"] = weights
    cmd = [sys.executable, "-m", "corpora", "chat",
           "--out", source, "--self-only",
           "--conversations", str(args.conversations),
           "--val", str(args.val), "--seed", str(args.seed)]
    print(f"+ (MINAGI_FACTS_WEIGHTS={weights}) " + " ".join(cmd), flush=True)
    if args.dry_run:
        print(f"would expand {source} -> {out}/train/{LANE} and "
              f"{out}/val/{LANE}", flush=True)
        if args.read:
            _run(_read_command(weights, os.path.join(out, "train", LANE),
                               held_out, args.read_chunk, args.read_passes,
                               args.read_lr), dry_run=True)
        return 0
    rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    if rc:
        return rc
    try:
        _expand(source, out, args.shard_chars)
    except (OSError, ValueError) as e:
        print(f"expand failed: {e}", file=sys.stderr)
        return 1

    if args.read:
        cmd = _read_command(weights, os.path.join(out, "train", LANE),
                            held_out, args.read_chunk, args.read_passes,
                            args.read_lr)
        rc = _run(cmd)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
