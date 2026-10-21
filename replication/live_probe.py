#!/usr/bin/env python3
"""
Does learning from a conversation damage what the model already knows? (M1)

`serve.py` learns from every exchange by default, but `minagi/live.py` says
plainly that the cost of that is UNKNOWN: an earlier, denser variant cost
around +0.65 nats on held-out against +0.02 for real documents, and a lower
learning rate was not the fix. This probe measures the current streaming
variant instead of guessing.

It answers the question the same way `forgetting_probe.py` does for corpus
reading, with one substitution: the "reading" is a fixed set of conversation
turns pushed through `LiveLearner`, exactly the path `serve.py::remember`
uses, and the held-out domains are scored before and after with
`FolderEvaluator`. The files are read from position zero every call, so a
difference is the model changing, not a different sample of text.

Two knobs matter most:

  --chunk   characters per optimiser step. This IS the step density, and
            live.py's warning is about density more than about the text.
  --mode    `both` feeds the user turn and the model's own reply (what serve
            does today); `user` feeds only what you typed. The difference is
            the cost of learning from its own output.

A run NEVER writes the directory given in --weights unless --no-copy is
passed: the tree is copied to a temporary directory first, because paging an
expert in marks it dirty and a learning step can write it back. --no-copy is
for a directory you have already copied yourself.

    # one configuration
    python3 replication/live_probe.py --weights /tmp/w_copy \
        --held-out data/val --chunk 512 --mode both --turns 64

    # the M1 sweep: densities x modes, each on its own fresh copy
    python3 replication/live_probe.py --weights /tmp/w_copy \
        --held-out data/val --sweep

The JSON it writes is what the roadmap records; `DEVPLAN.local.md` says where
the real measurement runs (the RTX 5080 box, with real weights).
"""
import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minagi.live import LiveLearner, exchange_text            # noqa: E402
from minagi.precision import set_compute_dtype                # noqa: E402
from minagi.stream import FolderEvaluator                     # noqa: E402
from minagi.tokenizer import ByteTokenizer                    # noqa: E402
import train as T                                             # noqa: E402


# A fixed, built-in conversation set so two runs are comparable and the probe
# needs no corpus or network. It is deliberately varied: the point is the
# density of learning, but a one-subject script would confound that with the
# massed-domain effect the forgetting probe already measures.
DEFAULT_CONVERSATIONS = [
    ("who are you", "I am mini-AGI. I read and write one character at a time, "
                    "and I keep learning from what I read."),
    ("how do you decide which experts to use",
     "The router scores every expert for the text, and the whole text votes. "
     "The experts it asks for most are the ones that compute each character."),
    ("what is 4917 + 388", "5205"),
    ("what is the capital of France", "Paris is the capital of France."),
    ("write a haiku about rain",
     "Soft rain on the roof. A grey sky leans on the town. The street shines "
     "like glass."),
    ("def merge_sorted(a, b)",
     "Merge two sorted lists by walking both with a pointer and appending the "
     "smaller head until one runs out."),
    ("explain a for loop",
     "A for loop repeats a block once for each item in a sequence, binding "
     "the current item to a name on every pass."),
    ("what is a neural network",
     "A neural network is a stack of parameterised functions trained by "
     "gradient descent to map inputs to outputs."),
    ("tell me a story about a lighthouse",
     "The lighthouse keeper counted the ships by their lights. One night a "
     "light appeared where no reef was charted, and it was moving."),
    ("what is the speed of light",
     "About 299,792 kilometres per second in a vacuum."),
    ("how do you avoid forgetting",
     "The trunk learns slower than the experts, and reading rotates between "
     "subjects so one domain cannot overwrite the rest."),
    ("what is a mixture of experts",
     "A pool of small feed-forward networks where a router sends each token "
     "to a few of them and combines their outputs."),
    ("write a function that reverses a string",
     "Return s[::-1] in Python, or walk from the end and append."),
    ("what is entropy",
     "A measure of the average surprise of a distribution, or of how many "
     "bits it takes to encode its outcomes."),
    ("why is the sky blue",
     "Air scatters shorter wavelengths more strongly, so blue light reaches "
     "your eye from every direction."),
    ("what is 17 * 23", "391"),
    ("describe a binary search",
     "Repeatedly halve a sorted range, comparing the middle element with the "
     "target, until it is found or the range is empty."),
    ("what is a git rebase",
     "It replays commits onto a new base, rewriting their hashes, so history "
     "looks as if the work started from that base."),
    ("tell me a joke",
     "A byte walks into a bar and asks for a bit. The bartender says, we only "
     "serve whole numbers here."),
    ("what is the difference between a list and a tuple",
     "A list is mutable; a tuple is not. Tuples can therefore be hashed and "
     "used as dictionary keys."),
    ("how does attention work",
     "Each position forms a query, key and value. Scores of query against key "
     "decide how much of each value is mixed into the output."),
    ("what is gradient descent",
     "Move each parameter a small step against the gradient of the loss, so "
     "the loss falls, and repeat."),
    ("summarise continual learning",
     "Training on a stream without a fixed end, keeping what was learned "
     "earlier while still absorbing new material."),
    ("what is 100 divided by 8", "12.5"),
]


def _turn_text(user, bot, mode):
    if mode == "user":
        return f"<user>\n{user}\n</user>\n"
    return exchange_text(user, bot)


def _load_conversations(path, turns):
    if path:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                rows.append((str(j.get("user", "")),
                             str(j.get("bot", j.get("assistant", "")))))
        if not rows:
            raise SystemExit(f"{path}: no conversations")
    else:
        rows = list(DEFAULT_CONVERSATIONS)
    if turns:
        rows = rows[:turns]
    return rows


def _score(ev, chunks):
    d = ev.run(chunks)
    d.pop("stderr", None)
    return {k: float(v) for k, v in d.items()}


def run_probe(weights, held_out, conversations, device="cpu", chunk=2048,
              mode="both", lr=3e-4, trunk_lr_mult=0.1, wd=0.1, clip=1.0,
              eval_chunks=8, eval_every=0, passes=1, resident=32,
              ram_cache=96, context=None, precision="bf16", tag="",
              read_domain="chat"):
    """Run one configuration and return the report dict. No file is written."""
    device = torch.device(device)
    set_compute_dtype(precision)
    model, cfg, pool, man = T.build_paged(
        weights, device, resident=resident, ram_capacity=ram_cache)
    # Growth would confound "did not forget" with "bought new capacity".
    model._grower = None
    ctx = int(context or man.get("context_now") or cfg.block)
    chunk = max(8, min(int(chunk), ctx))
    tok = ByteTokenizer()

    ev = FolderEvaluator(model, held_out, chunk, ctx, device)
    doms = sorted(ev.groups)
    before = _score(ev, eval_chunks)

    learner = LiveLearner(
        model, lr=lr, trunk_lr_mult=trunk_lr_mult, wd=wd, clip=clip,
        save_every=0,                      # the probe never calls save()
        weights_dir=weights, manifest=man, chunk=chunk, context=ctx)

    rows, losses, chars, turns = [], [], 0, 0
    t0 = time.time()
    for _ in range(max(1, passes)):
        for user, bot in conversations:
            text = _turn_text(user, bot, mode)
            if not text:
                continue
            chars += len(tok.encode(text).ids)
            recs = learner.feed(text, tok, note=f"turn {turns}")
            losses.extend(float(r["loss"]) for r in recs if "loss" in r)
            turns += 1
            if eval_every and turns % eval_every == 0:
                s = _score(ev, eval_chunks)
                rows.append({"turn": turns, "chars": chars,
                             "steps": learner.steps, **s})

    after = _score(ev, eval_chunks)
    delta = {k: after.get(k, float("nan")) - before.get(k, float("nan"))
             for k in doms}
    read = [read_domain] if read_domain in delta else []
    others = [k for k in doms if k not in read]
    report = {
        "tag": tag or f"{mode}-c{chunk}",
        "weights": weights,
        "step": man.get("step"),
        "experts": int(getattr(pool, "_n", man.get("n_experts", 0)) or 0),
        "device": str(device),
        "precision": precision,
        "context": ctx,
        "chunk": chunk,
        "mode": mode,
        "lr": lr,
        "trunk_lr_mult": trunk_lr_mult,
        "wd": wd,
        "clip": clip,
        "turns": turns,
        "passes": passes,
        "chars": chars,
        "steps": learner.steps,
        "train_loss": float(np.mean(losses)) if losses else None,
        "elapsed_s": round(time.time() - t0, 2),
        "domains": doms,
        "read_domain": read_domain,
        "before": before,
        "after": after,
        "delta": delta,
        "rows": rows,
        "others_mean_delta": (float(np.mean([delta[k] for k in others]))
                              if others else None),
        "read_mean_delta": (float(np.mean([delta[k] for k in read]))
                            if read else None),
    }
    return report


def _copy_weights(src):
    root = tempfile.mkdtemp(prefix="miniagi-liveprobe-")
    dst = os.path.join(root, "weights")
    shutil.copytree(src, dst)
    return root, dst


def _out_path(base, chunk, mode, sweep):
    if not sweep:
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}-{mode}-c{chunk}{ext or '.json'}"


def _print_row(r):
    def f(x):
        return "  n/a " if x is None or not np.isfinite(x) else f"{x:+.4f}"
    print(f"  {r['mode']:<5} chunk {r['chunk']:>5}  "
          f"steps {r['steps']:>5}  chars {r['chars']:>9,}  "
          f"train {r['train_loss'] if r['train_loss'] is not None else float('nan'):.4f}  "
          f"chat {f(r['read_mean_delta'])}  "
          f"others {f(r['others_mean_delta'])}  "
          f"[{r['elapsed_s']:.0f}s]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="weights directory; copied to a temp dir by default")
    ap.add_argument("--held-out", default="data/val",
                    help="folder of per-domain held-out text")
    ap.add_argument("--conversations", default="",
                    help="JSONL of {user, bot}; default is the built-in set")
    ap.add_argument("--turns", type=int, default=0,
                    help="cap the conversation set (0 = all)")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=2048,
                    help="characters per optimiser step (step density)")
    ap.add_argument("--mode", choices=("both", "user"), default="both")
    ap.add_argument("--read-domain", default="chat",
                    help="the held-out domain the conversations represent; "
                         "its delta is reported as the 'read' side")
    ap.add_argument("--sweep", action="store_true",
                    help="run chunks x modes, each on a fresh copy")
    ap.add_argument("--chunks", default="2048,512,256,96")
    ap.add_argument("--modes", default="both,user")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--trunk-lr-mult", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-chunks", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="score every N turns, not only before/after")
    ap.add_argument("--resident", type=int, default=32)
    ap.add_argument("--ram-cache", type=int, default=96)
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--device", default="", help="cuda/cpu (default: auto)")
    ap.add_argument("--no-copy", action="store_true",
                    help="run directly on --weights (must already be a copy)")
    ap.add_argument("--out", default="runs/live/probe.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    device = torch.device(a.device) if a.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    conv = _load_conversations(a.conversations, a.turns)
    print(f"  {len(conv)} turns, {a.passes} pass(es), "
          f"{a.eval_chunks} eval chunks/domain, device {device}")

    if a.sweep:
        chunks = [int(x) for x in a.chunks.split(",") if x.strip()]
        modes = [x.strip() for x in a.modes.split(",") if x.strip()]
        if a.no_copy:
            raise SystemExit("--sweep needs a fresh copy per config; "
                             "drop --no-copy")
        configs = [(c, m) for c in chunks for m in modes]
    else:
        configs = [(a.chunk, a.mode)]

    reports = []
    for chunk, mode in configs:
        tmp = None
        wdir = a.weights
        if not a.no_copy:
            tmp, wdir = _copy_weights(a.weights)
        try:
            print(f"\n  == {mode} / chunk {chunk} ==")
            r = run_probe(
                wdir, a.held_out, conv, device=device, chunk=chunk, mode=mode,
                lr=a.lr, trunk_lr_mult=a.trunk_lr_mult, wd=a.wd, clip=a.clip,
                eval_chunks=a.eval_chunks, eval_every=a.eval_every,
                passes=a.passes, resident=a.resident, ram_cache=a.ram_cache,
                context=a.context or None, precision=a.precision, tag=a.tag,
                read_domain=a.read_domain)
            r["weights_source"] = a.weights
            out = _out_path(a.out, chunk, mode, a.sweep)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w") as f:
                json.dump(r, f, indent=1)
            print(f"  -> {out}")
            _print_row(r)
            reports.append((out, r))
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
            # A sweep builds one model per config. Reference cycles and the
            # CUDA allocator can otherwise keep the previous one alive, which
            # OOMs a unified-memory board (an Orin) on the third config.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if len(reports) > 1:
        print("\n  summary (delta vs baseline; positive = worse):")
        print(f"  {'mode':<5} {'chunk':>6} {'steps':>6} {'chars':>10} "
              f"{'chat':>9} {'others':>9}")
        for _, r in reports:
            def f(x):
                return "      n/a" if x is None or not np.isfinite(x) else f"{x:+.4f}"
            print(f"  {r['mode']:<5} {r['chunk']:>6} {r['steps']:>6} "
                  f"{r['chars']:>10,} {f(r['read_mean_delta']):>9} "
                  f"{f(r['others_mean_delta']):>9}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
