#!/usr/bin/env python3
"""
A chat window for talking to the model.

    python3 serve.py                 then open http://127.0.0.1:8080

Nothing but the conversation. Generation streams a character at a time, so
what you watch is the model writing rather than a spinner followed by a wall
of text - at this size that difference is most of the information.

Decoding is greedy and deterministic: the same conversation gives the same
reply twice. The loops greedy decoding falls into are held off by the
adaptation trace in minagi/decode.py, at the strength config.yaml sets.

IT LEARNS WHILE YOU TALK TO IT, and this CHANGES THE WEIGHTS ON DISK. The
exchange joins one continuous stream and the model takes an optimiser step
every `training.chunk` characters of it - the same path minagi/stream.py
takes through a corpus, which is why there is no second mechanism to reason
about. Read minagi/live.py for what it costs; it is not free, and on short
chat it is a far denser diet than documents. `--no-learn` serves read-only.

Binds to localhost.

TOOL USE IS OFF BY DEFAULT. `--tools --tool-policy <file>` lets the model emit
`<tool>{"name": ..., "args": ...}</tool>` and gives it the result, under an
explicit allowlist and filesystem roots. The sandbox has no shell, network, or
arbitrary Python. Tool results are not fed back into the live-learning update.
"""

import argparse
import atexit
import gc
import json
import os
import signal
import sys
import threading
import time

# see train.py: the allocator reads this once at CUDA init, so it has to be
# set before torch loads - under the name this PyTorch reads (2.9 renamed it)
def _alloc_var():
    try:
        from importlib.metadata import version
        major, minor = (int(v) for v in version("torch").split(".")[:2])
        return "PYTORCH_ALLOC_CONF" if (major, minor) >= (2, 9) else "PYTORCH_CUDA_ALLOC_CONF"
    except Exception:                                      # noqa: BLE001
        return "PYTORCH_CUDA_ALLOC_CONF"


if not any(v in os.environ for v in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")):
    os.environ[_alloc_var()] = "expandable_segments:True"

import torch
from flask import Flask, Response, jsonify, request

from minagi.recur import load_any
from minagi.tokenizer import ByteTokenizer
from minagi.writer_lock import WriterBusy, writer_lock

app = Flask(__name__)

# One model, one conversation at a time. The lock is what makes that true:
# two overlapping requests would interleave their forward passes through the
# same KV caches and produce two corrupted replies instead of one good one.
LOCK = threading.Lock()
STATE = {"model": None, "tok": None, "weights": None, "learner": None,
         "watchdog": None, "gate": None, "experience": None,
         "replay": None, "session": None, "turn": 0}

# Tool use is off unless --tools is given with a policy file. Keeping the
# imports lazy means a normal server never imports the sandbox at all.
TOOLS = {"on": False, "policy": None, "ledger": None, "registry": None,
         "max_calls": 0, "parse": None, "result_text": None, "run": None}

U0, U1, B0, B1 = "<user>", "</user>", "<bot>", "</bot>"

# A passage of the corpus the conversation opens with. Empty when priming is
# off or no corpus is on disk.
PRIME = ""
WRITER_LOCK = None


def load_prime(chars, root="data/train/self-knowledge"):
    """
    Corpus text to open the conversation with, or "" if there is none.

    TWO THINGS IT BUYS, and they are separate. The reply has context to stand
    on: "hi" alone is two characters, and every character the model writes
    attends over what came before it and routes among the experts that whole
    context has voted for. And the learning stream starts part-way to its
    first step rather than twenty exchanges short of one.

    Taken from the self-knowledge lane because it is already in the register
    the chat is in - <user>/<bot> turns about what the model is - so this is
    text the model has read, not a preamble invented here and trained on.
    The `self-` files are the short question-and-answer ones; `auto-src-` are
    whole source files quoted back, and priming with one of those biases the
    router toward code and teaches the model to open by dumping a module.

    CUT WHERE THE MARKERS BALANCE, not at the last closing tag. This corpus
    documents its own format, so a file contains a code block reading
    "<bot>   </bot>" as prose - and those are real marker tokens, not
    spellings. Trimming to the last </bot> landed inside that block and
    returned a passage with an unclosed turn in it, which hands the model an
    open <bot> to answer inside before the conversation has started.

    Deterministic: the first file that yields a usable passage, every time. A
    passage picked at random would make the same conversation give different
    answers across restarts, and repeating a reply is worth more than variety.
    """
    if chars <= 0 or not os.path.isdir(root):
        return ""
    names = sorted(f for f in os.listdir(root) if f.endswith(".txt"))
    # conversational turns first, source dumps only if there are none
    names = ([n for n in names if n.startswith("self-")]
             + [n for n in names if not n.startswith("self-")])
    for name in names[:20]:
        with open(os.path.join(root, name), errors="replace") as f:
            text = f.read(chars * 3)
        best = ""
        at = text.find(f"{B1}\n")
        while 0 <= at < chars:
            end = at + len(B1) + 1
            head = text[:end]
            if (head.count(U0) == head.count(U1) == head.count(B0)
                    == head.count(B1)):
                best = head
            at = text.find(f"{B1}\n", end)
        if len(best) > chars // 3:
            return best
    return ""


def _manifest(path):
    """As loaded, so a save does not overwrite it with a blank one."""
    try:
        with open(os.path.join(path, "manifest.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load(weights, device=None, learn=True, lr=None, save_every=8,
         chunk=None, context=0):
    global WRITER_LOCK
    if learn and WRITER_LOCK is None:
        WRITER_LOCK = writer_lock(weights, "serve.py live learning")
        atexit.register(WRITER_LOCK.release)
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    # read_only marks nothing dirty, so an expert paged in is never written
    # back. That is right for serving and wrong for learning - what the model
    # learned would live in VRAM until the slot was reused and then be gone.
    ro = not learn
    try:
        model, _ = load_any(weights, dev, read_only=ro)
    except torch.cuda.OutOfMemoryError:
        # training may be holding most of the card; serve slowly rather than
        # not at all
        torch.cuda.empty_cache()
        dev = torch.device("cpu")
        model, _ = load_any(weights, dev, read_only=ro)
        print("[warn] CUDA is full, serving on CPU", file=sys.stderr)
    model.eval()
    STATE.update(model=model, tok=ByteTokenizer(), weights=weights)
    print(f"[loaded] {weights} on {dev}", file=sys.stderr)

    # A paged model loads with an EMPTY card - every slot -1. A reply is
    # admitted by its own text's vote, so this does not change a reply; what
    # it changes is that the page can show experts on the card before the
    # first question instead of nothing. It is one forward over the priming
    # text, which admits the experts that text asks for, as any forward does.
    if PRIME:
        from minagi.precision import amp
        ids = STATE["tok"].encode(PRIME).ids[-model.cfg.block:]
        with torch.no_grad(), amp(dev):
            model(torch.tensor([ids], device=dev))

    if learn:
        from minagi.live import LiveLearner
        # Set up as training is, at the rate training would step at next. The
        # chunk defaults to the corpus granularity; --learn-chunk trades that
        # against the denser diet live.py warns about.
        STATE["learner"] = LiveLearner.for_run(
            model, weights_dir=weights, manifest=_manifest(weights), lr=lr,
            chunk=chunk, context=context or None, save_every=save_every)
        # The prime goes into the stream too, so the first step arrives
        # sooner. Kept SHORTER than a chunk on purpose: long enough to
        # advance the counter, short enough that it never takes a step by
        # itself, so restarting cannot train on the same passage repeatedly.
        if PRIME:
            STATE["learner"].feed(PRIME[-(STATE["learner"].chunk - 1):],
                                  STATE["tok"], note="prime")
        ln = STATE["learner"]
        ch = ln.chunk
        left = max(0, ch - ln.pending)
        base = ln.opt.param_groups[-1]["base_lr"]
        print(f"[learning] on - one step per {ch} characters, "
              f"{left} to the first (about {max(1, round(left / 93))} "
              f"exchanges), at the run's rate: {base:g} x{ln.factor():.4f} "
              f"x live {ln.scale:g} = {base * ln.effective_scale():.3g}; "
              f"saved to {weights}/ every {save_every} steps",
              file=sys.stderr)
    return model


def resident_experts():
    """
    Which experts are on the card right now, by stable id.

    `slots` is slot -> expert index, and the index is not an identity:
    pruning renumbers the pool, so the same index means different experts at
    different times. `uid` is what survives that, which is why it is what the
    page shows.
    """
    pool = getattr(STATE.get("model"), "pool", None)
    if pool is None or not hasattr(pool, "slots"):
        return None
    gate = pool.gate.detach().abs()
    top = float(gate.max().clamp_min(1e-9))
    out = []
    for j, e in enumerate(pool.slots):
        e = int(e)
        if e < 0:
            continue
        out.append({"slot": j, "uid": int(pool.uid[e]),
                    "gate": round(float(gate[e]) / top, 3)})
    return {"resident": out, "n_experts": int(pool.n_experts()),
            "n_slots": len(pool.slots),
            "admitted": len(getattr(pool, "_admitted", ()) or ())}


def learn_state():
    """How far the stream is from its next optimiser step."""
    lr_ = STATE.get("learner")
    if lr_ is None:
        return None
    out = {"pending": int(lr_.pending), "chunk": int(lr_.chunk),
           "steps": int(lr_.steps), "lr_scale": round(float(lr_.scale), 4),
           "model_step": int(lr_.model_step)}
    g = STATE.get("gate")
    if g:
        out["gate"] = {k: g.get(k) for k in
                       ("action", "val", "best", "limit", "lr_scale")}
    exp = STATE.get("experience")
    if exp is not None:
        out["experience"] = {"recent": len(exp), "appended": exp.appended,
                             "loaded": exp.loaded, "log": exp.path}
    rp = STATE.get("replay")
    if rp is not None:
        out["replay"] = {"every": rp.every, "turns": rp.turn,
                         "corpus": bool(rp.corpus),
                         "user_only": rp.user_only}
    kp = STATE.get("keeper")
    if kp is not None:
        out["growth"] = {"every": kp.every, "checks": kp.checks,
                         "events": len(kp.events)}
    return out


def build_prompt(messages, budget, prime=""):
    """
    The conversation as the model reads it, newest turns kept.

    The trailing open `<bot>` tag is what asks for a reply, so the trim is
    from the FRONT. Cutting the end to fit would remove the question.
    """
    parts = []
    for m in messages:
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if m.get("role") == "user":
            parts.append(f"{U0}\n{text}\n{U1}\n")
        else:
            parts.append(f"{B0}\n{text}\n{B1}\n")
    prompt = prime + "".join(parts) + f"{B0}\n"
    return prompt[-budget:] if len(prompt) > budget else prompt


@torch.no_grad()
def stream(prompt, max_new):
    from minagi.config import get as _g, load as _lc
    from minagi.decode import pick_next
    from minagi.precision import amp

    c = _lc()
    strength = _g(c, "decoding.adapt_strength", 2.5)
    decay = _g(c, "decoding.adapt_decay", 0.88)

    model, tok = STATE["model"], STATE["tok"]
    device = next(model.parameters()).device
    block = model.cfg.block
    ids = tok.encode(prompt).ids[-block:]
    out = torch.tensor([ids or [10]], device=device)

    # THE TEXT CHOOSES. The prompt is read in chunks, and each chunk adds its
    # characters' requests to the text's vote and is admitted the most-voted
    # experts; then every character of the reply is a forward of its own that
    # adds its requests and routes among what the whole text has voted for,
    # loading whatever is not already on the card. Nothing here chooses
    # experts - the forward does.
    pool = getattr(model, "pool", None)

    # Prefill in chunks, the way training reads a corpus. Feeding a long
    # prompt in one pass materialises activations for every position across
    # every block application at once, which is what puts a long context out
    # of reach; the cache carries the reach instead. Every forward computes
    # in the process-wide precision, as reading and model.generate do.
    CHUNK = 512
    caches = None

    def read(text):
        """`text` into fresh caches from position 0, a chunk at a time; the
        last chunk's logits. Position 0 begins a new text, so the vote that
        admits experts is cast by what is read here and nothing before it."""
        nonlocal caches
        caches = model.empty_caches()
        lg = None
        for i in range(0, text.shape[1], CHUNK):
            with amp(device):
                lg = model(text[:, i:i + CHUNK], caches=caches, pos_offset=i)[0]
        return lg

    logits = read(out)
    pos = out.shape[1]                       # where the next character sits

    cur = out[:, -1:]
    produced = []
    n_tool = 0
    # The prompt's forwards have admitted its experts. Report them, so the
    # page starts the reply showing what is actually answering it.
    yield {"swap": {"at": 0, "moved": None, "pool": resident_experts()}}
    for i in range(max_new):
        loads = getattr(pool, "loads", 0)
        if logits is None:
            if pos + cur.shape[1] > block:
                # THE WINDOW IS FULL. A cached key keeps the rotary position
                # it was read at, so dropping the oldest key and placing the
                # next character at the end again - block-1, every time -
                # gives the newest characters one position between them, and
                # attention loses their order. Re-read the newest half of the
                # text from position 0 instead: what model.generate does, and
                # what every window of training is, a text from position 0.
                ctx = out[:, -(block // 2):]
                logits = read(ctx)
                pos = ctx.shape[1]
            else:
                with amp(device):
                    logits = model(cur, caches=caches, pos_offset=pos)[0]
                pos += cur.shape[1]
            moved = getattr(pool, "loads", 0) - loads
            if moved:
                # this character asked for experts that were not on the card;
                # `moved` counts the ones copied onto it
                yield {"swap": {"at": i, "moved": int(moved),
                                "pool": resident_experts()}}
        nxt = pick_next(logits[:, -1, :].float(), out, temperature=0.0,
                        adapt_strength=strength, adapt_decay=decay)
        out = torch.cat([out, nxt], dim=1)
        cur = nxt
        logits = None                       # spent; the next pass recomputes
        produced.append(int(nxt[0, 0]))
        text = tok.decode(produced)
        if TOOLS["on"] and n_tool < TOOLS["max_calls"]:
            calls = TOOLS["parse"](text)
            if calls:
                call = calls[0]
                n_tool += 1
                yield {"tool": {"name": call.name, "args": call.args}}
                result = TOOLS["run"](TOOLS["registry"], call.name,
                                      call.args, TOOLS["policy"],
                                      TOOLS["ledger"])
                body = TOOLS["result_text"](result.as_dict())
                ids = tok.encode(body).ids
                extra = torch.tensor([ids], device=device)
                out = torch.cat([out, extra], dim=1)
                if out.shape[1] > block:
                    out = out[:, -block:]
                logits = read(out)
                pos = out.shape[1]
                cur = out[:, -1:]
                produced = []
                yield {"t": body, "tool_result": True}
                continue
        if text.endswith(B1):
            break
        yield {"t": tok.decode([produced[-1]])}


def remember(user_text, bot_text):
    """
    Read the finished exchange, the way the corpus reader reads a file.

    ONLY THIS TURN goes in. The prompt above carries the whole conversation,
    so re-feeding it every time would put the earliest turns through the
    stream once per exchange - re-reading rather than reading on, which is
    the dense regime minagi/live.py warns about.

    The caller already holds LOCK.
    """
    learner = STATE.get("learner")
    if learner is None:
        return None
    from minagi.live import exchange_text
    text = exchange_text(user_text, bot_text.replace(B1, ""))
    if not text:
        return None
    try:
        recs = learner.feed(text, STATE["tok"], note="chat")
        # Reported even when no step was taken. A step needs `chunk`
        # characters and an exchange is about ninety, so the first twenty or
        # so take none at all - and silence there is indistinguishable from
        # learning being broken, which is exactly the wrong thing to leave
        # ambiguous.
        out = {"steps": len(recs),
               "pending": learner.pending,
               "chunk": learner.chunk,
               "total": learner.steps}
        if recs:
            out["loss"] = round(float(recs[-1].get("loss", 0.0)), 4)
            if learner.due_to_save():
                learner.save()
                out["saved"] = True
        watch = STATE.get("watchdog")
        if watch is not None:
            try:
                ev = watch.maybe_check(STATE.get("gate_every", 32))
                if ev is not None:
                    STATE["gate"] = ev
                    out["gate"] = {k: ev.get(k) for k in
                                   ("action", "val", "best", "limit",
                                    "lr_scale", "domains")}
            except Exception as e:                            # noqa: BLE001
                # A gate failure must not cost the reply or the learning step.
                print(f"[gate] failed: {e}", file=sys.stderr)
                out["gate_error"] = str(e)

        # A gate revert may have rebuilt the model and learner. The keeper
        # still points at the pre-revert pool; rebind it before the next
        # prune/grow decision so that decision acts on the restored model.
        learner = STATE.get("learner") or learner
        keeper = STATE.get("keeper")
        if keeper is not None and keeper.learner is not learner:
            keeper.rebind(STATE["model"], learner,
                          base_step=int(learner.start_step))
        out["pending"] = int(learner.pending)
        out["chunk"] = int(learner.chunk)
        out["total"] = int(learner.steps)

        exp = STATE.get("experience")
        if exp is not None:
            STATE["turn"] = int(STATE.get("turn", 0)) + 1
            exp.append(user_text, bot_text.replace(B1, ""),
                       session=STATE.get("session"), turn=STATE["turn"],
                       steps=learner.steps,
                       model_step=int(learner.model_step))

        keeper = STATE.get("keeper")
        if (keeper is not None and out.get("gate")
                and out["gate"].get("action") != "reverted"
                and out["gate"].get("val") is not None):
            try:
                change = keeper.after_check(out["gate"]["val"])
                if change:
                    print(f"  [pool] {change}", file=sys.stderr)
                    # New and deleted experts only exist in memory until this
                    # save; the rebuild is what gives the learner an optimizer
                    # that names the pool it now has.
                    step = keeper.model_step()
                    learner.fold_steps(step)
                    learner.save()
                    rebuild = STATE.get("rebuild")
                    if rebuild is None:
                        raise RuntimeError("no rebuild callback after growth")
                    fresh, live = rebuild(learner.weights_dir)
                    watch = STATE.get("watchdog")
                    if watch is not None:
                        watch.rebind(fresh, live)
                    keeper.rebind(fresh, live,
                                  base_step=int(live.start_step))
                    learner = live
                    out["pool"] = change
            except Exception as e:                            # noqa: BLE001
                print(f"[pool] change failed: {e}", file=sys.stderr)
                out["pool_error"] = str(e)
                # Do not try the same model change again from a half-applied
                # state; the next session can start clean.
                keeper.every = 0

        replay = STATE.get("replay")
        if replay is not None:
            gate = STATE.get("gate") or {}
            item = replay.maybe_next(allow=gate.get("action") != "reverted")
            if item is not None:
                replay_text, note = item
                try:
                    rr = learner.feed(replay_text, STATE["tok"], note=note)
                    out["replay"] = {"note": note, "steps": len(rr),
                                     "chars": len(replay_text)}
                    if rr and learner.due_to_save():
                        learner.save()
                        out["saved"] = True
                except Exception as e:                        # noqa: BLE001
                    print(f"[replay] failed: {e}", file=sys.stderr)
                    out["replay"] = {"error": str(e)}

        learner = STATE.get("learner") or learner
        out["pending"] = int(learner.pending)
        out["chunk"] = int(learner.chunk)
        out["total"] = int(learner.steps)
        out["model_step"] = int(learner.model_step)
        return out
    except Exception as e:                                    # noqa: BLE001
        # A failed step must not cost the reply that was already written.
        print(f"[learn] failed: {e}", file=sys.stderr)
        return {"error": str(e)}


@app.route("/api/chat", methods=["POST"])
def api_chat():
    body = request.get_json(force=True)
    model = STATE["model"]
    msgs = body.get("messages", [])
    prompt = build_prompt(msgs, int(model.cfg.block * 0.9), PRIME)
    max_new = int(body.get("max_new", 400))
    # the half of the exchange the model did not predict, which is where the
    # signal in a conversation is
    last_user = next((m.get("content") for m in reversed(msgs)
                      if m.get("role") == "user"), "")

    def events():
        t0, n = time.time(), 0
        reply = []
        learned = None
        try:
            with LOCK:
                for ev in stream(prompt, max_new):
                    if "t" in ev and not ev.get("tool_result"):
                        n += len(ev["t"])
                        reply.append(ev["t"])
                    yield f"data: {json.dumps(ev)}\n\n"
                learned = remember(last_user, "".join(reply))
        except torch.cuda.OutOfMemoryError:
            # Almost always a training run holding the card. Say so and stay
            # up: dying here closes the socket, and all the browser can tell
            # you then is that the fetch failed.
            torch.cuda.empty_cache()
            yield "data: " + json.dumps({"error":
                "the GPU is out of memory - something else is probably using "
                "it. Restart with --device cpu, or stop the other process."
                }) + "\n\n"
        except Exception as e:                            # noqa: BLE001
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        dt = max(time.time() - t0, 1e-6)
        done = {"done": True, "n": n, "cps": round(n / dt, 1),
                "learn": learn_state(), "pool": resident_experts()}
        if learned:
            done["learned"] = learned
        yield "data: " + json.dumps(done) + "\n\n"

    return Response(events(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/state")
def api_state():
    """
    What the two mechanisms are doing between replies.

    The page asks once on load and after every exchange, so the counters are
    right even when nothing is being generated - otherwise "how far from the
    next update" would only exist while the model was talking.
    """
    return jsonify({"learn": learn_state(),
                    "pool": resident_experts()})


@app.route("/api/prime")
def api_prime():
    """
    What the conversation is opened with.

    The prime is prepended to every prompt and never appears in the replies,
    so without this the model is answering from context the person talking to
    it cannot see. That is the kind of thing that should be inspectable rather
    than trusted.
    """
    return jsonify({"chars": len(PRIME),
                    "turns": PRIME.count(U0),
                    "text": PRIME})


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


PAGE = r'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mini-AGI</title>
<style>
  :root{
    --bg:#ffffff; --fg:#1f2328; --dim:#6b7280; --line:#e5e7eb;
    --user:#f3f4f6; --accent:#2563eb; --err:#b91c1c;
    --ok:#15803d; --glow:rgba(37,99,235,.28);
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#0d1117; --fg:#e6edf3; --dim:#8b949e; --line:#21262d;
           --user:#1c2128; --accent:#58a6ff; --err:#f85149;
           --ok:#3fb950; --glow:rgba(88,166,255,.30); }
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--fg);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       display:flex;flex-direction:column}
  header{border-bottom:1px solid var(--line);padding:10px 16px;
         display:flex;align-items:center;gap:12px;flex:0 0 auto}
  header b{font-weight:600}
  header .sp{flex:1}
  button{font:inherit;color:var(--fg);background:none;cursor:pointer;
         border:1px solid var(--line);border-radius:6px;padding:5px 11px}
  button:hover{border-color:var(--accent);color:var(--accent)}
  #log{flex:1 1 auto;overflow-y:auto;padding:24px 16px}
  .wrap{max-width:720px;margin:0 auto}
  .msg{margin:0 0 20px}
  .who{font-size:12px;color:var(--dim);margin-bottom:5px}
  .body{white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere}
  .user .body{background:var(--user);padding:10px 13px;border-radius:10px}
  .err{color:var(--err)}
  .caret::after{content:"\258B";color:var(--accent);
                animation:blink 1s step-end infinite}
  @keyframes blink{50%{opacity:0}}
  details.prime{border:1px solid var(--line);border-radius:8px;
                margin:0 0 22px;background:var(--user)}
  details.prime summary{cursor:pointer;padding:8px 12px;font-size:12px;
                        color:var(--dim);list-style:none}
  details.prime summary::-webkit-details-marker{display:none}
  details.prime summary::before{content:"\25B8  ";display:inline-block;
                                transition:transform .15s}
  details.prime[open] summary::before{transform:rotate(90deg)}
  details.prime pre{margin:0;padding:0 12px 12px;white-space:pre-wrap;
                    word-wrap:break-word;overflow-wrap:anywhere;
                    font-size:12px;color:var(--dim);max-height:320px;
                    overflow-y:auto}
  /* the two mechanisms, always on screen */
  #mech{border-bottom:1px solid var(--line);padding:9px 16px;flex:0 0 auto;
        background:var(--bg)}
  .mrow{max-width:720px;margin:0 auto;display:flex;gap:22px;
        align-items:center;flex-wrap:wrap}
  .meter{flex:1 1 210px;min-width:190px}
  .mlab{font-size:11px;color:var(--dim);display:flex;
        justify-content:space-between;margin-bottom:4px}
  .mlab b{font-weight:600;color:var(--fg)}
  .bar{height:5px;border-radius:3px;background:var(--line);overflow:hidden}
  .fill{height:100%;width:0;border-radius:3px;background:var(--accent);
        transition:width .18s linear}
  .fill.mem{background:var(--ok)}
  /* the moment it fires */
  @keyframes flare{0%{transform:scale(1);box-shadow:0 0 0 0 var(--glow)}
                   40%{transform:scale(1.03);box-shadow:0 0 0 7px transparent}
                   100%{transform:scale(1);box-shadow:0 0 0 0 transparent}}
  .fired{animation:flare .75s ease-out}
  .note{font-size:11px;color:var(--ok);opacity:0;transition:opacity .2s}
  .note.on{opacity:1}
  .note.swap{color:var(--accent)}
  /* which experts are on the card */
  #chips{max-width:720px;margin:9px auto 0;display:flex;flex-wrap:wrap;gap:3px}
  .chip{font-size:10px;font-variant-numeric:tabular-nums;padding:1px 5px;
        border-radius:3px;border:1px solid var(--line);color:var(--dim);
        transition:background .3s,color .3s,border-color .3s}
  .chip.new{background:var(--accent);border-color:var(--accent);color:#fff}
  .empty{color:var(--dim);text-align:center;margin-top:18vh;line-height:1.9}
  .empty code{background:var(--user);padding:2px 6px;border-radius:4px;
              font-size:13px}
  footer{border-top:1px solid var(--line);padding:12px 16px;flex:0 0 auto}
  form{max-width:720px;margin:0 auto;display:flex;gap:8px;align-items:flex-end}
  textarea{flex:1;resize:none;font:inherit;color:var(--fg);
           background:var(--bg);border:1px solid var(--line);
           border-radius:9px;padding:10px 12px;max-height:180px;
           overflow-y:auto}
  textarea:focus{outline:none;border-color:var(--accent)}
  .stat{max-width:720px;margin:7px auto 0;font-size:12px;color:var(--dim);
        height:16px;display:flex;justify-content:space-between}
  .hint{color:var(--dim)}
</style></head>
<body>
<header>
  <b>mini-AGI</b>
  <span class="sp"></span>
  <button id="reset">New chat</button>
</header>

<div id="mech">
  <div class="mrow">
    <div class="meter" id="m-mem">
      <div class="mlab"><span>long-term memory</span><b id="mem-n">-</b></div>
      <div class="bar"><div class="fill mem" id="mem-f"></div></div>
    </div>
    <div class="meter" id="m-exp">
      <div class="mlab"><span>experts the last character admitted</span><b id="exp-n">-</b></div>
      <div class="bar"><div class="fill" id="exp-f"></div></div>
    </div>
    <span class="note" id="note"></span>
  </div>
  <div id="chips"></div>
</div>

<div id="log"><div class="wrap"><div id="primebox"></div><div id="thread"></div></div></div>

<footer>
  <form id="f">
    <textarea id="q" rows="1" placeholder="Say something"></textarea>
    <button id="send" type="submit">Send</button>
  </form>
  <div class="stat"><span id="stat"></span><span class="hint">Enter to send, Shift+Enter for a newline</span></div>
</footer>

<script>
const thread = document.getElementById('thread');
const log    = document.getElementById('log');
const q      = document.getElementById('q');
const stat   = document.getElementById('stat');
const send   = document.getElementById('send');
let messages = [];
let busy = false;

// ------------------------------------------------ the two mechanisms
// Both are counters the server keeps; the page mirrors them so that what
// the model does to itself is visible while it is happening, rather than
// inferred afterwards from the fact that nothing changed.
let MECH = {pending:0, chunk:0, steps:0, admitted:0, slots:0, uids:[], loaded:0};
const $ = id => document.getElementById(id);

function note(text, kind){
  const n = $('note');
  n.textContent = text;
  n.className = 'note on' + (kind ? ' ' + kind : '');
  clearTimeout(note._t);
  note._t = setTimeout(() => { n.className = 'note'; }, 2600);
}
function flare(el){ el.classList.remove('fired');
                    void el.offsetWidth;            // restart the animation
                    el.classList.add('fired'); }

function drawMem(){
  const {pending, chunk} = MECH;
  if (!chunk){ $('mem-n').textContent = 'off'; $('mem-f').style.width = '0';
               return; }
  const left = Math.max(0, chunk - pending);
  // The step is taken after the reply finishes, not mid-sentence, so once
  // the stream is full "0 characters to go" is true but says the wrong
  // thing - it reads as though nothing is going to happen.
  $('mem-n').textContent = left
      ? left + ' characters to the next update'
      : (busy ? 'updates when this reply finishes' : 'ready to update');
  $('mem-f').style.width = (100 * Math.min(pending, chunk) / chunk) + '%';
}
function drawExp(){
  // A character may use at most as many experts as the card has slots: the
  // ones its text - the prompt and the reply so far - has voted for most.
  const {admitted, slots} = MECH;
  if (!slots){ $('exp-n').textContent = '-'; return; }
  $('exp-n').textContent = admitted + ' of ' + slots;
  $('exp-f').style.width = (100 * Math.min(admitted, slots) / slots) + '%';
}
// Chips are keyed by uid, not by slot. A slot number says where an expert
// sits; the uid says WHICH expert it is, and that is what survives pruning.
function drawChips(pool, changed){
  if (!pool) return;
  const box = $('chips');
  box.innerHTML = '';
  for (const e of pool.resident){
    const c = document.createElement('span');
    c.className = 'chip' + (changed && changed.has(e.uid) ? ' new' : '');
    c.textContent = e.uid;
    c.title = 'expert ' + e.uid + ' - gate ' + e.gate + ' of the largest';
    box.appendChild(c);
  }
  if (changed && changed.size) setTimeout(() => {
    box.querySelectorAll('.chip.new').forEach(c => c.classList.remove('new'));
  }, 1400);
}

function applyState(d){
  if (d.learn){ MECH.pending = d.learn.pending; MECH.chunk = d.learn.chunk;
                MECH.steps = d.learn.steps; }
  if (d.pool){
    MECH.admitted = d.pool.admitted || 0; MECH.slots = d.pool.n_slots || 0;
    drawExp();
    const seen = new Set(MECH.uids);
    const now = d.pool.resident.map(e => e.uid);
    const changed = MECH.uids.length ? new Set(now.filter(u => !seen.has(u)))
                                     : null;
    MECH.uids = now;
    drawChips(d.pool, changed);
    $('m-exp').title = d.pool.resident.length + ' of ' + d.pool.n_experts
                     + ' experts on the card';
  }
  drawMem(); drawExp();
}
fetch('/api/state').then(r => r.json()).then(applyState).catch(() => {});

// What the model is answering from, besides what you typed. It is prepended
// to every prompt server-side and never appears in a reply, so it is shown
// here rather than left to be taken on trust.
fetch('/api/prime').then(r => r.json()).then(p => {
  if (!p.chars) return;
  const d = document.createElement('details');
  d.className = 'prime';
  d.innerHTML = '<summary></summary><pre></pre>';
  d.querySelector('summary').textContent =
    'primed with ' + p.chars + ' characters of the corpus'
    + (p.turns ? ' (' + p.turns + ' turns)' : '')
    + ' - in front of every prompt';
  d.querySelector('pre').textContent = p.text;
  document.getElementById('primebox').appendChild(d);
}).catch(() => {});

function empty(){
  thread.innerHTML = '<div class="empty">Ask it something.<br>'
    + 'It is small, it is mid-training, and it will show.</div>';
}
function add(role, text){
  if (thread.querySelector('.empty')) thread.innerHTML = '';
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = '<div class="who"></div><div class="body"></div>';
  d.querySelector('.who').textContent = role === 'user' ? 'You' : 'mini-AGI';
  d.querySelector('.body').textContent = text || '';
  thread.appendChild(d);
  return d.querySelector('.body');
}
// Only follow the text down while the reader is already at the bottom, so
// scrolling up to re-read something is not undone by the next character.
function atBottom(){ return log.scrollHeight - log.scrollTop - log.clientHeight < 60; }
function follow(was){ if (was) log.scrollTop = log.scrollHeight; }

// Collapse to zero before measuring. 'auto' does not shrink a textarea
// inside a flex row, so scrollHeight reports the height it already has and
// the box only ever grows - an empty input opened at the 180px maximum.
function grow(){ q.style.height = '0px';
                 q.style.height = Math.min(q.scrollHeight, 180) + 'px'; }
q.addEventListener('input', grow);
q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault();
                                         document.getElementById('f').requestSubmit(); }
});

document.getElementById('reset').onclick = () => {
  if (busy) return;
  messages = []; empty(); stat.textContent = ''; q.focus();
};

document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text || busy) return;
  busy = true; send.disabled = true;
  q.value = ''; grow();

  add('user', text);
  messages.push({role:'user', content:text});
  const body = add('bot', '');
  body.classList.add('caret');
  log.scrollTop = log.scrollHeight;
  stat.textContent = 'thinking...';

  let reply = '';
  MECH.loaded = 0;
  try {
    const r = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages, max_new: 400})
    });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for(;;){
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      // SSE frames are separated by a blank line; a frame can arrive split
      // across two reads, so anything after the last separator stays in buf.
      const frames = buf.split('\n\n');
      buf = frames.pop();
      for (const f of frames){
        if (!f.startsWith('data: ')) continue;
        const d = JSON.parse(f.slice(6));
        if (d.t){ const was = atBottom(); reply += d.t;
                  body.textContent = reply; follow(was);
                  // every character the model writes goes into the stream
                  // it learns from, so both counters advance as it speaks
                  MECH.pending++;
                  if (MECH.chunk && MECH.pending >= MECH.chunk)
                    MECH.pending = MECH.chunk;
                  drawMem(); }
        else if (d.swap){
          applyState(d);
          if (d.swap.moved){
            // a load comes when the text's vote shifts, now and then; the
            // note keeps the reply's running total rather than flashing one
            // message per load
            MECH.loaded += d.swap.moved;
            flare($('m-exp'));
            note(MECH.loaded + (MECH.loaded === 1 ? ' expert loaded'
                                                  : ' experts loaded')
                 + ' for this reply', 'swap');
          }
        }
        else if (d.error){ body.classList.add('err');
                           body.textContent = reply + '\n\n[' + d.error + ']'; }
        else if (d.done){
          // The server's counters are authoritative: the page has been
          // guessing per character, and a step resets pending to something
          // it cannot work out on its own.
          const stepped = d.learned && d.learned.steps;
          applyState(d);
          if (stepped){
            flare($('m-mem'));
            note(stepped + (stepped === 1 ? ' weight update' : ' weight updates')
                 + (d.learned.saved ? ', saved to disk' : ''));
          }
          let line = d.n + ' characters, ' + d.cps + '/s';
          const L = d.learned;
          if (!L) line += '  ·  not learning';
          else if (L.error) line += '  ·  learning failed: ' + L.error;
          else if (L.steps) line += '  ·  ' + L.steps
                    + (L.steps === 1 ? ' step' : ' steps')
                    + ', loss ' + L.loss + (L.saved ? ', saved' : '')
                    + '  ·  ' + L.total + ' total';
          // No step yet is the normal state for the first several turns, so
          // say how far off one is rather than nothing at all.
          else line += '  ·  learning: ' + L.pending + '/' + L.chunk
                    + ' characters to the next step';
          stat.textContent = line;
        }
      }
    }
  } catch (err) {
    // A dropped connection is the server going away, not the model failing.
    // The browser reports both as "Failed to fetch", so say which is likely.
    const gone = String(err).includes('Failed to fetch');
    body.classList.add('err');
    body.textContent = reply + '\n\n[' + (gone
      ? 'lost the server - it stopped while answering. Check the terminal '
        + 'it is running in; if the GPU was busy, restart with --device cpu.'
      : err) + ']';
  }
  body.classList.remove('caret');
  // Keep the turn even when it came back empty, so the conversation the
  // model sees next time matches the one on screen.
  messages.push({role:'bot', content:reply});
  busy = false; send.disabled = false; q.focus();
  drawExp();
};

empty(); q.focus();
// After layout, not during it. At the end of the script the flex row has no
// resolved width yet, so the placeholder wraps and scrollHeight reports a
// tall box - the input opened at its maximum every time.
requestAnimationFrame(grow);
</script>
</body></html>'''


def _persist():
    """Flush the live stream's unsaved steps. Safe to call more than once."""
    learner = STATE.get("learner")
    if learner is None or not learner.weights_dir or not learner.unsaved:
        return
    got = LOCK.acquire(timeout=10)
    if not got:
        print("[save] skipped on exit: the model is busy", file=sys.stderr)
        return
    try:
        if learner.unsaved:
            learner.save()
            print("[saved] live steps written on exit", file=sys.stderr)
    except Exception as e:                                    # noqa: BLE001
        print(f"[save] failed on exit: {e}", file=sys.stderr)
    finally:
        LOCK.release()


def _install_exit_hooks():
    atexit.register(_persist)

    def stop(signum, frame):
        _persist()
        raise SystemExit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, stop)
        except (ValueError, OSError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights",
                    help="the weights directory to talk to")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default=None,
                    help="cuda / cpu; default auto, falling back to CPU if "
                         "the card is full")
    ap.add_argument("--no-learn", dest="learn", action="store_false",
                    help="serve without learning. The weights are then opened "
                         "read-only and nothing is written back")
    ap.add_argument("--learn-lr", type=float, default=None,
                    help="the base rate the live stream learns at, which "
                         "training's plasticity factor and the live gate scale "
                         "then multiply. Defaults to the rate the training run "
                         "used, then training.lr in config.yaml")
    ap.add_argument("--prime-chars", type=int, default=1024,
                    help="characters of corpus to open the conversation "
                         "with, so the router has something to choose "
                         "experts from and the learning stream starts part "
                         "of the way to its first step. 0 disables it")
    ap.add_argument("--learn-chunk", type=int, default=None,
                    help="characters of conversation per optimiser step. "
                         "Defaults to training.chunk, which is tuned for "
                         "reading a corpus; try 256 for a short session, at "
                         "the cost of a denser diet than documents give")
    ap.add_argument("--save-every", type=int, default=8,
                    help="optimiser steps between writing the weights out")
    ap.add_argument("--learn-context", type=int, default=0,
                    help="characters the live stream keeps in its cache; "
                         "0 = the model's block. Lower it on a small card")
    ap.add_argument("--precision", default=None,
                    choices=["bf16", "fp32"],
                    help="what the forward computes in; defaults to whatever "
                         "config.yaml trains with. No fp16: the recurrent "
                         "state reaches ~10,000 and RMSNorm squares it, past "
                         "fp16's 65,504")
    ap.add_argument("--held-out", default="data/val",
                    help="held-out corpus the gate scores. The live directory "
                         "is snapshotted to <weights>.best and restored from "
                         "there if this gets worse")
    ap.add_argument("--gate", dest="gate", action="store_true", default=None,
                    help="watch held-out while learning (default: on when "
                         "--held-out exists)")
    ap.add_argument("--no-gate", dest="gate", action="store_false",
                    help="learn without a held-out watchdog")
    ap.add_argument("--gate-every", type=int, default=32,
                    help="live steps between held-out checks")
    ap.add_argument("--gate-chunks", type=int, default=4,
                    help="chunks per held-out domain at each check")
    ap.add_argument("--gate-chunk", type=int, default=2048,
                    help="characters per held-out chunk (a fixed instrument, "
                         "independent of --learn-chunk)")
    ap.add_argument("--gate-context", type=int, default=0,
                    help="context for held-out chunks; 0 = the model's block")
    ap.add_argument("--gate-factor", type=float, default=1.10,
                    help="revert when held-out is this many times the best "
                         "(a three-standard-error floor is applied too)")
    ap.add_argument("--best-weights", default=None,
                    help="where to keep the best snapshot "
                         "(default: <weights>.best)")
    ap.add_argument("--retention-log",
                    default=os.path.join("runs", "retention.jsonl"),
                    help="append held-out domain readings here for the "
                         "longitudinal dashboard; empty disables")
    ap.add_argument("--allow-ungated-dense", action="store_true",
                    help="permit --learn-chunk below 2048 without a gate. M1 "
                         "measured material held-out damage at 512")
    ap.add_argument("--tools", action="store_true",
                    help="let the model call filesystem tools. Off by default; "
                         "needs --tool-policy")
    ap.add_argument("--tool-policy", default="",
                    help="JSON policy with allow/read_roots/write_roots")
    ap.add_argument("--tool-receipts",
                    default=os.path.join("runs", "tool_receipts.jsonl"),
                    help="append-only tool receipt ledger")
    ap.add_argument("--max-tool-calls", type=int, default=2,
                    help="tool calls allowed in one reply")
    ap.add_argument("--experience-log", default="",
                    help="append every exchange to this JSONL file. It holds "
                         "the user's and the model's text; empty keeps only "
                         "the in-memory recent window")
    ap.add_argument("--replay-every", type=int, default=0,
                    help="replay one past exchange every N turns. Needs the "
                         "held-out gate; 0 disables")
    ap.add_argument("--replay-corpus", default="",
                    help="also replay a slice of a .txt folder every other "
                         "replay")
    ap.add_argument("--replay-chars", type=int, default=512,
                    help="characters per corpus replay slice")
    ap.add_argument("--replay-user-only", action="store_true",
                    help="replay only the user's half, not the model's reply")
    ap.add_argument("--replay-seed", type=int, default=0)
    ap.add_argument("--grow-every", type=int, default=0,
                    help="held-out checks between prune/grow decisions; 0 "
                         "leaves the pool size fixed. Needs the gate and a "
                         "disk cap")
    ap.add_argument("--grow-k", type=int, default=1,
                    help="experts added per growth decision")
    ap.add_argument("--grow-min-age", type=int, default=3000,
                    help="steps a grown expert is safe from pruning, and the "
                         "window prune uses")
    ap.add_argument("--grow-mem-frac", type=float, default=0.80,
                    help="do not grow above this fraction of the card")
    ap.add_argument("--grow-max-experts", type=int, default=512)
    ap.add_argument("--grow-max-gb", type=float, default=0.0,
                    help="disk ceiling for growth; required with --grow-every")
    ap.add_argument("--no-prune", dest="prune", action="store_false",
                    help="grow without deleting stale experts")
    args = ap.parse_args()

    if args.tools:
        if not args.tool_policy:
            raise SystemExit("--tools needs an explicit --tool-policy file")
        if args.max_tool_calls < 1:
            raise SystemExit("--max-tool-calls must be at least 1")
        from minagi.agency import (Policy, ReceiptLedger, builtin_registry,
                                   parse_tool_calls, result_text, run)
        try:
            policy = Policy.from_file(args.tool_policy)
            ledger = ReceiptLedger(args.tool_receipts)
        except Exception as e:                            # noqa: BLE001
            raise SystemExit(f"cannot enable tools: {e}")
        if not policy.allowed:
            raise SystemExit("tool policy grants no tools")
        registry = builtin_registry()
        for name in policy.allowed:
            spec = registry.get(name)
            if spec is None:
                raise SystemExit(f"tool policy names unregistered tool: {name}")
            if spec.capability == "filesystem.read" and not policy.read_roots:
                raise SystemExit(f"tool {name} needs a read root")
            if spec.capability == "filesystem.write" and not policy.write_roots:
                raise SystemExit(f"tool {name} needs a write root")
        TOOLS.update(on=True, policy=policy, ledger=ledger,
                     registry=registry, max_calls=args.max_tool_calls,
                     parse=parse_tool_calls, result_text=result_text,
                     run=run)
        print(f"  [tools] ON: {len(policy.allowed)} granted, receipts "
              f"{args.tool_receipts}", file=sys.stderr)
    elif args.tool_policy:
        raise SystemExit("--tool-policy has no effect without --tools")

    from minagi.config import get as _g, load as _lc
    from minagi.precision import set_compute_dtype
    cfg = _lc()
    set_compute_dtype(args.precision
                      or _g(cfg, "training.precision", "bf16"))

    live_chunk = args.learn_chunk or _g(cfg, "training.chunk", 2048)
    gate_on = bool(args.learn) and (args.gate if args.gate is not None
                                    else os.path.isdir(args.held_out))
    if args.learn and args.gate and not os.path.isdir(args.held_out):
        raise SystemExit(f"--gate needs a held-out directory; "
                         f"{args.held_out} is missing")
    # M1: at 2048 the held-out change was inside the instrument; at 512 it
    # was material, and the model's own reply roughly tripled it. A lower
    # chunk is allowed only with the watchdog that can undo it.
    if args.learn and live_chunk < 2048 and not gate_on \
            and not args.allow_ungated_dense:
        raise SystemExit(
            f"live learning at {live_chunk}-character steps is denser than "
            f"the 2048 measured safe in M1 (material held-out damage at 512). "
            f"Use --gate with --held-out {args.held_out}, or "
            f"--allow-ungated-dense if you accept that.")

    if not os.path.exists(os.path.join(args.weights, "core.npz")):
        raise SystemExit(
            f"no model in {args.weights}. Train one first:\n"
            f"    python3 train.py read data/train --save")

    global PRIME
    PRIME = load_prime(args.prime_chars)
    if args.prime_chars and not PRIME:
        print("  no corpus to prime from - the router will choose experts "
              "from the prompt alone", file=sys.stderr)

    if args.learn:
        print(f"\n  LEARNING IS ON. Talking to this model CHANGES "
              f"{args.weights}/ on disk.\n  Use --no-learn to serve it "
              f"read-only.\n")
    if args.learn_context < 0:
        raise SystemExit("--learn-context must be 0 or more")
    try:
        model = load(args.weights, args.device, learn=args.learn,
                     lr=args.learn_lr, save_every=args.save_every,
                     chunk=args.learn_chunk,
                     context=args.learn_context or None)
    except WriterBusy as e:
        raise SystemExit(f"cannot start live learning: {e}") from None
    if PRIME:
        print(f"  primed with {len(PRIME)} characters of self-knowledge")
    pool = getattr(model, "pool", None)
    if pool is not None:
        # Both numbers are the POOL. Comparing model.n_params(), which counts
        # the trunk plus whatever is resident, against the pool total reads as
        # the card holding more than the model has.
        #
        # An in-memory SharedPool has no `vram_params`: everything it holds is
        # on the card, so its resident size is its whole size.
        vram = (pool.vram_params() if hasattr(pool, "vram_params")
                else pool.n_params())
        print(f"  {pool.n_experts()} experts, "
              f"{vram / 1e6:.1f}M of {pool.n_params() / 1e6:.1f}M "
              f"on the card, context {model.cfg.block:,}")
    else:
        print(f"  {model.n_params() / 1e6:.1f}M parameters, "
              f"context {model.cfg.block:,}")

    # Registered before the baseline evaluation, which can take a minute on a
    # large held-out: an interrupt during it should still save the live state.
    _install_exit_hooks()

    if args.learn and gate_on:
        def _rebuild(path):
            """Fresh model + learner from the restored directory."""
            from minagi.live import LiveLearner
            from minagi.recur import load_any
            dev = next(STATE["model"].parameters()).device
            fresh, _ = load_any(path, dev, read_only=False)
            fresh.eval()
            old = STATE["learner"]
            learner = LiveLearner.for_run(
                fresh, weights_dir=args.weights, manifest=_manifest(path),
                lr=args.learn_lr, chunk=old.chunk, context=old.context,
                save_every=args.save_every)
            learner.set_scale(old.scale)
            STATE["model"], STATE["learner"] = fresh, learner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return fresh, learner

        STATE["rebuild"] = _rebuild

        try:
            from minagi.watchdog import HeldOutWatchdog
            watch = HeldOutWatchdog(
                model, STATE["learner"], args.held_out,
                best_dir=args.best_weights, chunk=args.gate_chunk,
                context=args.gate_context or None, chunks=args.gate_chunks,
                factor=args.gate_factor,
                device=next(model.parameters()).device, rebuild=_rebuild,
                retention_log=args.retention_log or None)
            ev = watch.start()
            STATE.update(watchdog=watch, gate=ev,
                         gate_every=args.gate_every)
        except Exception as e:                                # noqa: BLE001
            print(f"  [gate] disabled: {e}", file=sys.stderr)
            gate_on = False
            if live_chunk < 2048 and not args.allow_ungated_dense:
                raise SystemExit(
                    "the gate could not start and the live chunk is denser "
                    "than the measured-safe 2048; use --allow-ungated-dense "
                    "to learn without one")
    elif args.learn and args.gate is False:
        print("  [gate] off - held-out is not being watched; a diverged "
              "stream will be written to disk", file=sys.stderr)
    elif args.learn:
        print(f"  [gate] off - {args.held_out} is not a directory; "
              f"held-out is not being watched", file=sys.stderr)

    if args.replay_every < 0:
        raise SystemExit("--replay-every must be 0 or more")
    if args.replay_every and not gate_on:
        raise SystemExit("--replay-every needs the held-out gate: replay is "
                         "another live stream and only the gate can tell "
                         "whether it is helping")
    if args.grow_every < 0:
        raise SystemExit("--grow-every must be 0 or more")
    if args.grow_every and not gate_on:
        raise SystemExit("--grow-every needs the held-out gate: a bigger "
                         "pool is a model change, and only the gate can "
                         "judge it")
    if args.grow_every and args.grow_max_gb <= 0:
        raise SystemExit("--grow-every needs --grow-max-gb: an expert is "
                         "about 25 MB with its Adam moments, and growth "
                         "must not be allowed to fill the disk")
    if args.grow_every and args.grow_k < 1:
        raise SystemExit("--grow-k must be at least 1")
    if args.grow_every and args.grow_min_age < 1:
        raise SystemExit("--grow-min-age must be at least 1")
    if args.replay_every or args.experience_log:
        from minagi.experience import CorpusSampler, ExchangeLog, Replay
        corpus = None
        if args.replay_corpus:
            corpus = CorpusSampler(args.replay_corpus,
                                   characters=args.replay_chars,
                                   seed=args.replay_seed)
            if not corpus.files:
                print(f"  [experience] no .txt files under "
                      f"{args.replay_corpus}", file=sys.stderr)
                corpus = None
        log = ExchangeLog(args.experience_log or None, seed=args.replay_seed)
        replay = (Replay(log, every=args.replay_every, corpus=corpus,
                         user_only=args.replay_user_only,
                         seed=args.replay_seed)
                  if args.replay_every else None)
        STATE.update(experience=log, replay=replay, turn=0,
                     session=time.strftime("%Y%m%d-%H%M%S")
                     + f"-{os.getpid()}")
        bits = [f"{len(log)} recent"]
        if log.path:
            bits.append(f"log {log.path} (user + model text)")
        if replay is not None:
            bits.append(f"replay every {replay.every} turns"
                        + (", corpus" if corpus is not None else ""))
        print("  [experience] " + "; ".join(bits), file=sys.stderr)

    if args.grow_every:
        pool = getattr(model, "pool", None)
        if (pool is None or not hasattr(pool, "add_experts")
                or not hasattr(pool, "born") or not hasattr(pool, "disk_bytes")):
            raise SystemExit("--grow-every needs a paged checkpoint with "
                             "per-expert ages and on-disk sizes")
        from minagi.growth import PoolKeeper
        from minagi.pool import AutoGrow
        base_step = int(STATE["learner"].start_step)
        # The checkpoint's experts are the originals for this session: their
        # age starts here, so a short serve cannot judge and delete them.
        pool.born.fill_(float(base_step))
        if hasattr(pool, "gate_seen") and hasattr(pool, "gate"):
            pool.gate_seen.copy_(pool.gate.data.abs())
        pool.trial = int(args.grow_min_age)
        if args.grow_max_experts <= pool.n_experts():
            raise SystemExit(f"--grow-max-experts {args.grow_max_experts} is "
                             f"not above the checkpoint's {pool.n_experts()}")
        grower = AutoGrow(grow_k=args.grow_k,
                          max_experts=args.grow_max_experts,
                          mem_frac_max=args.grow_mem_frac,
                          max_disk_gb=args.grow_max_gb)
        STATE["keeper"] = PoolKeeper(
            model, STATE["learner"], grower, survival=args.grow_min_age,
            base_step=base_step, every=args.grow_every, prune=args.prune)
        print(f"  [pool] growth armed: {pool.n_experts()} experts, ceiling "
              f"{args.grow_max_experts}, disk cap {args.grow_max_gb:g} GB, "
              f"every {args.grow_every} gate check(s)", file=sys.stderr)

    print(f"\n  http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
