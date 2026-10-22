"""
Learning from a conversation while it happens.

ONE STREAM. The model reads everything it is given and everything it writes as
a single continuous stream of characters, and takes an optimiser step every
`chunk` of them - the same mechanism minagi/stream.py uses on a corpus, with
the interleaving between subjects removed, because a conversation is one
subject. There is no per-exchange boundary and nothing is re-read: the cursor
only moves forward. What the model sees at every step is the last `context`
characters, and the gradient reaches all of them.

That symmetry is the point. The thing that learns from a book and the thing
that learns from you are the same code path at the same granularity, so there
is no second mechanism to reason about and no regime the model was never
trained in.

WHAT THIS COSTS, AND IT IS NOT MEASURED IN THE CURRENT REGIME. Learning from a
conversation is learning partly from the model's own output, and reading a few
hundred characters of chat under many gradient steps is a far denser diet than
a corpus. In an earlier, denser variant - short turns re-read many times - that
cost around +0.65 nats on held-out against +0.02 for real documents of the same
length, so most of the damage came from the step density rather than from
self-training as such.

Streaming continuously is a lighter regime: the cursor only advances, text
leaves the window, and nothing is read twice at the same position. Whether the
cost is still material here is UNKNOWN and would have to be measured. It is
stated rather than assumed away.

A smaller learning rate is not the fix. Descent has a direction, and on the
model's own argmax that direction is to confirm it; a lower rate confirms it
more slowly.

THE RATE IS THE RUN'S. A live stream continues the stream training reads, so
it continues training's rate as well when it is opened through `for_run`: the
base rate the run used times the plasticity controller's factor, restored from
the weights. A flat rate, as this once used, could be far above the rate
training had come down to. The live gate's own scale stays separate: effective
rate = base x training factor x live scale.
"""

import time

import torch

from .precision import amp


class LiveLearner:
    """One optimiser, and one step per `chunk` characters of the stream."""

    # what minagi/store.py writes into a manifest itself; everything else a
    # manifest holds is carried through a save untouched
    OWN = ("step", "val", "cfg", "telemetry", "n_experts", "d_model", "d_ff",
           "paged", "core_tensors", "router_tensors", "experts",
           "total_bytes")

    def __init__(self, model, lr=3e-4, trunk_lr_mult=0.2, wd=0.1, clip=1.0,
                 aux_weight=0.0, save_every=8, weights_dir=None, manifest=None,
                 chunk=512, context=None, run_rate=False):
        self.model = model
        # What the model was loaded as. save() writes the manifest that names
        # every tensor's shape, and without this it writes an empty one: the
        # directory then claims the default vocabulary of 8192 while holding
        # 265, and will not load again. Caught by reloading after a save,
        # which is the only way this kind of damage shows up.
        self.manifest = dict(manifest or {})
        self.clip = clip
        # Multiplier on the rates below, moved by the held-out watchdog. Kept
        # apart from the optimiser's lr and the training controller so a nudge
        # or a revert never has to reconstruct the per-group ratio.
        self.scale = 1.0
        self.run_rate = bool(run_rate)
        self.plast = None
        if self.run_rate:
            from .plasticity import Plasticity
            self.plast = Plasticity.restore(self.manifest.get("plasticity"))
        self.chars0 = float(self.manifest.get("read_chars") or 0.0)
        self.nats0 = float(self.manifest.get("read_nats") or 0.0)
        self.chars, self.nats = 0.0, 0.0
        # Live steps already folded into the checkpoint's `step` by an
        # earlier session, and steps taken since this learner opened.
        self.steps0 = int(self.manifest.get("live_steps") or 0)
        self.aux_weight = aux_weight
        self.save_every = save_every
        self.weights_dir = weights_dir
        self.steps = 0
        self.unsaved = 0
        self.log = []
        # the stream. `buf` holds the last `context` character ids and nothing
        # older; `pending` counts characters read since the last step.
        self.chunk = int(chunk)
        self.context = int(context or getattr(model.cfg, "block", 16384))
        self.buf = []
        self.pending = 0
        dev = next(model.parameters()).device
        if getattr(model, "pool", None) is not None:
            # training's own split: router rows and the depth embedding are
            # the pool's, not the trunk's
            from .pool import split_trunk_pool
            trunk, pool_ps = split_trunk_pool(model)
        else:
            trunk = [p for n, p in model.named_parameters()
                     if not n.startswith("pool.")]
            pool_ps = [p for n, p in model.named_parameters()
                       if n.startswith("pool.")]
        self.opt = torch.optim.AdamW(
            [{"params": trunk, "name": "trunk", "weight_decay": wd,
              "lr": lr * trunk_lr_mult, "base_lr": lr * trunk_lr_mult},
             {"params": pool_ps, "name": "pool", "weight_decay": wd,
              "lr": lr, "base_lr": lr}],
            lr=lr, betas=(0.9, 0.95), fused=(dev.type == "cuda"))
        pool = getattr(model, "pool", None)
        if pool is not None and hasattr(pool, "attach_optimiser"):
            pool.attach_optimiser(self.opt)
        # ADAM'S MOMENTS carry on from where the weights directory left them,
        # as they do when the trainer resumes. A fresh optimiser would start
        # every tensor from zero moments at the first step - the trunk's and
        # every expert on the card, instead of their own - and its first
        # save would write that short history over the trainer's in
        # optim.npz, step counters included. The experts' own moments are
        # put in place by the pool before each step (PagedPool._own_moments).
        if weights_dir:
            from . import store as weights_store
            try:
                weights_store._load_optim(self.opt, model, weights_dir)
            except Exception as e:             # never stop the server for it
                print(f"  [note] optimiser moments not restored: {e}")
        self._apply_rate()

    def feed(self, text, tok, note=""):
        """
        Add text to the stream and take a step for every `chunk` characters.

        Everything goes in - what you typed and what the model wrote - because
        the stream is what the model is reading, and it does not get to skip
        the parts it produced. Returns one record per step taken, which may be
        none for a short message and several for a long one. The caller holds
        whatever lock guards the model.
        """
        ids = list(tok.encode(text).ids)
        recs, i = [], 0
        while i < len(ids):
            take = min(self.chunk - self.pending, len(ids) - i)
            self.buf.extend(ids[i:i + take])
            self.pending += take
            i += take
            if self.pending >= self.chunk:
                # trim to the window BEFORE the step: what the model sees is
                # the last `context` characters, exactly as the corpus reader
                # sees them, and the gradient reaches all of it
                if len(self.buf) > self.context:
                    del self.buf[:-self.context]
                rec = self._step(note)
                if rec is not None:
                    recs.append(rec)
                self.pending = 0
        if len(self.buf) > self.context:
            del self.buf[:-self.context]
        return recs

    def learn(self, text, tok, note=""):
        """Compatibility: feed the text and return the last step, if any."""
        recs = self.feed(text, tok, note=note)
        return recs[-1] if recs else None

    def factor(self):
        """Training's controller factor on the base rate, or 1 without one."""
        return self.plast.factor() if self.plast is not None else 1.0

    @property
    def start_step(self):
        """The model step this learner starts from, including live steps
        carried in the manifest from an earlier process."""
        return int(self.manifest.get("step") or 0) + self.steps0

    @property
    def model_step(self):
        """The model step including this session's live steps."""
        return self.start_step + int(self.steps)

    def fold_steps(self, step):
        """Record that `step` now lives in the checkpoint manifest, and count
        the session's live steps from zero again. Growth uses this before its
        save so the new learner does not count the same steps twice."""
        self.manifest["step"] = int(step)
        self.steps0 = 0
        self.steps = 0

    def effective_scale(self):
        """The multiplier the optimiser is actually using."""
        return self.scale * self.factor()

    def _apply_rate(self):
        scale = self.effective_scale()
        for group in self.opt.param_groups:
            group["lr"] = group.get("base_lr", group["lr"]) * scale

    def _step(self, note=""):
        """One optimiser step on the window as it stands."""
        ids = self.buf
        if len(ids) < 8:
            return None
        dev = next(self.model.parameters()).device
        was_training = self.model.training
        self._apply_rate()
        self.model.train()
        try:
            with amp(dev):
                _, loss = self.model(
                    torch.tensor([ids[:-1]], device=dev),
                    torch.tensor([ids[1:]], device=dev))
                if self.aux_weight and getattr(self.model, "pool", None):
                    loss = loss + self.aux_weight * self.model.pool_aux()
                if hasattr(self.model, "pool_balance"):
                    loss = loss + self.model.pool_balance()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip))
            self.opt.step()
        finally:
            self.opt.zero_grad(set_to_none=True)
            if not was_training:
                self.model.eval()
        self.steps += 1
        self.unsaved += 1
        loss_value = float(loss.detach())
        self.chars += self.chunk
        self.nats += loss_value * self.chunk
        if self.plast is not None:
            self.plast.tick()
        rec = {"step": self.steps, "chars": len(ids), "loss": loss_value,
               "grad_norm": gn, "lr": self.opt.param_groups[-1]["lr"],
               "scale": self.effective_scale(), "note": note,
               "at": time.time()}
        self.log.append(rec)
        return rec

    def set_scale(self, scale):
        """Multiply every group's base rate by `scale` (bounded LR control)."""
        self.scale = max(0.0, float(scale))
        self._apply_rate()

    def scale_lr(self, factor):
        """Move the current scale by `factor`; used when the gate reverts."""
        self.set_scale(self.scale * float(factor))

    def reset_stream(self):
        """Drop the partial window after a revert; the next turn starts clean."""
        self.buf.clear()
        self.pending = 0

    def due_to_save(self):
        return (self.weights_dir and self.save_every
                and self.unsaved >= self.save_every)

    def save(self, extra=None):
        """Persist what has been learned. Without this it dies with the process."""
        if not self.weights_dir:
            return False
        from . import store as weights_store
        from dataclasses import asdict, is_dataclass
        cfg = self.manifest.get("cfg")
        if not cfg:
            mc = getattr(self.model, "cfg", None)
            cfg = asdict(mc) if is_dataclass(mc) else (dict(mc.__dict__)
                                                       if mc else None)
        # everything the directory carried that the store does not write
        # itself; live saves used to drop the controller and the base rate
        keep = {k: v for k, v in self.manifest.items() if k not in self.OWN}
        keep.update({"read_chars": int(self.chars0 + self.chars),
                     "read_nats": self.nats0 + self.nats,
                     # the checkpoint's step is the base; live steps are
                     # carried separately so a restart can continue counting
                     "live_steps": int(self.steps0 + self.steps),
                     # an old checkpoint may predate lr in the manifest;
                     # record the base this session actually used
                     "lr": float(self.opt.param_groups[-1]["base_lr"])})
        if self.plast is not None:
            keep["plasticity"] = self.plast.state()
        keep.update(extra or {})
        weights_store.save(self.model, self.weights_dir, opt=self.opt,
                           step=self.manifest.get("step"),
                           val=self.manifest.get("val"),
                           cfg=cfg, extra=keep)
        self.unsaved = 0
        return True

    @classmethod
    def for_run(cls, model, weights_dir=None, manifest=None, lr=None,
                chunk=None, context=None, save_every=8, config=None):
        """A learner set up as training is: config.yaml's training settings,
        the base rate the run used (the manifest's `lr`) unless `lr` says
        otherwise, and the controller where the run left it."""
        from .config import get, load
        c = config if config is not None else load()
        man = dict(manifest or {})
        base = (float(lr) if lr is not None
                else float(man.get("lr") or get(c, "training.lr", 3e-4)))
        return cls(model, lr=base,
                   aux_weight=float(getattr(getattr(model, "cfg", None),
                                            "pool_aux", 0.0) or 0.0),
                   trunk_lr_mult=get(c, "training.trunk_lr_mult", 0.1),
                   wd=get(c, "training.weight_decay", 0.1),
                   clip=get(c, "training.clip", 1.0),
                   chunk=chunk or get(c, "training.chunk", 2048),
                   context=context, save_every=save_every,
                   weights_dir=weights_dir, manifest=man, run_rate=True)


def exchange_text(user_text, bot_text, u0="<user>", u1="</user>",
                  b0="<bot>", b1="</bot>"):
    """
    One turn, marked up the way the chat corpus is.

    The user's half is the part that carries information: the model did not
    predict it, and that surprise is the whole signal in the exchange. Its own
    half is the argmax and mostly confirms what it already believed - kept
    because what it said is part of what happened, not because it teaches
    much.
    """
    user_text = (user_text or "").strip()
    bot_text = (bot_text or "").strip()
    if not user_text and not bot_text:
        return ""
    return f"{u0}\n{user_text}\n{u1}\n{b0}\n{bot_text}\n{b1}\n"
