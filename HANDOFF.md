# Handoff

State of the mini-AGI hardening work, written so it can move to another machine
without the original context. If you are picking this up fresh, read this file
first, then [`UPSTREAM.md`](UPSTREAM.md) and [`HARDENING.md`](HARDENING.md).

## What this is

An **unofficial hardening mirror** of [`volotat/mini-AGI`](https://github.com/volotat/mini-AGI):
the upstream tree plus a small, one-commit-per-fix patchset of correctness bugs
that upstream has not merged. Two goals:

1. **Run/train the model** on real GPU hardware.
2. **Own a runnable, non-corrupting fork** that tracks upstream cheaply.

It is not affiliated with, endorsed by, or a replacement for upstream.

## Current state

- Upstream synced: `volotat/mini-AGI` @ `ce31731`, mirror tag `upstream-20261014`
  (previous checkpoint `60f7ae3` kept as `upstream-20261014-1`). The mirror
  branch is at `95e901e`; upstream `main` has advanced to `aa9d53f`, whose
  `serve.py`/`LiveLearner` API changes still need a real port.
- `main` is at `9a02281`, which carries upstream PR #32 (Windows runtime fixes)
  and the selective `aa9d53f` backport, in addition to the fork-only
  live-serving, ingestion, self-model, scoped-tool, retention, and soak
  additions documented in [README.md](README.md).
- Patchset: **7 patches**, tracked as `patches/*.patch`; test suite: **126 tests**
  (4 skipped on CPU; 3 CUDA-gated). `make test` is green on CPU
  (`make test-gpu` also runs the CUDA tests on a card); `make smoke` runs the
  end-to-end path.
- A selective backport of upstream `aa9d53f` was merged into `main` at
  `9a02281`: live rate/plasticity restoration, shared trunk/pool split, and
  spread whole-block held-out evaluation with level migration. Paged
  newborn-rate and lane-scheduler changes are intentionally deferred.
- Verified on an RTX 5080 Laptop (Blackwell, `sm_120`): CUDA suite **126 pass /
  1 skip**, `make smoke` OK, and all four soak scenarios pass.
  Real-weight calibration on a copied 174-expert checkpoint: old top `0.7279`,
  new spread whole `0.7538`, gate baseline `0.7669 +/-0.1010`; live effective
  rate `1.78e-4` from base `3e-4` and training factor `0.5946`.
  The P0 follow-up fixes (gate-best persistence and live-save `lr`) are in the
  working tree on top of `9a02281`, not yet committed.
- Documentation rule: reconcile `HANDOFF.md`, `DEVPLAN.local.md`, `UPSTREAM.md`,
  and `CHANGELOG.md` at the end of every iteration, before the next one starts.
- Single-writer enforcement (2026-10-22): `minagi/writer_lock.py` locks a
  weights directory beside the model; `serve.py` learning, `train.py read
  --save`, and `ingestd` hold it; read-only serving does not. Dragon refused a
  second writer with pid/host/purpose in the message.
- Retention rows now record the held-out instrument (directory, chunk, context,
  chunks, spread, domains); `summary()` uses the latest instrument segment, so
  a migration starts a fresh baseline instead of subtracting unlike scores.
- Live-step restart verified on Dragon (2026-10-22): one session step saved
  `live_steps=1` with base step `223506`; restarting reported
  `model_step=223507`, so the counter continues without double-counting.
- Growth calibration (bounded, 2026-10-22): 0.92M characters of real corpus,
  17.8 min, normal LR, 174 -> 179 experts (+5) with no prune; held-out
  `0.7706 -> 0.8913` (+0.1207, inside 2 SE). Used `--prune-survival 400000`
  for the bounded run; the full default-survival run is still pending.
- P0 fresh-baseline soak (2026-10-22): 80 turns / 23 live steps on a fresh copy
  of the 174-expert checkpoint at `9a02281`; three `improved` gate readings,
  no revert or crash, clean shutdown save, and a successful restart.
  Baseline `0.9095`, best `0.9006`, final `0.9070`; artifacts under
  `/home/jg/aa9d-calib/p0-*`. Restart reloaded 82 exchanges but reset the gate
  baseline (`0.9096`) and `model_step` (`223506`). All three are fixed: the gate
  keeps a compatible better snapshot, live saves record `lr`, and `live_steps`
  now carries the model-step count across restart; growth folds it before
  rebuilding so it is not counted twice.
- `main` (default) is `hardening` (upstream + patch commits) plus the mirror
  scaffolding and the fork-only live-serving, ingestion, self-model,
  scoped-tool, retention, and soak additions. These additions are not upstream
  features and are not in the exported `patches/` series. The exported series is
  `upstream..hardening` in apply order; regenerate it with `make patches` after
  rebasing.
- CI: `tests` on every push; scheduled `upstream-sync` dry-run.

The patch list and rationale live in [`HARDENING.md`](HARDENING.md); the changelog
is [`CHANGELOG.md`](CHANGELOG.md).

## Rehydrating on a new machine

```sh
git clone <fork-url> mini-AGI
cd mini-AGI
git remote add upstream https://github.com/volotat/mini-AGI   # if not present
git remote -v                                                  # expect: origin=fork, upstream=volotat
make test                                                      # CPU suite must pass first
```

### Python environment

Any Python 3.10+ works. This project is commonly driven through `uv`, but a
plain `python3 -m venv` is fine:

```sh
uv venv .venv && . .venv/bin/activate
uv pip install -r requirements.txt                 # runtime deps (torch not pinned)
make test                                          # tests pass on CPU torch
```

`make setup` installs the **CPU** torch wheel, which is only enough for the
tests. For a GPU box, install the CUDA/ROCm build for that card into the same
virtualenv instead — see the upstream README's install section.

### GPU notes

The card only matters for the code paths that page experts and compute in bf16:
`minagi/precision.py` enables bf16 autocast on any NVIDIA card from compute
capability 8 on, and `minagi/paged.py` keeps 32 experts resident. The reference
environment is an RTX 3070 Laptop (8 GB); newer cards (RTX 40/50 series, sm_120
Blackwell) need a torch build with the matching CUDA architecture. Run
`python3 tools/device_check.py` on a new card to measure its kernels, precisions
and one learning step before trusting a run.

## Backlog

Ordered by risk to the next iteration:

- **P1 upstream decision.** Compare the remaining `aa9d53f` API port against
  the value of lane scheduling, paged newborn rates, and performance changes.
- **P1 operational hardening.** Done 2026-10-22: writers take an advisory
  per-directory lock; retention rows carry their held-out instrument and start
  a new baseline when it changes; the P0 restart accounting is verified.
- **Orin disposition: short checks only.** Do not target Orin for long or
  unattended gate/replay soaks; use Dragon.
- **Upstream decision: stay selective.** No full `aa9d53f` sync now. The
  remaining work is lane scheduling, paged newborn rates, and performance-only
  code; revisit only if a concrete capability needs it.
- **Growth calibration.** The bounded 0.92M-character run exercised growth at
  normal LR (174 -> 179, no prune). Still run a full default-survival
  growth/prune sequence before deciding whether serve-time growth stays
  experimental.
- **Operational hardening.** The `live_steps` restart path, single-writer
  enforcement, and retention instrument migration are done.
- **Scope decision.** Decide whether “improve the codebase itself” is a real
  autonomous goal or an operator-assisted maintenance workflow.

Done on this machine: the `ce31731` sync (rebased `hardening`, dropped the
superseded text-probe patch, re-exported `patches/`, replayed scaffolding);
`make smoke` (tiny create → read --save → serve → one request); the CUDA-gated
test tier (`tests/test_gpu_paging.py`); the `gcnArchName`-on-CUDA fix in
`tools/device_check.py`; the fork-only live-serving path (probe, held-out gate,
experience/replay), the ingestion daemon, the self-knowledge rebuild tool, the
scoped tool registry and receipt ledger, the retention log/dashboard, the
fault-injection soak, and its then-102-test suite; upstream PR #32 (Windows UTF-8
logs, direct plot spawns, retrying checkpoint replace) integrated with the
fork's manifest writer at `74bec51`; and this documentation pass. The suite is
now 126 tests. The selective `aa9d53f` backport (live rate/manifest, shared
trunk/pool split, spread whole-block held-out with level banners) is merged at
`9a02281` and validated on Dragon. P0 soak accounting is closed.

## Risks / open questions

- **Network for corpora.** `python3 -m corpora all` fetches public datasets;
  confirm the build host can reach Hugging Face before a first run.
- **Architecture match.** Confirm the torch wheel's CUDA architecture covers the
  card (sm_120 for Blackwell) before assuming a plain CUDA install works.
- **WSL2.** GPU passthrough works under WSL2, but the card is shared with the
  host: check it is free before `tools/device_check.py` (which refuses a busy
  card unless `--force`).
- **Orin 8 GB live soak.** Longer gated live/replay sessions can leave the
  Orin host unresponsive even though `ping` still works. Keep Orin runs short
  and use Dragon for long soaks until the hang is understood.
- **One writer per weights directory.** `serve.py`, `train.py`, `ingestd.py`,
  and `tools/rebuild_self_knowledge.py --read` write the same weights and
  optimiser files. Do not run two of them against one directory.
- **Tool use is off by default and is a capability sandbox.** `serve.py` cannot
  call a tool without `--tools` plus a policy file. Grants name exact tools and
  roots; there is no shell, network, arbitrary Python, or wildcard. Path checks
  resolve symlinks, but a same-user race between check and open remains. Do not
  grant `/` or a secrets directory. Receipts contain paths, decisions, and
  digests, not file contents; the hash chain detects edited or truncated lines,
  not a full rewrite by someone with write access to the ledger.
- **Model tool loop.** A reply may call at most `--max-tool-calls` tools. Tool
  results are escaped before re-injection and are not fed to the live learner;
  the model's own call text is. This path is opt-in and does not replace the
  held-out gate.
- **Saves are atomic per file, not one transaction across shape changes.**
  Bundles are written to `.tmp` and renamed, the manifest is published last,
  and pruned expert files are deleted only after it. An interrupted stable-shape
  save leaves the old manifest loadable; a crash during growth can leave a new
  core/router with an old manifest, so retry growth from the published
  checkpoint rather than resuming a half-written one. `scripts/soak.py` covers
  the stable and prune cases.
- **Retention log.** `serve.py` writes `runs/retention.jsonl` by default while
  the gate is on. It contains domain names and losses, not conversation text;
  `--retention-log ""` disables it. Rows are append-only by convention, not a
  security boundary; a restarted run appends at the end.
- **Branch model.** Fork-only feature/docs commits live on `main`; `make sync`
  rebases `main` onto the new `hardening`, so they stay on `main` and are not
  exported in `patches/`. Keep features out of files the hardening patches
  touch to reduce rebase conflicts.
