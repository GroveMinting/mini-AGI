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
  (previous checkpoint `60f7ae3` kept as `upstream-20261014-1`).
- Latest release tag: `v0.1.2-mirror`. `main` also carries the unreleased,
  fork-only live-serving, ingestion, and self-model additions documented in
  [README.md](README.md).
- Patchset: **7 patches**, tracked as `patches/*.patch`; test suite: **78 tests**
  (75 CPU + 3 CUDA-gated). `make test` is green on CPU (`make test-gpu` also runs
  the CUDA tests on a card); `make smoke` runs the end-to-end path.
- Verified on an RTX 5080 Laptop (Blackwell, `sm_120`): `tools/device_check.py`,
  `make smoke`, and the CUDA test tier all pass. The fork-only live,
  ingestion, and self-model paths are covered by the test suite; the live path
  was exercised on real weights.
- `main` (default) is `hardening` (upstream + patch commits) plus the mirror
  scaffolding and the fork-only live-serving, ingestion, and self-model
  additions. These additions are not upstream features and are not in the
  exported `patches/` series. The exported series is `upstream..hardening` in
  apply order; regenerate it with `make patches` after rebasing.
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

Ordered for GPU work:

- **Training run.** Start a short real run to exercise growth/pruning/continual
  learning on the card.
- **Upstream outreach** (open PRs, issue reporters). Deliberately **not** done; a
  public-communication decision for the owner.

Done on this machine: the `ce31731` sync (rebased `hardening`, dropped the
superseded text-probe patch, re-exported `patches/`, replayed scaffolding);
`make smoke` (tiny create → read --save → serve → one request); the CUDA-gated
test tier (`tests/test_gpu_paging.py`); the `gcnArchName`-on-CUDA fix in
`tools/device_check.py`; the fork-only live-serving path (probe, held-out gate,
experience/replay), the ingestion daemon, the self-knowledge rebuild tool, and
its 78-test suite; and this documentation pass.

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
- **Branch model.** Fork-only feature/docs commits live on `main`; `make sync`
  rebases `main` onto the new `hardening`, so they stay on `main` and are not
  exported in `patches/`. Keep features out of files the hardening patches
  touch to reduce rebase conflicts.
