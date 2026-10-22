# Relationship to upstream

This repository is an **unofficial hardening mirror** of
[`volotat/mini-AGI`](https://github.com/volotat/mini-AGI). It exists because
upstream is a fast-moving research project whose author focuses on model and
training quality, while a handful of correctness fixes reported by the community
have stayed unmerged. This fork keeps a runnable, non-corrupting build on top of
the current upstream, with the smallest possible patch surface.

It is not affiliated with, endorsed by, or a replacement for upstream. Model
architecture, training, benchmarks and the continual-learning results are
upstream's work — go there for those, and open model/training issues there.

## Synced to

`volotat/mini-AGI` at commit
[`ce31731`](https://github.com/volotat/mini-AGI/commit/ce31731903467468ebb77e68a6d79c352491a87a)
(mirror tag `upstream-20261014`; the previous checkpoint `60f7ae3` is kept as
`upstream-20261014-1`). The exported `patches/` apply to exactly this commit;
see the bottom of [`HARDENING.md`](HARDENING.md).

## Branch model

| Ref | What it is |
|-----|------------|
| `main` (default) | `hardening` **plus** the mirror scaffolding and fork-only feature/docs commits, including the live-serving additions documented in the README. This is what you clone and run. |
| `hardening` | Upstream `main` **plus** the correctness patches, one commit per fix. Rebased onto `upstream` on every sync. |
| `upstream` | A pure mirror of `volotat/mini-AGI` `main`, fast-forward only. Never edited. |
| tags `upstream-YYYYMMDD` | Sync checkpoints of the mirror. |

The patches are the commits on `hardening` above `upstream`, one per fix, listed
in [`HARDENING.md`](HARDENING.md) with their upstream issue and PR. The exported
series in `patches/` is formatted from `upstream..hardening`, so it never
includes the mirror scaffolding or fork-only features; regenerate it with
`make patches`.

Fork-only features live on `main`, not on `hardening`. `make sync` rebases
`main` onto the new `hardening`, so those commits stay on `main` and are not
exported in `patches/`. Keeping feature changes in separate files from the
patched code reduces rebase conflicts.

## Syncing upstream (on demand)

```sh
make sync          # or: scripts/sync-upstream.sh
```

`make sync` fetches upstream, moves the `upstream` mirror, rebases the patches
onto it, re-exports `patches/`, and runs the CPU test suite. If a patch no longer
applies it stops and tells you, rather than forcing a broken tree. Review the
result, then push.

The script finds the upstream remote automatically: an `upstream` remote if you
have one, otherwise any remote whose URL points at `volotat/mini-AGI`. Override
with `UPSTREAM_REMOTE=<name> make sync`.

## Fork-only candidates for upstream

The exported `patches/` are the correctness series; the live, ingestion,
self-model, and scoped-tool work on `main` is not in it. No external pull
request has been opened from this checkout. The handoff list, in order of
readiness, is:

1. **Checkpoint prune ordering (M7).** Publish the manifest before deleting
   stale expert files, so an interrupted prune cannot leave the old manifest
   naming a deleted file. This is a small correctness fix with
   `tests/test_soak.py`; it is the strongest candidate for a separate upstream
   PR. It currently lives on `main`, not on `hardening`.

   **Prepared handoff:** commit `858093c` is the candidate. The full diff
   (store + soak + test) applies cleanly to `ce31731` and the current mirror
   `95e901e`, and its three `test_soak.py` cases pass on both. No generated
   patch is tracked. No external PR has been opened. Produce the full or
   store-only diff with:

   ```sh
   mkdir -p runs/upstream-handoff
   git diff 858093c^ 858093c > runs/upstream-handoff/prune-order-full.patch
   git diff 858093c^ 858093c -- minagi/store.py \
       > runs/upstream-handoff/prune-order-minimal.patch
   ```
2. **Retention log and dashboard (M7).** Analysis-only: per-domain held-out
   readings in JSONL and a plot. Low risk, but it introduces a log format and a
   default write path, so it needs a maintainer decision.
3. **Scoped tool registry (M6).** Overlaps upstream #24, which the maintainer
   signalled may be rejected. Do not submit without agreement; this fork keeps
   it off by default and operator-invoked.
4. **Live gate/experience (M1-M3), ingestion (M4), self-model rebuild (M5).**
   Behavior-changing or corpus-policy changes; each needs its own long-run
   evidence and should not ride on the correctness series.

Local acceptance for the fork-only set is the 102-test CPU/CUDA suite plus
`scripts/soak.py` and the retention dashboard. None of it is a substitute for
upstream review.

## Running it

See the upstream README for the model itself. For this mirror:

```sh
make setup     # venv + deps (CPU torch, enough for the tests)
make test      # CPU unit tests
make run       # read a corpus into weights/
make serve     # local web UI
```

`make setup` installs the **CPU** build of PyTorch, which is all the tests need.
To train or serve on a GPU, install the CUDA or ROCm build of PyTorch for your
platform (see the upstream README's install section) into the same virtualenv.
