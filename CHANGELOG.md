# Changelog

Notable changes to **this hardening mirror**. It tracks
[`volotat/mini-AGI`](https://github.com/volotat/mini-AGI); upstream's own changes
are not listed here — see [`UPSTREAM.md`](UPSTREAM.md) for the synced commit.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

Fork-only additions. These are not upstream features.

### Added

- `replication/live_probe.py` + `make live-probe` — measure the held-out cost of
  the live conversation stream without writing the source weights.
- `minagi/watchdog.py` — held-out gate for serving: keep the best state, revert
  on a real regression, and halve the live learning rate.
- `minagi/experience.py` — bounded exchange history, an optional JSONL log, and
  replay through the same learner and gate.
- `serve.py` flags for the held-out gate, `--learn-context`, the experience log,
  and replay.
- Tests for the probe, gate, experience/replay, and serve wiring.
- `ingestd.py` + `minagi/ingestd.py` — opt-in scheduled ingestion of files and
  directories, per-source rate limits, persistent offsets, and the same
  held-out gate; `--dry-run` and `--once` for inspection.
- Tests for ingestion offsets, source opt-in, rate limits, and gate rebinding.
- `tools/rebuild_self_knowledge.py` — regenerate the self-knowledge lane from a
  checkpoint manifest, expand train/val, and optionally re-read the lane;
  `self_knowledge.yaml` now points at an existing tool.
- Tests for manifest-driven self-knowledge rebuilds and the non-growing read
  command.
- `minagi/agency.py` + `tools/agency.py` — opt-in filesystem tool registry with
  exact-name grants, explicit read/write roots, path confinement, bounded
  results, and a hash-chained append-only receipt ledger. `serve.py` runs a
  bounded `<tool>` loop only with `--tools --tool-policy`; tool use is off by
  default. There is no shell, network, or arbitrary Python tool.
- Tests for deny-by-default, root escapes and symlinks, argument validation,
  result bounds, the tool parser/result escaping, receipt tampering, and the
  CLI; the suite is now 95 tests (3 CUDA-gated).

### Fixed

- `PoolKeeper` now protects the checkpoint's original experts from serve-time
  pruning; a short survival window could previously delete them. Added a test
  that prunes a stale grown expert while keeping the checkpoint.
- The self-knowledge YAML's rebuild-tool reference now resolves.

### Changed

- `serve.py` learns by default. With `data/val` present, the held-out gate is on
  by default; a live chunk below 2048 characters is refused unless the gate is
  enabled.
- `README.md` documents the fork-only live, ingestion, self-model, and scoped
  tool paths; `HANDOFF.md` current state refreshed.

## [v0.1.2-mirror] — 2026-10-14

GPU bring-up on an NVIDIA Blackwell card, and the first end-to-end verification
of the fixed paths outside the unit tests.

### Added

- `scripts/smoke.py` + `make smoke` — tiny `create` → short `read --save` →
  `serve.py` → one request, all in a temp directory. Exercises the paths
  patches 0001/0004/0006 touch, on a chosen device. Nothing touches `weights/`
  or `data/`.
- `tests/test_gpu_paging.py` — CUDA-gated tests (`skipUnless`): a
  never-saved directory loads paged, a forward/backward reaches the resident
  experts, and a save-less read leaves the source weights byte-identical. The
  CPU CI suite skips them; `make test-gpu` runs them on a card.
- `make test-gpu`.

### Fixed

- `tools/device_check.py` printed the device name twice on NVIDIA: it read
  `gcnArchName` unconditionally, but torch ≥ 2.11 exposes that attribute on CUDA
  too (holding the marketing name). It is now used only on HIP builds; CUDA
  reports `sm_<major><minor>` again.
- `make patches` exported from `HEAD`, which would fold the mirror scaffolding
  into the patch series; it now exports `upstream..hardening`.

### Changed

- Restored `HARDENING.md` and `HANDOFF.md`, rewritten without machine-specific
  session or endpoint details.
- `UPSTREAM.md` now documents the `hardening` branch (upstream + patches) that
  `main` (upstream + patches + scaffolding) is built from.
- **Synced to upstream `ce31731`** (from `60f7ae3`): `hardening` rebased,
  `patches/` re-exported, scaffolding replayed onto `main`.

### Removed

- Patch **0007** (boundary-safe text probe) was dropped: upstream `ce31731`
  independently fixes **#30** in `minagi/ingest.py` (`why_not_text`, decoding the
  sample incrementally with `final=not cut` and bumping the verdict-cache
  version). `tests/test_text_probe.py` is kept as a regression test; it passes
  against upstream's implementation unchanged. With 0007 gone the MPS patch is
  now **0007** and the series is seven patches.

### Measured

- RTX 5080 Laptop (Blackwell, `sm_120`, 16 GB), torch 2.11.0+cu128, driver
  610.53: `tools/device_check.py` uses the fused SDPA kernel for both the
  4,096-window and cached-chunk cases, computes bf16 natively, and reads at
  **1,313 char/s bf16** (819 char/s fp32) for a 4,096-character window at chunk
  2,048, peaking at 3.66 GB (5.73 GB fp32). The full suite is 32 tests green
  (29 CPU + 3 GPU).

## [v0.1.1-mirror] — 2026-10-14

Fixes a crash introduced by patch 0007 in v0.1.0.

### Fixed

- **0007 (text probe).** `looks_like_text` divided by zero on a 1-2 byte file
  that is a truncated UTF-8 lead sequence (decodes to nothing), aborting a
  corpus scan. The empty-decode guard is folded into patch 0007 itself, so the
  series stays correct commit-by-commit. Upstream's original strict decode
  rejected such a file rather than crashing; this restores that behaviour while
  keeping the boundary-split fix.

### Documentation

- Corrected the patch-number column in `HARDENING.md` and `CHANGELOG.md` to
  match the exported `patches/` order (patches 0005-0007 were mislabelled).

## [v0.1.0-mirror] — 2026-10-14

First tagged snapshot. Tracks upstream `60f7ae3` (mirror tag `upstream-20261014`).

### Carried patches (one commit each)

- **0001** — dry `read` writes go to an overlay, never the source `weights/`
  (upstream #6 / PR #7).
- **0002** — build the dedicated self-knowledge corpus lane (upstream #10 / PR #11).
- **0003** — isolate the Hugging Face dataset cache per fetch (upstream #12 / PR #13).
- **0004** — cap `FileReader._measure` reads by the rotary context, fixing the
  held-out path when `--chunk` exceeds it (upstream #18 / PR #27).
- **0005** — drop the bias correction from `GradSNR.ratio()` (upstream PR #28).
- **0006** — load a created-but-never-saved weights directory through the paging
  path, fixing the `serve.py` startup crash (upstream #17; authored here).
- **0007** — decode the ingest text probe incrementally so a character split by
  the 4 KiB boundary is not rejected; a sample that decodes to nothing is
  rejected rather than divided by zero (upstream #30; authored here).
- **0008** — auto-detect MPS as a first-class device (upstream PR #3; carried).

### Infrastructure

- `make` targets (`setup`, `test`, `run`, `serve`, `sync`, `patches`).
- `scripts/sync-upstream.sh` — on-demand rebase of the patchset onto upstream,
  re-export of `patches/`, and the CPU test suite; tags each sync
  `upstream-YYYYMMDD`.
- CI: `tests` on every push, and a scheduled `upstream-sync` check.
- Docs: `HARDENING.md` (patch inventory and rationale), `UPSTREAM.md` (branch
  model and sync), `HANDOFF.md` (state for moving the work to another machine).

### Recorded, not carried

- Upstream **#9** and **#20** are covered by current upstream; dropped.
- Upstream **#24** (interop tool-loop, self-directed learning, governance ledger,
  health certificate) is out of scope for a correctness mirror; not carried.
