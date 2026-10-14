# Changelog

Notable changes to **this hardening mirror**. It tracks
[`volotat/mini-AGI`](https://github.com/volotat/mini-AGI); upstream's own changes
are not listed here — see [`UPSTREAM.md`](UPSTREAM.md) for the synced commit.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [v0.1.0-mirror] — 2026-10-14

First tagged snapshot. Tracks upstream `60f7ae3` (mirror tag `upstream-20261014`).

### Carried patches (one commit each)

- **0001** — dry `read` writes go to an overlay, never the source `weights/`
  (upstream #6 / PR #7).
- **0002** — build the dedicated self-knowledge corpus lane (upstream #10 / PR #11).
- **0003** — isolate the Hugging Face dataset cache per fetch (upstream #12 / PR #13).
- **0004** — cap `FileReader._measure` reads by the rotary context, fixing the
  held-out path when `--chunk` exceeds it (upstream #18 / PR #27).
- **0005** — load a created-but-never-saved weights directory through the paging
  path, fixing the `serve.py` startup crash (upstream #17; authored here).
- **0006** — decode the ingest text probe incrementally so a character split by
  the 4 KiB boundary is not rejected (upstream #30; authored here).
- **0007** — drop the bias correction from `GradSNR.ratio()` (upstream PR #28).
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
