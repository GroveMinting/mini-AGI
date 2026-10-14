# Handoff

State of the mini-AGI hardening work, written so it can move to another machine
(a GPU box, then a DGX Spark) without the original chat history. If you are
picking this up fresh, read this file first, then [`UPSTREAM.md`](UPSTREAM.md)
and [`HARDENING.md`](HARDENING.md).

## What this is

An **unofficial hardening mirror** of [`volotat/mini-AGI`](https://github.com/volotat/mini-AGI):
the upstream tree plus a small, one-commit-per-fix patchset of correctness bugs
that upstream has not merged. Two goals:

1. **Run/train the model** (eventually on real GPU hardware; a DGX Spark is the
   target).
2. **Own a runnable, non-corrupting fork** that tracks upstream cheaply.

Fork: <https://github.com/GroveMinting/mini-AGI>.

## Current state (as of this handoff)

- Repository: `GroveMinting/mini-AGI` (GitHub user `GroveMinting`).
- Upstream synced: `volotat/mini-AGI` @ `60f7ae3`, mirror tag `upstream-20261014`.
- Release tag: `v0.1.0-mirror` on `main`.
- Patchset: **8 patches**; test suite: **28 tests**, all green (`make test`).
- Branches:
  - `upstream` — pure mirror of upstream `main`, fast-forward only.
  - `hardening` — upstream + the 8 patch commits.
  - `main` (default) — `hardening` + the mirror scaffolding (docs, Makefile,
    sync script, CI).
- CI: `tests` on every push; scheduled `upstream-sync` check.

The patch list and rationale live in [`HARDENING.md`](HARDENING.md); the changelog
is [`CHANGELOG.md`](CHANGELOG.md).

## Rehydrating on a new machine

**Clone to the *same absolute path*, `/home/j/mini-AGI`.** opencode and the
mirror scaffolding assume it; a different path is workable but breaks session
reattachment (below).

```sh
git clone git@github.com:GroveMinting/mini-AGI.git /home/j/mini-AGI
cd /home/j/mini-AGI
git remote add upstream https://github.com/volotat/mini-AGI   # if not present
git remote -v                                                  # expect: origin=fork, upstream=volotat
make sync          # fetch upstream, rebase patches, re-export patches/, run tests
```

### Python environment

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # runtime deps, CPU torch by default
make test                                # 28 tests should pass
```

For a GPU box, replace the CPU torch wheel with the CUDA (or ROCm/ARM64) build
for that machine in the same virtualenv — `make setup` installs the CPU wheel,
which is only enough for the tests. See the upstream README's install section.

### opencode (the agent session)

The agent session is **not** carried by git. Two options:

1. **Fresh session, primed by this file** (chosen transport). Start opencode in
   `/home/j/mini-AGI` and point it at `HANDOFF.md`; it resumes the work.
2. **Carry the transcript** (optional). Export here, import there:
   ```sh
   # on the old machine
   opencode export ses_ec81abcf2ffe6AK40CmAubBdNg > mini-agi-session.json
   # on the new machine, from /home/j
   opencode import mini-agi-session.json
   opencode -s ses_ec81abcf2ffe6AK40CmAubBdNg
   ```
   Sessions are directory-scoped; reattaching cleanly is why the clone path
   matters.

Provider config: `~/.config/opencode/opencode.jsonc` defines the `vllm` provider
pointing at `http://192.168.8.158:8000/v1` (a Spark-hosted vLLM endpoint). The
new box must be able to reach that address, or repoint the config at its own
endpoint. Credentials live in `~/.local/share/opencode/auth.json` (providers
`openai`, `vllm`) — **not committed**; re-run `opencode auth login` on the new
box rather than copying the file.

## Next steps (backlog)

Ordered for the move to GPU hardware:

- **A — end-to-end smoke test, for real.** Tiny `create` → short `read` on a
  small corpus → one `serve.py` request, then add a `make smoke` target. This is
  the first run of the fixed paths (#17/#18/#30) on real data. CPU-feasible
  small; a real short run wants the GPU box.
- **F — DGX Spark target.** GB10 is Grace-ARM + Blackwell: needs an aarch64 +
  CUDA-13 PyTorch wheel. `#3`'s `_auto_device()` already returns `cuda`, so this
  is mostly install recipe + docs, not code.
- **E — upstream outreach** (open PRs, issue reporters). Deliberately **not**
  done; a public-communication decision for the owner.

Done in this pass: carried upstream #3 as patch 0008; recorded #24 as
deliberately not carried; added `CHANGELOG.md`, tag `v0.1.0-mirror`, the synced
SHA, and sync tags.

## Risks / open questions

- **Remote endpoint.** `192.168.8.158:8000` must be reachable from the new box,
  or the provider config repointed.
- **Path matching** for session import (`/home/j/mini-AGI`).
- **Secrets.** Never commit `auth.json`; re-auth on the target.
- **Spark arch.** Confirm the exact torch wheel for GB10 (ARM64 + Blackwell)
  before assuming a plain CUDA install works.
