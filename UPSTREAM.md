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

## Branch model

| Ref | What it is |
|-----|------------|
| `main` (default) | Upstream `main` **plus** the hardening patches. This is what you clone and run. |
| `upstream` | A pure mirror of `volotat/mini-AGI` `main`, fast-forward only. Never edited. |
| tags `upstream-YYYYMMDD` | Sync checkpoints of the mirror. |

The patches are the commits on `main` above `upstream`, one per fix, listed in
[`HARDENING.md`](HARDENING.md) with their upstream issue and PR.

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
