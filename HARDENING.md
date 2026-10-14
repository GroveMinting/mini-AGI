# Hardening patches

This fork tracks [`volotat/mini-AGI`](https://github.com/volotat/mini-AGI) and
carries a small set of correctness patches that upstream has not merged. The
default branch `main` is the upstream tree **plus** these commits, one commit
per fix, so a rebase is the whole update operation (see
[`scripts/sync-upstream.sh`](scripts/sync-upstream.sh)).

Each patch maps to an upstream issue and pull request. A patch is carried only
while it still fixes something real; if upstream covers it, the patch is dropped
and recorded below.

## Carried

| # | Patch | Upstream issue | Upstream PR | Test |
|---|-------|----------------|-------------|------|
| 0001 | Keep dry reads off source weights | [#6](https://github.com/volotat/mini-AGI/issues/6) | [#7](https://github.com/volotat/mini-AGI/pull/7) | `tests/test_store_shadow.py` |
| 0002 | Build a dedicated self-knowledge corpus lane | [#10](https://github.com/volotat/mini-AGI/issues/10) | [#11](https://github.com/volotat/mini-AGI/pull/11) | `tests/test_self_knowledge_corpus.py` |
| 0003 | Isolate the Hugging Face dataset cache | [#12](https://github.com/volotat/mini-AGI/issues/12) | [#13](https://github.com/volotat/mini-AGI/pull/13) | `tests/test_fetch_cache.py` |
| 0004 | Cap `FileReader._measure` by context | [#18](https://github.com/volotat/mini-AGI/issues/18) | [#27](https://github.com/volotat/mini-AGI/pull/27) | `tests/test_file_reader_measure_context.py` |
| 0005 | Drop GradSNR `ratio()` bias correction | [#28](https://github.com/volotat/mini-AGI/pull/28) | [#28](https://github.com/volotat/mini-AGI/pull/28) | `tests/test_gradsnr.py` |
| 0006 | Load a never-saved dir through paging | [#17](https://github.com/volotat/mini-AGI/issues/17) | — | `tests/test_paged_load.py` |
| 0007 | Boundary-safe text probe | [#30](https://github.com/volotat/mini-AGI/issues/30) | — | `tests/test_text_probe.py` |
| 0008 | Auto-detect MPS as a first-class device | — | [#3](https://github.com/volotat/mini-AGI/pull/3) | `tests/test_device_autodetect.py` |

The `#` column is the patch number in the exported `patches/` series, which
follows commit order. Patches 0005 and 0008 come from upstream PRs (#28, #3) that
are unmerged; 0006 and 0007 have no upstream PR at all — they were authored here
from the reports (credited in the commit trailers). All four are this mirror's
own upkeep.

### 0001 — dry reads must not touch the source checkpoint

`train.py read` without `--save` still trains in memory. Previously a dirty
expert evicted from a full RAM cache was written back into the source
`weights/` directory while the trunk, routers, optimiser bundle and manifest
were not — a hybrid checkpoint. The patch routes those writes to a temporary
overlay beside the model, reads unchanged experts from the source, prefers the
overlay after eviction, and refuses to checkpoint an overlay rather than
producing a mixed save.

### 0002 — the self-knowledge lane is actually built

`corpora all` declared `data/train/self-knowledge`, but the builder ran the
generic chat generator and expanded the `chat` lane, so the folder `serve.py`
primes from was never created. The patch adds a `--self-only` mode backed by
`self_knowledge.yaml`, expands it into its own lane, bounds it with `--limit`,
and treats a nonzero builder return code as a failure even when stale output
files exist.

### 0003 — the dataset cache is not someone else's to delete

`corpora fetch` recursively deleted the shared `~/.cache/huggingface/datasets`
on every run, which can hold unrelated datasets from other projects. The patch
gives each non-`--keep-cache` fetch a process-owned temporary cache inside the
output directory (so large Arrow intermediates use the corpus filesystem, not a
small `/tmp`), passes it to `load_dataset(cache_dir=...)`, and removes only that
path. `--keep-cache` still uses the normal shared cache.

### 0004 — evaluation must not read past the rotary tables

`FileReader._measure` (the `learn=False` held-out path) reset its cache when the
window filled but never capped the read length. With `--chunk` larger than the
context it forwarded a full chunk from position 0 and tripped the rotary-table
check (`reading at position N but the rotary tables were built to M`). The patch
caps each read at the remaining table (`min(n, context - seen)`) and returns
`None` when nothing is left. Adapted from PR #27 (kevin9327); the test was ported
from pytest to the repo's unittest runner.

### 0005 — GradSNR `ratio()` must not inflate the reading

`GradSNR.ratio()` divided both running averages by `1 - beta**n` (the Adam-style
bias correction), but these averages start at the first reading, not at zero, so
the correction only inflated the ratio — 6.7x at the eighth reading, where it is
first reported, so a constant gradient read above 1. The patch returns
`||m||^2 / sq` directly. Diagnostic only: `ratio()` feeds the `signal` field in
the progress line and nothing steers on it. From PR #28 (10xCrazyHorse); the test
was ported from pytest to unittest.

### 0006 — a created-but-never-saved directory loads paged

`minagi/recur.py::_load_dir` only took the paging path when the manifest carried
`"paged": true`, a flag first written on a paged save. A directory written by
`create()` or by a save-less read already has the on-disk `experts/` layout but
no flag, so it loaded into the in-memory `SharedPool` — which has no
`vram_params`, crashing `serve.py`'s startup banner. The patch prefers the
paging path when `experts/` is on disk and guards the banner call. From the
report by eletec; upstream has no PR.

### 0007 — the text probe must not split a character

`minagi/ingest.py::looks_like_text` decoded its 4,096-byte sample strictly, so a
file whose 4,096th byte fell inside a multi-byte character raised
`UnicodeDecodeError` and was silently dropped from training and evaluation. The
patch decodes incrementally (`final=False`) so a partial trailing sequence is
buffered instead of treated as an error; a genuinely invalid sequence earlier in
the sample still fails, and a sample that decodes to nothing at all (a 1-2 byte
file cut inside a character) is rejected rather than divided by zero. From the
report by Dshamir; upstream has no PR.

Note: `collect()` caches its verdicts per directory signature in
`runs/corpus_index.json`, so a file fixed in place keeps its old verdict. Delete
that cache if you re-scan a directory whose contents changed under the same
names. This is upstream's documented trade-off, not changed by the patch.

### 0008 — Apple Silicon should not silently train on the CPU

`--device` defaulted to `"cuda" if torch.cuda.is_available() else "cpu"`, so on
an Apple Silicon machine mini-AGI fell back to CPU even though MPS trains fine
when passed explicitly — the user got no signal to pass it. The patch adds
`_auto_device()`: CUDA first (the only backend that enables fused AdamW, bf16
autocast and the VRAM telemetry), MPS the deliberate second choice, CPU the
safety net. CUDA and CPU behaviour is byte-identical to the old inline default.
From PR #3 (MetalstormHS); the ordering test stubs both availability probes so it
runs without CUDA or MPS hardware.

## Upstream-covered (dropped)

| Patch | Upstream issue | Upstream PR | Why dropped |
|-------|----------------|-------------|-------------|
| Reset serving cache before RoPE rollover | [#8](https://github.com/volotat/mini-AGI/issues/8) | [#9](https://github.com/volotat/mini-AGI/pull/9) | Upstream's current `serve.py` rewrites this exact path — it re-reads the newest half-window from position 0 with explicit offsets instead of trimming the cache and rebasing the offset, which is what the patch did. Verified against upstream `main` at sync time; no longer needed. |
| GradSNR: missing grad as zeros | [#19](https://github.com/volotat/mini-AGI/issues/19) | [#20](https://github.com/volotat/mini-AGI/pull/20) | Upstream already zero-fills a parameter that took no part in a step, and additionally resets the meter when the observed set changes shape. Verified against upstream `main` at sync time; no longer needed. |

## Deliberately not carried

| PR | Title | Why not |
|----|-------|---------|
| [#24](https://github.com/volotat/mini-AGI/pull/24) | Interop tool-loop + self-directed learning + governance ledger + health certificate | 24 commits that add features, not correctness fixes: a `<tool>` protocol and registry in `serve.py`, an off-by-default `--selfdir` channel that lets the model adjust LR/growth, a hashed receipt ledger, and a monotonicity "health certificate". Outside this mirror's scope (correctness patches upstream has not merged), it changes the generation loop and training controller rather than fixing a bug, and the upstream maintainer asked the author to justify it and signalled he may reject it. Left to evolve upstream. |

## Applying to a plain upstream checkout

The exported series in [`patches/`](patches) applies to upstream
`60f7ae318cd54f6ed2995335de870107a04d74d4` — the commit this mirror was last
synced to (recorded in [`UPSTREAM.md`](UPSTREAM.md)):

```sh
git clone https://github.com/volotat/mini-AGI
cd mini-AGI
git checkout 60f7ae318cd54f6ed2995335de870107a04d74d4
git am /path/to/mini-AGI/patches/*.patch
```
