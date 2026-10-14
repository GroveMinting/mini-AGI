#!/usr/bin/env bash
#
# Rebase the hardening patchset onto current upstream and verify it.
#
# Run on demand when you want upstream's latest. It fetches upstream, moves the
# `upstream` mirror branch, rebases the code-patch branch (`hardening`) onto it,
# re-exports patches/, then replays the mirror scaffolding onto `main`. A
# conflict stops it with the tree mid-rebase; abort with `git rebase --abort`.
#
# Branch model (see UPSTREAM.md):
#   upstream  - pure mirror of volotat/mini-AGI main, fast-forward only
#   hardening - upstream + the code patches (one commit per fix)
#   main      - hardening + one mirror-scaffolding commit (default branch)
#
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

hardening="${HARDENING_BRANCH:-hardening}"
main_branch="${MAIN_BRANCH:-main}"

pick_upstream_remote() {
    if [ -n "${UPSTREAM_REMOTE:-}" ]; then echo "$UPSTREAM_REMOTE"; return; fi
    if git remote | grep -qx upstream; then echo upstream; return; fi
    for r in $(git remote); do
        case "$(git remote get-url "$r")" in
            *volotat/mini-AGI*) echo "$r"; return;;
        esac
    done
    echo ""
}

upstream_remote="$(pick_upstream_remote)"
if [ -z "$upstream_remote" ]; then
    echo "!! no remote points at volotat/mini-AGI." >&2
    echo "   add one:  git remote add upstream https://github.com/volotat/mini-AGI" >&2
    echo "   or set UPSTREAM_REMOTE=<name>." >&2
    exit 1
fi

push_remote="$upstream_remote"
for r in fork origin; do
    if git remote | grep -qx "$r"; then push_remote="$r"; break; fi
done

echo "== upstream: $upstream_remote   hardening: $hardening   main: $main_branch =="
git fetch "$upstream_remote" --prune

echo "== moving the upstream mirror =="
git branch -f upstream "$upstream_remote/main"
sync_tag="upstream-$(date -u +%Y%m%d)"
git tag -f "$sync_tag" upstream >/dev/null
echo "   mirror at $(git rev-parse --short upstream), tagged $sync_tag"

old_hardening="$(git rev-parse "$hardening")"

echo "== rebasing $hardening onto upstream =="
git checkout "$hardening"
if ! git rebase upstream; then
    echo >&2
    echo "!! conflict rebasing $hardening. Resolve, then 'git rebase --continue'," >&2
    echo "   or 'git rebase --abort'. If a patch is now upstream-covered, drop it" >&2
    echo "   and record it in HARDENING.md." >&2
    exit 1
fi

echo "== re-exporting patches/ =="
rm -f patches/*.patch
git format-patch --no-signature -o patches upstream.."$hardening" >/dev/null
ls -1 patches/

echo "== replaying mirror scaffolding onto $main_branch =="
git checkout "$main_branch"
if ! git rebase --onto "$hardening" "$old_hardening" "$main_branch"; then
    echo >&2
    echo "!! conflict re-applying the scaffolding commit onto the new $hardening." >&2
    echo "   Resolve, then 'git rebase --continue'." >&2
    exit 1
fi

echo "== running tests =="
if [ -x "$root/.venv/bin/python" ]; then
    py="$root/.venv/bin/python"
elif [ -x "$HOME/.venvs/minagi-hardening/bin/python" ]; then
    py="$HOME/.venvs/minagi-hardening/bin/python"
else
    py="python3"
fi
"$py" -m unittest discover -s tests

echo
echo "OK: rebased onto upstream main and the tests pass."
echo "Review 'git log --oneline upstream..$hardening', then push with:"
echo "    git push $push_remote $hardening"
echo "    git push --force-with-lease $push_remote $main_branch   # the scaffolding commit is rebased each sync"
echo "    git push $push_remote $sync_tag"
