# Contributing and releasing

`main` holds released code, `develop` holds integration, and everything else is
a short-lived topic branch. This page is the procedure those three imply: the
daily loop, the release steps, and the rules that are easy to get wrong.

## Branch model

| ref | role | rule |
|---|---|---|
| `main` | released code | receives merge commits from `develop` (or a hotfix); one tag per release |
| `develop` | integration | where topic branches land; tests and docs must pass here |
| `feature/*`, `fix/*` | one unit of work | short-lived, deleted after merging |
| `vX.Y.Z` | annotated tag on `main` | the release identity, and what `git describe` reports |

`AmrCore_refactor` predates this model: it was the only branch carrying real
work while `main` still held three commits. It was merged into `develop` for
`v0.1.0` and then deleted -- the tag records where that release came from, so a
live branch adds nothing. Its commits stay in `develop`'s history.

The invariant that makes the model work: **at a release, `main` and `develop`
are tree-identical.** `develop` then moves ahead with new work while `main`
stays at the last tag.

## Working on a change

```bash
git switch develop
git pull --ff-only
git switch -c feat/terrain-resampling

# ... commits ...

git switch develop
git merge --no-ff feat/terrain-resampling -F /tmp/msg.txt
git push origin develop
```

`-F /tmp/msg.txt` rather than `-F -`: merge accepts a message file but does not
read `-` as stdin, unlike `git commit -F -`. `-m 'subject' -m 'body'` also
works.

### If your working tree is dirty

A branch can only be checked out in one worktree at a time, and switching with
uncommitted changes either drags them along or refuses. Use a worktree instead
of stashing:

```bash
git worktree add /tmp/work develop          # second checkout, same repository
git -C /tmp/work add docs/contributing.md
git -C /tmp/work commit -m "docs: explain the branch model"
git worktree remove --force /tmp/work
```

The checkout in `/tmp/work` commits against `develop` without touching the tree
you are working in. Remove it when you are done — a stale worktree blocks future
checkouts of the same branch.

### Commit messages

`.gitmessage` is the convention: `<type>: <summary under 70 chars>`, where type
is one of `feat fix perf refactor docs build ci test chore`; the body explains
what changed and why, wrapped at 72 columns; footers carry issue references.
Enable it once per clone:

```bash
git config commit.template .gitmessage
```

## Releasing

```bash
# 1. Stamp the version on develop, not on main
#    (CMakeLists.txt project(... VERSION ...) and the CHANGELOG)
git switch develop
git commit -m "build: stamp the 0.2.0 release"
git push origin develop

# 2. Merge into main with a real merge commit, then tag it
git switch main
git pull --ff-only
git merge --no-ff develop -F /tmp/msg.txt
git tag -a v0.2.0 -m "HydroEXA v0.2.0"

# 3. Publish both branches and the tag
git push origin main develop
git push origin v0.2.0
```

`git push` sends commits, never tags. `git push --follow-tags origin main
develop` covers both in one command, for annotated tags reachable from what it
just pushed.

### Verify before you tag

```bash
git rev-list --left-right --count origin/main...main   # "0  <ahead>": nothing to merge back
git log --oneline --graph -8                           # does the history read as intended?
git status --porcelain                                 # your WIP only, nothing stray
```

and, on the release candidate, the two gates: `pytest PyHydroEXA/tests` and
`make -C docs html` (blocking — `-W` for Sphinx, `WARN_AS_ERROR` for Doxygen).

After the merge, `git diff --stat main develop` is empty. If it is not, the
release is not the integration branch, and something was committed on the wrong
side of it.

## Hotfixes

A fix that cannot wait for the next release starts from `main` and has to be
merged back:

```bash
git switch -c fix/cfl-crash main
# fix, commit
git switch main && git merge --no-ff fix/cfl-crash
git tag -a v0.1.1 -m "HydroEXA v0.1.1"

git switch develop && git merge --no-ff main    # do not skip this
```

Without the merge back, `develop` either reverts the fix at the next release or
conflicts on it. This is the only direction in which code flows from `main` into
`develop`.

## Rules that are easy to get wrong

**Merge with `--no-ff`.** When a topic branch is a strict descendant of
`develop`, a plain `git merge` fast-forwards: the pointer slides and the graph
records nothing. `--no-ff` forces a merge commit, which is what you tag and what
shows where unreleased work became released. `AmrCore_refactor` was 65 commits
ahead and 0 behind, so a fast-forward would have erased the release boundary
entirely.

**Stamp the version on `develop` first.** Bumping the version on `main` after
the merge makes the two trees diverge immediately, and the next release merge
conflicts in the changelog.

**Tags are immutable once published.** Moving a tag that anyone has fetched
means their `v0.1.0` is not yours; cut a patch release instead. Before it is
pushed, moving it is harmless.

**Branch names are configuration.** `.github/workflows/docs.yml` names the
branches it builds and deploys, and GitHub Pages has to be set to publish from
GitHub Actions for that workflow to be the publisher. Renaming a long-lived
branch is a code change plus a GitHub setting, which is why this model keeps
`main` instead of introducing `master`.
