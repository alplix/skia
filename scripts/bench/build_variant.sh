#!/bin/bash
# Prepare one benchmark label: a copy of the code and a `run` script that starts skia-app the way the volunteer wrapper
# does (boinc/job.xml, task 2: its <setenv> lines, then `skia_app task.zip -o result.txt`).
# usage: scripts/bench/build_variant.sh <name> [patch ...]
#   env: REF        a git ref to take src/ and boinc/ from (git archive); default: this working tree's tracked files as
#                   they are on disk (a new file must be `git add -N`ed), so an untracked or stale file cannot leak in
#        ENV        extra environment for the run script, "NAME=value NAME=value", set after job.xml's (e.g. PJRT_NPROC=1)
#        JOBXML=0   leave out job.xml's environment (then the app runs with whatever the shell has)
#        SKIA_WORK  (default ~/skia-work): the label goes to $SKIA_WORK/trees/<name>/
#   patches: git-diff files applied to the copy (scripts/bench/patches/ holds the result-changing experiments)
# Nothing is compiled: the label is Python source run by the pinned environment that $SKIA_PYTHON names at run time
# (setup_env.sh on a Mac, linux_ab.sh in a container), so one label runs on every platform the suite reaches.
set -e -o pipefail
[ $# -ge 1 ] || { sed -n 2,13p "$0"; exit 2; }
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
NAME=$1; shift
SKIA_WORK=${SKIA_WORK:-$HOME/skia-work}
T=$SKIA_WORK/trees/$NAME
rm -rf "$T"; mkdir -p "$T"
if [ -n "${REF:-}" ]; then
  git -C "$repo" archive "$REF" src boinc | tar -x -C "$T"
  what="$REF ($(git -C "$repo" rev-parse --short "$REF"))"
else
  ( cd "$repo" && git ls-files -z -- src boinc | while IFS= read -r -d '' f; do
      [ -e "$f" ] || continue                       # deleted in the working tree
      mkdir -p "$T/$(dirname "$f")"; cp -p "$f" "$T/$f"
    done )
  what="working tree on $(git -C "$repo" rev-parse --short HEAD)$(git -C "$repo" diff --quiet HEAD -- src boinc || echo ', with uncommitted changes')"
fi
for p in "$@"; do
  case $p in /*) ;; *) p=$PWD/$p ;; esac
  ( cd "$T" && git apply "$p" ) || { echo "patch $p does not apply to $what"; exit 2; }
done

{
  echo '#!/bin/sh'
  echo "# label $NAME: $what${*:+, patches: $*}"
  echo '# Runs skia-app as boinc/job.xml task 2 does; SKIA_PYTHON names the python of the pinned environment.'
  echo 'here=$(cd "$(dirname "$0")" && pwd)'
  echo 'export PYTHONPATH="$here/src" PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu'
  if [ "${JOBXML:-1}" = 1 ] && [ -f "$T/boinc/job.xml" ]; then
    # the <setenv> lines of the task that runs skia_app (the wrapper sets them for that task only)
    awk '/<task>/{t=""} /<setenv>/{s=$0; sub(/.*<setenv>/,"",s); sub(/<\/setenv>.*/,"",s); t=t s "\n"}
         /<application>/ && /skia_app\/skia_app/{app=1} /<\/task>/{if(app){printf "%s", t; exit}}' "$T/boinc/job.xml" |
      while IFS= read -r kv; do [ -n "$kv" ] && printf "export '%s'\n" "$kv"; done
  fi
  for kv in ${ENV:-}; do printf "export '%s'\n" "$kv"; done
  echo 'exec "${SKIA_PYTHON:?set SKIA_PYTHON to the python of the pinned environment}" -B -m skia.app_cli "$@"'
} > "$T/run"
chmod +x "$T/run"
{
  echo "label $NAME"
  echo "code $what"
  [ $# -gt 0 ] && echo "patches $*"
  echo "environment: $(grep '^export' "$T/run" | sed 1d | sed "s/^export //; s/'//g" | tr '\n' ' ')"
} > "$T/build.info"
echo "$NAME: $T/run ($what${*:+, patches: $*})"
