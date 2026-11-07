#!/bin/bash
# One-command A/B on a Mac: BASE_REF against this working tree, both run as the volunteer wrapper runs skia-app
# (boinc/job.xml task 2: its environment, then `skia_app task.zip -o result.txt`), on the profiles in
# scripts/bench/profiles, interleaved.
# usage: scripts/bench/mac_ab.sh [rounds per profile, default 3] [profiles, default "sim q1-fast q1-mid q1-slow"]
#   env: BASE_REF   the baseline (default upstream/main if that remote branch exists, else origin/main)
#        NEW_ENV    extra environment for the working tree's label, "NAME=value ..." (build_variant.sh ENV)
#        NEW_PATCHES  space-separated git-diff files applied to the working tree's copy (scripts/bench/patches/)
#        COPIES=N   run N copies of each label at once (run_bench.sh): the throughput of a machine whose every core runs
#                   a task, which is what a volunteer's machine does; COPIES=$(sysctl -n hw.perflevel0.physicalcpu)
#                   fills the performance cores
#        SKIA_WORK  (default ~/skia-work): the Python environment (setup_env.sh makes it on the first run), the labels
#                   and the results (bench-runs/<date>-<profile>/)
# A q1-* or task profile needs its public Euclid Q1 cutout (make_bench_data.py says where it is looked up); profiles
# whose cutout is missing are skipped with a note. With no change under src/ or boinc/ and no NEW_*, `new` equals
# `base`: run that once, because its spread is the noise a real change has to beat.
#
# Labels:
#   base  BASE_REF's src/ and boinc/job.xml
#   new   the working tree (tracked files, as on disk), with NEW_ENV and NEW_PATCHES
set -e -o pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
ROUNDS=${1:-3}
PROFILES=${2:-sim q1-fast q1-mid q1-slow}
export SKIA_WORK=${SKIA_WORK:-$HOME/skia-work}
export SKIA_PYTHON=${SKIA_PYTHON:-$SKIA_WORK/venv/bin/python}
[ -x "$SKIA_PYTHON" ] || bash "$here/setup_env.sh"
if [ -z "${BASE_REF:-}" ]; then
  if git -C "$repo" rev-parse -q --verify upstream/main > /dev/null; then BASE_REF=upstream/main; else BASE_REF=origin/main; fi
fi
git -C "$repo" rev-parse -q --verify "$BASE_REF^{commit}" > /dev/null || { echo "no git ref $BASE_REF: set BASE_REF"; exit 2; }
if git -C "$repo" diff --quiet "$BASE_REF" -- src boinc && [ -z "${NEW_ENV:-}${NEW_PATCHES:-}" ]; then
  echo "note: src/ and boinc/ equal $BASE_REF's and no NEW_* is set: this run measures the noise between identical labels"
fi
REF=$BASE_REF bash "$here/build_variant.sh" base
# NEW_PATCHES is deliberately unquoted: a list of file names
ENV="${NEW_ENV:-}" bash "$here/build_variant.sh" new ${NEW_PATCHES:-}
T=$SKIA_WORK/trees
have_input() { # profile file: true when it is simulated, its cutout is found, or its bundle was made before
  local name
  name=$(awk '$1 == "cutout" && $2 == "q1" { print $3 }' "$1")
  [ -z "$name" ] && return 0
  for d in "${SKIA_DATA:-}" "$repo/data/q1_cutouts" "$SKIA_WORK/data/q1"; do [ -n "$d" ] && [ -f "$d/$name" ] && return 0; done
  ls "$SKIA_WORK/bench-data/$(basename "$1" .prof)"-*/task.zip > /dev/null 2>&1
}
for prof in $PROFILES; do
  f=$here/profiles/$prof.prof; [ -f "$prof" ] && f=$prof
  if ! have_input "$f"; then
    echo "=== profile $prof skipped: its Q1 cutout was not found (docs/fetch_q1_cutouts.py fetches it)"
    continue
  fi
  echo "=== profile $prof, $ROUNDS rounds ==="
  bash "$here/run_bench.sh" "$prof" "$ROUNDS" base="$T/base/run" new="$T/new/run"
done
