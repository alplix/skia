#!/bin/bash
# The A/B of mac_ab.sh on Linux in a container: the labels are prepared on the host (build_variant.sh needs git), then
# run in python:3.14 on Debian with the pinned requirements.lock installed in a cached environment. Linux x86_64 is
# the platform the volunteer app ships for (README); on an Apple Silicon Mac it runs under emulation, so use it there
# for "builds, runs, same output as base", not for speed. Linux arm64 runs at native speed.
# usage: scripts/bench/linux_ab.sh [rounds, default 3] [profiles, default "sim q1-mid"]
#   env: ARCH        arm64 or amd64 (default: the host's), passed to docker run --platform linux/$ARCH
#        IMAGE       default python:3.14-slim-trixie (its digest is printed; pin it with IMAGE=python@sha256:...)
#        BASE_REF, NEW_ENV, NEW_PATCHES, COPIES   as for mac_ab.sh
#        SKIA_WORK   (default ~/skia-work), mounted at /work: the environment goes to linux-venv-$ARCH (made on the first
#                    run, a few minutes), results to bench-runs-linux-$ARCH/; the repository is mounted read-only at /repo
# Needs Docker. The cached bundles of bench-data/ are shared with the Mac runs, so both scan the same bytes.
set -e -o pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
ROUNDS=${1:-3}
PROFILES=${2:-sim q1-mid}
case $(uname -m) in arm64|aarch64) host=arm64 ;; *) host=amd64 ;; esac
ARCH=${ARCH:-$host}
IMAGE=${IMAGE:-python:3.14-slim-trixie}
export SKIA_WORK=${SKIA_WORK:-$HOME/skia-work}
mkdir -p "$SKIA_WORK"
if [ -z "${BASE_REF:-}" ]; then
  if git -C "$repo" rev-parse -q --verify upstream/main > /dev/null; then BASE_REF=upstream/main; else BASE_REF=origin/main; fi
fi
REF=$BASE_REF bash "$here/build_variant.sh" base
ENV="${NEW_ENV:-}" bash "$here/build_variant.sh" new ${NEW_PATCHES:-}
[ "$ARCH" = "$host" ] || echo "note: linux/$ARCH runs under emulation here: compare outputs, not times"
# Bundles are made on the host when its environment exists, so a container never needs the Q1 cutouts
if [ -x "$SKIA_WORK/venv/bin/python" ]; then
  for prof in $PROFILES; do
    f=$here/profiles/$prof.prof; [ -f "$prof" ] && f=$prof
    PYTHONPATH="$repo/src" "$SKIA_WORK/venv/bin/python" -B "$here/make_bench_data.py" "$f" "$SKIA_WORK/bench-data" > /dev/null ||
      echo "note: no bundle for $prof on the host"
  done
fi
docker pull -q --platform "linux/$ARCH" "$IMAGE" > /dev/null
echo "image $IMAGE: $(docker image inspect "$IMAGE" --format '{{index .RepoDigests 0}}' 2> /dev/null)"
docker run --rm --platform "linux/$ARCH" -v "$SKIA_WORK:/work" -v "$repo:/repo:ro" \
  -e SKIA_WORK=/work -e SKIA_PYTHON="/work/linux-venv-$ARCH/bin/python" -e BENCH_OUT="/work/bench-runs-linux-$ARCH" \
  -e COPIES="${COPIES:-1}" -e PIP_DISABLE_PIP_VERSION_CHECK=1 -e ROUNDS="$ROUNDS" -e PROFILES="$PROFILES" -e ARCH="$ARCH" \
  "$IMAGE" bash -c '
    set -e
    V=/work/linux-venv-$ARCH
    if [ ! -x $V/bin/python ]; then
      python -m venv $V && $V/bin/python -m pip install -q pip==25.3 && $V/bin/python -m pip install -q -r /repo/requirements.lock
    fi
    for prof in $PROFILES; do
      echo "=== linux/$ARCH profile $prof, $ROUNDS rounds ==="
      bash /repo/scripts/bench/run_bench.sh "$prof" "$ROUNDS" base=/work/trees/base/run new=/work/trees/new/run
    done'
