#!/bin/bash
# Make the pinned Python environment the benchmark runs in, outside the repository: $SKIA_WORK/venv (default
# ~/skia-work/venv), the README's development setup (python3.14, pip 25.3, requirements.lock) without the editable
# install, because each label brings its own src/ (build_variant.sh, PYTHONPATH). Also installs skia itself from a
# clean copy of this checkout, only for the console scripts the test suite calls (skia-fit); a label's PYTHONPATH still
# wins over that copy.
# usage: scripts/bench/setup_env.sh            (does nothing when the environment exists; delete it to make it again)
#   env: PYTHON314 (default python3.14), SKIA_WORK
set -e -o pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
SKIA_WORK=${SKIA_WORK:-$HOME/skia-work}
V=$SKIA_WORK/venv
if [ -x "$V/bin/python" ]; then echo "$V exists"; exit 0; fi
mkdir -p "$SKIA_WORK"
"${PYTHON314:-python3.14}" -m venv "$V"
"$V/bin/python" -m pip install -q pip==25.3
"$V/bin/python" -m pip install -q -r "$repo/requirements.lock"
rm -rf "$SKIA_WORK/pkg-src"; mkdir -p "$SKIA_WORK/pkg-src"
git -C "$repo" archive HEAD | tar -x -C "$SKIA_WORK/pkg-src"
"$V/bin/python" -m pip install -q --no-deps --no-build-isolation "$SKIA_WORK/pkg-src"
"$V/bin/python" -c 'import sys, jax; print("ready:", sys.version.split()[0], "jax", jax.__version__)'
