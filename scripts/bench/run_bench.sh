#!/bin/bash
# Interleaved speed benchmark of skia-app labels (build_variant.sh) on one profile's task bundle.
# usage: scripts/bench/run_bench.sh <profile> <rounds> <label>=<run script> [<label>=<run script> ...]
#   profile: a file in scripts/bench/profiles/ without .prof, or a path to a .prof file (make_bench_data.py says what
#        the keys mean). Its bundle is made once and cached (make_bench_data.py), then the labels run A B C ... A B C ...
#        (rounds) in fresh run directories, so that load drift hits every label alike. A fresh directory matters: the
#        app resumes from a result.txt.checkpoint.json it finds.
#   env: SKIA_WORK (default ~/skia-work), SKIA_PYTHON (default $SKIA_WORK/venv/bin/python, see setup_env.sh),
#        BENCH_OUT (default $SKIA_WORK/bench-runs), RUN_PREFIX (e.g. "taskpolicy -b" or "nice -n 19"),
#        COPIES=N   run N copies of each label at once, each on its own copy of the bundle, as the BOINC client runs one
#                   task per core: the row's wall time is the whole batch's, its CPU time the sum, its RSS the largest
#                   copy's, and its hash the copies' common hash ("copies differ" otherwise). It measures throughput of
#                   a loaded machine instead of the time of one task.
# Output: one line per run (label, round, wall s, user s, sys s, max RSS MB, exit code, hash of result.txt, grid rows
# written), then the summary of summarize_bench.py. result.txt has no timings (save_scan), so the whole file is hashed.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
[ $# -ge 3 ] || { sed -n 2,16p "$0"; exit 2; }
PROFILE=$1; ROUNDS=$2; shift 2
PROF=$PROFILE; [ -f "$PROF" ] || PROF=$here/profiles/$PROFILE.prof
[ -f "$PROF" ] || { echo "no profile $PROFILE"; exit 2; }
for lb in "$@"; do
  case $lb in *=*) ;; *) echo "arguments after the rounds are label=run-script, not $lb"; exit 2 ;; esac
  [ -x "${lb#*=}" ] || { echo "not an executable: ${lb#*=}"; exit 2; }
done
export SKIA_WORK=${SKIA_WORK:-$HOME/skia-work}
export SKIA_PYTHON=${SKIA_PYTHON:-$SKIA_WORK/venv/bin/python}
[ -x "$SKIA_PYTHON" ] || { echo "no python at $SKIA_PYTHON: run scripts/bench/setup_env.sh or set SKIA_PYTHON"; exit 2; }
COPIES=${COPIES:-1}

md5_of() { if command -v md5sum > /dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }
hash12() { if command -v shasum > /dev/null; then shasum | cut -c1-12; else sha1sum | cut -c1-12; fi; }
cpu_name() { # macOS, x86 Linux, then arm64 Linux (whose /proc/cpuinfo has no model name)
  local n; n=$(sysctl -n machdep.cpu.brand_string 2> /dev/null || awk -F': ' '/model name/ { print $2; exit }' /proc/cpuinfo 2> /dev/null)
  echo "${n:-$(uname -m)}"
}
load() { if [ -r /proc/loadavg ]; then echo "load averages: $(cut -d' ' -f1-3 /proc/loadavg)"; else uptime | sed 's/.*load/load/'; fi; }
now() { "$SKIA_PYTHON" -c 'import time; print("%.3f" % time.time())'; }

TASK=$(PYTHONPATH="$repo/src" PYTHONDONTWRITEBYTECODE=1 "$SKIA_PYTHON" -B "$here/make_bench_data.py" "$PROF" "$SKIA_WORK/bench-data") || exit 2
[ -s "$TASK" ] || { echo "no task bundle for $PROF"; exit 2; }
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${BENCH_OUT:-$SKIA_WORK/bench-runs}/$STAMP-$(basename "$PROF" .prof)
mkdir -p "$OUT"
RESULTS=$OUT/results.tsv
{
  echo "# $(date '+%Y-%m-%d %H:%M:%S') profile $PROF rounds $ROUNDS copies $COPIES cpu $(cpu_name) ($(getconf _NPROCESSORS_ONLN 2> /dev/null) CPUs) os $(uname -sr) $(uname -m)"
  echo "# task: $TASK md5 $(md5_of "$TASK"); $(cat "$(dirname "$TASK")/about.txt" 2> /dev/null)"
  echo "# python: $SKIA_PYTHON $("$SKIA_PYTHON" -c 'import sys, importlib.metadata as m; print(sys.version.split()[0], " ".join(p + " " + m.version(p) for p in ("jax", "jaxlib", "numpy", "scipy", "herculens", "lenstronomy")))')"
  echo "# load: $(load)"
  for lb in "$@"; do
    echo "# ${lb%%=*} = ${lb#*=}"
    [ -f "$(dirname "${lb#*=}")/build.info" ] && sed 's/^/#     /' "$(dirname "${lb#*=}")/build.info"
  done
} | tee "$RESULTS"

one_copy() { # run script, directory: one app run, its numbers in measure.txt
  local run=$1 dir=$2
  mkdir -p "$dir"; cp "$TASK" "$dir/task.zip"
  ( cd "$dir" && $SKIA_PYTHON -B "$here/measure.py" ${RUN_PREFIX:-} "$run" task.zip -o result.txt > stdout.txt 2> stderr.txt )
  tail -n 1 "$dir/stderr.txt" > "$dir/measure.txt"
  rm -f "$dir/task.zip" "$dir/result.txt.checkpoint.json"
}
result_of() { # directory: "hash rows" of its result.txt
  if [ -f "$1/result.txt" ]; then
    echo "$(hash12 < "$1/result.txt") $(grep -v '^#' "$1/result.txt" | sed 1d | grep -c .)"
  else
    echo "none -"
  fi
}
run_one() { # label run-script round
  local label=$1 run=$2 round=$3 dir=$OUT/$1-$3 c
  if [ "$COPIES" = 1 ]; then
    one_copy "$run" "$dir"
    IFS=$'\t' read -r wall user sys rss code < "$dir/measure.txt"
    read -r h rows <<< "$(result_of "$dir")"
  else
    local t0 t1; t0=$(now)
    for c in $(seq 1 "$COPIES"); do one_copy "$run" "$dir/copy$c" & done
    wait
    t1=$(now)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')
    read -r user sys rss code <<< "$(cat "$dir"/copy*/measure.txt | awk -F'\t' '{u+=$2; s+=$3; if ($4>r) r=$4; if ($5!=0) c=$5} END {printf "%.3f %.3f %.0f %d", u, s, r, c}')"
    h=""; for c in $(seq 1 "$COPIES"); do
      read -r hc rows <<< "$(result_of "$dir/copy$c")"
      if [ -z "$h" ]; then h=$hc; elif [ "$h" != "$hc" ]; then h="copies-differ"; fi
    done
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$round" "$wall" "$user" "$sys" "$rss" "$code" "$h" "$rows" | tee -a "$RESULTS"
}
for r in $(seq 1 "$ROUNDS"); do
  for lb in "$@"; do run_one "${lb%%=*}" "${lb#*=}" "$r"; done
done
echo "# load at the end: $(load)" | tee -a "$RESULTS"
"$SKIA_PYTHON" -B "$here/summarize_bench.py" "$RESULTS" | tee "$OUT/summary.txt"
