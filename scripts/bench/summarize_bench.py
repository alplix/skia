#!/usr/bin/env python3
"""Summary of a run_bench.sh results.tsv: per label the median, minimum and maximum wall time, CPU time (user + sys)
and memory, the number of grid rows written, the speed relative to the first label, the verdict of the acceptance rule
below, and whether the label's result.txt is the same as the first label's.

Acceptance rule (as in docs/BENCHMARK.md): a candidate is faster when its median is at least 5 % below the first
label's and its slowest run is faster than the first label's fastest, or when its median is at least 10 % below. With
fewer than 3 runs of either label the verdict is "too few runs". The rule is applied to wall time; `cpu s` shows
whether a label got faster by using more cores (skia-app is a one-CPU BOINC task, so CPU time is what volunteers pay).

usage: summarize_bench.py <results.tsv>
"""
import statistics
import sys


def main(path):
    runs, order = {}, []
    for line in open(path, encoding='utf-8'):
        if line.startswith('#') or not line.strip():
            continue
        f = line.rstrip('\n').split('\t')
        if f[0] not in runs:
            runs[f[0]] = []
            order.append(f[0])
        num = lambda s: float(s) if s not in ('', '-') else float('nan')
        runs[f[0]].append(dict(wall=num(f[2]), user=num(f[3]), sys=num(f[4]), rss=num(f[5]), code=f[6], hash=f[7],
                               rows=f[8]))
    if not order:
        print('no runs')
        return
    ref = order[0]
    ref_hashes = {r['hash'] for r in runs[ref]}
    rmed = statistics.median(r['wall'] for r in runs[ref])
    rmin = min(r['wall'] for r in runs[ref])
    rcpu = statistics.median(r['user'] + r['sys'] for r in runs[ref])
    print('%-12s %4s %9s %8s %8s %8s %7s %5s  %7s %7s  %s' % ('label', 'runs', 'median s', 'min s', 'max s', 'cpu s',
                                                            'RSS MB', 'rows', 'vs 1st', 'cpu vs', 'verdict; output'))
    for lb in order:
        rs = runs[lb]
        walls = [r['wall'] for r in rs]
        med, lo, hi = statistics.median(walls), min(walls), max(walls)
        cpu = statistics.median(r['user'] + r['sys'] for r in rs)
        rss = statistics.median(r['rss'] for r in rs)
        rows = sorted({r['rows'] for r in rs})
        bad = sorted({r['code'] for r in rs if r['code'] != '0'})
        hashes = {r['hash'] for r in rs}
        if bad:
            out = 'FAILED (exit %s)' % ','.join(bad)
        elif 'none' in hashes:
            out = 'NO result.txt'
        elif 'copies-differ' in hashes:
            out = 'copies of one run differ'
        elif len(hashes) > 1:
            out = 'results differ between its own runs'
        elif hashes == ref_hashes:
            out = 'same output'
        else:
            out = 'OTHER output'
        if lb == ref:
            verdict = 'baseline'
        else:
            gain = 1 - med / rmed
            if min(len(rs), len(runs[ref])) < 3:
                verdict = 'too few runs (%+.1f %%)' % (100 * gain)  # two runs say nothing about the spread
            elif (gain >= 0.05 and hi < rmin) or gain >= 0.10:
                verdict = 'ACCEPT (%+.1f %%)' % (100 * gain)
            elif gain <= -0.05:
                verdict = 'slower (%+.1f %%)' % (100 * gain)
            else:
                verdict = 'no clear gain (%+.1f %%)' % (100 * gain)
        print('%-12s %4d %9.2f %8.2f %8.2f %8.1f %7.0f %5s  %6.3fx %6.3fx  %s; %s' % (
            lb, len(rs), med, lo, hi, cpu, rss, '/'.join(rows), rmed / med, rcpu / cpu, verdict, out))
    if all(r['rows'] in ('0', '-') for r in runs[ref]):
        print('warning: the first label wrote no grid rows, so "same output" proves little')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
