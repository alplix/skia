#!/usr/bin/env python3
"""Run one command and print its wall time, CPU time and peak memory as one tab-separated line.

usage: measure.py <command> [args ...]
Prints to stderr (so the command's own stdout stays clean): wall s, user s, sys s, max RSS MB, exit code.
The child's stdin is closed; its stdout and stderr are inherited. Works the same on macOS and Linux, unlike
/usr/bin/time, whose output format and memory unit differ (and which slim containers do not have).
"""
import os
import resource
import subprocess
import sys
import time


def main(argv):
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    start = time.perf_counter()
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
    _, status, usage = os.wait4(child.pid, 0)
    wall = time.perf_counter() - start
    code = os.waitstatus_to_exitcode(status)
    # ru_maxrss is in bytes on macOS and in KiB on Linux
    rss_mb = usage.ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
    print("%.3f\t%.3f\t%.3f\t%.0f\t%d" % (wall, usage.ru_utime, usage.ru_stime, rss_mb, code), file=sys.stderr)
    return 0


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # a crashing candidate must not leave core files behind
    sys.exit(main(sys.argv[1:]))
