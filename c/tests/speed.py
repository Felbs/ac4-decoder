# SPDX-License-Identifier: Apache-2.0
"""Real-time factor of c/ac4dec-c against the Python `ac4dec`, one thread.

    nice -n 10 python3 c/tests/speed.py CAPTURE.m4s [...]

Per capture, best of RUNS (default 2) runs each, every process on one
thread (OMP/OpenBLAS/MKL threads pinned to 1):
  * ac4dec (Python, ac4.cli, C kernels as built): the full decoder,
    A-SPX rendered;
  * the same with A-SPX rendering disabled (the work the C port does);
  * ac4dec-c.
Two figures each: the decoder's own timer (decode time, what the README's
table reports) and the wall time of the whole process (interpreter start,
imports and file reading included). Outputs go to AC4_CPORT_OUT.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
C_CLI = os.path.join(REPO, "c", "ac4dec-c")
OUT = os.path.expanduser(os.environ.get("AC4_CPORT_OUT", "~/.cache/ac4-cport"))
RUNS = int(os.environ.get("RUNS", "2"))
PY = os.environ.get("AC4_PYTHON", sys.executable)

ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
           PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))

CORE_ONLY = (
    "import sys\n"
    "from ac4 import aspx_render, cli\n"
    "def _off(*a, **k):\n"
    "    raise RuntimeError('A-SPX rendering disabled')\n"
    "aspx_render.apply_hf_pair = _off\n"
    "sys.exit(cli.main(sys.argv[1:]))\n")

PAT = re.compile(r"([0-9.]+) s -> .*\[([0-9.]+) s, ([0-9.]+)x real time")


def run(cmd):
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    m = PAT.search(r.stdout)
    if r.returncode != 0 or not m:
        raise RuntimeError(f"{cmd[0]} failed: {r.stdout[-300:]} "
                           f"{r.stderr[-300:]}")
    secs, dt = float(m.group(1)), float(m.group(2))
    return secs, dt, wall


def best(cmd):
    res = [run(cmd) for _ in range(RUNS)]
    secs = res[0][0]
    return secs, min(r[1] for r in res), min(r[2] for r in res)


def main(files):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for p in files:
        tag = os.path.splitext(os.path.basename(p))[0]
        o = os.path.join(OUT, "speed_" + tag)
        py = best([PY, "-m", "ac4", p, o + ".py.wav"])
        core = best([PY, "-c", CORE_ONLY, p, o + ".pycore.wav"])
        c = best([C_CLI, p, o + ".c.wav"])
        rows.append((p, py, core, c))
        secs = c[0]
        print(f"\n{os.path.basename(os.path.dirname(p))}/"
              f"{os.path.basename(p)}: {secs:.1f} s of audio")
        print(f"  {'decoder':34s} {'decode RTF':>11s} {'wall RTF':>9s}")
        for name, r in (("ac4dec (Python, A-SPX rendered)", py),
                        ("ac4dec (Python, core only)", core),
                        ("ac4dec-c (core only)", c)):
            print(f"  {name:34s} {secs / r[1]:10.1f}x {secs / r[2]:8.1f}x")
        print(f"  C vs Python core only: {core[1] / c[1]:.0f}x faster "
              f"(decode), {core[2] / c[2]:.0f}x (wall)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
