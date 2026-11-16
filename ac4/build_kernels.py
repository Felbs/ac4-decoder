# SPDX-License-Identifier: Apache-2.0
"""Build the two optional C kernels next to the package's Python files.

    python -m ac4.build_kernels

Both kernels are OPTIONAL. Without them the decoder runs the pure NumPy /
Python paths, which remain the reference; the kernels only make it faster.
They are pure C with no Python.h, loaded through ctypes, so one build serves
every Python on the machine. Never copy a built binary between machines.

Flags are IEEE-pinned: -O3 -ffp-contract=off, never -ffast-math, so no FMA
contraction can change a rounding relative to the NumPy path.

  qmf_kernel.c       64-band complex QMF analysis/synthesis (A-SPX)
  spectral_kernel.c  ASF spectral Huffman parse

Disable at run time with AC4_QMF_KERNEL=0 / AC4_SPECTRAL_KERNEL=0.
tests/test_kernels.py compares each kernel against the Python reference.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = ("qmf_kernel", "spectral_kernel")
FLAGS = ["-O3", "-ffp-contract=off", "-shared"]


def lib_path(name):
    return os.path.join(HERE, name + (".dll" if os.name == "nt" else ".so"))


def build(name):
    src = os.path.join(HERE, name + ".c")
    out = lib_path(name)
    extra = ["-static-libgcc"] if os.name == "nt" else ["-fPIC"]
    for cc in ("cc", "gcc", "clang"):
        if not shutil.which(cc):
            continue
        cmd = [cc] + FLAGS + extra + ["-o", out, src]
        print("  $", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out):
            return cc
        print(f"  {cc} failed (rc {r.returncode}): "
              f"{(r.stderr or r.stdout).strip()[:300]}")
    return None


def main(argv=None):
    rc = 0
    for name in KERNELS:
        tool = build(name)
        if tool:
            print(f"built {lib_path(name)} with {tool}")
        else:
            print(f"FAIL: no C compiler built {name}; the Python path stays "
                  f"in use (correct, just slower)")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
