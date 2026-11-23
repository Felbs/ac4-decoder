# SPDX-License-Identifier: Apache-2.0
"""The C spectral noise fill against the Python's, sample for sample.

    python3 c/tests/check_noise_fill.py      (after make -C c test)

tests/unit_test --dump-snf prints 400 random channels (long and short
framings, grouped and not, both halves of different transform lengths;
coded bands and noise bands with every dpcm_snf delta, the -17 escape
included; a random sequence_counter) and their packed spectrum after the
noise fill (TS 103 190-1 5.1.4). The Python (audio.packed_spectrum with
audio.SnfRandom) must give the same numbers exactly. No capture sets
b_snf_data_exists, so this and the unit tests are what check the tool.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from ac4 import audio as R               # noqa: E402
from ac4 import channels as CH           # noqa: E402
from ac4.toc_v1 import Bits              # noqa: E402

NONE = -32768


def main():
    exe = os.path.join(REPO, "c", "tests", "unit_test")
    out = subprocess.run([exe, "--dump-snf"], capture_output=True,
                         text=True, check=True).stdout.splitlines()
    i, n_cases, n_noise, bad = 0, 0, 0, 0
    while i < len(out):
        _c, idx, bits, seq, n_lines = out[i].split()
        i += 1
        lines = np.zeros(int(n_lines), dtype=np.int32)
        for kv in out[i].split()[1:]:
            k, v = kv.split(":")
            lines[int(k)] = int(v)
        i += 1
        bb = bits + "0" * (-len(bits) % 8) + "0" * 64
        fr = CH.Framing(Bits(bytes(int(bb[j:j + 8], 2)
                                   for j in range(0, len(bb), 8))))
        sfs, snf = [], []
        for _g in range(fr.num_groups):
            row = out[i].split()[2:]
            i += 1
            sfs.append([None if int(p.split(",")[0]) == NONE
                        else int(p.split(",")[0]) for p in row])
            snf.append([None if int(p.split(",")[1]) == NONE
                        else int(p.split(",")[1]) for p in row])
        ch = dict(lines=lines, sfs=sfs[0] if sfs else [], sfs_all=sfs,
                  snf=snf, framing=fr)
        c_line = out[i]
        i += 1
        n_cases += 1
        n_noise += sum(d is not None and d != -17 for g in snf for d in g)
        try:
            py = R.packed_spectrum(ch, fr, R.SnfRandom(int(seq)))
            py_raised = not (len(py) == 0 or np.abs(py).max() <= 1e9)
        except Exception as e:                             # noqa: BLE001
            print(f"case {idx}: Python raised {e!r}")
            bad += 1
            continue
        if c_line == "raised":
            if not py_raised:
                print(f"case {idx}: only the C raised")
                bad += 1
            continue
        cv = np.array([float(x) for x in c_line.split()[1:]])
        if py_raised or cv.shape != py.shape or not np.array_equal(cv, py):
            d = (np.abs(cv - py).max() if cv.shape == py.shape
                 else f"shapes {cv.shape} {py.shape}")
            print(f"case {idx} ({bits}, seq {seq}): differ, max {d}")
            bad += 1
    print(f"{n_cases} channels, {n_noise} noise bands: "
          f"{'identical' if not bad else f'{bad} DIFFER'}")
    return 1 if bad or n_cases == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
