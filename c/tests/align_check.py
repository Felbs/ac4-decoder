# SPDX-License-Identifier: Apache-2.0
"""Time alignment of the 5.1 channels, measured on captures, both decoders.

    python3 c/tests/align_check.py [CAPTURE.m4s ...]

Captures: arguments, or the 5_X captures under AC4_TEST_DIR whose L/R
pair goes through the A-SPX path (ASPX_ACPL_3 streams, whose other
channels are silent, are skipped). Each is decoded by the Python
(Decoder, one pass) and the C (libac4dec.so through ctypes), both at the
internal rate; the lag of L, R, Ls, Rs and LFE against the centre is
measured by cross-correlation (where the channels share programme: the
peak normalised correlation is printed). Every channel goes through the
QMF bank or an equal delay (TS 103 190-1 6.2.8 / 6.2.14), so the lag of
L and R against C, where they correlate (> 0.3), must be within 2 samples
(the programme itself puts L about 1 sample before C on one stream; a
measurement, not a decoder delay). Before the
fix L/R (and Ls/Rs) came out 577 samples after C and LFE.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import robust_compare as RC                           # noqa: E402
from ac4 import mp4                                   # noqa: E402
from ac4.decoder import Decoder, probe_element        # noqa: E402

NAMES = ("L", "R", "C", "LFE", "Ls", "Rs")


def lag(a, b, maxlag=1200):
    """-> (lag of b against a in samples, peak normalised correlation)."""
    n = len(a)
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, 2 * n)) * np.fft.rfft(b, 2 * n))
    cc = np.concatenate([cc[-maxlag:], cc[:maxlag + 1]])
    k = int(np.argmax(np.abs(cc)))
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return k - maxlag, (float(np.abs(cc[k]) / den) if den > 0 else 0.0)


def check(path):
    frames = mp4.read_frames(path)
    if probe_element(frames) != "5_X":
        return None
    d = Decoder("5_X")
    yp = d.decode_pass(frames)
    if d.acpl3:
        return None
    yc, _c, _w = RC.c_decode(frames, "5_X", lookahead=0)
    ok = True
    for nm, y in (("py", yp), ("c", yc)):
        row = []
        for k in (0, 1, 3, 4, 5):
            lg, r = lag(y[:, 2], y[:, k])
            row.append(f"{NAMES[k]} {lg:+d} ({r:.2f})")
            if k in (0, 1) and (r <= 0.3 or abs(lg) > 2):
                ok = False
        print(f"{os.path.basename(path)} {nm}: lag against C: "
              + ", ".join(row))
    return ok


def main(argv):
    files = argv
    if not files:
        dd = os.environ.get("AC4_TEST_DIR")
        if not dd:
            sys.exit("pass capture files or set AC4_TEST_DIR")
        files = sorted(glob.glob(os.path.join(dd, "**", "*.m4s"),
                                 recursive=True))
    res = [r for r in (check(p) for p in files) if r is not None]
    if not res:
        sys.exit("no 5.1 capture with the A-SPX path")
    print("PASS" if all(res) else "FAIL")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
