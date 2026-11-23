# SPDX-License-Identifier: Apache-2.0
"""Sanity checks of the A-CPL upmix (ASPX_ACPL_3) on captures.

    python3 c/tests/acpl_check.py [CAPTURE.m4s ...]

There is no reference decoder to compare with, so this measures what the
standard implies and what a listener would notice, on the Python decoder
(one pass, internal rate) and the C decoder (libac4dec.so):

* levels:     RMS and peak of every channel (dB of full scale, the CLI's
              level rule), and of the core pair (the stereo downmix the
              stream carries, decoded with channels=2);
* spectra:    each channel's level in five bands against the core's, and
              its bandwidth (the highest frequency within 50 dB of the
              channel's spectral peak);
* downmix:    TS 103 190-1 Pseudocode 118 makes z0 + z1 + z4 (L + Ls/sqrt2
              + C/sqrt2) a mix of the two core channels alone, the
              decorrelated y2 terms cancelling; with the gammas of a
              downmix (L + C/sqrt2 + Ls/sqrt2) / (1 + sqrt2) that mix is
              the core channel itself. The residual of that downmix
              against the core, overall and for the worst frame;
* boundaries: the level step from one frame to the next of every channel
              (99.9th percentile and maximum over all frames, and at the
              I-frames, where the A-CPL configuration is sent) against the
              core's, and the high-frequency energy folded on the frame
              period (a click at every A-CPL frame boundary would show as
              a peak of the folded profile);
* stability:  the poles of the three decorrelators' all-pass filters (Tables
              199-201) and the output is finite;
* alignment:  the lag of every channel against the centre (cross-
              correlation, the align_check measure);
* proxies:    the share of speech-band energy (250 Hz to 4 kHz) in C, L,
              R; correlation of C with the core's mid signal; correlation
              of the surrounds with each other and with the fronts.

Captures are never part of the repository; nothing is written.
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

from ac4 import mp4, acpl as ACPL, toc as M          # noqa: E402
from ac4.decoder import Decoder, REF_PEAK, probe_element  # noqa: E402
from align_check import lag                           # noqa: E402

FRAME = 1536
FS = 48000 * 960 / 1001          # internal rate
NAMES = ("L", "R", "C", "LFE", "Ls", "Rs")


def db(x):
    return 10 * np.log10(np.maximum(x, 1e-30))


def psd(x, n=4096):
    m = len(x) // n
    if m < 1:
        return np.zeros(n // 2 + 1)
    w = np.hanning(n)
    X = np.fft.rfft(x[:m * n].reshape(m, n) * w, axis=1)
    return (np.abs(X) ** 2).mean(axis=0)


def frame_rms(x):
    n = len(x) // FRAME
    return np.sqrt((x[:n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))


def poles():
    """Largest pole radius of each decorrelator / region (all-pass:
    denominator a[0..L])."""
    out = []
    for d in range(3):
        for r, (_lo, _hi, _dl, L) in enumerate(ACPL.REGIONS):
            a = (1.0,) + ACPL.COEF[d][r]
            out.append(float(np.abs(np.roots(a)).max()))
    return out


def check(path, use_c=True):
    frames = mp4.read_frames(path)
    if probe_element(frames) != "5_X":
        return None
    dec = Decoder("5_X")
    y = dec.decode_pass(frames)
    if not dec.acpl3:
        return None
    core = Decoder("5_X", channels=2).decode_pass(frames)
    fs_ = REF_PEAK["5_X"] / 0.9
    ifr = [i for i, f in enumerate(frames)
           if M.parse(f).get("b_iframe_global")]
    first = ifr[0] if ifr else 0
    # from the first frame with A-CPL data (the first I-frame) on, plus
    # the A-CPL frame's offset and the bank's delay
    s0 = first * FRAME + 577 + 4 * 64
    print(f"\n{os.path.basename(path)}: {len(frames)} frames, first I-frame "
          f"{first}, I-frames {len(ifr)}; concealed {dec.n_concealed}, "
          f"burst frames {dec.n_burst}")
    ok = True
    Y = y[s0:]
    Lo, Ro = core[s0:, 0], core[s0:, 1]
    print("  finite:", bool(np.isfinite(y).all()),
          " decorrelator pole radii max %.3f" % max(poles()))
    ok &= bool(np.isfinite(y).all()) and max(poles()) < 1.0
    # levels
    print("  ch     RMS dBFS  peak dBFS")
    for nm, x in (("Lo", Lo), ("Ro", Ro)) + tuple(
            (NAMES[k], Y[:, k]) for k in range(6)):
        r = np.sqrt((x ** 2).mean()) / fs_
        p = np.abs(x).max() / fs_
        print(f"  {nm:4s} {db(r * r):9.1f} {db(p * p):9.1f}")
        ok &= p < 1.0
    # spectra
    bands = ((20, 200), (200, 2000), (2000, 8000), (8000, 16000),
             (16000, 22000))
    f = np.fft.rfftfreq(4096, 1 / FS)
    P = {nm: psd(x) for nm, x in (("Lo", Lo), ("Ro", Ro))}
    P.update({NAMES[k]: psd(Y[:, k]) for k in range(6)})
    ref = P["Lo"] + P["Ro"]
    print("  band levels (dB) against the core Lo+Ro, and bandwidth:")
    print("  ch   " + " ".join(f"{a // 1000:>2}-{b // 1000:<2}k" if a >= 1000
                               else f"{a:>3}-{b // 1000}k" for a, b in bands)
          + "   bw kHz")
    for nm in ("L", "R", "C", "LFE", "Ls", "Rs"):
        row = []
        for a, b in bands:
            m = (f >= a) & (f < b)
            row.append(f"{db(P[nm][m].sum() / ref[m].sum()):7.1f}")
        pk = P[nm].max()
        bw = f[np.flatnonzero(P[nm] > pk * 1e-5)].max() / 1000 if pk > 0 \
            else 0.0
        print(f"  {nm:4s} " + " ".join(row) + f"   {bw:5.1f}")
    # downmix identity
    k = 1 / (1 + np.sqrt(2))
    print("  downmix (X + C/sqrt2 + Xs/sqrt2)/(1+sqrt2) against the core:")
    for nm, t, a, b in (("Lo", Lo, Y[:, 0], Y[:, 4]),
                        ("Ro", Ro, Y[:, 1], Y[:, 5])):
        est = (a + Y[:, 2] / np.sqrt(2) + b / np.sqrt(2)) * k
        e = t - est
        res = db((e ** 2).sum() / (t ** 2).sum())
        fe, ft = frame_rms(e), frame_rms(t)
        live = ft > 1e-3 * ft.max()
        worst = db((fe[live] / ft[live]).max() ** 2)
        print(f"    {nm}: residual {res:6.1f} dB overall, worst frame "
              f"{worst:6.1f} dB")
        ok &= res < -30
    # boundaries: frame-to-frame level steps
    print("  frame-to-frame level step |dB| (99.9th pct / max / max at an "
          "I-frame), core first:")
    iset = set(i - first for i in ifr if i >= first)
    for nm, x in (("Lo", Lo), ("Ro", Ro)) + tuple(
            (NAMES[k], Y[:, k]) for k in (0, 1, 2, 4, 5)):
        r = frame_rms(x)
        live = (r[1:] > 1e-3 * r.max()) & (r[:-1] > 1e-3 * r.max())
        st = np.abs(db(r[1:] ** 2) - db(r[:-1] ** 2))
        at_i = [st[i - 1] for i in range(1, len(r)) if i in iset
                and live[i - 1]]
        print(f"    {nm:3s} {np.percentile(st[live], 99.9):5.1f} "
              f"{st[live].max():5.1f} {max(at_i) if at_i else 0:5.1f}")
    # the step into the first A-CPL frame (core before, upmix after)
    pre = slice(max(0, s0 - 4 * FRAME), s0)
    post = slice(s0, s0 + 4 * FRAME)
    print("  at the first I-frame (4 frames before / after, dBFS): " + ", ".join(
        f"{NAMES[k]} {db((y[pre, k] ** 2).mean() / fs_ ** 2):.1f}/"
        f"{db((y[post, k] ** 2).mean() / fs_ ** 2):.1f}"
        for k in (0, 1, 2, 4, 5)))
    # folded high-frequency energy: clicks at the frame period?
    print("  HF energy folded on the frame period, peak / mean "
          "(1 = flat):", end="")
    for nm, x in (("Lo", Lo),) + tuple((NAMES[k], Y[:, k])
                                       for k in (0, 2, 4)):
        d = np.diff(x) ** 2
        n = len(d) // FRAME
        prof = d[:n * FRAME].reshape(n, FRAME).mean(axis=0)
        prof = np.convolve(prof, np.ones(32) / 32, mode="same")
        print(f" {nm} {prof.max() / prof.mean():.2f}", end="")
    print()
    # alignment
    row = []
    for kk in (0, 1, 3, 4, 5):
        lg, r = lag(y[s0:, 2], y[s0:, kk])
        row.append(f"{NAMES[kk]} {lg:+d} ({r:.2f})")
        if kk in (0, 1) and r > 0.3 and abs(lg) > 2:
            ok = False
    print("  lag against C: " + ", ".join(row))
    # proxies
    sp = {}
    for nm in ("L", "R", "C", "Ls", "Rs"):
        m = (f >= 250) & (f < 4000)
        sp[nm] = P[nm][m].sum() / max(P[nm].sum(), 1e-30)
    mid = Lo + Ro

    def corr(a, b):
        return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)
                                             + 1e-30))
    print("  speech band (250 Hz - 4 kHz) share: " + ", ".join(
        f"{k} {100 * v:.0f} %" for k, v in sp.items()))
    print(f"  correlation: C~mid {corr(Y[:, 2], mid):.2f}, "
          f"L~Lo {corr(Y[:, 0], Lo):.2f}, Ls~Lo {corr(Y[:, 4], Lo):.2f}, "
          f"Ls~L {corr(Y[:, 4], Y[:, 0]):.2f}, Ls~Rs "
          f"{corr(Y[:, 4], Y[:, 5]):.2f}, Rs~R {corr(Y[:, 5], Y[:, 1]):.2f}")
    if use_c:
        import robust_compare as RC
        yc, _c, _w = RC.c_decode(frames, "5_X", lookahead=0)
        n = min(len(yc), len(y))
        d = np.abs(yc[:n] - y[:n]).max() / fs_
        print(f"  C decoder (libac4dec): max |C - Python| {d:.2e} of full "
              "scale")
        ok &= d < 1e-6
    print("  " + ("PASS" if ok else "FAIL"))
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
        sys.exit("no ASPX_ACPL_3 capture")
    print("\nPASS" if all(res) else "\nFAIL")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
