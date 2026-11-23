# SPDX-License-Identifier: Apache-2.0
"""Compare the C decoder (c/ac4dec-c) with the Python reference (ac4/).

    python3 c/tests/compare.py [CAPTURE.m4s ...]

With no arguments the captures are the *.m4s / *.mp4 files under
AC4_TEST_DIR (recursively), as the Python integration tests find them.
Captures are never part of the repository; every output goes to
AC4_CPORT_OUT (default ~/.cache/ac4-cport).

Both sides decode with everything on and their default CLI settings:
  * Python: Decoder(element) (A-SPX rendered, output guard on) in passes
    of 1800 frames, then the StreamResampler -> float64, 48 kHz, decoder
    scale (exactly what `ac4dec` writes before its level rule);
  * C: c/ac4dec-c --f64 (A-SPX rendered, guard on, default lookahead).

Reported per channel: max |C - Python| in decoder units and relative to
full scale (the CLI's level rule writes the element's reference level at
0.9 of full scale, so full scale = REF_PEAK / 0.9), and the SNR of the C
output against the Python's; the per-frame status (concealed frames and
their reasons) of both; and the 16-bit WAVs of both CLIs (same level
rule): how many samples differ, and by how many LSB at most.

The reference is the plain Python package (no test-side changes).
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from ac4 import mp4                                   # noqa: E402
from ac4 import cli as PYCLI                          # noqa: E402
from ac4.decoder import Decoder, REF_PEAK, probe_element  # noqa: E402
from ac4.resample import StreamResampler              # noqa: E402

C_CLI = os.path.join(REPO, "c", "ac4dec-c")
OUT = os.path.expanduser(os.environ.get("AC4_CPORT_OUT", "~/.cache/ac4-cport"))
CHANS6 = ("L", "R", "C", "LFE", "Ls", "Rs")
MAX_PASS = 1800
FRAME = 1536

def python_reference(path, max_pass=MAX_PASS):
    """-> (float64 (n, ch) at 48 kHz, decoder scale; element; Decoder;
    per-frame status list)."""
    frames = mp4.read_frames(path)
    element = probe_element(frames)
    dec = Decoder(element)
    rs = StreamResampler()
    out, status = [], []
    for i in range(0, len(frames), max(8, max_pass)):
        y = dec.decode_pass(frames[i:i + max_pass])
        for s, c in zip(dec.last_status, dec.last_concealed):
            status.append(s if s is not None else
                          ("level" if c else None))
        if len(y):
            out.append(rs.push(y))
    pcm = np.vstack(out) if out else np.zeros((0, len(dec.chans)))
    return pcm, element, dec, status


def read_wav16(path):
    import wave
    with wave.open(path, "rb") as w:
        n, ch = w.getnframes(), w.getnchannels()
        return np.frombuffer(w.readframes(n), "<i2").reshape(n, ch)


def read_status(path):
    out = {}
    for line in open(path):
        i, st, why = line.split()
        out[int(i)] = None if st == "0" else why
    return out


def _row(nm, a, b, fs, sl=slice(None)):
    d = a[sl] - b[sl]
    md = float(np.abs(d).max()) if d.size else 0.0
    es = float((b[sl] ** 2).sum())
    en = float((d ** 2).sum())
    snr = (float("inf") if en == 0 else
           (10 * np.log10(es / en) if es > 0 else float("-inf")))
    return md, md / fs, snr


def compare(path, tag):
    os.makedirs(OUT, exist_ok=True)
    c_wav = os.path.join(OUT, tag + ".c.wav")
    c_raw = os.path.join(OUT, tag + ".c.f64")
    c_st = os.path.join(OUT, tag + ".c.status")
    t0 = time.time()
    r = subprocess.run([C_CLI, path, c_wav, "--f64", c_raw, "--status", c_st,
                        "-q"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{tag}: ac4dec-c failed rc={r.returncode}: {r.stderr}")
        return None
    ref, element, dec, py_status = python_reference(path)
    n_frames = len(py_status)
    got = np.fromfile(c_raw, "<f8").reshape(-1, ref.shape[1])
    c_status = read_status(c_st)
    fs = REF_PEAK[element] / 0.9
    names = CHANS6 if ref.shape[1] == 6 else CHANS6[:2]
    py_wav = os.path.join(OUT, tag + ".py.wav")
    PYCLI.decode_file(path, py_wav, quiet=True)
    w_py, w_c = read_wav16(py_wav), read_wav16(c_wav)
    print(f"\n{tag}  ({os.path.basename(path)}, element {element}"
          f"{', ASPX_ACPL_3 core' if dec.acpl3 else ''}, {n_frames} frames, "
          f"{len(ref)} samples; lengths py {len(ref)} c {len(got)})")
    if got.shape != ref.shape:
        print("  LENGTH MISMATCH")
        return None
    py_bad = {i: s for i, s in enumerate(py_status) if s is not None}
    c_bad = {i: s for i, s in c_status.items() if s is not None}
    print(f"  concealed: py {len(py_bad)} {sorted(set(py_bad.values()))}  "
          f"c {len(c_bad)} {sorted(set(c_bad.values()))}  "
          f"{'same frames' if set(py_bad) == set(c_bad) else 'DIFFERENT FRAMES'}")
    rows = []
    print(f"  {'ch':4s} {'peak (dec)':>11s} | {'max |C-py|':>11s} "
          f"{'SNR dB':>7s}")
    for c, nm in enumerate(names):
        pk = float(np.abs(ref[:, c]).max()) if len(ref) else 0.0
        _m, rel, snr = _row(nm, got[:, c], ref[:, c], fs)
        rows.append((tag, nm, pk, rel))
        print(f"  {nm:4s} {pk:11.5g} | {rel:11.2e} {snr:7.1f}")
    if w_py.shape == w_c.shape:
        dif = np.abs(w_py.astype(int) - w_c.astype(int))
        print(f"  16-bit WAVs: {int((dif > 0).sum())} of {dif.size} "
              f"samples differ, max {int(dif.max()) if dif.size else 0} LSB")
    else:
        print(f"  16-bit WAVs: shapes differ {w_py.shape} {w_c.shape}")
    print(f"  ({time.time() - t0:.1f} s)")
    return rows


def main(argv):
    files = argv
    if not files:
        d = os.environ.get("AC4_TEST_DIR")
        if not d:
            sys.exit("pass capture files or set AC4_TEST_DIR")
        for pat in ("*.m4s", "*.mp4"):
            files += glob.glob(os.path.join(d, "**", pat), recursive=True)
        files = sorted(files)
    if not os.path.exists(C_CLI):
        sys.exit(f"build {C_CLI} first (make -C c)")
    worst = 0.0
    seen = {}
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        parent = os.path.basename(os.path.dirname(p))
        tag = f"{parent}_{base}"
        seen[tag] = seen.get(tag, 0) + 1
        if seen[tag] > 1:
            tag += f"_{seen[tag]}"
        rows = compare(p, tag)
        if rows is None:
            worst = float("inf")
            continue
        worst = max([worst] + [r[3] for r in rows])
    ok = worst < 1e-6
    print(f"\nworst max |C - Python| relative to full scale: {worst:.2e} "
          f"({'PASS' if ok else 'FAIL'} against 1e-6)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
