# SPDX-License-Identifier: Apache-2.0
"""Compare the C decoder (c/ac4dec-c) with the Python reference (ac4/).

    python3 c/tests/compare.py [CAPTURE.m4s ...]

With no arguments the captures are the *.m4s / *.mp4 files under
AC4_TEST_DIR (recursively), as the Python integration tests find them.
Captures are never part of the repository; every output goes to
AC4_CPORT_OUT (default ~/.cache/ac4-cport).

Both sides decode with the same settings:
  * the Python Decoder with A-SPX rendering disabled (milestone 1 of the C
    port renders no A-SPX; the Python then plays the core band, undelayed)
    and the output guard off (not ported yet), in passes of 1800 frames,
    then the StreamResampler -> float64, 48 kHz, decoder scale;
  * c/ac4dec-c --f64 -> the same.
Reported per channel: max |C - Python| in decoder units and relative to
full scale (the CLI's level rule writes the element's reference level at
0.9 of full scale, so full scale = REF_PEAK / 0.9), and the SNR of the C
output against the Python's. Also the 16-bit WAVs of both CLIs (same
level rule): how many samples differ, and by how many LSB at most.
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
from ac4 import aspx_render                           # noqa: E402
from ac4 import cli as PYCLI                          # noqa: E402
from ac4.decoder import Decoder, REF_PEAK, probe_element  # noqa: E402
from ac4.resample import StreamResampler              # noqa: E402

C_CLI = os.path.join(REPO, "c", "ac4dec-c")
OUT = os.path.expanduser(os.environ.get("AC4_CPORT_OUT", "~/.cache/ac4-cport"))
CHANS6 = ("L", "R", "C", "LFE", "Ls", "Rs")


def _no_aspx(*a, **k):
    raise RuntimeError("A-SPX rendering disabled for the C comparison")


def python_reference(path, max_pass=1800):
    """-> (float64 (n, ch) at 48 kHz, decoder scale; element; Decoder)."""
    frames = mp4.read_frames(path)
    element = probe_element(frames)
    saved = aspx_render.apply_hf_pair
    aspx_render.apply_hf_pair = _no_aspx
    try:
        dec = Decoder(element, guard=False)
        rs = StreamResampler()
        out = []
        for i in range(0, len(frames), max_pass):
            y = dec.decode_pass(frames[i:i + max_pass])
            if len(y):
                out.append(rs.push(y))
    finally:
        aspx_render.apply_hf_pair = saved
    pcm = np.vstack(out) if out else np.zeros((0, len(dec.chans)))
    return pcm, element, dec


def python_wav(path, dst):
    saved = aspx_render.apply_hf_pair
    aspx_render.apply_hf_pair = _no_aspx
    try:
        PYCLI.decode_file(path, dst, quiet=True, guard=False)
    finally:
        aspx_render.apply_hf_pair = saved


def read_wav16(path):
    import wave
    with wave.open(path, "rb") as w:
        n, ch = w.getnframes(), w.getnchannels()
        return np.frombuffer(w.readframes(n), "<i2").reshape(n, ch)


def compare(path, tag):
    os.makedirs(OUT, exist_ok=True)
    ref, element, dec = python_reference(path)
    np.save(os.path.join(OUT, tag + ".py.npy"), ref)
    c_wav = os.path.join(OUT, tag + ".c.wav")
    c_raw = os.path.join(OUT, tag + ".c.f64")
    r = subprocess.run([C_CLI, path, c_wav, "--f64", c_raw, "-q"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{tag}: ac4dec-c failed rc={r.returncode}: {r.stderr}")
        return None
    got = np.fromfile(c_raw, "<f8").reshape(-1, ref.shape[1])
    fs = REF_PEAK[element] / 0.9
    names = CHANS6 if ref.shape[1] == 6 else CHANS6[:2]
    py_wav = os.path.join(OUT, tag + ".py.wav")
    python_wav(path, py_wav)
    w_py, w_c = read_wav16(py_wav), read_wav16(c_wav)
    rows = []
    print(f"\n{tag}  ({os.path.basename(path)}, element {element}"
          f"{', ASPX_ACPL_3 core' if dec.acpl3 else ''}, "
          f"{len(ref)} samples; Python concealed {dec.n_concealed}; "
          f"lengths py {len(ref)} c {len(got)})")
    if got.shape != ref.shape:
        print("  LENGTH MISMATCH")
        return None
    print(f"  {'ch':4s} {'peak (dec)':>12s} {'max|diff|':>11s} "
          f"{'rel. FS':>9s} {'SNR dB':>8s}")
    for c, nm in enumerate(names):
        d = got[:, c] - ref[:, c]
        md = float(np.abs(d).max()) if len(d) else 0.0
        pk = float(np.abs(ref[:, c]).max()) if len(d) else 0.0
        es = float((ref[:, c] ** 2).sum())
        en = float((d ** 2).sum())
        snr = (float("inf") if en == 0 else
               (10 * np.log10(es / en) if es > 0 else float("-inf")))
        rows.append((tag, nm, pk, md, md / fs, snr))
        print(f"  {nm:4s} {pk:12.5g} {md:11.3g} {md / fs:9.2e} "
              f"{snr:8.1f}")
    if w_py.shape == w_c.shape:
        dif = np.abs(w_py.astype(int) - w_c.astype(int))
        print(f"  16-bit WAVs: {int((dif > 0).sum())} of {dif.size} samples "
              f"differ, max {int(dif.max()) if dif.size else 0} LSB")
    else:
        print(f"  16-bit WAVs: shapes differ {w_py.shape} {w_c.shape}")
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
        t0 = time.time()
        rows = compare(p, tag)
        if rows is None:
            worst = float("inf")
            continue
        worst = max([worst] + [r[4] for r in rows])
        print(f"  ({time.time() - t0:.1f} s)")
    print(f"\nworst max |diff| relative to full scale: {worst:.2e} "
          f"({'PASS' if worst < 1e-6 else 'FAIL'} against 1e-6)")
    return 0 if worst < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
