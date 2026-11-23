# SPDX-License-Identifier: Apache-2.0
"""Damaged input: the C decoder against the Python, on the same damage.

    nice -n 10 python3 c/tests/robust_compare.py [CAPTURE.m4s ...]

The measurement of tests/test_robust.py (the damaged-input gate), run on
both decoders: each capture is cut to its first AC4_TEST_FRAGMENTS
fragments (default 4), damaged deterministically with tests/inject.py
(every kind, AC4_ROBUST_SEEDS seeds each, default 20, seed = 7919 s +
kind index as there), decoded, and assessed against the same decoder's
clean decode of the cut (inject.assess): worst peak over the damaged
decode relative to the clean peak, the verdict of every damaged frame
(concealed / identical / bounded / LOUD), undamaged frames concealed
(collateral), and the gate's failures (peak above 2x, a LOUD frame, more
than 4 collateral frames per hit). The clean cuts must conceal nothing.

Python: Decoder(element) (A-SPX, guard on), one decode pass, internal
rate. C: libac4dec.so through ctypes, internal rate, guard on, with the
default lookahead and (AC4_C_LOOKAHEADS, default "0,8") the minimum.

Captures: arguments, or AC4_TEST_DIR (the lanes/ and station_b captures
under it: the seven distinct streams). Cuts go to AC4_CPORT_OUT.
"""
from __future__ import annotations

import collections
import ctypes
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

import capture_util as CU                             # noqa: E402
import inject as I                                    # noqa: E402
from ac4 import mp4                                   # noqa: E402
from ac4.decoder import Decoder, probe_element        # noqa: E402

OUT = os.path.expanduser(os.environ.get("AC4_CPORT_OUT", "~/.cache/ac4-cport"))
N_FRAG = int(os.environ.get("AC4_TEST_FRAGMENTS", "4"))
N_SEEDS = int(os.environ.get("AC4_ROBUST_SEEDS", "20"))
LOOKS = [int(x) for x in os.environ.get("AC4_C_LOOKAHEADS", "0,8").split(",")]
WORKERS = int(os.environ.get("AC4_WORKERS", "2"))
FRAME = 1536
MAX_COLLATERAL = 4
LIB = os.path.join(REPO, "c", "libac4dec.so")


class Cfg(ctypes.Structure):
    _fields_ = [("element", ctypes.c_int), ("channels", ctypes.c_int),
                ("internal_rate", ctypes.c_int), ("lookahead", ctypes.c_int),
                ("no_guard", ctypes.c_int)]


class Frame(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int), ("reason", ctypes.c_char_p),
                ("index", ctypes.c_longlong), ("n_channels", ctypes.c_int),
                ("n_samples", ctypes.c_int),
                ("pcm", ctypes.POINTER(ctypes.c_double) * 6),
                ("sequence_counter", ctypes.c_int), ("iframe", ctypes.c_int),
                ("seq_jump", ctypes.c_int)]


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        L = ctypes.CDLL(LIB)
        L.ac4dec_open.restype = ctypes.c_void_p
        L.ac4dec_open.argtypes = [ctypes.POINTER(Cfg)]
        L.ac4dec_close.argtypes = [ctypes.c_void_p]
        L.ac4dec_decode_frame.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                          ctypes.c_size_t,
                                          ctypes.POINTER(Frame)]
        L.ac4dec_flush.argtypes = [ctypes.c_void_p, ctypes.POINTER(Frame)]
        _LIB = L
    return _LIB


def c_decode(frames, element, lookahead=0, guard=True):
    """-> (y (n * 1536, ch), concealed bool per frame, reasons per frame)."""
    L = lib()
    cfg = Cfg(1 if element == "5_X" else 2, 0, 1, lookahead,
              0 if guard else 1)
    d = L.ac4dec_open(ctypes.byref(cfg))
    if not d:
        raise RuntimeError("ac4dec_open failed")
    n_ch = 6 if element == "5_X" else 2
    y = np.zeros((len(frames) * FRAME, n_ch))
    conc = np.zeros(len(frames), dtype=bool)
    why = [None] * len(frames)
    fo = Frame()
    k = 0

    def take():
        nonlocal k
        i = fo.index
        assert i == k, (i, k)
        for c in range(fo.n_channels):
            y[i * FRAME:(i + 1) * FRAME, c] = np.ctypeslib.as_array(
                fo.pcm[c], shape=(FRAME,))
        conc[i] = fo.status != 0
        why[i] = fo.reason.decode() if fo.reason else None
        k += 1

    try:
        for f in frames:
            if L.ac4dec_decode_frame(d, f if f else None, len(f),
                                     ctypes.byref(fo)):
                take()
        while L.ac4dec_flush(d, ctypes.byref(fo)):
            take()
    finally:
        L.ac4dec_close(d)
    assert k == len(frames)
    return y, conc, why


def py_decode(frames, element):
    d = Decoder(element)
    y = d.decode_pass(frames)
    return y, np.asarray(d.last_concealed, dtype=bool), d


def captures():
    files = sys.argv[1:]
    if files:
        return files
    d = os.environ.get("AC4_TEST_DIR")
    if not d:
        sys.exit("pass capture files or set AC4_TEST_DIR")
    out = sorted(glob.glob(os.path.join(d, "**", "lanes", "*", "*.m4s"),
                           recursive=True))
    out += sorted(glob.glob(os.path.join(d, "**", "station_b*", "*.m4s"),
                            recursive=True))
    return out


def load():
    os.makedirs(os.path.join(OUT, "cuts"), exist_ok=True)
    caps = []
    for i, p in enumerate(captures()):
        dst = os.path.join(OUT, "cuts", f"cut{i}.m4s")
        if CU.cut_fragments(p, dst, N_FRAG) == 0:
            continue
        frames = mp4.read_frames(dst)
        el = probe_element(frames)
        if el is None:
            continue
        caps.append((p, frames, el))
    return caps


def _foreign(caps, ci):
    for j, (_p, f2, e2) in enumerate(caps):
        if j != ci and e2 == caps[ci][2]:
            return f2
    return None


_CAPS = None
_CLEAN = None


def _init(caps, clean):
    global _CAPS, _CLEAN
    _CAPS, _CLEAN = caps, clean


def job(args):
    """One (capture, kind, seed): assess every decoder on it."""
    ci, kind, s = args
    _p, frames, el = _CAPS[ci]
    seed = 7919 * s + I.KINDS.index(kind)
    dfr, hits = I.damage(frames, kind, seed, foreign=_foreign(_CAPS, ci))
    res = {}
    for name in ["py"] + [f"c{la}" for la in LOOKS]:
        clean = _CLEAN[(ci, name)]
        try:
            if name == "py":
                y, conc, _d = py_decode(dfr, el)
            else:
                y, conc, _w = c_decode(dfr, el, int(name[1:]))
        except Exception as e:                                 # noqa: BLE001
            res[name] = dict(error=repr(e))
            continue
        if y.shape != clean.shape:
            res[name] = dict(error=f"shape {y.shape} != {clean.shape}")
            continue
        a = I.assess(clean, y, hits, conc)
        coll = int(conc.sum()) - int(conc[hits].sum())
        res[name] = dict(peak=a["peak_ratio"], per_hit=a["per_hit"],
                         coll=coll, n_hits=len(hits))
    return ci, kind, s, res


def main():
    t0 = time.time()
    caps = load()
    names = ["py"] + [f"c{la}" for la in LOOKS]
    print(f"{len(caps)} cuts of {N_FRAG} fragments, {len(I.KINDS)} kinds x "
          f"{N_SEEDS} seeds; decoders {names}")
    clean = {}
    for ci, (p, frames, el) in enumerate(caps):
        yp, cp, dp = py_decode(frames, el)
        clean[(ci, "py")] = yp
        line = (f"  cut{ci} {os.path.basename(os.path.dirname(p))}/"
                f"{os.path.basename(p)} {el} {len(frames)} frames: clean "
                f"concealed py {int(cp.sum())} (guard fired "
                f"{dp.guard.n_fired})")
        for la in LOOKS:
            yc, cc, why = c_decode(frames, el, la)
            clean[(ci, f"c{la}")] = yc
            # the same decode without the guard: the guard must change nothing
            yg, _c, _w = c_decode(frames, el, la, guard=False)
            line += (f", c{la} {int(cc.sum())} "
                     f"{sorted(set(w for w in why if w))} guard-changed "
                     f"{int((np.abs(yg - yc) > 0).any(axis=1).sum())} samples")
        print(line, flush=True)
    jobs = [(ci, kind, s) for ci in range(len(caps)) for kind in I.KINDS
            for s in range(N_SEEDS)]
    tally = {n: collections.Counter() for n in names}
    worst = {n: (0.0, None) for n in names}
    wloc = {n: (0.0, None) for n in names}
    fails = {n: [] for n in names}
    per_kind = {n: collections.defaultdict(float) for n in names}
    with mp.Pool(WORKERS, initializer=_init, initargs=(caps, clean)) as pool:
        for done, (ci, kind, s, res) in enumerate(
                pool.imap_unordered(job, jobs, chunksize=2)):
            tag = f"cut{ci} {kind} seed {7919 * s + I.KINDS.index(kind)}"
            for n, r in res.items():
                if "error" in r:
                    fails[n].append(f"{tag}: {r['error']}")
                    continue
                tally[n]["collateral"] += r["coll"]
                tally[n]["hits"] += r["n_hits"]
                if r["coll"] > MAX_COLLATERAL * max(1, r["n_hits"]):
                    fails[n].append(f"{tag}: {r['coll']} undamaged frames "
                                    "concealed")
                if r["peak"] > worst[n][0]:
                    worst[n] = (r["peak"], tag)
                per_kind[n][kind] = max(per_kind[n][kind], r["peak"])
                if r["peak"] > 2.0:
                    fails[n].append(f"{tag}: peak {r['peak']:.3g}x clean")
                for i, v, loc, lev in r["per_hit"]:
                    tally[n][v] += 1
                    if v == "bounded" and loc > wloc[n][0]:
                        wloc[n] = (loc, f"{tag} frame {i}")
                    if v == "LOUD":
                        fails[n].append(f"{tag}: frame {i} not concealed, "
                                        f"{lev:.3g}x the clean peak")
            if (done + 1) % 100 == 0:
                print(f"  {done + 1}/{len(jobs)} ({time.time() - t0:.0f} s)",
                      flush=True)
    print()
    for n in names:
        t = tally[n]
        print(f"{n:4s}: worst peak {worst[n][0]:.3f}x clean ({worst[n][1]}); "
              f"damaged frames {t['hits']}: concealed {t['concealed']}, "
              f"identical {t['identical']}, bounded {t['bounded']}, LOUD "
              f"{t['LOUD']}; collateral {t['collateral']}; worst unflagged "
              f"{wloc[n][0]:.3g}x local ({wloc[n][1]}); gate failures "
              f"{len(fails[n])}")
        print("      worst peak by kind: " + ", ".join(
            f"{k} {per_kind[n][k]:.2f}" for k in I.KINDS))
        for f in fails[n][:10]:
            print("      FAIL " + f)
    print(f"({time.time() - t0:.0f} s)")
    return 0 if not any(fails.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
