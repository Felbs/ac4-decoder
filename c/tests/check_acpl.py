# SPDX-License-Identifier: Apache-2.0
"""The C A-CPL upmix against the Python's, on random parameters.

    python3 c/tests/check_acpl.py      (after make -C c)

The captures at hand exercise one A-CPL configuration (15 parameter bands,
fine quantisation, one parameter set per frame, smooth and a few steep
frames), which c/tests/compare.py checks end to end. This drives every
other branch through ac4_acpl_run (libac4dec.so, ctypes) and ac4/acpl.py
(render) on the same random input: 12, 9 and 7 bands, coarse
quantisation, two parameter sets, steep interpolation with change slots
inside and beyond the frame, out-of-range values (clamped), concealed
frames (held parameters, and a cold concealed frame), frames without data
(passthrough, state cleared) and a cut last frame. 40 streams of 12
frames; every output channel must agree to 1e-12 of its peak.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from ac4 import acpl as ACPL                          # noqa: E402
from ac4 import acpl_parse as AP                      # noqa: E402
from ac4.robust import CONCEALED                      # noqa: E402

LIB = os.path.join(REPO, "c", "libac4dec.so")
NONE, CONC, DATA = 0, 1, 2                            # AC4_GRP_*


class Cfg(ctypes.Structure):
    _fields_ = [("num_param_bands", ctypes.c_int),
                ("param_band", ctypes.c_int),
                ("quant_mode_0", ctypes.c_int),
                ("quant_mode_1", ctypes.c_int)]


class Frame(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("cfg", Cfg),
                ("interp", ctypes.c_int), ("n_sets", ctypes.c_int),
                ("ts", ctypes.c_int * 2),
                ("q", ((ctypes.c_int * 15) * 2) * 11)]


Slots = ctypes.c_double * (24 * 64)


def random_frame(rng, cfg, n_sets, interp):
    q = []
    for data_type, which in AP.DATA_2CH:
        mode = AP.QUANT[cfg["quant_mode_1" if which else "quant_mode_0"]]
        lo, hi = AP.Q_RANGE[(data_type, mode)]
        v = rng.integers(lo - 2, hi + 3, (n_sets, cfg["num_param_bands"]))
        q.append(v.tolist())
    ts = sorted(rng.integers(0, 32, n_sets).tolist()) if interp else []
    return dict(cfg=cfg, q=q, interp=interp, timeslots=tuple(ts))


def to_c(g):
    f = Frame()
    if g is None:
        f.kind = NONE
        return f
    if g is CONCEALED:
        f.kind = CONC
        return f
    f.kind = DATA
    c = g["cfg"]
    f.cfg = Cfg(c["num_param_bands"], 0, c["quant_mode_0"],
                c["quant_mode_1"])
    f.interp = g["interp"]
    f.n_sets = len(g["q"][0])
    for i, t in enumerate(g["timeslots"]):
        f.ts[i] = t
    for k in range(11):
        for s, row in enumerate(g["q"][k]):
            for b, v in enumerate(row):
                f.q[k][s][b] = int(v)
    return f


def run_c(L, QL, QR, groups):
    n_slots = QL.shape[1]
    state = ctypes.create_string_buffer(1 << 20)
    work = ctypes.create_string_buffer(1 << 21)
    out = {k: np.zeros((64, n_slots), complex)
           for k in ("L", "R", "C", "Ls", "Rs")}
    for f, g in enumerate(groups):
        a = 24 * f
        n = min(24, n_slots - a)
        xr = [Slots(), Slots()]
        xi = [Slots(), Slots()]
        for c, Q in enumerate((QL, QR)):
            blk = np.zeros((24, 64), complex)
            blk[:n] = Q[:, a:a + n].T
            xr[c][:] = blk.real.ravel()
            xi[c][:] = blk.imag.ravel()
        zr = [Slots() for _ in range(5)]
        zi = [Slots() for _ in range(5)]
        fr = to_c(g)
        PP = ctypes.c_void_p * 2
        P5 = ctypes.c_void_p * 5
        L.ac4_acpl_run(state, work,
                       None if g is None else ctypes.byref(fr), n,
                       PP(*[ctypes.addressof(x) for x in xr]),
                       PP(*[ctypes.addressof(x) for x in xi]),
                       P5(*[ctypes.addressof(x) for x in zr]),
                       P5(*[ctypes.addressof(x) for x in zi]))
        for k, name in enumerate(("L", "R", "C", "Ls", "Rs")):
            re = np.frombuffer(zr[k], float).reshape(24, 64)[:n]
            im = np.frombuffer(zi[k], float).reshape(24, 64)[:n]
            out[name][:, a:a + n] = (re + 1j * im).T
    return out


def main():
    L = ctypes.CDLL(LIB)
    L.ac4_acpl_run.restype = None
    rng = np.random.default_rng(2026)
    worst, n_frames, kinds = 0.0, 0, {}
    for case in range(40):
        nb = (15, 12, 9, 7)[case % 4]
        cfg = dict(num_param_bands=nb, param_band=0,
                   quant_mode_0=int(rng.integers(2)),
                   quant_mode_1=int(rng.integers(2)))
        groups = []
        for f in range(12):
            r = rng.random()
            if f in (0, 1) and case % 5 == 0:
                g = None if f == 0 else CONCEALED       # cold concealed
            elif r < 0.08:
                g = None
            elif r < 0.2:
                g = CONCEALED
            else:
                g = random_frame(rng, cfg, int(rng.integers(1, 3)),
                                 int(rng.random() < 0.4))
            groups.append(g)
            key = ("none" if g is None else "concealed" if g is CONCEALED
                   else f"{len(g['q'][0])} set(s) "
                        f"{'steep' if g['interp'] else 'smooth'}")
            kinds[key] = kinds.get(key, 0) + 1
        n_slots = 12 * 24 - (7 if case % 3 == 0 else 0)  # a cut last frame
        QL = (rng.standard_normal((64, n_slots))
              + 1j * rng.standard_normal((64, n_slots))) * 1e4
        QR = (rng.standard_normal((64, n_slots))
              + 1j * rng.standard_normal((64, n_slots))) * 1e4
        py = ACPL.render(QL, QR, groups, ts_offset=0)
        c = run_c(L, QL, QR, groups)
        for k in py:
            pk = max(np.abs(py[k]).max(), 1e-30)
            d = np.abs(py[k] - c[k]).max() / pk
            worst = max(worst, d)
        n_frames += len(groups)
    print(f"{n_frames} frames ({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))}): "
          f"max |C - Python| / peak {worst:.2e}")
    ok = worst < 1e-12
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
