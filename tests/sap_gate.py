# SPDX-License-Identifier: Apache-2.0
"""Measurements for the full-SAP (sap_mode 3) gates, on real captures.

    python tests/sap_gate.py CAPTURE [CAPTURE ...]     # prints a JSON report

Nothing here is part of the library. For one channel pair of a stream it
decodes the MDCT spectra after stereo processing, frame by frame, under the
decoder's own SAP coefficients and under deliberately wrong variants:

    spec         clause 5.3.2 / 5.3.3.2 as implemented
    passthrough  the old behaviour: the two tracks used as L and R
    sign_flip    alpha -> -alpha
    transposed   [a b; c d] read as [a c; b d]
    ms_only      alpha ignored (plain M/S)

and scores each variant three ways.

1. continuity: median |frame-to-frame change| of the per-band ILD and
   inter-channel correlation across boundaries where the pair switches into
   or out of sap_mode 3, over the same for boundaries between two non-SAP3
   frames. A band counts only where every non-SAP3 frame of the boundary has
   a non-zero side track in it: measured on both broadcasters, M/S frames
   very often send an all-zero side in the upper core bands, which makes
   them exactly mono there and no reference for the stereo image.

2. aspx_ref: the A-SPX signal envelopes carry each channel's high-band
   energy as the encoder measured it on the original signal, independent of
   the core (Pseudocodes 80, 82, 84). On SAP3 frames, the core's ILD just
   below the crossover is compared with the envelopes' ILD just above it:
   correlation r, and the scale sd(core) / sd(A-SPX) (signed with r), which
   is 1 for an image that is the same on both sides of the crossover.

3. level_step: mean signed change of the pair's total level (dB) on
   entering sap_mode 3 from a non-SAP3 frame. A decoding that adds or
   removes energy in proportion to alpha shows up as an offset.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from ac4 import audio, frameinfo, mp4
from ac4 import aspx_env as V
from ac4 import aspx_parse as P
from ac4 import aspx_render as R37
from ac4 import stream as SS
from ac4.decoder import probe_element

GOOD = audio.sap_coeffs
EDGES = [0, 250, 500, 1000, 1500, 2000, 3000, 4500, 6000, 9000, 12000]
NB = len(EDGES) - 1
CORE_LO, CORE_HI = 9000.0, 12000.0          # core band just below crossover
HF_SBG = 2                                  # A-SPX subband groups just above


def _remap(fn):
    return lambda sap: [[None if x is None else fn(x) for x in g]
                        for g in GOOD(sap)]


VARIANTS = {
    "spec": GOOD,
    "passthrough": lambda sap: [[None] * len(g) for g in GOOD(sap)],
    "sign_flip": _remap(lambda x: (x[2], x[1], x[0], x[3])),
    "transposed": _remap(lambda x: (x[0], x[2], x[1], x[3])),
    "ms_only": _remap(lambda x: (1.0, 1.0, 1.0, -1.0)),
}


def _bins(n):
    return np.digitize((np.arange(n) + 0.5) * 24000.0 / n, EDGES) - 1


def _frame_stats(wa, wb, side_packed, fr):
    """-> (band stats (NB, 3): EL, ER, ELR; side present (NB,); core-top
    energies (EL, ER))."""
    st = np.zeros((NB, 3))
    top = np.zeros(2)
    for a, b in zip(wa, wb):
        idx = _bins(len(a))
        for k in range(NB):
            m = idx == k
            st[k] += (np.dot(a[m], a[m]), np.dot(b[m], b[m]), np.dot(a[m], b[m]))
        f = (np.arange(len(a)) + 0.5) * 24000.0 / len(a)
        m = (f >= CORE_LO) & (f < CORE_HI)
        top += (np.dot(a[m], a[m]), np.dot(b[m], b[m]))
    side = np.zeros(NB, bool)
    if side_packed is not None:
        win, _ = audio.ungroup(side_packed, fr)
        for w in win:
            idx = _bins(len(w))
            for k in range(NB):
                side[k] |= bool(np.any(w[idx == k] != 0))
    return st, side, top


def _hf_ild(got, cfg, t, maps, prev):
    """Envelope ILD (dB) of one frame's aspx_data_2ch, first HF_SBG groups."""
    if got is None or len(got) < 2:
        return np.nan

    def a_of(d):                                  # Pseudocodes 82 / 84
        q = cfg["quant_mode_env"]
        if d["fr"]["int_class"] == P.FIXFIX and d["fr"]["num_env"] == 1:
            q = 0
        return 2 if q == 0 else 1
    try:
        qa, prev[0] = V.reconstruct(got[0], t, maps, prev[0])
        qb, prev[1] = V.reconstruct(got[1], t, maps, prev[1])
    except Exception:                                          # noqa: BLE001
        return np.nan
    ea = eb = 0.0
    if got[0].get("balance") == 1:
        a = a_of(got[0])
        for e in range(min(len(qa), len(qb))):
            for va, vb in list(zip(qa[e], qb[e]))[:HF_SBG]:
                nom = 2.0 ** (va / a + 1)
                ea += nom / (1 + 2.0 ** (12 - vb / a))
                eb += nom / (1 + 2.0 ** (vb / a - 12))
    else:
        a0, a1 = a_of(got[0]), a_of(got[1])
        ea = sum(2.0 ** (v / a0) for e in qa for v in e[:HF_SBG])
        eb = sum(2.0 ** (v / a1) for e in qb for v in e[:HF_SBG])
    return 10 * np.log10(ea / eb) if ea > 0 and eb > 0 else np.nan


def collect(frames, element, pair=0, variant="spec"):
    """Decode `frames` with one SAP variant. -> dict of per-frame arrays for
    channel pair `pair` (0 = L/R, 1 = Ls/Rs)."""
    info = frameinfo.walk(frames, element)
    acpl3 = any(r.get("codec_mode") == 4 for r in info)
    ka, kb = (("L", "R"), ("Ls", "Rs"))[pair]
    chans = ("L", "R") if element == "pair" or acpl3 else (
        "L", "R", "Ls", "Rs")
    gkey = ("lr", "sr")[pair]
    s = SS.Ac4Stream(element=element)
    cap = []
    orig = audio.unmix

    def spy(m, sd, st, fr):
        cap.append((sd.copy(), fr))
        return orig(m, sd, st, fr)

    rows = []
    hf_prev = {0: (None, 0), 1: (None, 0)}
    tmaps = None
    audio.sap_coeffs = VARIANTS[variant]
    SS.R.unmix = spy
    try:
        for f in frames:
            cap.clear()
            w, _l, g, _c = s.decode_frames([f], chans)
            sp = cap[pair] if pair < len(cap) else (None, None)
            rows.append(_frame_stats(w[ka], w[kb], *sp))
            hf = np.nan
            if s.cfg is not None:
                if tmaps is None or tmaps[0] is not s.cfg:
                    t = R37.t_for(s.cfg)
                    tmaps = (s.cfg, t, V.sbg_maps(t))
                hf = _hf_ild(g[gkey][0], s.cfg, tmaps[1], tmaps[2], hf_prev)
            rows[-1] = rows[-1] + (hf,)
    finally:
        SS.R.unmix = orig
        audio.sap_coeffs = GOOD
    is3 = np.array([len(r["sap"]) > pair and r["sap"][pair] == 3
                    for r in info])
    return dict(
        st=np.array([r[0] for r in rows]), side=np.array([r[1] for r in rows]),
        top=np.array([r[2] for r in rows]), hf=np.array([r[3] for r in rows]),
        is3=is3, info=info)


def score(c):
    st, is3 = c["st"], c["is3"]
    el, er, elr = st[..., 0], st[..., 1], st[..., 2]
    with np.errstate(all="ignore"):
        ok = (el > 0) & (er > 0)
        ild = np.where(ok, 10 * np.log10(el / er), np.nan)
        rho = np.where(ok, elr / np.sqrt(el * er), np.nan)
        lev = 10 * np.log10(np.maximum(el.sum(1) + er.sum(1), 1e-30))
    carry = c["side"] | is3[:, None]
    out = dict(frames=int(len(is3)), sap3_frames=int(is3.sum()))
    for name, X in (("ild", ild), ("corr", rho)):
        sw, ns = [], []
        for t in range(1, len(is3)):
            okb = carry[t] & carry[t - 1]
            d = np.abs(X[t] - X[t - 1])[okb]
            d = d[np.isfinite(d)]
            if is3[t] != is3[t - 1]:
                sw.extend(d)
            elif not is3[t]:
                ns.extend(d)
        out[f"continuity_{name}"] = (float(np.median(sw) / np.median(ns))
                                     if sw and ns and np.median(ns) > 0
                                     else None)
    steps = [lev[t] - lev[t - 1] for t in range(1, len(is3))
             if is3[t] and not is3[t - 1]]
    steps += [lev[t - 1] - lev[t] for t in range(1, len(is3))
              if is3[t - 1] and not is3[t]]
    out["level_step_db"] = float(np.mean(steps)) if steps else None
    out["level_step_n"] = len(steps)
    top = c["top"]
    with np.errstate(all="ignore"):
        core = np.where((top[:, 0] > 0) & (top[:, 1] > 0),
                        10 * np.log10(top[:, 0] / top[:, 1]), np.nan)
    sel = is3 & np.isfinite(core) & np.isfinite(c["hf"])
    if sel.sum() >= 10 and core[sel].std() > 0:
        x, y = c["hf"][sel], core[sel]
        r = float(np.corrcoef(x, y)[0, 1])
        out["aspx_ref_r"] = r
        out["aspx_ref_scale"] = float(np.sign(r) * y.std() / x.std())
        out["aspx_ref_n"] = int(sel.sum())
    else:
        out["aspx_ref_r"] = out["aspx_ref_scale"] = None
        out["aspx_ref_n"] = int(sel.sum())
    return out


def report(frames, element, pair=0):
    return {v: score(collect(frames, element, pair, v)) for v in VARIANTS}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        fr = mp4.read_frames(path)
        el = probe_element(fr)
        n_pairs = 1
        if el == "5_X":
            info = frameinfo.walk(fr[:40], el)
            n_pairs = 1 if any(r.get("codec_mode") == 4 for r in info) else 2
        for p in range(n_pairs):
            rep = report(fr, el, p)
            for v, row in rep.items():
                print(json.dumps(dict(pair=p, variant=v,
                                      **{k: (round(x, 3) if isinstance(x, float)
                                             else x) for k, x in row.items()})),
                      flush=True)
