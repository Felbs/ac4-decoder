# SPDX-License-Identifier: Apache-2.0
"""M37 -- render WITH the A-SPX high band: 0..21 kHz instead of 0..12 kHz.

Everything up to here produced either core audio band-limited at 12 kHz (m29)
or high-band stages gated in the QMF domain without writing a file (m35, m36).
This wires them together and writes audio that actually contains 12..21 kHz.

THE CHAIN
----------
    1. decode the MDCT core at NATIVE scale          m28 + m29 machinery
    2. QMF analyse                                    m33
    3. HF generate -- patch low subbands upward       m35
    4. envelope adjust + noise, per frame             m36
    5. QMF synthesise                                 m33
    6. write

Step 1 must be at native scale: m29's own renderer peak-normalises, which
destroys the absolute level the A-SPX envelopes are defined against and makes
the gain saturate (see M36).

THE GATES
----------
  1. energy APPEARS in 12..21 kHz, where the core render has none;
  2. energy does NOT appear above 21 kHz -- A-SPX stops at QMF subband 56, so
     a correct high band is bounded there and a wrong one is not;
  3. below 12 kHz the output still matches the core render closely -- the
     high-band work must not disturb the part that was already verified.

Gate 3 matters: it is easy to produce a "fuller" file by breaking the core.

ALIGNMENT -- MEASURED, NOT ASSUMED
-----------------------------------
The control data for frame i has to be applied to the QMF slots that actually
carry frame i's audio, and that is two delays deep.  Both were measured with
impulses rather than reasoned about:

    MDCT render delay   0 frames.  A spectrum placed in frame 6 alone peaks in
                        output frame 6 (spreading into 7 through the overlap),
                        so the render is frame-aligned.
    QMF analysis delay  4.0 slots.  An impulse at input slot 10 has its
                        subband-energy centroid at slot 14.0.

So TS_OFFSET = 4.  The first version of this file used 0 and smeared the high
band ~4 slots (~5.3 ms) early against the core.  Two statistical attempts to
find the offset by correlating the transmitted envelope against the core's
energy both failed first -- one gave a broad monotone curve with no peak, the
other (differencing to sharpen it) gave noise, because the transmitted envelope
is piecewise CONSTANT and differencing annihilates it.  A deterministic impulse
measurement settled in one step what two statistical ones could not.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from . import toc as M                                          # noqa: E402
from . import hcb as H                                               # noqa: E402
from . import spectral as S                                          # noqa: E402
from . import channels as C                                          # noqa: E402
from . import audio as R                                             # noqa: E402
from . import filterbank as FB                                       # noqa: E402
from . import aspx_bands as A                                        # noqa: E402
from . import aspx_parse as P                                        # noqa: E402
from . import qmf as Q                                               # noqa: E402
from . import aspx_env as V                                          # noqa: E402
from . import aspx_hfgen as G                                             # noqa: E402
from . import aspx_envadj as E                                            # noqa: E402

FS = 48000

# QMF ANALYSIS DELAY, MEASURED not assumed: an impulse at input slot 10 has its
# subband-energy centroid at slot 14.0, so frame i's audio occupies QMF slots
# [i*24 + TS_OFFSET, ...).  The MDCT render itself was measured to add ZERO
# frames of delay (frame 6's spectrum peaks in output frame 6), so this is the
# whole correction.  Applying the control at offset 0 -- the previous default --
# smears the high band ~4 slots (~5.3 ms) early against the core.
TS_OFFSET = 4


_T_CACHE = {}


def t_for(cfg):
    key = (cfg["start_freq"], cfg["stop_freq"], cfg["master_freq_scale"],
           cfg["noise_sbg"])
    if key not in _T_CACHE:
        _T_CACHE[key] = A.sbg_tables(cfg, 0)
    return _T_CACHE[key]


def apply_hf_pair(pcmL, pcmR, groups, cfg, state0=None, state_log=None):
    """Steps 2-5 for the L/R pair, JOINTLY.

    Three things run on from one frame to the next: the noise-table index
    and the two delta-coded envelopes (signal and noise, per channel).
    `state_log` (a list) receives that state as it stands at the start of
    every group, plus once more after the last; `state0` starts the call
    from a logged state instead of from nothing.  A caller that renders a
    stream in several calls passes the state back so the second call
    continues the first; without it every call restarts the noise table at
    0 and decodes its first TIME-coded envelopes against zeros (see
    the original receiver's audio worker for what that measured).

    The pair cannot be processed one channel at a time: when aspx_balance = 1
    -- 89.7 % of frames here -- channel B carries a PAN RATIO and its scale
    factors only become levels through Pseudocode 84, which needs both
    channels' indices at once.  Treating B's values as levels (Pseudocode 82,
    which explicitly covers only balance = 0) made the right channel 20x too
    loud and uncorrelated with its own core, while the left channel looked
    perfect -- so a single-channel gate could never have caught it.
    """
    t = t_for(cfg)
    lim = E.sbg_lim(t)
    nt = E.noise_table()
    w = Q.qwin()
    maps = V.sbg_maps(t)
    n_sb, sbx = t["num_sb_aspx"], t["sbx"]

    out = {}
    Qh = {}
    for k, x in (("L", pcmL), ("R", pcmR)):
        xx = x[:(len(x) // Q.NSB) * Q.NSB]
        Qh[k] = G.hf_generate(Q.analyse(xx, w, Q.analysis_matrix()), t)

    prev = {0: (None, 0), 1: (None, 0)}
    nprev = {0: None, 1: None}
    idx_noise = 0
    if state0 is not None:
        idx_noise, (prev[0], prev[1]), (nprev[0], nprev[1]) = state0
    slot = 0
    applied = 0
    for got in groups:
        if state_log is not None:
            state_log.append((idx_noise, (prev[0], prev[1]),
                              (nprev[0], nprev[1])))
        if got is None or len(got) < 2:
            slot += 24
            continue
        try:
            qa, prev[0] = V.reconstruct(got[0], t, maps, prev[0])
            qb, prev[1] = V.reconstruct(got[1], t, maps, prev[1])
            na, nprev[0] = E.noise_qscf(got[0], t, nprev[0])
            nb, nprev[1] = E.noise_qscf(got[1], t, nprev[1])
            fr0 = got[0]["fr"]
            borders = fr0.get("borders") or [0, 12]
            if got[0].get("balance") == 1:
                sigA, sigB = E.map_scf_joint(qa, qb, t, fr0, n_sb)
                nsA, nsB = E.map_noise_joint(na, nb, t, n_sb, sigA.shape[1])
            else:
                sigA = E.map_scf(qa, t, fr0, n_sb)
                sigB = E.map_scf(qb, t, got[1]["fr"], n_sb)
                dqa = [[2.0 ** (E.NOISE_FLOOR_OFFSET - v) for v in e]
                       for e in na]
                dqb = [[2.0 ** (E.NOISE_FLOOR_OFFSET - v) for v in e]
                       for e in nb]
                nsA = E.map_noise(dqa, t, n_sb, sigA.shape[1])
                nsB = E.map_noise(dqb, t, n_sb, sigB.shape[1])
        except Exception:                                      # noqa: BLE001
            slot += 24
            continue
        # WITH balance = 0 THE TWO CHANNELS HAVE INDEPENDENT FRAMINGS, so each
        # gets its OWN borders.  Using channel 0's for both silently skipped
        # 114 of 3480 frames (they failed a shape check and fell through to
        # core-only), which the per-frame data diagnostic could not see because
        # the data was fine -- the renderer was wrong.
        b0 = borders
        b1 = (borders if got[0].get("balance") == 1
              else (got[1]["fr"].get("borders") or borders))
        ok = True
        for k, sig, ns, bd in (("L", sigA, nsA, b0), ("R", sigB, nsB, b1)):
            est = E.estimate(Qh[k], t, bd, slot, 2, ts_off=TS_OFFSET)
            if est.shape != sig.shape:
                ok = False
                break
            g, nl = E.adjust(Qh[k], t, sig, est, lim, ns[:, :est.shape[1]])
            for e in range(est.shape[1]):
                ta = slot + int(bd[e]) * 2 + TS_OFFSET
                tz = slot + int(bd[e + 1]) * 2 + TS_OFFSET
                ta, tz = max(ta, 0), min(tz, Qh[k].shape[1])
                if tz <= ta:
                    continue
                Qh[k][sbx:sbx + n_sb, ta:tz] *= g[:, e:e + 1]
                base = idx_noise if k == "L" else idx_noise
                for ts in range(ta, tz):
                    kk = (base + n_sb * (ts - ta) + np.arange(n_sb) + 1) % 512
                    Qh[k][sbx:sbx + n_sb, ts] += nl[:, e] * nt[kk]
        if ok:
            applied += 1
            span = sum(max(0, min(slot + int(borders[e + 1]) * 2 + TS_OFFSET,
                                  Qh["L"].shape[1])
                           - max(slot + int(borders[e]) * 2 + TS_OFFSET, 0))
                       for e in range(len(borders) - 1))
            idx_noise = (idx_noise + n_sb * span) % 512
        slot += 24

    if state_log is not None:
        state_log.append((idx_noise, (prev[0], prev[1]),
                          (nprev[0], nprev[1])))
    for k in ("L", "R"):
        out[k] = Q.synthesise(Qh[k], w, Q.synthesis_matrix(255))
    return out, applied


def band(x, lo, hi):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    m = (f >= lo) & (f < hi)
    return float((X[m] ** 2).sum()), float((X ** 2).sum())
