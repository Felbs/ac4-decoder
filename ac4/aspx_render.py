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
from .robust import CONCEALED                                        # noqa: E402

FS = 48000

# THE TIME GRID OF THE QMF TOOLS (A-SPX, A-CPL, companding): frame f's
# A-SPX interval slot ts (atsg borders x num_ts_in_ats) acts on QMF
# analysis slot 24 f + TS_OFFSET + ts of the decoded core.
#
# TS 103 190-1 Pseudocode 75 feeds the HF generator Q_low[ts] =
# Q[ts - ts_offset_hfgen] (6 slots for frame lengths 1536 to 2048, Table
# 192), and 5.7.6.5 outputs Qin(i - ts_offset_hfgen) + Y(i): read
# literally, slot ts of frame f's interval is analysis slot 24 f + ts - 6.
# On air the data of frame f fits the core one frame later, at
# 24 f + ts - 6 + 24: the envelope-to-envelope swing of the envelope
# adjuster's E_ref / E_est (it jumps wherever envelope and core are
# misaligned) is smallest at offsets 16 to 18 on both broadcasters and a
# public test stream, and largest at -6 (c/tests/... see README,
# Interpretations); the A-SPX transient borders lead the core's own
# transients by 12-13 slots at the old offset 4, i.e. 16-17. So
# TS_OFFSET = 24 - ts_offset_hfgen = 18. (Was 4: an impulse measurement
# of the analysis filter's own delay.)
TS_HFGEN = 6
TS_OFFSET = 24 - TS_HFGEN


# COMPANDING (TS 103 190-1 5.7.5). The tool's levels L(ts) are not scale
# invariant (its gain is L^((1 - alpha) / alpha) * 2^(1 / alpha)), and the
# standard does not say what full scale they refer to. Its fixed point,
# L = 2^(-1 / (1 - alpha)) = 0.138, sits at programme level only for a
# full scale of 1, so the levels are taken relative to this decoder's full
# scale: IMDCT_K * 2^14.5 decoder units, the full scale the loudness
# metadata of five streams gives (BS.1770 against loudrelgat: 122.1 dB re
# 1, spread +-1.8 dB; IMDCT_K * 2^14.5 is 122.2 dB). See README,
# Interpretations.
COMPANDING = False      # off: see README, Interpretations (companding)
COMPAND_ALPHA = 0.65
COMPAND_SCALE = FB.IMDCT_K * 2.0 ** 14.5


def compand_levels(Q, sb0, sb1):
    """L(ts) = 0.9105 * mean over sb0..sb1-1 of max(|Re|, |Im|) + min / 2."""
    re, im = np.abs(Q[sb0:sb1].real), np.abs(Q[sb0:sb1].imag)
    e = np.maximum(re, im) + 0.5 * np.minimum(re, im)
    return 0.9105 * e.mean(axis=0) / COMPAND_SCALE


def compand(Qs, groups, sbx, ts_offset=None):
    """The companding tool (5.7.5.2) on the analysed core of the channels
    of one A-SPX group, in place, before HF generation (6.2.9). Qs: the
    channels' QMF matrices (list); groups: per frame the A-SPX data (one
    dict per channel, with its "compand" flags) or None / CONCEALED. Each
    frame acts on subbands 0 .. sbx - 1 and on the slots of its A-SPX
    interval, atsg_sig[0] .. atsg_sig[num_atsg_sig] (x num_ts_in_ats),
    placed like the envelopes."""
    off = TS_OFFSET if ts_offset is None else ts_offset
    a = COMPAND_ALPHA
    G = 2.0 ** (1.0 / a)
    expo = (1.0 - a) / a
    for f, got in enumerate(groups):
        if not isinstance(got, list):
            continue
        flags = [c.get("compand") for c in got]
        if any(fl is None for fl in flags):
            continue
        sync = flags[0]["sync"]
        spans, Ls = [], []
        for c, Q in zip(got, Qs):
            bd = c["fr"].get("borders") or [0, 12]
            ta = max(f * 24 + int(bd[0]) * 2 + off, 0)
            tz = min(f * 24 + int(bd[-1]) * 2 + off, Q.shape[1])
            spans.append((ta, tz))
            Ls.append(compand_levels(Q[:, ta:tz], 0, sbx) if tz > ta
                      else np.zeros(0))
        if sync:
            n = min(len(x) for x in Ls)
            if n == 0:
                continue
            gs = np.mean([x[:n] ** expo for x in Ls], axis=0)
            if flags[0]["on"]:
                gains = [gs] * len(Qs)
            elif flags[0]["avg"]:
                gains = [np.full(n, gs.mean())] * len(Qs)
            else:
                continue
            for (ta, tz), Q, g in zip(spans, Qs, gains):
                Q[:sbx, ta:ta + n] *= g * G
            continue
        for (ta, tz), Q, L, fl in zip(spans, Qs, Ls, flags):
            if tz <= ta:
                continue
            if fl["on"]:
                g = L ** expo
            elif fl["avg"]:
                g = np.full(tz - ta, L.mean() ** expo)
            else:
                continue
            Q[:sbx, ta:tz] *= g * G


_T_CACHE = {}


def t_for(cfg):
    key = (cfg["start_freq"], cfg["stop_freq"], cfg["master_freq_scale"],
           cfg["noise_sbg"])
    if key not in _T_CACHE:
        _T_CACHE[key] = A.sbg_tables(cfg, 0)
    return _T_CACHE[key]


def apply_hf_pair(pcmL, pcmR, groups, cfg, state0=None, state_log=None,
                  acpl_groups=None, acpl_state0=None, acpl_log=None):
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

    acpl_groups (ASPX_ACPL_3, one entry per group, see acpl.render): the
    A-CPL upmix runs on the A-SPX output in the QMF domain (TS 103 190-1
    6.2.11) and out["up"] receives its five channels, synthesised (L, R,
    C, Ls, Rs); out["L"], out["R"] stay the A-SPX output before the upmix
    (for the decoder's level checks). acpl_state0 / acpl_log as state0 /
    state_log, for the A-CPL state.
    """
    if cfg is None:
        # no A-SPX configuration yet: the pair only goes through the QMF
        # bank (analysis, synthesis), which is the delay every channel of
        # the decoder has (TS 103 190-1 6.2.8, 6.2.14)
        w = Q.qwin()
        if state_log is not None:             # the state is untouched
            state_log.extend([state0] * (len(groups) + 1))
        out = {}
        Qa = {}
        for k, x in (("L", pcmL), ("R", pcmR)):
            xx = x[:(len(x) // Q.NSB) * Q.NSB]
            Qa[k] = Q.analyse(xx, w, Q.analysis_matrix())
            out[k] = Q.synthesise(Qa[k], w, Q.synthesis_matrix(255))
        if acpl_groups is not None:
            out["up"] = _upmix(Qa["L"], Qa["R"], acpl_groups, acpl_state0,
                               acpl_log, w)
        return out, 0
    t = t_for(cfg)
    lim = E.sbg_lim(t)
    nt = E.noise_table()
    w = Q.qwin()
    maps = V.sbg_maps(t)
    n_sb, sbx = t["num_sb_aspx"], t["sbx"]

    out = {}
    Qh = {}
    Qan = {}
    for k, x in (("L", pcmL), ("R", pcmR)):
        xx = x[:(len(x) // Q.NSB) * Q.NSB]
        Qan[k] = Q.analyse(xx, w, Q.analysis_matrix())
    if COMPANDING:
        compand([Qan["L"], Qan["R"]], groups, sbx)
    for k in ("L", "R"):
        Qa = Qan[k]
        Qh[k] = G.hf_generate(Qa, t)
        # HF generation only on the QMF slots of frames that have A-SPX
        # data (or are concealed, whose high band is muted below): slot ts
        # carries frame (ts - TS_OFFSET) // 24. The others -- frames from
        # before the stream's first aspx_config, and the TS_OFFSET slots
        # before the first frame -- keep the analysed core and nothing
        # above it; patched there, they would play the copied low band
        # unadjusted (up to 0.88 of full scale was measured).
        f = (np.arange(Qh[k].shape[1]) - TS_OFFSET) // 24
        has = np.array([g is not None for g in groups] + [False], bool)
        keep = ~has[np.where((f >= 0) & (f < len(groups)), f, len(groups))]
        if keep.any():
            Qh[k][:, keep] = Qa[:, keep]

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
        if got is CONCEALED:
            # a concealed frame: no high band for its slots (its core is
            # the faded overlap tail); the envelope state is held as it was
            for k in ("L", "R"):
                ta = max(slot + TS_OFFSET, 0)
                tz = min(slot + 24 + TS_OFFSET, Qh[k].shape[1])
                if tz > ta:
                    Qh[k][sbx:sbx + n_sb, ta:tz] = 0.0
            # the noise table runs on as if the frame had been decoded
            # (one frame = 24 QMF slots), so the frames after it pick up
            # the noise sequence where a clean decode would be
            idx_noise = (idx_noise + n_sb * 24) % 512
            slot += 24
            continue
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
            # Pseudocodes 82, 84: 1.5 dB steps (aspx_qmode_env 0) -> a = 2
            a0 = E.env_a(got[0].get("qmode", 1))
            a1 = E.env_a(got[1].get("qmode", 1))
            if got[0].get("balance") == 1:
                sigA, sigB = E.map_scf_joint(qa, qb, t, fr0, n_sb, a=a0)
                nsA, nsB = E.map_noise_joint(na, nb, t, n_sb, sigA.shape[1])
            else:
                sigA = E.map_scf(qa, t, fr0, n_sb, a=a0)
                sigB = E.map_scf(qb, t, got[1]["fr"], n_sb, a=a1)
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
        for k, sig, ns, bd, frk in (("L", sigA, nsA, b0, got[0]["fr"]),
                                    ("R", sigB, nsB, b1, got[1]["fr"])):
            est = E.estimate(Qh[k], t, bd, slot, 2, ts_off=TS_OFFSET)
            if est.shape != sig.shape:
                ok = False
                break
            # Pseudocodes 95 and 99: the envelope at aspx_tsg_ptr (and at
            # p_sine_at_end) takes no noise factor in its gain or boost
            g, nl = E.adjust(Qh[k], t, sig, est, lim, ns[:, :est.shape[1]],
                             E.transient_envelopes(frk, est.shape[1]))
            t0 = slot + int(bd[0]) * 2 + TS_OFFSET     # atsg_sig[0], slots
            for e in range(est.shape[1]):
                ta = slot + int(bd[e]) * 2 + TS_OFFSET
                tz = slot + int(bd[e + 1]) * 2 + TS_OFFSET
                ta, tz = max(ta, 0), min(tz, Qh[k].shape[1])
                if tz <= ta:
                    continue
                Qh[k][sbx:sbx + n_sb, ta:tz] *= g[:, e:e + 1]
                # Pseudocode 103: noise_idx(sb, ts) = noise_idx_prev
                # + num_sb_aspx * (ts - atsg_sig[0]) + sb + 1 -- counted
                # from the start of the frame's first envelope, so the
                # envelopes of one frame take successive noise samples
                for ts in range(ta, tz):
                    kk = (idx_noise + n_sb * (ts - t0) + np.arange(n_sb)
                          + 1) % 512
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
    if acpl_groups is not None:
        out["up"] = _upmix(Qh["L"], Qh["R"], acpl_groups, acpl_state0,
                           acpl_log, w)
    return out, applied


def _upmix(QL, QR, acpl_groups, state0, log, w):
    """A-CPL (acpl.render) on the pair's QMF matrices, then synthesis of
    the five channels."""
    from . import acpl as ACPL
    up = ACPL.render(QL, QR, acpl_groups, state0=state0, state_log=log,
                     ts_offset=TS_OFFSET)
    return {k: Q.synthesise(v, w, Q.synthesis_matrix(255))
            for k, v in up.items()}


def band(x, lo, hi):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    m = (f >= lo) & (f < hi)
    return float((X[m] ** 2).sum()), float((X ** 2).sum())
