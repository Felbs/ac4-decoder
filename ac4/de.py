# SPDX-License-Identifier: Apache-2.0
"""Dialogue enhancement (TS 103 190-1 5.7.8), parametric channel
independent enhancement (de_method 0, 5.7.8.7), applied in the QMF domain
at a user gain G_DE in dB (0 = off, the default: the tool is then not run
at all and the output is bit-exact the decode without it).

Per frame f the enhancement matrix of the processed front channels (L, R,
C by de_channel_config, Table 171) is diagonal, 1 + g p_i(b) for channel
i in parameter band b (Tables 173, 209), with g = 10^(min(G_DE, G_max) /
20) - 1 and G_max = (de_max_gain + 1) 3 dB (4.3.14.3.2); with
de_ms_proc_flag the one parameter set acts on the mid signal of the two
channels: (1/2) [1 1; 1 -1] diag(1 + g p0, 1) [1 1; 1 -1]. Between frames
the matrix is interpolated per QMF slot n of the frame's N:
H(n) = (1 - (n + 1/2) / N) H(f - 1) + (n + 1/2) / N H(f) (5.7.8.6).

Readings (README, Interpretations): frame f's matrix acts on the QMF slots
of frame f's A-SPX / A-CPL interval (aspx_render.TS_OFFSET); QMF subbands
above Table 173's last (40) take parameter band 7; a frame without DE
data, or whose method is not 0, has the identity (H = I); a concealed
frame holds the previous frame's matrix.

Phase D: methods 1 - 3. Cross-channel enhancement (de_method 1, 5.7.8.8):
Y = (I + g r p^T) m over the processed channels, p from Table 210 (the
indices x 0.1, -3 .. 3), r the rendering vector of 5.7.8.5 from the
de_mix_coef indices (Table 172; the last coefficient energy preserving,
sqrt(1 - sum of the others' squares), 1 for one channel). The hybrid
methods 2 and 3 are run on their parametric data alone, as 5.7.8.1
allows a low-complexity decoder (with g as in 5.7.8.7 / 5.7.8.8): the
waveform-coded dialogue d_c is not located in the standards at hand.
Readings: a cross-channel matrix whose processed channels are not all
in one render call (L / R and C in separate A-SPX pairs, e.g. a 5_X
element in codec mode ASPX) is not applied (identity); a cross-channel
frame without de_mix_coef ever sent (two or three channels) has the
identity."""
from __future__ import annotations

import numpy as np

# Table 173: parameter band -> first QMF band (the last band runs on to 63)
BAND_FIRST = (0, 1, 2, 4, 7, 11, 17, 27)
SB_TO_BAND = np.array([max(i for i, f in enumerate(BAND_FIRST) if f <= sb)
                       for sb in range(64)])
# Table 209 (channel independent)
PAR_CI = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                   1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0,
                   4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
# Table 172
MIX_COEF = (0.0, 6.32e-3, 1e-2, 1.79e-2, 3.16e-2, 5.65e-2, 7.87e-2, 0.111,
            0.156, 0.218, 0.303, 0.37, 0.448, 0.533, 0.577, 0.622, 0.7071,
            0.783, 0.846, 0.894, 0.929, 0.953, 0.976, 0.9877, 0.9938,
            0.9969, 0.9984, 0.9995, 0.99984, 0.99995, 0.99998, 1.0)
# de_channel_config bits (Table 171): 4 Left, 2 Right, 1 Centre
CH_BITS = (("L", 4), ("R", 2), ("C", 1))

HOLD = "hold"


def frame_gains(de, level_db):
    """One frame's matrix as {channel: (64,) diagonal gain}, plus "ms":
    the (64,) mid gain when de_ms_proc_flag is set (then L and R are not
    in the dict). de: the stream's DE entry (config, par, ms) or None."""
    if de is None or level_db == 0:
        return {}
    cfg = de["config"]
    if not de.get("par"):
        return {}
    gmax = (cfg["max_gain"] + 1) * 3.0
    g = 10.0 ** (min(level_db, gmax) / 20.0) - 1.0
    chans = [k for k, bit in CH_BITS if cfg["channel_config"] & bit]
    par = de["par"]
    out = {}
    if cfg["method"] in (1, 3):
        return _cross(chans, par, de.get("mix"), g)

    def gain(row):
        idx = np.clip(np.asarray(row[:8], int), 0, len(PAR_CI) - 1)
        return 1.0 + g * PAR_CI[idx][SB_TO_BAND]

    if de.get("ms") and len(chans) == 2:
        out["ms"] = gain(par[0])
        out["ms_pair"] = tuple(chans)
        return out
    for i, k in enumerate(chans):
        out[k] = gain(par[i])
    return out


def _cross(chans, par, mix, g):
    """5.7.8.8: {"xc": H (n, n, 64), "xc_ch": the processed channels}."""
    n = len(chans)
    if n == 0:
        return {}
    if n == 1:
        r = [1.0]
    else:
        if not mix or len(mix) < n - 1:
            return {}
        c = [MIX_COEF[min(max(int(v), 0), 31)] for v in mix[:n - 1]]
        rest = 1.0
        for v in c:
            rest = rest - v * v
        r = c + [float(np.sqrt(max(0.0, rest)))]
    p = [np.clip(np.asarray(par[i][:8], int), -30, 30)[SB_TO_BAND] * 0.1
         for i in range(n)]
    H = np.zeros((n, n, 64))
    for a in range(n):
        for b in range(n):
            H[a, b] = (1.0 if a == b else 0.0) + g * r[a] * p[b]
    return {"xc": H, "xc_ch": tuple(chans)}


def _mat(m, names):
    """A frame's matrix over the channels `names` (n, n, 64), or the
    identity when it is cross-channel on channels not all in names."""
    n = len(names)
    H = np.zeros((n, n, 64))
    if "xc" in m:
        xc = m["xc_ch"]
        if not all(k in names for k in xc):
            for i in range(n):
                H[i, i] = 1.0
            return H
        for i in range(n):
            H[i, i] = 1.0
        for a, ka in enumerate(xc):
            for b, kb in enumerate(xc):
                H[names.index(ka), names.index(kb)] = m["xc"][a, b]
        return H
    f = _full(m, names)
    pair = m.get("ms_pair", ())
    for i, k in enumerate(names):
        H[i, i] = f[k][0]
        if k in pair:
            other = pair[1] if k == pair[0] else pair[0]
            if other in names:
                H[i, names.index(other)] = f[k][1]
    return H


def _full(m, names):
    """A frame's matrix as per-channel diagonal gains and an optional M/S
    block: -> dict name -> (a, b) where channel name's output is
    a * x_name + b * x_other (x_other the M/S partner), (64,) each."""
    res = {}
    ms = m.get("ms")
    pair = m.get("ms_pair", ())
    for k in names:
        if ms is not None and k in pair:
            # 1/2 [1 1; 1 -1] diag(G, 1) [1 1; 1 -1]: own (G + 1) / 2,
            # partner (G - 1) / 2
            res[k] = ((ms + 1.0) / 2.0, (ms - 1.0) / 2.0)
        else:
            res[k] = (m.get(k, np.ones(64)), np.zeros(64))
    return res


def apply(Qs, frames, level_db, nts, ts_offset, prev=None):
    """Apply DE in place to the QMF matrices Qs (dict name -> (64, n_slots)
    complex; names among L, R, C). frames: per frame the stream's DE entry
    (None: no data, HOLD: concealed); frame i acts on slots nts i +
    ts_offset .. + nts. prev: the matrix in force before frames[0]. ->
    the matrix in force after the last frame."""
    if level_db == 0:
        return prev
    names = [k for k in Qs]
    m_prev = prev if prev is not None else {}
    n_slots = next(iter(Qs.values())).shape[1]
    w = (np.arange(nts) + 0.5) / nts
    for i, de in enumerate(frames):
        m_cur = m_prev if de is HOLD else frame_gains(de, level_db)
        a = nts * i + ts_offset
        z = min(a + nts, n_slots)
        if z > max(a, 0) and ("xc" in m_cur or "xc" in m_prev):
            # a cross-channel matrix: the general form (y_k = h_kk x_k,
            # then + h_kj x_j for the other channels in order)
            h0, h1 = _mat(m_prev, names), _mat(m_cur, names)
            lo = max(a, 0)
            ww = w[lo - a:z - a][None, :]
            src = {k: Qs[k][:, lo:z].copy() for k in names}
            for i, k in enumerate(names):
                y = ((1 - ww) * h0[i, i][:, None]
                     + ww * h1[i, i][:, None]) * src[k]
                for j, kj in enumerate(names):
                    if j != i:
                        y = y + ((1 - ww) * h0[i, j][:, None]
                                 + ww * h1[i, j][:, None]) * src[kj]
                Qs[k][:, lo:z] = y
        elif z > max(a, 0) and (m_cur or m_prev):
            f0, f1 = _full(m_prev, names), _full(m_cur, names)
            lo = max(a, 0)
            ww = w[lo - a:z - a][None, :]
            src = {k: Qs[k][:, lo:z].copy() for k in names}
            pair = m_cur.get("ms_pair") or m_prev.get("ms_pair") or ()
            for k in names:
                own = (1 - ww) * f0[k][0][:, None] + ww * f1[k][0][:, None]
                y = own * src[k]
                if k in pair:
                    other = pair[1] if k == pair[0] else pair[0]
                    if other in src:
                        cr = ((1 - ww) * f0[k][1][:, None]
                              + ww * f1[k][1][:, None])
                        y = y + cr * src[other]
                Qs[k][:, lo:z] = y
        m_prev = m_cur
    return m_prev
