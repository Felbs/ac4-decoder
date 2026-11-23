# SPDX-License-Identifier: Apache-2.0
"""The A-SPX tools beyond the patch copy (TS 103 190-1 5.7.6.4, 5.7.6.5):

    pre-flattening            Pseudocode 85 (aspx_preflat)
    tonal to noise adjustment Pseudocodes 86-88, Table 195 (aspx_tna_mode)
    HF signal creation        Pseudocode 89
    estimation, interpolation 0  Pseudocode 90 (per subband group)
    additional sinusoids      Pseudocodes 92-94, 104, 105, Table 196
                              (aspx_add_harmonic)
    interleaved waveform      5.7.6.5.3 (aspx_fic_used_in_sfb,
      coding                  aspx_tic_used_in_slot)

Times are QMF analysis slots of the decoder's matrices; an A-SPX
interval's slot ts (Q_low[ts], Q_high[ts]) is analysis slot A0 + ts with
A0 = num_qmf_timeslots f + TS_OFFSET (aspx_render, the phase A reading).
Q_low_ext[ts + ts_offset_hfadj] is Q_low[ts], so Q_low_ext[n - 2] is two
analysis slots back. The pseudocodes' loops run over whole subbands and
slots; they are written vectorised here and as plain loops in the C port.
"""
from __future__ import annotations

import numpy as np

TS_OFFSET_HFADJ = 4
# Table 195: new_chirp[aspx_tna_mode_prev][aspx_tna_mode]
NEW_CHIRP = ((0.0, 0.6, 0.9, 0.98),
             (0.6, 0.75, 0.9, 0.98),
             (0.0, 0.75, 0.9, 0.98),
             (0.0, 0.75, 0.9, 0.98))
EPSILON_INV = 2.0 ** -20
# Table 196
SINE_RE = (1.0, 0.0, -1.0, 0.0)
SINE_IM = (0.0, 1.0, 0.0, -1.0)


def new_state(n_sb=64):
    """One channel's tool state between A-SPX intervals: the previous
    chirp factors and TNA modes (Pseudocode 88: zero for the first
    interval), the previous interval's sine_idx_sb (last envelope, per
    A-SPX subband) and the tone generator's index (Pseudocode 105:
    first_frame)."""
    return dict(chirp=np.zeros(0), tna=[], sine_sb=np.zeros(n_sb, bool),
                sine_idx=None)


def copy_state(s):
    return dict(chirp=s["chirp"].copy(), tna=list(s["tna"]),
                sine_sb=s["sine_sb"].copy(), sine_idx=s["sine_idx"])


# ------------------------------------------------------------- generator
def tna_alpha(Qa, sba, a0, nts, hfgen):
    """Pseudocodes 86, 87: the linear prediction coefficients alpha0,
    alpha1 (complex, per QMF subband below sba) from the covariance of
    Q_low_ext over the interval's window (every other slot of Q_low,
    analysis slots a0 .. a0 + nts + hfgen - 1, lags of 2 and 4 slots;
    slots before the matrix are 0)."""
    ms = np.arange(0, nts + hfgen, 2)
    idx = a0 + ms
    n = Qa.shape[1]

    def lag(k):
        j = idx - k
        ok = (j >= 0) & (j < n)
        out = np.zeros((sba, len(j)), complex)
        out[:, ok] = Qa[:sba, j[ok]]
        return out
    x0, x2, x4 = lag(0), lag(2), lag(4)
    xs = (x0, x2, x4)

    def cov(i, j):
        return (xs[i] * np.conj(xs[j])).sum(axis=1)
    c01, c02, c11, c12, c22 = cov(0, 1), cov(0, 2), cov(1, 1), cov(1, 2), \
        cov(2, 2)
    denom = c22 * c11 - np.abs(c12) * np.abs(c12) * (1.0 / (1.0 + EPSILON_INV))
    a1 = np.zeros(sba, complex)
    nz = denom != 0
    a1[nz] = (c01[nz] * c12[nz] - c02[nz] * c11[nz]) / denom[nz]
    a0_ = np.zeros(sba, complex)
    nz1 = c11 != 0
    a0_[nz1] = (-c01[nz1] + a1[nz1] * np.conj(c12[nz1])) / c11[nz1]
    big = (np.abs(a0_) >= 4) | (np.abs(a1) >= 4)
    a0_[big] = 0
    a1[big] = 0
    return a0_, a1


def chirps(tna, st):
    """Pseudocode 88 per noise subband group; updates st (chirp, tna)."""
    n = len(tna)
    prev_c = st["chirp"] if len(st["chirp"]) == n else np.zeros(n)
    prev_t = st["tna"] if len(st["tna"]) == n else [0] * n
    out = np.zeros(n)
    for g in range(n):
        nc = NEW_CHIRP[prev_t[g]][tna[g]]
        if nc < prev_c[g]:
            nc = 0.75 * nc + 0.25 * prev_c[g]
        else:
            nc = 0.90625 * nc + 0.09375 * prev_c[g]
        out[g] = 0.0 if nc < 0.015625 else nc
    st["chirp"] = out
    st["tna"] = list(tna)
    return out


def polyfit3(x, y):
    """The least-squares cubic of Pseudocode 85 (polynomial_fit): the
    normal equations, solved by Gaussian elimination with partial
    pivoting. -> coefficients, highest power first."""
    n = len(x)
    A = np.zeros((4, 4))
    v = np.zeros(4)
    for r in range(4):
        for c in range(4):
            s = 0.0
            for i in range(n):
                s += float(x[i]) ** (r + c)
            A[r, c] = s
        s = 0.0
        for i in range(n):
            s += float(x[i]) ** r * float(y[i])
        v[r] = s
    for c in range(4):
        p = c + int(np.argmax(np.abs(A[c:, c])))
        if p != c:
            A[[c, p]] = A[[p, c]]
            v[[c, p]] = v[[p, c]]
        if A[c, c] == 0:
            continue
        for r in range(c + 1, 4):
            f = A[r, c] / A[c, c]
            for k in range(c, 4):
                A[r, k] = A[r, k] - f * A[c, k]
            v[r] = v[r] - f * v[c]
    coef = np.zeros(4)
    for r in range(3, -1, -1):
        s = v[r]
        for k in range(r + 1, 4):
            s = s - A[r, k] * coef[k]
        coef[r] = s / A[r, r] if A[r, r] != 0 else 0.0
    return coef[::-1]                  # x^3, x^2, x, 1


def preflat_gain(Qa, sbx, ta, tz):
    """Pseudocode 85: gain_vec per QMF subband below sbx, from the cubic
    fit to the low band's energy (dB) over the interval's slots
    [ta, tz)."""
    seg = Qa[:sbx, ta:tz]
    pw = np.zeros(sbx)
    for sb in range(sbx):
        e = 0.0
        for v in seg[sb]:
            e += v.real ** 2
            e += v.imag ** 2
        pw[sb] = 10.0 * np.log10(e / max(tz - ta, 1) + 1.0)
    mean = 0.0
    for sb in range(sbx):
        mean += pw[sb]
    mean /= sbx
    x = np.arange(sbx, dtype=float)
    poly = polyfit3(x, pw)
    slope = np.zeros(sbx)
    for k in range(3, -1, -1):
        for sb in range(sbx):
            slope[sb] += x[sb] ** k * poly[3 - k]
    return 10.0 ** ((mean - slope) / 20.0)


def generate(Qa, Qh, t, ta, tz, alpha0, alpha1, chirp, gain_vec):
    """Pseudocode 89 on analysis slots [ta, tz): Q_high from Q_low with
    the TNA filter and (gain_vec not None) the pre-flattening, into Qh."""
    sbx = t["sbx"]
    noise = t["noise"]
    if tz <= ta:
        return
    n = Qa.shape[1]
    sl = np.arange(ta, tz)
    prev2 = sl - 2
    prev4 = sl - 4
    sum_sb = 0
    g = 0
    for i, (num, st) in enumerate(zip(t["patch_num_sb"],
                                      t["patch_start_sb"])):
        for sb in range(num):
            hi = sbx + sum_sb + sb
            if g + 1 < len(noise) and noise[g + 1] == hi:
                g += 1
            p = st + sb
            x = Qa[p, ta:tz].copy()
            c = chirp[g] if g < len(chirp) else 0.0
            if c != 0.0:
                x2 = np.where(prev2 >= 0, Qa[p, np.clip(prev2, 0, n - 1)],
                              0)
                x4 = np.where(prev4 >= 0, Qa[p, np.clip(prev4, 0, n - 1)],
                              0)
                x = x + c * alpha0[p] * x2
                x = x + c ** 2 * alpha1[p] * x4
            if gain_vec is not None:
                x = x * (1.0 / gain_vec[p])
            Qh[hi, ta:tz] = x
        sum_sb += num


# ------------------------------------------------------------- sinusoids
def sine_idx_sb(fr, ah, t, prev_sb, n_sb):
    """Pseudocode 92: sine_idx_sb[sb][atsg] (n_sb x num_env, bool); the
    last envelope's column is the next interval's sine_idx_sb_prev."""
    n_env = len(fr["borders"]) - 1
    ptr = fr.get("tsg_ptr", -1)
    p_end = fr.get("sine_at_end", -1)
    out = np.zeros((n_sb, n_env), bool)
    hi, sbx = t["hi"], t["sbx"]
    for e in range(n_env):
        for g in range(t["n_hi"]):
            if not (g < len(ah) and ah[g]):
                continue
            a, z = hi[g] - sbx, hi[g + 1] - sbx
            mid = int(0.5 * (z + a))
            if (e >= ptr or p_end == 0
                    or (mid < len(prev_sb) and prev_sb[mid])):
                out[mid, e] = True
    return out


def sine_area(sidx, t, fr):
    """Pseudocode 93: per envelope, every subband of a subband group (of
    that envelope's resolution) that holds a sinusoid."""
    n_sb, n_env = sidx.shape
    out = np.zeros_like(sidx)
    sbx = t["sbx"]
    for e in range(n_env):
        tab = t["hi"] if fr["freq_res"][e] else t["lo"]
        for g in range(len(tab) - 1):
            a, z = tab[g] - sbx, tab[g + 1] - sbx
            if sidx[a:z, e].any():
                out[a:z, e] = True
    return out


def fic_mask(flags, t, n_sb):
    """aspx_fic_used_in_sfb (per high-resolution subband group) -> per
    A-SPX subband."""
    m = np.zeros(n_sb, bool)
    if flags:
        for g, f in enumerate(flags):
            if f and g + 1 < len(t["hi"]):
                m[t["hi"][g] - t["sbx"]:t["hi"][g + 1] - t["sbx"]] = True
    return m
