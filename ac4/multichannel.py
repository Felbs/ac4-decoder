# SPDX-License-Identifier: Apache-2.0
"""Multichannel coding tools of TS 103 190-1: the channel data elements of
more than two tracks (4.2.6.8 - 4.2.6.13, 5.3.3.3 - 5.3.3.5), the track to
channel mapping of the channel elements (5.3.4, Tables 180 - 184) and the
max_sfb_master side tracks of ASPX_ACPL_1 (4.3.5.13, Tables B.8 - B.19).

Every matrix of clause 5.3.3 is a cascade of the 2 x 2 stereo operation of
clause 5.3.3.2, [O0; O1] = [a b; c d] [I0; I1] (one per chparam_info), on
tracks that share one sf_info and so one packed layout:

  * three_channel_data (Table 178): pair 0 on two of the inputs gives
    (P, Q); pair 1 on one of P, Q and the third input gives (R, S); the
    outputs are three of P, Q, R, S. MATSEL3 lists, per chel_matsel, the
    inputs of both pairs and the outputs (checked against the table in
    tests/test_phase_b2.py).
  * four_channel_data: (P0, Q0) = pair0(I0, I1), (P1, Q1) = pair1(I2, I3),
    (O0, O2) = pair2(P0, P1), (O1, O3) = pair3(Q0, Q1).
  * five_channel_data (Table 179): (X, Y, Z) = the three_channel_data
    matrix of the same chel_matsel on I0..I2 with pairs 0 and 1,
    (U, V) = pair2(I3, I4), (O0, O3) = pair3(X, U), (O1, O4) = pair4(Y, V),
    O2 = Z.
"""
from __future__ import annotations

import copy

import numpy as np

from . import channels as C
from .robust import Damaged

# chel_matsel -> (pair 0 inputs, pair 1 inputs, outputs O0..O2); ints are
# the input tracks, "P" / "Q" pair 0's outputs, "R" / "S" pair 1's
MATSEL3 = {
    0: ((0, 1), ("P", 2), ("R", "Q", "S")),
    1: ((1, 0), ("P", 2), ("Q", "R", "S")),
    2: ((0, 2), ("P", 1), ("R", "S", "Q")),
    3: ((1, 2), (0, "Q"), ("R", "P", "S")),
    4: ((0, 2), (1, "Q"), ("P", "R", "S")),
    5: ((2, 1), (0, "Q"), ("R", "S", "P")),
    6: ((1, 0), (2, "Q"), ("S", "P", "R")),
    7: ((0, 1), (2, "Q"), ("P", "S", "R")),
    8: ((2, 0), (1, "Q"), ("S", "R", "P")),
    9: ((2, 1), ("P", 0), ("S", "Q", "R")),
    10: ((2, 0), ("P", 1), ("Q", "S", "R")),
    11: ((1, 2), ("P", 0), ("S", "R", "Q")),
}

# Tables B.8 - B.19: max_sfb_master at the largest signalled transform
# length -> the max_sfb of a block of another transform length
MAX_SFB_MASTER = {
    2048: {
        1024: (0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12,
            13, 13, 14, 15, 15, 16, 17, 18, 19, 19, 20, 21),
        512: (0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
            9, 10, 10, 11, 12, 13, 14, 15, 16, 17),
        256: (0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
            5, 5, 5, 6, 6, 7, 7, 8, 8, 8),
        128: (0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 4, 4, 4, 5, 5),
    },
    1024: {
        512: (0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
        256: (0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8,
            9, 9, 10, 11, 11, 12, 12, 13, 13, 14),
        128: (0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5,
            5, 6, 6, 6, 7, 7, 8, 8, 8, 9),
    },
    512: {
        256: (0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
            11, 11, 12, 12, 13, 13, 14, 15, 16, 17, 17),
        128: (0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 6,
            7, 7, 8, 8, 9, 9, 10, 10, 11, 12),
    },
    256: {
        128: (0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10),
    },
    1920: {
        960: (0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12,
            13, 13, 14, 15, 15, 16, 17, 18, 19, 19, 20, 21),
        480: (0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
            9, 10, 10, 11, 12, 13, 14, 15, 16, 17),
        240: (0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
            5, 5, 5, 6, 6, 7, 7, 8, 8, 8),
        120: (0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 4, 4, 4, 5, 5),
    },
    960: {
        480: (0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
        240: (0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8,
            9, 9, 10, 11, 11, 12, 12, 13, 13, 14),
        120: (0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5,
            5, 6, 6, 6, 7, 7, 8, 8, 8, 9),
    },
    480: {
        240: (0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
            11, 11, 12, 12, 13, 13, 14, 15, 16, 17, 17),
        120: (0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 6,
            7, 7, 8, 8, 9, 9, 10, 10, 11, 12),
    },
    240: {
        120: (0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10),
    },
    1536: {
        768: (0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12,
            13, 13, 14, 15, 15, 16, 17, 18, 19, 19, 20, 21),
        384: (0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
            9, 10, 10, 11, 12, 13, 14, 15, 16, 17),
        192: (0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
            5, 5, 5, 6, 6, 7, 7, 8, 8, 8),
        96: (0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 4, 4, 4, 5, 5),
    },
    768: {
        384: (0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27),
        192: (0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8,
            9, 9, 10, 11, 11, 12, 12, 13, 13, 14),
        96: (0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5,
            5, 6, 6, 6, 7, 7, 8, 8, 8, 9),
    },
    384: {
        192: (0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8),
        96: (0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4),
    },
    192: {
        96: (0, 1, 1, 2, 2, 3, 3, 4),
    },
}


def st_of(sap_mode, ms_used, sap):
    return dict(sap_mode=sap_mode, ms_used=ms_used, sap=sap)


NO_ST = dict(sap_mode=None, ms_used=None, sap=None)


def read_chparam(b, fr, T):
    return st_of(*C.chparam_info(b, fr, T["sf"]))


def chel_matsel(b):
    m = b.u(4)
    if m > 11:
        raise Damaged("asf.chel_matsel", f"{m} (reserved)")
    return m


def three_channel_data(b, T):
    """Table 27 -> (three tracks, the processing step)."""
    fr = C.Framing(b, C.frame_len(T))
    m = chel_matsel(b)
    sts = [read_chparam(b, fr, T) for _ in range(2)]
    tr = [C.sf_data(b, fr, T) for _ in range(3)]
    return tr, dict(kind="three", matsel=m, st=sts, framing=fr)


def four_channel_data(b, T):
    """Table 28."""
    fr = C.Framing(b, C.frame_len(T))
    sts = [read_chparam(b, fr, T) for _ in range(4)]
    tr = [C.sf_data(b, fr, T) for _ in range(4)]
    return tr, dict(kind="four", st=sts, framing=fr)


def five_channel_data(b, T):
    """Table 29."""
    fr = C.Framing(b, C.frame_len(T))
    m = chel_matsel(b)
    sts = [read_chparam(b, fr, T) for _ in range(5)]
    tr = [C.sf_data(b, fr, T) for _ in range(5)]
    return tr, dict(kind="five", matsel=m, st=sts, framing=fr)


def two_channel_step(st):
    """two_channel_data's own step (5.3.3.2)."""
    return dict(kind="pair", st=[st], framing=st.get("framing"))


# ------------------------------------------------------------ processing
def _pair(x, y, st, fr):
    from .audio import unmix
    if st is None or st.get("sap_mode") in (None, 0):
        return x, y
    return unmix(x, y, st, fr)


def matrix3(tr, m, st0, st1, fr):
    (i, j), (u, v), outs = MATSEL3[m]
    P, Q = _pair(tr[i], tr[j], st0, fr)
    env = {0: tr[0], 1: tr[1], 2: tr[2], "P": P, "Q": Q}
    R, S = _pair(env[u], env[v], st1, fr)
    env.update(R=R, S=S)
    return [env[k] for k in outs]


def matrix4(tr, sts, fr):
    P0, Q0 = _pair(tr[0], tr[1], sts[0], fr)
    P1, Q1 = _pair(tr[2], tr[3], sts[1], fr)
    O0, O2 = _pair(P0, P1, sts[2], fr)
    O1, O3 = _pair(Q0, Q1, sts[3], fr)
    return [O0, O1, O2, O3]


def matrix5(tr, m, sts, fr):
    X, Y, Z = matrix3(tr[:3], m, sts[0], sts[1], fr)
    U, V = _pair(tr[3], tr[4], sts[2], fr)
    O0, O3 = _pair(X, U, sts[3], fr)
    O1, O4 = _pair(Y, V, sts[4], fr)
    return [O0, O1, Z, O3, O4]


def apply_step(pk, keys, step):
    """Apply one processing step to the packed spectra pk[k] of the tracks
    `keys` (in bitstream order; replaced by the output channels, which the
    keys name: the track to channel mapping of Tables 180 - 184 is in the
    keys). Tracks of one step share one framing, so one packed length."""
    if not all(k in pk for k in keys):
        return
    fr = step["framing"]
    tr = [pk[k] for k in keys]
    n = max(len(x) for x in tr)
    tr = [x if len(x) == n else np.concatenate([x, np.zeros(n - len(x))])
          for x in tr]
    kind = step["kind"]
    if kind == "pair":
        out = list(_pair(tr[0], tr[1], step["st"][0], fr))
    elif kind == "three":
        out = matrix3(tr, step["matsel"], step["st"][0], step["st"][1], fr)
    elif kind == "four":
        out = matrix4(tr, step["st"], fr)
    else:
        out = matrix5(tr, step["matsel"], step["st"], fr)
    for k, v in zip(keys, out):
        pk[k] = v


# ---------------------------------------- max_sfb_master (ASPX_ACPL_1)
def largest_length(framings):
    return max(n for fr in framings for n in fr.lens)


def master_framing(fr, master, lmax):
    """The framing of a max_sfb_master track (4.3.5.13): fr's transform
    lengths and grouping; each block's max_sfb is max_sfb_master when its
    transform length is the largest signalled one (lmax), else the
    n_sfb_side of Tables B.8 - B.19 for its length."""
    f = copy.copy(fr)
    tab = MAX_SFB_MASTER.get(lmax, {})
    side = []
    for n in fr.lens:
        if n == lmax:
            side.append(master)
        else:
            col = tab.get(n)
            if col is None or master >= len(col):
                raise Damaged("asf.max_sfb", f"master {master} at {n}")
            side.append(col[master])
    for i, m in enumerate(side):
        if m > len(C.SFB_TABLE[fr.lens[i]]) - 1:
            raise Damaged("asf.max_sfb", f"{m}")
    f.max_sfb_side = side
    f.side = True
    return f


def residual_tracks(b, T, frA, frB, lmax):
    """max_sfb_master, two chparam_info, two sf_data (Table 25, Table 33):
    the side tracks of the A and B tracks. -> ((sA, stA), (sB, stB)).

    Readings: each side track takes the transform lengths and grouping of
    the track it is paired with (A, B), and its chparam_info is read with
    that track's own max_sfb (as the channel_pair_element reads its
    chparam_info with max_sfb, not max_sfb_side, Pseudocode 5)."""
    master = b.u(C.N_SIDE_BY_LEN[lmax])
    stA = read_chparam(b, frA, T)
    stB = read_chparam(b, frB, T)
    sA = C.sf_data(b, master_framing(frA, master, lmax), T)
    sB = C.sf_data(b, master_framing(frB, master, lmax), T)
    return (sA, stA), (sB, stB), master


def abcd(st, fr):
    """chparam_info -> per window group, per scale factor band, (a, b, c,
    d) or None (identity) (clause 5.3.2)."""
    from .audio import sap_coeffs
    mode = st.get("sap_mode") if st else None
    G = fr.num_groups
    if mode == 1 and st.get("ms_used"):
        return [[(1.0, 1.0, 1.0, -1.0) if on else None for on in u]
                for u in st["ms_used"]]
    if mode == 2:
        return "ms"
    if mode == 3 and st.get("sap"):
        return sap_coeffs(st["sap"])
    return [[] for _ in range(G)]


def unmix_windows(wa, wb, st, fr):
    """The stereo operation of 5.3.3.2 on per-window spectra (after
    ungrouping): wa, wb lists of arrays, window w in group fr.w2g[w].
    Used where the two tracks have different max_sfb (the side tracks of
    ASPX_ACPL_1), so different packed layouts."""
    co = abcd(st, fr)
    oa, ob = [x.copy() for x in wa], [x.copy() for x in wb]
    for w in range(min(len(wa), len(wb))):
        x, y = wa[w], wb[w]
        n = min(len(x), len(y))
        if co == "ms":
            oa[w][:n] = x[:n] + y[:n]
            ob[w][:n] = x[:n] - y[:n]
            continue
        g = fr.w2g[w]
        if g >= len(co):
            continue
        tbl = C.SFB_TABLE[len(x)]
        for sfb, c in enumerate(co[g]):
            if c is None or sfb + 1 >= len(tbl):
                continue
            lo, hi = tbl[sfb], min(tbl[sfb + 1], n)
            if hi > lo:
                a, bb, cc, d = c
                oa[w][lo:hi] = a * x[lo:hi] + bb * y[lo:hi]
                ob[w][lo:hi] = cc * x[lo:hi] + d * y[lo:hi]
    return oa, ob
