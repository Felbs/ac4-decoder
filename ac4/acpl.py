# SPDX-License-Identifier: Apache-2.0
"""A-CPL, advanced coupling: the 5.1 upmix of 5_X_codec_mode ASPX_ACPL_3
(TS 103 190-1 5.7.7, 6.2.11).

The stream carries a two-channel core (L, R in the bitstream's terms: a
downmix) plus, per frame, eleven parameter sets (acpl_data_2ch, Table 62).
In the QMF domain, after A-SPX, the tool rebuilds L, R, C, Ls and Rs from
the two core channels and three decorrelated signals made from them:

    dequantisation         5.7.7.7, Tables 203-208
    parameter bands        5.7.7.2, Table 197 (sb_to_pb)
    interpolation          5.7.7.3, Pseudocodes 109, 110
    decorrelators          5.7.7.4.2, Pseudocode 111, Tables 198-201
    transient ducker       5.7.7.4.3, Pseudocodes 112-114
    the upmix              5.7.7.6.2, Pseudocodes 118, 119

Readings where the standard leaves room (see the README, Interpretations):

* Time alignment. Frame f's A-CPL acts on the same QMF slots as frame f's
  A-SPX in this decoder: slots 24 f + TS_OFFSET .. 24 f + TS_OFFSET + 23
  (aspx_render.TS_OFFSET), so the two tools see one matrix per frame.
* The ducker runs once per QMF time slot (Pseudocode 113 takes the energy
  of one subsample per band; "previous frame" in its text is read as the
  previous time slot), on the 15-band partition of Table 197 (the loop
  runs over acpl_max_num_param_bands = 15), and the energy is that of the
  ducker's own input, the decorrelator output (Pseudocodes 113 and 114 use
  the same x).
* Cold start. acpl_param_prev is 0 "when decoding the first AC-4 frame";
  this decoder plays the stereo core before the stream's first
  acpl_config_2ch, so at the first frame with A-CPL data (and after a
  frame without any) the previous parameters are taken equal to that
  frame's first parameter set instead: no fade from silence.
* Frames without A-CPL data (before the first acpl_config_2ch): the core
  passes through on L/R, C/Ls/Rs are silent, and the tool's state is
  cleared. A concealed frame holds the previous parameters (the core is
  already faded by the concealment) and runs the decorrelators on.
"""
from __future__ import annotations

import math

import numpy as np

from .robust import CONCEALED

NSB = 64
NTS = 24                     # num_qmf_timeslots (1536 / 64)

# ---------------------------------------------------------------- tables
# Table 197: QMF subband -> parameter band, per acpl_num_param_bands
_GROUPS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
           (8, 8), (9, 10), (11, 13), (14, 17), (18, 22), (23, 34), (35, 63))
_PB = {15: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
       12: (0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11),
       9: (0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8),
       7: (0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6)}


def sb_to_pb(num_param_bands):
    """-> int array (64,): the parameter band of each QMF subband."""
    out = np.zeros(NSB, int)
    for (lo, hi), pb in zip(_GROUPS, _PB[num_param_bands]):
        out[lo:hi + 1] = pb
    return out


SB_TO_PB15 = sb_to_pb(15)
# the 15 bands' subband runs (Pseudocode 113: first_sb_in_pb..last_sb_in_pb)
PB15_RUNS = tuple((lo, hi + 1) for lo, hi in _GROUPS)

# Table 203 (fine) and 205 (coarse): alpha_dq and ibeta per alpha_q
ALPHA_FINE = (-2.0, -1.809375, -1.6375, -1.484375, -1.35, -1.234375,
              -1.1375, -1.059375, -1.0, -0.940625, -0.8625, -0.765625,
              -0.65, -0.515625, -0.3625, -0.190625, 0.0, 0.190625, 0.3625,
              0.515625, 0.65, 0.765625, 0.8625, 0.940625, 1.0, 1.059375,
              1.1375, 1.234375, 1.35, 1.484375, 1.6375, 1.809375, 2.0)
IBETA_FINE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4,
              5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0)
ALPHA_COARSE = (-2.0, -1.6375, -1.35, -1.1375, -1.0, -0.8625, -0.65,
                -0.3625, 0.0, 0.3625, 0.65, 0.8625, 1.0, 1.1375, 1.35,
                1.6375, 2.0)
IBETA_COARSE = (0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0)
# Table 204 (fine) and 206 (coarse): beta_dq[beta_q][ibeta]
BETA_FINE = (
    (0.0,) * 9,
    (0.2375, 0.2035449, 0.1729297, 0.1456543, 0.1217188, 0.101123,
     0.0838672, 0.0699512, 0.059375),
    (0.55, 0.4713672, 0.4004688, 0.3373047, 0.281875, 0.2341797,
     0.1942188, 0.1619922, 0.1375),
    (0.9375, 0.8034668, 0.6826172, 0.5749512, 0.4804688, 0.3991699,
     0.3310547, 0.276123, 0.234375),
    (1.4, 1.199844, 1.019375, 0.8585938, 0.7175, 0.5960938, 0.494375,
     0.4123438, 0.35),
    (1.9375, 1.660498, 1.410742, 1.1882319, 0.9929688, 0.8249512,
     0.6841797, 0.5706543, 0.484375),
    (2.55, 2.18543, 1.856719, 1.563867, 1.306875, 1.085742, 0.9004688,
     0.7510547, 0.6375),
    (3.2375, 2.7746389, 2.357305, 1.985498, 1.659219, 1.378467, 1.143242,
     0.9535449, 0.809375),
    (4.0, 3.4281249, 2.9124999, 2.453125, 2.05, 1.703125, 1.4125, 1.178125,
     1.0))
BETA_COARSE = (
    (0.0,) * 5,
    (0.55, 0.4004688, 0.281875, 0.1942188, 0.1375),
    (1.4, 1.019375, 0.7175, 0.494375, 0.35),
    (2.55, 1.856719, 1.306875, 0.9004688, 0.6375),
    (4.0, 2.9124999, 2.05, 1.4125, 1.0))
# Table 207 (beta3) and 208 (gamma): the step per quantisation mode
BETA3_DELTA = {0: 0.125, 1: 0.25}                 # 0 fine, 1 coarse
GAMMA_DELTA = {0: 1638.0 / 16384.0, 1: 3276.0 / 16384.0}

# Tables 198-201: the decorrelators. Region k0 (subbands 0-6): delay 7,
# 7 coefficients; k1 (7-22): delay 10, 4; k2 (23-63): delay 12, 2. a[0]
# is 1 everywhere. COEF[d][region] = (a[1], ..., a[L]).
REGIONS = ((0, 7, 7, 7), (7, 23, 10, 4), (23, 64, 12, 2))  # lo, hi, delay, L
COEF = (
    ((0.5306, -0.4533, -0.6248, 0.0424, 0.4237, 0.4311, 0.1688),
     (0.5561, -0.3039, -0.5024, -0.1850),
     (0.5773, 0.3321)),
    ((-0.4178, 0.1082, -0.2368, -0.1014, -0.1052, -0.3528, 0.4665),
     (0.0425, 0.3235, -0.1556, 0.4958),
     (0.2327, -0.3901)),
    ((0.4007, 0.4747, 0.2611, -0.1211, -0.4248, -0.2989, -0.1932),
     (-0.4361, 0.0345, 0.5215, -0.4178),
     (-0.6057, 0.3804)))
HIST = 14                    # delay + L, the same (14) in every region
MAXL = 7

# Pseudocode 112
DUCK_ALPHA = 0.76592833836465
DUCK_ALPHA_SMOOTH = 0.25
DUCK_GAMMA = 1.5
DUCK_EPSILON = 1.0e-9

GAIN_IN = 1.0 + 2.0 * math.sqrt(0.5)            # Pseudocode 118
SQRT2 = math.sqrt(2.0)


def _filter_arrays():
    """Per decorrelator and subband: delay, and the a / b coefficients of
    Pseudocode 111 padded to 7 with zeros (b[i] = a[L - i], b[0] = a[L])."""
    delay = np.zeros(NSB, int)
    a = np.zeros((3, MAXL + 1, NSB))
    b = np.zeros((3, MAXL + 1, NSB))
    for r, (lo, hi, d, L) in enumerate(REGIONS):
        delay[lo:hi] = d
        for k in range(3):
            full = (1.0,) + COEF[k][r]
            for i in range(L + 1):
                a[k, i, lo:hi] = full[i]
                b[k, i, lo:hi] = full[L - i]
    return delay, a, b


DELAY, A_COEF, B_COEF = _filter_arrays()
_SB = np.arange(NSB)


# ---------------------------------------------------------------- params
def dequantise_2ch(q, cfg):
    """5.7.7.7: acpl_<SET>_q (quantised_2ch) -> dict of float arrays
    (num_param_sets, num_param_bands): a1, a2, b1, b2, b3, g1..g6.

    A value outside the range the set's F0 codebook can code (which for
    alpha and beta is also the range of Tables 203-206) is clamped to it
    first (acpl_parse.Q_RANGE: time-differential decoding after frames
    lost on air)."""
    from .acpl_parse import Q_RANGE, QUANT
    qm0, qm1 = cfg["quant_mode_0"], cfg["quant_mode_1"]
    alpha, ibeta, beta = ((ALPHA_FINE, IBETA_FINE, BETA_FINE) if qm0 == 0
                          else (ALPHA_COARSE, IBETA_COARSE, BETA_COARSE))

    def clamp(k, kind, mode):
        lo, hi = Q_RANGE[(kind, QUANT[mode])]
        return np.clip(np.asarray(q[k], int), lo, hi)

    out = {}
    for name, k_a, k_b in (("1", 0, 2), ("2", 1, 3)):
        qa = clamp(k_a, "ALPHA", qm0)
        qb = clamp(k_b, "BETA", qm0)
        out["a" + name] = np.asarray(alpha, float)[qa]
        out["b" + name] = np.asarray(beta, float)[qb, np.asarray(ibeta)[qa]]
    out["b3"] = clamp(4, "BETA3", qm0) * BETA3_DELTA[qm0]
    for i in range(6):
        out[f"g{i + 1}"] = clamp(5 + i, "GAMMA", qm1) * GAMMA_DELTA[qm1]
    return out


# The quantities Pseudocode 118 / 119 interpolate (each one its own
# acpl_param_prev): the parameters, the products g * a and b3 * a, and the
# two sums of v3.
QUANTITIES = ("g1", "g2", "g3", "g4", "g5", "g6", "a1", "a2", "b1", "b2",
              "b3", "g1a1", "g2a1", "g3a2", "g4a2", "b3a1", "b3a2", "g135",
              "g246")


def quantities(dq):
    """-> dict name -> (num_param_sets, num_param_bands) at band level."""
    out = {k: dq[k] for k in ("g1", "g2", "g3", "g4", "g5", "g6", "a1",
                              "a2", "b1", "b2", "b3")}
    out["g1a1"] = dq["g1"] * dq["a1"]
    out["g2a1"] = dq["g2"] * dq["a1"]
    out["g3a2"] = dq["g3"] * dq["a2"]
    out["g4a2"] = dq["g4"] * dq["a2"]
    out["b3a1"] = dq["b3"] * dq["a1"]
    out["b3a2"] = dq["b3"] * dq["a2"]
    out["g135"] = (dq["g1"] + dq["g3"]) + dq["g5"]
    out["g246"] = (dq["g2"] + dq["g4"]) + dq["g6"]
    return out


def interpolate(prev, sets, interp_type, timeslots, ts, nts=NTS):
    """Pseudocode 109 for every subband at once. prev (64,), sets
    (num_pset, 64) (already mapped to subbands), ts: the frame's time
    slots (ints). -> (len(ts), 64)."""
    t = np.asarray(ts)[:, None]
    n_pset = len(sets)
    if interp_type == 0:                                # smooth
        if n_pset == 1:
            delta = sets[0] - prev
            return prev + (t + 1) * delta / nts
        ts2 = nts // 2
        first = prev + (t + 1) * (sets[0] - prev) / ts2
        second = sets[0] + (t - ts2 + 1) * (sets[1] - sets[0]) / (nts - ts2)
        return np.where(t < ts2, first, second)
    if n_pset == 1:                                     # steep
        return np.where(t < timeslots[0], prev, sets[0])
    return np.where(t < timeslots[0], prev,
                    np.where(t < timeslots[1], sets[0], sets[1]))


# ---------------------------------------------------------------- state
def new_state():
    """The A-CPL tool's state between frames (cold)."""
    return dict(
        warm=False,
        prev={k: np.zeros(NSB) for k in QUANTITIES},
        # decorrelator input history (the last HIST slots of v1, v2, v3)
        # and output history (the last MAXL slots of u0, u1, u2)
        xh=np.zeros((3, HIST, NSB), complex),
        yh=np.zeros((3, MAXL, NSB), complex),
        # ducker (Pseudocode 112), per decorrelator and 15 bands
        peak=np.zeros((3, 15)), smooth=np.zeros((3, 15)),
        spd=np.zeros((3, 15)))


def copy_state(s):
    return dict(warm=s["warm"], prev={k: v.copy() for k, v in
                                      s["prev"].items()},
                xh=s["xh"].copy(), yh=s["yh"].copy(), peak=s["peak"].copy(),
                smooth=s["smooth"].copy(), spd=s["spd"].copy())


def decorrelate(v, s):
    """Pseudocode 111, the three decorrelators at once. v (3, n, 64)
    complex -> u (3, n, 64); s["xh"], s["yh"] carried on."""
    n = v.shape[1]
    X = np.concatenate([s["xh"], v], axis=1)            # (3, HIST + n, 64)
    Y = np.concatenate([s["yh"], np.zeros((3, n, NSB), complex)], axis=1)
    for ts in range(n):
        p = HIST + ts                                   # ts in X
        q = MAXL + ts                                   # ts in Y
        y = B_COEF[:, 0, :] * X[:, p - DELAY, _SB]
        for i in range(1, MAXL + 1):
            # past a region's L the coefficients are 0 (and the index is
            # held inside the history): y + 0 leaves y exactly as it is
            y = y + (B_COEF[:, i, :] * X[:, np.maximum(p - i - DELAY, 0), _SB]
                     - A_COEF[:, i, :] * Y[:, q - i, :])
        Y[:, q, :] = y
    s["xh"] = X[:, -HIST:, :].copy()
    s["yh"] = Y[:, -MAXL:, :].copy()
    return Y[:, MAXL:, :]


def duck(u, s):
    """Pseudocodes 112-114, once per QMF time slot: u (3, n, 64) ->
    the ducked y (3, n, 64); the ducker state carried on."""
    n = u.shape[1]
    e2 = np.abs(u) ** 2
    energy = np.stack([e2[:, :, lo:hi].sum(axis=2) for lo, hi in PB15_RUNS],
                      axis=2)                           # (3, n, 15)
    gain = np.ones((3, n, 15))
    peak, smooth, spd = s["peak"], s["smooth"], s["spd"]
    for ts in range(n):
        e = energy[:, ts, :]
        peak = np.where(DUCK_ALPHA * peak < e, e, DUCK_ALPHA * peak)
        smooth = (1.0 - DUCK_ALPHA_SMOOTH) * smooth
        smooth = smooth + DUCK_ALPHA_SMOOTH * e
        spd = (1.0 - DUCK_ALPHA_SMOOTH) * spd
        spd = spd + DUCK_ALPHA_SMOOTH * (peak - e)
        hot = DUCK_GAMMA * spd > smooth
        with np.errstate(divide="ignore", invalid="ignore"):
            g = smooth / (DUCK_GAMMA * (spd + DUCK_EPSILON))
        gain[:, ts, :] = np.where(hot, g, 1.0)
    s["peak"], s["smooth"], s["spd"] = peak, smooth, spd
    return u * gain[:, :, SB_TO_PB15]


def upmix_frame(x0, x1, frame, s, nts=NTS):
    """One frame of Pseudocode 118. x0, x1: (n, 64) complex, the QMF slots
    of the frame (n <= 24: the end of a decode pass can cut a frame).
    frame: dict(cfg, interp, timeslots, q) -- or None to hold the previous
    parameters (a concealed frame). -> dict L, R, C, Ls, Rs of (n, 64)."""
    n = x0.shape[0]
    ts = np.arange(n)
    pbmap = None
    if frame is not None:
        cfg = frame["cfg"]
        pbmap = sb_to_pb(cfg["num_param_bands"])
        qs = quantities(dequantise_2ch(frame["q"], cfg))
        sets = {k: v[:, pbmap] for k, v in qs.items()}  # (n_pset, 64)
        if not s["warm"]:
            s["prev"] = {k: v[0].copy() for k, v in sets.items()}
        it, tsl = frame["interp"], frame["timeslots"]
    else:
        sets = {k: s["prev"][k][None, :] for k in QUANTITIES}
        it, tsl = 0, ()
    s["warm"] = True
    ip = {k: interpolate(s["prev"][k], sets[k], it, tsl, ts, nts)
          for k in QUANTITIES}
    if frame is not None:                               # Pseudocode 110
        s["prev"] = {k: v[-1].copy() for k, v in sets.items()}

    x0in = x0 * GAIN_IN
    x1in = x1 * GAIN_IN
    v = np.stack([x0in * ip["g1"] + x1in * ip["g2"],    # Transform()
                  x0in * ip["g3"] + x1in * ip["g4"],
                  x0in * ip["g135"] + x1in * ip["g246"]])
    y0, y1, y2 = duck(decorrelate(v, s), s)
    # ACplModule2
    z0 = 0.5 * (x0in * (ip["g1"] + ip["g1a1"]) + x1in * (ip["g2"] + ip["g2a1"])
                + y0 * ip["b1"])
    z1 = 0.5 * (x0in * (ip["g1"] - ip["g1a1"]) + x1in * (ip["g2"] - ip["g2a1"])
                - y0 * ip["b1"])
    z2 = 0.5 * (x0in * (ip["g3"] + ip["g3a2"]) + x1in * (ip["g4"] + ip["g4a2"])
                + y1 * ip["b2"])
    z3 = 0.5 * (x0in * (ip["g3"] - ip["g3a2"]) + x1in * (ip["g4"] - ip["g4a2"])
                - y1 * ip["b2"])
    z4 = 0.5 * (x0in * (ip["g5"] + ip["g5"]) + x1in * (ip["g6"] + ip["g6"]))
    # ACplModule3 (z5 is a temporary that is never output)
    nb3 = -ip["b3"]
    z0 = z0 + 0.25 * y2 * (ip["b3"] + ip["b3a1"])
    z1 = z1 + 0.25 * y2 * (ip["b3"] - ip["b3a1"])
    z2 = z2 + 0.25 * y2 * (ip["b3"] + ip["b3a2"])
    z3 = z3 + 0.25 * y2 * (ip["b3"] - ip["b3a2"])
    z4 = z4 + 0.25 * y2 * (nb3 + nb3)
    return dict(L=z0, R=z2, C=z4 * SQRT2, Ls=z1 * SQRT2, Rs=z3 * SQRT2)


def render(QL, QR, groups, state0=None, state_log=None, ts_offset=4,
           nts=NTS):
    """The A-CPL upmix of a decode pass. QL, QR: (64, n_slots) complex,
    the A-SPX output of the core pair; groups: per frame None (no A-CPL
    data: the core passes through), CONCEALED (hold) or the frame's
    parameters (dict cfg, interp, timeslots, q). Frame f acts on slots
    nts f + ts_offset .. + nts - 1 (nts = num_qmf_timeslots). `state_log` (a list) receives a copy of the
    state at the start of every frame, plus one after the last; `state0`
    starts from a logged state. -> dict L, R, C, Ls, Rs of (64, n_slots)."""
    n_slots = QL.shape[1]
    out = {"L": QL.copy(), "R": QR.copy()}
    for k in ("C", "Ls", "Rs"):
        out[k] = np.zeros_like(QL)
    s = copy_state(state0) if state0 is not None else new_state()
    for f, g in enumerate(groups):
        if state_log is not None:
            state_log.append(copy_state(s))
        a = nts * f + ts_offset
        z = min(a + nts, n_slots)
        if g is None or (g is CONCEALED and not s["warm"]):
            s = new_state()                 # passthrough, state cleared
            continue
        if z <= a:
            continue
        res = upmix_frame(QL[:, a:z].T, QR[:, a:z].T,
                          None if g is CONCEALED else g, s, nts)
        for k, val in res.items():
            out[k][:, a:z] = val.T
    if state_log is not None:
        state_log.append(copy_state(s))
    return out

