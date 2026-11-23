# SPDX-License-Identifier: Apache-2.0
"""A-CPL (TS 103 190-1 5.7.7): the upmix of ASPX_ACPL_3, without captures.

The reference here is the standard's pseudocode written out loop by loop,
sample by sample (Pseudocodes 109-114, 118, 119, 121), independent of the
vectorised ac4/acpl.py; c/tests/check_acpl.py holds the C to the Python.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ac4 import acpl as ACPL
from ac4 import acpl_parse as AP
from ac4.robust import CONCEALED


# ------------------------------------------------------------ tables
def test_dequantisation_tables():
    """Tables 203-206: alpha is odd around its centre, ibeta even, and the
    coarse tables are the fine ones at every other step (q 2k, ibeta 2k);
    Table 197 maps 64 subbands onto 0..n-1, monotone."""
    a, ib = np.array(ACPL.ALPHA_FINE), np.array(ACPL.IBETA_FINE)
    assert np.allclose(a, -a[::-1]) and np.array_equal(ib, ib[::-1])
    assert np.allclose(ACPL.ALPHA_COARSE, a[::2])
    assert np.array_equal(ACPL.IBETA_COARSE, ib[::2] // 2)
    bf, bc = np.array(ACPL.BETA_FINE), np.array(ACPL.BETA_COARSE)
    assert np.allclose(bc, bf[::2, ::2])
    assert np.all(np.diff(bf, axis=0) >= 0)           # rises with beta_q
    for n in (15, 12, 9, 7):
        m = ACPL.sb_to_pb(n)
        assert m[0] == 0 and m[-1] == n - 1 and np.all(np.diff(m) >= 0)
        assert set(m) == set(range(n))
    assert list(ACPL.sb_to_pb(7)[[3, 8, 9, 23, 35]]) == [2, 4, 4, 6, 6]


def test_decorrelators_are_stable_allpass_filters():
    """Pseudocode 111 with Tables 198-201: every section is an all-pass
    (b[i] = a[L - i]) with its poles inside the unit circle, so an impulse
    keeps its energy, and nothing comes out before the region's delay."""
    for d in range(3):
        for r, (_lo, _hi, _dl, L) in enumerate(ACPL.REGIONS):
            assert np.abs(np.roots((1.0,) + ACPL.COEF[d][r])).max() < 1.0
    s = ACPL.new_state()
    v = np.zeros((3, 24, 64), complex)
    v[:, 0, :] = 1.0
    u = [ACPL.decorrelate(v, s)]
    for _ in range(40):                          # 41 frames of the tail
        u.append(ACPL.decorrelate(np.zeros((3, 24, 64), complex), s))
    h = np.concatenate(u, axis=1)
    for lo, hi, dl, _L in ACPL.REGIONS:
        assert np.all(h[:, :dl, lo:hi] == 0)
        assert np.all(h[:, dl, lo:hi] != 0)
        e = (np.abs(h[:, :, lo:hi]) ** 2).sum(axis=1)
        assert np.allclose(e, 1.0, atol=1e-6)


# ------------------------------------------------------------ parsing
def test_differential_decoding_with_codebook_offsets():
    """Pseudocode 121 and huff_decode_diff (cb_off, Annex A.3): DIFF_FREQ
    from the F0 value up the bands, DIFF_TIME against the previous set
    (the last set of the previous frame), each set on its own."""
    cfg = dict(num_param_bands=7, param_band=0, quant_mode_0=0,
               quant_mode_1=1)
    f0, df = 0, 32                              # ALPHA fine: F0 0, DF 32
    params = []
    for k, (dt, which) in enumerate(AP.DATA_2CH):
        mode = AP.QUANT[cfg["quant_mode_1" if which else "quant_mode_0"]]
        o_f0 = AP.CB_OFF[(dt, mode, "F0")]
        o_df = AP.CB_OFF[(dt, mode, "DF")]
        o_dt = AP.CB_OFF[(dt, mode, "DT")]
        if k == 0:
            assert (o_f0, o_df) == (f0, df)
        # set 0 by frequency: 3, then +1, -2, 0, 0, +1, 0; set 1 by time:
        # +1 on every band
        sets = [(0, [3 + o_f0] + [d + o_df for d in (1, -2, 0, 0, 1, 0)]),
                (1, [1 + o_dt] * 7)]
        params.append(sets)
    prev = [None] * 11
    q = AP.quantised_2ch(dict(params=params), cfg, prev)
    for k in range(11):
        assert q[k][0] == [3, 4, 2, 2, 2, 3, 3]
        assert q[k][1] == [4, 5, 3, 3, 3, 4, 4]
        assert prev[k] == [4, 5, 3, 3, 3, 4, 4]
    # a later frame by time continues from the last set
    def dt_off(dt, w):
        mode = AP.QUANT[cfg["quant_mode_1" if w else "quant_mode_0"]]
        return AP.CB_OFF[(dt, mode, "DT")]
    params2 = [[(1, [dt_off(dt, w) - 1] * 7)] for dt, w in AP.DATA_2CH]
    q2 = AP.quantised_2ch(dict(params=params2), cfg, prev)
    assert all(q2[k][0] == [3, 4, 2, 2, 2, 3, 3] for k in range(11))


def test_out_of_range_values_are_clamped_at_dequantisation():
    cfg = dict(num_param_bands=7, param_band=0, quant_mode_0=0,
               quant_mode_1=1)
    q = [[[40, -3, 16, 16, 16, 16, 16]]] * 2 + [[[9, -1, 0, 0, 0, 0, 0]]] * 2 \
        + [[[17, -1, 0, 0, 0, 0, 0]]] + [[[11, -11, 0, 0, 0, 0, 0]]] * 6
    assert AP.out_of_range_2ch(q, cfg)
    dq = ACPL.dequantise_2ch(q, cfg)
    assert dq["a1"][0, 0] == 2.0 and dq["a1"][0, 1] == -2.0
    assert dq["b1"][0, 0] == ACPL.BETA_FINE[8][0] and dq["b1"][0, 1] == 0.0
    assert dq["b3"][0, 0] == 16 * 0.125 and dq["b3"][0, 1] == 0.0
    assert math.isclose(dq["g1"][0, 0], 10 * 3276 / 16384)
    assert math.isclose(dq["g1"][0, 1], -10 * 3276 / 16384)


# ------------------------------------------------------------ the upmix
def _dq_ref(q, cfg, ps):
    """5.7.7.7 for one parameter set, band by band."""
    fine0 = cfg["quant_mode_0"] == 0
    fine1 = cfg["quant_mode_1"] == 0
    alpha = ACPL.ALPHA_FINE if fine0 else ACPL.ALPHA_COARSE
    ibeta = ACPL.IBETA_FINE if fine0 else ACPL.IBETA_COARSE
    beta = ACPL.BETA_FINE if fine0 else ACPL.BETA_COARSE
    out = {}
    for b in range(cfg["num_param_bands"]):
        for i, (ka, kb) in enumerate(((0, 2), (1, 3))):
            qa, qb = q[ka][ps][b], q[kb][ps][b]
            out.setdefault(f"a{i + 1}", []).append(alpha[qa])
            out.setdefault(f"b{i + 1}", []).append(beta[qb][ibeta[qa]])
        out.setdefault("b3", []).append(q[4][ps][b]
                                        * (0.125 if fine0 else 0.25))
        for i in range(6):
            out.setdefault(f"g{i + 1}", []).append(
                q[5 + i][ps][b] * ((1638 if fine1 else 3276) / 16384))
    return out


class _Ref:
    """Pseudocodes 109-114, 118, 119 sample by sample."""

    TABLE_A = {}

    def __init__(self):
        self.prev = None              # per quantity: 64 values
        self.xh = [[[0j] * 64 for _ in range(14)] for _ in range(3)]
        self.yh = [[[0j] * 64 for _ in range(7)] for _ in range(3)]
        self.peak = [[0.0] * 15 for _ in range(3)]
        self.smooth = [[0.0] * 15 for _ in range(3)]
        self.spd = [[0.0] * 15 for _ in range(3)]

    @staticmethod
    def params(dq):
        p = dict(dq)
        p["g1a1"] = [g * a for g, a in zip(dq["g1"], dq["a1"])]
        p["g2a1"] = [g * a for g, a in zip(dq["g2"], dq["a1"])]
        p["g3a2"] = [g * a for g, a in zip(dq["g3"], dq["a2"])]
        p["g4a2"] = [g * a for g, a in zip(dq["g4"], dq["a2"])]
        p["b3a1"] = [g * a for g, a in zip(dq["b3"], dq["a1"])]
        p["b3a2"] = [g * a for g, a in zip(dq["b3"], dq["a2"])]
        p["g135"] = [(a + b) + c for a, b, c in zip(dq["g1"], dq["g3"],
                                                     dq["g5"])]
        p["g246"] = [(a + b) + c for a, b, c in zip(dq["g2"], dq["g4"],
                                                     dq["g6"])]
        return p

    def interpolate(self, name, sets, pb, it, tsl, sb, ts):
        num_ts = 24
        prev = self.prev[name][sb]
        if it == 0:
            if len(sets) == 1:
                delta = sets[0][name][pb[sb]] - prev
                return prev + (ts + 1) * delta / num_ts
            ts_2 = num_ts // 2
            if ts < ts_2:
                delta = sets[0][name][pb[sb]] - prev
                return prev + (ts + 1) * delta / ts_2
            delta = sets[1][name][pb[sb]] - sets[0][name][pb[sb]]
            return sets[0][name][pb[sb]] + (ts - ts_2 + 1) * delta / (
                num_ts - ts_2)
        if len(sets) == 1:
            return prev if ts < tsl[0] else sets[0][name][pb[sb]]
        if ts < tsl[0]:
            return prev
        return sets[0][name][pb[sb]] if ts < tsl[1] else \
            sets[1][name][pb[sb]]

    def decorrelate(self, d, v):
        """Pseudocode 111 on one frame of v[ts][sb]."""
        n = len(v)
        y = [[0j] * 64 for _ in range(n)]
        for sb in range(64):
            r = 0 if sb < 7 else (1 if sb < 23 else 2)
            _lo, _hi, delay, L = ACPL.REGIONS[r]
            a = (1.0,) + ACPL.COEF[d][r]

            def x(t):
                return v[t][sb] if t >= 0 else self.xh[d][14 + t][sb]

            def yy(t):
                return y[t][sb] if t >= 0 else self.yh[d][7 + t][sb]
            for ts in range(n):
                b0 = a[L]
                acc = b0 * x(ts - delay) / a[0]
                for i in range(1, L + 1):
                    bi = a[L - i]
                    acc += (bi * x(ts - i - delay) - a[i] * yy(ts - i)) / a[0]
                y[ts][sb] = acc
        hist = self.xh[d] + v
        self.xh[d] = hist[-14:]
        yhist = self.yh[d] + y
        self.yh[d] = yhist[-7:]
        return y

    def duck(self, d, u):
        """Pseudocodes 112-114, once per time slot."""
        ALPHA, AS, GAMMA, EPS = 0.76592833836465, 0.25, 1.5, 1.0e-9
        out = []
        for ts in range(len(u)):
            g = [1.0] * 15
            for pb in range(15):
                lo, hi = ACPL.PB15_RUNS[pb]
                e = 0.0
                for sb in range(lo, hi):
                    e += abs(u[ts][sb]) ** 2
                if ALPHA * self.peak[d][pb] < e:
                    self.peak[d][pb] = e
                else:
                    self.peak[d][pb] = ALPHA * self.peak[d][pb]
                self.smooth[d][pb] = (1 - AS) * self.smooth[d][pb] + AS * e
                self.spd[d][pb] = ((1 - AS) * self.spd[d][pb]
                                   + AS * (self.peak[d][pb] - e))
                if GAMMA * self.spd[d][pb] > self.smooth[d][pb]:
                    g[pb] = self.smooth[d][pb] / (GAMMA * (self.spd[d][pb]
                                                           + EPS))
            out.append([u[ts][sb] * g[ACPL.SB_TO_PB15[sb]]
                        for sb in range(64)])
        return out

    def frame(self, x0, x1, fr):
        """x0, x1: [ts][sb] complex -> dict of [ts][sb]."""
        cfg = fr["cfg"]
        pb = ACPL.sb_to_pb(cfg["num_param_bands"])
        sets = [self.params(_dq_ref(fr["q"], cfg, ps))
                for ps in range(len(fr["q"][0]))]
        names = list(sets[0])
        if self.prev is None:
            self.prev = {k: [sets[0][k][pb[sb]] for sb in range(64)]
                         for k in names}
        n = len(x0)
        it, tsl = fr["interp"], fr["timeslots"]

        def ip(name, sb, ts):
            return self.interpolate(name, sets, pb, it, tsl, sb, ts)
        k = 1 + 2 * math.sqrt(0.5)
        x0in = [[x0[t][s] * k for s in range(64)] for t in range(n)]
        x1in = [[x1[t][s] * k for s in range(64)] for t in range(n)]

        def transform(g1, g2):
            return [[x0in[t][s] * ip(g1, s, t) + x1in[t][s] * ip(g2, s, t)
                     for s in range(64)] for t in range(n)]
        v = [transform("g1", "g2"), transform("g3", "g4"),
             transform("g135", "g246")]
        y = [self.duck(d, self.decorrelate(d, v[d])) for d in range(3)]

        def module2(g1, g2, a, b, yy):
            z0 = [[0j] * 64 for _ in range(n)]
            z1 = [[0j] * 64 for _ in range(n)]
            for s in range(64):
                for t in range(n):
                    i1, i2 = ip(g1, s, t), ip(g2, s, t)
                    i1a = ip(g1 + a, s, t) if a else i1
                    i2a = ip(g2 + a, s, t) if a else i2
                    ib = ip(b, s, t) if b else 0.0
                    yv = yy[t][s] if yy else 0.0
                    z0[t][s] = 0.5 * (x0in[t][s] * (i1 + i1a)
                                      + x1in[t][s] * (i2 + i2a) + yv * ib)
                    z1[t][s] = 0.5 * (x0in[t][s] * (i1 - i1a)
                                      + x1in[t][s] * (i2 - i2a) - yv * ib)
            return z0, z1

        def module3(sign, a, z0, z1):
            for s in range(64):
                for t in range(n):
                    b3 = sign * ip("b3", s, t)
                    b3a = sign * ip("b3" + a, s, t) if a else b3
                    z0[t][s] += 0.25 * y[2][t][s] * (b3 + b3a)
                    z1[t][s] += 0.25 * y[2][t][s] * (b3 - b3a)
        z0, z1 = module2("g1", "g2", "a1", "b1", y[0])
        z2, z3 = module2("g3", "g4", "a2", "b2", y[1])
        z4, z5 = module2("g5", "g6", None, None, None)
        module3(1, "a1", z0, z1)
        module3(1, "a2", z2, z3)
        module3(-1, None, z4, z5)
        r2 = math.sqrt(2)
        self.prev = {k: [sets[-1][k][pb[sb]] for sb in range(64)]
                     for k in names}
        return dict(L=z0, R=z2, C=[[c * r2 for c in row] for row in z4],
                    Ls=[[c * r2 for c in row] for row in z1],
                    Rs=[[c * r2 for c in row] for row in z3])


def _random_frame(rng, cfg, n_sets, interp):
    q = []
    for data_type, which in AP.DATA_2CH:
        mode = AP.QUANT[cfg["quant_mode_1" if which else "quant_mode_0"]]
        lo, hi = AP.Q_RANGE[(data_type, mode)]
        q.append(rng.integers(lo, hi + 1,
                              (n_sets, cfg["num_param_bands"])).tolist())
    ts = sorted(rng.integers(0, 24, n_sets).tolist()) if interp else []
    return dict(cfg=cfg, q=q, interp=interp, timeslots=tuple(ts))


@pytest.mark.parametrize("nb,qm0,qm1", [(15, 0, 0), (9, 1, 0), (7, 0, 1)])
def test_upmix_against_the_pseudocode(nb, qm0, qm1):
    """acpl.render against Pseudocodes 109-114, 118 and 119 written out
    sample by sample, over four frames (one and two parameter sets, smooth
    and steep), the state carried from frame to frame."""
    rng = np.random.default_rng(nb)
    cfg = dict(num_param_bands=nb, param_band=0, quant_mode_0=qm0,
               quant_mode_1=qm1)
    frames = [_random_frame(rng, cfg, 1, 0), _random_frame(rng, cfg, 2, 0),
              _random_frame(rng, cfg, 1, 1), _random_frame(rng, cfg, 2, 1)]
    n = 24 * len(frames)
    QL = rng.standard_normal((64, n)) + 1j * rng.standard_normal((64, n))
    QR = rng.standard_normal((64, n)) + 1j * rng.standard_normal((64, n))
    got = ACPL.render(QL, QR, frames, ts_offset=0)
    ref = _Ref()
    for f, fr in enumerate(frames):
        sl = slice(24 * f, 24 * f + 24)
        r = ref.frame(QL[:, sl].T.tolist(), QR[:, sl].T.tolist(), fr)
        for k in ("L", "R", "C", "Ls", "Rs"):
            want = np.array(r[k]).T
            assert np.allclose(got[k][:, sl], want, rtol=1e-11,
                               atol=1e-11), (f, k)


def test_upmix_sums_back_to_the_core(monkeypatch):
    """Pseudocode 118: z0 + z1 + z4 = (g1 + g5) x0in + (g2 + g6) x1in, the
    decorrelated terms cancelling; with the gammas of the downmix (g1 + g5
    = 1, g2 + g6 = 0) L + Ls/sqrt2 + C/sqrt2 is the core's left channel
    times 1 + sqrt2, whatever alpha, beta and beta3 are (likewise right,
    g3 + g5 = 0, g4 + g6 = 1)."""
    rng = np.random.default_rng(3)
    nb = 15

    def dq(q, cfg):
        d = {k: rng.uniform(-2, 2, (1, nb)) for k in ("a1", "a2", "g1",
                                                      "g2", "g3", "g4")}
        d.update({k: rng.uniform(0, 2, (1, nb)) for k in ("b1", "b2", "b3")})
        d["g5"] = 1.0 - d["g1"]                       # g1 + g5 = 1
        d["g6"] = -d["g2"]                            # g2 + g6 = 0
        d["g3"] = -d["g5"]                            # g3 + g5 = 0
        d["g4"] = 1.0 - d["g6"]                       # g4 + g6 = 1
        return d
    monkeypatch.setattr(ACPL, "dequantise_2ch", dq)
    cfg = dict(num_param_bands=nb, param_band=0, quant_mode_0=0,
               quant_mode_1=0)
    frames = [dict(cfg=cfg, q=None, interp=0, timeslots=())] * 3
    n = 72
    QL = rng.standard_normal((64, n)) + 1j * rng.standard_normal((64, n))
    QR = rng.standard_normal((64, n)) + 1j * rng.standard_normal((64, n))
    z = ACPL.render(QL, QR, frames, ts_offset=0)
    k = 1 + math.sqrt(2)
    lo = z["L"] + z["Ls"] / math.sqrt(2) + z["C"] / math.sqrt(2)
    ro = z["R"] + z["Rs"] / math.sqrt(2) + z["C"] / math.sqrt(2)
    assert np.allclose(lo, k * QL, atol=1e-9)
    assert np.allclose(ro, k * QR, atol=1e-9)
    assert np.abs(z["Ls"]).max() > 0.1 and np.abs(z["C"]).max() > 0.1


def test_frames_without_data_pass_the_core_and_concealed_frames_hold():
    """No A-CPL data: L/R are the core, C/Ls/Rs silent, the state cleared
    (a cold concealed frame too). A concealed frame after data holds the
    previous parameters: the same output as a data frame repeating the
    last parameter set."""
    rng = np.random.default_rng(9)
    cfg = dict(num_param_bands=12, param_band=0, quant_mode_0=0,
               quant_mode_1=0)
    fr = _random_frame(rng, cfg, 1, 0)
    n = 24 * 4
    QL = rng.standard_normal((64, n)) + 1j * rng.standard_normal((64, n))
    QR = rng.standard_normal((64, n)) + 1j * rng.standard_normal((64, n))
    z = ACPL.render(QL, QR, [None, CONCEALED, fr, CONCEALED], ts_offset=0)
    assert np.array_equal(z["L"][:, :48], QL[:, :48])
    assert np.array_equal(z["R"][:, :48], QR[:, :48])
    for k in ("C", "Ls", "Rs"):
        assert not np.any(z[k][:, :48])
    held = ACPL.render(QL, QR, [None, None, fr, fr], ts_offset=0)
    for k in z:
        assert np.allclose(z[k], held[k], rtol=1e-13, atol=1e-13)
    assert np.isfinite(np.stack(list(z.values()))).all()


def test_decoder_upmixes_aspx_acpl_3_to_six_channels(monkeypatch):
    """Decoder("5_X") on an ASPX_ACPL_3 stream (stubbed: the stream gives
    the core pair, the LFE and A-CPL parameters): C, Ls and Rs come from
    the upmix, every channel 577 samples after its core, and the upmix
    sums back to the core pair (gammas of a downmix, fixed alpha/beta).
    Before A-CPL was built C, Ls, Rs (and the LFE) were silent."""
    from ac4 import decoder as D
    rng = np.random.default_rng(4)
    n = 12
    specs = {k: [rng.standard_normal(1536) * (np.arange(1536) < 300)
                 for _ in range(n)] for k in ("L", "R", "lfe")}

    def dq(q, cfg):
        o = {k: np.full((1, 15), v) for k, v in dict(
            a1=0.3, a2=-0.2, b1=0.5, b2=0.4, b3=0.6, g1=0.6, g2=0.1,
            g3=-0.4, g4=0.7).items()}
        o["g5"], o["g6"] = 1.0 - o["g1"], -o["g2"]
        o["g3"], o["g4"] = -o["g5"], 1.0 - o["g6"]
        return o
    monkeypatch.setattr(ACPL, "dequantise_2ch", dq)
    cfg = dict(num_param_bands=15, param_band=0, quant_mode_0=0,
               quant_mode_1=0)
    fr = dict(cfg=cfg, q=None, interp=0, timeslots=())
    dec = D.Decoder("5_X", guard=False)
    dec.hf_ratio = None

    def fake_decode_frames(frames, chans):
        assert "C" in chans
        w = {k: list(specs[k][:len(frames)]) for k in ("L", "R", "lfe")}
        ln = {k: [1536] * len(frames) for k in w}
        ct = {k: [1] * len(frames) for k in w}
        dec.stream.mode = ("5_X", 4)
        dec.stream.acpl3 = True
        dec.stream.cfg = None
        dec.stream.status = [None] * len(frames)
        return w, ln, {"lr": [None] * len(frames), "sr": [None] * len(frames),
                       "acpl": [None] * 2 + [fr] * (len(frames) - 2)}, ct

    monkeypatch.setattr(dec.stream, "decode_frames", fake_decode_frames)
    y = dec.decode_pass([b""] * n)
    core = D.Decoder("5_X", channels=2, guard=False)
    core.hf_ratio = None
    monkeypatch.setattr(core.stream, "decode_frames", lambda f, c: (
        {k: list(specs[k][:len(f)]) for k in ("L", "R")},
        {k: [1536] * len(f) for k in ("L", "R")},
        {"lr": [None] * len(f), "sr": [None] * len(f)},
        {k: [1] * len(f) for k in ("L", "R")}))
    core.stream.mode = ("5_X", 4)
    yc = core.decode_pass([b""] * n)
    L, R, C, LFE, Ls, Rs = y.T
    s = slice(4 * 1536, None)
    for x in (C, Ls, Rs, LFE):
        assert np.sqrt(np.mean(x[s] ** 2)) > 0.05 * np.sqrt(np.mean(L[s] ** 2))
    k = 1 + math.sqrt(2)
    r2 = math.sqrt(2)
    for est, ref in (((L + C / r2 + Ls / r2) / k, yc[:, 0]),
                     ((R + C / r2 + Rs / r2) / k, yc[:, 1])):
        err = est[s] - ref[s]
        assert 10 * np.log10(np.sum(ref[s] ** 2) / np.sum(err ** 2)) > 60
    # the LFE through the delay line: its core, 577 samples late
    lfe_core = D.FB.synthesise_frames(
        {"lfe": specs["lfe"]}, {"lfe": [1536] * n}, {"lfe": [1] * n},
        1536)["lfe"]
    ref = np.concatenate([np.zeros(577), lfe_core])[:len(LFE)]
    assert np.allclose(LFE, ref, atol=1e-9 * np.abs(ref).max())
