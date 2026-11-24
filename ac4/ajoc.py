# SPDX-License-Identifier: Apache-2.0
"""A-JOC side information (TS 103 190-2 6.2.5, 6.2.3.5, 6.3.6, 5.7.3.1 -
5.7.3.3): parsed, differentially decoded and dequantised.

ajoc() (ajoc_num_decorr, ajoc_ctrl_info, ajoc_data with ajoc_huff_data
and the twelve AJOC_HCB_* codebooks of Annex A.1.1, banked in
tables/ac4_huffman_p2.json) and ajoc_dmx_de_data. Pseudocode 16 (the
differential decoding, with the quantised values of the last data point
of the previous frame held per object), Tables 29 - 32 (dequantisation:
uniform, dry over +-5.0048828, wet over +-2.0019531), Table 28 (the
parameter band of each QMF subband).

Full decoding (phase D): the reconstruction of the upmix objects from
these matrices (5.7.2, 5.7.3.4 - 5.7.3.6, Pseudocodes 14a, 15, 17, 18):
input_order (Pseudocode 14a), frame_targets (the dequantised matrices of
a frame per QMF subband, and the decorrelation input matrix of
5.7.3.6.2), Ramp (the linear interpolation of Pseudocode 17) and Engine
(Pseudocode 18 slot by slot, with the A-CPL decorrelators and ducker of
TS 103 190-1 5.7.7.4, used cyclically 0, 2, 1, 0, 2, 1, 0). Readings
(README, Phase D):

* the ramp: a data point at slot s with ajoc_ramp_len L leaves the value
  of slot s as it was and reaches its own value at slot s + L, linearly
  (Pseudocode 17's "curr_ramp_len <= target_ramp_len" would take L + 1
  steps and overshoot); the first data point of a stream (cold start) is
  taken at once;
* the decorrelation input matrix is |Csub2^T| x Csub1 per QMF subband
  (5.7.3.6.2) at each data point, then interpolated like the others
  (Pseudocode 18 sums the objects' parameter bands, which differ between
  objects with different ajoc_num_bands);
* a disabled decorrelator (ajoc_decorr_enable 0) gives no output;
* a frame's data points act from its first QMF slot (the A-CPL frame
  start, nts f + ts_offset); a start_pos past the frame's slots acts in
  the next frame's."""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(HERE, "tables", "ac4_huffman_p2.json")

# Table 78
NUM_BANDS = (23, 15, 12, 9, 7, 5, 3, 1)
# Table 28: the parameter band of each QMF subband, per ajoc_num_bands
_T28_SB = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 23, 26,
           30, 35, 41, 48, 64)
_T28 = {23: range(23),
        15: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
             13, 14, 14, 14),
        12: (0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10,
             11, 11, 11),
        9: (0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8,
            8),
        7: (0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6,
            6),
        5: (0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
            4),
        3: (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2),
        1: (0,) * 23}


def sb_to_pb(num_bands):
    """Table 28 -> array of 64 (the parameter band of each subband)."""
    rows = list(_T28[num_bands])
    out = np.zeros(64, int)
    for r, v in enumerate(rows):
        out[_T28_SB[r]:_T28_SB[r + 1]] = v
    return out


# Tables 29 - 32: uniform; (levels, max)
DRY_STEPS = {1: 51, 0: 101}               # ajoc_quant_select 1 coarse
WET_STEPS = {1: 21, 0: 41}
DRY_MAX = 5.0048828125
WET_MAX = 2.001953125


def dequant_dry(q, qs):
    n = DRY_STEPS[qs]
    return (np.asarray(q, float) - (n - 1) // 2) * (DRY_MAX / ((n - 1) // 2))


def dequant_wet(q, qs):
    n = WET_STEPS[qs]
    return (np.asarray(q, float) - (n - 1) // 2) * (WET_MAX / ((n - 1) // 2))


_BOOKS = None


def books():
    """The twelve A-JOC codebooks: {(data_type, quant, kind): (Huff,
    cb_off)} (Annex A.1.1: cb_off 0 for F0 and DF, (length - 1) / 2 for
    DT)."""
    global _BOOKS
    if _BOOKS is None:
        from . import spectral as S
        from . import hcb as H
        with open(BANK, encoding="utf-8") as fh:
            arrays = json.load(fh)
        out = {}
        for dt in ("DRY", "WET"):
            for qm in ("COARSE", "FINE"):
                for kind in ("F0", "DF", "DT"):
                    n = f"AJOC_HCB_{dt}_{qm}_{kind}"
                    ln = arrays[n + "_LEN"]
                    if abs(H.kraft(ln) - 1.0) > 1e-9:
                        raise SystemExit(f"codebook {n} failed its Kraft "
                                         "gate")
                    off = (len(ln) - 1) // 2 if kind == "DT" else 0
                    out[(dt, qm, kind)] = (S.Huff(ln, arrays[n + "_CW"]),
                                           off)
        _BOOKS = out
    return _BOOKS


def huff_data(b, data_type, n_bands, quant_select, b_dfonly):
    """ajoc_huff_data (6.2.5.5) -> (diff_type, values) (huff_decode_diff:
    the symbol minus cb_off)."""
    bk = books()
    qm = "COARSE" if quant_select else "FINE"
    diff_type = 0 if b_dfonly else b.u(1)
    vals = []
    if diff_type == 0:
        h, off = bk[(data_type, qm, "F0")]
        vals.append(h.decode(b) - off)
        h, off = bk[(data_type, qm, "DF")]
        for _ in range(1, n_bands):
            vals.append(h.decode(b) - off)
    else:
        h, off = bk[(data_type, qm, "DT")]
        for _ in range(n_bands):
            vals.append(h.decode(b) - off)
    return diff_type, vals


def ajoc(b, n_dmx, n_umx):
    """ajoc() (6.2.5.1 - 6.2.5.4) -> dict: num_decorr, decorr_enable,
    present, dpoints [(start_pos, ramp_len)], per object num_bands,
    quant_select, sparse_select, dry_present, wet_present, and data[o] =
    per data point (dry [(diff_type, values) or None per downmix signal],
    wet [... per decorrelator]), b_nodt."""
    nd = b.u(3)                              # ajoc_num_decorr
    out = dict(num_decorr=nd)
    out["decorr_enable"] = [b.u(1) for _ in range(nd)]
    out["present"] = [b.u(1) for _ in range(n_umx)]
    ndp = b.u(2)                             # ajoc_num_dpoints
    out["dpoints"] = [(b.u(5), b.u(6) + 1) for _ in range(ndp)]
    nb = [0] * n_umx
    qsel = [0] * n_umx
    ssel = [0] * n_umx
    dryp = [[1] * n_dmx for _ in range(n_umx)]
    wetp = [[1] * nd for _ in range(n_umx)]
    if ndp:
        for o in range(n_umx):
            if out["present"][o]:
                nb[o] = NUM_BANDS[b.u(3)]
                qsel[o] = b.u(1)
                ssel[o] = b.u(1)
                if ssel[o]:
                    dryp[o] = [b.u(1) for _ in range(n_dmx)]
                    wetp[o] = [b.u(1) if out["decorr_enable"][d] else 0
                               for d in range(nd)]
    out.update(num_bands=nb, quant_select=qsel, sparse_select=ssel,
               dry_present=dryp, wet_present=wetp)
    # ajoc_data (6.2.5.3)
    out["b_nodt"] = nodt = b.u(1)
    data = []
    for o in range(n_umx):
        per = []
        if out["present"][o]:
            for dp in range(ndp):
                dfonly = dp == 0 and nodt
                dry, wet = [], []
                for ch in range(n_dmx):
                    dry.append(huff_data(b, "DRY", nb[o], qsel[o], dfonly)
                               if (not ssel[o] or dryp[o][ch]) else None)
                for de in range(nd):
                    wet.append(huff_data(b, "WET", nb[o], qsel[o], dfonly)
                               if (not ssel[o] or wetp[o][de]) else None)
                per.append((dry, wet))
        data.append(per)
    out["data"] = data
    return out


def dmx_de_data(b, n_dmx, n_umx, st):
    """ajoc_dmx_de_data (6.2.3.5; Table 82) with the dialogue object
    flags held in st (dict) from the frame that configured them."""
    cfg = b.u(1)                             # b_dmx_de_cfg
    keep = b.u(1)                            # b_keep_dmx_de_coeffs
    if cfg:
        st["de_max_gain"] = b.u(2)
        st["dlg"] = [o for o in range(n_umx) if b.u(1)]
    out = dict(cfg=cfg, keep=keep, coeffs=None)
    if not keep:
        co = []
        for _ in st.get("dlg", []):
            row = []
            for _ in range(n_dmx):
                if b.u(1) == 0:                  # 0b0
                    row.append(0.0)
                else:
                    v = b.u(3)
                    if v == 7:                   # 0b1111
                        row.append(1.0)
                    else:                        # 0b1xxxx
                        row.append((((v << 1) | b.u(1)) + 1) / 15.0)
            co.append(row)
        out["coeffs"] = co
    return out


def quantised(side, prev):
    """Pseudocode 16 for every object: -> per object per data point (dry
    [n_dmx][nb], wet [nd][nb]) quantised values; prev[(o, "d", ch)] /
    prev[(o, "w", de)]: the values of the last data point of the previous
    frame (updated). A time-differential value is taken against the held
    one when it has the same band count (else against the middle index,
    the value 0), and clamped into the table (Tables 29 - 32; a reading:
    the standard does not say what an index outside it means)."""
    out = []
    nd = side["num_decorr"]
    for o, per in enumerate(side["data"]):
        nb = side["num_bands"][o]
        qs = side["quant_select"][o]
        res = []
        for dry_d, wet_d in per:
            dq = []
            for ch, v in enumerate(dry_d):
                dq.append(_diff(v, nb, DRY_STEPS[qs], prev.get((o, "d", ch))))
                prev[(o, "d", ch)] = dq[-1]
            wq = []
            for de in range(nd):
                wq.append(_diff(wet_d[de], nb, WET_STEPS[qs],
                                prev.get((o, "w", de))))
                prev[(o, "w", de)] = wq[-1]
            res.append((dq, wq))
        out.append(res)
    return out


def _diff(v, nb, nquant, prev):
    mid = (nquant - 1) // 2
    if v is None:                            # sparse: not sent
        return [mid] * nb
    diff_type, vals = v
    if diff_type == 0:
        q = [min(max(vals[0], 0), nquant - 1)]
        for i in range(1, nb):
            q.append((q[-1] + vals[i]) % nquant)
        return q
    base = prev if prev is not None and len(prev) == nb else [mid] * nb
    return [min(max(base[i] + vals[i], 0), nquant - 1) for i in range(nb)]


# ---------------------------------------------------------- full decoding
DECORR_CYCLE = (0, 2, 1, 0, 2, 1, 0)        # 5.7.3.5
MAX_DECORR = 7


def input_order(n_fb):
    """Pseudocode 14a: -> the position, among the var_channel_element's
    full-band signals in bitstream order, of A-JOC input i (i = 0 ..
    n_fb - 1). The LFE is not an input."""
    if n_fb > 3:
        n_off = 2 + n_fb % 2
        return ([n_fb - n_off + i for i in range(n_off)]
                + list(range(n_fb - n_off)))
    return list(range(n_fb))


def frame_targets(side, q, n_dmx, n_umx):
    """The data points of one frame -> list of dict(start, ramp, dry
    (n_umx, n_dmx, 64), wet (n_umx, 7, 64), pre (7, n_dmx, 64)): the
    dequantised matrices (Tables 29 - 32; 0 for absent objects and
    elements) mapped to QMF subbands (Table 28), and the decorrelation
    input matrix D = |Csub2^T| x Csub1 (5.7.3.6.2)."""
    nd = side["num_decorr"]
    out = []
    for dp, (start, ramp) in enumerate(side["dpoints"]):
        dry = np.zeros((n_umx, n_dmx, 64))
        wet = np.zeros((n_umx, MAX_DECORR, 64))
        for o in range(n_umx):
            if not side["present"][o] or dp >= len(q[o]):
                continue
            pbm = sb_to_pb(side["num_bands"][o])
            qs = side["quant_select"][o]
            dq, wq = q[o][dp]
            for ch in range(n_dmx):
                dry[o, ch] = dequant_dry(dq[ch], qs)[pbm]
            for de in range(nd):
                wet[o, de] = dequant_wet(wq[de], qs)[pbm]
        pre = np.zeros((MAX_DECORR, n_dmx, 64))
        for o in range(n_umx):
            aw = np.abs(wet[o])
            for de in range(nd):
                for ch in range(n_dmx):
                    pre[de, ch] = pre[de, ch] + aw[de] * dry[o, ch]
        out.append(dict(start=start, ramp=ramp, dry=dry, wet=wet, pre=pre))
    return out


class Ramp:
    """Linear interpolation of a parameter array over QMF slots
    (Pseudocode 17, as read above): event(target, n) starts a ramp of n
    slots from the current value (n 0: a step); value() is the value of
    the current slot; step() moves to the next slot."""

    def __init__(self, shape):
        self.prev = np.zeros(shape)
        self.delta = np.zeros(shape)
        self.curr = 0
        self.target = 0
        self.warm = False

    def event(self, target, n):
        if not self.warm or n <= 0:
            self.prev = np.array(target, float)
            self.delta = np.zeros_like(self.prev)
            self.curr = self.target = 0
            self.warm = True
            return
        self.delta = (target - self.prev) / n
        self.curr, self.target = 0, n

    def step(self):
        if self.curr < self.target:
            self.prev = self.prev + self.delta
            self.curr += 1

    def copy(self):
        c = Ramp.__new__(Ramp)
        c.prev, c.delta = self.prev.copy(), self.delta.copy()
        c.curr, c.target, c.warm = self.curr, self.target, self.warm
        return c


class Engine:
    """Pseudocode 18, slot by slot, across frames: n_dmx inputs, n_umx
    full-band output objects. frame(f_slot, targets, nd, enable) installs
    a frame's data points (absolute slots f_slot + start); run(X) takes
    the next slots of the inputs, X (n_dmx, n, 64) complex in A-JOC input
    order, -> Z (n_umx, n, 64)."""

    def __init__(self, n_dmx, n_umx):
        from . import acpl as AC
        self.n_dmx, self.n_umx = n_dmx, n_umx
        self.dry = Ramp((n_umx, n_dmx, 64))
        self.wet = Ramp((n_umx, MAX_DECORR, 64))
        self.pre = Ramp((MAX_DECORR, n_dmx, 64))
        self.nd = 0
        self.enable = np.zeros(MAX_DECORR)
        self.pending = []                   # (abs slot, target dict)
        self.slot = 0                       # absolute slot of the next run
        s = AC.new_state()
        self.dec = dict(xh=np.zeros((MAX_DECORR,) + s["xh"].shape[1:],
                                    complex),
                        yh=np.zeros((MAX_DECORR,) + s["yh"].shape[1:],
                                    complex),
                        peak=np.zeros((MAX_DECORR, 15)),
                        smooth=np.zeros((MAX_DECORR, 15)),
                        spd=np.zeros((MAX_DECORR, 15)))

    def frame(self, f_slot, targets, nd, enable):
        self.nd = nd
        self.enable = np.array([1.0 if (d < nd and enable[d]) else 0.0
                                for d in range(MAX_DECORR)])
        # data points of an earlier frame not reached yet are dropped
        self.pending = [(f_slot + t["start"], t) for t in targets]

    def run(self, X):
        from . import acpl as AC
        n = X.shape[1]
        nd = self.nd
        dry = np.zeros((n, self.n_umx, self.n_dmx, 64))
        wet = np.zeros((n, self.n_umx, MAX_DECORR, 64))
        U = np.zeros((MAX_DECORR, n, 64), complex)
        for ts in range(n):
            a = self.slot + ts
            for s_, t in self.pending:
                if s_ == a:
                    self.dry.event(t["dry"], t["ramp"])
                    self.wet.event(t["wet"], t["ramp"])
                    self.pre.event(t["pre"], t["ramp"])
            dry[ts] = self.dry.prev
            wet[ts] = self.wet.prev
            pre = self.pre.prev
            for de in range(nd):
                u = np.zeros(64, complex)
                for ch in range(self.n_dmx):
                    u = u + pre[de, ch] * X[ch, ts]
                U[de, ts] = u
            self.dry.step()
            self.wet.step()
            self.pre.step()
        self.pending = [(s_, t) for s_, t in self.pending
                        if s_ >= self.slot + n]
        Y = np.zeros((MAX_DECORR, n, 64), complex)
        if nd:
            rows = DECORR_CYCLE[:nd]
            sub = {k: v[:nd] for k, v in self.dec.items()}
            y = AC.duck(AC.decorrelate(U[:nd], sub, rows=rows), sub)
            for k, v in sub.items():
                self.dec[k][:nd] = v
            Y[:nd] = y * self.enable[:nd, None, None]
        Z = np.zeros((self.n_umx, n, 64), complex)
        for o in range(self.n_umx):
            z = np.zeros((n, 64), complex)
            for ch in range(self.n_dmx):
                z = z + X[ch] * dry[:, o, ch]
            for de in range(nd):
                z = z + Y[de] * wet[:, o, de]
            Z[o] = z
        self.slot += n
        return Z
