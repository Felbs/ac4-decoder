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

The reconstruction of the upmix objects from these matrices (5.7.3.4 -
5.7.3.6, Pseudocodes 17, 18: full decoding) is NOT built: the decoder
decodes A-JOC substreams in core decoding mode (the downmix objects with
their own OAMD; README, Phase C)."""
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
    [n_dmx][nb], wet [nd][nb]) quantised values; prev[o] = (dry, wet) of
    the last data point of the previous frame (updated)."""
    out = []
    nd = side["num_decorr"]
    for o, per in enumerate(side["data"]):
        nb = side["num_bands"][o]
        qs = side["quant_select"][o]
        res = []
        for dry_d, wet_d in per:
            p = prev.get(o)
            dq = []
            for ch, v in enumerate(dry_d):
                dq.append(_diff(v, nb, DRY_STEPS[qs],
                                None if p is None else p[0][ch]))
            wq = []
            for de in range(nd):
                v = wet_d[de]
                wq.append(_diff(v, nb, WET_STEPS[qs],
                                None if p is None else p[1][de]))
            prev[o] = (dq, wq)
            res.append((dq, wq))
        out.append(res)
    return out


def _diff(v, nb, nquant, prev):
    mid = (nquant - 1) // 2
    if v is None:                            # sparse: not sent
        return [mid] * nb
    diff_type, vals = v
    if diff_type == 0:
        q = [vals[0]]
        for i in range(1, nb):
            q.append((q[-1] + vals[i]) % nquant)
        return q
    base = prev if prev is not None and len(prev) == nb else [mid] * nb
    return [base[i] + vals[i] for i in range(nb)]
