# SPDX-License-Identifier: Apache-2.0
"""A-CPL side information: PARSING ONLY (TS 103 190-1 clause 4.2.13).

The A-CPL upmix is not built. This module reads acpl_config_2ch (Table 60)
and acpl_data_2ch (Tables 62 to 65) so that an ASPX_ACPL_3 frame can be
walked to the end of its audio_data() and the bit count checked against
audio_size (clause 4.3.4.1). The decoded values are returned as Huffman
symbol indices, not dequantised.
"""
from __future__ import annotations

# Table 142
NUM_PARAM_BANDS = {0: 15, 1: 12, 2: 9, 3: 7}
# Table 143: 0 = fine, 1 = coarse
QUANT = {0: "FINE", 1: "COARSE"}

# Table 62, in bitstream order: (data_type, which quant mode)
DATA_2CH = (("ALPHA", 0), ("ALPHA", 0), ("BETA", 0), ("BETA", 0),
            ("BETA3", 0), ("GAMMA", 1), ("GAMMA", 1), ("GAMMA", 1),
            ("GAMMA", 1), ("GAMMA", 1), ("GAMMA", 1))


def acpl_config_2ch(b):
    """Table 60: four bits, no conditionals."""
    nid = b.u(2)
    return dict(num_param_bands=NUM_PARAM_BANDS[nid], param_band=0,
                quant_mode_0=b.u(1), quant_mode_1=b.u(1))


def load_books(arrays):
    """The 24 A-CPL codebooks (Annex A.34 to A.57) from the parsed tables."""
    from . import spectral as S
    return {n[:-4]: S.Huff(arrays[n], arrays[n[:-4] + "_CW"])
            for n in arrays if n.startswith("ACPL_HCB_") and n.endswith("_LEN")}


def _framing(b):
    """Table 63."""
    interp = b.u(1)
    n_sets = b.u(1) + 1                       # Table 145
    ts = [b.u(5) for _ in range(n_sets)] if interp == 1 else []
    return dict(interpolation_type=interp, num_param_sets=n_sets,
                param_timeslot=ts)


def _huff_data(b, books, data_type, qmode, n_bands, start):
    """Table 65 and Pseudocode 8. -> (diff_type, symbol indices)."""
    q = QUANT[qmode]
    diff_type = b.u(1)
    vals = []
    if diff_type == 0:                        # DIFF_FREQ
        vals.append(books[f"ACPL_HCB_{data_type}_{q}_F0"].decode(b))
        book = books[f"ACPL_HCB_{data_type}_{q}_DF"]
        for _ in range(start + 1, n_bands):
            vals.append(book.decode(b))
    else:                                     # DIFF_TIME
        book = books[f"ACPL_HCB_{data_type}_{q}_DT"]
        for _ in range(start, n_bands):
            vals.append(book.decode(b))
    return diff_type, vals


def acpl_data_2ch(b, books, cfg):
    """Table 62 (with Tables 63 to 65). -> dict of the raw symbols."""
    fr = _framing(b)
    n, st = cfg["num_param_bands"], cfg["param_band"]
    qm = (cfg["quant_mode_0"], cfg["quant_mode_1"])
    params = []
    for data_type, which in DATA_2CH:
        params.append([_huff_data(b, books, data_type, qm[which], n, st)
                       for _ in range(fr["num_param_sets"])])
    return dict(framing=fr, params=params)
