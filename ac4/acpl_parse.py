# SPDX-License-Identifier: Apache-2.0
"""A-CPL side information (TS 103 190-1 clause 4.2.13, 4.3.11, 5.7.7.7).

Reads acpl_config_2ch (Table 60) and acpl_data_2ch (Tables 62 to 65), so
that an ASPX_ACPL_3 frame can be walked to the end of its audio_data() and
the bit count checked against audio_size (clause 4.3.4.1), and turns the
Huffman symbols into the quantised parameter values acpl_<SET>_q
(huff_decode_diff with the codebooks' cb_off, Annex A.3; differential
decoding, Pseudocode 121). Dequantisation and the upmix are in acpl.py.
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


# Table 142's helper acpl_1ch_mode
PARTIAL, FULL = 0, 1


def acpl_config_1ch(b, mode):
    """Table 59. mode PARTIAL (ASPX_ACPL_1): acpl_qmf_band_minus1 follows,
    and the parameters start at the band holding it (sb_to_pb); FULL
    (ASPX_ACPL_2): every band, acpl_qmf_band 0."""
    from .acpl import sb_to_pb
    nb = NUM_PARAM_BANDS[b.u(2)]
    qm = b.u(1)
    qmf_band = param_band = 0
    if mode == PARTIAL:
        qmf_band = b.u(3) + 1
        param_band = int(sb_to_pb(nb)[qmf_band])
    return dict(num_param_bands=nb, param_band=param_band,
                qmf_band=qmf_band, quant_mode=qm, mode=mode)


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


# Table 61, in bitstream order
DATA_1CH = (("ALPHA", 0), ("BETA", 0))
SETS_1CH = ("alpha1", "beta1")


def acpl_data_1ch(b, books, cfg):
    """Table 61 (with Tables 63 to 65). -> dict of the raw symbols."""
    fr = _framing(b)
    n, st = cfg["num_param_bands"], cfg["param_band"]
    params = []
    for data_type, _ in DATA_1CH:
        params.append([_huff_data(b, books, data_type, cfg["quant_mode"], n,
                                  st)
                       for _ in range(fr["num_param_sets"])])
    return dict(framing=fr, params=params)


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


# Annex A.3, Tables A.34 to A.57: cb_off of each codebook. huff_decode_diff
# (4.3.10.8.3) returns the codeword's index minus cb_off.
CB_OFF = {
    ("ALPHA", "COARSE", "F0"): 0, ("ALPHA", "FINE", "F0"): 0,
    ("ALPHA", "COARSE", "DF"): 16, ("ALPHA", "FINE", "DF"): 32,
    ("ALPHA", "COARSE", "DT"): 16, ("ALPHA", "FINE", "DT"): 32,
    ("BETA", "COARSE", "F0"): 0, ("BETA", "FINE", "F0"): 0,
    ("BETA", "COARSE", "DF"): 4, ("BETA", "FINE", "DF"): 8,
    ("BETA", "COARSE", "DT"): 4, ("BETA", "FINE", "DT"): 8,
    ("BETA3", "COARSE", "F0"): 0, ("BETA3", "FINE", "F0"): 0,
    ("BETA3", "COARSE", "DF"): 8, ("BETA3", "FINE", "DF"): 16,
    ("BETA3", "COARSE", "DT"): 8, ("BETA3", "FINE", "DT"): 16,
    ("GAMMA", "COARSE", "F0"): 10, ("GAMMA", "FINE", "F0"): 20,
    ("GAMMA", "COARSE", "DF"): 20, ("GAMMA", "FINE", "DF"): 40,
    ("GAMMA", "COARSE", "DT"): 20, ("GAMMA", "FINE", "DT"): 40,
}

# The range of a quantised value: what the F0 codebook (an absolute value)
# can code, and what the dequantisation tables (203 to 206) cover for alpha
# and beta. Time-differential decoding can leave it when the reference is
# not the one the encoder used: after frames lost on air (measured: 33
# frames after a 4-frame gap in broadcaster B's 5.1, until each set's next
# DIFF_FREQ). acpl.dequantise_2ch clamps to it; the q values themselves
# stay as Pseudocode 121 gives them (they are the next frame's reference).
Q_RANGE = {
    ("ALPHA", "FINE"): (0, 32), ("ALPHA", "COARSE"): (0, 16),
    ("BETA", "FINE"): (0, 8), ("BETA", "COARSE"): (0, 4),
    ("BETA3", "FINE"): (0, 16), ("BETA3", "COARSE"): (0, 8),
    ("GAMMA", "FINE"): (-20, 20), ("GAMMA", "COARSE"): (-10, 10),
}

# the names of the eleven sets of Table 62, in bitstream order
SETS_2CH = ("alpha1", "alpha2", "beta1", "beta2", "beta3", "gamma1",
            "gamma2", "gamma3", "gamma4", "gamma5", "gamma6")


def _modes(cfg):
    if "quant_mode" in cfg:                     # acpl_config_1ch
        return (cfg["quant_mode"], cfg["quant_mode"]), DATA_1CH
    return (cfg["quant_mode_0"], cfg["quant_mode_1"]), DATA_2CH


def quantised_2ch(data, cfg, q_prev):
    """Pseudocode 121 for the eleven sets of one acpl_data_2ch() (or the
    two of an acpl_data_1ch(), cfg from acpl_config_1ch): the symbols ->
    acpl_<SET>_q, a list of 11 (2) lists of num_param_sets lists of
    num_param_bands ints (bands below param_band are 0). q_prev (list of 11
    (2) lists, or None entries: the last set of the previous frame) is
    updated in place."""
    n, st = cfg["num_param_bands"], cfg["param_band"]
    qm, kinds = _modes(cfg)
    out = []
    for k, ((data_type, which), sets) in enumerate(zip(kinds,
                                                        data["params"])):
        q = QUANT[qm[which]]
        per_set = []
        for diff_type, vals in sets:
            cur = [0] * n
            if diff_type == 0:                # DIFF_FREQ: F0, then DF
                cur[st] = vals[0] - CB_OFF[(data_type, q, "F0")]
                off = CB_OFF[(data_type, q, "DF")]
                for i in range(st + 1, n):
                    cur[i] = cur[i - 1] + (vals[i - st] - off)
            else:                             # DIFF_TIME
                off = CB_OFF[(data_type, q, "DT")]
                prev = q_prev[k] or []
                for i in range(st, n):
                    cur[i] = ((prev[i] if i < len(prev) else 0)
                              + (vals[i - st] - off))
            per_set.append(cur)
            q_prev[k] = cur
        out.append(per_set)
    return out


quantised_1ch = quantised_2ch


def out_of_range_2ch(q, cfg):
    """-> True when a value of quantised_2ch's output leaves Q_RANGE."""
    qm, kinds = _modes(cfg)
    for (data_type, which), sets in zip(kinds, q):
        lo, hi = Q_RANGE[(data_type, QUANT[qm[which]])]
        if any(not lo <= v <= hi for cur in sets for v in cur):
            return True
    return False
