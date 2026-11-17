# SPDX-License-Identifier: Apache-2.0
"""Full SAP (sap_mode 3) without broadcast audio: the sap_data() parse
(TS 103 190-1 Table 48), the alpha_q reconstruction (Pseudocode 59), the
two-track matrix (clause 5.3.3.2), and the A-CPL side-data parse used to
walk ASPX_ACPL_3 frames to the end of audio_data()."""
import numpy as np
import pytest

from ac4 import acpl_parse as AP
from ac4 import audio
from ac4 import channels as C
from ac4 import hcb as H
from ac4 import spectral as S
from ac4.toc_v1 import Bits

ARRAYS = H.parse_c(H.DEFAULT_C)
SF_LEN, SF_CW = ARRAYS["ASF_HCB_SCALEFAC_LEN"], ARRAYS["ASF_HCB_SCALEFAC_CW"]
SF = S.Huff(SF_LEN, SF_CW)


def _code(lens, cws, sym):
    return format(cws[sym] & ((1 << lens[sym]) - 1), f"0{lens[sym]}b")


def _bytes(bits):
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def _long(max_sfb):
    """A long-frame Framing (b_long_frame = 1, 6-bit max_sfb)."""
    return C.Framing(Bits(_bytes("1" + format(max_sfb, "06b"))))


def _short_4x384(max_sfb):
    """Four 384-line windows, each its own group (grouping bits all 0)."""
    fr = C.Framing(Bits(_bytes("0" + "10" + "10" + format(max_sfb, "06b")
                               + "000")))
    assert fr.num_groups == 4
    return fr


def _sap(max_sfb, used, dpcm, dct=0):
    return dict(coeff_all=0, coeff_used=used, delta_code_time=dct,
                dpcm=dpcm, max_sfb=max_sfb)


# ------------------------------------------------------ Pseudocode 59
def test_alpha_q_frequency_dpcm_and_odd_bands_copy():
    s = _sap([5], [[1] * 5], [[62, None, 59, None, 63]])
    assert audio.sap_alpha_q(s) == [[2, 2, 1, 1, 4]]


def test_alpha_q_time_delta_against_previous_group():
    d0 = [61, None, 62, None]
    d1 = [63, None, 57, None]
    s = _sap([4, 4], [[1] * 4] * 2, [d0, d1], dct=1)
    q = audio.sap_alpha_q(s)
    assert q[0] == [1, 1, 3, 3]
    assert q[1] == [1 + 3, 4, 3 - 3, 0]
    s["delta_code_time"] = 0                     # same data, frequency coded
    assert audio.sap_alpha_q(s)[1] == [3, 3, 0, 0]


def test_alpha_q_time_delta_off_when_max_sfb_changes():
    s = _sap([4, 2], [[1] * 4, [1] * 2], [[61, None, 62, None], [63, None]],
             dct=1)
    assert audio.sap_alpha_q(s)[1] == [3, 3]     # not 1 + 3


def test_alpha_q_unused_bands_and_references_to_them():
    used = [[1, 1, 0, 0, 1, 1]]
    s = _sap([6], used, [[64, None, None, None, 58, None]])
    # band 4 refers to band 2, which is unused: its alpha_q counts as 0
    assert audio.sap_alpha_q(s) == [[4, 4, None, None, -2, -2]]


def test_coeffs_follow_clause_5_3_2():
    s = _sap([2], [[1, 1]], [[63, None]])
    a, b, c, d = audio.sap_coeffs(s)[0][0]
    assert (a, b, c, d) == pytest.approx((1.3, 1.0, 0.7, -1.0))


# ------------------------------------------------- the matrix, 5.3.3.2
def _encoder(L, R, fr):
    """A minimal SAP encoder: M = (L + R) / 2, alpha_q per band pair from a
    least-squares prediction of the side from M, residual S = side - alpha M.
    -> (M, S, sap dict)."""
    M = (L + R) / 2
    side = (L - R) / 2
    off = fr.sect_sfb_offset(0)
    m = fr.max_sfb_g(0)
    used = [1] * m
    q = []
    for sfb in range(0, m, 2):
        lo, hi = off[sfb], off[min(sfb + 2, m)]
        q.append(int(round(10 * np.dot(side[lo:hi], M[lo:hi])
                           / np.dot(M[lo:hi], M[lo:hi]))))
    dpcm, prev = [None] * m, 0
    for i, sfb in enumerate(range(0, m, 2)):
        dpcm[sfb] = 60 + q[i] - prev
        prev = q[i]
    alpha = np.zeros_like(M)
    for i, sfb in enumerate(range(0, m, 2)):
        alpha[off[sfb]:off[min(sfb + 2, m)]] = 0.1 * q[i]
    return M, side - alpha * M, _sap([m], [used], [dpcm])


def test_matrix_inverts_a_prediction_encoder_and_wrong_variants_do_not():
    rng = np.random.default_rng(3)
    fr = _long(20)
    n = fr.sect_sfb_offset(0)[20]
    src = rng.standard_normal(n)
    L = 1.4 * src + 0.2 * rng.standard_normal(n)       # panned left
    R = 0.6 * src + 0.2 * rng.standard_normal(n)
    M, Sres, sap = _encoder(L, R, fr)
    assert max(v for v in sap["dpcm"][0] if v is not None) > 60  # alpha > 0
    st = dict(sap_mode=3, sap=sap)
    l2, r2 = audio.unmix(M, Sres, st, fr)
    assert np.allclose(l2, L, atol=1e-12) and np.allclose(r2, R, atol=1e-12)

    good = audio.sap_coeffs
    wrong = {
        "sign_flip": lambda s: [[None if x is None else (x[2], x[1], x[0], x[3])
                                 for x in g] for g in good(s)],
        "transposed": lambda s: [[None if x is None else (x[0], x[2], x[1], x[3])
                                  for x in g] for g in good(s)],
        "passthrough": lambda s: [[None] * len(g) for g in good(s)],
    }
    for name, fn in wrong.items():
        audio.sap_coeffs = fn
        try:
            l3, r3 = audio.unmix(M, Sres, st, fr)
        finally:
            audio.sap_coeffs = good
        err = np.abs(np.concatenate([l3 - L, r3 - R])).max()
        assert err > 0.1, name


def test_unused_band_is_identity_and_short_groups_use_their_own_offsets():
    fr = _short_4x384(4)
    off = [fr.sect_sfb_offset(g) for g in range(4)]
    n = off[3][4]
    rng = np.random.default_rng(1)
    m, s = rng.standard_normal(n), rng.standard_normal(n)
    used = [[1, 1, 0, 0]] * 4
    dpcm = [[60 + g, None, None, None] for g in range(4)]
    st = dict(sap_mode=3, sap=_sap([4] * 4, used, dpcm))
    l, r = audio.unmix(m, s, st, fr)
    for g in range(4):
        a = 0.1 * g
        lo, mid, hi = off[g][0], off[g][2], off[g][4]
        assert np.allclose(l[lo:mid], (1 + a) * m[lo:mid] + s[lo:mid])
        assert np.allclose(r[lo:mid], (1 - a) * m[lo:mid] - s[lo:mid])
        assert np.array_equal(l[mid:hi], m[mid:hi])     # sap_used 0
        assert np.array_equal(r[mid:hi], s[mid:hi])


def test_other_modes_unchanged():
    fr = _long(10)
    n = fr.sect_sfb_offset(0)[10]
    rng = np.random.default_rng(2)
    m, s = rng.standard_normal(n), rng.standard_normal(n)
    l, r = audio.unmix(m, s, dict(sap_mode=2), fr)
    assert np.array_equal(l, m + s) and np.array_equal(r, m - s)
    l, r = audio.unmix(m, s, dict(sap_mode=0), fr)
    assert l is m and r is s


# ------------------------------------------------- Table 48 parse
def test_sap_data_parse_long_frame():
    fr = _long(7)
    flags = [1, 0, 1, 1]                            # bands 0,2,4,6
    syms = [55, 70, 60]                             # for bands 0, 4, 6
    bits = "0" + "".join(map(str, flags)) + "".join(
        _code(SF_LEN, SF_CW, x) for x in syms)
    b = Bits(_bytes(bits + "1"))
    sap = C.sap_data(b, fr, SF)
    assert b.p == len(bits)                         # no delta_code_time
    assert sap["coeff_used"] == [[1, 1, 0, 0, 1, 1, 1]]
    assert sap["dpcm"] == [[55, None, None, None, 70, None, 60]]
    assert sap["delta_code_time"] == 0


def test_sap_data_parse_multi_group_reads_delta_code_time():
    fr = _short_4x384(3)
    syms = [61, 59, 62, 60, 63, 57, 64, 56]         # two coded bands x 4
    bits = "1" + "1" + "".join(_code(SF_LEN, SF_CW, x) for x in syms)
    b = Bits(_bytes(bits))
    sap = C.sap_data(b, fr, SF)
    assert b.p == len(bits)
    assert sap["coeff_all"] == 1 and sap["delta_code_time"] == 1
    assert sap["dpcm"] == [[61, None, 59], [62, None, 60], [63, None, 57],
                           [64, None, 56]]


# ------------------------------------------- A-CPL parse (Tables 60-65)
def test_acpl_config_2ch():
    cfg = AP.acpl_config_2ch(Bits(_bytes("0101")))
    assert cfg == dict(num_param_bands=12, param_band=0, quant_mode_0=0,
                       quant_mode_1=1)


def test_acpl_data_2ch_round_trip():
    books = AP.load_books(ARRAYS)
    assert len(books) == 24
    cfg = dict(num_param_bands=7, param_band=0, quant_mode_0=1,
               quant_mode_1=0)
    rng = np.random.default_rng(4)
    bits = "1" + "1" + "00011" + "10100"            # steep, 2 sets, slots
    expect = []
    for data_type, which in AP.DATA_2CH:
        q = AP.QUANT[(cfg["quant_mode_0"], cfg["quant_mode_1"])[which]]
        sets = []
        for ps in range(2):
            dt = int(rng.integers(0, 2))
            names = ([f"F0"] + ["DF"] * 6) if dt == 0 else ["DT"] * 7
            vals = []
            bits += str(dt)
            for nm in names:
                L = ARRAYS[f"ACPL_HCB_{data_type}_{q}_{nm}_LEN"]
                W = ARRAYS[f"ACPL_HCB_{data_type}_{q}_{nm}_CW"]
                sym = int(rng.integers(0, len(L)))
                vals.append(sym)
                bits += _code(L, W, sym)
            sets.append((dt, vals))
        expect.append(sets)
    b = Bits(_bytes(bits))
    got = AP.acpl_data_2ch(b, books, cfg)
    assert b.p == len(bits)
    assert got["framing"] == dict(interpolation_type=1, num_param_sets=2,
                                  param_timeslot=[3, 20])
    assert got["params"] == expect
