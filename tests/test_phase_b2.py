# SPDX-License-Identifier: Apache-2.0
"""Phase B2 tests (no captures): the channel data elements of three to
five tracks (TS 103 190-1 5.3.3.3 - 5.3.3.5, Tables 178, 179), the track
to channel mapping of the 5_X element (Table 180, with 2ch_mode), the
max_sfb_master side tracks of ASPX_ACPL_1 (Tables B.8 - B.19, 5.3.4.3.2),
streams written bit by bit (elem_writer)."""
import numpy as np
import pytest

from ac4 import channels as CH
from ac4 import multichannel as MC
from ac4 import stream as SC
import elem_writer as EW
from test_phase_a import BW

# Table 178, transcribed from the standard: rows O0..O2, columns I0..I2;
# each entry a product of a_i, b_i, c_i, d_i (i = pair) or "0" / "1"
T178 = {
    0: ["a0a1 b0a1 b1", "c0 d0 0", "a0c1 b0c1 d1"],
    1: ["d0 c0 0", "b0a1 a0a1 b1", "b0c1 a0c1 d1"],
    2: ["a0a1 b1 b0a1", "a0c1 d1 b0c1", "c0 0 d0"],
    3: ["a1 c0b1 d0b1", "0 a0 b0", "c1 c0d1 d0d1"],
    4: ["a0 0 b0", "c0b1 a1 d0b1", "c0d1 c1 d0d1"],
    5: ["a1 d0b1 c0b1", "c1 d0d1 c0d1", "0 b0 a0"],
    6: ["d0d1 c0d1 c1", "b0 a0 0", "d0b1 c0b1 a1"],
    7: ["a0 b0 0", "c0d1 d0d1 c1", "c0b1 d0b1 a1"],
    8: ["d0d1 c1 c0d1", "d0b1 a1 c0b1", "b0 0 a0"],
    9: ["d1 b0c1 a0c1", "0 d0 c0", "b1 b0a1 a0a1"],
    10: ["d0 0 c0", "b0c1 d1 a0c1", "b0a1 b1 a0a1"],
    11: ["d1 a0c1 b0c1", "b1 a0a1 b0a1", "0 c0 d0"],
}
# Table 179 (five_channel_data), chel_matsel 0, 3, 6, 9, 11 transcribed
T179 = {
    0: ["a0a1a3 b0a1a3 b1a3 a2b3 b2b3", "c0a4 d0a4 0 c2b4 d2b4",
        "a0c1 b0c1 d1 0 0", "a0a1c3 b0a1c3 b1c3 a2d3 b2d3",
        "c0c4 d0c4 0 c2d4 d2d4"],
    3: ["a1a3 c0b1a3 d0b1a3 a2b3 b2b3", "0 a0a4 b0a4 c2b4 d2b4",
        "c1 c0d1 d0d1 0 0", "a1c3 c0b1c3 d0b1c3 a2d3 b2d3",
        "0 a0c4 b0c4 c2d4 d2d4"],
    6: ["d0d1a3 c0d1a3 c1a3 a2b3 b2b3", "b0a4 a0a4 0 c2b4 d2b4",
        "d0b1 c0b1 a1 0 0", "d0d1c3 c0d1c3 c1c3 a2d3 b2d3",
        "b0c4 a0c4 0 c2d4 d2d4"],
    9: ["d1a3 b0c1a3 a0c1a3 a2b3 b2b3", "0 d0a4 c0a4 c2b4 d2b4",
        "b1 b0a1 a0a1 0 0", "d1c3 b0c1c3 a0c1c3 a2d3 b2d3",
        "0 d0c4 c0c4 c2d4 d2d4"],
    11: ["d1a3 a0c1a3 b0c1a3 a2b3 b2b3", "b1a4 a0a1a4 b0a1a4 c2b4 d2b4",
         "0 c0 d0 0 0", "d1c3 a0c1c3 b0c1c3 a2d3 b2d3",
         "b1c4 a0a1c4 b0a1c4 c2d4 d2d4"],
}
# 5.3.3.4 (four_channel_data)
T4 = ["a0a2 b0a2 a1b2 b1b2", "c0a3 d0a3 c1b3 d1b3",
      "a0c2 b0c2 a1d2 b1d2", "c0c3 d0c3 c1d3 d1d3"]


def _eval(entry, p):
    if entry == "0":
        return 0.0
    v = 1.0
    for i in range(0, len(entry), 2):
        v *= p[entry[i]][int(entry[i + 1])]
    return v


def _matrix(rows, p):
    return np.array([[_eval(e, p) for e in r.split()] for r in rows])


def _params(rng, n):
    return {k: rng.normal(size=n) for k in "abcd"}


@pytest.fixture
def scalar_pairs(monkeypatch):
    """MC._pair on scalar (a, b, c, d) 'chparam' tuples."""
    def pair(x, y, st, fr):
        a, b, c, d = st
        return a * x + b * y, c * x + d * y
    monkeypatch.setattr(MC, "_pair", pair)


@pytest.mark.parametrize("m", range(12))
def test_three_channel_matrix_is_table_178(scalar_pairs, m):
    rng = np.random.default_rng(m)
    p = _params(rng, 2)
    M = _matrix(T178[m], p)
    sts = [tuple(p[k][i] for k in "abcd") for i in range(2)]
    eye = np.eye(3)
    got = np.array(MC.matrix3([eye[0], eye[1], eye[2]], m, sts[0],
                              sts[1], None))
    assert np.allclose(got, M, rtol=0, atol=1e-12)


@pytest.mark.parametrize("m", sorted(T179))
def test_five_channel_matrix_is_table_179(scalar_pairs, m):
    rng = np.random.default_rng(100 + m)
    p = _params(rng, 5)
    M = _matrix(T179[m], p)
    sts = [tuple(p[k][i] for k in "abcd") for i in range(5)]
    eye = np.eye(5)
    got = np.array(MC.matrix5(list(eye), m, sts, None))
    assert np.allclose(got, M, rtol=0, atol=1e-12)


def test_four_channel_matrix_is_5_3_3_4(scalar_pairs):
    rng = np.random.default_rng(7)
    p = _params(rng, 4)
    sts = [tuple(p[k][i] for k in "abcd") for i in range(4)]
    got = np.array(MC.matrix4(list(np.eye(4)), sts, None))
    assert np.allclose(got, _matrix(T4, p), rtol=0, atol=1e-12)


def test_reserved_chel_matsel_is_damage():
    from ac4.robust import Damaged
    from ac4.toc import Bits
    for m in (12, 15):
        with pytest.raises(Damaged):
            MC.chel_matsel(Bits(BW().u(4, m).bytes()))


# ------------------------------------------------ Tables B.8 - B.19
def test_max_sfb_master_tables():
    T = MC.MAX_SFB_MASTER
    assert sorted(T) == [192, 240, 256, 384, 480, 512, 768, 960, 1024,
                         1536, 1920, 2048]
    assert sorted(T[1536]) == [96, 192, 384, 768]
    # spot values (Tables B.16 - B.19)
    assert [T[1536][n][31] for n in (768, 384, 192, 96)] == [21, 17, 8, 5]
    assert [T[768][n][18] for n in (384, 192, 96)] == [14, 7, 4]
    assert [T[384][n][15] for n in (192, 96)] == [8, 4]
    assert T[192][96] == (0, 1, 1, 2, 2, 3, 3, 4)
    # rows fill the n_side_bits field of the master's length (Table 106)
    for lm, cols in T.items():
        for v in cols.values():
            assert len(v) == 1 << CH.N_SIDE_BY_LEN[lm]
            assert list(v) == sorted(v)          # monotone


def test_master_framing_maps_each_block():
    class F:
        lens = [768, 384]
    f = MC.master_framing(F(), 20, 768)
    assert f.max_sfb_side == [20, 16] and f.side
    f = MC.master_framing(F(), 5, 1536)
    assert f.max_sfb_side == [3, 2]


# ------------------------------------------------ 5_X element streams
def _expect(tracks, keys, rows, p):
    """Expected per-channel spectra: the matrix (rows of the table, with
    per-band parameters p[sfb]) applied band by band."""
    n = 1536
    out = {k: np.zeros(n) for k in keys}
    X = np.array([t.deq() for t in tracks])
    for sfb in range(len(EW.SFB) - 1):
        lo, hi = EW.SFB[sfb], EW.SFB[sfb + 1]
        M = _matrix(rows, p(sfb))
        Y = M @ X[:, lo:hi]
        for k, y in zip(keys, Y):
            out[k][lo:hi] = y
    return out


def _pp(sts):
    def p(sfb):
        v = [s.abcd(sfb) for s in sts]
        return {k: [x[i] for x in v] for i, k in enumerate("abcd")}
    return p


def _simple_5x(rng, coding_config, two_ch_mode=0, matsel=0, modes=None):
    """A 5_X element in SIMPLE codec mode (no A-SPX). -> (bytes,
    expected spectra per channel)."""
    T = CH.load_tables()
    w = BW().u(3, 0)                                 # 5_X_codec_mode SIMPLE
    lfe = EW.Track(rng, 2)
    EW.lfe(w, T, lfe)
    w.u(2, coding_config)
    exp = {}
    nb = lambda: int(rng.integers(3, 12))              # noqa: E731
    modes = modes or [2, 1, 0, 2, 1]
    if coding_config == 0:
        w.u(1, two_ch_mode)
        k1, k2 = ((("L", "R"), ("Ls", "Rs")) if two_ch_mode == 0
                  else (("L", "Ls"), ("R", "Rs")))
        for keys, mode in ((k1, modes[0]), (k2, modes[1])):
            n = nb()
            tr = [EW.Track(rng, n), EW.Track(rng, n)]
            st = EW.St(rng, mode, n)
            EW.two_channel(w, T, tr, st)
            exp.update(_expect(tr, keys, ["a0 b0", "c0 d0"], _pp([st])))
        c = EW.Track(rng, nb())
        EW.mono(w, T, c)
        exp["C"] = c.deq()
    elif coding_config == 1:
        n = nb()
        tr = [EW.Track(rng, n) for _ in range(3)]
        sts = [EW.St(rng, modes[i], n) for i in range(2)]
        EW.n_channel(w, T, tr, sts, matsel)
        exp.update(_expect(tr, ("L", "R", "C"), T178[matsel], _pp(sts)))
        n = nb()
        tr = [EW.Track(rng, n), EW.Track(rng, n)]
        st = EW.St(rng, modes[2], n)
        EW.two_channel(w, T, tr, st)
        exp.update(_expect(tr, ("Ls", "Rs"), ["a0 b0", "c0 d0"],
                           _pp([st])))
    elif coding_config == 2:
        n = nb()
        tr = [EW.Track(rng, n) for _ in range(4)]
        sts = [EW.St(rng, modes[i], n) for i in range(4)]
        EW.n_channel(w, T, tr, sts)
        exp.update(_expect(tr, ("L", "R", "Ls", "Rs"), T4, _pp(sts)))
        c = EW.Track(rng, nb())
        EW.mono(w, T, c)
        exp["C"] = c.deq()
    else:
        n = nb()
        tr = [EW.Track(rng, n) for _ in range(5)]
        sts = [EW.St(rng, modes[i], n) for i in range(5)]
        EW.n_channel(w, T, tr, sts, matsel)
        exp.update(_expect(tr, ("L", "R", "C", "Ls", "Rs"), T179[matsel],
                           _pp(sts)))
    exp["lfe"] = lfe.deq()
    return EW.element(w), exp


def _spectra(frames):
    s = SC.Ac4Stream("5_X")
    wins, lengths, groups, counts = s.decode_frames(frames)
    return s, wins


@pytest.mark.parametrize("cc,mode2,matsel", [
    (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 5), (1, 0, 9), (2, 0, 0),
    (3, 0, 0), (3, 0, 6), (3, 0, 11)])
def test_simple_5x_coding_configs(cc, mode2, matsel):
    """5_X SIMPLE, every coding_config (Table 180, 2ch_mode both ways,
    several chel_matsel): each output channel's spectrum is the table's
    matrix of the tracks, band by band; nothing concealed."""
    rng = np.random.default_rng(1000 * cc + 10 * mode2 + matsel)
    frames, exps = [], []
    for i in range(3):
        sub, exp = _simple_5x(rng, cc, mode2, matsel)
        frames.append(EW.frame(sub, seq=5 + i))
        exps.append(exp)
    s, wins = _spectra(frames)
    assert s.n_bad == 0, s.status
    for i, exp in enumerate(exps):
        for k in ("L", "R", "C", "Ls", "Rs", "lfe"):
            got = np.asarray(wins[k][i])
            ref = exp[k]
            assert np.allclose(got, ref, rtol=1e-12, atol=1e-9), (i, k)


def test_2ch_mode_was_ignored_before():
    """2ch_mode 1 puts the first pair on L / Ls (Table 180): the second
    track of the first two_channel_data is Ls, not R."""
    rng = np.random.default_rng(3)
    sub, exp = _simple_5x(rng, 0, 1, modes=[0, 0])
    s, wins = _spectra([EW.frame(sub)])
    assert np.allclose(wins["Ls"][0], exp["Ls"])
    assert not np.allclose(exp["Ls"], exp["R"])


def test_simple_5x_decodes_to_pcm():
    """The whole decoder on a SIMPLE 5_X stream: six channels, nothing
    concealed, the centre's level as its spectrum's."""
    from ac4.decoder import Decoder
    rng = np.random.default_rng(11)
    frames = [EW.frame(_simple_5x(rng, 3, 0, 6)[0], seq=5 + i)
              for i in range(6)]
    dec = Decoder("5_X", guard=False)
    y = dec.decode_pass(frames)
    assert dec.n_concealed == 0 and y.shape == (6 * 1536, 6)
    assert all(np.abs(y[:, c]).max() > 0 for c in range(6))


# ------------------------------------------ 5_X ASPX_ACPL_1 (residuals)
ASPX_CFG = dict(quant_mode_env=0, start_freq=2, stop_freq=1,
                master_freq_scale=1, interpolation=0, preflat=0, limiter=1,
                noise_sbg=1, num_env_bits_fixfix=0, freq_res_mode=1)


def _books():
    from ac4 import acpl_parse as AP
    from ac4 import hcb as H
    from ac4 import spectral as S
    arrays = H.parse_c(H.DEFAULT_C)
    abooks = {n[:-4]: S.Huff(arrays[n], arrays[n[:-4] + "_CW"])
              for n in arrays if n.endswith("_LEN") and n.startswith("ASPX_")}
    return AP.load_books(arrays), abooks


def _aspx_cfg_bits(w, cfg):
    for k in ("quant_mode_env", "start_freq", "stop_freq",
              "master_freq_scale", "interpolation", "preflat", "limiter",
              "noise_sbg", "num_env_bits_fixfix", "freq_res_mode"):
        w.u({"start_freq": 3, "stop_freq": 2, "noise_sbg": 2,
             "freq_res_mode": 2}.get(k, 1), cfg[k])


def _ec(w, abooks, t, n_ch):
    """ec_data of n_ch channels, signal then noise (frequency coded)."""
    dfc = (len(abooks["ASPX_HCB_ENV_LEVEL_15_DF"].map) - 1) // 2
    nfc = (len(abooks["ASPX_HCB_NOISE_LEVEL_DF"].map) - 1) // 2
    for _ in range(n_ch):
        EW.huff_put(w, abooks["ASPX_HCB_ENV_LEVEL_15_F0"], 30)
        for _ in range(t["n_lo"] - 1):
            EW.huff_put(w, abooks["ASPX_HCB_ENV_LEVEL_15_DF"], dfc)
    for _ in range(n_ch):
        EW.huff_put(w, abooks["ASPX_HCB_NOISE_LEVEL_F0"], 10)
        for _ in range(t["n_noise"] - 1):
            EW.huff_put(w, abooks["ASPX_HCB_NOISE_LEVEL_DF"], nfc)


def _aspx_1ch(w, abooks, t, iframe):
    if iframe:
        w.u(3, 0)                                    # xover offset
    w.u(1, 0).u(1, 0)                                # FIXFIX, one envelope
    w.u(1, 0).u(1, 0)                                # delta_dir: freq
    for _ in range(t["n_noise"]):
        w.u(2, 1)                                    # aspx_tna_mode
    w.u(1, 0).u(1, 0).u(1, 0)                        # no ah, fic, tic
    _ec(w, abooks, t, 1)


def _aspx_2ch(w, abooks, t, iframe):
    if iframe:
        w.u(3, 0)
    w.u(1, 0).u(1, 0)                                # ch 0: FIXFIX, 1 env
    w.u(1, 0)                                        # aspx_balance 0
    w.u(1, 0).u(1, 0)                                # ch 1: FIXFIX, 1 env
    w.u(2, 0).u(2, 0)                                # delta_dir both
    for _ in range(2 * t["n_noise"]):
        w.u(2, 1)                                    # tna modes
    w.u(1, 0).u(1, 0).u(1, 0).u(1, 0)                # ah l/r, fic, tic
    _ec(w, abooks, t, 2)


def _acpl_1ch(w, books, rng, nb, start):
    w.u(1, 0).u(1, 0)                                # smooth, one set
    for name, f0, off in (("ALPHA", 22, 32), ("BETA", 3, 8)):
        w.u(1, 0)                                    # frequency coded
        EW.huff_put(w, books[f"ACPL_HCB_{name}_FINE_F0"], f0)
        for _ in range(start + 1, nb):
            EW.huff_put(w, books[f"ACPL_HCB_{name}_FINE_DF"],
                        off + int(rng.integers(-1, 2)))


def _acpl1_5x(rng, iframe=True, coding_config=0, master=6, side_mode=2):
    """A 5_X element in ASPX_ACPL_1, written bit by bit. -> (bytes,
    expected spectra of L, Ls, R, Rs after the side processing)."""
    from ac4 import acpl
    from ac4 import aspx_bands as A
    T = CH.load_tables()
    books, abooks = _books()
    t = A.tables_for(ASPX_CFG, 0)
    nb, qb = 9, 4
    w = BW().u(3, 2)                                 # ASPX_ACPL_1
    if iframe:
        _aspx_cfg_bits(w, ASPX_CFG)
        w.u(2, 2).u(1, 0).u(3, qb - 1)               # 9 bands, fine, qmf 4
    EW.lfe(w, T, EW.Track(rng, 1))
    w.u(1, 1).u(1, 1)                                # companding: sync, on
    w.u(1, coding_config)
    n = 20
    if coding_config == 0:
        tr = [EW.Track(rng, n), EW.Track(rng, n)]
        st = EW.St(rng, 1, n)
        EW.two_channel(w, T, tr, st)
        A_, B_ = _expect(tr, ("A", "B"), ["a0 b0", "c0 d0"],
                         _pp([st])).values()
    else:
        tr = [EW.Track(rng, n) for _ in range(3)]
        sts = [EW.St(rng, 2, n), EW.St(rng, 1, n)]
        EW.n_channel(w, T, tr, sts, 4)
        A_, B_, _c = _expect(tr, ("A", "B", "C"), T178[4],
                             _pp(sts)).values()
    w.u(5, master)                                   # max_sfb_master
    ss = [EW.St(rng, side_mode, n), EW.St(rng, 1, n)]
    for s in ss:
        s.write(w)
    res = [EW.Track(rng, master), EW.Track(rng, master)]
    for r in res:
        r.write(w, T)
    if coding_config == 0:
        EW.mono(w, T, EW.Track(rng, 8))
    _aspx_2ch(w, abooks, t, iframe)
    _aspx_1ch(w, abooks, t, iframe)
    start = int(acpl.sb_to_pb(nb)[qb])
    for _ in range(2):
        _acpl_1ch(w, books, rng, nb, start)
    exp = {}
    for (k0, k1), core, r, s in ((("L", "Ls"), A_, res[0], ss[0]),
                                 (("R", "Rs"), B_, res[1], ss[1])):
        class Tr:
            pass
        ta, tb = Tr(), Tr()
        ta.deq, tb.deq = (lambda c=core: c), r.deq
        exp.update(_expect([ta, tb], (k0, k1), ["a0 b0", "c0 d0"],
                           _pp([s])))
    return EW.element(w), exp


@pytest.mark.parametrize("cc,side_mode", [(0, 2), (0, 1), (1, 2), (1, 0)])
def test_acpl1_5x_side_tracks(cc, side_mode):
    """ASPX_ACPL_1 in 5_X (coding_config 0 and 1): every frame closes on
    audio_size (nothing concealed), and L, Ls, R, Rs are the side
    processing of 5.3.4.3.2 of the A / B tracks and the two
    max_sfb_master tracks."""
    rng = np.random.default_rng(50 + cc + 10 * side_mode)
    frames, exps = [], []
    for i in range(4):
        sub, exp = _acpl1_5x(rng, iframe=(i == 0), coding_config=cc,
                             side_mode=side_mode)
        frames.append(EW.frame(sub, seq=5 + i, iframe=int(i == 0)))
        exps.append(exp)
    s, wins = _spectra(frames)
    assert s.n_bad == 0, s.status
    assert s.acpl1 == "5_X"
    for i, exp in enumerate(exps):
        for k in ("L", "Ls", "R", "Rs"):
            assert np.allclose(np.asarray(wins[k][i]), exp[k], rtol=1e-12,
                               atol=1e-9), (i, k)


def test_acpl1_5x_decodes_with_the_residual():
    """The whole decoder: six channels, nothing concealed; below
    acpl_qmf_band Ls carries the residual (it differs from the same
    stream with the residual tracks zeroed)."""
    from ac4.decoder import Decoder
    outs = []
    for zero in (False, True):
        rng = np.random.default_rng(77)
        frames = []
        for i in range(8):
            sub, _ = _acpl1_5x(rng, iframe=(i == 0), side_mode=0,
                               master=0 if zero else 6)
            frames.append(EW.frame(sub, seq=5 + i, iframe=int(i == 0)))
        dec = Decoder("5_X", guard=False)
        dec.hf_ratio = None
        y = dec.decode_pass(frames)
        assert dec.n_concealed == 0, dict(dec.concealed.reasons)
        outs.append(y)
    s = slice(3 * 1536, None)
    ls, ls0 = outs[0][s, 4], outs[1][s, 4]
    assert np.sqrt(np.mean(ls ** 2)) > 0
    assert np.abs(ls - ls0).max() > 0.01 * np.abs(ls).max()


# ------------------------------------------------ the C decoder
C_CLI = __import__("os").path.join(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__)),
    "..", "c", "ac4dec-c")


def _c_vs_python(tmp_path, frames, channels=6, element="5_X", ch_mode=None):
    import os
    import subprocess
    from ac4.decoder import Decoder
    from test_inputs import syncframe
    if not os.path.exists(C_CLI):
        pytest.skip("c/ac4dec-c not built")
    src = tmp_path / "e.ac4"
    src.write_bytes(b"".join(syncframe(f) for f in frames))
    raw = tmp_path / "e.f64"
    subprocess.run([C_CLI, str(src), str(tmp_path / "e.wav"),
                    "--legacy-level", "--f64", str(raw), "--internal",
                    "--no-guard", "-q", "--channels", str(channels)],
                   check=True)
    got = np.fromfile(raw, "<f8").reshape(-1, channels)
    dec = Decoder(element, channels=channels, guard=False, ch_mode=ch_mode)
    dec.hf_ratio = None
    ref = dec.decode_pass(frames)
    assert dec.n_concealed == 0
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()


@pytest.mark.parametrize("cc,mode2,matsel", [(0, 1, 0), (1, 0, 9), (2, 0, 0),
                                             (3, 0, 11)])
@pytest.mark.parametrize("channels", [6, 2])
def test_c_simple_5x(tmp_path, cc, mode2, matsel, channels):
    rng = np.random.default_rng(3000 + cc)
    frames = [EW.frame(_simple_5x(rng, cc, mode2, matsel)[0], seq=5 + i)
              for i in range(5)]
    _c_vs_python(tmp_path, frames, channels)


@pytest.mark.parametrize("cc,side_mode", [(0, 2), (1, 1), (1, 0)])
@pytest.mark.parametrize("channels", [6, 2])
def test_c_acpl1_5x(tmp_path, cc, side_mode, channels):
    rng = np.random.default_rng(4000 + cc)
    frames = []
    for i in range(8):
        sub, _ = _acpl1_5x(rng, iframe=(i == 0), coding_config=cc,
                           side_mode=side_mode)
        frames.append(EW.frame(sub, seq=5 + i, iframe=int(i == 0)))
    _c_vs_python(tmp_path, frames, channels)
