# SPDX-License-Identifier: Apache-2.0
"""Phase C (no captures): the immersive_channel_element (TS 103 190-2
6.2.4.1) in immersive_codec_mode ASPX_ACPL_2, written bit by bit, and its
A-CPL (5.5.2, Pseudocode 2)."""
import numpy as np
import pytest

from ac4 import acpl as ACPL
from ac4 import channels as CH
from ac4 import elements as EL
from ac4 import stream as SC
import elem_writer as EW
from test_phase_a import BW
from test_phase_b2 import (T178, T179, T4, _expect, _pp, _books, ASPX_CFG,
                           _aspx_cfg_bits, _aspx_1ch, _aspx_2ch, _acpl_1ch,
                           C_CLI)

PAIR = ["a0 b0", "c0 d0"]
# Table 56 codes
CODE = {11: "11111100", 12: "11111101"}


def _ice(rng, ch_mode=12, iframe=True, grouping=0, mode2=0, sap_add=False,
         matsel=0, acpl=True):
    """An immersive_channel_element (b_5fronts 0) in ASPX_ACPL_2. ->
    (bytes, expected spectra of L R C [lfe] Ls Rs Tfl Tfr after SMP)."""
    from ac4 import aspx_bands as A
    T = CH.load_tables()
    books, abooks = _books()
    t = A.tables_for(ASPX_CFG, 0)
    nb = 9
    w = BW().u(1, 0).u(2, 3)                         # ASPX_ACPL_2 (0b011)
    if iframe:
        _aspx_cfg_bits(w, ASPX_CFG)
        w.u(2, 2).u(1, 0)                            # 9 bands, fine (FULL)
    pre = {}
    if ch_mode == 12:
        lfe = EW.Track(rng, 2)
        EW.lfe(w, T, lfe)
        pre["lfe"] = lfe.deq()
    w.u(2, grouping)
    n = int(rng.integers(3, 12))
    mono_c = False
    if grouping == 0:
        w.u(1, mode2)
        keys = [("A", "B"), ("D", "E")] if mode2 == 0 else \
            [("A", "D"), ("B", "E")]
        for kk in keys:
            tr = [EW.Track(rng, n), EW.Track(rng, n)]
            st = EW.St(rng, 1, n)
            EW.two_channel(w, T, tr, st)
            pre.update(_expect(tr, kk, PAIR, _pp([st])))
        mono_c = True
    elif grouping == 1:
        tr = [EW.Track(rng, n) for _ in range(3)]
        sts = [EW.St(rng, 2, n), EW.St(rng, 1, n)]
        EW.n_channel(w, T, tr, sts, matsel)
        pre.update(_expect(tr, ("A", "B", "C"), T178[matsel], _pp(sts)))
        tr = [EW.Track(rng, n), EW.Track(rng, n)]
        st = EW.St(rng, 2, n)
        EW.two_channel(w, T, tr, st)
        pre.update(_expect(tr, ("D", "E"), PAIR, _pp([st])))
    elif grouping == 2:
        tr = [EW.Track(rng, n) for _ in range(4)]
        sts = [EW.St(rng, m, n) for m in (2, 1, 0, 2)]
        EW.n_channel(w, T, tr, sts)
        pre.update(_expect(tr, ("A", "B", "D", "E"), T4, _pp(sts)))
        mono_c = True
    else:
        tr = [EW.Track(rng, n) for _ in range(5)]
        sts = [EW.St(rng, m, n) for m in (2, 1, 0, 2, 1)]
        EW.n_channel(w, T, tr, sts, matsel)
        pre.update(_expect(tr, ("A", "B", "C", "D", "E"), T179[matsel],
                           _pp(sts)))
    if mono_c:
        c = EW.Track(rng, 6)
        EW.mono(w, T, c)
        pre["C"] = c.deq()
    w.u(1, int(sap_add))                             # b_use_sap_add_ch
    sap = [EW.St(rng, 1, n), EW.St(rng, 2, n)] if sap_add else None
    if sap_add:
        for s in sap:
            s.write(w)
    tr = [EW.Track(rng, n), EW.Track(rng, n)]
    st = EW.St(rng, 1, n)
    EW.two_channel(w, T, tr, st)
    pre.update(_expect(tr, ("F", "G"), PAIR, _pp([st])))
    for _ in range(3):
        _aspx_2ch(w, abooks, t, iframe)              # (A, B), (D, E), (F, G)
    _aspx_1ch(w, abooks, t, iframe)                  # C
    if acpl:
        for _ in range(4):
            _acpl_1ch(w, books, rng, nb, 0)
    exp = {"L": pre["A"], "R": pre["B"], "C": pre["C"]}
    if ch_mode == 12:
        exp["lfe"] = pre["lfe"]
    for i, ((x, y), (u, v)) in enumerate(((("D", "F"), ("Ls", "Tfl")),
                                          (("E", "G"), ("Rs", "Tfr")))):
        class Tr:
            pass
        ta, tb = Tr(), Tr()
        ta.deq, tb.deq = (lambda a=pre[x]: a), (lambda a=pre[y]: a)
        p = _pp([sap[i]]) if sap_add else \
            (lambda sfb: {k: [d] for k, d in zip("abcd", (1, 0, 0, 1))})
        exp.update(_expect([ta, tb], (u, v), PAIR, p))
    return EW.element(w), exp


def _ice_frames(rng, n=4, ch_mode=12, top=(1, 1, 3), **kw):
    frames, exps = [], []
    for i in range(n):
        sub, exp = _ice(rng, ch_mode, iframe=(i == 0), **kw)
        frames.append(EW.frame(sub, seq=5 + i, ch_mode_code=CODE[ch_mode],
                               iframe=int(i == 0), top=top))
        exps.append(exp)
    return frames, exps


@pytest.mark.parametrize("ch_mode", [11, 12])
@pytest.mark.parametrize("grouping,mode2,matsel,sap", [
    (0, 0, 0, False), (0, 1, 0, True), (1, 0, 7, True), (2, 0, 0, False),
    (3, 0, 3, True)])
def test_ice_tracks(ch_mode, grouping, mode2, matsel, sap):
    """Every core grouping of Table 19 (2ch_mode included), the
    b_use_sap_add_ch step of 5.2.3.2 on (D, F), (E, G): the spectra after
    SMP are the tracks' as the tables say, and every frame closes on
    audio_size with its A-SPX and four acpl_data_1ch()."""
    rng = np.random.default_rng(1000 * ch_mode + 10 * grouping + mode2)
    frames, exps = _ice_frames(rng, ch_mode=ch_mode, grouping=grouping,
                               mode2=mode2, matsel=matsel, sap_add=sap)
    s = SC.Ac4Stream("ice", ch_mode=ch_mode)
    wins, lengths, groups, counts = s.decode_frames(frames)
    assert s.n_bad == 0, s.status
    assert s.acpl1 == "ice"
    for i, exp in enumerate(exps):
        for k, ref in exp.items():
            assert np.allclose(np.asarray(wins[k][i]), ref, rtol=1e-12,
                               atol=1e-9), (i, k)
        if i:
            assert len(groups["acpl"][i]) == 4
            for g in ("lr", "sr", "c", "xr"):
                assert groups[g][i] not in (None, SC.CONCEALED), (i, g)


def test_ice_other_modes_are_concealed():
    """SCPL, ASPX_SCPL, ASPX_ACPL_1, ASPX_AJCC and 9.X.4 are not built:
    concealed as element.parse, not misdecoded."""
    T = CH.load_tables()
    for bits, ch_mode in (((1, 0, 2, 0), 12), ((1, 0, 2, 1), 12),
                          ((1, 0, 2, 2), 12), ((1, 1), 12),
                          ((1, 0, 2, 3), 14)):
        w = BW()
        for i in range(0, len(bits), 2):
            w.u(bits[i], bits[i + 1])
        w.u(8, 0)
        sub = EW.element(w)
        with pytest.raises(Exception) as e:
            EL.immersive_channel_element(sub, T, ch_mode, b_iframe=True)
        assert getattr(e.value, "reason", "") == "element.parse"


def test_ice_channels():
    assert EL.ice_channels(12, (1, 1, 3)) == EL.ICE_FULL
    assert EL.ice_channels(12, (0, 1, 3)) == (
        "L", "R", "C", "lfe", "Ls", "Rs", "Tfl", "Tfr", "Tbl", "Tbr")
    assert EL.ice_channels(11, (0, 0, 0)) == ("L", "R", "C", "Ls", "Rs")
    assert EL.ice_channels(12, (1, 1, 1))[-2:] == ("Tfl", "Tfr")
    assert EL.ice_channels(12, (1, 1, 2))[-2:] == ("Tbl", "Tbr")


def test_two_instances_of_one_decorrelator():
    """Pseudocode 2 runs D0 on both Ls and Rs (and D1 on both tops): each
    instance is the decorrelator alone on its own input."""
    rng = np.random.default_rng(2)
    v = (rng.standard_normal((3, 40, 64))
         + 1j * rng.standard_normal((3, 40, 64)))
    both = ACPL.decorrelate(v.copy(), ACPL.new_state(), rows=(0, 0, 1))
    for inst, d in ((0, 0), (1, 0), (2, 1)):
        x = np.zeros_like(v)
        x[d] = v[inst]
        alone = ACPL.decorrelate(x, ACPL.new_state())[d]
        assert np.allclose(both[inst], alone, atol=1e-12)
    # and the default rows are the three decorrelators
    assert np.array_equal(ACPL.decorrelate(v.copy(), ACPL.new_state()),
                          ACPL.decorrelate(v.copy(), ACPL.new_state(),
                                           rows=(0, 1, 2)))


def _pseudocode2_module(x, frames, d, nts=24):
    """One ASPX_ACPL_2 module of Pseudocode 2 written out per time slot
    (TS 103 190-1 5.7.7.4 / 5.7.7.5): x (64, n) the module's A-SPX
    output, frames per frame dict(q, cfg) (one smooth parameter set), d
    the decorrelator. -> (z0, z1) times sqrt 2."""
    n = x.shape[1]
    s = ACPL.new_state()
    v = np.zeros((3, n, 64), complex)
    v[d] = 2.0 * x.T
    y = ACPL.duck(ACPL.decorrelate(v, s), s)[d]
    z0 = np.zeros((n, 64), complex)
    z1 = np.zeros((n, 64), complex)
    prev = None
    for f, fr in enumerate(frames):
        dq = ACPL.dequantise_1ch(fr["q"], fr["cfg"])
        pb = ACPL.sb_to_pb(fr["cfg"]["num_param_bands"])
        a1, b1 = dq["a"][0][pb], dq["b"][0][pb]
        if prev is None:
            prev = (a1, b1)
        for ts in range(nts):
            g = (ts + 1) / nts
            a = prev[0] + g * (a1 - prev[0])
            b = prev[1] + g * (b1 - prev[1])
            t = f * nts + ts
            xin = 2.0 * x[:, t]
            z0[t] = 0.5 * (xin * (1.0 + a) + y[t] * b)
            z1[t] = 0.5 * (xin * (1.0 - a) - y[t] * b)
        prev = (a1, b1)
    return z0.T * np.sqrt(2.0), z1.T * np.sqrt(2.0)


def test_ice_upmix_is_pseudocode_2():
    """The decoder's immersive upmix of one pair (Ls / Rs, both on D0)
    against Pseudocode 2 written out: z5 .. z8 times sqrt 2, Lb / Rb the
    modules' second outputs."""
    from ac4 import aspx_render as R
    from ac4 import aspx_bands as A
    from ac4 import qmf as Q
    rng = np.random.default_rng(9)
    n_f, nts = 3, 24
    QL = rng.standard_normal((64, n_f * nts)) + 1j * rng.standard_normal(
        (64, n_f * nts))
    QR = rng.standard_normal((64, n_f * nts)) + 1j * rng.standard_normal(
        (64, n_f * nts))
    cfg = dict(num_param_bands=9, param_band=0, qmf_band=0, quant_mode=0,
               mode=1)
    groups = []
    for f in range(n_f):
        mods = []
        for m in range(2):
            qa = rng.integers(-8, 9, (1, 9))
            qb = rng.integers(0, 5, (1, 9))
            mods.append(dict(cfg=cfg, q=(qa, qb), interp=0, timeslots=()))
        groups.append(mods)
    zs = ACPL.render_1ch([QL, QR], [None, None], groups, ts_offset=0,
                         nts=nts, dec=(0, 0))
    for m, X in enumerate((QL, QR)):
        e0, e1 = _pseudocode2_module(X, [g[m] for g in groups], 0, nts)
        assert np.allclose(zs[m][0] * np.sqrt(2.0), e0, atol=1e-9)
        assert np.allclose(zs[m][1] * np.sqrt(2.0), e1, atol=1e-9)
    # through the renderer: names and the sqrt 2
    geo = A.geometry(1536)
    up = R._upmix(QL, QR, groups, None, None, Q.qwin(),
                  dict(geo, ts_offset=0),
                  dict(res=[None, None], split=False, five=False,
                       names=(("Ls", "Lb"), ("Rs", "Rb")), dec=(0, 0)))
    assert set(up) == {"Ls", "Lb", "Rs", "Rb"}


def test_ice_decodes_to_pcm():
    """The whole decoder: 7.1.4 (12 channels) and, with the back pair
    signalled absent, 5.1.4 (10); L / R / C carry twice their A-SPX
    output (Pseudocode 2: z = 2 x); nothing concealed."""
    from ac4.decoder import (Decoder, probe_element, probe_ch_mode,
                             probe_present)
    rng = np.random.default_rng(12)
    frames, _ = _ice_frames(rng, n=6)
    assert probe_element(frames) == "ice"
    assert probe_ch_mode(frames) == 12
    assert probe_present(frames) == (1, 1, 3)
    dec = Decoder("ice", guard=False, ch_mode=12, present=(1, 1, 3))
    dec.hf_ratio = None
    y = dec.decode_pass(frames)
    assert dec.n_concealed == 0, dict(dec.concealed.reasons)
    assert y.shape == (6 * 1536, 12)
    assert dec.chans == EL.ICE_FULL
    assert all(np.abs(y[3 * 1536:, c]).max() > 0 for c in range(12))
    dec5 = Decoder("ice", guard=False, ch_mode=12, present=(0, 1, 3))
    dec5.hf_ratio = None
    y5 = dec5.decode_pass(frames)
    assert y5.shape == (6 * 1536, 10)
    sel = [EL.ICE_FULL.index(k) for k in dec5.chans]
    assert np.array_equal(y5, y[:, sel])
    # stereo: L / R only, the same samples as in the full decode
    dec2 = Decoder("ice", channels=2, guard=False, ch_mode=12)
    dec2.hf_ratio = None
    y2 = dec2.decode_pass(frames)
    assert np.allclose(y2, y[:, :2], atol=1e-9 * np.abs(y).max())


# ------------------------------------------------ the C decoder
def _c_vs_python_ice(tmp_path, frames, channels, present):
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
    dec = Decoder("ice", channels=channels, guard=False, ch_mode=12,
                  present=present)
    dec.hf_ratio = None
    ref = dec.decode_pass(frames)
    assert dec.n_concealed == 0
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()


@pytest.mark.parametrize("grouping,sap,top", [
    (0, False, (1, 1, 3)), (1, True, (0, 1, 3)), (3, True, (1, 1, 2))])
def test_c_ice(tmp_path, grouping, sap, top):
    """The C decoder against the Python on synthetic 7.1.4 streams (the
    A-SPX of the fourth pair, the four A-CPL modules, the output
    selection by the presence flags): within 1e-9."""
    rng = np.random.default_rng(60 + grouping)
    frames, _ = _ice_frames(rng, n=8, grouping=grouping, sap_add=sap,
                            top=top)
    n = len(EL.ice_channels(12, top))
    _c_vs_python_ice(tmp_path, frames, n, top)
    _c_vs_python_ice(tmp_path, frames, 2, top)


# ------------------------------------------------ rendering (render.py)
def test_render_tables():
    """Entries of TS 103 190-2 Tables 39, 41, 43 with Table 130's
    defaults and with sent gains, and TS 103 190-1 Tables 217, 218 and
    6.2.17.6."""
    from ac4 import render as RD
    m3 = 10 ** (-3 / 20)
    c514 = ("L", "R", "C", "lfe", "Ls", "Rs", "Tfl", "Tfr", "Tbl", "Tbr")
    R = RD.matrix(c514, "5.1")
    # 5.X.4 -> 5.X.0: Ls = Ls + gain_t2b Tfl + gain_t2e Tbl (-3 dB each)
    assert np.allclose(R[4], [0, 0, 0, 0, 1, 0, m3, 0, m3, 0])
    assert np.allclose(R[0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    c714 = EL.ICE_FULL
    cd = dict(bs_ch_config=1, configs={0: dict(gain_b=-6.0, gain_t2a=0.0,
                                               gain_t2b=math_inf())})
    R = RD.matrix(c714, "5.1", cd)
    i = {k: n for n, k in enumerate(c714)}
    g6 = 10 ** (-6 / 20)
    assert np.isclose(R[4, i["Ls"]], g6) and np.isclose(R[4, i["Lb"]], g6)
    assert np.isclose(R[0, i["Tfl"]], 1.0) and R[4, i["Tfl"]] == 0
    assert np.isclose(R[4, i["Tbl"]], m3)          # gain_t2e default
    # 7.X.4 -> 7.X.2: Tsl = gain_t1 (Tfl + Tbl)
    R = RD.matrix(c714, "7.1.2", dict(bs_ch_config=1,
                                      configs={4: dict(gain_t1=-1.5)}))
    assert np.isclose(R[8, i["Tfl"]], 10 ** (-1.5 / 20))
    assert np.isclose(R[8, i["Tbl"]], 10 ** (-1.5 / 20))
    # 5.1 -> 2.0, LoRo with the stream's LFE gain (5.5 - 13 dB)
    c51 = ("L", "R", "C", "lfe", "Ls", "Rs")
    R = RD.matrix(c51, "2.0", dict(bs_ch_config=-1, configs={},
                                   stereo=dict(loro_centre=-3.0,
                                               loro_surround=-1.5,
                                               lfe_mixgain=13)))
    assert np.allclose(R[0], [1, 0, m3, 10 ** (-7.5 / 20),
                              10 ** (-1.5 / 20), 0])
    assert np.allclose(RD.matrix(("C",), "2.0"), [[m3], [m3]])
    assert np.allclose(RD.matrix(("L", "R", "C"), "2.0"),
                       [[1, 0, m3], [0, 1, m3]])
    # 5.1 into 7.1.4: routed, nothing invented
    R = RD.matrix(c51, "7.1.4")
    assert R.shape == (12, 6) and np.array_equal(R[:6], np.eye(6))
    assert not R[6:].any()


def math_inf():
    import math
    return -math.inf


def test_custom_dmx_data_bits():
    """custom_dmx_data (TS 103 190-2 6.2.9.2) written bit by bit: a 7.1.4
    presentation (bs_ch_config 1) sending out_ch_config 0 (tool_t4_to_f_s
    and tool_b4_to_b2) and the stereo coefficients."""
    from ac4 import metadata as MD
    from ac4.toc import Bits
    w = BW()
    w.u(1, 1).u(2, 0)                   # b_cdmx_data_present, one config
    w.u(3, 0)                           # out_ch_config 0 (5.X.0)
    w.u(1, 1).u(3, 2)                   # top front to front, t2a -3 dB
    w.u(1, 0).u(3, 4)                   # top back: t2e -6 dB
    w.u(3, 1)                           # gain_b -1.5 dB
    w.u(1, 1).u(3, 3).u(3, 5)           # stereo: loro c -1.5, s -4.5
    w.u(1, 0)                           # no ltrt
    w.u(1, 1).u(5, 9)                   # lfe_mixgain 9
    w.u(2, 1)                           # preferred: loro
    b = Bits(w.align().bytes())
    c = MD.custom_dmx_data(b, 12, 12, 1, 2, 1)
    assert c["bs_ch_config"] == 1
    p = c["configs"][0]
    assert p["gain_t2a"] == -3.0 and p["gain_t2b"] == float("-inf")
    assert p["gain_t2e"] == -6.0 and p["gain_b"] == -1.5
    assert c["stereo"]["loro_centre"] == -1.5
    assert c["stereo"]["loro_surround"] == -4.5
    assert c["stereo"]["lfe_mixgain"] == 9
    assert MD.pres_modes_of([dict(channel_mode=12,
                                  b_4_back_channels_present=0,
                                  top_channels_present=3)]) == (12, 12, 0,
                                                                2, 1)


def test_decoder_layout():
    """Decoder(layout=): the decoded channels through render.matrix."""
    from ac4.decoder import Decoder
    from ac4 import render as RD
    rng = np.random.default_rng(13)
    frames, _ = _ice_frames(rng, n=5, top=(0, 1, 3))
    d0 = Decoder("ice", guard=False, ch_mode=12, present=(0, 1, 3))
    d0.hf_ratio = None
    y = d0.decode_pass(frames)
    for lay in ("2.0", "5.1", "7.1.4"):
        d = Decoder("ice", guard=False, ch_mode=12, present=(0, 1, 3),
                    layout=lay)
        d.hf_ratio = None
        z = d.decode_pass(frames)
        assert d.chans == RD.LAYOUTS[lay]
        assert np.allclose(z, y @ RD.matrix(d0.chans, lay).T, atol=1e-6)


@pytest.mark.parametrize("layout", ["2.0", "5.1", "7.1.2"])
def test_c_layout(tmp_path, layout):
    """The C renderer (--layout) against the Python on a synthetic 5.1.4
    stream."""
    import os
    import subprocess
    from ac4.decoder import Decoder
    from test_inputs import syncframe
    if not os.path.exists(C_CLI):
        pytest.skip("c/ac4dec-c not built")
    rng = np.random.default_rng(14)
    frames, _ = _ice_frames(rng, n=6, top=(0, 1, 3))
    src = tmp_path / "e.ac4"
    src.write_bytes(b"".join(syncframe(f) for f in frames))
    raw = tmp_path / "e.f64"
    subprocess.run([C_CLI, str(src), str(tmp_path / "e.wav"),
                    "--legacy-level", "--f64", str(raw), "--internal",
                    "--no-guard", "-q", "--layout", layout], check=True)
    dec = Decoder("ice", guard=False, ch_mode=12, present=(0, 1, 3),
                  layout=layout)
    dec.hf_ratio = None
    ref = dec.decode_pass(frames)
    got = np.fromfile(raw, "<f8").reshape(-1, len(dec.chans))
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()


# ------------------------------------------------ OAMD, A-JOC side info
def test_oamd_object_blocks():
    """object_info_block (TS 103 190-2 6.2.8.5 - 6.2.8.11) written bit by
    bit: an I-frame block with gain and position, a differential position
    update, a reuse, extended precision, an inactive object."""
    import math
    from ac4 import oamd as O
    from ac4.toc import Bits
    w = BW()
    # block 1 (no delta, dynamic): active, basic info: gain 0b0 + value 3
    # (12 dB), position x 10 y 50 z + 9
    w.u(1, 0)                                  # b_object_not_active
    w.u(1, 0).u(1, 0).u(1, 0).u(6, 3)          # not default, md 0b0, gain
    w.u(6, 10).u(6, 50).u(1, 1).u(4, 9)        # pos3D_X Y, Z sign, Z
    w.u(1, 1)                                  # b_grouped_zone_defaults
    w.u(1, 1)                                  # b_grouped_other_defaults
    w.u(1, 0)                                  # b_add_table_data
    # block 2 (delta allowed): basic reuse, render partial reuse with a
    # differential position (+3, -4, -1) and ext precision on X (+2)
    w.u(1, 0).u(1, 1)                          # active, b_basic_info_reuse
    w.u(1, 0).u(1, 1)                          # not reuse, partial reuse
    w.u(1, 0).u(1, 0).u(1, 1)                  # mask: position only
    w.u(1, 1).u(3, 3).u(3, 4).u(3, 7)          # diff coding: 3, -4, -1
    w.u(1, 1).u(4, 0)                          # add_table_data, 1 byte
    w.u(1, 0).u(1, 1).u(3, 0b100).u(2, 1)      # trim, ext X = 2
    w.u(1, 0)                                  # b_headphone (8 bits)
    # block 3: inactive
    w.u(1, 1).u(1, 0)
    b = Bits(w.align().bytes())
    st = O.ObjState()
    O.object_info_block(b, st, True, True)
    assert st.active and st.gain == 12.0
    assert st.position() == (10 / 62, 50 / 62, 9 / 15)
    O.object_info_block(b, st, False, True)
    assert st.gain == 12.0
    assert np.allclose(st.position(), (13 / 62 + 2 / 310, 46 / 62, 8 / 15))
    O.object_info_block(b, st, False, True)
    assert not st.active and st.gain == -math.inf
    assert st.position() == (0.5, 0.5, 0.0)


def test_oamd_timing():
    from ac4 import oamd as O
    from ac4.toc import Bits
    w = BW().u(1, 1).u(1, 0).u(1, 1).u(1, 1)    # 0b10, code 0b11: 24
    w.u(3, 2)
    w.u(6, 5).u(2, 1)                            # 512
    w.u(6, 9).u(2, 3).u(1, 0).u(11, 1000)
    t = O.oamd_timing_data(Bits(w.align().bytes()))
    assert t == dict(sample_offset=24, blocks=[(5, 512), (9, 1000)])


def _ajoc_put(w, dt, qs, kind, val):
    from ac4 import ajoc as AJ
    h, off = AJ.books()[(dt, "COARSE" if qs else "FINE", kind)]
    EW.huff_put(w, h, val + off)


def test_ajoc_side_info():
    """ajoc() (6.2.5) written with the Annex A.1.1 codebooks: control
    info, a sparse object, frequency then time differential data;
    Pseudocode 16 and Tables 29 - 32."""
    from ac4 import ajoc as AJ
    from ac4.toc import Bits
    n_dmx, n_umx = 2, 2
    w = BW()
    w.u(3, 1).u(1, 1)                            # one decorrelator, on
    w.u(1, 1).u(1, 1)                            # both objects present
    w.u(2, 1).u(5, 3).u(6, 7)                    # one data point 3, 8
    w.u(3, 7).u(1, 1).u(1, 0)                    # obj 0: 1 band, coarse
    w.u(3, 6).u(1, 0).u(1, 1)                    # obj 1: 3 bands, fine,
    w.u(1, 1).u(1, 0).u(1, 1)                    # sparse: dry 0, wet 0
    w.u(1, 1)                                    # ajoc_b_nodt
    # object 0: dry ch0, ch1, wet d0 (one band each, F0)
    _ajoc_put(w, "DRY", 1, "F0", 30)
    _ajoc_put(w, "DRY", 1, "F0", 25)
    _ajoc_put(w, "WET", 1, "F0", 12)
    # object 1 (fine, 3 bands): dry ch0 F0 60, DF +1, DF +45 (wraps)
    _ajoc_put(w, "DRY", 0, "F0", 60)
    _ajoc_put(w, "DRY", 0, "DF", 1)
    _ajoc_put(w, "DRY", 0, "DF", 45)
    _ajoc_put(w, "WET", 0, "F0", 21)
    _ajoc_put(w, "WET", 0, "DF", 0)
    _ajoc_put(w, "WET", 0, "DF", 40)
    b = Bits(w.align().bytes())
    s = AJ.ajoc(b, n_dmx, n_umx)
    assert s["dpoints"] == [(3, 8)] and s["num_bands"] == [1, 3]
    assert s["dry_present"][1] == [1, 0]
    prev = {}
    q = AJ.quantised(s, prev)
    assert q[0][0][0] == [[30], [25]] and q[0][0][1] == [[12]]
    assert q[1][0][0] == [[60, 61, 5], [50, 50, 50]]
    assert q[1][0][1] == [[21, 21, 20]]
    assert np.allclose(AJ.dequant_dry([0, 25, 50], 1),
                       [-5.0048828, 0, 5.0048828], atol=1e-7)
    assert np.allclose(AJ.dequant_dry([1, 99], 0), [-4.90478516, 4.904785156],
                       atol=1e-8)
    assert np.allclose(AJ.dequant_wet([0, 11], 1), [-2.001953125,
                                                     0.200195313], atol=1e-8)
    assert np.allclose(AJ.dequant_wet([39], 0), [1.901855469], atol=1e-8)
    # a time-differential frame against the held values
    w = BW().u(1, 0)                             # ajoc_b_nodt 0
    w.u(1, 1)
    _ajoc_put(w, "DRY", 0, "DT", -2)
    for _ in range(2):
        _ajoc_put(w, "DRY", 0, "DT", 1)
    w.u(1, 1)
    for v in (1, 0, -1):
        _ajoc_put(w, "WET", 0, "DT", v)
    side = dict(s)
    side.update(present=[0, 1])
    b = Bits(w.align().bytes())
    side["b_nodt"] = b.u(1)
    side["data"] = [[], [([AJ.huff_data(b, "DRY", 3, 0, False), None],
                          [AJ.huff_data(b, "WET", 3, 0, False)])]]
    q = AJ.quantised(side, prev)
    assert q[1][0][0][0] == [58, 62, 6]
    assert q[1][0][1][0] == [22, 21, 19]


def test_ajoc_band_map():
    from ac4 import ajoc as AJ
    m = AJ.sb_to_pb(23)
    assert m[0] == 0 and m[11] == 11 and m[13] == 12 and m[63] == 22
    assert list(AJ.sb_to_pb(7)[[3, 4, 13, 14, 25]]) == [2, 3, 4, 5, 6]
    assert set(AJ.sb_to_pb(1)) == {0}


def test_object_panner():
    """objrender: power preserved, a source at a loudspeaker feeds it
    alone, left / right mirror, the top ring used from z > 0."""
    from ac4 import objrender as OR
    from ac4 import render as RD
    lay = RD.LAYOUTS["7.1.4"]
    i = {k: n for n, k in enumerate(lay)}
    for pos in ((0.1, 0.2, 0.0), (0.7, 0.9, 0.4), (0.5, 0.5, 1.0),
                (0.5, 0.0, 0.0), (0.3, 0.6, -0.5)):
        g = OR.gains(pos, lay)
        assert np.isclose((g ** 2).sum(), 1.0)
    g = OR.gains((0.0, 0.0, 0.0), lay)              # front left corner
    assert np.isclose(g[i["L"]], 1.0) and np.isclose((g ** 2).sum(), 1.0)
    gl = OR.gains((0.2, 0.3, 0.5), lay)
    gr = OR.gains((0.8, 0.3, 0.5), lay)
    for a, b_ in (("L", "R"), ("Ls", "Rs"), ("Tfl", "Tfr"), ("Tbl", "Tbr")):
        assert np.isclose(gl[i[a]], gr[i[b_]])
    assert gl[i["Tfl"]] > 0
    g2 = OR.gains((0.2, 0.3, 0.5), RD.LAYOUTS["5.1"])
    assert np.isclose((g2 ** 2).sum(), 1.0)          # no top ring
    assert OR.gains((0.5, 0.5, 0), lay, lfe=True)[i["lfe"]] == 1.0


def test_ajoc_dmx_de_data():
    """ajoc_dmx_de_data (6.2.3.5, Table 82)."""
    from ac4 import ajoc as AJ
    from ac4.toc import Bits
    w = BW().u(1, 1).u(1, 0).u(2, 2)             # cfg, new coeffs, max 2
    w.u(1, 0).u(1, 1).u(1, 0)                    # objects: 1 is dialogue
    w.u(1, 0)                                    # dmx 0: 0
    w.u(1, 1).u(3, 7)                            # dmx 1: 1
    w.u(1, 1).u(3, 2).u(1, 1)                    # dmx 2: 0b10101 -> 6/15
    st = {}
    d = AJ.dmx_de_data(Bits(w.align().bytes()), 3, 3, st)
    assert st == dict(de_max_gain=2, dlg=[1])
    assert np.allclose(d["coeffs"], [[0.0, 1.0, 6 / 15]])


# ------------------------------------------------ A-JOC core decoding
def _ajoc_toc(sub_len, seq, iframe, n_dmx, n_umx, lfe, static=False):
    """A TOC with one presentation whose only audio substream is A-JOC
    coded (ac4_substream_info_ajoc, 6.2.1.9): dynamic objects only."""
    from test_phase_a import presentation
    w = BW().u(2, 2).u(10, seq).u(1, 0).u(1, 1).u(4, 3).u(1, iframe)
    w.u(1, 1).u(1, 0).u(1, 0)            # one presentation, no base, no id
    presentation(w, groups=[0], pres_sub=0)
    # ac4_substream_group_info: present, no hsf, single, object coded
    w.u(1, 1).u(1, 0).u(1, 1).u(1, 0)
    w.u(1, 0)                            # b_oamd_substream
    w.u(1, 1)                            # b_ajoc
    if static:
        w.u(1, lfe).u(1, 1)              # b_lfe, b_static_dmx
    else:
        w.u(1, lfe).u(1, 0).u(4, n_dmx - 1)  # b_lfe, not static, n_dmx
        w.u(1, 1)                        # dmx: b_dyn_objects_only
    w.u(1, 0)                            # b_oamd_common_data_present
    w.u(4, n_umx - 1).u(1, 1)            # n_umx, dynamic only
    w.u(1, 0).u(1, 0)                    # b_sf_multiplier, b_bitrate_info
    w.u(1, iframe)                       # b_audio_ndot
    w.u(2, 1)                            # substream_index 1
    w.u(1, 0)                            # b_content_type
    w.u(2, 2).u(1, 0).u(10, 2).u(1, 0).u(10, sub_len)
    return w.bytes()


def _ajoc_sub(rng, n_dmx, n_umx, lfe, pos, iframe=True, aspx=False,
              static=False):
    """audio_data_ajoc (6.2.3.4) with a SIMPLE var_channel_element of
    n_dmx signals (+ LFE), downmix OAMD with the positions pos, an A-JOC
    side info with no data points, no DE data, the upmix OAMD inactive.
    -> (bytes, the dequantised tracks in bitstream order)."""
    from ac4 import aspx_bands as A
    T = CH.load_tables()
    books, abooks = _books()
    t_ = A.tables_for(ASPX_CFG, 0)
    if static:
        sub5, exp = _static_5x(rng, lfe)
        w = BW()
        tracks = []
        return _ajoc_rest(w, sub5, n_dmx, n_umx, lfe, pos, iframe, True), \
            exp
    w = BW().u(1, 0)                     # b_some_signals_inactive
    w.u(1, int(aspx))                    # var_codec_mode
    if aspx:
        if iframe:
            _aspx_cfg_bits(w, ASPX_CFG)
        if n_dmx <= 5:
            w.u(1, 1).u(1, 1) if n_dmx > 1 else w.u(1, 1)   # companding
    tracks = []
    if lfe:
        t = EW.Track(rng, 2)
        EW.lfe(w, T, t)
        tracks.append(t.deq())
    n = 10
    for p in range(n_dmx // 2 - (1 if n_dmx % 2 else 0)):
        tr = [EW.Track(rng, n), EW.Track(rng, n)]
        EW.two_channel(w, T, tr, EW.St(rng, 0, n))
        tracks += [x.deq() for x in tr]
    if n_dmx % 2:
        if n_dmx > 1:
            w.u(1, 0)                    # var_coding_config: pair + mono
            tr = [EW.Track(rng, n), EW.Track(rng, n)]
            EW.two_channel(w, T, tr, EW.St(rng, 0, n))
            tracks += [x.deq() for x in tr]
        t = EW.Track(rng, n)
        EW.mono(w, T, t)
        tracks.append(t.deq())
    if aspx:
        for _ in range(n_dmx // 2):
            _aspx_2ch(w, abooks, t_, iframe)
        if n_dmx % 2:
            _aspx_1ch(w, abooks, t_, iframe)
    return _ajoc_rest(w, None, n_dmx, n_umx, lfe, pos, iframe, False), \
        tracks


def _static_5x(rng, lfe):
    """A SIMPLE 5_X element body (5.1 or 5.0) for b_static_dmx."""
    T = CH.load_tables()
    w = BW().u(3, 0)                     # 5_X_codec_mode SIMPLE
    exp = {}
    if lfe:
        t = EW.Track(rng, 2)
        EW.lfe(w, T, t)
        exp["lfe"] = t.deq()
    w.u(2, 0).u(1, 0)                    # coding_config 0, 2ch_mode 0
    for keys in (("L", "R"), ("Ls", "Rs")):
        tr = [EW.Track(rng, 8), EW.Track(rng, 8)]
        EW.two_channel(w, T, tr, EW.St(rng, 0, 8))
        exp.update(zip(keys, (x.deq() for x in tr)))
    c = EW.Track(rng, 8)
    EW.mono(w, T, c)
    exp["C"] = c.deq()
    return w, exp


def _ajoc_rest(w, w5, n_dmx, n_umx, lfe, pos, iframe, static):
    if static:
        w = w5
    else:
        _dmx_oamd(w, n_dmx, lfe, pos, iframe)
    return _ajoc_side_bits(w, n_umx, lfe)


def _dmx_oamd(w, n_dmx, lfe, pos, iframe):
    w.u(1, 1)                            # b_dmx_timing
    w.u(1, 0).u(3, 1).u(6, 0).u(2, 0)    # offset 0, one block, ramp 0
    for k in range(n_dmx + (1 if lfe else 0)):
        w.u(1, 0)                        # active
        if not iframe:
            w.u(1, 0)                    # basic info: new
        w.u(1, 1)                        # b_default_basic_info_md (0 dB)
        if not (lfe and k == 0):
            if not iframe:
                w.u(1, 0).u(1, 0)        # render: new
            x, y, z = pos[k - (1 if lfe else 0)]
            if not iframe:
                w.u(1, 0)                # b_diff_pos_coding
            w.u(6, x).u(6, y).u(1, 1).u(4, z)
            w.u(1, 1).u(1, 1)            # zone, other: defaults
        w.u(1, 0)                        # b_add_table_data
    w.u(1, 0)                            # b_oamd_extension_present


def _ajoc_side_bits(w, n_umx, lfe):
    w.u(3, 0)                            # ajoc_num_decorr 0
    for _ in range(n_umx):
        w.u(1, 0)                        # ajoc_object_present
    w.u(2, 0)                            # no data points
    w.u(1, 1)                            # ajoc_b_nodt
    w.u(1, 0).u(1, 1)                    # dmx_de: no cfg, keep
    w.u(1, 0).u(1, 1)                    # b_umx_timing 0, derive
    for k in range(n_umx + (1 if lfe else 0)):
        w.u(1, 1).u(1, 0)                # inactive, no table data
    return EW.element(w)


def _ajoc_frames(rng, n_dmx=3, n_umx=4, lfe=1, pos=None, n=4, aspx=False,
                 static=False):
    from test_inputs import syncframe  # noqa: F401
    pos = pos or [(0, 0, 0), (62, 0, 0), (31, 62, 15)][:n_dmx] + \
        [(31, 31, 0)] * max(0, n_dmx - 3)
    frames, tracks = [], []
    for i in range(n):
        sub, tr = _ajoc_sub(rng, n_dmx, n_umx, lfe, pos, iframe=(i == 0),
                            aspx=aspx, static=static)
        toc = _ajoc_toc(len(sub), 5 + i, int(i == 0), n_dmx, n_umx, lfe,
                        static)
        frames.append(toc + bytes(2) + sub)
        tracks.append(tr)
    return frames, tracks


@pytest.mark.parametrize("n_dmx", [1, 2, 3, 4, 5, 7])
def test_ajoc_core_parse(n_dmx):
    """An A-JOC substream (6.2.3.4) walked to audio_size: the downmix
    tracks of the var_channel_element (6.2.4.4) on their slots, the
    downmix objects' positions, the side information."""
    from ac4 import decoder as D
    from ac4 import toc as M
    rng = np.random.default_rng(70 + n_dmx)
    frames, tracks = _ajoc_frames(rng, n_dmx=n_dmx)
    st = M.parse(frames[0])
    info = M.audio_substream(st, 0)
    assert info["kind"] == "ajoc" and info["n_dmx"] == n_dmx
    assert D.probe_element(frames) == "ajoc"
    s = SC.Ac4Stream("ajoc", ajoc_info=info)
    wins, lengths, groups, counts = s.decode_frames(frames)
    assert s.n_bad == 0, s.status
    slots = EL.ajoc_slots(info)
    order = ("lfe",) + EL.VAR_SLOTS[n_dmx]
    for i in range(4):
        for k, ref in zip(order, tracks[i]):
            assert np.allclose(np.asarray(wins[k][i]), ref), (i, k)
        o = groups["obj"][i]
        assert len(o["dmx"]) == n_dmx + 1 and o["side"]["num_decorr"] == 0
    p = groups["obj"][0]["dmx"][1].position()
    assert p == (0.0, 0.0, 0.0)
    assert set(slots) == set(order)


def test_ajoc_core_renders_objects():
    """Core decoding to 5.1: an object at the front left corner plays on
    L only, one at the front right corner on R, one at the back ceiling
    on the surrounds (no top ring in 5.1)."""
    from ac4.decoder import decode
    rng = np.random.default_rng(81)
    frames, _ = _ajoc_frames(rng, n_dmx=3, n=6)
    pcm, dec = decode(frames, max_pass=100)
    assert dec.n_concealed == 0 and dec.chans == ("L", "R", "C", "lfe",
                                                   "Ls", "Rs")
    # each object alone: zero the others' tracks through the renderer
    R = dec.obj_renderer._matrix()
    i = {k: n for n, k in enumerate(dec.dchans)}
    assert np.allclose(R[:, i["L"]], [1, 0, 0, 0, 0, 0])
    assert np.allclose(R[:, i["R"]], [0, 1, 0, 0, 0, 0])
    assert np.allclose(R[:, i["C"]], [0, 0, 0, 0, np.sqrt(.5), np.sqrt(.5)])
    assert R[3, i["lfe"]] == 1.0
    assert np.abs(pcm).max() > 0


@pytest.mark.parametrize("n_dmx", [2, 3, 5, 6])
def test_ajoc_core_aspx(n_dmx):
    """The var_channel_element in codec mode ASPX (aspx_config,
    companding_control, the pairs' and the odd signal's A-SPX blocks):
    every frame closes, every downmix signal sounds."""
    from ac4.decoder import Decoder, probe_ajoc_info
    rng = np.random.default_rng(100 + n_dmx)
    frames, _ = _ajoc_frames(rng, n_dmx=n_dmx, n=6, aspx=True)
    info = probe_ajoc_info(frames)
    dec = Decoder("ajoc", guard=False, ajoc_info=info, layout="7.1.4")
    dec.hf_ratio = None
    y = dec.decode_pass(frames)
    assert dec.n_concealed == 0, dict(dec.concealed.reasons)
    assert np.abs(y[3 * 1536:]).max() > 0


def test_ajoc_static_dmx():
    """b_static_dmx: the downmix is a 5.1 audio_data_chan; rendered as a
    5.1 bed."""
    from ac4.decoder import Decoder, probe_ajoc_info
    rng = np.random.default_rng(111)
    frames, exps = _ajoc_frames(rng, n_dmx=5, n=4, static=True)
    info = probe_ajoc_info(frames)
    assert info["static"] == 1
    s = SC.Ac4Stream("ajoc", ajoc_info=info)
    wins, lengths, groups, counts = s.decode_frames(frames)
    assert s.n_bad == 0, s.status
    for i, exp in enumerate(exps):
        for k, ref in exp.items():
            assert np.allclose(np.asarray(wins[k][i]), ref), (i, k)
    dec = Decoder("ajoc", guard=False, ajoc_info=info)
    dec.hf_ratio = None
    y = dec.decode_pass(frames)
    d0 = Decoder("5_X", guard=False)
    d0.hf_ratio = None
    assert dec.chans == d0.chans


def _c_vs_python_ajoc(tmp_path, frames, layout):
    import os
    import subprocess
    from ac4.decoder import Decoder, probe_ajoc_info
    from test_inputs import syncframe
    cli = os.environ.get("AC4_C_CLI", C_CLI)
    if not os.path.exists(cli):
        pytest.skip("c/ac4dec-c not built")
    src = tmp_path / "e.ac4"
    src.write_bytes(b"".join(syncframe(f) for f in frames))
    raw = tmp_path / "e.f64"
    args = [cli, str(src), str(tmp_path / "e.wav"), "--legacy-level",
            "--f64", str(raw), "--internal", "--no-guard", "-q"]
    if layout:
        args += ["--layout", layout]
    subprocess.run(args, check=True)
    dec = Decoder("ajoc", guard=False, ajoc_info=probe_ajoc_info(frames),
                  layout=layout)
    dec.hf_ratio = None
    ref = dec.decode_pass(frames)
    assert dec.n_concealed == 0
    got = np.fromfile(raw, "<f8").reshape(-1, len(dec.chans))
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()


@pytest.mark.parametrize("n_dmx,aspx,static,layout", [
    (1, False, False, None), (3, False, False, "2.0"),
    (5, True, False, "7.1.4"), (7, False, False, "5.1.2"),
    (6, True, False, None), (5, False, True, "2.0")])
def test_c_ajoc(tmp_path, n_dmx, aspx, static, layout):
    """The C decoder's A-JOC core decoding and object rendering against
    the Python, within 1e-9."""
    rng = np.random.default_rng(120 + n_dmx)
    frames, _ = _ajoc_frames(rng, n_dmx=n_dmx, n=6, aspx=aspx,
                             static=static,
                             pos=[(9 * k, 61 - 8 * k, 2 * k)
                                  for k in range(n_dmx)])
    _c_vs_python_ajoc(tmp_path, frames, layout)


def test_ice_upmix_with_dialogue_enhancement():
    """DE on: the immersive upmix (no L / R / C in it) still runs (it
    used to raise inside DE and fall back to the core band)."""
    from ac4.decoder import Decoder
    rng = np.random.default_rng(15)
    frames, _ = _ice_frames(rng, n=6)
    ys = []
    for de in (0.0, 9.0):
        d = Decoder("ice", guard=False, ch_mode=12, present=(1, 1, 3),
                    de_level=de)
        d.hf_ratio = None
        ys.append(d.decode_pass(frames))
        assert d.n_hf_fail == 0, d.log
    assert np.array_equal(ys[0][:, 4:], ys[1][:, 4:])
