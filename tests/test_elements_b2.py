# SPDX-License-Identifier: Apache-2.0
"""Phase B2 item 3 (no captures): single_channel_element (mono),
3_0_channel_element, 5_X without LFE (5.0) and 7_X_channel_element
(SIMPLE / ASPX, the three layouts of Table 88), written bit by bit;
TS 103 190-1 Tables 19 - 25, 33, 180 - 183."""
import numpy as np
import pytest

from ac4 import channels as CH
from ac4 import stream as SC
import elem_writer as EW
from test_phase_a import BW
from test_phase_b2 import T178, T179, T4, _expect, _pp, _c_vs_python

PAIR = ["a0 b0", "c0 d0"]
# Table 88 codes
CODE = {"mono": "0", "3_0": "1100", "5_0": "1101", 5: "1111000",
        6: "1111001", 7: "1111010", 8: "1111011", 9: "1111100",
        10: "1111101"}


def _mono(rng):
    T = CH.load_tables()
    w = BW().u(1, 0)                                 # SIMPLE
    t = EW.Track(rng, int(rng.integers(3, 12)))
    EW.mono(w, T, t)
    return EW.element(w), {"C": t.deq()}


def _three0(rng, cc, matsel=0):
    T = CH.load_tables()
    w = BW().u(1, 0).u(1, cc)                        # SIMPLE, coding_config
    n = int(rng.integers(3, 12))
    if cc == 0:
        tr = [EW.Track(rng, n), EW.Track(rng, n)]
        st = EW.St(rng, 1, n)
        EW.two_channel(w, T, tr, st)                 # stereo_data, proc on
        exp = _expect(tr, ("L", "R"), PAIR, _pp([st]))
        c = EW.Track(rng, 5)
        EW.mono(w, T, c)
        exp["C"] = c.deq()
    else:
        tr = [EW.Track(rng, n) for _ in range(3)]
        sts = [EW.St(rng, 2, n), EW.St(rng, 1, n)]
        EW.n_channel(w, T, tr, sts, matsel)
        exp = _expect(tr, ("L", "R", "C"), T178[matsel], _pp(sts))
    return EW.element(w), exp


def _five0(rng):
    """5_X SIMPLE without LFE (5_X_channel_element(0, ...))."""
    T = CH.load_tables()
    w = BW().u(3, 0).u(2, 0).u(1, 0)                 # SIMPLE, cc 0, 2ch 0
    exp = {}
    for keys in (("L", "R"), ("Ls", "Rs")):
        n = int(rng.integers(3, 12))
        tr = [EW.Track(rng, n), EW.Track(rng, n)]
        st = EW.St(rng, 2, n)
        EW.two_channel(w, T, tr, st)
        exp.update(_expect(tr, keys, PAIR, _pp([st])))
    c = EW.Track(rng, 4)
    EW.mono(w, T, c)
    exp["C"] = c.deq()
    return EW.element(w), exp


FIN = {5: (("D", "F"), ("Ls", "Lb"), ("E", "G"), ("Rs", "Rb"),
           {"A": "L", "B": "R"}),
       7: (("A", "F"), ("L", "Lw"), ("B", "G"), ("R", "Rw"),
           {"D": "Ls", "E": "Rs"}),
       9: (("A", "F"), ("L", "Tfl"), ("B", "G"), ("R", "Tfr"),
           {"D": "Ls", "E": "Rs"})}


def _seven(rng, ch_mode, cc, mode2=0, matsel=0, sap_add=True):
    """7_X SIMPLE. -> (bytes, expected spectra per channel)."""
    T = CH.load_tables()
    lay = ch_mode - (ch_mode + 1) % 2                 # 5, 7, 9
    has_lfe = ch_mode % 2 == 0
    w = BW().u(2, 0)                                 # SIMPLE
    pre = {}
    if has_lfe:
        lfe = EW.Track(rng, 2)
        EW.lfe(w, T, lfe)
        pre["lfe"] = lfe.deq()
    w.u(2, cc)
    n = int(rng.integers(3, 12))                     # one framing for all
    if cc == 0:
        w.u(1, mode2)
        keys = [("A", "B"), ("D", "E")] if mode2 == 0 else \
            [("A", "D"), ("B", "E")]
        for kk in keys:
            tr = [EW.Track(rng, n), EW.Track(rng, n)]
            st = EW.St(rng, 1, n)
            EW.two_channel(w, T, tr, st)
            pre.update(_expect(tr, kk, PAIR, _pp([st])))
    elif cc == 1:
        tr = [EW.Track(rng, n) for _ in range(3)]
        sts = [EW.St(rng, 2, n), EW.St(rng, 1, n)]
        EW.n_channel(w, T, tr, sts, matsel)
        pre.update(_expect(tr, ("A", "B", "C"), T178[matsel], _pp(sts)))
        tr = [EW.Track(rng, n), EW.Track(rng, n)]
        st = EW.St(rng, 2, n)
        EW.two_channel(w, T, tr, st)
        pre.update(_expect(tr, ("D", "E"), PAIR, _pp([st])))
    elif cc == 2:
        tr = [EW.Track(rng, n) for _ in range(4)]
        sts = [EW.St(rng, m, n) for m in (2, 1, 0, 2)]
        EW.n_channel(w, T, tr, sts)
        pre.update(_expect(tr, ("A", "B", "D", "E"), T4, _pp(sts)))
    else:
        tr = [EW.Track(rng, n) for _ in range(5)]
        sts = [EW.St(rng, m, n) for m in (2, 1, 0, 2, 1)]
        EW.n_channel(w, T, tr, sts, matsel)
        pre.update(_expect(tr, ("A", "B", "C", "D", "E"), T179[matsel],
                           _pp(sts)))
    b0, o0, b1, o1, plain = FIN[lay]
    w.u(1, int(sap_add))                             # b_use_sap_add_ch
    sap = [EW.St(rng, 1, n), EW.St(rng, 2, n)] if sap_add else None
    if sap_add:
        for s in sap:
            s.write(w)
    tr = [EW.Track(rng, n), EW.Track(rng, n)]
    st = EW.St(rng, 1, n)
    EW.two_channel(w, T, tr, st)
    pre.update(_expect(tr, ("F", "G"), PAIR, _pp([st])))
    if cc in (0, 2):
        c = EW.Track(rng, 6)
        EW.mono(w, T, c)
        pre["C"] = c.deq()
    exp = {"C": pre["C"]}
    if has_lfe:
        exp["lfe"] = pre["lfe"]
    for k, v in plain.items():
        exp[v] = pre[k]
    for i, ((x, y), (u, v)) in enumerate(((b0, o0), (b1, o1))):
        class Tr:
            pass
        ta, tb = Tr(), Tr()
        ta.deq, tb.deq = (lambda a=pre[x]: a), (lambda a=pre[y]: a)
        p = _pp([sap[i]]) if sap_add else \
            (lambda sfb: {k: [d] for k, d in zip("abcd", (1, 0, 0, 1))})
        exp.update(_expect([ta, tb], (u, v), PAIR, p))
    return EW.element(w), exp


def _check(element, frames, exps, ch_mode=None):
    s = SC.Ac4Stream(element, ch_mode=ch_mode)
    wins, lengths, groups, counts = s.decode_frames(frames)
    assert s.n_bad == 0, s.status
    for i, exp in enumerate(exps):
        for k, ref in exp.items():
            assert np.allclose(np.asarray(wins[k][i]), ref, rtol=1e-12,
                               atol=1e-9), (i, k)


def _frames(fn, code, n=3, **kw):
    frames, exps = [], []
    for i in range(n):
        sub, exp = fn(**kw)
        frames.append(EW.frame(sub, seq=5 + i, ch_mode_code=code))
        exps.append(exp)
    return frames, exps


def test_mono():
    rng = np.random.default_rng(1)
    f, e = _frames(lambda: _mono(rng), CODE["mono"])
    _check("mono", f, e)


@pytest.mark.parametrize("cc,matsel", [(0, 0), (1, 3), (1, 10)])
def test_three_0(cc, matsel):
    rng = np.random.default_rng(10 + cc + matsel)
    f, e = _frames(lambda: _three0(rng, cc, matsel), CODE["3_0"])
    _check("3_0", f, e)


def test_five_0_reads_no_lfe():
    rng = np.random.default_rng(5)
    f, e = _frames(lambda: _five0(rng), CODE["5_0"])
    _check("5_0", f, e)


@pytest.mark.parametrize("ch_mode", [5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("cc,mode2,matsel,sap", [
    (0, 0, 0, True), (0, 1, 0, False), (1, 0, 7, True), (2, 0, 0, True),
    (3, 0, 3, False)])
def test_seven_x(ch_mode, cc, mode2, matsel, sap):
    rng = np.random.default_rng(100 * ch_mode + 10 * cc + mode2)
    f, e = _frames(lambda: _seven(rng, ch_mode, cc, mode2, matsel, sap),
                   CODE[ch_mode])
    _check("7_X", f, e, ch_mode)


@pytest.mark.parametrize("element,code,fn,nch", [
    ("mono", "0", "_mono", 1), ("3_0", "1100", "_three0", 3),
    ("5_0", "1101", "_five0", 5)])
def test_decoder_pcm(element, code, fn, nch):
    from ac4.decoder import Decoder, probe_element
    rng = np.random.default_rng(3)
    g = globals()[fn]
    frames = [EW.frame((g(rng, 1) if fn == "_three0" else g(rng))[0],
                       seq=5 + i, ch_mode_code=code) for i in range(5)]
    assert probe_element(frames) == element
    dec = Decoder(element, guard=False)
    y = dec.decode_pass(frames)
    assert dec.n_concealed == 0 and y.shape == (5 * 1536, nch)
    assert all(np.abs(y[:, c]).max() > 0 for c in range(nch))


@pytest.mark.parametrize("ch_mode", [6, 7])
def test_decoder_pcm_7x(ch_mode):
    from ac4.decoder import Decoder, probe_element, probe_ch_mode
    rng = np.random.default_rng(4)
    frames = [EW.frame(_seven(rng, ch_mode, 1, matsel=4)[0], seq=5 + i,
                       ch_mode_code=CODE[ch_mode]) for i in range(5)]
    assert probe_element(frames) == "7_X"
    assert probe_ch_mode(frames) == ch_mode
    dec = Decoder("7_X", guard=False, ch_mode=ch_mode)
    y = dec.decode_pass(frames)
    n = 8 if ch_mode == 6 else 7
    assert dec.n_concealed == 0 and y.shape == (5 * 1536, n)
    assert all(np.abs(y[:, c]).max() > 0 for c in range(n))


# ------------------------------------------------ the C decoder
@pytest.mark.parametrize("element,code,fn,nch", [
    ("mono", "0", "_mono", 1), ("3_0", "1100", "_three0", 3),
    ("5_0", "1101", "_five0", 5)])
def test_c_elements(tmp_path, element, code, fn, nch):
    rng = np.random.default_rng(30)
    g = globals()[fn]
    frames = [EW.frame((g(rng, 1, 5) if fn == "_three0" else g(rng))[0],
                       seq=5 + i, ch_mode_code=code) for i in range(5)]
    _c_vs_python(tmp_path, frames, nch, element)
    if nch > 2:
        _c_vs_python(tmp_path, frames, 2, element)


@pytest.mark.parametrize("ch_mode", [5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("cc,sap", [(0, True), (1, False), (3, True)])
def test_c_seven_x(tmp_path, ch_mode, cc, sap):
    rng = np.random.default_rng(40 + ch_mode + cc)
    frames = [EW.frame(_seven(rng, ch_mode, cc, 1 if cc == 0 else 0, 3,
                              sap)[0],
                       seq=5 + i, ch_mode_code=CODE[ch_mode])
              for i in range(5)]
    n = 8 if ch_mode % 2 == 0 else 7
    _c_vs_python(tmp_path, frames, n, "7_X", ch_mode)
    _c_vs_python(tmp_path, frames, 2, "7_X", ch_mode)
