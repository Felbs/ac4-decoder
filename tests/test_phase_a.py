# SPDX-License-Identifier: Apache-2.0
"""Phase A: stereo_data, per-substream I-frames, the structured TOC and
presentation choice. No captures needed: the bitstreams are written here
from the syntax of TS 103 190-1/-2 with a small bit writer."""
import numpy as np
import pytest

from ac4 import channels as CH
from ac4 import toc as M
from ac4.toc_v1 import Bits


class BW:
    """MSB-first bit writer."""

    def __init__(self):
        self.bits = []

    def u(self, n, v):
        for i in range(n - 1, -1, -1):
            self.bits.append((v >> i) & 1)
        return self

    def vb(self, n, v):
        """variable_bits(n) of v (TS 103 190-1 Table 3)."""
        groups = []
        while True:
            groups.append(v & ((1 << n) - 1))
            if v < (1 << n):
                break
            v = (v >> n) - 1
        for i, g in enumerate(reversed(groups)):
            self.u(n, g)
            self.u(1, 0 if i == len(groups) - 1 else 1)
        return self

    def align(self):
        while len(self.bits) % 8:
            self.bits.append(0)
        return self

    def bytes(self):
        self.align()
        return bytes(int("".join(map(str, self.bits[i:i + 8])), 2)
                     for i in range(0, len(self.bits), 8))


def test_bit_writer_variable_bits_round_trip():
    for n in (2, 3, 7):
        for v in (0, 1, 3, 4, 5, 17, 100, 1000):
            w = BW().vb(n, v)
            assert Bits(w.bytes()).vb(n) == v


# ------------------------------------------------------------ stereo_data
def _long_empty_channel(w):
    """sf_info for a long frame with max_sfb 0 (asf_transform_info +
    asf_psy_info at frame_len_base 1536: b_long_frame, 6-bit max_sfb)."""
    w.u(1, 1).u(6, 0)


def _pair_substream(stereo_proc):
    """A channel_pair_element in stereo_codec_mode SIMPLE whose two
    channels are long frames with no bands, then 13 marker bits."""
    w = BW().u(15, 0).u(1, 0)                # audio_size_value, b_more_bits
    w.u(2, 0)                                # stereo_codec_mode SIMPLE
    if stereo_proc:
        w.u(1, 1)
        _long_empty_channel(w)
        w.u(2, 0)                            # chparam_info: sap_mode 0
    else:
        w.u(1, 0)
        w.u(1, 0)                            # spec_frontend_l = ASF
        _long_empty_channel(w)
        w.u(1, 0)                            # spec_frontend_r = ASF
        _long_empty_channel(w)
    # sf_data x2: no sections or lines; reference_scale_factor (8 bits),
    # b_snf_data_exists
    for _ in range(2):
        w.u(8, 100).u(1, 0)
    end = len(w.bits)
    w.u(13, 0x1ACE)
    return w.bytes(), end


@pytest.mark.parametrize("stereo_proc", [1, 0])
def test_stereo_data_reads_spec_frontend_bits(stereo_proc):
    """Table 23: without MDCT stereo processing each channel's sf_info is
    preceded by spec_frontend. Reading two_channel_data (Table 26) there
    misread the frame (kit 2.0 streams: 32 frames concealed)."""
    from ac4 import pair as PAIR
    sub, end = _pair_substream(stereo_proc)
    T = CH.load_tables()
    r = PAIR.decode_pair(sub, T)
    assert r["bits"] == end
    assert r["L"]["framing"].long and r["R"]["framing"].long


def test_stereo_data_rejects_ssf():
    from ac4 import pair as PAIR
    w = BW().u(15, 0).u(1, 0).u(2, 0).u(1, 0).u(1, 1).u(16, 0)
    with pytest.raises(ValueError, match="SSF"):
        PAIR.decode_pair(w.bytes(), CH.load_tables())


# ------------------------------------------------------------ the TOC
def emdf_info(w):
    w.u(2, 0).u(3, 0).u(1, 0).u(2, 1).u(8, 0).u(2, 0)


def presentation(w, groups, pres_sub, md_compat=0, pres_id=None,
                 enabled=None):
    """ac4_presentation_v1_info, bitstream_version 2."""
    if len(groups) == 1:
        w.u(1, 1)                            # b_single_substream_group
    else:
        w.u(1, 0).u(3, 5)                    # presentation_config 5
    w.u(1, 1).u(1, 0)                        # presentation_version 1
    w.u(3, md_compat)
    if pres_id is None:
        w.u(1, 0)
    else:
        w.u(1, 1).vb(2, pres_id)
    w.u(1, 0)                                # frame_rate_multiply_info (fri 3)
    emdf_info(w)
    if enabled is None:
        w.u(1, 0)
    else:
        w.u(1, 1).u(1, enabled)
    if len(groups) == 1:
        w.u(3, groups[0])
    else:
        w.u(1, 0)                            # b_multi_pid
        w.u(2, len(groups) - 2)
        for g in groups:
            w.u(3, g)
    w.u(1, 0).u(1, 0)                        # b_pre_virtualized, b_add_emdf
    w.u(1, 0).u(1, 1).u(2, pres_sub)         # presentation_substream_info


def group(w, ch_mode_code, index, ndot=1, cc=None, lang=None):
    """ac4_substream_group_info with one channel-coded substream."""
    w.u(1, 1).u(1, 0).u(1, 1)                # present, no hsf, single
    w.u(1, 1)                                # b_channel_coded
    w.u(len(ch_mode_code), int(ch_mode_code, 2))
    w.u(1, 0)                                # b_sf_multiplier (fs_index 1)
    w.u(1, 0)                                # b_bitrate_info
    w.u(1, ndot)                             # b_audio_ndot
    w.u(2, index)
    if cc is None:
        w.u(1, 0)
    else:
        w.u(1, 1).u(3, cc)
        if lang is None:
            w.u(1, 0)
        else:
            w.u(1, 1).u(1, 0).u(6, len(lang))
            for c in lang.encode():
                w.u(8, c)


def build_toc(pres, groups, sizes, iframe=1, seq=5):
    w = BW().u(2, 2).u(10, seq).u(1, 0).u(1, 1).u(4, 3).u(1, iframe)
    if len(pres) == 1:
        w.u(1, 1)
    else:
        w.u(1, 0).u(1, 1).vb(2, len(pres) - 2)
    w.u(1, 0)                                # b_payload_base
    w.u(1, 0)                                # b_program_id
    for p in pres:
        presentation(w, **p)
    for g in groups:
        group(w, **g)
    w.u(2, len(sizes) if len(sizes) < 4 else 0)
    if len(sizes) >= 4:
        w.vb(2, len(sizes) - 4)
    for s in sizes:
        w.u(1, 0).u(10, s)
    return w.bytes()


# two presentations (English 5.1 and a French stereo commentary) sharing
# the presentation substream (index 0): substream groups 0 and 1, audio
# substreams 1 and 2
TWO_PRES = dict(
    pres=[dict(groups=[0], pres_sub=0, pres_id=10),
          dict(groups=[1], pres_sub=0, pres_id=11)],
    groups=[dict(ch_mode_code="1110", index=1, cc=0, lang="en"),
            dict(ch_mode_code="10", index=2, ndot=0, cc=5, lang="fr")],
    sizes=[3, 5, 7])


def test_toc_keeps_presentations_groups_and_indices():
    f = build_toc(**TWO_PRES)
    st = M.parse(f)
    assert st["n_presentations"] == 2
    assert [p["groups"] for p in st["presentations"]] == [[0], [1]]
    assert [p["presentation_id"] for p in st["presentations"]] == [10, 11]
    # total_n_substream_groups = 1 + max group index (TS 103 190-2
    # 6.3.2.1.8), not the sum over presentations
    assert st["total_n_substream_groups"] == 2
    g0, g1 = st["groups"]
    assert g0["substreams"][0]["channel_mode"] == 4
    assert g0["substreams"][0]["index"] == 1
    assert g1["substreams"][0]["channel_mode"] == 1
    assert g1["substreams"][0]["index"] == 2
    assert g1["substreams"][0]["ndot"] == [0]
    assert (g0["language"], g1["language"]) == ("en", "fr")
    assert M.CONTENT_CLASS[g1["content_classifier"]] == "commentary"
    assert st["substream_sizes"] == [3, 5, 7]
    assert M.substream_offset(st, 2) == st["toc_bytes"] + 8


def test_shared_group_is_read_once():
    """Two presentations referencing the same group: one group info
    follows (the old parser read one per presentation and misread the
    substream table)."""
    t = dict(pres=[dict(groups=[0], pres_sub=0),
                   dict(groups=[0], pres_sub=0)],
             groups=[dict(ch_mode_code="10", index=1)], sizes=[2, 4])
    st = M.parse(build_toc(**t))
    assert st["total_n_substream_groups"] == 1
    assert st["substream_sizes"] == [2, 4]


def test_choose_presentation():
    st = M.parse(build_toc(**TWO_PRES))
    assert M.choose(st) == 0                       # complete main, first
    assert M.choose(st, language="fr") == 1
    assert M.choose(st, language="de") == 0        # no match: default
    assert M.choose(st, presentation=1) == 1
    assert M.choose(st, presentation="id:11") == 1
    assert M.choose(st, presentation="id:12") is None
    assert M.choose(st, presentation=2) is None
    assert M.audio_substream(st, 1)["index"] == 2
    d = M.describe(st)
    assert d[1]["content"] == ["commentary"] and d[1]["language"] == "fr"
    assert d[0]["channel_modes"] == [4]


def test_disabled_presentation_is_not_chosen():
    t = dict(TWO_PRES)
    t["pres"] = [dict(groups=[0], pres_sub=0, enabled=0),
                 dict(groups=[1], pres_sub=0)]
    st = M.parse(build_toc(**t))
    assert not st["presentations"][0]["enabled"]
    assert M.choose(st) == 1


def test_substream_iframe_is_its_b_audio_ndot(monkeypatch):
    """The audio substream is an I-frame when ITS b_audio_ndot is set,
    whatever b_iframe_global says (TS 103 190-2 6.3.2.7.6); the kit
    streams send such frames halfway between global I-frames, with an
    aspx_config the decoder must read."""
    from ac4 import pair as PAIR
    from ac4 import robust as RB
    from ac4 import stream as ST
    seen = []

    def fake_pair(sub, T, b_iframe=False):
        seen.append((len(sub), b_iframe))
        raise ValueError("stop")

    monkeypatch.setattr(PAIR, "decode_pair", fake_pair)
    monkeypatch.setattr(RB, "CHECKS", False)
    t = dict(pres=[dict(groups=[0], pres_sub=0)],
             groups=[dict(ch_mode_code="10", index=1, ndot=1)],
             sizes=[2, 4])
    f = build_toc(iframe=0, **t)
    f += bytes(6)
    s = ST.Ac4Stream("pair")
    s.decode_frames([f])
    assert seen == [(4, True)]
    t["groups"][0]["ndot"] = 0
    seen.clear()
    s.decode_frames([build_toc(iframe=1, **t) + bytes(6)])
    assert seen == [(4, False)]


def test_stream_decodes_the_chosen_presentations_substream(monkeypatch):
    from ac4 import pair as PAIR
    from ac4 import robust as RB
    from ac4 import stream as ST
    seen = []

    def fake_pair(sub, T, b_iframe=False):
        seen.append(len(sub))
        raise ValueError("stop")

    monkeypatch.setattr(PAIR, "decode_pair", fake_pair)
    monkeypatch.setattr(RB, "CHECKS", False)
    f = build_toc(**TWO_PRES) + bytes(15)
    for kw, want in ((dict(), 5), (dict(language="fr"), 7),
                     (dict(presentation="id:11"), 7)):
        seen.clear()
        ST.Ac4Stream("pair", **kw).decode_frames([f])
        assert seen == [want]


# ------------------------------------------------------------ A-SPX envelopes
def test_envelope_dequantisation_uses_the_frames_step():
    """Pseudocodes 82 and 84: a = 2 for 1.5 dB steps (aspx_qmode_env 0, which
    aspx_data forces for a FIXFIX interval with one envelope), a = 1 for
    3 dB; the energies are num_qmf_subbands * 2^(q / a), in this decoder's
    units (x IMDCT_K^2)."""
    from ac4 import aspx_envadj as E
    from ac4 import filterbank as FB
    t = dict(lo=[10, 14, 18], hi=[10, 12, 14, 16, 18], sbx=10)
    fr = dict(freq_res=[0])
    k2 = FB.IMDCT_K ** 2
    for qmode, a in ((0, 2), (1, 1)):
        out = E.map_scf([[20, 30]], t, fr, 8, a=E.env_a(qmode))
        assert np.allclose(out[:4, 0], 64 * k2 * 2.0 ** (20 / a))
        assert np.allclose(out[4:8, 0], 64 * k2 * 2.0 ** (30 / a))
        A, B = E.map_scf_joint([[20, 30]], [[12, 16]], t, fr, 8,
                               a=E.env_a(qmode))
        nom = 2.0 ** (20 / a + 1) * 64 * k2
        assert np.isclose(A[0, 0], nom / (1 + 2.0 ** (12 - 12 / a)))
        assert np.isclose(B[0, 0], nom / (1 + 2.0 ** (12 / a - 12)))


def test_fixfix_single_envelope_reports_qmode_0(monkeypatch):
    """aspx_data hands the renderer the step it decoded with."""
    from ac4 import aspx_parse as P
    calls = {}

    def fake_framing(b, cfg, nats, ifr, state, key):
        return dict(int_class=P.FIXFIX, num_env=1, num_noise=1,
                    freq_res=[0], borders=[0, 12], tsg_ptr=-1,
                    sine_at_end=-1)

    monkeypatch.setattr(P, "aspx_framing", fake_framing)
    monkeypatch.setattr(P, "delta_dir", lambda b, f: ([0], [0]))
    monkeypatch.setattr(P, "hfgen_iwc", lambda *a, **k: [[0], [0]])
    monkeypatch.setattr(P, "ec_data",
                        lambda b, tabs, t, kind, n, fres, q, sm, d:
                        calls.setdefault(kind, []).append(q) or [[0]])
    out = P.aspx_data(Bits(bytes(4)), {}, {}, dict(quant_mode_env=1), 12,
                      False, False, {}, 0)
    assert out[0]["qmode"] == 0 and calls["SIGNAL"] == [0]


def test_aspx_grid_is_one_frame_after_pseudocode_75(monkeypatch):
    """Frame f's A-SPX interval slot ts acts on analysis slot 24 f + 24 -
    ts_offset_hfgen + ts (TS_OFFSET 18; see aspx_render.TS_OFFSET): a
    noise-only high band from frame 1's interval [0, 12) A-SPX slots
    occupies exactly analysis slots 42 .. 65."""
    from ac4 import aspx_render as R37
    from ac4 import aspx_env as V
    from ac4 import aspx_envadj as E
    from ac4 import qmf as Q
    from test_units import _aspx_cfg
    cfg, t = _aspx_cfg()
    n_sb, sbx = t["num_sb_aspx"], t["sbx"]
    got = [{"fr": {"borders": [0, 12]}, "balance": 1},
           {"fr": {"borders": [0, 12]}, "balance": 1}]
    z = (lambda *a, **k: (np.zeros((n_sb, 1)), np.zeros((n_sb, 1))))
    monkeypatch.setattr(V, "reconstruct", lambda d, t, m, p: (None, p))
    monkeypatch.setattr(E, "noise_qscf", lambda d, t, p: (None, p))
    monkeypatch.setattr(E, "map_scf_joint", z)
    monkeypatch.setattr(E, "map_noise_joint", z)
    monkeypatch.setattr(E, "estimate", lambda *a, **k: np.zeros((n_sb, 1)))
    monkeypatch.setattr(E, "adjust", lambda *a: (np.zeros((n_sb, 1)),
                                                 np.ones((n_sb, 1))))
    seen = []
    real = Q.synthesise
    monkeypatch.setattr(Q, "synthesise",
                        lambda Qm, *a: (seen.append(Qm.copy()),
                                        real(Qm, *a))[1])
    x = np.zeros(4 * 1536)
    R37.apply_hf_pair(x, x, [None, got, None, None], cfg)
    hf = np.abs(seen[0][sbx:sbx + n_sb]).sum(axis=0)
    assert R37.TS_OFFSET == 24 - R37.TS_HFGEN == 18
    assert list(np.flatnonzero(hf)) == list(range(42, 66))


def test_centre_goes_through_the_aspx_renderer(monkeypatch):
    """A 5_X element in codec mode 1: the centre's aspx_data_1ch is
    rendered (as the same channel twice) like the pairs'; before, the
    centre bypassed A-SPX and kept its core band only."""
    from ac4 import aspx_render as R37
    from ac4 import decoder as D
    calls = []

    def fake_render(pa, pb, gg, cfg, **kw):
        calls.append((pa is pb or np.array_equal(pa, pb), len(gg)))
        if kw.get("state_log") is not None:
            kw["state_log"].extend([None] * (len(gg) + 1))
        n = len(pa) + 577
        return {"L": np.zeros(n), "R": np.zeros(n)}, 0

    monkeypatch.setattr(R37, "apply_hf_pair", fake_render)
    dec = D.Decoder("5_X")
    dat = [{"fr": {"borders": [0, 12]}, "balance": 0}] * 2
    rng = np.random.default_rng(3)

    def fake_frames(frames, chans):
        n = len(frames)
        wins = {k: [rng.standard_normal(1536) for _ in range(n)]
                for k in chans}
        lens = {k: [1536] * n for k in chans}
        cnts = {k: [1] * n for k in chans}
        dec.stream.status = [None] * n
        dec.stream.mode = ("5_X", 1)
        dec.stream.cfg = {"x": 1}
        grp = {"lr": [dat] * n, "sr": [dat] * n, "c": [dat] * n,
               "acpl": [None] * n}
        return wins, lens, grp, cnts

    monkeypatch.setattr(dec.stream, "decode_frames", fake_frames)
    dec.decode_pass([b""] * 3)
    # L/R, Ls/Rs and the centre: three renderer calls, the last with the
    # centre's core as both channels
    assert len(calls) == 3 and calls[2][0]


@pytest.mark.parametrize("fri", [2, 4])
def test_frames_at_other_rates_are_concealed(fri):
    """Only frame_rate_index 3 (29.97 fps, 1536 samples) is rendered; a
    frame at another rate is concealed with its own reason instead of being
    decoded as 1536 samples (noise)."""
    from ac4 import stream as ST
    t = dict(pres=[dict(groups=[0], pres_sub=0)],
             groups=[dict(ch_mode_code="10", index=1)], sizes=[2, 4])
    f = bytearray(build_toc(**t) + bytes(6))
    # rewrite frame_rate_index (4 bits after fs_index, bit 14 of the TOC;
    # 2, 3 and 4 share the TOC layout)
    bits = list(format(int.from_bytes(f[:3], "big"), "024b"))
    bits[14:18] = format(fri, "04b")
    f[:3] = int("".join(bits), 2).to_bytes(3, "big")
    assert M.parse(bytes(f))["frame_rate_index"] == fri
    s = ST.Ac4Stream("pair")
    s.decode_frames([bytes(f)])
    assert s.status == ["toc.unsupported_rate"]
