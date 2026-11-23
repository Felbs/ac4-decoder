# SPDX-License-Identifier: Apache-2.0
"""Phase B tests (no captures): every frame rate of TS 103 190-1 Table 83,
written bit by bit with the helpers of test_phase_a."""
import numpy as np
import pytest

from ac4 import channels as CH
from ac4 import rates as RT
from ac4 import toc as M
from ac4.toc_v1 import Bits
from test_phase_a import BW, build_toc

# Table 83 (48 kHz): frame_rate_index -> (frame_len_base, ratio)
TABLE_83 = {0: (1920, (1001, 960)), 1: (1920, (25, 24)), 2: (2048, (15, 16)),
            3: (1536, (1001, 960)), 4: (1536, (25, 24)), 5: (960, (1001, 960)),
            6: (960, (25, 24)), 7: (1024, (15, 16)), 8: (768, (1001, 960)),
            9: (768, (25, 24)), 10: (512, (15, 16)), 11: (384, (1001, 960)),
            12: (384, (25, 24)), 13: (2048, (1, 1))}


def test_rate_tables():
    for fri, (fl, ratio) in TABLE_83.items():
        assert RT.frame_len_base(1, fri) == fl
        assert RT.ratio(1, fri) == ratio
    for fri in (14, 15):
        assert RT.frame_len_base(1, fri) is None
    # Table 84: 44,1 kHz only at index 13
    assert RT.frame_len_base(0, 13) == 2048 and RT.ratio(0, 13) == (1, 1)
    for fri in range(13):
        assert RT.frame_len_base(0, fri) is None
    assert RT.out_rate(0) == 44100 and RT.out_rate(1) == 48000


def _frame(sub, seq=5, **rate):
    t = dict(pres=[dict(groups=[0], pres_sub=0)],
             groups=[dict(ch_mode_code="10", index=1)],
             sizes=[2, len(sub)])
    return build_toc(**t, seq=seq, **rate) + bytes(2) + sub


@pytest.mark.parametrize("fri", sorted(TABLE_83))
def test_every_rate_sets_the_frame_length(fri):
    """A frame at any rate of Table 83 is taken at that rate (it used to be
    concealed as toc.unsupported_rate unless frame_rate_index was 3)."""
    from ac4 import stream as ST
    f = _frame(bytes(4), fri=fri)
    assert M.parse(f)["frame_rate_index"] == fri
    s = ST.Ac4Stream("pair")
    s.decode_frames([f])
    assert s.status != ["toc.unsupported_rate"]
    assert s.frame_len == TABLE_83[fri][0]
    assert s.rate == (1, fri)


@pytest.mark.parametrize("rate", [dict(fri=14), dict(fs=0, fri=3),
                                  dict(fri=3, frf=2), dict(fri=2, frf=4),
                                  dict(fri=7, fraction=2),
                                  dict(fri=11, fraction=4)])
def test_rates_not_decoded_are_concealed(rate):
    """Reserved indices, 44,1 kHz other than index 13, frame_rate_factor and
    frame_rate_fraction above 1 (rates.py): concealed with their reason."""
    from ac4 import stream as ST
    f = _frame(bytes(4), **rate)
    s = ST.Ac4Stream("pair")
    s.decode_frames([f])
    assert s.status == ["toc.unsupported_rate"]
    assert s.frame_len is None


def test_sf_multiplier_is_concealed():
    from ac4 import stream as ST
    t = dict(pres=[dict(groups=[0], pres_sub=0)],
             groups=[dict(ch_mode_code="10", index=1, sf_mult=0)],
             sizes=[2, 4])
    f = build_toc(**t) + bytes(6)
    s = ST.Ac4Stream("pair")
    s.decode_frames([f])
    assert s.status == ["toc.unsupported_rate"]


def test_44k1_index_13():
    from ac4 import stream as ST
    s = ST.Ac4Stream("pair")
    s.decode_frames([_frame(bytes(4), fs=0, fri=13)])
    assert s.frame_len == 2048 and s.rate == (0, 13)


def test_the_first_rate_must_be_confirmed_by_the_next_frame():
    """One damaged rate field in the first frame cannot fix the stream's
    frame length: the stream's rate is taken when the next frame agrees."""
    from ac4 import stream as ST
    s = ST.Ac4Stream("pair")
    s.decode_frames([_frame(bytes(4), seq=5, fri=2),
                     _frame(bytes(4), seq=6, fri=3),
                     _frame(bytes(4), seq=7, fri=3)])
    assert s.status[0] == "toc.rate"
    assert s.rate == (1, 3) and s.frame_len == 1536
    # a later change of rate is concealed, not followed
    s.decode_frames([_frame(bytes(4), seq=8, fri=2)])
    assert s.status == ["toc.rate"] and s.frame_len == 1536


# --------------------------------------------------------- the ASF framing
def _transform_bits(w, fl, long=True, tl=(3, 3), tl_short=None):
    if fl >= 1536:
        w.u(1, int(long))
        if not long:
            w.u(2, tl[0]).u(2, tl[1])
    else:
        w.u(2, tl_short)


@pytest.mark.parametrize("fl,tl,nwin,lens", [
    (2048, (0, 0), 16, [128] * 16), (2048, (2, 3), 3, [512, 512, 1024]),
    (1920, (1, 1), 8, [240] * 8), (1536, (3, 0), 9, [768] + [96] * 8)])
def test_short_framing_frame_len_ge_1536(fl, tl, nwin, lens):
    """Table 100 (partial blocks per half), Table 109 (grouping bits),
    Pseudocode 3; every grouping bit 1 -> one group per half."""
    w = BW()
    _transform_bits(w, fl, long=False, tl=tl)
    n0, n1 = (TL_LONG := CH.TRANSF_LEN_LONG[fl])[tl[0]], TL_LONG[tl[1]]
    w.u(CH.N_MSFB_BY_LEN[n0], 3)
    if tl[0] != tl[1]:
        w.u(CH.N_MSFB_BY_LEN[n1], 2)
    for _ in range(CH.N_GRP_BITS[tl]):
        w.u(1, 1)
    fr = CH.Framing(Bits(w.bytes()), fl)
    assert fr.num_windows == nwin
    assert [fr.length_g(fr.w2g[i]) for i in range(nwin)] == lens
    assert fr.num_groups == (2 if tl[0] != tl[1] else 1)


@pytest.mark.parametrize("fl,t,nwin,length,nbits", [
    (1024, 0, 8, 128, 4), (1024, 3, 1, 1024, 6), (960, 1, 4, 240, 5),
    (768, 2, 2, 384, 6), (512, 0, 4, 128, 4), (512, 2, 1, 512, 6),
    (384, 1, 2, 192, 5)])
def test_framing_frame_len_below_1536(fl, t, nwin, length, nbits):
    """Table 103 (one transf_length), Table 110 (n_grp_bits), Table 106
    (n_msfb_bits by transform length), Pseudocode 6 (n_sect_bits)."""
    w = BW()
    _transform_bits(w, fl, tl_short=t)
    w.u(nbits, 2)                                     # max_sfb
    for _ in range(CH.N_GRP_BITS_SHORT[fl][t]):
        w.u(1, 0)                                     # every window apart
    fr = CH.Framing(Bits(w.bytes()), fl)
    assert fr.num_windows == nwin and fr.num_groups == nwin
    assert fr.length_g(0) == length and fr.max_sfb == [2]
    assert fr.n_sect_bits(0) == (3 if t <= 2 else 5)
    assert fr.long == (length == fl)


def test_reserved_transform_length_is_damage():
    from ac4.robust import Damaged
    w = BW().u(2, 3)                                  # x at frame 512
    with pytest.raises(Damaged):
        CH.Framing(Bits(w.bytes()), 512)


def test_lfe_max_sfb_width():
    """Table 106: n_msfbl_bits is 3 for 1536-2048, 2 below."""
    for fl, n in ((2048, 3), (1536, 3), (1024, 2), (384, 2)):
        fr = CH.lfe_framing(Bits(bytes([0b10100000])), fl)
        assert fr.max_sfb == [0b101 >> (3 - n)]


# ------------------------------------------- a whole stream at every rate
def _huff_code(book, sym):
    for (n, code), s in book.map.items():
        if s == sym:
            return n, code
    raise KeyError(sym)


def _simple_pair(fl, q0, sf):
    """channel_pair_element, stereo_codec_mode SIMPLE: L carries one line
    of value q0 (codebook 11) in scale factor band 0 at scale factor sf; R
    is empty. Long blocks at frame length fl."""
    T = CH.load_tables()
    w = BW()
    w.u(2, 0)                                          # SIMPLE
    w.u(1, 0)                                          # no MDCT stereo proc
    for ch in range(2):
        w.u(1, 0)                                      # spec_frontend ASF
        if fl >= 1536:
            w.u(1, 1)                                  # b_long_frame
        else:
            w.u(2, 3 if fl in (1024, 960, 768) else 2)
        w.u(6, 1 if ch == 0 else 0)                    # max_sfb
    # L: one section of codebook 11 over band 0
    # sect_len 1 (n_sect_bits 3 when transf_length <= 2, Pseudocode 6:
    # the long block of frame length 512 and 384)
    w.u(4, 11).u(3 if fl in (512, 384) else 5, 0)
    band = CH.SFB_TABLE[fl][1]
    pairs = [(q0, 0)] + [(0, 0)] * (band // 2 - 1)
    for a, b in pairs:
        n, code = _huff_code(T["cb"][11], a * 17 + b)
        w.u(n, code)
        for v in (a, b):
            if v:
                w.u(1, 0)                              # sign: positive
    w.u(8, sf)                                         # reference sf
    w.u(1, 0)                                          # no noise fill
    # R: no bands
    w.u(8, 0).u(1, 0)
    w.align()
    body = w.bytes()
    head = BW().u(15, len(body)).u(1, 0).bytes()
    return head + body


@pytest.mark.parametrize("fri", [0, 2, 3, 5, 7, 8, 10, 11, 13])
def test_stream_decodes_at_every_frame_length(fri):
    """Three frames of a SIMPLE stereo stream at the rate's frame length
    decode to exactly the filterbank's output of the dequantised line (one
    long block per frame), delayed 577 samples like every channel, with
    nothing concealed; at 3 x frame_len_base samples."""
    from ac4 import filterbank as FB
    from ac4.decoder import Decoder, HF_DELAY
    fl = TABLE_83[fri][0]
    sub = _simple_pair(fl, 5, 120)
    frames = [_frame(sub, seq=5 + i, fri=fri) for i in range(3)]
    dec = Decoder("pair")
    y = dec.decode_pass(frames)
    assert dec.n_concealed == 0 and dec.stream.frame_len == fl
    assert y.shape == (3 * fl, 2)
    X = np.zeros(fl)
    X[0] = 5 ** (4 / 3) * 2.0 ** (0.25 * (120 - 100))
    ref = FB.synthesise([X] * 3, [fl] * 3, fl)
    ref = np.concatenate([np.zeros(HF_DELAY), ref])[:3 * fl]
    assert np.abs(y[:, 0] - ref).max() <= 1e-9 * np.abs(ref).max()
    assert np.abs(y[:, 1]).max() == 0.0
    assert dec.ratio == TABLE_83[fri][1]


# ------------------------------------------------------ A-SPX at all rates
@pytest.mark.parametrize("fl,nts,nats,tso", [
    (2048, 32, 16, 26), (1920, 30, 15, 24), (1536, 24, 12, 18),
    (1024, 16, 16, 13), (960, 15, 15, 12), (768, 12, 12, 9),
    (512, 8, 8, 5), (384, 6, 6, 3)])
def test_aspx_geometry(fl, nts, nats, tso):
    """Table 192 / Pseudocode 75a; TS_OFFSET = num_qmf_timeslots -
    ts_offset_hfgen (the phase A reading at 1536, see README)."""
    from ac4 import aspx_bands as A
    g = A.geometry(fl)
    assert (g["nts"], g["nats"], g["ts_offset"]) == (nts, nats, tso)


@pytest.mark.parametrize("fl", [2048, 1024, 384])
def test_aspx_grid_at_other_frame_lengths(monkeypatch, fl):
    """A noise-only high band from frame 1's whole FIXFIX interval
    occupies exactly analysis slots nts + TS_OFFSET .. 2 nts + TS_OFFSET - 1
    at every frame length."""
    from ac4 import aspx_bands as A
    from ac4 import aspx_render as R37
    from ac4 import aspx_env as V
    from ac4 import aspx_envadj as E
    from ac4 import qmf as Q
    from test_units import _aspx_cfg
    g = A.geometry(fl)
    cfg, t = _aspx_cfg()
    n_sb, sbx = t["num_sb_aspx"], t["sbx"]
    got = [{"fr": {"borders": [0, g["nats"]]}, "balance": 1},
           {"fr": {"borders": [0, g["nats"]]}, "balance": 1}]
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
    x = np.zeros(5 * fl)
    R37.apply_hf_pair(x, x, [None, got, None, None, None], cfg, frame_len=fl)
    hf = np.abs(seen[0][sbx:sbx + n_sb]).sum(axis=0)
    a = g["nts"] + g["ts_offset"]
    assert list(np.flatnonzero(hf)) == list(range(a, a + g["nts"]))


def test_tab_border_is_table_194():
    """FIXFIX borders: Table 194, not an even split rounded (15 slots in 4
    envelopes are 0, 4, 8, 12, 15; rounding to nearest gives 11)."""
    from ac4 import aspx_parse as P
    assert P.tab_border(15, 4) == [0, 4, 8, 12, 15]
    assert P.tab_border(6, 4) == [0, 2, 3, 4, 6]
    assert P.tab_border(16, 2) == [0, 8, 16]
    from ac4.robust import Damaged
    with pytest.raises(Damaged):
        P.tab_border(12, 8)          # Table 127: FIXFIX carries at most 4


def _cfg(**k):
    d = dict(quant_mode_env=1, start_freq=3, stop_freq=3,
             master_freq_scale=1, interpolation=1, preflat=1, limiter=1,
             noise_sbg=3, num_env_bits_fixfix=0, freq_res_mode=2)
    d.update(k)
    return d


def test_xover_offset_moves_the_aspx_range():
    """Pseudocode 68: sbg_sig_highres starts aspx_xover_subband_offset
    groups into the master table; Pseudocode 70 counts the noise groups
    over log2(sbz / sbx), from the crossover (it used sba: equal only at
    offset 0). Both change how many envelope values a frame carries."""
    from ac4 import aspx_bands as A
    t0, t1 = A.tables_for(_cfg(), 0), A.tables_for(_cfg(), 1)
    assert t0["master"] == t1["master"]
    assert t1["n_hi"] == t0["n_hi"] - 1
    assert t1["hi"] == t0["master"][1:]
    assert t1["sbx"] == t0["master"][1]
    assert t1["n_lo"] == t1["n_hi"] - t1["n_hi"] // 2
    # 3 * log2(44 / 26) = 2.28 -> 2 noise groups (3 from sba = 24)
    assert t1["sbx"] == 26 and t1["master"][-1] == 44
    assert t1["n_noise"] == 2 and t0["n_noise"] == 3


def test_aspx_data_reads_and_holds_the_xover_offset(monkeypatch):
    """aspx_data reads aspx_xover_subband_offset in I-frames, parses the
    block with the tables at that offset, and holds the offset per block
    for the frames up to the next I-frame."""
    from ac4 import aspx_parse as P
    seen = []
    def framing(b, cfg, nats, ifr, st, key):
        st[key] = 16
        return dict(int_class=P.FIXFIX, num_env=1, num_noise=1,
                    freq_res=[1], borders=[0, 16], tsg_ptr=-1,
                    sine_at_end=-1)

    monkeypatch.setattr(P, "aspx_framing", framing)
    monkeypatch.setattr(P, "delta_dir", lambda b, f: ([0], [0]))
    monkeypatch.setattr(P, "hfgen_iwc",
                        lambda b, t, *a, **k: seen.append(t) or [[0], [0]])
    monkeypatch.setattr(P, "ec_data", lambda *a: [[0]])
    state = {}
    w = BW().u(3, 1).u(1, 1)                 # offset 1, then aspx_balance
    out = P.aspx_data(Bits(w.bytes()), {}, None, _cfg(), 16, True, True,
                      state, 0)
    assert out[0]["xover"] == 1 and seen[-1]["sbx"] == 26
    out = P.aspx_data(Bits(BW().u(1, 1).bytes()), {}, None, _cfg(), 16,
                      False, True, state, 0)
    assert out[0]["xover"] == 1 and out[0]["t"]["n_hi"] == 9
    # another block keeps its own offset
    out = P.aspx_data(Bits(BW().u(1, 1).bytes()), {}, None, _cfg(), 16,
                      False, True, state, 1)
    assert out[0]["xover"] == 0


# ------------------------------------------------------ the C decoder too
C_CLI = __import__("os").path.join(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__)),
    "..", "c", "ac4dec-c")


@pytest.mark.skipif(not __import__("os").path.exists(C_CLI),
                    reason="C decoder not built")
@pytest.mark.parametrize("fri", [0, 2, 3, 5, 7, 8, 10, 11, 13])
def test_c_decoder_at_every_frame_length(tmp_path, fri):
    """The C decoder (ac4dec-c, internal rate) gives the Python's samples
    for the synthetic stream at every frame length, and the external rate
    with the ratio of Table 83."""
    import subprocess
    from ac4.decoder import Decoder
    from test_inputs import syncframe
    fl = TABLE_83[fri][0]
    sub = _simple_pair(fl, 7, 110)
    frames = [_frame(sub, seq=5 + i, fri=fri) for i in range(5)]
    src = tmp_path / "s.ac4"
    src.write_bytes(b"".join(syncframe(f) for f in frames))
    raw = tmp_path / "s.f64"
    subprocess.run([C_CLI, str(src), str(tmp_path / "s.wav"), "--f64",
                    str(raw), "--internal", "-q"], check=True)
    got = np.fromfile(raw, "<f8").reshape(-1, 2)
    ref = Decoder("pair").decode_pass(frames)
    assert got.shape == ref.shape == (5 * fl, 2)
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()
    raw2 = tmp_path / "e.f64"
    subprocess.run([C_CLI, str(src), str(tmp_path / "e.wav"), "--f64",
                    str(raw2), "-q"], check=True)
    up, down = TABLE_83[fri][1]
    n_ext = -(-5 * fl * up // down)
    assert np.fromfile(raw2, "<f8").size == 2 * n_ext
