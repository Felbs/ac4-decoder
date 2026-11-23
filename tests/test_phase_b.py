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
    got = [{"fr": {"borders": [0, g["nats"]], "freq_res": [1]},
            "balance": 1},
           {"fr": {"borders": [0, g["nats"]], "freq_res": [1]},
            "balance": 1}]
    z = (lambda *a, **k: (np.zeros((n_sb, 1)), np.zeros((n_sb, 1))))
    monkeypatch.setattr(V, "reconstruct", lambda d, t, m, p: (None, p))
    monkeypatch.setattr(E, "noise_qscf", lambda d, t, p: (None, p))
    monkeypatch.setattr(E, "map_scf_joint", z)
    monkeypatch.setattr(E, "map_noise_joint", z)
    monkeypatch.setattr(E, "estimate", lambda *a, **k: np.zeros((n_sb, 1)))
    monkeypatch.setattr(E, "estimate_groups",
                        lambda *a, **k: np.zeros((n_sb, 1)))
    monkeypatch.setattr(E, "adjust", lambda *a, **k: (np.zeros((n_sb, 1)),
                                                      np.ones((n_sb, 1)),
                                                      np.zeros((n_sb, 1))))
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
                        lambda b, t, *a, **k: seen.append(t) or [
                            dict(tna=[0], ah=[0], fic=None, tic=None)] * 2)
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


# --------------------------------------------- A-CPL: ASPX_ACPL_1 / _2
from test_acpl import _Ref                                   # noqa: E402


class _Ref1(_Ref):
    """Pseudocodes 109-116 for the modules of acpl_data_1ch(), sample by
    sample (decorrelators and ducker from test_acpl._Ref)."""

    def __init__(self, n_mod):
        super().__init__()
        self.prevs = [None] * n_mod

    def module(self, m, x0, x1, fr):
        from ac4 import acpl as ACPL
        cfg = fr["cfg"]
        pb = ACPL.sb_to_pb(cfg["num_param_bands"])
        tab = ((ACPL.ALPHA_FINE, ACPL.IBETA_FINE, ACPL.BETA_FINE)
               if cfg["quant_mode"] == 0 else
               (ACPL.ALPHA_COARSE, ACPL.IBETA_COARSE, ACPL.BETA_COARSE))
        sets = [{"a": [tab[0][qa] for qa in fr["q"][0][ps]],
                 "b": [tab[2][qb][tab[1][qa]]
                       for qa, qb in zip(fr["q"][0][ps], fr["q"][1][ps])]}
                for ps in range(len(fr["q"][0]))]
        if self.prevs[m] is None:
            self.prevs[m] = {k: [sets[0][k][pb[sb]] for sb in range(64)]
                             for k in ("a", "b")}
        self.prev = self.prevs[m]
        n = len(x0)
        x0in = [[2 * x0[t][s] for s in range(64)] for t in range(n)]
        x1in = [[2 * x1[t][s] for s in range(64)] for t in range(n)]
        y = self.duck(m, self.decorrelate(m, x0in))
        z0 = [[0j] * 64 for _ in range(n)]
        z1 = [[0j] * 64 for _ in range(n)]
        for sb in range(64):
            for ts in range(n):
                a = self.interpolate("a", sets, pb, fr["interp"],
                                     fr["timeslots"], sb, ts)
                b = self.interpolate("b", sets, pb, fr["interp"],
                                     fr["timeslots"], sb, ts)
                if sb < cfg["qmf_band"]:
                    z0[ts][sb] = 0.5 * (x0in[ts][sb] + x1in[ts][sb])
                    z1[ts][sb] = 0.5 * (x0in[ts][sb] - x1in[ts][sb])
                else:
                    z0[ts][sb] = 0.5 * (x0in[ts][sb] * (1 + a)
                                        + y[ts][sb] * b)
                    z1[ts][sb] = 0.5 * (x0in[ts][sb] * (1 - a)
                                        - y[ts][sb] * b)
        self.prevs[m] = {k: [sets[-1][k][pb[sb]] for sb in range(64)]
                         for k in ("a", "b")}
        return z0, z1


def _frame_1ch(rng, cfg, n_sets, interp):
    from ac4 import acpl_parse as AP
    q = []
    for data_type, _ in AP.DATA_1CH:
        lo, hi = AP.Q_RANGE[(data_type, AP.QUANT[cfg["quant_mode"]])]
        q.append(rng.integers(lo, hi + 1,
                              (n_sets, cfg["num_param_bands"])).tolist())
    ts = sorted(rng.integers(0, 24, n_sets).tolist()) if interp else []
    return dict(cfg=cfg, q=q, interp=interp, timeslots=tuple(ts))


@pytest.mark.parametrize("nb,qm,qmf_band,n_mod", [
    (15, 0, 0, 2), (9, 1, 0, 1), (12, 0, 5, 2), (7, 1, 3, 1)])
def test_acpl_1ch_against_the_pseudocode(nb, qm, qmf_band, n_mod):
    """acpl.render_1ch against Pseudocodes 109-116 written out sample by
    sample: one or two modules (decorrelators D0, D1), full and partial
    (acpl_qmf_band, with a residual), one and two parameter sets, smooth
    and steep interpolation, the state carried over four frames."""
    from ac4 import acpl as ACPL
    rng = np.random.default_rng(nb + qmf_band)
    cfg = dict(num_param_bands=nb, quant_mode=qm, qmf_band=qmf_band,
               param_band=int(ACPL.sb_to_pb(nb)[qmf_band]) if qmf_band
               else 0)
    frames = [[_frame_1ch(rng, cfg, ns, it) for _ in range(n_mod)]
              for ns, it in ((1, 0), (2, 0), (1, 1), (2, 1))]
    n = 24 * len(frames)

    def cplx():
        return rng.standard_normal((64, n)) + 1j * rng.standard_normal(
            (64, n))
    Q0 = [cplx() for _ in range(n_mod)]
    Q1 = [cplx() if qmf_band else None for _ in range(n_mod)]
    got = ACPL.render_1ch(Q0, Q1, frames, ts_offset=0)
    ref = _Ref1(n_mod)
    for f, fr in enumerate(frames):
        sl = slice(24 * f, 24 * f + 24)
        for m in range(n_mod):
            x1 = (Q1[m][:, sl] if Q1[m] is not None
                  else np.zeros((64, 24))).T.tolist()
            z0, z1 = ref.module(m, Q0[m][:, sl].T.tolist(), x1, fr[m])
            assert np.allclose(got[m][0][:, sl], np.array(z0).T,
                               rtol=1e-11, atol=1e-11), (f, m, "z0")
            assert np.allclose(got[m][1][:, sl], np.array(z1).T,
                               rtol=1e-11, atol=1e-11), (f, m, "z1")


def test_acpl_1ch_without_data_and_concealed():
    """No data: 5_X keeps the core on z0 and silence on z1; a pair splits
    (x0 + x1, x0 - x1). A concealed frame after data holds the
    parameters."""
    from ac4 import acpl as ACPL
    rng = np.random.default_rng(2)
    cfg = dict(num_param_bands=15, quant_mode=0, qmf_band=0, param_band=0)
    fr = [_frame_1ch(rng, cfg, 1, 0)]
    n = 96
    Q0 = [rng.standard_normal((64, n)) + 0j]
    Q1 = [rng.standard_normal((64, n)) + 0j]
    z = ACPL.render_1ch(Q0, Q1, [None, ACPL.CONCEALED, fr, ACPL.CONCEALED],
                        ts_offset=0)
    assert np.array_equal(z[0][0][:, :48], Q0[0][:, :48])
    assert not np.any(z[0][1][:, :48])
    zs = ACPL.render_1ch(Q0, Q1, [None, None], ts_offset=0, split=True)
    assert np.allclose(zs[0][0][:, :48], Q0[0][:, :48] + Q1[0][:, :48])
    assert np.allclose(zs[0][1][:, :48], Q0[0][:, :48] - Q1[0][:, :48])
    held = ACPL.render_1ch(Q0, Q1, [None, None, fr, fr], ts_offset=0)
    assert np.allclose(z[0][0], held[0][0], atol=1e-13)
    assert np.allclose(z[0][1], held[0][1], atol=1e-13)


def _huff_put(w, book, sym):
    n, code = _huff_code(book, sym)
    w.u(n, code)


def test_acpl_config_and_data_1ch_parse():
    """Tables 59, 61, 63-65: acpl_config_1ch (PARTIAL reads
    acpl_qmf_band_minus1, the parameters starting at its band) and
    acpl_data_1ch, written with the codebooks, differentially decoded
    (Pseudocode 121) with the books' cb_off."""
    from ac4 import acpl as ACPL
    from ac4 import acpl_parse as AP
    from ac4 import hcb as H
    books = AP.load_books(H.parse_c(H.DEFAULT_C))
    w = BW().u(2, 1).u(1, 0).u(3, 4)        # 12 bands, fine, qmf_band 5
    cfg = AP.acpl_config_1ch(Bits(w.bytes()), AP.PARTIAL)
    assert cfg["num_param_bands"] == 12 and cfg["qmf_band"] == 5
    assert cfg["param_band"] == int(ACPL.sb_to_pb(12)[5]) == 4
    full = AP.acpl_config_1ch(Bits(BW().u(2, 3).u(1, 1).bytes()), AP.FULL)
    assert (full["num_param_bands"], full["qmf_band"],
            full["param_band"], full["quant_mode"]) == (7, 0, 0, 1)
    # one parameter set, steep at slot 9; alpha DIFF_FREQ, beta DIFF_TIME
    w = BW().u(1, 1).u(1, 0).u(5, 9)
    w.u(1, 0)
    _huff_put(w, books["ACPL_HCB_ALPHA_FINE_F0"], 20)
    for d in range(4 + 1, 12):
        _huff_put(w, books["ACPL_HCB_ALPHA_FINE_DF"], 32 + (d % 3) - 1)
    w.u(1, 1)
    for d in range(4, 12):
        _huff_put(w, books["ACPL_HCB_BETA_FINE_DT"], 8 + 1)
    data = AP.acpl_data_1ch(Bits(w.bytes()), books, cfg)
    assert data["framing"] == dict(interpolation_type=1, num_param_sets=1,
                                   param_timeslot=[9])
    q_prev = [None, [2] * 12]
    q = AP.quantised_1ch(data, cfg, q_prev)
    alpha = [0] * 4 + [20]
    for d in range(5, 12):
        alpha.append(alpha[-1] + (d % 3) - 1)
    assert q[0] == [alpha] and q[1] == [[0] * 4 + [3] * 8]
    assert q_prev == [alpha, [0] * 4 + [3] * 8]


def _aspx_acpl_pair(mode, seed):
    """A channel_pair_element I-frame in stereo_codec_mode 2 (ASPX_ACPL_1)
    or 3 (ASPX_ACPL_2), written bit by bit: aspx_config, acpl_config_1ch,
    companding_control(1), the core track(s), aspx_data_1ch (FIXFIX, one
    envelope), acpl_data_1ch (one parameter set, frequency coded)."""
    from ac4 import aspx_bands as A
    from ac4 import aspx_parse as P
    from ac4 import acpl_parse as AP
    from ac4 import hcb as H
    from ac4 import spectral as S
    rng = np.random.default_rng(seed)
    T = CH.load_tables()
    arrays = H.parse_c(H.DEFAULT_C)
    books = AP.load_books(arrays)
    abooks = {n[:-4]: S.Huff(arrays[n], arrays[n[:-4] + "_CW"])
              for n in arrays if n.endswith("_LEN") and n.startswith("ASPX_")}
    cfg = dict(quant_mode_env=0, start_freq=2, stop_freq=1,
               master_freq_scale=1, interpolation=0, preflat=0, limiter=1,
               noise_sbg=1, num_env_bits_fixfix=0, freq_res_mode=1)
    t = A.tables_for(cfg, 0)
    w = BW().u(2, mode)
    for k in ("quant_mode_env", "start_freq", "stop_freq",
              "master_freq_scale", "interpolation", "preflat", "limiter",
              "noise_sbg", "num_env_bits_fixfix", "freq_res_mode"):
        w.u({"start_freq": 3, "stop_freq": 2, "noise_sbg": 2,
             "freq_res_mode": 2}.get(k, 1), cfg[k])
    nb = 9
    w.u(2, 2).u(1, 0)                         # acpl_config_1ch: 9, fine
    if mode == 2:
        w.u(3, 3)                             # acpl_qmf_band 4
    w.u(1, 1)                                 # companding: on
    band = CH.SFB_TABLE[1536]

    def track(nbands, nbits):
        w.u(4, 11).u(5, nbands - 1)           # one section, codebook 11
        for _ in range(band[nbands] // 2):
            a, b = rng.integers(1, 6), rng.integers(0, 6)
            _huff_put(w, T["cb"][11], int(a) * 17 + int(b))
            for v in (a, b):
                if v:
                    w.u(1, int(rng.integers(0, 2)))
        w.u(8, 100)                           # reference scale factor
        for _ in range(nbands - 1):
            _huff_put(w, T["sf"], 60)         # same scale factor
        w.u(1, 0)                             # no noise fill
    if mode == 2:
        w.u(1, 0)                             # no MDCT stereo processing
        w.u(1, 0).u(1, 1).u(6, 20)            # m: ASF, long, max_sfb 20
        w.u(1, 0).u(1, 1).u(5, 6)             # s: ASF, long, max_sfb_side 6
        track(20, 6)
        track(6, 6)
    else:
        w.u(1, 0).u(1, 1).u(6, 20)
        track(20, 6)
    # aspx_data_1ch
    w.u(3, 0)                                 # aspx_xover_subband_offset
    w.u(1, 0).u(1, 0)                         # FIXFIX, one envelope
    w.u(1, 0).u(1, 0)                         # delta_dir: frequency
    for _ in range(t["n_noise"]):
        w.u(2, 1)                             # aspx_tna_mode
    w.u(1, 0).u(1, 0).u(1, 0)                 # no ah, fic, tic
    _huff_put(w, abooks["ASPX_HCB_ENV_LEVEL_15_F0"], 30)
    dfc = (len(abooks["ASPX_HCB_ENV_LEVEL_15_DF"].map) - 1) // 2
    for _ in range(t["n_lo"] - 1):
        _huff_put(w, abooks["ASPX_HCB_ENV_LEVEL_15_DF"], dfc)
    _huff_put(w, abooks["ASPX_HCB_NOISE_LEVEL_F0"], 10)
    nfc = (len(abooks["ASPX_HCB_NOISE_LEVEL_DF"].map) - 1) // 2
    for _ in range(t["n_noise"] - 1):
        _huff_put(w, abooks["ASPX_HCB_NOISE_LEVEL_DF"], nfc)
    # acpl_data_1ch: smooth, one set; alpha and beta frequency coded
    w.u(1, 0).u(1, 0)
    start = int(__import__("ac4.acpl", fromlist=["x"]).sb_to_pb(nb)[4]) \
        if mode == 2 else 0
    for name, f0, off in (("ALPHA", 22, 32), ("BETA", 3, 8)):
        w.u(1, 0)
        _huff_put(w, books[f"ACPL_HCB_{name}_FINE_F0"], f0)
        for _ in range(start + 1, nb):
            _huff_put(w, books[f"ACPL_HCB_{name}_FINE_DF"],
                      off + int(rng.integers(-1, 2)))
    w.align()
    body = w.bytes()
    return BW().u(15, len(body)).u(1, 0).bytes() + body


@pytest.mark.parametrize("mode", [2, 3])
def test_pair_aspx_acpl_stream(tmp_path, mode):
    """A stereo stream in ASPX_ACPL_1 / _2 (written bit by bit) decodes
    with nothing concealed (every frame closes on audio_size), to two
    different, non-silent channels; the C decoder gives the same samples
    when it is built."""
    import os
    import subprocess
    from ac4.decoder import Decoder
    from test_inputs import syncframe
    frames = [_frame(_aspx_acpl_pair(mode, i), seq=5 + i, fri=3)
              for i in range(8)]
    dec = Decoder("pair", guard=False)
    dec.hf_ratio = None              # (the synthetic envelopes are loud)
    y = dec.decode_pass(frames)
    assert dec.n_concealed == 0, dict(dec.concealed.reasons)
    assert dec.stream.acpl1 == "pair"
    s = slice(4 * 1536, None)
    rl, rr = (np.sqrt(np.mean(y[s, c] ** 2)) for c in (0, 1))
    assert rl > 0 and rr > 0
    assert np.abs(y[s, 0] - y[s, 1]).max() > 0.01 * max(rl, rr)
    if os.path.exists(C_CLI):
        src = tmp_path / "p.ac4"
        src.write_bytes(b"".join(syncframe(f) for f in frames))
        raw = tmp_path / "p.f64"
        subprocess.run([C_CLI, str(src), str(tmp_path / "p.wav"), "--f64",
                        str(raw), "--internal", "--no-guard", "-q"],
                       check=True)
        got = np.fromfile(raw, "<f8").reshape(-1, 2)
        dref = Decoder("pair", guard=False)
        ref = dref.decode_pass(frames)
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()


# ------------------------------------------------ A-SPX tools (item 3)
def test_tna_alpha_against_pseudocodes_86_87():
    """The covariance method of Pseudocodes 86, 87 written out with
    loops (complex), against aspx_tools.tna_alpha."""
    from ac4 import aspx_tools as TL
    rng = np.random.default_rng(5)
    nts, hf, sba = 24, 6, 20
    Qa = rng.standard_normal((64, 96)) + 1j * rng.standard_normal((64, 96))
    a0 = 30
    a0g, a1g = TL.tna_alpha(Qa, sba, a0, nts, hf)
    for sb in range(sba):
        ext = [Qa[sb, a0 - 4 + i] for i in range(nts + hf + 4)]
        cov = [[0j] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(1, 3):
                for ts in range(4, nts + hf + 4, 2):
                    cov[i][j] += ext[ts - 2 * i] * np.conj(ext[ts - 2 * j])
        den = cov[2][2] * cov[1][1] - abs(cov[1][2]) ** 2 / (1 + 2 ** -20)
        a1 = (cov[0][1] * cov[1][2] - cov[0][2] * cov[1][1]) / den
        a0_ = (-cov[0][1] + a1 * np.conj(cov[1][2])) / cov[1][1]
        if abs(a0_) >= 4 or abs(a1) >= 4:
            a0_ = a1 = 0
        assert abs(a0g[sb] - a0_) < 1e-12 and abs(a1g[sb] - a1) < 1e-12


def test_chirp_factors_table_195():
    """Pseudocode 88: new_chirp from Table 195, smoothed against the
    previous interval's chirp, zeroed below 0.015625."""
    from ac4 import aspx_tools as TL
    st = TL.new_state()
    c = TL.chirps([2, 0, 3], st)                 # previous modes 0, chirp 0
    assert np.allclose(c, [0.90625 * 0.9, 0.0, 0.90625 * 0.98])
    c2 = TL.chirps([1, 1, 0], st)                # prev 2, 0, 3
    want = [0.75 * 0.75 + 0.25 * c[0], 0.90625 * 0.6, 0.25 * c[2]]
    assert np.allclose(c2, want)


def test_preflat_gain_is_the_cubic_fit():
    """Pseudocode 85: the gain vector from a least-squares cubic through
    the low band's energies in dB (numpy's polyfit as the reference)."""
    from ac4 import aspx_tools as TL
    rng = np.random.default_rng(6)
    sbx = 26
    Qa = (rng.standard_normal((64, 40)) + 1j * rng.standard_normal((64, 40))) \
        * np.linspace(30, 1, 64)[:, None]
    gv = TL.preflat_gain(Qa, sbx, 5, 29)
    pw = 10 * np.log10((np.abs(Qa[:sbx, 5:29]) ** 2).sum(axis=1) / 24 + 1)
    poly = np.polyfit(np.arange(sbx), pw, 3)
    want = 10 ** ((pw.mean() - np.polyval(poly, np.arange(sbx))) / 20)
    assert np.allclose(gv, want, rtol=1e-9)


def test_hf_generation_pseudocode_89():
    """Pseudocode 89 written out: each patched subband is the source plus
    chirp * alpha0 * the source two slots back plus chirp^2 * alpha1 *
    four slots back, over 1 / gain_vec; the chirp of its noise group."""
    from ac4 import aspx_bands as A
    from ac4 import aspx_tools as TL
    t = A.tables_for(_cfg(), 0)
    rng = np.random.default_rng(7)
    Qa = rng.standard_normal((64, 60)) + 1j * rng.standard_normal((64, 60))
    al0 = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    al1 = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    ch = np.linspace(0.3, 0.9, t["n_noise"])
    gv = rng.uniform(0.5, 2, 64)
    Qh = np.zeros_like(Qa)
    TL.generate(Qa, Qh, t, 10, 40, al0, al1, ch, gv)
    sum_sb, g = 0, 0
    for i in range(len(t["patch_num_sb"])):
        for sb in range(t["patch_num_sb"][i]):
            hi = t["sbx"] + sum_sb + sb
            if g + 1 < len(t["noise"]) and t["noise"][g + 1] == hi:
                g += 1
            p = t["patch_start_sb"][i] + sb
            for ts in range(10, 40):
                want = (Qa[p, ts] + ch[g] * al0[p] * Qa[p, ts - 2]
                        + ch[g] ** 2 * al1[p] * Qa[p, ts - 4]) / gv[p]
                assert abs(Qh[hi, ts] - want) < 1e-12
        sum_sb += t["patch_num_sb"][i]


def test_sinusoid_placement_pseudocodes_92_93():
    """A flagged high-resolution group gets its sinusoid in its middle
    subband from aspx_tsg_ptr on (or from envelope 0 when the previous
    interval had one there, or ended on its pointer); sine_area covers the
    group of the envelope's own resolution."""
    from ac4 import aspx_bands as A
    from ac4 import aspx_tools as TL
    t = A.tables_for(_cfg(), 0)
    n_sb = t["num_sb_aspx"]
    ah = [0] * t["n_hi"]
    ah[2] = 1
    fr = dict(borders=[0, 6, 16], tsg_ptr=1, sine_at_end=-1,
              freq_res=[0, 1])
    s = TL.sine_idx_sb(fr, ah, t, np.zeros(64, bool), n_sb)
    a, z = t["hi"][2] - t["sbx"], t["hi"][3] - t["sbx"]
    mid = int(0.5 * (a + z))
    assert not s[:, 0].any() and s[mid, 1] and s[:, 1].sum() == 1
    prev = np.zeros(64, bool)
    prev[mid] = True
    assert TL.sine_idx_sb(fr, ah, t, prev, n_sb)[mid, 0]
    area = TL.sine_area(s, t, fr)
    assert area[a:z, 1].all() and area[:, 1].sum() == z - a
    lo = t["lo"]
    g = max(i for i in range(len(lo) - 1) if lo[i] - t["sbx"] <= mid)
    assert not area[:, 0].any()


def test_adjust_with_sinusoids_pseudocodes_94_101():
    """aspx_envadj.adjust with sinusoids against Pseudocodes 94-101
    written out with loops."""
    from ac4 import aspx_bands as A
    from ac4 import aspx_envadj as E
    t = A.tables_for(_cfg(), 0)
    lim = E.sbg_lim(t)
    n_sb, n_env, sbx = t["num_sb_aspx"], 2, t["sbx"]
    rng = np.random.default_rng(8)
    sig = rng.uniform(1, 100, (n_sb, n_env))
    est = rng.uniform(1, 100, (n_sb, n_env))
    noi = rng.uniform(0.01, 2, (n_sb, n_env))
    sidx = np.zeros((n_sb, n_env), bool)
    sidx[3, 1] = sidx[9, 0] = True
    area = np.zeros_like(sidx)
    area[2:5, 1] = area[8:11, 0] = True
    tr = np.array([False, True])
    g, nl, sl = E.adjust(None, t, sig, est, lim, noi, tr, sine_idx=sidx,
                         sine_area=area, with_sine=True)
    for e in range(n_env):
        snf = sig[:, e] / (1 + noi[:, e])
        sine = np.sqrt(snf * sidx[:, e])
        nlev = np.sqrt(snf * noi[:, e])
        gain = np.empty(n_sb)
        for sb in range(n_sb):
            if not area[sb, e]:
                den = 1.0 + est[sb, e]
                if not tr[e]:
                    den *= 1 + noi[sb, e]
                gain[sb] = np.sqrt(sig[sb, e] / den)
            else:
                den = (1.0 + est[sb, e]) * (1 + noi[sb, e])
                gain[sb] = np.sqrt(sig[sb, e] * noi[sb, e] / den)
        maxg = np.empty(n_sb)
        for i in range(len(lim) - 1):
            a, z = lim[i] - sbx, lim[i + 1] - sbx
            maxg[a:z] = min(np.sqrt(sig[a:z, e].sum()
                                    / (1e-12 + est[a:z, e].sum())) * 1.41254,
                            1e5)
        nlim = np.minimum(nlev, nlev * maxg / gain)
        glim = np.minimum(gain, maxg)
        for i in range(len(lim) - 1):
            a, z = lim[i] - sbx, lim[i + 1] - sbx
            nom = 1e-12 + sig[a:z, e].sum()
            den = 1e-12 + (est[a:z, e] * glim[a:z] ** 2).sum()
            den += (sine[a:z] ** 2).sum()
            for sb in range(a, z):
                if not (sine[sb] != 0 or tr[e]):
                    den += nlim[sb] ** 2
            bst = min(np.sqrt(nom / den), 1.584893192)
            assert np.allclose(g[a:z, e], glim[a:z] * bst, rtol=1e-12)
            assert np.allclose(nl[a:z, e], nlim[a:z] * bst, rtol=1e-12)
            assert np.allclose(sl[a:z, e], sine[a:z] * bst, rtol=1e-12)


def test_tna_off_and_no_preflat_is_the_plain_copy():
    """With every aspx_tna_mode 0 (chirp 0) and aspx_preflat 0 the HF
    generator is the patch copy of before."""
    from ac4 import aspx_bands as A
    from ac4 import aspx_tools as TL
    from ac4 import aspx_hfgen as G
    t = A.tables_for(_cfg(), 0)
    rng = np.random.default_rng(9)
    Qa = rng.standard_normal((64, 50)) + 1j * rng.standard_normal((64, 50))
    Qh = Qa.copy()
    st = TL.new_state()
    ch = TL.chirps([0] * t["n_noise"], st)
    TL.generate(Qa, Qh, t, 6, 40, np.zeros(64), np.zeros(64), ch, None)
    ref = G.hf_generate(Qa, t)
    assert np.array_equal(Qh[:, 6:40], ref[:, 6:40])


def test_fixfix_freq_res_mode_0_reads_one_bit():
    """Table 53: in aspx_freq_res_mode 0 a FIXFIX interval sends a single
    aspx_freq_res (Pseudocode 76 copies it to every envelope); it used to
    read one per envelope."""
    from ac4 import aspx_parse as P
    w = BW().u(1, 0).u(2, 2).u(1, 1).u(1, 0)    # FIXFIX, 4 env, freq_res 1
    b = Bits(w.bytes())
    fr = P.aspx_framing(b, dict(num_env_bits_fixfix=1, freq_res_mode=0), 12,
                        False, {}, (0, 0))
    assert fr["num_env"] == 4 and fr["freq_res"] == [1] * 4 and b.p == 4


# ------------------------------------------------------ companding (item 4)
def _levels_ref(Q, ta, tz, sbx, scale):
    out = []
    for ts in range(ta, tz):
        acc = 0.0
        for sb in range(sbx):
            re, im = abs(Q[sb, ts].real), abs(Q[sb, ts].imag)
            acc += max(re, im) + 0.5 * min(re, im)
        out.append(0.9105 * (acc / sbx) / scale)
    return np.array(out)


@pytest.mark.parametrize("sync,on,avg", [(0, (1, 0), 1), (1, (1,), None),
                                         (1, (0,), 1), (0, (0, 1), 0)])
def test_companding_frame_is_5_7_5_2(sync, on, avg):
    """compand_frame against 5.7.5.2 written out: per slot of each
    channel's interval L = 0.9105 mean(max(|re|,|im|) + min/2) over
    subbands 0..sbx-1 (levels relative to COMPAND_SCALE), gain
    L^((1-a)/a) 2^(1/a); b_compand_avg: the interval's mean level; sync:
    the channels' mean gain per slot. Companding is on by default, with
    the levels relative to IMDCT_K * 2^16.5 (README, Interpretations)."""
    from ac4 import aspx_bands as A
    from ac4 import aspx_render as R
    from ac4 import filterbank as FB
    assert R.COMPANDING
    assert R.COMPAND_SCALE == FB.IMDCT_K * 2.0 ** 16.5
    rng = np.random.default_rng(11)
    geo = A.geometry(1536)
    Qs = [(rng.standard_normal((64, 96)) + 1j * rng.standard_normal((64, 96)))
          * 1e5 for _ in range(2)]
    orig = [q.copy() for q in Qs]
    fl = [dict(on=on[0] if sync else on[i], avg=avg, sync=sync)
          for i in range(2)]
    got = [{"fr": {"borders": [0, 12]}, "compand": fl[0]},
           {"fr": {"borders": [2, 14]}, "compand": fl[1]}]
    sbx = 20
    R.compand_frame(Qs, got, sbx, 1, geo)
    a = 0.65
    G, expo = 2 ** (1 / a), (1 - a) / a
    spans = [(24 + 18 + 0, 24 + 18 + 24), (24 + 18 + 4, 24 + 18 + 28)]
    Ls = [_levels_ref(o, ta, tz, sbx, R.COMPAND_SCALE)
          for o, (ta, tz) in zip(orig, spans)]
    for c in range(2):
        ta, tz = spans[c]
        if sync:
            n = min(len(x) for x in Ls)
            g = (Ls[0][:n] ** expo + Ls[1][:n] ** expo) / 2
            if not on[0]:
                g = np.full(n, g.mean())
            tz = ta + n
        elif on[c]:
            g = Ls[c] ** expo
        elif avg:
            g = np.full(tz - ta, Ls[c].mean() ** expo)
        else:
            assert np.array_equal(Qs[c], orig[c])
            continue
        want = orig[c].copy()
        want[:sbx, ta:tz] *= g * G
        assert np.allclose(Qs[c], want, rtol=1e-12)
