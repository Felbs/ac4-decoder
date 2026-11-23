# SPDX-License-Identifier: Apache-2.0
"""Unit tests that need no broadcast audio: tables, the bit reader, the
transforms, Huffman decoding, the resampler, and the MP4 reader."""
import numpy as np
import pytest

from ac4 import filterbank as FB
from ac4 import hcb as H
from ac4 import mdct as T
from ac4 import mp4
from ac4 import qmf as Q
from ac4 import resample as RS
from ac4 import sfb as B
from ac4 import spectral as S
from ac4 import aspx_envadj as E
from ac4.toc_v1 import Bits


# ---------------------------------------------------------------- tables
def test_huffman_tables_complete_and_prefix_free():
    arrays = H.parse_c(H.DEFAULT_C)          # no ETSI file -> banked JSON
    books = [n[:-4] for n in arrays if n.endswith("_LEN")]
    assert len(books) >= 20
    for nm in books:
        lens, cw = arrays[nm + "_LEN"], arrays[nm + "_CW"]
        assert len(lens) == len(cw), nm
        assert abs(H.kraft(lens) - 1.0) < 1e-9, nm
        ok, why = H.prefix_free(lens, cw)
        assert ok, (nm, why)


def test_sfb_offsets_gate():
    for length in (1536, 768, 384, 192, 96):
        off = B.offsets_for(length)
        assert off[0] == 0 and off[-1] == length
        assert all(a < b for a, b in zip(off, off[1:]))
    assert len(B.offsets_for(1536)) - 1 == B.NUM_SFB_B1[1536]


def test_qmf_window_and_noise_table_shapes():
    w = Q.qwin()
    assert w.shape == (640,) and np.isfinite(w).all()
    nz = E.noise_table()
    assert nz.shape == (512,) and np.iscomplexobj(nz)


# ------------------------------------------------------------- bit reader
def test_bits_reads_msb_first_and_raises_at_end():
    b = Bits(bytes([0b10110011, 0b01000000]))
    assert b.u(1) == 1
    assert b.u(3) == 0b011
    assert b.u(4) == 0b0011
    assert b.u(2) == 0b01
    b.align()
    assert b.p == 16
    with pytest.raises(EOFError):
        b.u(1)


def test_variable_bits_offsets_each_continuation():
    # variable_bits(2): chunk 01, more=1, chunk 10, more=0
    # -> ((1) << 2) + (1 << 2) + 2 = 10  (TS 103 190-1 Table 3 rule)
    b = Bits(bytes([0b01110000]))
    assert b.vb(2) == 10
    b = Bits(bytes([0b11000000]))            # chunk 11, more=0 -> 3
    assert b.vb(2) == 3


# --------------------------------------------------------------- Huffman
def _encode(lens, cws, syms):
    bits = "".join(format(cws[s] & ((1 << lens[s]) - 1), f"0{lens[s]}b")
                   for s in syms)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


@pytest.mark.parametrize("book", ["ASF_HCB_SCALEFAC", "ASF_HCB_1", "ASF_HCB_11"])
def test_huffman_round_trip_on_real_codebooks(book):
    arrays = H.parse_c(H.DEFAULT_C)
    if book + "_LEN" not in arrays:
        pytest.skip(f"{book} not in the banked tables")
    lens, cws = arrays[book + "_LEN"], arrays[book + "_CW"]
    hf = S.Huff(lens, cws)
    rng = np.random.default_rng(5)
    valid = [i for i, L in enumerate(lens) if L > 0]
    syms = [valid[k] for k in rng.integers(0, len(valid), 400)]
    b = Bits(_encode(lens, cws, syms))
    assert [hf.decode(b) for _ in syms] == syms


# ------------------------------------------------------------ transforms
def test_mdct_perfect_reconstruction():
    assert T.gate_perfect_reconstruction(64, verbose=False)


def test_kbd_princen_bradley():
    for n, al in ((16, 3.0), (32, 4.0), (96, 6.0), (192, 5.0), (1536, 3.0)):
        w = FB.kbd_left(n, al)
        full = np.concatenate([w, w[::-1]])
        assert np.abs(full[:n] ** 2 + full[n:] ** 2 - 1.0).max() < 1e-12


@pytest.mark.parametrize("seq", [
    [16] * 6,
    [16, 16, 8, 8, 16, 16],
    [16, 8, 8, 4, 4, 4, 4, 16, 16],
    [16, 4, 4, 8, 16, 16],
])
def test_filterbank_block_switching_is_orthogonal(seq):
    assert FB.gate(seq, 16, "reversed", trim=16, verbose=False) < 1e-9


def test_filterbank_literal_reading_is_rejected():
    # the negative control: the other reading of the window rule must fail
    seq = [16, 8, 8, 4, 4, 4, 4, 16, 16]
    assert FB.gate(seq, 16, "literal", trim=16, verbose=False) > 1e-6


def test_qmf_round_trip():
    lag, snr = Q.gate(Q.qwin(), 255, verbose=False)
    assert snr > 70.0
    assert 500 < lag < 700


def test_stream_resampler_is_cut_invariant(capsys):
    assert RS.stream_gate()


def test_resampler_length():
    assert RS.expected_length(300) == 300 * 1536 * 1001 // 960


# ------------------------------------------------------------------- MP4
def _box(t, payload):
    return (8 + len(payload)).to_bytes(4, "big") + t + payload


def _full(t, ver, flags, payload):
    return _box(t, bytes([ver]) + flags.to_bytes(3, "big") + payload)


def _trak(tid, entry):
    tkhd = _full(b"tkhd", 0, 0, bytes(8) + tid.to_bytes(4, "big") + bytes(68))
    stsd = _full(b"stsd", 0, 0, (1).to_bytes(4, "big") + _box(entry, bytes(8)))
    return _box(b"trak", tkhd + _box(b"mdia", _box(b"minf", _box(b"stbl", stsd))))


def _fragment(samples, hint_n=0, hint_size=0, ac4_tid=1, hint_tid=2,
              hint_first=False):
    """moof (+ optional hint traf with default size) + mdat."""
    def traf_media(doff):
        tfhd = _full(b"tfhd", 0, 0x20000, ac4_tid.to_bytes(4, "big"))
        body = len(samples).to_bytes(4, "big") + doff.to_bytes(4, "big", signed=True)
        body += b"".join(len(s).to_bytes(4, "big") for s in samples)
        return _box(b"traf", tfhd + _full(b"trun", 0, 0x201, body))

    def traf_hint(doff):
        tfhd = _full(b"tfhd", 0, 0x20010, hint_tid.to_bytes(4, "big")
                     + hint_size.to_bytes(4, "big"))
        body = hint_n.to_bytes(4, "big") + doff.to_bytes(4, "big", signed=True)
        return _box(b"traf", tfhd + _full(b"trun", 0, 0x001, body))

    media = b"".join(samples)
    hint = bytes(range(hint_size)) * hint_n
    # box sizes do not depend on the offset values, so lay out once
    mfhd = _full(b"mfhd", 0, 0, (1).to_bytes(4, "big"))
    moof_len = 8 + len(mfhd) + len(traf_media(0)) + (len(traf_hint(0)) if hint_n else 0)
    m_off, h_off = moof_len + 8, moof_len + 8 + len(media)
    if hint_first:
        h_off, m_off = moof_len + 8, moof_len + 8 + len(hint)
    trafs = [traf_media(m_off)] + ([traf_hint(h_off)] if hint_n else [])
    if hint_first:
        trafs.reverse()
    moof = _box(b"moof", mfhd + b"".join(trafs))
    data = (hint + media) if hint_first else (media + hint)
    return moof + _box(b"mdat", data)


def test_mp4_reads_samples_and_ignores_hint_track():
    s1 = [bytes([i]) * (10 + i) for i in range(5)]
    s2 = [bytes([0x40 + i]) * (7 + 2 * i) for i in range(3)]
    init = _box(b"ftyp", b"iso6" + bytes(4)) + _box(
        b"moov", _trak(1, b"ac-4") + _trak(2, b"mmth"))
    data = init + _fragment(s1, 5, 3) + _fragment(s2, 3, 3)
    assert mp4.ac4_track_ids(data) == [1]
    assert mp4.frames(data) == s1 + s2


def test_mp4_track_choice_with_hint_traf_first():
    s1 = [b"abc", b"defgh"]
    init = _box(b"moov", _trak(7, b"ac-4") + _trak(2, b"mmth"))
    data = init + _fragment(s1, 2, 4, ac4_tid=7, hint_first=True)
    assert mp4.frames(data) == s1


def test_mp4_bare_fragments_use_per_sample_sizes():
    s1 = [b"x" * 9, b"y" * 4]
    data = _fragment(s1, 4, 5)               # no moov at all
    assert mp4.frames(data) == s1


def test_mp4_skips_incomplete_trailing_fragment():
    s1, s2 = [b"a" * 5, b"b" * 6], [b"c" * 7, b"d" * 8]
    full = _fragment(s1) + _fragment(s2)
    assert mp4.frames(full) == s1 + s2
    assert mp4.frames(full[:-3]) == s1


def test_mp4_default_sample_size():
    tid = 3
    samples = [b"Q" * 6] * 4
    tfhd = _full(b"tfhd", 0, 0x20010, tid.to_bytes(4, "big") + (6).to_bytes(4, "big"))
    mfhd = _full(b"mfhd", 0, 0, (1).to_bytes(4, "big"))
    trun = _full(b"trun", 0, 0x001, bytes(8))
    doff = len(_box(b"moof", mfhd + _box(b"traf", tfhd + trun))) + 8
    trun = _full(b"trun", 0, 0x001, (4).to_bytes(4, "big") + doff.to_bytes(4, "big"))
    moof = _box(b"moof", mfhd + _box(b"traf", tfhd + trun))
    data = _box(b"moov", _trak(tid, b"ac-4")) + moof + _box(b"mdat", b"".join(samples))
    assert mp4.frames(data) == samples


# ---------------------------------------------------------------- TOC
def _bits(s):
    """'0101..' -> Bits over the string padded to whole bytes."""
    s = s + "0" * (-len(s) % 8)
    return Bits(bytes(int(s[i:i + 8], 2) for i in range(0, len(s), 8)))


# TS 103 190-2 table 56: channel_mode code -> ch_mode
CHANNEL_MODE_CODES = [
    ("0", 0), ("10", 1), ("1100", 2), ("1101", 3), ("1110", 4),
    ("1111000", 5), ("1111001", 6), ("1111010", 7), ("1111011", 8),
    ("1111100", 9), ("1111101", 10), ("11111100", 11), ("11111101", 12),
    ("111111100", 13), ("111111101", 14), ("111111110", 15),
    # 0b111111111 + variable_bits(2): chunk 00, more 0 -> 16 + 0;
    # chunk 01, more 1, chunk 10, more 0 -> 16 + ((1 << 2) + 4 + 2)
    ("111111111" + "000", 16), ("111111111" + "011" + "100", 26),
]


@pytest.mark.parametrize("code,ch_mode", CHANNEL_MODE_CODES)
def test_channel_mode_prefix_code(code, ch_mode):
    from ac4 import toc
    b = _bits(code)
    assert toc.channel_mode(b) == ch_mode
    assert b.p == len(code)


@pytest.mark.parametrize("code,ch_mode", [c for c in CHANNEL_MODE_CODES
                                          if c[1] in (11, 12, 13, 14)])
def test_substream_info_chan_reads_top_channel_fields(code, ch_mode):
    """7.0.4 / 7.1.4 / 9.0.4 / 9.1.4 carry b_4_back_channels_present,
    b_centre_present and top_channels_present (4 bits) before
    b_bitrate_info (TS 103 190-2 6.2.1.8)."""
    from ac4 import toc
    # code, the 4 bits, b_bitrate_info = 1, bitrate_indicator 010,
    # b_audio_ndot = 1; fs_index 0, frame_rate_factor 1, no substream index
    s = code + "1011" + "1" + "010" + "1"
    st = {"channel_modes": [], "fs_index": 0, "ndot_bits": []}
    b = _bits(s)
    toc.substream_info_chan(b, False, 1, st)
    assert st["channel_modes"] == [ch_mode]
    assert b.p == len(s)
    assert st["ndot_bits"] == [len(s) - 1]


# ---------------------------------------------------------------- framing
def _framing(bits):
    from ac4 import channels as CH
    return CH.Framing(_bits(bits))


def test_pair_without_stereo_processing_uses_each_channels_framing(
        monkeypatch):
    """b_enable_mdct_stereo_proc = 0: L and R carry separate sf_info, and
    R must be dequantised and ungrouped with its own (here: L one long
    block, R four 384-sample blocks), not with L's."""
    from ac4 import pair as PAIR
    from ac4 import robust as RB
    from ac4 import stream as ST
    fr_l = _framing("1" + "000101")                  # long, max_sfb 5
    # short: transf_length 2, 2 (384), max_sfb 5, no grouping (3 bits)
    fr_r = _framing("0" + "10" + "10" + "000101" + "000")
    assert fr_r.num_windows == 4 and fr_r.num_groups == 4

    def ch(fr):
        return dict(lines=np.zeros(1536, dtype=np.int32), sfs=[None] * 5,
                    sfs_all=[[None] * 5] * fr.num_groups, framing=fr)

    def fake_pair(sub, T, b_iframe=False):
        return dict(L=ch(fr_l), R=ch(fr_r),
                    st_lr=dict(sap_mode=None, ms_used=None, sap=None,
                               max_sfb=5, framing=fr_l),
                    aspx=None, codec_mode=0, bits=8, nbits=64,
                    audio_size=1, audio_start=0, mode=("pair", 0))

    monkeypatch.setattr(PAIR, "decode_pair", fake_pair)
    monkeypatch.setattr(ST.M, "parse", lambda f: dict(
        sequence_counter=1, toc_bytes=0, payload_base=0,
        substream_sizes=[0, 8], b_iframe_global=1))
    monkeypatch.setattr(RB, "CHECKS", False)
    s = ST.Ac4Stream("pair")
    wins, lengths, _g, counts = s.decode_frames([bytes(16)])
    assert s.status == [None]
    assert lengths["L"] == [1536] and counts["L"] == [1]
    assert lengths["R"] == [384] * 4 and counts["R"] == [4]
    assert [len(w) for w in wins["R"]] == [384] * 4


# ---------------------------------------------------------- noise fill
# GetRandomNoiseValue (TS 103 190-1 Pseudocode 57) after
# ResetRandGenStateSnf (Pseudocode 24): the state and the first values for
# a few sequence counters (float32 results, exact). c/tests/unit_test.c
# checks the same numbers. sequence_counter 0 by hand: state (0, 0, 1, 0);
# T[0] + T[1] = 0.643120 + 0.970001; then uiOffsetA 1, uiStateIdx
# 2 + 0 + 1 = 3, uiCurrentIdx 0 + 1 = 1: T[1] + T[3] = 0.970001 - 0.643120.
SNF_FIRST = {
    0: ((0, 0, 1, 0), [1.6131210327148438, 0.3268809914588928,
                       -0.3108939826488495, 1.3197810649871826,
                       1.1259419918060303, -0.547692060470581]),
    1: ((0, 1, 130, 130), [-0.04938599839806557, 1.502228021621704,
                           -0.13683800399303436, -0.9735279679298401,
                           1.7364200353622437, -0.6638969779014587]),
    255: ((0, 255, 255, 126), [-0.05372801423072815, 0.9366300106048584,
                               -1.3645939826965332, -1.0417840480804443,
                               -0.7589679956436157, -1.3996400833129883]),
    1020: ((0, 252, 115, 248), [2.1083831787109375, -0.534654974937439,
                                -0.47258099913597107, 0.04947199672460556,
                                -0.47846800088882446, 1.1031620502471924]),
}


def test_snf_generator_state_and_statistics():
    from ac4 import audio as R
    for seq, (state, first) in SNF_FIRST.items():
        r = R.SnfRandom(seq)
        assert (r.a, r.b, r.state, r.cur) == state, seq
        assert [r.next() for _ in range(len(first))] == first, seq
    # 65280 draws (a full uiOffsetA/uiOffsetB cycle) are zero-mean,
    # unit-variance noise ("normal distributed with unit variance")
    for seq in (0, 1, 77, 255, 1020):
        r = R.SnfRandom(seq)
        x = np.array([r.next() for _ in range(65280)])
        assert abs(x.mean()) < 0.02 and abs(x.var() - 1.0) < 0.05, seq
        assert np.all(x == x.astype(np.float32))


def _snf_channel(fr, lines, sfs, snf):
    return dict(lines=np.asarray(lines, dtype=np.int32), sfs=sfs[0],
                sfs_all=sfs, snf=snf, framing=fr)


def test_noise_fill_levels_escape_and_order():
    """Pseudocodes 22/23 on a long block: band 0 coded, band 1 noise at
    +2 (log2 mean square), band 2 escaped (-17: zero, level held), band 3
    noise at -1 against band 1's level; band 4 coded again."""
    from ac4 import audio as R
    fr = _framing("1" + "000101")                    # long, max_sfb 5
    off = fr.sect_sfb_offset(0)
    lines = np.zeros(off[5], dtype=np.int32)
    lines[off[0]:off[1]] = [1, -1, 2, 0][:off[1] - off[0]]
    lines[off[4]] = 3
    sfs = [[100, None, None, None, 104]]
    snf = [[None, 2, -17, -1, None]]
    ch = _snf_channel(fr, lines, sfs, snf)
    plain = R.packed_spectrum(ch, fr)
    spec = R.packed_spectrum(ch, fr, R.SnfRandom(5))
    b = [slice(off[i], off[i + 1]) for i in range(5)]
    assert np.array_equal(spec[b[0]], plain[b[0]])
    assert np.array_equal(spec[b[4]], plain[b[4]])
    assert not plain[b[1]].any() and not spec[b[2]].any()
    lev0 = 1.44269504 * np.log(np.mean(plain[b[0]] ** 2))
    rnd = R.SnfRandom(5)
    want1 = np.array([rnd.next() for _ in range(off[2] - off[1])]) \
        * 2.0 ** (0.5 * (lev0 + 2))
    want3 = np.array([rnd.next() for _ in range(off[4] - off[3])]) \
        * 2.0 ** (0.5 * (lev0 + 2 - 1))
    assert np.allclose(spec[b[1]], want1, rtol=1e-12, atol=0)
    assert np.allclose(spec[b[3]], want3, rtol=1e-12, atol=0)
    # no noise data: unchanged
    ch0 = _snf_channel(fr, lines, sfs, None)
    assert np.array_equal(R.packed_spectrum(ch0, fr, R.SnfRandom(5)), plain)


def test_noise_fill_is_applied_by_the_stream(monkeypatch):
    """Ac4Stream.decode_frames applies the noise fill of a frame that
    carries asf_snf_data (before the fix the rng was ignored and the
    bands stayed silent), with one generator per frame reset from
    sequence_counter and run L then R (bitstream order)."""
    from ac4 import audio as R
    from ac4 import pair as PAIR
    from ac4 import robust as RB
    from ac4 import stream as ST
    fr = _framing("1" + "000101")
    off = fr.sect_sfb_offset(0)
    lines = np.zeros(off[5], dtype=np.int32)
    lines[off[0]] = 4

    def ch():
        return _snf_channel(fr, lines, [[100, None, None, None, None]],
                            [[None, 0, 0, 0, 0]])

    def fake_pair(sub, T, b_iframe=False):
        return dict(L=ch(), R=ch(),
                    st_lr=dict(sap_mode=None, ms_used=None, sap=None,
                               max_sfb=5, framing=fr),
                    aspx=None, codec_mode=0, bits=8, nbits=64,
                    audio_size=1, audio_start=0, mode=("pair", 0))

    monkeypatch.setattr(PAIR, "decode_pair", fake_pair)
    monkeypatch.setattr(ST.M, "parse", lambda f: dict(
        sequence_counter=9, toc_bytes=0, payload_base=0,
        substream_sizes=[0, 8], b_iframe_global=1))
    monkeypatch.setattr(RB, "CHECKS", False)
    s = ST.Ac4Stream("pair")
    wins, _l, _g, _c = s.decode_frames([bytes(16)])
    L, Rr = wins["L"][0], wins["R"][0]
    n = off[5] - off[1]
    assert np.sum(L[off[1]:off[5]] ** 2) > 0
    assert np.sum(Rr[off[1]:off[5]] ** 2) > 0
    rnd = R.SnfRandom(9)
    seq = np.array([rnd.next() for _ in range(2 * n)])
    amp = 2.0 ** (0.5 * 1.44269504 * np.log(
        np.mean(L[off[0]:off[1]] ** 2)))
    assert np.allclose(L[off[1]:off[5]], seq[:n] * amp, rtol=1e-12)
    assert np.allclose(Rr[off[1]:off[5]], seq[n:] * amp, rtol=1e-12)


# ---------------------------------------------------------------- A-SPX
def _aspx_cfg():
    from ac4 import aspx_render as R37
    cfg = dict(start_freq=2, stop_freq=1, master_freq_scale=1, noise_sbg=1,
               quant_mode_env=0, interpolation=0, preflat=0, limiter=1,
               num_env_bits_fixfix=0, freq_res_mode=0)
    t = R37.t_for(cfg)
    assert t["n_hi"] >= 1 and t["num_sb_aspx"] >= 1
    return cfg, t


def test_aspx_frames_without_data_are_core_only():
    """Frames with no A-SPX data (before the stream's first aspx_config)
    come out of the QMF bank as their core, 577 samples late, with nothing
    patched above the crossover. Before the fix every slot of the call was
    HF-generated, so those frames played the copied low band unadjusted."""
    from ac4 import aspx_render as R37
    cfg, _t = _aspx_cfg()
    rng = np.random.default_rng(1)
    n = 8
    x = {k: rng.standard_normal(n * 1536) for k in ("L", "R")}
    out, applied = R37.apply_hf_pair(x["L"], x["R"], [None] * n, cfg)
    assert applied == 0
    for k in ("L", "R"):
        y = out[k]
        ref = np.concatenate([np.zeros(577), x[k]])[:len(y)]
        err = y[1536:] - ref[1536:]
        snr = 10 * np.log10(np.sum(ref[1536:] ** 2) / np.sum(err ** 2))
        assert snr > 60, snr


# ---------------------------------------------------------- alignment
def _xcorr_lag(a, b, maxlag=1200):
    """Lag of b against a, in samples (positive: b is later)."""
    n = len(a)
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, 2 * n)) * np.fft.rfft(b, 2 * n))
    cc = np.concatenate([cc[-maxlag:], cc[:maxlag + 1]])
    return int(np.argmax(np.abs(cc))) - maxlag


@pytest.mark.parametrize("with_cfg", [False, True])
def test_all_channels_time_aligned(monkeypatch, with_cfg):
    """The same core in all six channels of a 5.1 element: L/R and Ls/Rs
    go through the QMF bank (577 samples), so C and LFE must come out
    equally late (TS 103 190-1 6.2.8 / 6.2.14). Before the fix C and LFE
    led the others by 577 samples (12 ms) whenever the pairs were on the
    A-SPX path (here: an aspx_config, and every frame's A-SPX concealed,
    which mutes only the band above the crossover, where the test signal
    has nothing), and before any aspx_config nothing was on the QMF path.
    Two passes, so the delay line is carried over a pass boundary too."""
    from ac4 import decoder as D
    from ac4 import robust as RB
    cfg = _aspx_cfg()[0] if with_cfg else None
    grp = RB.CONCEALED if with_cfg else None
    rng = np.random.default_rng(3)
    n = 10
    specs = [rng.standard_normal(1536) * (np.arange(1536) < 400)
             for _ in range(2 * n)]
    dec = D.Decoder("5_X", guard=False)
    dec.hf_ratio = None
    state = {"i": 0}

    def fake_decode_frames(frames, chans):
        i0 = state["i"]
        state["i"] += len(frames)
        w = {k: [specs[i0 + j] for j in range(len(frames))] for k in chans}
        ln = {k: [1536] * len(frames) for k in chans}
        ct = {k: [1] * len(frames) for k in chans}
        dec.stream.mode = ("5_X", 1)
        dec.stream.cfg = cfg
        dec.stream.status = [None] * len(frames)
        return w, ln, {"lr": [grp] * len(frames),
                       "sr": [grp] * len(frames)}, ct

    monkeypatch.setattr(dec.stream, "decode_frames", fake_decode_frames)
    y = np.vstack([dec.decode_pass([b""] * n), dec.decode_pass([b""] * n)])
    L, R, C, LFE, Ls, Rs = y.T
    for other in (R, LFE, Ls, Rs):
        assert _xcorr_lag(C, other) == 0
    assert _xcorr_lag(C, L) == 0
    s = slice(2 * 1536, None)
    for k in (L, R, Ls, Rs):
        err = k[s] - C[s]
        assert 10 * np.log10(np.sum(C[s] ** 2) / np.sum(err ** 2)) > 60


def test_aspx_noise_index_runs_on_across_envelopes(monkeypatch):
    """Pseudocode 103: noise_idx = noise_idx_prev + num_sb_aspx * (ts -
    atsg_sig[0]) + sb + 1, ts counted from the frame's FIRST envelope. A
    frame with two envelopes (A-SPX slots 0-4 and 4-8): with the signal
    gain 0 and noise level 1, the high band is the noise table itself, and
    its index must run on from the first envelope into the second, and on
    into the next frame. Before the fix it restarted at every envelope."""
    from ac4 import aspx_render as R37
    from ac4 import aspx_env as V
    from ac4 import aspx_envadj as E
    from ac4 import qmf as Q
    cfg, t = _aspx_cfg()
    n_sb, sbx = t["num_sb_aspx"], t["sbx"]
    got = [{"fr": {"borders": [0, 4, 8]}, "balance": 1},
           {"fr": {"borders": [0, 4, 8]}, "balance": 1}]
    zeros = lambda *a, **k: np.zeros((n_sb, 2))             # noqa: E731
    monkeypatch.setattr(V, "reconstruct", lambda d, t, m, p: (None, p))
    monkeypatch.setattr(E, "noise_qscf", lambda d, t, p: (None, p))
    monkeypatch.setattr(E, "map_scf_joint",
                        lambda *a: (np.zeros((n_sb, 2)), np.zeros((n_sb, 2))))
    monkeypatch.setattr(E, "map_noise_joint",
                        lambda *a: (np.zeros((n_sb, 2)), np.zeros((n_sb, 2))))
    monkeypatch.setattr(E, "estimate", zeros)
    monkeypatch.setattr(E, "adjust", lambda *a: (np.zeros((n_sb, 2)),
                                                 np.ones((n_sb, 2))))
    seen = []
    real = Q.synthesise
    monkeypatch.setattr(Q, "synthesise",
                        lambda Qm, *a: (seen.append(Qm.copy()),
                                        real(Qm, *a))[1])
    x = np.zeros(4 * 1536)
    R37.apply_hf_pair(x, x, [None, got, got, None], cfg)
    nt = E.noise_table()
    Qh = seen[0]
    # frame 1: slots 24 + 4 .. 24 + 4 + 16; frame 2 continues
    for f in (1, 2):
        t0 = 24 * f + R37.TS_OFFSET
        for ts in range(t0, t0 + 16):
            idx = (n_sb * 16 * (f - 1) + n_sb * (ts - t0)
                   + np.arange(n_sb) + 1) % 512
            assert np.allclose(Qh[sbx:sbx + n_sb, ts], nt[idx]), (f, ts)
