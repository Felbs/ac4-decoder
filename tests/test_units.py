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
