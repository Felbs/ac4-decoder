# SPDX-License-Identifier: Apache-2.0
"""Phase B3 item 3 (no captures): bitstream_version 0 and 1, whose table
of contents is TS 103 190-1's ac4_presentation_info() (Tables 4, 5, 9;
TS 103 190-2 6.2.1.2), written bit by bit. Unverified on real streams:
every stream at hand is bitstream_version 2."""
import numpy as np
import pytest

from ac4 import mix as MX
from ac4 import toc as M
from ac4.decoder import Decoder, decode, probe_element
from test_phase_a import BW, build_toc, emdf_info
from test_mix import mono_sub, pair_sub


def substream_info(w, ch_mode_code, index, iframe=1, cc=None, lang=None):
    """ac4_substream_info (TS 103 190-1 Table 9), 48 kHz."""
    w.u(len(ch_mode_code), int(ch_mode_code, 2))
    w.u(1, 0)                                # b_sf_multiplier
    w.u(1, 0)                                # b_bitrate_info
    if cc is None:
        w.u(1, 0)                            # b_content_type
    else:
        w.u(1, 1).u(3, cc)
        if lang is None:
            w.u(1, 0)
        else:
            w.u(1, 1).u(1, 0).u(6, len(lang))
            for c in lang.encode():
                w.u(8, c)
    w.u(1, iframe)                           # b_iframe (frame_rate_factor 1)
    w.u(2, index) if index < 3 else w.u(2, 3).vb(2, index - 3)


def toc_v0(subs, sizes, config=None, bv=0, seq=5, version=0, iframe=1,
           size_present=True):
    """ac4_toc with one ac4_presentation_info: subs = [(code, cc), ...]
    in presentation_config order."""
    w = BW().u(2, bv).u(10, seq).u(1, 0).u(1, 1).u(4, 3).u(1, iframe)
    w.u(1, 1)                                # b_single_presentation
    w.u(1, 0)                                # b_payload_base
    if config is None:
        w.u(1, 1)                            # b_single_substream
    else:
        w.u(1, 0).u(3, config)
    w.u(version, (1 << version) - 1).u(1, 0)  # presentation_version
    w.u(3, 0)                                # md_compat
    w.u(1, 0)                                # b_presentation_id
    w.u(1, 0)                                # frame_rate_multiply_info
    emdf_info(w)
    if config is not None:
        w.u(1, 0)                            # b_hsf_ext
    for i, (code, cc) in enumerate(subs):
        substream_info(w, code, i, iframe, cc)
    w.u(1, 0).u(1, 0)                        # b_pre_virtualized, add emdf
    n = len(sizes)
    w.u(2, n if n < 4 else 0)
    if n == 1:
        w.u(1, int(size_present))
    if n > 1 or size_present:
        for s in sizes:
            w.u(1, 0).u(10, s)
    return w.bytes()


def _stereo_frames(n=6, bv=0, size_present=True, seed=3):
    rng = np.random.default_rng(seed)
    v0, v2 = [], []
    for i in range(n):
        a = pair_sub(rng)
        v0.append(toc_v0([("10", 0)], [len(a)], bv=bv, seq=5 + i,
                         size_present=size_present) + a)
        ps = bytes(2)
        v2.append(build_toc(pres=[dict(groups=[0], pres_sub=0)],
                            groups=[dict(ch_mode_code="10", index=1, cc=0)],
                            sizes=[len(ps), len(a)], seq=5 + i) + ps + a)
    return v0, v2


def test_v0_toc_parses_into_presentations_and_groups():
    v0, _ = _stereo_frames(1)
    st = M.parse(v0[0])
    assert st["bitstream_version"] == 0
    p = st["presentations"][0]
    assert p["config"] is None and p["version"] == 0
    assert p["pres_substream"] is None and p["groups"] == [0]
    s = M.audio_substream(st, 0)
    assert s["channel_mode"] == 1 and s["index"] == 0 and s["ndot"] == [1]
    assert st["groups"][0]["content_classifier"] == 0
    assert st["substream_sizes"] == [len(v0[0]) - st["toc_bytes"]]


@pytest.mark.parametrize("bv,size_present", [(0, True), (1, True),
                                             (0, False)])
def test_v0_decodes_like_the_same_audio_under_a_v2_toc(bv, size_present):
    v0, v2 = _stereo_frames(bv=bv, size_present=size_present)
    assert probe_element(v0) == "pair"
    a = Decoder("pair", guard=False)
    b = Decoder("pair", guard=False)
    ya, yb = a.decode_pass(v0), b.decode_pass(v2)
    assert a.n_concealed == 0 and b.n_concealed == 0
    assert np.abs(ya).max() > 0 and np.array_equal(ya, yb)


def test_v0_me_plus_dialogue_mixes_without_dividing():
    rng = np.random.default_rng(9)
    frames = []
    for i in range(5):
        a, d = pair_sub(rng), mono_sub(rng)
        frames.append(toc_v0([("10", None), ("0", None)], [len(a), len(d)],
                             config=0, seq=5 + i) + a + d)
    subs = MX.probe(frames)
    assert [(x["type"], x["element"]) for x in subs] == \
        [("main", "pair"), ("dialogue", "mono")]
    mx = MX.MixDecoder(subs, guard=False)
    y = mx.decode_pass(frames)
    ya = Decoder("pair", guard=False).decode_pass(frames)
    yb = Decoder("mono", guard=False, substream=1).decode_pass(frames)
    # presentation_version 0: TS 103 190-1 6.2.16.1, a plain sum
    exp = ya + yb[:, [0, 0]] * 0.5
    assert np.abs(y - exp).max() <= 1e-12 * np.abs(exp).max()


def test_v0_dialnorm_and_truepk_come_from_basic_metadata():
    from ac4 import metadata as MD
    from ac4.stream import Ac4Stream
    w = BW().u(7, 92).u(1, 1).u(1, 1)        # dialnorm -23, more, loudness
    w.u(2, 0).u(4, 0).u(1, 0).u(1, 0).u(1, 0).u(1, 0)
    w.u(1, 1).u(11, 1024 - 31)               # truepk -3.1
    meta = w.align().bytes() + bytes(2)
    rng = np.random.default_rng(4)
    frames = []
    for i in range(3):
        a = pair_sub(rng) + meta
        frames.append(toc_v0([("10", 0)], [len(a)], seq=5 + i) + a)
    s = Ac4Stream("pair")
    s.decode_frames(frames)
    assert s.n_bad == 0
    assert [m["dialnorm"] for m in s.meta] == [-23.0] * 3
    assert [m["truepk"] for m in s.meta] == [-3.1] * 3


@pytest.mark.parametrize("bv,size_present", [(0, True), (1, False)])
def test_c_v0_equals_python(tmp_path, bv, size_present):
    from test_mix import _c_mix
    v0, _ = _stereo_frames(bv=bv, size_present=size_present)
    ref = Decoder("pair", guard=False).decode_pass(v0)
    got = _c_mix(tmp_path, v0).reshape(-1, 2)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()


def test_c_v0_mix_equals_python(tmp_path):
    from test_mix import _c_mix
    rng = np.random.default_rng(9)
    frames = []
    for i in range(5):
        a, d = pair_sub(rng), mono_sub(rng)
        frames.append(toc_v0([("10", None), ("0", None)], [len(a), len(d)],
                             config=0, seq=5 + i) + a + d)
    ref = MX.MixDecoder(MX.probe(frames), guard=False).decode_pass(frames)
    got = _c_mix(tmp_path, frames).reshape(-1, 2)
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()
