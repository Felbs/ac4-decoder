# SPDX-License-Identifier: Apache-2.0
"""Phase B3 item 2 (no captures): presentations with several substream
groups -- music and effects + dialogue, main + associated audio --
written bit by bit, decoded substream by substream and mixed (TS 103
190-2 4.8.3.17, 4.8.3.18, 4.8.4; TS 103 190-1 6.2.16). Unverified on
real streams: no stream at hand carries more than one audio substream
per presentation."""
import math

import numpy as np
import pytest

from ac4 import channels as CH
from ac4 import metadata as MD
from ac4 import mix as MX
from ac4 import toc as M
from ac4.decoder import Decoder, decode
import elem_writer as EW
from test_phase_a import BW, build_toc


def pair_sub(rng):
    """channel_pair_element, SIMPLE, stereo processing on."""
    T = CH.load_tables()
    w = BW().u(2, 0)
    n = int(rng.integers(3, 12))
    EW.two_channel(w, T, [EW.Track(rng, n), EW.Track(rng, n)],
                   EW.St(rng, 1, n))
    return EW.element(w)


def mono_sub(rng):
    T = CH.load_tables()
    w = BW().u(1, 0)
    EW.mono(w, T, EW.Track(rng, int(rng.integers(3, 12))))
    return EW.element(w)


def pres_sub(sg=None, keep=False, assoc=None, n_groups=2):
    """ac4_presentation_substream (TS 103 190-2 6.2.2.3): dialnorm -24,
    an empty drc_frame, the substream group gains and the associated
    audio mixing metadata (raw field values)."""
    w = BW().u(1, 0).u(7, 96).u(1, 0)        # no add. data, dialnorm, loud.
    w.u(5, 1).u(1, 0).u(1, 0)                # drc_metadata_size 1, no drc
    if n_groups > 1:
        if sg is None and not keep:
            w.u(1, 0)
        else:
            w.u(1, 1).u(1, int(keep))
            if not keep:
                for v in sg:
                    w.u(6, v)
    if assoc is None:
        w.u(1, 0)
    else:
        w.u(1, 1)
        for key in ("scale_main", "scale_main_centre", "scale_main_front",
                    "pan_associated"):
            if assoc.get(key) is None:
                w.u(1, 0)
            else:
                w.u(1, 1).u(8, assoc[key])
    w.u(16, 0)                               # custom_dmx_data, ... (unread)
    return w.align().bytes()


def two_group_frames(n=6, config=5, cc=(0, 4), sg=(4, 12), assoc=None,
                     second="mono", seed=7, keep_from=None):
    rng = np.random.default_rng(seed)
    codes = {"mono": "0", "pair": "10"}
    frames = []
    for i in range(n):
        a = pair_sub(rng)
        b = mono_sub(rng) if second == "mono" else pair_sub(rng)
        ps = pres_sub(sg=sg, keep=keep_from is not None and i >= keep_from,
                      assoc=assoc)
        toc = build_toc(
            pres=[dict(groups=[0, 1], pres_sub=0)],
            groups=[dict(ch_mode_code="10", index=1, cc=cc[0]),
                    dict(ch_mode_code=codes[second], index=2, cc=cc[1])],
            sizes=[len(ps), len(a), len(b)], seq=5 + i)
        frames.append(toc + ps + a + b)
    return frames




# ------------------------------------------------------------ metadata
def test_presentation_substream_mixing_metadata():
    b = pres_sub(sg=[0, 63], assoc=dict(scale_main=10, pan_associated=20))
    md = MD.presentation_substream(b, 0, 0, 3, MD.DrcState(), 1, 1536,
                                   n_groups=2)
    assert md["dialnorm"] == -24.0
    assert md["sg_gain"] == [0.0, -math.inf]
    a = md["assoc"]
    assert a["scale_main"] == pytest.approx(-3.0)
    assert a["scale_main_centre"] is None and a["scale_main_front"] is None
    assert a["pan_associated"] == 30.0
    md = MD.presentation_substream(pres_sub(keep=True), 0, 0, 3,
                                   MD.DrcState(), 1, 1536, n_groups=2)
    assert md["sg_gain"] == "keep"
    # one group: no sg_gain fields are read
    md = MD.presentation_substream(pres_sub(n_groups=1), 0, 0, 2,
                                   MD.DrcState(), 1, 1536, n_groups=1)
    assert "sg_gain" not in md and "assoc" not in md


def test_tables_70_and_scale_and_pan():
    assert MD.sg_gain_db(0) == 0.0 and MD.sg_gain_db(62) == -15.5
    assert MD.sg_gain_db(63) == -math.inf
    assert MD.scale_db(0xFE) == pytest.approx(-76.2)
    assert MD.scale_db(0xFF) == -math.inf
    assert MD.pan_deg(0x14) == 30.0 and MD.pan_deg(0xEF) == 358.5


def test_pan_gains_meet_table_216():
    six = ("L", "R", "C", "lfe", "Ls", "Rs")
    assert MX.pan_gains(330, six) == pytest.approx({"L": 1.0, "Ls": 0.0})
    g = MX.pan_gains(0, six)
    assert g["C"] == pytest.approx(1.0) and sum(g.values()) == 1.0
    assert MX.pan_gains(30, six)["R"] == pytest.approx(1.0)
    assert MX.pan_gains(0, ("L", "R")) == {"L": 0.5, "R": 0.5}
    g = MX.pan_gains(15, six)
    assert g == pytest.approx({"C": 0.5, "R": 0.5})


def test_substream_types():
    f = two_group_frames(n=1, cc=(1, 4))[0]
    st = M.parse(f)
    subs = M.audio_substreams(st, 0)
    assert [s["index"] for _, _, s in subs] == [1, 2]
    assert [M.substream_type(st, 0, gi) for gi, _, _ in subs] == \
        ["main", "dialogue"]
    f = two_group_frames(n=1, cc=(0, 2))[0]
    st = M.parse(f)
    assert M.substream_type(st, 0, 1) == "associated"
    # Table 53: config 1 (main + DE) leaves the DE group out
    st["presentations"][0]["config"] = 1
    assert len(M.audio_substreams(st, 0)) == 1


# ------------------------------------------------------------ mixing
def _parts(frames):
    a = Decoder("pair", guard=False, substream=0)
    b = Decoder("mono", guard=False, substream=1)
    ya, yb = a.decode_pass(frames), b.decode_pass(frames)
    assert a.n_concealed == 0 and b.n_concealed == 0
    assert np.abs(ya).max() > 0 and np.abs(yb).max() > 0
    return ya, yb


def test_me_plus_dialogue_mix():
    frames = two_group_frames(cc=(1, 4), sg=(4, 12))
    subs = MX.probe(frames)
    assert [(d["type"], d["element"], d["sg"]) for d in subs] == \
        [("main", "pair", 0), ("dialogue", "mono", 1)]
    mx = MX.MixDecoder(subs, guard=False)
    y = mx.decode_pass(frames)
    ya, yb = _parts(frames)
    g0, g1 = 10 ** (-1.0 / 20), 10 ** (-3.0 / 20)
    # mono dialogue on a stereo output: L and R at 0.5 (Table 216)
    exp = (ya * g0 + yb[:, [0, 0]] * 0.5 * g1) / 2
    assert mx.chans == ("L", "R") and mx.n_concealed == 0
    assert np.abs(y - exp).max() <= 1e-12 * np.abs(exp).max()


def test_main_plus_associated_mix_with_scale_and_pan():
    assoc = dict(scale_main=10, scale_main_front=5, pan_associated=0x14)
    frames = two_group_frames(cc=(0, 2), sg=(0, 0), assoc=assoc)
    mx = MX.MixDecoder(MX.probe(frames), guard=False, assoc_gain=-6.0)
    y = mx.decode_pass(frames)
    ya, yb = _parts(frames)
    gm = 10 ** (-3.0 / 20) * 10 ** (-1.5 / 20)   # scale_main, _front
    ga = 10 ** (-6.0 / 20)
    exp = ya * gm
    exp[:, 1] += yb[:, 0] * ga                    # panned to 30 deg: R
    exp /= 2
    assert np.abs(y - exp).max() <= 1e-12 * np.abs(exp).max()


def test_sg_gain_keep_and_mute():
    frames = two_group_frames(sg=(0, 63), keep_from=3)
    mx = MX.MixDecoder(MX.probe(frames), guard=False)
    y = mx.decode_pass(frames)
    ya, _yb = _parts(frames)
    # the dialogue group muted (63 = -inf), and kept so after b_keep
    assert np.abs(y - ya / 2).max() <= 1e-12 * np.abs(ya).max()


def test_decode_mixes_by_default_and_not_with_mix_false():
    frames = two_group_frames(sg=(0, 0))
    p1, d1 = decode(frames)
    p0, d0 = decode(frames, mix=False)
    assert isinstance(d1, MX.MixDecoder) and isinstance(d0, Decoder)
    assert p1.shape == p0.shape
    assert not np.allclose(p1, p0)


def test_mix_conceals_when_a_substream_is_damaged():
    frames = two_group_frames(sg=(0, 0), n=8)
    f = bytearray(frames[4])
    f[-3] ^= 0xFF                              # the mono substream
    frames[4] = bytes(f)
    mx = MX.MixDecoder(MX.probe(frames), guard=True)
    y = mx.decode_pass(frames)
    assert y.shape == (8 * 1536, 2)
    assert mx.last_concealed[4] or not mx.decs[1].last_concealed[4]


# ------------------------------------------------------------ the C decoder
def _c_mix(tmp_path, frames, extra=()):
    import os
    import subprocess
    from test_inputs import syncframe
    from test_phase_b2 import C_CLI
    if not os.path.exists(C_CLI):
        pytest.skip("c/ac4dec-c not built")
    src = tmp_path / "m.ac4"
    src.write_bytes(b"".join(syncframe(f) for f in frames))
    raw = tmp_path / "m.f64"
    subprocess.run([C_CLI, str(src), str(tmp_path / "m.wav"),
                    "--legacy-level", "--f64", str(raw), "--internal",
                    "--no-guard", "-q", *extra], check=True)
    return np.fromfile(raw, "<f8")


@pytest.mark.parametrize("kw,assoc_gain", [
    (dict(cc=(1, 4), sg=(4, 12)), 0.0),
    (dict(cc=(0, 2), sg=(0, 3), assoc=dict(scale_main=10,
                                            scale_main_front=5,
                                            pan_associated=0x14)), -6.0),
    (dict(cc=(0, 2), sg=(2, 0), assoc=dict(scale_main_centre=7)), 0.0),
    (dict(sg=(0, 63), keep_from=3), 0.0),
    (dict(cc=(0, 4), sg=(1, 2), second="pair"), 0.0)])
def test_c_mix_equals_python(tmp_path, kw, assoc_gain):
    frames = two_group_frames(**kw)
    mx = MX.MixDecoder(MX.probe(frames), guard=False, assoc_gain=assoc_gain)
    ref = mx.decode_pass(frames)
    got = _c_mix(tmp_path, frames, ("--assoc-gain", str(assoc_gain)))
    got = got.reshape(-1, ref.shape[1])
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()
    # --no-mix: the main substream alone
    one = Decoder("pair", guard=False).decode_pass(frames)
    got1 = _c_mix(tmp_path, frames, ("--no-mix",)).reshape(-1, 2)
    assert np.abs(got1 - one).max() <= 1e-9 * np.abs(one).max()
