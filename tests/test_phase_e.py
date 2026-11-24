# SPDX-License-Identifier: Apache-2.0
"""Phase E: object substreams without A-JOC (audio_data_objs, the
oamd_substream, ISF; TS 103 190-2 6.2.1.11, 6.2.2.4, 6.2.3.2, 5.10.3),
written bit by bit from the syntax. Synthetic streams only."""
import math

import numpy as np
import pytest

from ac4 import channels as CH
from ac4 import elements as EL
from ac4 import objrender as OR
from ac4 import stream as SC
from ac4 import toc as M
import elem_writer as EW
from test_phase_a import BW, presentation

C_CLI = __import__("os").path.join(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__)), "..", "c", "ac4dec-c")


# ---- writers -------------------------------------------------------------

def _info_obj(w, s, iframe):
    """ac4_substream_info_obj (6.2.1.11) for one substream description s:
    dict(n=coded objects, kind "dyn" / "bed" / "isf", lfe (dyn), start
    (bed / ISF: the assignment bits written, None: a continuation),
    index)."""
    w.u(3, {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}[s["n"]])
    if s["kind"] == "dyn":
        w.u(1, 1).u(1, s.get("lfe", 0))
    elif s["kind"] == "bed":
        w.u(1, 0).u(1, 1)
        st = s.get("start")
        w.u(1, int(st is not None))
        if st is not None:
            if st[0] == "code":
                w.u(1, 1).u(3, st[1])
            elif st[0] == "std":
                w.u(1, 0).u(1, 0).u(10, st[1])
            else:
                w.u(1, 0).u(1, 1).u(17, st[1])
    else:
        w.u(1, 0).u(1, 0).u(1, 1)
        st = s.get("start")
        w.u(1, int(st is not None))
        if st is not None:
            w.u(3, st)
    w.u(1, 0).u(1, 0)                    # b_sf_multiplier, b_bitrate_info
    w.u(1, iframe)                       # b_audio_ndot
    _index(w, s["index"])


def _index(w, i):
    """substream_index (2 bits, 3 escapes to variable_bits(2))."""
    if i < 3:
        w.u(2, i)
    else:
        w.u(2, 3).vb(2, i - 3)


def objs_toc(subs, sizes, seq, iframe, oamd_index, alternative=0):
    """One presentation, one object-coded substream group with an
    oamd_substream and the direct-coded substreams subs."""
    w = BW().u(2, 2).u(10, seq).u(1, 0).u(1, 1).u(4, 3).u(1, iframe)
    w.u(1, 1).u(1, 0).u(1, 0)            # one presentation, no base, no id
    presentation(w, groups=[0], pres_sub=0)
    if alternative:                      # patch b_alternative: rewrite
        raise NotImplementedError
    n = len(subs)
    w.u(1, 1).u(1, 0)                    # b_substreams_present, no hsf
    if n == 1:
        w.u(1, 1)
    else:
        w.u(1, 0).u(2, n - 2)
    w.u(1, 0)                            # object coded
    w.u(1, 1)                            # b_oamd_substream
    w.u(1, iframe)                       # b_oamd_ndot
    _index(w, oamd_index)
    for s in subs:
        w.u(1, 0)                        # b_ajoc
        _info_obj(w, s, iframe)
    w.u(1, 0)                            # b_content_type
    w.u(2, len(sizes) if len(sizes) < 4 else 0)
    if len(sizes) >= 4:
        w.vb(2, len(sizes) - 4)
    for z in sizes:
        if z < 1024:
            w.u(1, 0).u(10, z)
        else:
            w.u(1, 1).u(10, z & 1023).vb(2, z >> 10)
    return w.bytes()


def obj_block(w, iframe, dyn, pos=None, gain=None, active=True):
    """object_info_block (6.2.8.5), all new."""
    if not active:
        w.u(1, 1).u(1, 0)                # not active, no table data
        return
    w.u(1, 0)
    if not iframe:
        w.u(1, 0)                        # basic info: new
    if gain is None:
        w.u(1, 1)                        # default basic info (0 dB)
    else:                                # object_gain_code 0b0
        w.u(1, 0).u(1, 0).u(1, 0)
        v = 15 - gain if gain >= 1 else 14 - gain
        w.u(6, v)
    if dyn:
        if not iframe:
            w.u(1, 0).u(1, 0)            # render: new
        x, y, z = pos
        if not iframe:
            w.u(1, 0)                    # b_diff_pos_coding
        w.u(6, x).u(6, y).u(1, int(z >= 0)).u(4, abs(z))
        w.u(1, 1).u(1, 1)                # zone, other: defaults
    w.u(1, 0)                            # b_add_table_data


def oamd_sub(objs, iframe, timing=True):
    """oamd_substream (6.2.2.4): no common data, one block at sample 0,
    oamd_dyndata_multi over objs = [(dyn, pos, gain)]."""
    w = BW().u(1, 0)                     # b_oamd_common_data_present
    w.u(1, int(timing))
    if timing:
        w.u(1, 0).u(3, 1).u(6, 0).u(2, 0)  # offset 0, one block, ramp 0
    for dyn, pos, gain in objs:
        obj_block(w, iframe, dyn, pos, gain)
    return w.align().bytes()


def objs_body(rng, n, lfe):
    """audio_data_objs (6.2.3.2) in SIMPLE codec mode: the LFE's
    mono_data, then the element of objs_to_channel_mode(n). -> (bytes,
    {slot: dequantised track})."""
    T = CH.load_tables()
    w = BW()
    exp = {}
    if lfe:
        t = EW.Track(rng, 2)
        EW.lfe(w, T, t)
        exp["lfe"] = t.deq()
    nb = 8
    if n == 1:
        w.u(1, 0)                        # mono_codec_mode SIMPLE
        t = EW.Track(rng, nb)
        EW.mono(w, T, t)
        exp["C"] = t.deq()
    elif n == 2:
        w.u(2, 0)                        # stereo_codec_mode SIMPLE
        tr = [EW.Track(rng, nb), EW.Track(rng, nb)]
        EW.two_channel(w, T, tr, EW.St(rng, 0, nb))
        exp.update(L=tr[0].deq(), R=tr[1].deq())
    elif n == 3:
        w.u(1, 0).u(1, 0)                # 3_0 SIMPLE, coding_config 0
        tr = [EW.Track(rng, nb), EW.Track(rng, nb)]
        EW.two_channel(w, T, tr, EW.St(rng, 0, nb))
        c = EW.Track(rng, nb)
        EW.mono(w, T, c)
        exp.update(L=tr[0].deq(), R=tr[1].deq(), C=c.deq())
    else:
        w.u(3, 0).u(2, 0).u(1, 0)        # 5_X SIMPLE, cc 0, 2ch_mode 0
        for keys in (("L", "R"), ("Ls", "Rs")):
            tr = [EW.Track(rng, nb), EW.Track(rng, nb)]
            EW.two_channel(w, T, tr, EW.St(rng, 0, nb))
            exp.update(zip(keys, (x.deq() for x in tr)))
        c = EW.Track(rng, nb)
        EW.mono(w, T, c)
        exp["C"] = c.deq()
    return EW.element(w), exp


def objs_frames(rng, subs, objs, n=4, iframe_every=None):
    """Frames of an object presentation: the presentation substream
    (index 0, empty), the oamd_substream (index 1), the object
    substreams (2, 3, ...). subs as _info_obj's (index set here); objs
    the group's objects [(dyn, pos, gain)] in OAMD order. -> (frames,
    per frame [{slot: track}] per substream)."""
    frames, exps = [], []
    for i in range(n):
        ifr = int(i == 0 or bool(iframe_every and i % iframe_every == 0))
        bodies, ex = [], []
        for k, s in enumerate(subs):
            s["index"] = 2 + k
            lfe = s.get("lfe", 0) if s["kind"] == "dyn" else s.get("blfe", 0)
            body, e = objs_body(rng, s["n"], lfe)
            bodies.append(body)
            ex.append(e)
        om = oamd_sub(objs, ifr)
        sizes = [2, len(om)] + [len(x) for x in bodies]
        toc = objs_toc(subs, sizes, 5 + i, ifr, 1)
        frames.append(toc + bytes(2) + om + b"".join(bodies))
        exps.append(ex)
    return frames, exps


# ---- tests ---------------------------------------------------------------

def test_obj_split():
    """The objects of each substream of a group (toc.obj_split): a 5.1
    bed shared by two substreams (L R C + LFE, then Ls Rs), then two
    dynamic objects."""
    rng = np.random.default_rng(1)
    subs = [dict(kind="bed", n=3, blfe=1, start=("code", 2)),
            dict(kind="bed", n=2, start=None),
            dict(kind="dyn", n=2, lfe=0)]
    objs = [(False, None, None)] * 6 + [(True, (0, 0, 0), None)] * 2
    frames, _ = objs_frames(rng, subs, objs, n=1)
    st = M.parse(frames[0])
    g = st["groups"][0]
    s0, s1, s2 = g["substreams"]
    assert s0["types"] == ["L", "R", "C", "lfe"] and s0["lfe"] == 1
    assert s1["types"] == ["Ls", "Rs"] and s1["lfe"] == 0
    assert s2["types"] == ["DYN", "DYN"]
    assert [s["oamd_offset"] for s in (s0, s1, s2)] == [0, 4, 6]
    assert [t for t, _l, _a in g["obj_types"]] == [
        "L", "R", "C", "lfe", "Ls", "Rs", "DYN", "DYN"]
    assert [len(M.audio_substreams(st, 0))] == [3]


@pytest.mark.parametrize("n,lfe", [(1, 0), (2, 1), (3, 0), (5, 1)])
def test_objs_tracks(n, lfe):
    """audio_data_objs walked to audio_size: the LFE and the element's
    tracks on their slots; the objects' OAMD from the oamd_substream."""
    rng = np.random.default_rng(10 + n)
    subs = [dict(kind="dyn", n=n, lfe=lfe)]
    pos = [(10 * k, 62 - 10 * k, k) for k in range(n)]
    objs = ([(False, None, None)] if lfe else []) + \
        [(True, p, None) for p in pos]
    frames, exps = objs_frames(rng, subs, objs)
    info = M.audio_substream(M.parse(frames[0]), 0)
    assert info["kind"] == "obj" and EL.ajoc_slots(info) == (
        ("lfe",) if lfe else ()) + EL.OBJS_ELEMENT[n][1]
    s = SC.Ac4Stream("ajoc", ajoc_info=info)
    wins, lengths, groups, counts = s.decode_frames(frames)
    assert s.n_bad == 0, s.status
    for i, ex in enumerate(exps):
        for k, ref in ex[0].items():
            assert np.allclose(np.asarray(wins[k][i]), ref), (i, k)
        o = groups["obj"][i]
        assert len(o["dmx"]) == n + lfe
        for k, p in enumerate(pos):
            x, y, z = p
            got = o["dmx"][k + lfe]
            assert (got.px, got.py, got.pz) == (x, y, z)


def test_objs_render_and_mix():
    """A bed (L R C LFE, then Ls Rs) and two dynamic objects in three
    substreams, mixed (4.8.4: the sum over 3): each bed channel on its
    loudspeaker, a dynamic object at the front left corner on L, one at
    the front right corner on R at -6 dB."""
    from ac4.decoder import decode
    rng = np.random.default_rng(3)
    subs = [dict(kind="bed", n=3, blfe=1, start=("code", 2)),
            dict(kind="bed", n=2, start=None),
            dict(kind="dyn", n=2, lfe=0)]
    objs = [(False, None, None)] * 6 + [(True, (0, 0, 0), None),
                                        (True, (62, 0, 0), -6)]
    frames, exps = objs_frames(rng, subs, objs, n=6)
    pcm, dec = decode(frames, max_pass=100)
    assert dec.n_concealed == 0, dict(dec.concealed.reasons)
    assert dec.chans == ("L", "R", "C", "lfe", "Ls", "Rs")
    m = dec.gains(None)
    assert len(m) == 3
    R0 = dec.decs[0].obj_renderer.ramp.cur * 3     # undo the 1 / n_sub
    R1 = dec.decs[1].obj_renderer.ramp.cur
    R2 = dec.decs[2].obj_renderer.ramp.cur
    # substream 0: slots lfe, L, R, C -> objects lfe, L, R, C
    assert np.allclose(R0 / 3, np.eye(6)[:, [3, 0, 1, 2]])
    assert np.allclose(R1, np.eye(6)[:, [4, 5]])
    assert np.allclose(R2[:, 0], [1, 0, 0, 0, 0, 0])
    assert np.allclose(R2[:, 1], [0, 10 ** (-6 / 20), 0, 0, 0, 0])
    assert np.abs(pcm).max() > 0


def test_isf_matrix_rows():
    """Annex A.2.1 as banked: SR3.1.0.0 on 5.1.4 sends U1 to the four
    top loudspeakers only and M1 to the front."""
    m = np.array(OR.isf_matrix(0, "5.1.4"))
    assert m.shape == (4, 9)
    assert np.allclose(m[3, :5], 0) and np.allclose(m[3, 5:], 0.3751586799)
    assert m[0, 0] == m[0, 1] > 0.6


@pytest.mark.parametrize("layout", ["2.0", "5.1", "7.1.4"])
def test_isf_render(layout):
    """ISF objects (isf_config 0, SR3.1.0.0: four objects, shared 3 + 1
    over two substreams) rendered by the matrix of the layout (5.10.3);
    the OAMD gain of an ISF object scales its column."""
    from ac4.decoder import decode
    from ac4.render import LAYOUTS
    rng = np.random.default_rng(5)
    subs = [dict(kind="isf", n=3, start=0), dict(kind="isf", n=1,
                                                start=None)]
    objs = [(False, None, None)] * 3 + [(False, None, -6)]
    frames, _ = objs_frames(rng, subs, objs, n=4)
    st = M.parse(frames[0])
    assert [s["types"] for s in st["groups"][0]["substreams"]] == [
        ["ISF:0:0", "ISF:0:1", "ISF:0:2"], ["ISF:0:3"]]
    pcm, dec = decode(frames, max_pass=100, layout=layout)
    assert dec.n_concealed == 0, dict(dec.concealed.reasons)
    out = LAYOUTS[layout]
    spk = [c for c in out if c != "lfe"]
    ref = np.array(OR.isf_matrix(0, layout))
    R0 = dec.decs[0].obj_renderer.ramp.cur
    R1 = dec.decs[1].obj_renderer.ramp.cur
    for j, row in enumerate(ref[:3]):
        assert np.allclose([R0[out.index(c), j] for c in spk], row)
    assert np.allclose([R1[out.index(c), 0] for c in spk],
                       ref[3] * 10 ** (-6 / 20))


def _c_vs_python(tmp_path, frames, layout=None):
    import os
    import subprocess
    from ac4.decoder import decode
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
    import ac4.decoder as D
    from ac4 import mix as MX
    subs = MX.probe(frames)
    if len(subs) > 1:
        dec = MX.MixDecoder(subs, guard=False, layout=layout)
        for d in dec.decs:
            d.hf_ratio = None
    else:
        dec = D.Decoder("ajoc", guard=False, ajoc_info=subs[0]["info"],
                        layout=layout)
        dec.hf_ratio = None
    ref = dec.decode_pass(frames)
    assert dec.n_concealed == 0
    got = np.fromfile(raw, "<f8").reshape(-1, len(dec.chans))
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()


@pytest.mark.parametrize("case,layout", [
    ("dyn2", None), ("dyn5", "7.1.4"), ("bed", "2.0"), ("isf", "5.1.4"),
    ("mono", "5.1")])
def test_c_objs(tmp_path, case, layout):
    """The C decoder's object substreams against the Python, within
    1e-9."""
    rng = np.random.default_rng(40 + len(case))
    if case == "dyn2":
        subs = [dict(kind="dyn", n=2, lfe=1)]
        objs = [(False, None, None), (True, (5, 9, 3), -3),
                (True, (50, 30, -2), None)]
    elif case == "dyn5":
        subs = [dict(kind="dyn", n=5, lfe=0)]
        objs = [(True, (12 * k, 62 - 12 * k, 3 * k - 6), None)
                for k in range(5)]
    elif case == "mono":
        subs = [dict(kind="dyn", n=1, lfe=0)]
        objs = [(True, (31, 10, 0), None)]
    elif case == "bed":
        subs = [dict(kind="bed", n=3, blfe=1, start=("code", 2)),
                dict(kind="bed", n=2, start=None),
                dict(kind="dyn", n=2, lfe=0)]
        objs = [(False, None, None)] * 6 + [(True, (0, 0, 0), None),
                                            (True, (62, 0, 0), -6)]
    else:
        subs = [dict(kind="isf", n=5, start=1), dict(kind="isf", n=3,
                                                    start=None)]
        objs = [(False, None, None)] * 8
    frames, _ = objs_frames(rng, subs, objs, n=5)
    _c_vs_python(tmp_path, frames, layout)


# ---- bitstream_version 0: extended_metadata, DRC (item 4) ------------------

def v0_meta(cm=1, dialnorm=92, assoc=None, dmax=None, drc=None):
    """metadata() of a sus_ver 0 substream (TS 103 190-1 Table 66) as
    bytes: basic_metadata (dialnorm, nothing more), extended_metadata
    with b_associated (assoc = (scale_main, _centre, _front raw codes or
    None, pan code for a mono substream)) or b_dialog (dmax:
    dialog_max_gain code), tools_metadata_size, drc_frame (drc: None,
    "config": one mode, default profile, eac3_profile 1 (I-frame), or
    "data": a later frame), no DE, no EMDF."""
    w = BW().u(7, dialnorm).u(1, 0)
    if assoc is not None:
        for v in assoc[:3]:
            if v is None:
                w.u(1, 0)
            else:
                w.u(1, 1).u(8, v)
        if cm == 0:
            w.u(8, assoc[3])
    if dmax is not None:
        w.u(1, 1).u(2, dmax).u(1, 0)     # dialog_max_gain, no pan_dialog
    w.u(1, 0).u(1, 0)                    # channels classifier, event prob.
    tools = BW()
    if drc is None:
        tools.u(1, 0)
    else:
        tools.u(1, 1)
        if drc == "config":
            tools.u(3, 0).u(3, 0).u(1, 0).u(1, 1).u(3, 1)
        tools.u(1, 0).u(2, 0)            # drc_reset_flag, reserved
    tools.u(1, 0).u(1, 0)                # no DE data, no EMDF
    w.u(7, len(tools.bits)).u(1, 0)
    w.bits += tools.bits
    return w.bytes()


def v0_sub(rng, kind, **kw):
    """An audio substream (SIMPLE pair or mono) with its sus_ver 0
    metadata."""
    from test_mix import mono_sub, pair_sub
    body = pair_sub(rng) if kind == "pair" else mono_sub(rng)
    return body + v0_meta(cm=1 if kind == "pair" else 0, **kw)


def test_v0_associated_gains_and_dialnorm():
    """presentation_config 2 (main + associated): the associated mono
    substream's extended_metadata (scale_main -3 dB, scale_main_centre
    absent, scale_main_front -6 dB, pan_associated 30 degrees) reaches
    the mixer; dialnorm comes from the associated substream (4.8.5.2)."""
    from test_bitstream_v0 import toc_v0
    from ac4 import mix as MX
    from ac4.decoder import Decoder
    rng = np.random.default_rng(21)
    frames = []
    for i in range(5):
        a = v0_sub(rng, "pair", dialnorm=92)
        d = v0_sub(rng, "mono", dialnorm=80, assoc=(10, None, 20, 20))
        frames.append(toc_v0([("10", None), ("0", None)], [len(a), len(d)],
                             config=2, seq=5 + i, iframe=int(i == 0))
                      + a + d)
    subs = MX.probe(frames)
    assert [x["type"] for x in subs] == ["main", "associated"]
    mx = MX.MixDecoder(subs, guard=False)
    y = mx.decode_pass(frames)
    meta = mx.decs[0].stream.meta
    assert [m["dialnorm"] for m in meta] == [-20.0] * 5
    a = meta[0]["assoc"]
    assert a["scale_main"] == pytest.approx(-3.0)
    assert a["scale_main_centre"] is None
    assert a["scale_main_front"] == pytest.approx(-6.0)
    assert a["pan_associated"] == pytest.approx(30.0)
    ya = Decoder("pair", guard=False).decode_pass(frames)
    yb = Decoder("mono", guard=False, substream=1).decode_pass(frames)
    g = 10 ** (-3 / 20) * 10 ** (-6 / 20)
    exp = ya * g + np.column_stack([0 * yb[:, 0], yb[:, 0]])
    assert np.abs(y - exp).max() <= 1e-12 * np.abs(exp).max()


def test_v0_dialog_max_gain():
    """presentation_config 0 (M&E + dialogue): dialog_max_gain 1 (6 dB)
    caps the user's dialogue gain; an I-frame without it resets it to 0
    dB (4.3.12.4.11)."""
    from test_bitstream_v0 import toc_v0
    from ac4 import mix as MX
    from ac4.decoder import Decoder
    rng = np.random.default_rng(22)
    frames = []
    for i in range(6):
        a = v0_sub(rng, "pair")
        d = v0_sub(rng, "mono", dmax=None if i >= 4 else 1)
        frames.append(toc_v0([("10", None), ("0", None)], [len(a), len(d)],
                             config=0, seq=5 + i, iframe=int(i in (0, 4)))
                      + a + d)
    mx = MX.MixDecoder(MX.probe(frames), guard=False, dialog_gain=9.0)
    y = mx.decode_pass(frames)
    assert [m["dmax"] for m in mx.decs[0].stream.meta] == [6.0] * 4 + [
        0.0, 0.0]
    ya = Decoder("pair", guard=False).decode_pass(frames)
    yb = Decoder("mono", guard=False, substream=1).decode_pass(frames)
    F = 1536
    for f, gd in ((1, 6.0), (5, 0.0)):
        seg = slice(f * F + F // 2, f * F + F // 2 + 1)
        exp = ya[seg] + yb[seg][:, [0, 0]] * 0.5 * 10 ** (gd / 20)
        # (frames 1 and 5 hold their matrix from the frame before)
        assert np.allclose(y[seg], exp, rtol=1e-9, atol=1e-9)


def test_v0_drc_frame():
    """The sus_ver 0 drc_frame (TS 103 190-1 Table 66) is read: its
    configuration from the I-frame, held after it."""
    from test_bitstream_v0 import toc_v0
    from ac4.stream import Ac4Stream
    rng = np.random.default_rng(23)
    frames = []
    for i in range(4):
        a = v0_sub(rng, "pair", drc="config" if i == 0 else "data")
        frames.append(toc_v0([("10", 0)], [len(a)], seq=5 + i,
                             iframe=int(i == 0)) + a)
    s = Ac4Stream("pair")
    s.decode_frames(frames)
    assert s.n_bad == 0
    d = [m["drc"] for m in s.meta]
    assert all(x is not None for x in d)
    assert d[0]["config"]["eac3_profile"] == 1
    assert [m["id"] for m in d[3]["config"]["modes"]] == [0]


@pytest.mark.parametrize("case", ["assoc", "drc"])
def test_c_v0_metadata(tmp_path, case):
    """The C decoder against the Python on the v0 metadata streams,
    loudness-normalised (dialnorm and DRC from the right substream)."""
    import os
    import subprocess
    from test_bitstream_v0 import toc_v0
    from test_inputs import syncframe
    from ac4 import mix as MX
    from ac4.decoder import Decoder
    cli = os.environ.get("AC4_C_CLI", C_CLI)
    if not os.path.exists(cli):
        pytest.skip("c/ac4dec-c not built")
    rng = np.random.default_rng(24)
    frames = []
    for i in range(6):
        if case == "assoc":
            a = v0_sub(rng, "pair", dialnorm=92)
            d = v0_sub(rng, "mono", dialnorm=80, assoc=(10, 3, 20, 200),
                       drc="config" if i == 0 else "data")
            frames.append(toc_v0([("10", None), ("0", None)],
                                 [len(a), len(d)], config=2, seq=5 + i,
                                 iframe=int(i == 0)) + a + d)
        else:
            a = v0_sub(rng, "pair", dialnorm=70,
                       drc="config" if i == 0 else "data")
            frames.append(toc_v0([("10", 0)], [len(a)], seq=5 + i,
                                 iframe=int(i == 0)) + a)
    src = tmp_path / "e.ac4"
    src.write_bytes(b"".join(syncframe(f) for f in frames))
    raw = tmp_path / "e.f64"
    subprocess.run([cli, str(src), str(tmp_path / "e.wav"), "--f64",
                    str(raw), "--internal", "--no-guard", "-q"], check=True)
    subs = MX.probe(frames)
    if len(subs) > 1:
        dec = MX.MixDecoder(subs, guard=False, target_level=-24.0)
        for d_ in dec.decs:
            d_.hf_ratio = None
    else:
        dec = Decoder("pair", guard=False, target_level=-24.0)
        dec.hf_ratio = None
    ref = dec.decode_pass(frames)
    got = np.fromfile(raw, "<f8").reshape(-1, len(dec.chans))
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()


def toc_v1_cfg7(sub_len, seq=5, iframe=1, sus_ver=1, skip_extra=0):
    """bitstream_version 1, one ac4_presentation_info of
    presentation_config 7 whose presentation_config_ext_info holds an
    ac4_presentation_v1_info (single group, inline: one stereo
    substream, index 1; presentation substream index 0)."""
    from test_phase_a import emdf_info, rate_info
    v = BW().u(1, 1)                     # b_single_substream_group
    v.u(1, 1).u(1, 0)                    # presentation_version 1
    v.u(3, 0).u(1, 0)                    # md_compat, b_presentation_id
    rate_info(v, 3)
    emdf_info(v)
    v.u(1, 0)                            # b_presentation_filter
    # ac4_sgi_specifier (bitstream_version 1): ac4_substream_group_info
    v.u(1, 1).u(1, 0).u(1, 1).u(1, 1)    # present, no hsf, single, chan
    v.u(1, sus_ver).u(2, 0b10)           # sus_ver, channel_mode stereo
    v.u(1, 0).u(1, 0).u(1, iframe).u(2, 1)   # sf_mult, bitrate, ndot, idx
    v.u(1, 0)                            # b_content_type
    v.u(1, 0).u(1, 0)                    # b_pre_virtualized, b_add_emdf
    v.u(1, 0).u(1, 1).u(2, 0)            # presentation_substream_info
    vb = v.bytes()
    w = BW().u(2, 1).u(10, seq).u(1, 0).u(1, 1).u(4, 3).u(1, iframe)
    w.u(1, 1).u(1, 0)                    # single presentation, no base
    w.u(1, 0).u(3, 7).vb(2, 0)           # presentation_config 7
    w.u(1, 0)                            # presentation_version 0
    w.u(3, 0).u(1, 0)                    # md_compat, b_presentation_id
    rate_info(w, 3)
    emdf_info(w)
    w.u(1, 0)                            # b_hsf_ext
    n = len(vb) + skip_extra
    w.u(5, n & 31).u(1, int(n > 31))
    if n > 31:
        w.vb(2, n >> 5)
    for byte in vb + bytes(skip_extra):
        w.u(8, byte)
    w.u(1, 0).u(1, 0)                    # b_pre_virtualized, b_add_emdf
    w.u(2, 2).u(1, 0).u(10, 2).u(1, 0).u(10, sub_len)
    return w.bytes()


@pytest.mark.parametrize("skip_extra", [0, 3])
def test_v1_presentation_config_7(skip_extra):
    """bitstream_version 1, presentation_config 7: the
    ac4_presentation_v1_info in presentation_config_ext_info (TS 103
    190-2 6.2.1.5) is the presentation decoded; it decodes as the same
    audio under a bitstream_version 2 TOC."""
    from test_mix import pair_sub, pres_sub
    from ac4.decoder import Decoder
    import elem_writer as EW_
    rng = np.random.default_rng(31)
    f1, f2 = [], []
    for i in range(5):
        a = pair_sub(rng)
        ps = bytes(2)
        f1.append(toc_v1_cfg7(len(a), 5 + i, int(i == 0),
                              skip_extra=skip_extra) + ps + a)
        t = dict(pres=[dict(groups=[0], pres_sub=0)],
                 groups=[dict(ch_mode_code="10", index=1,
                              ndot=int(i == 0))], sizes=[2, len(a)])
        from test_phase_a import build_toc
        f2.append(build_toc(**t, seq=5 + i, iframe=int(i == 0)) + ps + a)
    st = M.parse(f1[0])
    p = st["presentations"][0]
    assert p.get("config_ext") == 7 and not p.get("v0")
    assert M.audio_substream(st, 0)["channel_mode"] == 1
    a = Decoder("pair", guard=False)
    b = Decoder("pair", guard=False)
    ya, yb = a.decode_pass(f1), b.decode_pass(f2)
    assert a.n_concealed == 0 and b.n_concealed == 0
    assert np.abs(ya).max() > 0 and np.array_equal(ya, yb)


def test_c_v1_presentation_config_7(tmp_path):
    import os
    import subprocess
    from test_mix import pair_sub
    from test_inputs import syncframe
    from ac4.decoder import Decoder
    cli = os.environ.get("AC4_C_CLI", C_CLI)
    if not os.path.exists(cli):
        pytest.skip("c/ac4dec-c not built")
    rng = np.random.default_rng(32)
    frames = []
    for i in range(5):
        a = pair_sub(rng)
        frames.append(toc_v1_cfg7(len(a), 5 + i, int(i == 0),
                                  skip_extra=2) + bytes(2) + a)
    src = tmp_path / "e.ac4"
    src.write_bytes(b"".join(syncframe(f) for f in frames))
    raw = tmp_path / "e.f64"
    subprocess.run([cli, str(src), str(tmp_path / "e.wav"), "--f64",
                    str(raw), "--internal", "--no-guard", "--legacy-level",
                    "-q"], check=True)
    dec = Decoder("pair", guard=False)
    dec.hf_ratio = None
    ref = dec.decode_pass(frames)
    got = np.fromfile(raw, "<f8").reshape(-1, 2)
    assert dec.n_concealed == 0 and got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()
