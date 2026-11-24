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
