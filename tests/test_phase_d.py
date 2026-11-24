# SPDX-License-Identifier: Apache-2.0
"""Phase D (no captures): A-JOC full decoding (TS 103 190-2 5.7), OAMD
timing (5.9), written bit by bit. Synthetic streams only: no A-JOC
stream was at hand."""
import math

import numpy as np
import pytest

from ac4 import ajoc as AJ
from ac4 import channels as CH
import elem_writer as EW
from test_phase_a import BW
from test_phase_b2 import _books, ASPX_CFG, _aspx_cfg_bits, _aspx_1ch, \
    _aspx_2ch, C_CLI
from test_phase_c import _ajoc_toc, _ajoc_put, _dmx_oamd, _static_5x


# --------------------------------------------------------------- writer
def _side(rng, n_dmx, n_umx, nd=1, start=0, ramp=1, nb_code=7, qs=1,
          q=None, sparse=None):
    """An ajoc() with one data point: every object present, nb_code /
    qs for all; q[o] = (dry [n_dmx][nb], wet [nd][nb]) quantised values
    (random if None). -> (side dict for the writer, q)."""
    nb = AJ.NUM_BANDS[nb_code]
    nqd, nqw = AJ.DRY_STEPS[qs], AJ.WET_STEPS[qs]
    if q is None:
        q = [([list(rng.integers(0, nqd, nb)) for _ in range(n_dmx)],
              [list(rng.integers(0, nqw, nb)) for _ in range(nd)])
             for _ in range(n_umx)]
    return dict(nd=nd, start=start, ramp=ramp, nb_code=nb_code, qs=qs,
                q=q, sparse=sparse), q


def _side_bits(w, n_dmx, n_umx, s):
    """ajoc() (6.2.5) for _side's dict; data frequency differential."""
    nd, qs = s["nd"], s["qs"]
    w.u(3, nd)
    for _ in range(nd):
        w.u(1, 1)                                # ajoc_decorr_enable
    for _ in range(n_umx):
        w.u(1, 1)                                # ajoc_object_present
    if s["start"] is None:
        w.u(2, 0)                                # no data points
    else:
        w.u(2, 1).u(5, s["start"]).u(6, s["ramp"] - 1)
        for o in range(n_umx):
            w.u(3, s["nb_code"]).u(1, qs).u(1, 0)    # not sparse
    w.u(1, 1)                                    # ajoc_b_nodt
    if s["start"] is None:
        return
    nqd, nqw = AJ.DRY_STEPS[qs], AJ.WET_STEPS[qs]
    for o in range(n_umx):
        dry, wet = s["q"][o]
        for vals, kind, nq in [(v, "DRY", nqd) for v in dry] + \
                [(v, "WET", nqw) for v in wet]:
            _ajoc_put(w, kind, qs, "F0", int(vals[0]))
            for b in range(1, len(vals)):
                _ajoc_put(w, kind, qs, "DF", int((vals[b] - vals[b - 1])
                                                 % nq))


def _umx_oamd(w, n_umx, lfe, pos, iframe, timing=None, gains=None):
    """b_umx_timing and the upmix's oamd_dyndata_single: every object
    active at pos[o] (and the LFE); timing: None (derive from the
    downmix) or [(block_offset_factor, ramp_code, ramp_extra)] with
    per-block positions pos[blk][o]."""
    if timing is None:
        w.u(1, 0).u(1, 1)                        # derive from dmx
        blocks = [pos]
    else:
        w.u(1, 1).u(1, 0).u(3, len(timing))      # sample_offset 0
        for off, rc, extra in timing:
            w.u(6, off).u(2, rc)
            if rc == 3:
                w.u(1, 0).u(11, extra)
        blocks = pos
    n_all = n_umx + (1 if lfe else 0)
    for k in range(n_all):
        for bi, bpos in enumerate(blocks):
            nodelta = iframe and bi == 0
            w.u(1, 0)                            # active
            if not nodelta:
                w.u(1, 0)                        # basic info: new
            if gains is None:
                w.u(1, 1)                        # default basic (0 dB)
            else:
                # not default, md 0 (gain only), gain code 0b0 + 6 bits
                w.u(1, 0).u(1, 0).u(1, 0).u(6, gains[k])
            if not (lfe and k == 0):
                if not nodelta:
                    w.u(1, 0).u(1, 0)            # render: new
                x, y, z = bpos[k - (1 if lfe else 0)]
                if not nodelta:
                    w.u(1, 0)                    # b_diff_pos_coding
                w.u(6, x).u(6, y).u(1, 1).u(4, z)
                w.u(1, 1).u(1, 1)                # zone, other: defaults
            w.u(1, 0)                            # b_add_table_data


def _ajoc_sub(rng, n_dmx, n_umx, lfe, dpos, upos, side, iframe=True,
              aspx=False, static=False, timing=None, ugains=None):
    from ac4 import aspx_bands as A
    T = CH.load_tables()
    books, abooks = _books()
    t_ = A.tables_for(ASPX_CFG, 0)
    if static:
        w, exp = _static_5x(rng, lfe)
    else:
        w = BW().u(1, 0)                     # b_some_signals_inactive
        w.u(1, int(aspx))                    # var_codec_mode
        if aspx:
            if iframe:
                _aspx_cfg_bits(w, ASPX_CFG)
            if n_dmx <= 5:
                w.u(1, 1).u(1, 1) if n_dmx > 1 else w.u(1, 1)
        if lfe:
            EW.lfe(w, T, EW.Track(rng, 2))
        n = 10
        for p in range(n_dmx // 2 - (1 if n_dmx % 2 else 0)):
            tr = [EW.Track(rng, n), EW.Track(rng, n)]
            EW.two_channel(w, T, tr, EW.St(rng, 0, n))
        if n_dmx % 2:
            if n_dmx > 1:
                w.u(1, 0)                    # var_coding_config
                tr = [EW.Track(rng, n), EW.Track(rng, n)]
                EW.two_channel(w, T, tr, EW.St(rng, 0, n))
            EW.mono(w, T, EW.Track(rng, n))
        if aspx:
            for _ in range(n_dmx // 2):
                _aspx_2ch(w, abooks, t_, iframe)
            if n_dmx % 2:
                _aspx_1ch(w, abooks, t_, iframe)
        _dmx_oamd(w, n_dmx, lfe, dpos, iframe)
    _side_bits(w, n_dmx, n_umx, side)
    w.u(1, 0).u(1, 1)                            # dmx_de: no cfg, keep
    _umx_oamd(w, n_umx, lfe, upos, iframe, timing, ugains)
    return EW.element(w)


def ajoc_frames(rng, n_dmx=3, n_umx=4, lfe=1, n=6, aspx=False,
                static=False, sides=None, upos=None, dpos=None,
                timing=None, ugains=None, iframe_every=0):
    """n frames of an A-JOC substream with data (sides[i] or random);
    an I-frame every iframe_every frames (0: the first only)."""
    dpos = dpos or [(9 * k % 63, 61 - 8 * k % 62, 2 * k % 16)
                    for k in range(n_dmx)]
    upos = upos or [(7 * k % 63, (5 + 11 * k) % 63, (3 * k) % 16)
                    for k in range(n_umx)]
    frames, qs = [], []
    for i in range(n):
        side = sides[i] if sides else _side(rng, n_dmx, n_umx)[0]
        ifr = i == 0 or (iframe_every and i % iframe_every == 0)
        sub = _ajoc_sub(rng, n_dmx, n_umx, lfe, dpos, upos, side,
                        iframe=ifr, aspx=aspx, static=static,
                        timing=timing, ugains=ugains)
        toc = _ajoc_toc(len(sub), (5 + i) % 1020 + 1, int(ifr), n_dmx,
                        n_umx, lfe, static)
        frames.append(toc + bytes(2) + sub)
        qs.append(side["q"])
    return frames, qs


# ---------------------------------------------------------------- tests
def test_input_order_pseudocode_14a():
    assert AJ.input_order(1) == [0]
    assert AJ.input_order(3) == [0, 1, 2]
    assert AJ.input_order(4) == [2, 3, 0, 1]
    assert AJ.input_order(5) == [2, 3, 4, 0, 1]
    assert AJ.input_order(6) == [4, 5, 0, 1, 2, 3]
    assert AJ.input_order(7) == [4, 5, 6, 0, 1, 2, 3]


def test_ramp():
    """A data point at slot s with ramp L: slot s keeps the old value,
    slot s + L has the new one, linearly in between; the first is taken
    at once."""
    r = AJ.Ramp((1,))
    r.event(np.array([2.0]), 5)
    assert r.prev[0] == 2.0
    vals = []
    r.event(np.array([4.0]), 4)
    for _ in range(6):
        vals.append(r.prev[0])
        r.step()
    assert np.allclose(vals, [2.0, 2.5, 3.0, 3.5, 4.0, 4.0])


def _ref_pc18(X, events, n_umx, nd, enable):
    """Pseudocode 18 written out scalar by scalar (as read in ajoc.py):
    X (n_dmx, n, 64); events {slot: (dry (n_umx, n_dmx, 64), wet (n_umx,
    7, 64), ramp)} -> Z."""
    from ac4 import acpl as AC
    n_dmx, n = X.shape[0], X.shape[1]
    dry = np.zeros((n_umx, n_dmx, 64))
    wet = np.zeros((n_umx, 7, 64))
    pre = np.zeros((7, n_dmx, 64))
    dd = np.zeros_like(dry)
    dw = np.zeros_like(wet)
    dp = np.zeros_like(pre)
    curr = tgt = 0
    warm = False
    U = np.zeros((nd, n, 64), complex)
    D = np.zeros((n, n_umx, n_dmx, 64))
    W = np.zeros((n, n_umx, 7, 64))
    for ts in range(n):
        if ts in events:
            ndry, nwet, L = events[ts]
            npre = np.zeros_like(pre)
            for sb in range(64):
                for de in range(nd):
                    for ch in range(n_dmx):
                        acc = 0.0
                        for o in range(n_umx):
                            acc += abs(nwet[o, de, sb]) * ndry[o, ch, sb]
                        npre[de, ch, sb] = acc
            if not warm:
                dry, wet, pre = ndry.copy(), nwet.copy(), npre
                dd[:] = dw[:] = dp[:] = 0
                curr = tgt = 0
                warm = True
            else:
                dd, dw, dp = (ndry - dry) / L, (nwet - wet) / L, \
                    (npre - pre) / L
                curr, tgt = 0, L
        D[ts], W[ts] = dry, wet
        for de in range(nd):
            for sb in range(64):
                u = 0j
                for ch in range(n_dmx):
                    u += pre[de, ch, sb] * X[ch, ts, sb]
                U[de, ts, sb] = u
        if curr < tgt:
            dry, wet, pre = dry + dd, wet + dw, pre + dp
            curr += 1
    st = AC.new_state()
    st = {k: (np.zeros((nd,) + v.shape[1:], v.dtype)
              if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    Y = AC.duck(AC.decorrelate(U, st, rows=AJ.DECORR_CYCLE[:nd]), st) \
        if nd else U
    Z = np.zeros((n_umx, n, 64), complex)
    for o in range(n_umx):
        for ts in range(n):
            for sb in range(64):
                z = 0j
                for ch in range(n_dmx):
                    z += X[ch, ts, sb] * D[ts, o, ch, sb]
                for de in range(nd):
                    z += Y[de, ts, sb] * enable[de] * W[ts, o, de, sb]
                Z[o, ts, sb] = z
    return Z


@pytest.mark.parametrize("n_dmx,n_umx,nd", [(1, 2, 0), (2, 3, 1),
                                            (3, 4, 3), (2, 2, 7)])
def test_engine_is_pseudocode_18(n_dmx, n_umx, nd):
    """Engine against Pseudocode 18 written out (one frame, two data
    points with ramps) and through the parameter band mapping."""
    rng = np.random.default_rng(n_dmx * 10 + nd)
    X = rng.standard_normal((n_dmx, 40, 64)) + \
        1j * rng.standard_normal((n_dmx, 40, 64))
    ev = {}
    tg = []
    for start, ramp in ((2, 1), (11, 9)):
        dry = rng.uniform(-2, 2, (n_umx, n_dmx, 64))
        wet = np.zeros((n_umx, 7, 64))
        wet[:, :nd] = rng.uniform(-1, 1, (n_umx, nd, 64))
        ev[start] = (dry, wet, ramp)
        pre = np.zeros((7, n_dmx, 64))
        for o in range(n_umx):
            for de in range(nd):
                for ch in range(n_dmx):
                    pre[de, ch] += np.abs(wet[o, de]) * dry[o, ch]
        tg.append(dict(start=start, ramp=ramp, dry=dry, wet=wet, pre=pre))
    en = [1] * nd
    eng = AJ.Engine(n_dmx, n_umx)
    eng.frame(0, tg, nd, en)
    Z = np.concatenate([eng.run(X[:, :7]), eng.run(X[:, 7:])], axis=1)
    R = _ref_pc18(X, ev, n_umx, nd, en)
    assert np.abs(Z - R).max() < 1e-12 * max(1.0, np.abs(R).max())


def test_frame_targets_band_mapping():
    """frame_targets: Tables 28 - 32 on the QMF subbands, the
    decorrelation input matrix of 5.7.3.6.2."""
    rng = np.random.default_rng(5)
    side, q = _side(rng, 2, 2, nd=2, nb_code=4, qs=0)
    s = dict(num_decorr=2, decorr_enable=[1, 1], present=[1, 1],
             dpoints=[(0, 1)], num_bands=[7, 7], quant_select=[0, 0],
             data=None)
    t = AJ.frame_targets(s, [[(q[o][0], q[o][1])] for o in range(2)], 2, 2)
    pbm = AJ.sb_to_pb(7)
    assert np.allclose(t[0]["dry"][1, 0],
                       AJ.dequant_dry(q[1][0][0], 0)[pbm])
    assert np.allclose(t[0]["wet"][0, 1],
                       AJ.dequant_wet(q[0][1][1], 0)[pbm])
    D = np.abs(t[0]["wet"][:, :2]).transpose(1, 0, 2)   # (de, o, sb)
    exp = np.einsum("dos,ocs->dcs", D, t[0]["dry"])
    assert np.allclose(t[0]["pre"][:2], exp)


def test_full_decoding_parse():
    """The A-JOC data with data points written bit by bit closes on
    audio_size and decodes to the written quantised values."""
    from ac4 import decoder as D
    from ac4 import stream as SC
    rng = np.random.default_rng(3)
    frames, qs = ajoc_frames(rng, n_dmx=3, n_umx=4, n=3)
    info = D.probe_ajoc_info(frames)
    s = SC.Ac4Stream("ajoc", ajoc_info=info)
    _w, _l, groups, _c = s.decode_frames(frames)
    assert s.n_bad == 0, s.status
    for i in range(3):
        o = groups["obj"][i]
        for k in range(4):
            dq, wq = o["q"][k][0]
            assert [list(v) for v in dq] == [list(v) for v in qs[i][k][0]]
            assert [list(v) for v in wq] == [list(v) for v in qs[i][k][1]]
        assert len(o["umx_blocks"]) == 1


def _q_gain(g, qs=1):
    """The dry quantisation index of gain g (Tables 29 / 30)."""
    n = AJ.DRY_STEPS[qs]
    mid = (n - 1) // 2
    return int(round(g / (AJ.DRY_MAX / mid))) + mid


def test_full_decoding_dry_upmix_is_scaled_downmix():
    """One downmix signal, objects with dry gains only (no
    decorrelators), each at a loudspeaker of 5.1: every loudspeaker
    plays its object's gain times the downmix (the core decoding of the
    same signal at that position), to rounding."""
    from ac4.decoder import Decoder, probe_ajoc_info
    rng = np.random.default_rng(11)
    gq = [_q_gain(g) for g in (1.0, -0.6, 0.4)]
    q = [([[v]], []) for v in gq]
    sides = [_side(rng, 1, 3, nd=0, start=0, ramp=1, q=q)[0]
             for _ in range(6)]
    upos = [(0, 0, 0), (62, 0, 0), (31, 0, 0)]       # L, R, C
    frames, _ = ajoc_frames(rng, n_dmx=1, n_umx=3, lfe=0, n=6,
                            sides=sides, upos=upos, dpos=[(31, 0, 0)])
    info = probe_ajoc_info(frames)
    full = Decoder("ajoc", guard=False, ajoc_info=info, layout="5.1")
    core = Decoder("ajoc", guard=False, ajoc_info=info, layout="5.1",
                   ajoc_core=True)
    for d in (full, core):
        d.hf_ratio = None
    yf = full.decode_pass(frames)
    yc = core.decode_pass(frames)
    assert full.n_concealed == 0 and core.n_concealed == 0
    c = yc[:, 2]                         # the downmix at C
    assert np.abs(c).max() > 0
    g = [AJ.dequant_dry([v], 1)[0] for v in gq]
    pk = np.abs(c).max()
    # from the first data point on (slot ts_offset 18 of frame 0) and
    # the QMF synthesis's ten slots after it
    s0 = (18 + 10) * 64
    for ch, gg in ((0, g[0]), (1, g[1]), (2, g[2])):
        assert np.abs(yf[s0:, ch] - gg * c[s0:]).max() < 1e-9 * pk, ch
    assert np.abs(yf[s0:, 3:]).max() < 1e-9 * pk


def test_full_decoding_decorrelated_objects():
    """Decorrelators on (the decorrelator's input is |wet| x dry,
    5.7.3.6.2): two objects with dry 0.5 and wet +2 / -2: their sum is
    the downmix, their difference (4 y) has energy and is decorrelated
    from it."""
    from ac4.decoder import Decoder, probe_ajoc_info
    rng = np.random.default_rng(12)
    q = [([[_q_gain(0.5)]], [[20]]), ([[_q_gain(0.5)]], [[0]])]
    sides = [_side(rng, 1, 2, nd=1, start=0, ramp=1, q=q)[0]
             for _ in range(8)]
    frames, _ = ajoc_frames(rng, n_dmx=1, n_umx=2, lfe=0, n=8,
                            sides=sides, upos=[(0, 0, 0), (62, 0, 0)],
                            dpos=[(31, 0, 0)])
    info = probe_ajoc_info(frames)
    d = Decoder("ajoc", guard=False, ajoc_info=info, layout="2.0")
    d.hf_ratio = None
    y = d.decode_pass(frames)
    assert d.n_concealed == 0
    a, b = y[3072:, 0], y[3072:, 1]
    s_, d_ = a + b, a - b
    assert np.abs(s_).max() > 0 and np.abs(d_).max() > 0.1 * np.abs(s_).max()
    assert abs(np.corrcoef(s_, d_)[0, 1]) < 0.5


@pytest.mark.parametrize("n_dmx,aspx,static", [(2, False, False),
                                                (5, True, False),
                                                (7, False, False),
                                                (5, False, True)])
def test_full_decoding_streams(n_dmx, aspx, static):
    """Random A-JOC data (three decorrelators, 15 bands): every frame
    closes, the output is finite and not silent."""
    from ac4.decoder import Decoder, probe_ajoc_info
    rng = np.random.default_rng(40 + n_dmx)
    sides = [_side(rng, n_dmx, 5, nd=3, start=3, ramp=8, nb_code=1,
                   qs=i % 2)[0] for i in range(6)]
    frames, _ = ajoc_frames(rng, n_dmx=n_dmx, n_umx=5, n=6, aspx=aspx,
                            static=static, sides=sides)
    info = probe_ajoc_info(frames)
    d = Decoder("ajoc", guard=False, ajoc_info=info, layout="7.1.4")
    d.hf_ratio = None
    y = d.decode_pass(frames)
    assert d.n_concealed == 0, dict(d.concealed.reasons)
    assert np.isfinite(y).all() and np.abs(y[2048:]).max() > 0


def test_oamd_block_updates_core_decoding():
    """Core decoding with two block updates per frame: the object is at
    L from sample 0 and moves to R at sample 32 x 20 = 640 with a ramp
    of 512 samples (Tables 94; 5.9.2)."""
    from ac4 import oamd as O
    from ac4.decoder import ObjectRenderer
    info = dict(b_lfe=0, static=0, n_dmx=1, dmx_types=["DYN"])
    r = ObjectRenderer(("C",), info, "2.0")
    L, R = O.ObjState(), O.ObjState()
    for st, x in ((L, 0), (R, 62)):
        st.active, st.gain = True, 0.0
        st.set_pos(x, 0, 0)
    obj = dict(dmx=[R], dmx_blocks=[[L], [R]],
               dmx_upd=O.block_updates(dict(sample_offset=0, blocks=[
                   (0, 0), (20, 512)]), 2))
    y = r.process(np.ones((1536, 1)), [obj], 1536)
    assert np.allclose(y[:641], [1, 0])
    k = np.arange(512)
    assert np.allclose(y[640:1152, 1], np.sin(0) + k / 512)
    assert np.allclose(y[1152:], [0, 1])
    assert O.block_updates(dict(sample_offset=8, blocks=[(3, ("table", 6))]),
                           2) == [(104, 1000), (8, 0)]


def _c_vs_python(tmp_path, frames, layout, core=False):
    import os
    import subprocess
    from ac4.decoder import Decoder, probe_ajoc_info
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
    if core:
        args += ["--core-decoding"]
    subprocess.run(args, check=True)
    dec = Decoder("ajoc", guard=False, ajoc_info=probe_ajoc_info(frames),
                  layout=layout, ajoc_core=core)
    dec.hf_ratio = None
    ref = dec.decode_pass(frames)
    assert dec.n_concealed == 0
    got = np.fromfile(raw, "<f8").reshape(-1, len(dec.chans))
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()


@pytest.mark.parametrize("n_dmx,aspx,static,layout,nd", [
    (1, False, False, None, 0), (3, False, False, "2.0", 2),
    (5, True, False, "7.1.4", 3), (7, False, False, "5.1.2", 7),
    (6, True, False, None, 1), (5, False, True, "5.1.4", 3)])
def test_c_full_decoding(tmp_path, n_dmx, aspx, static, layout, nd):
    """The C decoder's A-JOC full decoding against the Python."""
    rng = np.random.default_rng(200 + n_dmx)
    sides = [_side(rng, n_dmx, 6, nd=nd, start=(5 * i) % 20,
                   ramp=1 + 7 * i, nb_code=i % 8, qs=i % 2)[0]
             for i in range(6)]
    tm = [(0, 0, 0), (12, 1, 0), (30, 3, 700)]
    upos = [[(9 * k % 63, (5 + 11 * k + 20 * b) % 63, (3 * k + b) % 16)
             for k in range(6)] for b in range(3)]
    frames, _ = ajoc_frames(rng, n_dmx=n_dmx, n_umx=6, n=6, aspx=aspx,
                            static=static, sides=sides, upos=upos,
                            timing=tm, ugains=[3, 9, 0, 20, 5, 1, 7])
    _c_vs_python(tmp_path, frames, layout)


def test_c_core_decoding_timing(tmp_path):
    """Core decoding with OAMD block updates and ramps, C against
    Python."""
    rng = np.random.default_rng(300)
    frames, _ = ajoc_frames(rng, n_dmx=5, n_umx=2, n=6,
                            sides=[_side(rng, 5, 2)[0]] * 6)
    _c_vs_python(tmp_path, frames, "5.1", core=True)


# ------------------------------------------- LtRt, loudness correction
def test_table_218_ltrt():
    """Table 218: LtRt and the Pro Logic II weights; LtRt gains default
    to the LoRo ones without b_ltrt_mixinfo."""
    from ac4 import render as RD
    ch = ("L", "R", "C", "lfe", "Ls", "Rs")
    st = dict(loro_centre=-3.0, loro_surround=-4.5, lfe_mixgain=None,
              preferred_dmx_method=2)
    cd = dict(stereo=st)
    i = {k: n for n, k in enumerate(ch)}
    g = 10 ** (-4.5 / 20)
    R = RD.matrix(ch, "2.0", cd, "ltrt")
    assert np.allclose(R[0, [i["Ls"], i["Rs"]]], [-g, -g])
    assert np.allclose(R[1, [i["Ls"], i["Rs"]]], [g, g])
    assert np.allclose(RD.matrix(ch, "2.0", cd, "auto"), R)
    st2 = dict(st, ltrt_centre=0.0, ltrt_surround=-6.0)
    R = RD.matrix(ch, "2.0", dict(stereo=st2), "pl2")
    s = 10 ** (-6 / 20)
    assert np.allclose(R[0, [i["C"], i["Ls"], i["Rs"]]],
                       [1.0, -s * 10 ** (1.8 / 20), -s * 10 ** (-3.2 / 20)])
    assert np.allclose(R[1, [i["Ls"], i["Rs"]]],
                       [s * 10 ** (-3.2 / 20), s * 10 ** (1.8 / 20)])
    R = RD.matrix(ch, "2.0", dict(stereo=st2), "loro")
    assert np.allclose(R[0, [i["Ls"], i["Rs"]]], [g, 0.0])
    assert RD.stereo_method("auto", None) == "loro"


def test_loud_corr_bits():
    """loud_corr (TS 103 190-2 6.2.9.1) written bit by bit for a 7.1.4
    presentation (core 5.1.2): (15 - v) / 2 dB, 31 = 0 dB."""
    from ac4 import metadata as MD
    from ac4.toc import Bits
    w = BW().u(1, 1)                             # immersive out
    w.u(1, 1).u(5, 9).u(1, 0)                    # loro 3 dB, no ltrt
    w.u(1, 1).u(5, 31)                           # 5_X: 0 dB (reserved)
    w.u(1, 0).u(1, 1).u(5, 17)                   # no 5_X_2, 7_X -1 dB
    w.u(1, 0).u(1, 0).u(1, 1).u(5, 0)            # 5_X_4: 7.5 dB
    w.u(1, 0)                                    # core_5_X_2
    w.u(1, 0).u(1, 1).u(5, 15).u(5, 16)          # core loro / ltrt
    b = Bits(w.align().bytes())
    lc = MD.loud_corr(b, 12, 6, False)
    assert lc == {"loro": 3.0, "5_X": 0.0, "7_X": -1.0, "5_X_4": 7.5,
                  "core_loro": 0.0, "core_ltrt": -0.5}
    assert MD.pres_modes_of([dict(channel_mode=12)])[1] == 6
    assert MD.pres_modes_of([dict(channel_mode=4)])[1] == -1


def test_loud_corr_applied():
    """A downmix takes the held correction of its output configuration;
    a layout that drops nothing does not."""
    from ac4 import render as RD
    ch = ("L", "R", "C", "lfe", "Ls", "Rs")
    r = RD.Renderer(ch, "2.0")
    y = np.ones((4, 6))
    o0 = r.process(y, [None], 4)
    o1 = r.process(y, [dict(loud_corr={"loro": -6.0})], 4)
    o2 = r.process(y, [None], 4)
    assert np.allclose(o1, o0 * 10 ** (-6 / 20)) and np.allclose(o2, o1)
    r = RD.Renderer(ch, "5.1")
    assert np.allclose(r.process(y, [dict(loud_corr={"5_X": -6.0})], 4), y)


@pytest.mark.parametrize("dmx", ["ltrt", "pl2", "auto"])
def test_c_stereo_dmx(tmp_path, dmx):
    """--stereo-dmx in the C decoder against the Python, on a synthetic
    5.1 stream."""
    import os
    import subprocess
    from ac4.decoder import Decoder
    from test_inputs import syncframe
    from test_phase_b2 import _simple_5x
    cli = os.environ.get("AC4_C_CLI", C_CLI)
    if not os.path.exists(cli):
        pytest.skip("c/ac4dec-c not built")
    rng = np.random.default_rng(9)
    frames = [EW.frame(_simple_5x(rng, 3, 0, 6)[0], seq=5 + i)
              for i in range(6)]
    src = tmp_path / "e.ac4"
    src.write_bytes(b"".join(syncframe(f) for f in frames))
    raw = tmp_path / "e.f64"
    subprocess.run([cli, str(src), str(tmp_path / "e.wav"),
                    "--legacy-level", "--f64", str(raw), "--internal",
                    "--no-guard", "-q", "--layout", "2.0", "--stereo-dmx",
                    dmx], check=True)
    d = Decoder("5_X", guard=False, layout="2.0", stereo_dmx=dmx)
    d.hf_ratio = None
    ref = d.decode_pass(frames)
    got = np.fromfile(raw, "<f8").reshape(-1, 2)
    assert np.abs(got - ref).max() <= 1e-9 * np.abs(ref).max()


# ------------------------------------------------- dialogue enhancement
def test_de_cross_channel_matrix():
    """5.7.8.8: H = I + g r p^T with p from Table 210 and r from Table
    172 (the last coefficient energy preserving)."""
    from ac4 import de as DE
    par = [[30, -30, 5, 0, 10, 0, 0, 0], [-3] * 8, [7] * 8]
    e = dict(config=dict(method=1, max_gain=3, channel_config=0b111),
             par=par, ms=0, mix=[20, 8])
    g = 10 ** (6.0 / 20) - 1
    m = DE.frame_gains(e, 6.0)
    c1, c2 = 0.929, 0.156
    r = [c1, c2, np.sqrt(1 - c1 * c1 - c2 * c2)]
    p = [np.array([v / 10 for v in row])[DE.SB_TO_BAND] for row in par]
    assert m["xc_ch"] == ("L", "R", "C")
    for a in range(3):
        for b in range(3):
            assert np.allclose(m["xc"][a, b], (a == b) + g * r[a] * p[b])
    # one channel: r = 1; two without de_mix_coef: identity
    e1 = dict(e, config=dict(method=1, max_gain=3, channel_config=1))
    assert np.allclose(DE.frame_gains(e1, 6.0)["xc"][0, 0], 1 + g * p[0])
    e2 = dict(e, config=dict(method=3, max_gain=3, channel_config=6),
              mix=None)
    assert DE.frame_gains(e2, 6.0) == {}
    # method 2: its parametric part, channel independent
    e3 = dict(e, config=dict(method=2, max_gain=3, channel_config=1),
              par=[[20] * 8, [0] * 8, [0] * 8])
    assert np.allclose(DE.frame_gains(e3, 6.0)["C"], 1 + g * DE.PAR_CI[20])


def test_de_cross_channel_apply():
    """The cross-channel matrix mixes the channels of one call; a call
    without all its channels leaves them untouched."""
    from ac4 import de as DE
    rng = np.random.default_rng(4)
    n = 24
    QL = rng.normal(size=(64, 3 * n)) + 1j * rng.normal(size=(64, 3 * n))
    QR = rng.normal(size=(64, 3 * n)) + 0j
    L0, R0 = QL.copy(), QR.copy()
    e = dict(config=dict(method=1, max_gain=3, channel_config=0b110),
             par=[[12] * 8, [-4] * 8, [0] * 8], ms=0, mix=[16])
    DE.apply({"L": QL, "R": QR}, [e] * 3, 6.0, n, 0)
    m = DE.frame_gains(e, 6.0)["xc"]
    s = slice(n, 2 * n)
    assert np.allclose(QL[:, s], m[0, 0][:, None] * L0[:, s]
                       + m[0, 1][:, None] * R0[:, s])
    assert np.allclose(QR[:, s], m[1, 0][:, None] * L0[:, s]
                       + m[1, 1][:, None] * R0[:, s])
    Q = L0.copy()
    DE.apply({"L": Q}, [e] * 3, 6.0, n, 0)
    assert np.array_equal(Q, L0)


def _de_metadata(method, cc, par, mix=None, ms=0, contrib=None):
    """metadata() (6.2.7.1, sus_ver 1) of an I-frame with dialogue
    enhancement: basic and extended metadata empty, the tools block
    with de_config and de_data (Huffman coded, absolute then
    differential)."""
    from ac4 import metadata as MD
    t = BW()
    t.u(1, 1)                                    # b_de_data_present
    t.u(2, method).u(2, 3).u(3, cc)              # de_config
    n = MD.DE_NR_CHANNELS[cc]
    if method in (1, 3) and n > 1:
        for v in mix[:n - 1]:
            t.u(5, v)
    if method in (0, 2) and n == 2:
        t.u(1, ms)
    k = method % 2
    bk = MD.books()
    ref = None
    for ch in range(n - ms):
        for band in range(8):
            v = par[ch][band]
            if ref is None:
                EW.huff_put(t, bk[f"DE_HCB_ABS_{k}"],
                            v + MD.CB_OFF[f"DE_HCB_ABS_{k}"])
            else:
                EW.huff_put(t, bk[f"DE_HCB_DIFF_{k}"],
                            v - ref + MD.CB_OFF[f"DE_HCB_DIFF_{k}"])
            ref = v
        ref = par[ch][0]
    if method >= 2:
        t.u(5, contrib)
    return t


@pytest.mark.parametrize("method,cc,ms", [(1, 0b110, 0), (2, 0b110, 1),
                                          (1, 0b001, 0), (3, 0b100, 0),
                                          (1, 0b111, 0)])
def test_c_de_methods(tmp_path, method, cc, ms):
    """DE methods 1 - 3 on a SIMPLE 5_X stream at 9 dB, C against the
    Python; and the output differs from the decode without DE where the
    matrix is applied."""
    import os
    import subprocess
    from ac4.decoder import Decoder
    from test_inputs import syncframe
    from test_phase_b2 import _simple_5x
    cli = os.environ.get("AC4_C_CLI", C_CLI)
    if not os.path.exists(cli):
        pytest.skip("c/ac4dec-c not built")
    rng = np.random.default_rng(31 + method)
    par = [[12, 14, 10, 9, 11, 13, 12, 8], [-5, -4, -6, -5, -3, -2, -4, -5],
           [3] * 8]
    if method in (0, 2):
        par = [[abs(v) % 32 for v in row] for row in par]
    frames = []
    for i in range(6):
        sub, _ = _simple_5x(rng, 3, 0, 6)
        t = _de_metadata(method, cc, par, mix=[18, 9], ms=ms, contrib=12)
        frames.append(EW.frame(_with_md(sub, t), seq=5 + i, iframe=1))
    src = tmp_path / "e.ac4"
    src.write_bytes(b"".join(syncframe(f) for f in frames))
    raw = tmp_path / "e.f64"
    subprocess.run([cli, str(src), str(tmp_path / "e.wav"),
                    "--legacy-level", "--f64", str(raw), "--internal",
                    "--no-guard", "-q", "--de-level", "9"], check=True)
    ys = []
    for lvl in (0.0, 9.0):
        d = Decoder("5_X", guard=False, de_level=lvl)
        d.hf_ratio = None
        ys.append(d.decode_pass(frames))
        assert d.n_concealed == 0
    got = np.fromfile(raw, "<f8").reshape(-1, 6)
    assert np.abs(got - ys[1]).max() <= 1e-9 * np.abs(ys[1]).max()
    changed = np.abs(ys[1] - ys[0]).max() > 1e-6 * np.abs(ys[0]).max()
    assert changed == (cc != 0b111), cc


def _with_md(sub, tools):
    """An element (audio_size + body) followed by metadata(): basic and
    extended metadata empty (4 bits), the tools block's size and bits,
    no EMDF payloads."""
    from ac4.toc import Bits
    b = Bits(sub)
    asz = b.u(15)
    if b.u(1):
        asz += b.vb(7) << 15
    body = sub[:(b.p + 7) // 8 + asz]
    w = BW().u(1, 0).u(3, 0)                     # basic, extended
    w.u(7, len(tools.bits)).u(1, 0)              # tools size
    w.bits.extend(tools.bits)
    w.u(1, 0)                                    # no EMDF payloads
    return body + w.align().bytes()


# ------------------------------------------------- mixing-gain changes
def _sg_frames(sgs):
    from test_mix import pair_sub, mono_sub, pres_sub
    from test_phase_a import build_toc
    rng = np.random.default_rng(17)
    frames = []
    for i, sg in enumerate(sgs):
        a, b = pair_sub(rng), mono_sub(rng)
        ps = pres_sub(sg=sg)
        toc = build_toc(pres=[dict(groups=[0, 1], pres_sub=0)],
                        groups=[dict(ch_mode_code="10", index=1, cc=1),
                                dict(ch_mode_code="0", index=2, cc=4)],
                        sizes=[len(ps), len(a), len(b)], seq=5 + i)
        frames.append(toc + ps + a + b)
    return frames


def test_mix_gain_change_crossfades(tmp_path):
    """A substream group gain that changes crossfades over the frame:
    (1 - w) old + w new, w = (n + 1/2) / N; C against the Python."""
    from ac4 import mix as MX
    from ac4.decoder import Decoder
    from test_mix import _c_mix
    sgs = [(0, 0), (0, 0), (0, 24), (0, 24), (12, 24), (12, 24)]
    frames = _sg_frames(sgs)
    mx = MX.MixDecoder(MX.probe(frames), guard=False)
    y = mx.decode_pass(frames)
    ya = Decoder("pair", guard=False, substream=0).decode_pass(frames)
    yb = Decoder("mono", guard=False, substream=1).decode_pass(frames)
    F = 1536
    w = (np.arange(F) + 0.5) / F
    g1 = [10 ** (-v[1] * 0.25 / 20) for v in sgs]
    g0 = [10 ** (-v[0] * 0.25 / 20) for v in sgs]
    for f in (2, 3, 4):
        s = slice(f * F, (f + 1) * F)
        e1 = g1[f - 1] + (g1[f] - g1[f - 1]) * w
        e0 = g0[f - 1] + (g0[f] - g0[f - 1]) * w
        exp = (ya[s] * e0[:, None] + yb[s][:, [0, 0]] * 0.5 * e1[:, None]) / 2
        assert np.abs(y[s] - exp).max() <= 1e-12 * np.abs(exp).max(), f
    got = _c_mix(tmp_path, frames).reshape(-1, 2)
    assert np.abs(got - y).max() <= 1e-9 * np.abs(y).max()
