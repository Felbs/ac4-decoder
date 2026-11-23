# SPDX-License-Identifier: Apache-2.0
"""Damaged input: the error-injection gate.

Without captures (always run): garbage, empty and truncated frames, the
MP4 reader on damaged boxes, the output guard and the sequence rules.

With captures (AC4_TEST_DIR, see test_integration.py), each capture is cut
to its first AC4_TEST_FRAGMENTS fragments and damaged deterministically
(tests/inject.py: bit flips, byte bursts, truncation, zeroed bytes,
swapped, repeated, extended, empty and garbage frames, frames from
another stream, runs of 3..20 damaged frames in a row, and a damaged TOC in
the first frame) with AC4_ROBUST_SEEDS seeds per kind (default 3). The
gate, for every damaged decode:

  * no exception escapes Decoder.decode_pass;
  * the output has exactly 1536 samples per input frame;
  * no sample is above 2x the clean decode's own peak;
  * at most MAX_COLLATERAL undamaged frames per hit are concealed;
  * every damaged frame is concealed, or provably harmless: frames
    i-1 .. i+8 equal the clean decode ("identical"), or no sample in them
    is above 2x the clean decode's own peak ("bounded"; inject.assess,
    which also reports how loud such a frame is against its local clean
    level).

And on the clean cuts: nothing is concealed and the output guard never
changes a sample. (Byte identity with the original receiver is
test_integration.test_identical_to_reference_worker.)

NEGATIVE CONTROL: with the bitstream checks and the output guard switched
off (robust.CHECKS = False, Decoder(guard=False)), the same gate must FAIL
on the same damaged inputs.
"""
import os

import numpy as np
import pytest

import capture_util as CU
import inject as I
from ac4 import mp4
from ac4 import robust as RB
from ac4.decoder import Decoder, probe_element

FRAME = 1536
CAPTURES = CU.capture_files()
N_FRAG = int(os.environ.get("AC4_TEST_FRAGMENTS", "2"))
N_SEEDS = int(os.environ.get("AC4_ROBUST_SEEDS", "3"))
MAX_COLLATERAL = RB.RELEARN + 1     # undamaged frames concealed per hit


# ------------------------------------------------------- no captures needed
def test_garbage_frames_never_raise_and_keep_length():
    rng = np.random.default_rng(7)
    frames = [b"", b"\x00", b"\xff" * 3]
    frames += [rng.integers(0, 256, int(rng.integers(4, 2000)),
                            dtype=np.uint8).tobytes() for _ in range(40)]
    for element in ("5_X", "pair"):
        d = Decoder(element)
        y = d.decode_pass(frames)
        assert y.shape == (len(frames) * FRAME, len(d.chans))
        assert np.isfinite(y).all()
        assert d.n_concealed == len(frames)
        assert sum(d.concealed.reasons.values()) == len(frames)
        y2 = d.decode_pass([])
        assert y2.shape == (0, len(d.chans))


def test_mp4_reader_survives_damaged_boxes():
    from test_units import _box, _fragment, _trak
    s1 = [bytes([i]) * (10 + i) for i in range(5)]
    init = _box(b"moov", _trak(1, b"ac-4"))
    data = init + _fragment(s1) + _fragment(s1)
    rng = np.random.default_rng(3)
    for _ in range(400):
        b = bytearray(data)
        for _ in range(int(rng.integers(1, 6))):
            j = int(rng.integers(0, len(b)))
            b[j] = int(rng.integers(0, 256))
        out = mp4.frames(bytes(b))              # must not raise or hang
        assert all(isinstance(f, bytes) for f in out)


def test_mp4_huge_sample_count_is_rejected():
    from test_units import _box, _full, _trak
    tfhd = _full(b"tfhd", 0, 0x20000, (1).to_bytes(4, "big"))
    body = (0xFFFFFFF0).to_bytes(4, "big") + (0).to_bytes(4, "big")
    moof = _box(b"moof", _box(b"traf", tfhd + _full(b"trun", 0, 0x201, body)))
    data = _box(b"moov", _trak(1, b"ac-4")) + moof + _box(b"mdat", b"x" * 9)
    assert mp4.frames(data) == []


def test_mp4_sample_outside_mdat_keeps_its_place():
    from test_units import _fragment
    s1 = [b"a" * 5, b"b" * 6, b"c" * 7]
    data = bytearray(_fragment(s1))
    # the second sample's size (in the trun table) made far too large
    i = data.index((6).to_bytes(4, "big") + (7).to_bytes(4, "big"))
    data[i:i + 4] = (600).to_bytes(4, "big")
    out = mp4.frames(bytes(data))
    assert len(out) == 3 and out[0] == s1[0] and out[1] == b"" \
        and out[2] == b""


def test_sequence_rules():
    assert RB.seq_continues(5, 4)
    assert RB.seq_continues(1, 1020)
    assert RB.seq_continues(7, 0)              # after a splice
    assert RB.seq_continues(3, None)
    assert not RB.seq_continues(6, 4)
    assert not RB.seq_continues(4, 4)
    assert RB.seq_next(1020) == 1 and RB.seq_prev_of(1) == 1020
    # bits 2..11 of the frame
    assert RB.peek_counter(bytes([0b10000011, 0b11110000])) == 0x3F


def test_output_guard_mutes_a_burst_and_ramps():
    rng = np.random.default_rng(1)
    g = RB.OutputGuard(1.0)
    y = rng.standard_normal((20 * FRAME, 2)) * 0.2
    y0 = y.copy()
    y[10 * FRAME + 100] = 50.0                  # one frame far too loud
    fired = g.process(y)
    assert list(np.flatnonzero(fired)) == [10]
    assert np.abs(y[10 * FRAME:11 * FRAME]).max() <= np.abs(y0).max()
    assert np.array_equal(y[:10 * FRAME], y0[:10 * FRAME])
    # ramp back in, then untouched
    assert np.array_equal(y[11 * FRAME + g.RAMP:], y0[11 * FRAME + g.RAMP:])
    assert np.abs(y[11 * FRAME:11 * FRAME + g.RAMP]).max() <= \
        np.abs(y0[11 * FRAME:11 * FRAME + g.RAMP]).max()


def test_output_guard_judges_the_frames_after_a_muted_frame_at_2x():
    """Every channel is 577 samples late, so a damaged frame's core reaches
    into the second output frame after it. A frame within 3 after one the
    guard muted is judged like one near a concealed frame: at 2x the
    stream level. Here frames 10-11 are far too loud (muted) and frame 12
    peaks at 2.5x the stream: muted too (before: let through, under 3x)."""
    rng = np.random.default_rng(2)
    g = RB.OutputGuard(0.01)
    y = np.clip(rng.standard_normal((40 * FRAME, 2)) * 0.2, -0.6, 0.6)
    y[0, 0] = 1.0                               # the stream level, 1.0
    for f in (10, 11, 30, 31):
        y[f * FRAME + 50] = 50.0
    y[12 * FRAME + 10] = 2.5
    y[35 * FRAME + 10] = 1.5                    # within 2x: untouched
    fired = g.process(y)
    assert list(np.flatnonzero(fired)) == [10, 11, 12, 30, 31]


def test_output_guard_does_not_lock_after_a_long_quiet_stretch():
    """Loud, then 6 minutes far quieter, then loud again: the return is
    programme, not damage (a guard that only releases slowly muted it for
    good). Fed in 300-frame passes, as a caller might."""
    g = RB.OutputGuard(1.0)

    def feed(level, n):
        out = []
        for i in range(0, n, 300):
            m = min(300, n - i)
            y = (np.sin(0.07 * np.arange(m * FRAME)) * level)[:, None]
            out.append(g.process(np.repeat(y, 2, axis=1)))
        return np.concatenate(out)

    assert not feed(0.9, 900).any()
    assert not feed(0.3, int(6 * 60 * 29.97)).any()
    assert not feed(0.9, 900).any()


def test_output_guard_is_not_ratcheted_by_one_loud_frame():
    """One accepted frame well above the stream level raises the
    reference by at most UP x (here 2x, not the 2.8x of the frame), so a
    louder frame after it is still caught."""
    g = RB.OutputGuard(1.0)
    n = 400
    y = np.repeat((np.sin(0.05 * np.arange(n * FRAME)) * 0.1)[:, None],
                  2, axis=1)
    g.init = 0.0                       # judge against the stream level
    y[300 * FRAME + 7] = 0.28          # 2.8x: below RATIO, accepted
    y[340 * FRAME + 7] = 0.8           # 8x: must still fire
    fired = g.process(y)
    assert not fired[300] and fired[340]
    assert g.ref < 0.2


def test_output_guard_ignores_a_steady_crescendo():
    g = RB.OutputGuard(1.0)
    t = np.arange(300 * FRAME)
    env = np.minimum(1.0, 0.01 * 1.02 ** (t / FRAME))    # +0.17 dB / frame
    y = (np.sin(t * 0.05) * env)[:, None] * np.ones((1, 2))
    y0 = y.copy()
    assert not g.process(y).any()
    assert np.array_equal(y, y0)


# ------------------------------------------------------- with captures
pytestmark_caps = pytest.mark.skipif(
    not CAPTURES, reason="set AC4_TEST_DIR to a folder of AC-4 fMP4 captures")

_CACHE = {}


def _load(tmpdir_factory):
    """-> list of (frames, element, clean decode) for every capture."""
    if "all" in _CACHE:
        return _CACHE["all"]
    out = []
    d = tmpdir_factory.mktemp("cuts")
    for i, path in enumerate(CAPTURES):
        dst = str(d.join(f"c{i}.m4s"))
        if CU.cut_fragments(path, dst, N_FRAG) == 0:
            continue
        frames = mp4.read_frames(dst)
        el = probe_element(frames)
        if el is None:
            continue
        out.append((frames, el))
    _CACHE["all"] = out
    return out


def _clean(frames, el, guard=True):
    key = (id(frames), guard)
    if key not in _CACHE:
        d = Decoder(el, guard=guard)
        _CACHE[key] = (d.decode_pass(frames), d)
    return _CACHE[key]


def _foreign(caps, frames, el):
    for f2, e2 in caps:
        if f2 is not frames and e2 == el:
            return f2
    return None


def _run_gate(caps, kinds, seeds, guard):
    """-> (failures (str), worst peak ratio, worst local ratio of a frame
    that was neither concealed nor identical, verdict counts).

    Also counted (tally["collateral"]): undamaged frames concealed. A hit
    may cost its neighbours (a swapped pair, the frame after a damaged
    sequence counter), but a decode that conceals more than
    MAX_COLLATERAL undamaged frames per hit is a failure: that is the
    shape of a decoder that locked onto a damaged frame's layout and muted
    the rest."""
    import collections
    fails, worst, wloc = [], (0.0, None), (0.0, None)
    tally = collections.Counter()
    for ci, (frames, el) in enumerate(caps):
        clean, _ = _clean(frames, el, guard=guard)
        foreign = _foreign(caps, frames, el)
        for kind in kinds:
            for s in range(seeds):
                seed = 7919 * s + I.KINDS.index(kind)
                dfr, hits = I.damage(frames, kind, seed, foreign=foreign)
                d = Decoder(el, guard=guard)
                tag = f"capture{ci} {kind} seed {seed}"
                try:
                    y = d.decode_pass(dfr)
                except Exception as e:                         # noqa: BLE001
                    fails.append(f"{tag}: raised {e!r}")
                    continue
                if y.shape != clean.shape:
                    fails.append(f"{tag}: shape {y.shape} != {clean.shape}")
                    continue
                a = I.assess(clean, y, hits, d.last_concealed)
                coll = int(d.last_concealed.sum()) - int(
                    d.last_concealed[hits].sum())
                tally["collateral"] += coll
                if coll > MAX_COLLATERAL * max(1, len(hits)):
                    fails.append(f"{tag}: {coll} undamaged frames "
                                 f"concealed ({dict(d.concealed.reasons)})")
                if a["peak_ratio"] > worst[0]:
                    worst = (a["peak_ratio"], tag)
                if a["peak_ratio"] > 2.0:
                    fails.append(f"{tag}: peak {a['peak_ratio']:.3g}x clean")
                for i, v, loc, lev in a["per_hit"]:
                    tally[v] += 1
                    if v == "bounded" and loc > wloc[0]:
                        wloc = (loc, f"{tag} frame {i}")
                    if v == "LOUD":
                        fails.append(f"{tag}: frame {i} not concealed, "
                                     f"{lev:.3g}x the clean stream's peak")
    return fails, worst, wloc, tally


@pytestmark_caps
def test_clean_captures_conceal_nothing_and_guard_is_silent(
        tmpdir_factory):
    for frames, el in _load(tmpdir_factory):
        y, d = _clean(frames, el, guard=True)
        y0, _ = _clean(frames, el, guard=False)
        assert d.n_concealed == 0, dict(d.concealed.reasons)
        assert d.guard.n_fired == 0
        assert np.array_equal(y, y0)


@pytestmark_caps
@pytest.mark.parametrize("kind", I.KINDS)
def test_damaged_frames_are_concealed_or_harmless(kind, tmpdir_factory):
    caps = _load(tmpdir_factory)
    fails, worst, wloc, tally = _run_gate(caps, [kind], N_SEEDS, guard=True)
    print(f"\n{kind}: {dict(tally)}; worst peak {worst[0]:.3g}x clean "
          f"({worst[1]}); worst unflagged frame {wloc[0]:.3g}x its local "
          f"clean level ({wloc[1]})")
    assert not fails, "\n".join(fails[:20])


@pytestmark_caps
def test_negative_control_old_behaviour_fails_the_gate(tmpdir_factory,
                                                       monkeypatch):
    """Checks and guard off -> the same gate must fail."""
    caps = _load(tmpdir_factory)
    monkeypatch.setattr(RB, "CHECKS", False)
    fails, worst, _, _ = _run_gate(caps, ["zero", "burst", "foreign"],
                                   N_SEEDS, guard=False)
    print(f"negative control: {len(fails)} failures, worst peak "
          f"{worst[0]:.3g}x clean ({worst[1]})")
    assert fails, "the gate passed with the checks and guard off"


def _first_iframe(frames, start=0):
    from ac4 import toc as M
    for i in range(start, len(frames)):
        try:
            if M.parse(frames[i])["b_iframe_global"]:
                return i
        except Exception:                                      # noqa: BLE001
            pass
    return None


@pytestmark_caps
def test_splice_to_another_stream_recovers(tmpdir_factory):
    """Stream A, then stream B (same element, counter left as is: an
    uncontrolled source change, TS 103 190-1 4.3.3.2.2). From just after
    B's first I-frame on, nothing is concealed, and no sample is above 2x the louder clean stream."""
    caps = _load(tmpdir_factory)
    n_pairs = 0
    for ia, (fa, ea) in enumerate(caps):
        for ib, (fb, eb) in enumerate(caps):
            if ia == ib or ea != eb or len(fa) < 60 or len(fb) < 120:
                continue
            k = 60
            sp = fa[:k] + fb[k:]
            d = Decoder(ea)
            y = d.decode_pass(sp)
            assert y.shape[0] == len(sp) * FRAME
            i0 = _first_iframe(fb, k)
            # (before B's first I-frame its frames need B's A-SPX
            # configuration, which only an I-frame carries; the output
            # guard judges the NEAR frames after concealed ones strictly,
            # so a louder stream B may lose those too)
            settle = (i0 + 1 + RB.OutputGuard.NEAR if i0 is not None
                      else len(sp))
            tail = d.last_concealed[settle:]
            assert not tail.any(), (ia, ib, int(tail.sum()),
                                    dict(d.concealed.reasons))
            ca, _ = _clean(fa, ea)
            cb, _ = _clean(fb, eb)
            top = max(np.abs(ca).max(), np.abs(cb).max())
            assert np.abs(y).max() <= 2.0 * top
            n_pairs += 1
    if not n_pairs:
        pytest.skip("needs two captures of the same element")


@pytestmark_caps
def test_cli_reports_concealed_frames(tmp_path, capsys):
    """Garbage written over 5 samples inside a real file: the CLI still
    writes a WAV of the exact length and reports 5 concealed frames."""
    from ac4 import cli
    src = CAPTURES[0]
    cut = str(tmp_path / "in.m4s")
    if CU.cut_fragments(src, cut, N_FRAG) == 0:
        pytest.skip("capture has no complete fragment")
    data = bytearray(open(cut, "rb").read())
    tops = list(mp4.boxes(bytes(data)))
    samples = []
    for k, (o, sz, t) in enumerate(tops):
        if t == b"moof" and k + 1 < len(tops):
            ids = mp4.ac4_track_ids(bytes(data))
            samples += mp4.fragment_samples(bytes(data), o, sz,
                                            ids[0] if ids else None) or []
    n = len(mp4.frames(bytes(data)))
    rng = np.random.default_rng(5)
    for i in (10, 20, 30, 40, 50):
        p, s = samples[i]
        data[p:p + s] = rng.integers(0, 256, s, dtype=np.uint8).tobytes()
    bad = str(tmp_path / "bad.m4s")
    open(bad, "wb").write(bytes(data))
    out = str(tmp_path / "out.wav")
    assert cli.main([bad, out]) == 0
    text = capsys.readouterr().out
    n_ch, rate, raw = CU.wav_data(out)
    assert len(raw) // (2 * n_ch) == -(-n * 1536 * 1001 // 960)
    assert "5 concealed" in text, text


def test_late_closure_counts_an_already_concealed_frame_once(monkeypatch):
    """A frame put on the pending list and then concealed for another
    reason keeps that reason: _late_closure neither re-marks it nor counts
    it in n_bad a second time (nor takes it off n_ok, which never had
    it); a frame that was accepted and does not close is concealed."""
    from ac4 import stream as ST
    s = ST.Ac4Stream("pair")
    s.cfg = {"stub": 1}
    monkeypatch.setattr(ST.R37, "t_for", lambda cfg: {})

    def fake_aspx(b, tabs, t, cfg, nats, ifr, two, state, gi):
        if bytes(b.d[:3]) == b"BAD":
            b.p += 100                  # runs past audio_size: no close
    monkeypatch.setattr(ST.P, "aspx_data", fake_aspx)
    n = 20
    bad_already, bad_late = 5, 9
    pending, wins, lengths, counts = [], {}, {}, {}
    for k in ("L", "R"):
        wins[k] = [np.ones(1536) for _ in range(n)]
        lengths[k] = [1536] * n
        counts[k] = [1] * n
    for fi in range(n):
        sub = b"BAD" if fi in (bad_already, bad_late) else b"\0\0\0"
        pending.append(dict(fi=fi, sub=sub + bytes(13), bits=0, ifr=False,
                            start=0, size=0, acpl3=False, five=False,
                            wpos={"L": fi, "R": fi}))
    groups = {"lr": [None] * n, "sr": [None] * n}
    s.status = [None] * n
    s.status[bad_already] = "asf.level"
    s.n_ok, s.n_bad = n - 1, 1
    s.damage.add("asf.level")
    s._late_closure(pending, wins, lengths, counts, groups)
    assert s.status[bad_already] == "asf.level"
    assert s.status[bad_late] == "audio_size.late"
    assert s.n_bad == 2 and s.n_ok == n - 2
    assert s.n_ok + s.n_bad == n
    assert s.damage.reasons["audio_size.late"] == 1
    assert groups["lr"][bad_late] is ST.CONCEALED
    assert not wins["L"][bad_late].any()
