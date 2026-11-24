# SPDX-License-Identifier: Apache-2.0
"""Phase B3 item 1 (no captures): the output guard's true-peak rules
(TS 103 190-1 4.3.12.3.17, robust.OutputGuard) and the burst limit."""
import math

import numpy as np

from ac4 import metadata as MD
from ac4 import robust as RB
from ac4.decoder import BURST_LIMIT, Decoder, REF_PEAK
from ac4.loudness import FULL_SCALE

F = 1536


def _frames(peaks):
    """One channel, frame f a sine burst peaking at peaks[f]."""
    t = np.arange(F)
    y = np.concatenate([p * np.sin(2 * np.pi * 7 * t / F) for p in peaks])
    return y[:, None].copy()


def _db(x):
    return 20 * math.log10(x / FULL_SCALE)


def test_truepk_rule_mutes_what_the_level_rule_lets_through():
    # a quiet pair stream: frames at 1e3, truepk saying so (-62 dBTP);
    # one damaged frame at 3e4: under 3x the start level (0.75 x 2e5)
    peaks = [1e3] * 60
    peaks[40] = 3e4
    tp = [_db(1e3)] * 60
    g = RB.OutputGuard(REF_PEAK["pair"])
    fired = g.process(_frames(peaks), truepk=tp)
    assert list(np.flatnonzero(fired)) == [40]
    assert g.last_truepk[40] and g.n_truepk == 1
    g = RB.OutputGuard(REF_PEAK["pair"])            # without truepk
    assert not g.process(_frames(peaks)).any()


def test_truepk_rule_spares_programme_within_its_true_peak():
    # a loud transient the stream announces: not damage
    peaks = [1e3] * 60
    peaks[40] = 3e4
    tp = [_db(1e3)] * 60
    tp[41] = _db(3e4)                    # sent a frame late: in the window
    g = RB.OutputGuard(REF_PEAK["pair"])
    assert not g.process(_frames(peaks), truepk=tp).any()


def test_near_damage_rule_takes_the_true_peak_into_account():
    # a frame next to a concealed one, 2.5x the stream level: muted by the
    # near-damage rule unless the stream's truepk covers it
    peaks = [1e4] * 60
    peaks[41] = 2.5e4
    force = np.zeros(60, dtype=bool)
    force[40] = True
    g = RB.OutputGuard(REF_PEAK["pair"])
    assert g.process(_frames(peaks), force=force)[41]
    g = RB.OutputGuard(REF_PEAK["pair"])
    tp = [_db(1e4) + 3] * 60
    assert not g.process(_frames(peaks), force=force, truepk=tp)[41]
    # far above it, still muted
    peaks[41] = 2e5
    g = RB.OutputGuard(REF_PEAK["pair"])
    assert g.process(_frames(peaks), force=force, truepk=tp)[41]


def test_offset_tracks_a_stream_quieter_than_its_metadata():
    # decoded 17 dB under its truepk: the offset follows (the limits
    # tighten by as much), and stays at 0 for a stream above it
    peaks = [1e4] * 300
    g = RB.OutputGuard(REF_PEAK["pair"])
    g.process(_frames(peaks), truepk=[_db(1e4) + 17] * 300)
    assert abs(g.tp_off + 17) <= RB.OutputGuard.TP_STEP
    g = RB.OutputGuard(REF_PEAK["pair"])
    g.process(_frames(peaks), truepk=[_db(1e4) - 5] * 300)
    assert g.tp_off > 0 and g.tp_limit(0.0, 0.0) == FULL_SCALE


def test_truepk_window_crosses_passes():
    g = RB.OutputGuard(REF_PEAK["pair"])
    assert g.truepk_window([None, -10.0, None, None], 4) == \
        [-10.0, -10.0, -10.0, -10.0]
    # the last TP_WIN values are carried into the next pass
    assert g.truepk_window([None] * 4, 4) == [-10.0, None, None, None]


def test_truepk_is_read_from_further_loudness_info():
    from test_phase_a import BW
    w = BW().u(1, 0).u(7, 96).u(1, 1)        # dialnorm -24, loudness info
    w.u(2, 0).u(4, 0)                        # version 0, prac type 0
    w.u(1, 0).u(1, 0)                        # no loudrelgat / loudspchgat
    w.u(1, 0).u(1, 0)                        # no loudstrm3s / max
    w.u(1, 1).u(11, 1024 - 63)               # truepk -6.3 dBTP
    w.u(1, 0)                                # no max_truepk
    w.u(1, 0).u(1, 0).u(1, 0).u(1, 0)        # prgmbndy, lra, mntry x2
    w.u(1, 0).u(1, 0)                        # rtll_comp, extension
    w.u(5, 1).u(1, 0).u(1, 0)                # drc size 1, no drc
    w.u(8, 0)
    md = MD.presentation_substream(w.align().bytes(), 0, 0, 2,
                                   MD.DrcState(), 1, 1536)
    assert md["loudness"]["truepk"] == -6.3


def test_burst_limit_is_twice_full_scale():
    # it used to be 2 x REF_PEAK, which fired on clean frames of streams
    # louder than that reference (a public stereo test stream: 27 frames)
    assert BURST_LIMIT == 2 * FULL_SCALE
    for el in ("pair", "5_X"):
        assert Decoder(el).burst_limit == BURST_LIMIT > 2 * REF_PEAK[el]
