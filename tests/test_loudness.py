# SPDX-License-Identifier: Apache-2.0
"""Loudness normalisation and DRC (TS 103 190-1 5.7.9)."""
import numpy as np

from ac4 import loudness as L


def _profile(n):
    pts, tc = L.curve_points({"default_profile": 1}, n)
    return pts, tc


def test_film_standard_curve_is_table_162():
    p, tc = _profile(1)
    g = lambda x: L.curve_gain(x, p)                              # noqa: E731
    # null band 0..5 dB, max boost 6 dB below -12, cut section to 15 dB
    # (-5 dB there), max cut -24 dB from 35 dB
    assert g(2.0) == 0.0 and g(0.0) == 0.0 and g(5.0) == 0.0
    assert g(-30.0) == 6.0 and g(-12.0) == 6.0
    assert abs(g(-6.0) - 3.0) < 1e-12
    assert abs(g(15.0) + 5.0) < 1e-12 and abs(g(10.0) + 2.5) < 1e-12
    assert abs(g(25.0) + 14.5) < 1e-12
    assert g(35.0) == -24.0 and g(60.0) == -24.0
    assert tc[:4] == (100, 3000, 10, 1000)


def test_profile_none_and_music_light():
    assert L.curve_points({"default_profile": 1}, 0) is None
    p, _ = _profile(4)          # no cut section: straight to -15 at 40
    assert abs(L.curve_gain(25.0, p) + 7.5) < 1e-12


def test_transmitted_curve_table_166():
    c = dict(nullband_low=3, nullband_high=2, gain_max_boost=6,
             lev_max_boost=9, nr_boost_sections=0, gain_max_cut=20,
             lev_max_cut=19, nr_cut_sections=1, gain_section_cut=4,
             lev_section_cut=9, tc_default=1)
    p, tc = L.curve_points({"curve": c}, 0)
    assert (p["l0l"], p["l0h"], p["lmb"], p["lsc"], p["gsc"], p["lmc"],
            p["gmc"]) == (-3, 2, -13, 12, -5, 32, -20)
    assert tc == L.DEFAULT_TC
    assert L.curve_gain(-20, p) == 6.0 and L.curve_gain(40, p) == -20.0
    assert abs(L.curve_gain(-8, p) - 3.0) < 1e-12


def test_mode_selection_5_7_9_2():
    cfg = {"modes": [{"id": 0, "out_range": (-31, -27)},
                     {"id": 1, "out_range": (-26, -17)},
                     {"id": 2, "out_range": (-16, 0)},
                     {"id": 3, "out_range": (-16, 0)},
                     {"id": 5, "out_range": (-20, -30)}]}
    assert L.select_mode(cfg, -24)["id"] == 5
    assert L.select_mode(cfg, -28)["id"] == 5
    assert L.select_mode(cfg, -14)["id"] == 3
    assert L.select_mode(cfg, -26.5) is None or \
        L.select_mode(cfg, -26.5)["id"] == 5


def test_normalisation_gain_and_drc_off():
    n = 1536 * 3
    y = np.full((n, 2), 0.25 * L.FULL_SCALE)
    lv = L.Leveler(("L", "R"), target=-24.0, drc="off")
    out = lv.process(y, [{"dialnorm": -18.0, "drc": None}] + [None] * 2,
                     1536)
    assert np.allclose(out, 0.25 * 2.0 ** (-6.0 / 6.0))


def test_transmitted_gains_channel_groups():
    cfg = {"modes": [{"id": 1, "out_range": (-26, -17), "curve_flag": 0,
                      "gains_config": 1}], "eac3_profile": 0}
    gains = [[[-6]] * 6, [[0]] * 6, [[6]] * 6]
    md = {"dialnorm": -24.0, "drc": {"config": cfg, "gains": {1: gains},
                                     "reset": None}}
    chans = ("L", "R", "C", "lfe", "Ls", "Rs")
    y = np.full((1536, 6), L.FULL_SCALE)
    out = L.Leveler(chans, -24.0).process(y, [md], 1536)
    assert np.allclose(out[0], [0.5, 0.5, 1.0, 0.5, 2.0, 2.0])


def test_curve_drc_compresses_loud_programme():
    rng = np.random.default_rng(1)
    n = 1536 * 60
    y = rng.standard_normal((n, 2)) * 0.5 * L.FULL_SCALE
    cfg = {"modes": [{"id": 1, "out_range": (-26, -17), "curve_flag": 1,
                      "default_profile": 1}], "eac3_profile": 1}
    md = [{"dialnorm": -31.0, "drc": {"config": cfg, "gains": {},
                                       "reset": 0}}] + [None] * 59
    off = L.Leveler(("L", "R"), -24.0, "off").process(y, md, 1536)
    on = L.Leveler(("L", "R"), -24.0, "auto").process(y, md, 1536)
    # K-weighted white noise at -0.5 LKFS, 30.5 dB above dialnorm: the
    # curve's cut there is -5 - 19 * 15.5 / 20 = -19.7 dB
    r = 20 * np.log10(np.abs(on[-1536:]).mean() / np.abs(off[-1536:]).mean())
    assert -22.0 < r < -18.0
