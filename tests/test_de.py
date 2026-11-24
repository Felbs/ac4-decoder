# SPDX-License-Identifier: Apache-2.0
"""Dialogue enhancement, de_method 0 (TS 103 190-1 5.7.8.3 - 5.7.8.7,
Tables 171, 173, 209), written out from the standard."""
import numpy as np

from ac4 import de as DE

# Table 209, by hand
T209 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
        6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
# Table 173: first, last QMF band of each parameter band
T173 = [(0, 0), (1, 1), (2, 3), (4, 6), (7, 10), (11, 16), (17, 26),
        (27, 40)]


def _entry(cc, par, max_gain=2, ms=0, method=0):
    return dict(config=dict(method=method, max_gain=max_gain,
                            channel_config=cc), par=par, ms=ms)


def test_tables():
    assert list(DE.PAR_CI) == T209
    for b, (lo, hi) in enumerate(T173):
        assert all(DE.SB_TO_BAND[sb] == b for sb in range(lo, hi + 1))
    assert all(DE.SB_TO_BAND[sb] == 7 for sb in range(41, 64))


def test_frame_gains_channel_config_and_gmax():
    par = [[3, 10, 17, 20, 31, 0, 1, 2], [5] * 8, [0] * 8]
    # 0b101: Left, Centre (Table 171), parameter subsets in that order
    m = DE.frame_gains(_entry(0b101, par, max_gain=1), 12.0)
    g = 10 ** (6.0 / 20) - 1                 # G_max = (1 + 1) 3 dB
    assert set(m) == {"L", "C"}
    for sb in range(64):
        b = DE.SB_TO_BAND[sb]
        assert np.isclose(m["L"][sb], 1 + g * T209[par[0][b]])
        assert np.isclose(m["C"][sb], 1 + g * T209[par[1][b]])
    g4 = 10 ** (4.0 / 20) - 1                # under G_max
    m = DE.frame_gains(_entry(0b001, par, max_gain=3), 4.0)
    assert np.isclose(m["C"][0], 1 + g4 * T209[3])
    assert DE.frame_gains(_entry(0b001, par), 0.0) == {}
    # method 1 (cross-channel, phase D): test_phase_d
    assert "xc" in DE.frame_gains(_entry(0b001, par, method=1), 6.0)
    assert DE.frame_gains(None, 6.0) == {}


def test_ms_processing():
    """de_ms_proc_flag: (1/2) [1 1; 1 -1] diag(1 + g p0, 1) [1 1; 1 -1]."""
    par = [[10] * 8, [0] * 8, [0] * 8]
    level = 6.0
    g = 10 ** (level / 20) - 1
    n = 24
    rng = np.random.default_rng(1)
    QL = rng.normal(size=(64, n * 3)) + 0j
    QR = rng.normal(size=(64, n * 3)) + 0j
    L0, R0 = QL.copy(), QR.copy()
    fr = [_entry(0b110, par, ms=1)] * 3
    DE.apply({"L": QL, "R": QR}, fr, level, n, 0)
    # frame 1 on: H(f - 1) = H(f): the full matrix
    s = slice(n, 2 * n)
    G = 1 + g * 1.0
    M = 0.5 * np.array([[1, 1], [1, -1]]) @ np.diag([G, 1]) \
        @ np.array([[1, 1], [1, -1]])
    assert np.allclose(QL[:, s], M[0, 0] * L0[:, s] + M[0, 1] * R0[:, s])
    assert np.allclose(QR[:, s], M[1, 0] * L0[:, s] + M[1, 1] * R0[:, s])


def test_interpolation_5_7_8_6():
    """H(n) = (1 - (n + 1/2) / N) H(f - 1) + (n + 1/2) / N H(f), from the
    identity before the first frame; a concealed frame holds; a frame
    without data has the identity."""
    n, off = 24, 18
    par = [[20] * 8, [0] * 8, [0] * 8]
    level = 9.0
    g = 10 ** (9.0 / 20) - 1
    G = 1 + g * T209[20]
    Q = np.ones((64, n * 4 + off), complex)
    DE.apply({"C": Q}, [_entry(1, par, max_gain=2), DE.HOLD, None], level,
             n, off)
    assert np.all(Q[:, :off] == 1)
    for i in range(n):
        w = (i + 0.5) / n
        assert np.allclose(Q[:, off + i], (1 - w) + w * G)      # frame 0
        assert np.allclose(Q[:, off + n + i], G)                # held
        assert np.allclose(Q[:, off + 2 * n + i], (1 - w) * G + w)
    assert np.all(Q[:, off + 3 * n:] == 1)


def test_level_zero_is_untouched():
    Q = np.ones((64, 48), complex)
    DE.apply({"C": Q}, [_entry(1, [[31] * 8] * 3)] * 2, 0.0, 24, 0)
    assert np.all(Q == 1)
