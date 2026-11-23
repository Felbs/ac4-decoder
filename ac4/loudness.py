# SPDX-License-Identifier: Apache-2.0
"""Loudness normalisation and dynamic range control (TS 103 190-1 5.7.9,
4.3.13; dialnorm 4.3.12.2.1 / TS 103 190-2 6.3.2.x).

The decoder's own output is in decoder units. This stage takes it to
units of full scale (FULL_SCALE decoder units = 1.0) and then:

  * normalises the programme's reference level, dialnorm (Lin, dBFS, from
    the presentation substream), to a target output level Lout (default
    -24, the "flat panel TV" range of Table 161): a gain of
    2^((Lout - Lin) / 6) (5.7.9.3.3; levels in dB2, 6 dB2 = a factor 2);
  * applies the DRC decoder mode the stream configures for that output
    level (5.7.9.2: the largest mode ID with Lout,min < Lout < Lout,max):
    a compression curve -- the (E-)AC-3 profile of Table 162 for a
    default profile (drc_eac3_profile), or the transmitted curve (Table
    166 control points) -- or the transmitted gains (4.3.13.6,
    5.7.9.3.2).

Readings (README, Interpretations):
  * The full scale is IMDCT_K * 2^14.5 decoder units (phase A: the
    loudness metadata of five streams against their BS.1770 loudness).
  * The level detector (5.7.9.3.1.1 leaves it to the implementation) is
    the BS.1770 K-weighted mean square of one QMF time slot (64 samples
    at the internal rate) summed over the channels with the BS.1770
    channel weights (LFE excluded, surrounds 1.41), as -0.691 + 10 log10
    in dB re full scale, minus dialnorm. The K-weighting filter is the
    48 kHz one at every internal rate.
  * DRC is applied as one wideband gain per QMF time slot to the output
    samples (the standard applies it to QMF samples before synthesis);
    the transmitted gains of the multi-band configurations (2 and 4
    bands) are applied as the mean of their bands in dB2.
  * Curve gains are 10^(G / 20) (5.7.9.3.1.2), transmitted gains and the
    normalisation 2^(G / 6) (dB2).
"""
from __future__ import annotations

import math

import numpy as np

from .filterbank import IMDCT_K

FULL_SCALE = IMDCT_K * 2.0 ** 14.5
TARGET_DEFAULT = -24.0
SLOT = 64                       # one QMF time slot

# Table 162: the (E-)AC-3 profiles: (L0low, L0high, Gmaxboost,
# Lmaxboost, Gmaxcut, Lmaxcut, Lsectioncut, Gsectioncut or None) and the
# time constants (attack, release, attack fast, release fast [ms],
# attack threshold, release threshold)
PROFILES = {
    0: None,                                            # None
    1: ((0, 5, 6, -12, -24, 35, 15, -5), (100, 3000, 10, 1000, 15, 20)),
    2: ((-10, 10, 6, -22, -24, 40, 20, -5), (100, 3000, 10, 1000, 15, 20)),
    3: ((0, 5, 12, -24, -24, 35, 15, -5), (100, 10000, 10, 1000, 15, 20)),
    4: ((-10, 10, 12, -34, -15, 40, None, None),
        (100, 3000, 10, 1000, 15, 20)),
    5: ((0, 5, 15, -19, -24, 35, 15, -5), (100, 1000, 10, 200, 10, 10)),
}
# Table 167
DEFAULT_TC = (100, 3000, 10, 1000, None, None)

# BS.1770 K-weighting at 48 kHz (two biquads) and the channel weights
K_B1 = (1.53512485958697, -2.69169618940638, 1.19839281085285)
K_A1 = (1.0, -1.69065929318241, 0.73248077421585)
K_B2 = (1.0, -2.0, 1.0)
K_A2 = (1.0, -1.99004745483398, 0.99007225036621)
CH_WEIGHT = {"L": 1.0, "R": 1.0, "C": 1.0, "lfe": 0.0, "Ls": 1.41,
             "Rs": 1.41}


def curve_points(m, eac3_profile):
    """A DRC mode's compression curve as (L0low, L0high, Gmaxboost,
    Lmaxboost, Gsectionboost, Lsectionboost, Gmaxcut, Lmaxcut,
    Gsectioncut, Lsectioncut) and its time constants, or None (profile
    None / no curve)."""
    if m.get("default_profile"):
        p = PROFILES.get(eac3_profile)
        if p is None:
            return None
        l0l, l0h, gmb, lmb, gmc, lmc, lsc, gsc = p[0]
        pts = dict(l0l=l0l, l0h=l0h, gmb=gmb, lmb=lmb, gsb=None, lsb=l0l,
                   gmc=gmc, lmc=lmc, gsc=gsc, lsc=lsc if lsc is not None
                   else l0h)
        return pts, p[1]
    c = m.get("curve")
    if not c:
        return None
    # Table 166
    l0l = 0 - c["nullband_low"]
    l0h = 0 + c["nullband_high"]
    gmb = c["gain_max_boost"]
    if c.get("nr_boost_sections"):
        gsb = 1 + c["gain_section_boost"]
        lsb = l0l - (1 + c["lev_section_boost"])
    else:
        gsb, lsb = None, l0l
    lmb = lsb - (1 + c["lev_max_boost"]) if gmb > 0 else l0l
    gmc = 0 - c["gain_max_cut"]
    if c.get("nr_cut_sections"):
        gsc = 0 - (1 + c["gain_section_cut"])
        lsc = l0h + (1 + c["lev_section_cut"])
    else:
        gsc, lsc = None, l0h
    lmc = lsc + (1 + c["lev_max_cut"]) if gmc < 0 else l0h
    if c.get("tc_default", 1):
        tc = DEFAULT_TC
    else:
        a, r, af, rf = c["tc"]
        th = c.get("thresholds") if c.get("adaptive") else None
        tc = (a * 5, r * 40, af * 5, rf * 20,
              th[0] if th else None, th[1] if th else None)
    return dict(l0l=l0l, l0h=l0h, gmb=gmb, lmb=lmb, gsb=gsb, lsb=lsb,
                gmc=gmc, lmc=lmc, gsc=gsc, lsc=lsc), tc


def curve_gain(x, p):
    """5.7.9.3.1.2: the gain (dB) for level x (dB re dialnorm), piecewise
    linear through the control points (a section that is not sent has
    zero gain at its inner end: Gsectionboost / Gsectioncut 0 at
    L0,low / L0,high)."""
    gsb = p["gsb"] if p["gsb"] is not None else 0.0
    gsc = p["gsc"] if p["gsc"] is not None else 0.0
    if x < p["lmb"]:
        return float(p["gmb"])
    if x <= p["lsb"]:                                   # G1
        d = p["lsb"] - p["lmb"]
        return gsb + (p["gmb"] - gsb) * ((p["lsb"] - x) / d) if d else gsb
    if x <= p["l0l"]:                                   # G2
        d = p["l0l"] - p["lsb"]
        return gsb * ((p["l0l"] - x) / d) if d else 0.0
    if x <= p["l0h"]:
        return 0.0
    if x <= p["lsc"]:                                   # G3
        d = p["lsc"] - p["l0h"]
        return gsc * ((x - p["l0h"]) / d) if d else 0.0
    if x <= p["lmc"]:                                   # G4
        d = p["lmc"] - p["lsc"]
        return gsc + (p["gmc"] - gsc) * ((x - p["lsc"]) / d) if d else gsc
    return float(p["gmc"])


def select_mode(config, target):
    """5.7.9.2: the mode with the largest ID whose output range holds the
    target (Lout,min < Lout < Lout,max), or None."""
    best = None
    for m in config.get("modes", []):
        rng = m.get("out_range")
        if not rng:
            continue
        lo, hi = min(rng), max(rng)
        if lo < target < hi and (best is None or m["id"] > best["id"]):
            best = m
    return best


class Leveler:
    """The stage for one stream: per output frame, its metadata (dialnorm
    in dBFS, the DRC frame) and its samples (decoder units, internal
    rate) -> samples in units of full scale, normalised and compressed.

    target: Lout (dB); drc: "auto" (the mode for the target), "off", or a
    DRC decoder mode ID. channel names: the decoder's (L R C lfe Ls Rs)."""

    def __init__(self, chans, target=TARGET_DEFAULT, drc="auto",
                 rate=48000.0):
        self.chans = tuple(chans)
        self.target = float(target)
        self.drc = drc
        self.rate = float(rate)
        self.w = np.array([CH_WEIGHT.get(k, 1.0) for k in self.chans])
        self.zi1 = np.zeros((2, len(self.chans)))
        self.zi2 = np.zeros((2, len(self.chans)))
        self.lin = None             # dialnorm in force
        self.g_s = None             # smoothed curve gain (linear)
        self.l_s = None             # smoothed level (linear power)
        self.mode_id = None

    def _kweight(self, x):
        """Two DF2T biquads per channel (scipy lfilter's arithmetic:
        y = b0 x + z0, z0 = b1 x + z1 - a1 y, z1 = b2 x - a2 y), state
        carried."""
        from scipy.signal import lfilter
        out = np.empty_like(x)
        for c in range(x.shape[1]):
            v, self.zi1[:, c] = lfilter(K_B1, K_A1, x[:, c],
                                        zi=self.zi1[:, c])
            out[:, c], self.zi2[:, c] = lfilter(K_B2, K_A2, v,
                                                zi=self.zi2[:, c])
        return out

    def process(self, y, meta, frame_len):
        """y: (n, ch) decoder units, n = frames x frame_len; meta: per
        frame a dict (dialnorm, drc) or None (the last kept). -> (n, ch)
        in units of full scale."""
        y = np.asarray(y, float) / FULL_SCALE
        n_fr = len(y) // frame_len
        n_slot = -(-frame_len // SLOT)  # a last short slot at 480, 960
        kw = self._kweight(y) if self.drc != "off" else None
        out = y.copy()
        for f in range(n_fr):
            md = meta[f] if f < len(meta) else None
            if md is not None and md.get("dialnorm") is not None:
                self.lin = md["dialnorm"]
            lin = self.lin if self.lin is not None else self.target
            norm = 2.0 ** ((self.target - lin) / 6.0)
            drc = md.get("drc") if md else None
            if drc is not None:
                self._drc_frame = drc
            dfr = getattr(self, "_drc_frame", None)
            gains = np.ones((n_slot, len(self.chans)))
            if self.drc != "off" and dfr is not None:
                gains = self._drc_gains(dfr, kw, f, frame_len, lin,
                                        md is not None and drc is not None)
            a, z = f * frame_len, (f + 1) * frame_len
            seg = out[a:z]
            for s in range(n_slot):
                seg[s * SLOT:(s + 1) * SLOT] *= norm * gains[s][None, :]
        return out

    def _drc_gains(self, dfr, kw, f, frame_len, lin, fresh):
        cfg = dfr["config"]
        n_slot = -(-frame_len // SLOT)
        g = np.ones((n_slot, len(self.chans)))
        if self.drc == "auto":
            m = select_mode(cfg, self.target)
        else:
            m = next((x for x in cfg["modes"] if x["id"] == self.drc), None)
        if m is None:
            return g
        self.mode_id = m["id"]
        if m.get("curve_flag", 1) == 0:
            gl = dfr.get("gains", {}).get(m["id"]) if fresh else \
                dfr.get("gains", {}).get(m["id"])
            if gl is None:
                return g
            return self._transmitted(gl, n_slot)
        cp = curve_points(m, cfg.get("eac3_profile", 0))
        if cp is None:
            return g
        pts, tc = cp
        if fresh and dfr.get("reset"):
            self.g_s = self.l_s = None
        T = SLOT * 1000.0 / self.rate
        att, rel, attf, relf, ath, rth = tc
        a0 = f * frame_len
        for s in range(n_slot):
            seg = kw[a0 + s * SLOT:min(a0 + (s + 1) * SLOT,
                                       a0 + frame_len)]
            ms = 0.0
            for c in range(len(self.chans)):
                acc = float(np.sum(np.ascontiguousarray(seg[:, c]) ** 2))
                ms += self.w[c] * (acc / len(seg))
            lvl = ms                                    # linear power
            x = -0.691 + 10.0 * math.log10(lvl + 1e-30) - lin
            gl = 10.0 ** (curve_gain(x, pts) / 20.0)
            if self.g_s is None:
                self.g_s, self.l_s = gl, lvl
            else:
                tau = att if lvl > self.l_s else rel
                if ath is not None and lvl > 0 and self.l_s > 0:
                    r = 10.0 * math.log10(lvl / self.l_s)
                    if r > ath:
                        tau = attf
                    elif -r > rth:
                        tau = relf
                al = 2.0 ** (-T / tau) if tau > 0 else 0.0
                self.g_s = al * self.g_s + (1.0 - al) * gl
                self.l_s = al * self.l_s + (1.0 - al) * lvl
            g[s, :] = self.g_s
        return g

    def _transmitted(self, gl, n_slot):
        """Transmitted gains (5.7.9.3.2): channel groups (Table 168: one
        group, or [L R LFE], [C], [the rest]), subframes spread over the
        frame's slots, bands averaged in dB2."""
        n_ch_g = len(gl)
        n_sf = len(gl[0])
        out = np.ones((n_slot, len(self.chans)))
        grp = {"L": 0, "R": 0, "lfe": 0, "C": 1}
        for s in range(n_slot):
            sf = min(s * n_sf // n_slot, n_sf - 1)
            for ci, k in enumerate(self.chans):
                gi = 0 if n_ch_g == 1 else grp.get(k, 2)
                gi = min(gi, n_ch_g - 1)
                bands = gl[gi][sf]
                db = sum(bands) / len(bands)
                out[s, ci] = 2.0 ** (db / 6.0)
        return out
