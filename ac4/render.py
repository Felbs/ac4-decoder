# SPDX-License-Identifier: Apache-2.0
"""Rendering the decoded channels to an output layout (`--layout`).

The channel audio renderer of TS 103 190-2 5.10.2: a matrix from the
decoded channel configuration to the output configuration, its
coefficients from Tables 38 - 43 (full decoding) with the custom downmix
parameters the presentation substream sends (custom_dmx_data, 6.2.9.2;
metadata.custom_dmx_data) or, where none were sent, their defaults
(Table 130). Stereo output is the 5.X.0 rendering downmixed by TS 103
190-1 6.2.17.4 (Table 218) in its LoRo form, with the stream's
loro_centre_mixgain / loro_surround_mixgain / lfe_mixgain when sent
(-3 dB, -3 dB and no LFE otherwise, 4.3.12.2.8 / .9 / .18); 3.0 to 2.0 by
Table 217, mono to 2.0 by 6.2.17.6, 7.X (5/2/0, 3/2/2) to 5.X by Table 219.

Readings (README, Interpretations): the input configuration is the
channels the decoder gives out, so a 7.1.4 channel mode whose back pair is
signalled absent renders as 5.X.4 (the bs_ch_config of 6.2.9.2 is 5.X.4
for it too); the 7.0 / 7.1 3/2/2 layout's Tfl / Tfr render as the top
front pair of a 5.X.4 input with silent top back channels; a decoder
choosing LtRt (preferred_dmx_method 2, 3) is not built: stereo is LoRo.
Gains change at frame boundaries, unsmoothed.

Phase D: the other rows of Table 218 (`dmx`: "loro", the default;
"ltrt", preferred_dmx_method 2; "pl2", method 3 with its +1.8 / -3.2 dB
surround weights; "auto", the stream's preferred_dmx_method, LoRo when
it indicates none), the LtRt gains defaulting to the LoRo ones without
b_ltrt_mixinfo (4.3.12.2.12); the downmix loudness correction of TS 103
190-2 4.8.5.3 (loud_corr, metadata.loud_corr): when the layout drops
channels the decoder gives out, its gain for that output configuration
(loro / ltrt for 2.0), held from the last sent, 0 dB before.

Layouts: "2.0", "5.1", "5.1.2", "5.1.4", "7.1", "7.1.2", "7.1.4", in WAV
order (L R C LFE, the surrounds, the back pair, then the top channels).
"""
from __future__ import annotations

import math

import numpy as np

LAYOUTS = {
    "2.0": ("L", "R"),
    "5.1": ("L", "R", "C", "lfe", "Ls", "Rs"),
    "5.1.2": ("L", "R", "C", "lfe", "Ls", "Rs", "Tsl", "Tsr"),
    "5.1.4": ("L", "R", "C", "lfe", "Ls", "Rs", "Tfl", "Tfr", "Tbl", "Tbr"),
    "7.1": ("L", "R", "C", "lfe", "Ls", "Rs", "Lb", "Rb"),
    "7.1.2": ("L", "R", "C", "lfe", "Ls", "Rs", "Lb", "Rb", "Tsl", "Tsr"),
    "7.1.4": ("L", "R", "C", "lfe", "Ls", "Rs", "Lb", "Rb", "Tfl", "Tfr",
              "Tbl", "Tbr"),
}
# out_ch_config of Table 127 for each layout family (2.0: the 5.X.0
# stage)
OUT_CH_CONFIG = {"5.X.0": 0, "5.X.2": 1, "5.X.4": 2, "7.X.0": 3, "7.X.2": 4}
FAMILY = {"2.0": "5.X.0", "5.1": "5.X.0", "5.1.2": "5.X.2",
          "5.1.4": "5.X.4", "7.1": "7.X.0", "7.1.2": "7.X.2",
          "7.1.4": "7.X.4"}
# Table 130
DEFAULTS = dict(gain_f1=-math.inf, gain_f2=0.0, gain_b=-3.0, gain_t1=-3.0,
                gain_t2a=-math.inf, gain_t2b=-3.0, gain_t2c=-math.inf,
                gain_t2d=-math.inf, gain_t2e=-3.0, gain_t2f=-math.inf)
M3DB = 10.0 ** (-3.0 / 20.0)        # the tables' fixed -3 dB


def db(x):
    return 0.0 if x == -math.inf else 10.0 ** (x / 20.0)


def input_config(chans):
    """The configuration of a decoded channel set: "7.X.4", "5.X.4",
    "7.X.0", "5/2/0" (7_X with Lw / Rw), "5.X.0", "3.0", "2.0", "1.0"."""
    s = set(chans)
    if s & {"Tfl", "Tfr", "Tbl", "Tbr"}:
        return "7.X.4" if "Lb" in s else "5.X.4"
    if "Lw" in s:
        return "5/2/0"
    if "Lb" in s:
        return "7.X.0"
    if "Ls" in s:
        return "5.X.0"
    if "L" in s and "C" in s:
        return "3.0"
    if "L" in s:
        return "2.0"
    return "1.0"


def _family_matrix(cfg_in, fam, g):
    """Tables 38 - 43: {out: {in: gain}} for the channels of the family
    fam from the configuration cfg_in, custom gains g (dB)."""
    m = {}

    def add(o, i, v):
        if v:
            m.setdefault(o, {})
            m[o][i] = m[o].get(i, 0.0) + v

    beds = {"7.X.4": ("L", "R", "C", "Ls", "Rs", "Lb", "Rb"),
            "5.X.4": ("L", "R", "C", "Ls", "Rs"),
            "7.X.0": ("L", "R", "C", "Ls", "Rs", "Lb", "Rb"),
            "5.X.0": ("L", "R", "C", "Ls", "Rs"),
            "3.0": ("L", "R", "C"), "2.0": ("L", "R"), "1.0": ("C",)}[cfg_in]
    tops = cfg_in in ("7.X.4", "5.X.4")
    backs = cfg_in in ("7.X.4", "7.X.0")
    out_backs = fam.startswith("7")
    add("lfe", "lfe", 1.0)
    for k in beds:
        if k in ("Lb", "Rb") and not out_backs:
            continue
        if k in ("Ls", "Rs") and backs and not out_backs:
            continue
        add(k, k, 1.0)
    if backs and not out_backs:
        # 7.X to 5.X: Ls = gain_b (Ls + Lb); -3 dB fixed from 7.X.0
        gb = db(g["gain_b"]) if tops else M3DB
        for s_, b_ in (("Ls", "Lb"), ("Rs", "Rb")):
            add(s_, s_, gb)
            add(s_, b_, gb)
    if not tops:
        return m
    if fam.endswith(".4"):
        for k in ("Tfl", "Tfr", "Tbl", "Tbr"):
            add(k, k, 1.0)
    elif fam.endswith(".2"):
        # r12,7 = r12,9 = r13,8 = r13,10: gain_t1 from 7.X.4 and into
        # 5.X.2; -3 dB fixed from 5.X.4 into 7.X.2
        gt = (M3DB if (cfg_in == "5.X.4" and fam == "7.X.2")
              else db(g["gain_t1"]))
        for o, a, b_ in (("Tsl", "Tfl", "Tbl"), ("Tsr", "Tfr", "Tbr")):
            add(o, a, gt)
            add(o, b_, gt)
    else:
        # to 7.X.0 from 5.X.4: Ls += -3 dB (Tfl + Tbl), fixed
        if fam == "7.X.0" and cfg_in == "5.X.4":
            for o, a, b_ in (("Ls", "Tfl", "Tbl"), ("Rs", "Tfr", "Tbr")):
                add(o, a, M3DB)
                add(o, b_, M3DB)
            return m
        for side, tf, tb in (("L", "Tfl", "Tbl"), ("R", "Tfr", "Tbr")):
            s_ = side + "s"
            add(side, tf, db(g["gain_t2a"]))
            add(s_, tf, db(g["gain_t2b"]))
            add(side, tb, db(g["gain_t2d"]))
            add(s_, tb, db(g["gain_t2e"]))
            if fam == "7.X.0":
                add(side + "b", tf, db(g["gain_t2c"]))
                add(side + "b", tb, db(g["gain_t2f"]))
    return m


DMX_MODES = ("loro", "ltrt", "pl2", "auto")


def stereo_method(dmx, cdmx):
    """The Table 218 row used: dmx (DMX_MODES) with the stream's
    preferred_dmx_method for "auto" -> "loro", "ltrt" or "pl2"."""
    if dmx != "auto":
        return dmx
    pm = ((cdmx or {}).get("stereo") or {}).get("preferred_dmx_method")
    return {2: "ltrt", 3: "pl2"}.get(pm, "loro")


def matrix(chans, layout, cdmx=None, dmx="loro"):
    """-> (n_out, n_in) array rendering the decoded channels `chans` to
    `layout`, with the custom downmix data cdmx (metadata.custom_dmx_data,
    or None) of the frame; dmx: the stereo downmix (DMX_MODES)."""
    out = LAYOUTS[layout]
    cfg_in = input_config(chans)
    m = {}
    if cfg_in == "5/2/0":
        # Table 219 (add_ch_base 0): Lw to L at 0.707, then a 5.X input
        m0 = {"L": {"L": 1.0, "Lw": M3DB}, "R": {"R": 1.0, "Rw": M3DB},
              "C": {"C": 1.0}, "Ls": {"Ls": 1.0}, "Rs": {"Rs": 1.0},
              "lfe": {"lfe": 1.0}}
        m1 = _family_matrix("5.X.0", FAMILY[layout], DEFAULTS)
        m = {o: {} for o in m1}
        for o, row in m1.items():
            for k, v in row.items():
                for i, w in m0.get(k, {}).items():
                    m[o][i] = m[o].get(i, 0.0) + v * w
    else:
        fam = FAMILY[layout]
        g = dict(DEFAULTS)
        if cdmx:
            occ = OUT_CH_CONFIG[fam] if fam in OUT_CH_CONFIG else None
            cf = cdmx.get("configs") or {}
            if occ is not None and occ in cf:
                g.update(cf[occ])
            if occ == 4 and cdmx.get("bs_ch_config") == 1 and 1 in cf \
                    and "gain_t1" in cf[1] and "gain_t1" not in cf.get(4, {}):
                g["gain_t1"] = cf[1]["gain_t1"]       # 6.3.10.3.10
        m = _family_matrix(cfg_in, fam, g)
    if layout == "2.0":
        st = (cdmx or {}).get("stereo") or {}
        meth = stereo_method(dmx, cdmx)
        cmg = db(st.get("loro_centre", -3.0))
        smg = db(st.get("loro_surround", -3.0))
        if meth != "loro":
            cmg = db(st.get("ltrt_centre", st.get("loro_centre", -3.0)))
            smg = db(st.get("ltrt_surround", st.get("loro_surround", -3.0)))
        lfe = st.get("lfe_mixgain")
        lmg = db(5.5 - lfe) if lfe is not None else 0.0
        if cfg_in == "1.0":
            m2 = {"L": {"C": M3DB}, "R": {"C": M3DB}}       # 6.2.17.6
        elif meth == "loro":
            m2 = {"L": {"L": 1.0, "C": cmg, "Ls": smg, "lfe": lmg},
                  "R": {"R": 1.0, "C": cmg, "Rs": smg, "lfe": lmg}}
        else:
            # Table 218, preferred_dmx_method 2 / 3
            a, b_ = smg, smg
            if meth == "pl2":
                a, b_ = smg * db(1.8), smg * db(-3.2)
            m2 = {"L": {"L": 1.0, "C": cmg, "Ls": -a, "Rs": -b_,
                        "lfe": lmg},
                  "R": {"R": 1.0, "C": cmg, "Rs": a, "Ls": b_,
                        "lfe": lmg}}
        mm = {}
        for o, row in m2.items():
            mm[o] = {}
            for k, v in row.items():
                if not v:
                    continue
                for i, w in m.get(k, {}).items():
                    mm[o][i] = mm[o].get(i, 0.0) + v * w
        m = mm
    R = np.zeros((len(out), len(chans)))
    idx = {k: i for i, k in enumerate(chans)}
    for oi, o in enumerate(out):
        for k, v in m.get(o, {}).items():
            if k in idx:
                R[oi, idx[k]] += v
    return R


# loud_corr's output configuration of each layout (4.8.5.3)
LOUD_CORR_OF = {"5.1": "5_X", "5.1.2": "5_X_2", "5.1.4": "5_X_4",
                "7.1": "7_X", "7.1.2": "7_X_2", "7.1.4": "7_X_4"}


def is_downmix(chans, layout):
    """Does rendering chans to layout drop a channel?"""
    out = set(LAYOUTS[layout])
    return any(k not in out for k in chans if k != "lfe")


class Renderer:
    """Renders decode passes (n, len(chans)) to a layout, frame by frame
    with the custom downmix data in force (the last sent, held), and the
    downmix loudness correction (held)."""

    def __init__(self, chans, layout, dmx="loro"):
        if layout not in LAYOUTS:
            raise ValueError(f"layout {layout!r}: one of {tuple(LAYOUTS)}")
        if dmx not in DMX_MODES:
            raise ValueError(f"dmx {dmx!r}: one of {DMX_MODES}")
        self.chans = tuple(chans)
        self.layout = layout
        self.dmx = dmx
        self.out = LAYOUTS[layout]
        self.cdmx = None
        self.lc = {}
        self._key = None
        self._R = matrix(self.chans, layout, dmx=dmx)
        self._down = is_downmix(self.chans, layout)

    def _gain(self):
        """The loudness correction (linear) in force."""
        if not self._down:
            return 1.0
        if self.layout == "2.0":
            k = "loro" if stereo_method(self.dmx, self.cdmx) == "loro" \
                else "ltrt"
        else:
            k = LOUD_CORR_OF[self.layout]
        return db(self.lc.get(k, 0.0))

    def _update(self, md):
        """Hold what each frame sends: the configurations sent and the
        stereo coefficients when sent; the loudness corrections sent."""
        if not md:
            return
        if md.get("loud_corr"):
            self.lc = dict(self.lc)
            self.lc.update(md["loud_corr"])
        c = md.get("cdmx")
        if not c:
            return
        cur = dict(self.cdmx or {"configs": {}})
        cur["bs_ch_config"] = c.get("bs_ch_config")
        cfg = dict(cur.get("configs") or {})
        cfg.update(c.get("configs") or {})
        cur["configs"] = cfg
        if c.get("stereo"):
            cur["stereo"] = c["stereo"]
        self.cdmx = cur
        key = repr(cur)
        if key != self._key:
            self._key = key
            self._R = matrix(self.chans, self.layout, cur, self.dmx)

    def process(self, y, meta, frame):
        out = np.zeros((len(y), len(self.out)))
        n_f = (len(y) + frame - 1) // frame
        for f in range(n_f):
            self._update(meta[f] if f < len(meta) else None)
            s = slice(f * frame, (f + 1) * frame)
            g = self._gain()
            R = self._R if g == 1.0 else self._R * g
            out[s] = y[s] @ R.T
        return out
