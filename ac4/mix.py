# SPDX-License-Identifier: Apache-2.0
"""Mixing the substreams of one presentation (TS 103 190-2 4.8.3.17,
4.8.3.18, 4.8.4; TS 103 190-1 6.2.16).

A presentation can reference several substream groups: music and effects
plus dialogue, main plus associated audio (audio description,
commentary), or both (Table 53; presentation_config 5 by
content_classifier, Table 54). Each channel-coded substream is decoded by
its own `Decoder` (substream=k of toc.audio_substreams), every decoder
sees every frame, and the outputs are mixed frame by frame:

  * the output layout is the main (or music and effects) substream's
    (TS 103 190-1 6.2.16.0: "No additional channels are allowed for the
    dialogue or associated audio substream, except for a mono channel");
  * every other substream is rendered onto it by channel name; a mono
    substream (C) goes to the output's C, or, with pan_associated (an
    associated substream), is panned by angle (render_gains);
  * gains: the substream group gain g_sg (Table 70, sg_gain of the
    presentation substream; b_keep keeps the last, 0 dB until the first);
    for an associated substream g_assoc (the user's, default 0 dB) and
    scale_main_centre, scale_main_front and scale_main on the main and
    dialogue substreams, in that order (4.8.3.17); the user's dialogue
    gain g_dialog (default 0 dB, at most dialog_max_gain: not read here,
    so only gains <= 0 dB are taken);
  * the sum is divided by the number of substreams mixed (4.8.4, as
    printed: Y = sum(X_s) / n_sub; MIX_NORMALISE = False drops it, TS 103
    190-1 6.2.16's sum).
Mixing metadata stays valid until new values are sent (6.2.16.0).
Phase D: where a substream's mixing matrix changes from one frame to the
next it is crossfaded over that frame, sample n of N taking (1 - w) the
previous matrix and w the new one, w = (n + 1/2) / N (a reading: the
standard does not say how; the form is that of dialogue enhancement's
interpolation, TS 103 190-1 5.7.8.6); unchanged matrices apply as
they are.

Dialogue enhancement substreams (presentation_config 1 and 4) are not
mixed (toc.audio_substreams leaves them out). Object audio substreams
(phase E) are decoded by their own Decoder, rendered by it to the main
substream's layout (or the one asked for), and mixed like the rest; an
A-JOC substream mixed with others is taken in core decoding mode. UNVERIFIED ON REAL STREAMS: no capture or public test signal
at hand carries more than one audio substream in a presentation (the
second substream of every one of them is the presentation substream);
the tests build such streams bit by bit.

Loudness normalisation and DRC act on the mix, with the presentation
substream's metadata as read by the main substream's decoder.
"""
from __future__ import annotations

import math

import numpy as np

from . import toc as M
from .decoder import Decoder, ELEMENT_OF_CH_MODE

MIX_NORMALISE = True

# loudspeaker azimuths (degrees, clockwise from the front) for panning a
# mono substream (pan_associated)
AZIMUTH = {"L": -30.0, "C": 0.0, "R": 30.0, "Ls": -110.0, "Rs": 110.0}


def probe(frames, presentation=None, language=None, limit=64):
    """-> [dict(k, type, group, element, ch_mode)] for the channel-coded
    substreams of the presentation that would be decoded, from the first
    frame whose TOC parses (empty when none)."""
    for f in frames[:limit]:
        try:
            st = M.parse(f)
        except Exception:                                      # noqa: BLE001
            continue
        pi = M.choose(st, presentation, language)
        subs = M.audio_substreams(st, pi)
        if not subs:
            continue
        p = st["presentations"][pi]
        out = []
        for k, (gi, g, s) in enumerate(subs):
            cm = s.get("channel_mode")
            obj = s["kind"] in ("obj", "ajoc")
            out.append(dict(k=k, type=M.substream_type(st, pi, gi),
                            group=gi, ch_mode=cm,
                            element=("ajoc" if obj else
                                     ELEMENT_OF_CH_MODE.get(cm)),
                            info=s if obj else None,
                            version=p.get("version")))
        # the substream group index (sg of Table 70) of each substream:
        # the position of its group among the presentation's mixed groups
        order = []
        for _gi, _g, _s in subs:
            if _gi not in order:
                order.append(_gi)
        for d in out:
            d["sg"] = order.index(d["group"])
        return out
    return []


def pan_gains(angle, chans):
    """Linear pairwise panning (gains sum to 1) between the two output
    loudspeakers either side of `angle` (degrees, clockwise, 0 = front).
    It meets Table 216 of TS 103 190-1: 330 -> L, 0 -> C (5.1) or L, R at
    0.5 each (2.0), 30 -> R. -> {channel: gain}."""
    spk = sorted((AZIMUTH[c], c) for c in chans if c in AZIMUTH)
    if not spk:
        return {}
    a = ((angle + 180.0) % 360.0) - 180.0
    if len(spk) == 1:
        return {spk[0][1]: 1.0}
    ring = spk + [(spk[0][0] + 360.0, spk[0][1])]
    for (a0, c0), (a1, c1) in zip(ring, ring[1:]):
        x = a if a >= a0 else a + 360.0
        if a0 <= x <= a1:
            t = (x - a0) / (a1 - a0) if a1 > a0 else 0.0
            g = {c0: 1.0 - t}
            g[c1] = g.get(c1, 0.0) + t
            return g
    return {spk[0][1]: 1.0}


def render_gains(src, dst, pan=None):
    """-> matrix (len(dst), len(src)): substream channels onto the output
    layout. Channels go to the output channel of the same name; a mono
    substream (C) to the output's C, else to L and R at 0.5 each (Table
    216, 0 degrees on 2.0), or by `pan` degrees (pan_gains)."""
    m = np.zeros((len(dst), len(src)))
    if tuple(src) == ("C",) and (pan is not None or "C" not in dst):
        for c, g in pan_gains(pan or 0.0, dst).items():
            m[dst.index(c), 0] = g
        return m
    for j, c in enumerate(src):
        if c in dst:
            m[dst.index(c), j] = 1.0
    return m


def db(x):
    return 0.0 if x == -math.inf else 10.0 ** (x / 20.0)


class MixState:
    """The mixing metadata in force (held until new values are sent)."""

    def __init__(self, n_groups):
        self.sg = [0.0] * n_groups          # dB
        self.assoc = None

    def update(self, meta):
        if not meta:
            return
        g = meta.get("sg_gain")
        if isinstance(g, list):
            self.sg = (list(g) + [0.0] * len(self.sg))[:len(self.sg)]
        if meta.get("assoc") is not None:
            self.assoc = dict(meta["assoc"])


class MixDecoder:
    """Decodes every channel-coded substream of a presentation and mixes
    them (see the module docstring). Same interface as Decoder for
    decode_pass, chans, n_concealed, concealed, last_concealed, log,
    leveler, ... . `subs` from probe()."""

    def __init__(self, subs, channels=None, guard=True, presentation=None,
                 language=None, target_level=None, drc="auto", de_level=0.0,
                 assoc_gain=0.0, dialog_gain=0.0, layout=None,
                 stereo_dmx="loro"):
        from . import robust as RB
        from .render import LAYOUTS
        self.subs = list(subs)
        self.decs = [None] * len(self.subs)
        mains = [i for i, d in enumerate(self.subs) if d["type"] == "main"]
        self.main = mains[0] if mains else 0
        # object substreams (phase E) are rendered to a layout by their
        # decoder: the main substream's, when that is a channel
        # substream of one of render.LAYOUTS; else `layout`, or 5.1
        order = [self.main] + [i for i in range(len(self.subs))
                               if i != self.main]
        obj_layout = layout or "5.1"
        for i in order:
            d = self.subs[i]
            el = d["element"]
            if el is None:
                raise ValueError(f"substream {d['k']}: channel mode "
                                 f"{d['ch_mode']} not decodable")
            self.decs[i] = Decoder(
                el, guard=guard, presentation=presentation,
                language=language, de_level=de_level, substream=d["k"],
                ch_mode=d["ch_mode"] if el in ("7_X", "ice") else None,
                ajoc_info=d.get("info"), ajoc_core=True,
                layout=obj_layout if el == "ajoc" else None)
            if i == self.main and el != "ajoc":
                hit = [k for k, v in LAYOUTS.items()
                       if v == tuple(self.decs[i].chans)]
                if hit:
                    obj_layout = hit[0]
        full = self.decs[self.main].chans
        if channels is not None and channels not in (2, len(full)):
            raise ValueError(f"channels must be {len(full)} or 2")
        self.full = full
        self.chans = full if channels in (None, len(full)) else ("L", "R")
        # an output layout (render.py), after the mix and the leveler
        self.dchans = self.chans
        self.renderer = None
        if layout is not None:
            from .render import Renderer
            self.renderer = Renderer(self.dchans, layout, stereo_dmx)
            self.chans = self.renderer.out
        self.element = self.decs[self.main].element
        self.target_level = target_level
        self.drc_mode = drc
        self.leveler = None
        self.assoc_gain = min(0.0, float(assoc_gain))
        self.dialog_gain = min(0.0, float(dialog_gain))
        self.state = MixState(1 + max(d["sg"] for d in self.subs))
        self.version = self.subs[0].get("version")
        self.concealed = RB.DamageLog()
        self.n_concealed = 0
        self.log = []
        self.last_concealed = np.zeros(0, dtype=bool)

    # the Decoder interface ------------------------------------------
    @property
    def frame(self):
        return self.decs[self.main].frame

    @property
    def ratio(self):
        return self.decs[self.main].ratio

    @property
    def out_rate(self):
        return self.decs[self.main].out_rate

    @property
    def guard(self):
        return self.decs[self.main].guard

    def _sum(self, name):
        return sum(getattr(d, name) for d in self.decs)

    n_ok = property(lambda self: self.decs[self.main].n_ok)
    n_bad = property(lambda self: self._sum("n_bad"))
    n_seq_jumps = property(lambda self: self.decs[self.main].n_seq_jumps)
    n_burst = property(lambda self: self._sum("n_burst"))
    n_hf_fail = property(lambda self: self._sum("n_hf_fail"))

    def gains(self, meta):
        """-> per substream the (len(full), n_ch) matrix of this frame."""
        self.state.update(meta)
        a = self.state.assoc
        has_assoc = any(d["type"] == "associated" for d in self.subs)
        out = []
        for d, dec in zip(self.subs, self.decs):
            src = dec.chans
            pan = (a.get("pan_associated") if a and d["type"] == "associated"
                   and tuple(src) == ("C",) else None)
            m = render_gains(src, self.full, pan) * db(self.state.sg[d["sg"]])
            if d["type"] == "associated":
                m = m * db(self.assoc_gain)
            if d["type"] == "dialogue":
                m = m * db(self.dialog_gain)
            if a and has_assoc and d["type"] in ("main", "dialogue"):
                # 4.8.3.17: centre, then L / R, then every channel
                for key, which in (("scale_main_centre", ("C",)),
                                   ("scale_main_front", ("L", "R")),
                                   ("scale_main", tuple(self.full))):
                    if a.get(key) is not None:
                        g = db(a[key])
                        for c in which:
                            if c in self.full:
                                m[self.full.index(c), :] *= g
            out.append(m)
        if MIX_NORMALISE and (self.version or 0) >= 1:
            out = [m / len(self.subs) for m in out]
        return out

    def decode_pass(self, frames):
        frames = list(frames)
        ys = [d.decode_pass(frames) for d in self.decs]
        F = self.frame
        n = len(frames)
        y = np.zeros((n * F, len(self.full)))
        meta = list(self.decs[self.main].stream.meta)
        meta = (meta + [None] * n)[:n]
        conc = np.zeros(n, dtype=bool)
        for d in self.decs:
            c = np.asarray(d.last_concealed, dtype=bool)
            conc[:len(c)] |= c[:n]
            self.log.extend(d.log)
            d.log.clear()
        w = (np.arange(F) + 0.5) / F
        for f in range(n):
            ms = self.gains(meta[f])
            prev = getattr(self, "_m_prev", None) or [None] * len(ms)
            seg = slice(f * F, (f + 1) * F)
            for yk, m, mp in zip(ys, ms, prev):
                if mp is None or mp.shape != m.shape or np.array_equal(mp,
                                                                       m):
                    y[seg] += yk[seg] @ m.T
                    continue
                for c in range(m.shape[0]):
                    for j in range(m.shape[1]):
                        if m[c, j] == 0.0 and mp[c, j] == 0.0:
                            continue
                        g = mp[c, j] + (m[c, j] - mp[c, j]) * w
                        y[seg, c] += yk[seg, j] * g[:len(yk[seg])]
            self._m_prev = [m.copy() for m in ms]
        # reasons: every substream decoder's, the others' prefixed with
        # their substream ("s1:asf.sect_len"); n_concealed counts frames
        # any of them concealed
        self.concealed.reasons.clear()
        for i, d in enumerate(self.decs):
            for r, v in d.concealed.reasons.items():
                key = r if i == self.main else f"s{self.subs[i]['k']}:{r}"
                self.concealed.reasons[key] += v
        self.n_concealed += int(conc.sum())
        self.last_concealed = conc
        if self.dchans != self.full:
            idx = [self.full.index(c) for c in self.dchans]
            y = y[:, idx]
        if self.target_level is not None:
            if self.leveler is None:
                from .loudness import Leveler
                up, down = self.ratio
                self.leveler = Leveler(self.dchans, self.target_level,
                                       self.drc_mode,
                                       rate=self.out_rate * down / up)
            y = self.leveler.process(y, meta, F)
        if self.renderer is not None:
            y = self.renderer.process(y, meta, F)
        return y
