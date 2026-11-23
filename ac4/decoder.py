# SPDX-License-Identifier: Apache-2.0
"""Frames in, PCM out.

`Ac4Stream` (stream.py) turns AC-4 frames into MDCT spectra and A-SPX side
information. This module holds the rest of the chain, lifted unchanged in
arithmetic and order from the original receiver's audio worker (see the
README's PROVENANCE section):

    spectra -> filterbank (overlap carried between passes)
            -> A-SPX high band per channel pair (QMF state, envelope state
               and a 4-frame lead-in carried between passes)
            -> burst guard (a frame whose A-SPX output exceeds a limit is
               replaced by its core band, plus the 2 frames after it)
    = PCM at the internal rate (1536 samples per frame, decoder scale)

`StreamResampler` (resample.py) then takes the internal rate to 48 kHz.

ASPX_ACPL_3 (a 5.1 programme sent as a two-channel core plus A-CPL): the
A-CPL upmix (acpl.py) runs on the core pair's A-SPX output in the QMF
domain and gives L, R, C, Ls, Rs; the LFE is decoded from its own
mono_data. With channels=2 such a stream gives its stereo core.

What it does NOT do:
companding, DRC / dialogue / loudness metadata,
presentation or substream choice (substream 1 is always decoded), frame
rates other than 29.97 and sample rates other than 48 kHz.

DAMAGED INPUT (robust.py): decode_pass never raises and always returns
1536 samples per frame. Frames that break a rule of the bitstream are
concealed by the stream (zeroed spectrum, A-SPX high band muted, state
held); after synthesis an output guard mutes any frame far above a running
reference. n_concealed / concealed.reasons / last_concealed report them.

THE BURST GUARD is a heuristic carried over from the receiver, not part of
the standard: on air, a few A-SPX frames have been seen to decode tens of
times louder than any programme (a decode error in the envelope chain whose
cause is not yet found). Its default limit is twice the receiver's fixed
reference level for the element and is in decoder units; pass
burst_limit=float("inf") to turn it off.
"""
from __future__ import annotations

import numpy as np

from . import aspx_render as R37
from . import filterbank as FB
from . import rates as RT
from . import robust as RB
from . import stream as SC
from .stream import Ac4Stream

FRAME = 1536                # internal samples per frame at 29.97 fps, 48 kHz
#                             (the default until a stream's rate is known;
#                             a Decoder's own is Decoder.frame)
LEAD = 4                    # frames of A-SPX/QMF lead-in carried per pair
HF_DELAY = 577              # samples the A-SPX path delays a channel
HF_TAIL = 2                 # frames after a burst frame that also play core

# The receiver's fixed output reference per element, in decoder units: the
# level written at 0.9 of full scale. Hand-tuned; the decoder's true full
# scale is NOT established. Used for the default burst limit and by the CLI.
REF_PEAK = {"5_X": 9.0e5, "pair": 2.0e5, "mono": 2.0e5, "3_0": 9.0e5,
            "5_0": 9.0e5, "7_X": 9.0e5}
ELEMENTS = tuple(REF_PEAK)


def core_for_bursts(hf_a, hf_b, core_a, core_b, lead, limit, owed,
                    hf_ratio=None, flagged=None, hf_floor=0.0, bad_out=None,
                    FRAME=FRAME):
    """One pair's A-SPX output with the core put back wherever A-SPX ran
    away. -> (a, b, frames replaced, frames still owed to the next pass).

    hf_ratio (damaged-input check, robust.py): a frame whose A-SPX output
    RMS is above hf_ratio x the RMS of the core it was made from is also
    replaced, unless its A-SPX peak is below hf_floor; `flagged` (a bool
    array of n frames) receives those frames; `bad_out` (a bool array)
    every frame replaced."""
    n = min(len(hf_a), len(hf_b)) // FRAME
    if not n:
        return hf_a, hf_b, 0, owed
    pk = np.maximum(np.abs(hf_a[:n * FRAME]).reshape(n, FRAME).max(axis=1),
                    np.abs(hf_b[:n * FRAME]).reshape(n, FRAME).max(axis=1))
    hot = pk > limit
    if hf_ratio is not None:
        over = np.zeros(n, dtype=bool)
        for hf, core in ((hf_a, core_a), (hf_b, core_b)):
            d = np.concatenate([np.zeros(HF_DELAY), core])[lead:lead + len(hf)]
            h = np.asarray(hf[:n * FRAME], dtype=float).reshape(n, FRAME)
            rh = np.sqrt((h ** 2).mean(axis=1))
            rc = np.sqrt((d[:n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))
            with np.errstate(invalid="ignore"):
                audible = ~(np.abs(h).max(axis=1) <= hf_floor)
                over |= ~(rh <= hf_ratio * rc) & audible
        if flagged is not None:
            flagged[:n] |= over
        hot = hot | over
    bad = np.zeros(n, dtype=bool)
    left = int(owed)
    for f in range(n):
        if hot[f]:
            bad[f], left = True, HF_TAIL
        elif left:
            bad[f], left = True, left - 1
    if bad_out is not None:
        bad_out[:n] |= bad
    if not bad.any():
        return hf_a, hf_b, 0, 0
    out = []
    for hf, core in ((hf_a, core_a), (hf_b, core_b)):
        d = np.concatenate([np.zeros(HF_DELAY), core])[lead:lead + len(hf)]
        hf = np.array(hf, dtype=float)
        for f in np.flatnonzero(bad):
            hf[f * FRAME:(f + 1) * FRAME] = d[f * FRAME:(f + 1) * FRAME]
        out.append(hf)
    return out[0], out[1], int(bad.sum()), left


class Decoder:
    """A resumable decoder: call decode_pass() with consecutive batches of
    frames; the filterbank overlap, the A-SPX envelope state, the QMF
    lead-in and the burst guard's carry-over are kept between calls.

    Every channel comes out HF_DELAY (577) samples after the MDCT core: the
    pairs that can carry A-SPX through the QMF bank from their first frame,
    the others (centre, LFE, a pair without A-SPX) through a delay line.

    KNOWN (inherited, measured): the output is NOT fully independent of
    where the batches are cut: the last frame of a batch can differ from a
    one-batch decode in its last 1279 samples (up to about 1 % of peak on
    one stereo stream, about -100 dB on others). Decode in the same batch
    sizes to get the same samples.

    element   "5_X" (5.1 channel element) or "pair" (channel_pair_element)
    channels  2 -> L, R;  6 -> L, R, C, LFE, Ls, Rs (WAV 5.1 order).
              A pair element always gives 2.
    """

    def __init__(self, element="5_X", channels=None, burst_limit=None,
                 guard=True, presentation=None, language=None,
                 target_level=None, drc="auto", ch_mode=None, de_level=0.0):
        """The frame length follows the stream (TS 103 190-1 Table 83):
        `frame` is FRAME (1536) until the first frame whose rate the
        decoder takes (see Ac4Stream), then that frame_len_base; out_rate
        and ratio give the output rate and the resampler ratio.

        target_level (dB, e.g. -24): the output is in units of full scale
        (loudness.FULL_SCALE), normalised from dialnorm to this level and
        compressed by the stream's DRC mode for it (drc "auto"; "off"; or
        a mode ID) -- loudness.Leveler. None: decoder scale, untouched."""
        if element not in ELEMENTS:
            raise ValueError(f"element {element!r}: one of {ELEMENTS}")
        self.stream = Ac4Stream(element=element, presentation=presentation,
                                language=language, ch_mode=ch_mode)
        full = self.stream.CHANS
        if channels is None:
            channels = len(full)
        if channels not in (2, len(full)) or (channels == 2
                                               and "L" not in full):
            raise ValueError(f"channels must be {len(full)}"
                             + (" or 2" if "L" in full else ""))
        self.element = element
        self.chans = full if channels == len(full) else ("L", "R")
        self.burst_limit = (2.0 * REF_PEAK[element] if burst_limit is None
                            else float(burst_limit))
        self.fb_state = {k: None for k in self.chans}
        self.pairs = [(self.stream.PAIRS[g], gk) for g, gk in
                      ((0, "lr"), (1, "sr"), (2, "c"), (3, "xr"))
                      if g in self.stream.PAIRS]
        self.lead_pcm = {k: np.zeros(0) for (ka, kb), _g in self.pairs
                         for k in (ka, kb)}
        self.lead_core = {}
        self.lead_groups = {"lr": [], "sr": [], "c": [], "xr": []}
        # the last HF_DELAY core samples of every channel: the delay line
        # of the channels that do not go through the QMF bank
        self.core_tail = {k: np.zeros(HF_DELAY) for k in self.chans}
        self.hf_state = {"lr": [None], "sr": [None], "c": [None],
                         "xr": [None]}
        # the A-SPX tables in force before the lead-in of the next pass
        # (aspx_render.apply_hf_pair t_start)
        self.t_hold = {"lr": None, "sr": None, "c": None, "xr": None}
        self.hf_hot = {"lr": 0, "sr": 0, "c": 0, "xr": 0}
        self.n_burst = 0
        self.n_hf_fail = 0
        self.acpl3 = False
        # A-CPL (ASPX_ACPL_3): the lead-in groups and the logged state, as
        # for the A-SPX pairs
        self.lead_acpl = []
        self.acpl_state = [None]
        # dialogue enhancement (de.py): the user gain in dB (0 = off) and
        # the lead-in frames' DE entries
        self.de_level = float(de_level)
        self.lead_de = []
        self.log = []               # human-readable notes, in order
        # damaged input (robust.py): frames concealed, by reason
        self.guard = RB.OutputGuard(REF_PEAK[element], FRAME) if guard \
            else None
        self.n_concealed = 0
        self.concealed = RB.DamageLog()
        # A-SPX level check (robust.py): one of the damaged-input checks,
        # switched with them (robust.CHECKS), not with the output guard;
        # None turns it off
        self.hf_ratio = RB.MAX_HF_RATIO
        self.hf_floor = RB.HF_MIN_FRAC * REF_PEAK[element]
        self._aspx_flag = np.zeros(0, dtype=bool)
        self.n_frames = 0
        self.frame = FRAME
        self.target_level = target_level
        self.drc_mode = drc
        self.leveler = None

    @property
    def rate(self):
        """(fs_index, frame_rate_index) of the stream, None until known."""
        return self.stream.rate

    @property
    def ratio(self):
        """The decoder resampling ratio (up, down) of Table 83 / 84."""
        r = self.stream.rate
        return RT.ratio(*r) if r else RT.ratio(1, 3)

    @property
    def out_rate(self):
        r = self.stream.rate
        return RT.out_rate(r[0]) if r else 48000

    def _adopt_frame(self):
        """The stream's frame length became known (or changed): the
        filterbank, QMF and delay-line state is reset to it."""
        fl = self.stream.frame_len
        if fl is None or fl == self.frame:
            return
        self.frame = fl
        self.fb_state = {k: None for k in self.chans}
        self.lead_pcm = {k: np.zeros(0) for k in self.lead_pcm}
        self.lead_core = {}
        self.lead_groups = {k: [] for k in self.lead_groups}
        self.lead_acpl = []
        self.lead_de = []
        self.hf_state = {k: [None] for k in self.hf_state}
        self.t_hold = {k: None for k in self.t_hold}
        self.acpl_state = [None]
        if self.guard is not None:
            self.guard.frame = fl

    @property
    def n_ok(self):
        return self.stream.n_ok

    @property
    def n_bad(self):
        return self.stream.n_bad

    @property
    def n_seq_jumps(self):
        return self.stream.n_seq_jumps

    def _pair_capable(self, gkey):
        """Can this pair carry A-SPX (then it goes through the QMF bank from
        its first frame on)? The L/R pair of a 5_X element (codec modes 1
        and 4), its Ls/Rs pair unless the stream is ASPX_ACPL_3, and a
        channel_pair_element with stereo_codec_mode 1."""
        mode = self.stream.mode
        if mode is None:
            return False
        if self.element == "pair":
            return gkey == "lr" and mode[1] >= 1
        if gkey == "lr":
            return "L" in self.chans
        if gkey == "c" and not SC.CENTRE_ASPX:
            return False
        if gkey == "sr" and self.stream.acpl1:
            return False                # ASPX_ACPL_1 / _2: no Ls/Rs A-SPX
        if self.acpl3 or self.stream.acpl3:
            return False
        ka = dict((g, p) for p, g in self.pairs)[gkey][0]
        return ka in self.chans and len(self.chans) > 2

    def decode_pass(self, frames):
        """-> float64 array (len(frames) * 1536, n_channels) at the internal
        rate (48000 * 960 / 1001 Hz), decoder scale.

        Never raises on bad input and always returns exactly 1536 samples
        per frame. A frame that breaks a rule of the bitstream is concealed
        (counted in n_bad and n_concealed, reasons in concealed.reasons):
        its spectrum is zeroed, so the previous frame's overlap fades out
        and the next good frame fades in through its window, and its A-SPX
        high band is muted with the envelope state held. A frame whose
        output still rises far above the running level is muted by the
        output guard (reason "guard.level")."""
        frames = list(frames)
        n_bad0 = self.stream.n_bad
        self._aspx_flag = np.zeros(len(frames), dtype=bool)
        try:
            y = self._decode_pass(frames)
            n_out = len(frames) * self.frame
            status = list(self.stream.status)
            self.log.extend(self.stream.log)
            self.stream.log.clear()
        except Exception as e:                                 # noqa: BLE001
            # last resort: nothing escapes, and the timeline holds
            self._adopt_frame()
            self.log.append(f"decode pass failed ({type(e).__name__}: "
                            f"{str(e)[:90]}) -- {len(frames)} frames muted")
            n_out = len(frames) * self.frame
            y = np.zeros((n_out, len(self.chans)))
            status = ["decoder.internal"] * len(frames)
            self.stream.n_bad = n_bad0 + len(frames)
            self.stream.damage.reasons["decoder.internal"] += len(frames)
        if y.shape != (n_out, len(self.chans)):
            self.log.append(f"pass gave {y.shape[0]} samples for "
                            f"{len(frames)} frames -- fixed to {n_out}")
            z = np.zeros((n_out, len(self.chans)))
            m = min(n_out, y.shape[0])
            z[:m, :y.shape[1]] = y[:m, :len(self.chans)]
            y = z
        if not np.isfinite(y).all():
            y = np.where(np.isfinite(y), y, np.inf)  # the guard mutes them
        if len(status) != len(frames):
            status = (status + [None] * len(frames))[:len(frames)]
        bad = np.array([s is not None for s in status], dtype=bool)
        fired = np.zeros(len(frames), dtype=bool)
        if self.guard is not None:
            fired = self.guard.process(y, force=bad)
        else:
            y[~np.isfinite(y)] = 0.0
        for s in status:
            if s is not None:
                self.concealed.add(s)
        hf_flag = self._aspx_flag[:len(frames)] & ~bad & ~fired
        for f in np.flatnonzero(fired & ~bad):
            self.concealed.add("guard.level")
        for f in np.flatnonzero(hf_flag):
            self.concealed.add("aspx.level")
        fired = fired | hf_flag
        self.n_concealed += int((bad | fired).sum())
        self.n_frames += len(frames)
        if self.target_level is not None:
            if self.leveler is None:
                from .loudness import Leveler
                up, down = self.ratio
                self.leveler = Leveler(self.chans, self.target_level,
                                       self.drc_mode,
                                       rate=self.out_rate * down / up)
            meta = list(self.stream.meta)
            meta = (meta + [None] * len(frames))[:len(frames)]
            y = self.leveler.process(y, meta, self.frame)
        self.last_status = status            # per frame of this pass
        self.last_concealed = bad | fired
        return y

    def _decode_pass(self, frames):
        chans = self.chans
        dec = self.stream
        wins, lens, grp, cnts = dec.decode_frames(frames, chans)
        self._adopt_frame()
        FRAME = self.frame
        pcm, self.fb_state = FB.synthesise_frames(
            wins, lens, cnts, FRAME, states=self.fb_state, return_state=True)
        if len(pcm) < len(chans):
            # ASPX_ACPL_3: a two-channel core (and the LFE) plus A-CPL;
            # C / Ls / Rs come from the upmix below
            if not self.acpl3:
                self.acpl3 = True
                self.log.append(
                    "stream is ASPX_ACPL_3 (2-ch core + A-CPL): "
                    f"{[k for k in chans if k not in pcm]} from the A-CPL "
                    "upmix")
            n_ref = len(next(iter(pcm.values())))
            for k in chans:
                if k not in pcm:
                    pcm[k] = np.zeros(n_ref)
        ns = {len(v) for v in pcm.values()}
        if len(ns) != 1:
            n_min = min(ns)
            self.log.append(f"channel lengths disagree {sorted(ns)} -- "
                            f"cropping to {n_min}")
            pcm = {k: v[:n_min] for k, v in pcm.items()}
        # Every channel leaves the decoder HF_DELAY samples late: a pair
        # that can carry A-SPX goes through the QMF bank (analysis, HF
        # generation and adjustment where there is A-SPX data, synthesis),
        # whose delay that is; every other channel (centre, LFE, a pair
        # without A-SPX) through a plain delay line of the same length.
        # TS 103 190-1 6.2.8 / 6.2.14 put each channel through the QMF bank,
        # so all of them are time-aligned.
        core = dict(pcm)
        via_qmf = set()
        # the A-CPL upmix: an ASPX_ACPL_3 stream decoded to 5.1
        upmix = ((self.acpl3 or dec.acpl3) and "C" in chans
                 and self.element == "5_X")
        # ASPX_ACPL_1 / _2: the pair element always (its only stereo);
        # the 5_X element decoded to 5.1
        a1 = dec.acpl1
        up1 = bool(a1 == "pair" or (a1 == "5_X" and "C" in chans
                                    and self.element == "5_X"))
        if up1:
            upmix = True
        ga = grp.get("acpl") or [None] * len(grp["lr"])
        gde = grp.get("de") or [None] * len(grp["lr"])
        de_frames = self.lead_de[-LEAD:] + gde
        self.lead_de = (self.lead_de + gde)[-LEAD:]

        def de_arg(gkey, up):
            """Dialogue enhancement on this render call (de.py): on the
            upmix's L / R / C, or on the pair's own channels."""
            if not self.de_level:
                return None
            pair = dict((g, p) for p, g in self.pairs)[gkey]
            if up:
                return dict(frames=de_frames, level=self.de_level, up=True)
            if gkey == "lr" and a1 == "pair":
                pair = ("L", "R")
            if not set(pair) & {"L", "R", "C"}:
                return None
            if gkey == "lr" and up1:
                return None
            return dict(frames=de_frames, level=self.de_level, pair=pair)
        for (ka, kb), gkey in self.pairs:
            # the centre (aspx_data_1ch) goes through the pair renderer as
            # the same channel twice: with balance 0 each half is decoded
            # alone, so either half is the mono rendering
            if ka not in chans:
                continue
            gl = grp.get(gkey) or [None] * len(grp["lr"])
            if self._pair_capable(gkey):
                lead = min(len(self.lead_pcm[ka]), LEAD * FRAME)
                pa = np.concatenate([self.lead_pcm[ka][-lead:], pcm[ka]])
                kb_ = "L" if (a1 == "pair" and gkey == "lr") else kb
                pb = np.concatenate([self.lead_pcm[kb_][-lead:], pcm[kb_]])
                a1ch = None
                if up1 and gkey == "lr":
                    rk = ("R",) if a1 == "pair" else ("Ls", "Rs")
                    a1ch = dict(res=[np.concatenate(
                        [self.lead_core.get(k, np.zeros(0))[-lead:], core[k]])
                        for k in rk],
                        split=a1 == "pair", five=a1 == "5_X")
                gg = self.lead_groups[gkey][-LEAD:] + gl
                n_lead = len(gg) - len(gl)
                hs, slog = self.hf_state[gkey], []
                up = upmix and gkey == "lr"
                if up:
                    gacpl = self.lead_acpl[-LEAD:] + ga
                    acs, alog = self.acpl_state, []
                try:
                    hf, _n = R37.apply_hf_pair(
                        pa, pb, gg, dec.cfg,
                        state0=hs[max(0, len(hs) - 1 - n_lead)],
                        state_log=slog,
                        acpl_groups=gacpl if up else None,
                        acpl_state0=(acs[max(0, len(acs) - 1 - n_lead)]
                                     if up else None),
                        acpl_log=alog if up else None,
                        frame_len=FRAME, t_start=self.t_hold[gkey],
                        acpl_1ch=a1ch, de=de_arg(gkey, up))
                    bad = np.zeros(len(gl) + 1, dtype=bool)
                    pcm[ka], pcm[kb], nb_, self.hf_hot[gkey] = \
                        core_for_bursts(hf["L"][lead:], hf["R"][lead:],
                                        pa, pb, lead, self.burst_limit,
                                        self.hf_hot[gkey],
                                        hf_ratio=(self.hf_ratio
                                                  if RB.CHECKS
                                                  else None),
                                        flagged=self._aspx_flag,
                                        hf_floor=self.hf_floor,
                                        bad_out=bad, FRAME=FRAME)
                    if "de_L" in hf:
                        # dialogue enhancement: the frames the checks kept
                        n = len(pcm[ka])
                        keep = ~np.repeat(bad, FRAME)[:n]
                        for k_, h_ in ((ka, "de_L"), (kb, "de_R")):
                            pcm[k_] = np.where(keep, hf[h_][lead:lead + n],
                                               pcm[k_])
                    via_qmf |= {ka, kb}
                    if up:
                        # the channels of the upmix; a frame whose
                        # A-SPX output the checks above replaced by its
                        # core plays the core (the stereo downmix) on
                        # L / R and nothing on C / Ls / Rs (a pair element
                        # in ASPX_ACPL_1 / _2: its mono core on both)
                        n = len(pcm[ka])
                        if a1 == "pair":
                            pcm[kb] = pcm[ka]
                        for k in hf["up"]:
                            v = hf["up"][k][lead:lead + n]
                            if k in (ka, kb):
                                v = np.where(np.repeat(bad, FRAME)[:n],
                                             pcm[k], v)
                            else:
                                v = np.where(np.repeat(bad, FRAME)[:n],
                                             0.0, v)
                            pcm[k] = v
                        via_qmf |= set(hf["up"])
                        self.acpl_state = alog[-(LEAD + 1):]
                    if nb_:
                        self.n_burst += nb_
                        self.log.append(
                            f"A-SPX burst on {ka}/{kb}: {nb_} frame(s) "
                            f"above {self.burst_limit:.3g} play core-band")
                    self.hf_state[gkey] = slog[-(LEAD + 1):]
                    nl = len((self.lead_groups[gkey] + gl)[-LEAD:])
                    tb = hf.get("t_before")
                    if tb:
                        self.t_hold[gkey] = tb[len(gg) - nl]
                except Exception as e:                         # noqa: BLE001
                    self.hf_state[gkey] = (
                        hs + [hs[-1]] * len(gl))[-(LEAD + 1):]
                    if up:
                        self.acpl_state = [None]
                    self.n_hf_fail += 1
                    self.log.append(
                        f"A-SPX pass failed ({type(e).__name__}: "
                        f"{str(e)[:90]}) -- this pass is core-band only")
                self.lead_pcm[ka] = pa[-LEAD * FRAME:]
                self.lead_pcm[kb] = pb[-LEAD * FRAME:]
            else:
                self.lead_pcm[ka] = pcm[ka][-LEAD * FRAME:]
                self.lead_pcm[kb] = pcm[kb][-LEAD * FRAME:]
                self.hf_state[gkey] = (
                    self.hf_state[gkey] + [self.hf_state[gkey][-1]]
                    * len(gl))[-(LEAD + 1):]
            self.lead_groups[gkey] = (self.lead_groups[gkey] + gl)[-LEAD:]
            if gkey == "lr":
                self.lead_acpl = (self.lead_acpl + ga)[-LEAD:]
        for k in chans:
            x = np.concatenate([self.core_tail[k], core[k]])
            if k not in via_qmf:
                pcm[k] = x[:len(core[k])]
            self.core_tail[k] = x[-HF_DELAY:]
            # the core of the last LEAD frames (ASPX_ACPL_1 residuals)
            self.lead_core[k] = np.concatenate(
                [self.lead_core.get(k, np.zeros(0)), core[k]])[-LEAD * FRAME:]
        return np.column_stack([pcm[k] for k in chans])


def probe_rate(frames, limit=64):
    """(fs_index, frame_rate_index) of the first frame whose TOC parses and
    whose next frame agrees, or None."""
    prev = None
    for f in frames[:limit]:
        r = RB.peek_rate(f)
        if r is not None and r == prev:
            return r
        prev = r
    return prev


ELEMENT_OF_CH_MODE = {0: "mono", 1: "pair", 2: "3_0", 3: "5_0", 4: "5_X",
                      5: "7_X", 6: "7_X", 7: "7_X", 8: "7_X", 9: "7_X",
                      10: "7_X"}


# A presentation_version 2 presentation whose channel-coded substream
# signals a 7.x channel mode (the public kit's "IMS" streams): its
# substream is a channel_pair_element (every frame of both kit streams
# closes on audio_size so, none as a 7_X_channel_element); the rendering
# to the signalled layout is not in TS 103 190-1/-2 V1.3.1, so its stereo
# core is decoded (README, Interpretations)
IMS_CH_MODES = (5, 6, 7, 8, 9, 10)
IMS_STEREO = 1


def probe_ch_mode(frames, limit=64, presentation=None, language=None):
    """The channel_mode (Table 88 ch_mode) of the audio substream of the
    presentation that would be decoded (toc.choose), or None."""
    from . import toc as M
    for f in frames[:limit]:
        try:
            st = M.parse(f)
        except Exception:                                      # noqa: BLE001
            continue
        pi = M.choose(st, presentation, language)
        sel = M.audio_substream(st, pi)
        if sel is None:
            continue
        cm = sel["channel_mode"]
        if (pi is not None and cm in IMS_CH_MODES
                and st["presentations"][pi].get("version") == 2):
            return IMS_STEREO
        return cm
    return None


def probe_element(frames, limit=64, presentation=None, language=None):
    """Guess the element from the channel_mode of the audio substream of
    the presentation that would be decoded (toc.choose).
    -> "mono", "pair", "3_0", "5_0", "5_X" (5.1), "7_X", or None if not
    recognised."""
    return ELEMENT_OF_CH_MODE.get(probe_ch_mode(frames, limit, presentation,
                                                language))


def decode(frames, element=None, channels=None, max_pass=1800,
           presentation=None, language=None, target_level=None, drc="auto",
           de_level=0.0):
    """Batch convenience: -> (pcm float64 (n, ch) at the external rate,
    Decoder.out_rate (48 kHz, or 44,1 kHz for fs_index 0), decoder scale,
    Decoder). The resampler ratio is the stream's (Table 83: 1001/960,
    25/24, 15/16 or 1; with ratio 1 it passes the samples through). Decodes in passes of `max_pass` frames, exactly as a
    streaming caller would. presentation / language: see toc.choose."""
    from .resample import StreamResampler
    element = element or probe_element(frames, presentation=presentation,
                                       language=language)
    rate = probe_rate(frames)
    up, down = (RT.ratio(*rate) if rate and RT.ratio(*rate)
                else RT.ratio(1, 3))
    if element is None:
        raise ValueError("cannot tell the element from the stream; pass "
                         "element='5_X' or 'pair'")
    ch_mode = (probe_ch_mode(frames, presentation=presentation,
                             language=language) if element == "7_X" else None)
    dec = Decoder(element, channels, presentation=presentation,
                  language=language, target_level=target_level, drc=drc,
                  ch_mode=ch_mode, de_level=de_level)
    rs = StreamResampler(up, down)
    out = []
    for i in range(0, len(frames), max(1, max_pass)):
        y = dec.decode_pass(frames[i:i + max_pass])
        if len(y):
            out.append(rs.push(y))
    n_ch = len(dec.chans)
    pcm = np.vstack(out) if out else np.zeros((0, n_ch))
    return pcm, dec
