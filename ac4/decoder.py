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

What it does NOT do (stage 1): A-CPL upmix (an ASPX_ACPL_3 5.1 stream gives
L/R; the other four channels are silent), A-SPX on the centre channel, full
SAP (sap_mode 3), companding, DRC / dialogue / loudness metadata,
presentation or substream choice (substream 1 is always decoded), frame
rates other than 29.97 and sample rates other than 48 kHz.

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
from .stream import Ac4Stream

FRAME = 1536                # internal samples per frame (29.97 fps, 48 kHz)
LEAD = 4                    # frames of A-SPX/QMF lead-in carried per pair
HF_DELAY = 577              # samples the A-SPX path delays a channel
HF_TAIL = 2                 # frames after a burst frame that also play core

# The receiver's fixed output reference per element, in decoder units: the
# level written at 0.9 of full scale. Hand-tuned; the decoder's true full
# scale is NOT established. Used for the default burst limit and by the CLI.
REF_PEAK = {"5_X": 9.0e5, "pair": 2.0e5}


def core_for_bursts(hf_a, hf_b, core_a, core_b, lead, limit, owed):
    """One pair's A-SPX output with the core put back wherever A-SPX ran
    away. -> (a, b, frames replaced, frames still owed to the next pass)."""
    n = min(len(hf_a), len(hf_b)) // FRAME
    if not n:
        return hf_a, hf_b, 0, owed
    pk = np.maximum(np.abs(hf_a[:n * FRAME]).reshape(n, FRAME).max(axis=1),
                    np.abs(hf_b[:n * FRAME]).reshape(n, FRAME).max(axis=1))
    hot = pk > limit
    bad = np.zeros(n, dtype=bool)
    left = int(owed)
    for f in range(n):
        if hot[f]:
            bad[f], left = True, HF_TAIL
        elif left:
            bad[f], left = True, left - 1
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

    KNOWN (inherited, measured): the output is NOT fully independent of
    where the batches are cut. Before the first A-SPX configuration
    arrives, a batch that also contains that configuration delays its
    core-only frames by 577 samples and one that does not, does not; and
    the last frame of a batch can differ from a one-batch decode in its
    last 1279 samples (up to about 1 % of peak on one stereo stream,
    about -100 dB on others). Decode in the same batch sizes to get the
    same samples.

    element   "5_X" (5.1 channel element) or "pair" (channel_pair_element)
    channels  2 -> L, R;  6 -> L, R, C, LFE, Ls, Rs (WAV 5.1 order).
              A pair element always gives 2.
    """

    def __init__(self, element="5_X", channels=None, burst_limit=None):
        if element not in ("5_X", "pair"):
            raise ValueError(f"element {element!r}: '5_X' or 'pair'")
        if channels is None:
            channels = 6 if element == "5_X" else 2
        if channels not in (2, 6):
            raise ValueError("channels must be 2 or 6")
        if element == "pair":
            channels = 2
        self.element = element
        self.chans = Ac4Stream.CHANS if channels == 6 else ("L", "R")
        self.burst_limit = (2.0 * REF_PEAK[element] if burst_limit is None
                            else float(burst_limit))
        self.stream = Ac4Stream(element=element)
        self.fb_state = {k: None for k in self.chans}
        self.lead_pcm = {k: np.zeros(0) for k in ("L", "R", "Ls", "Rs")}
        self.lead_groups = {"lr": [], "sr": []}
        self.hf_state = {"lr": [None], "sr": [None]}
        self.hf_hot = {"lr": 0, "sr": 0}
        self.n_burst = 0
        self.n_hf_fail = 0
        self.acpl3 = False
        self.log = []               # human-readable notes, in order

    @property
    def n_ok(self):
        return self.stream.n_ok

    @property
    def n_bad(self):
        return self.stream.n_bad

    def decode_pass(self, frames):
        """-> float64 array (len(frames) * 1536, n_channels) at the internal
        rate (48000 * 960 / 1001 Hz), decoder scale. A frame that fails to
        decode gives silence of the right length (counted in n_bad)."""
        chans = self.chans
        dec = self.stream
        wins, lens, grp, cnts = dec.decode_frames(frames, chans)
        pcm, self.fb_state = FB.synthesise_frames(
            wins, lens, cnts, FRAME, states=self.fb_state, return_state=True)
        if len(pcm) < len(chans):
            # ASPX_ACPL_3: a two-channel core plus A-CPL (upmix not built)
            if not self.acpl3:
                self.acpl3 = True
                self.log.append(
                    "stream is ASPX_ACPL_3 (2-ch core + A-CPL): rendering L/R; "
                    f"{[k for k in chans if k not in pcm]} written silent")
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
        if dec.cfg is not None:
            for (ka, kb), gkey in ((("L", "R"), "lr"), (("Ls", "Rs"), "sr")):
                if ka not in chans:
                    continue
                gl = grp[gkey]
                if any(g is not None for g in gl):
                    lead = min(len(self.lead_pcm[ka]), LEAD * FRAME)
                    pa = np.concatenate([self.lead_pcm[ka][-lead:], pcm[ka]])
                    pb = np.concatenate([self.lead_pcm[kb][-lead:], pcm[kb]])
                    gg = self.lead_groups[gkey][-LEAD:] + gl
                    n_lead = len(gg) - len(gl)
                    hs, slog = self.hf_state[gkey], []
                    try:
                        hf, _n = R37.apply_hf_pair(
                            pa, pb, gg, dec.cfg,
                            state0=hs[max(0, len(hs) - 1 - n_lead)],
                            state_log=slog)
                        pcm[ka], pcm[kb], nb_, self.hf_hot[gkey] = \
                            core_for_bursts(hf["L"][lead:], hf["R"][lead:],
                                            pa, pb, lead, self.burst_limit,
                                            self.hf_hot[gkey])
                        if nb_:
                            self.n_burst += nb_
                            self.log.append(
                                f"A-SPX burst on {ka}/{kb}: {nb_} frame(s) "
                                f"above {self.burst_limit:.3g} play core-band")
                        self.hf_state[gkey] = slog[-(LEAD + 1):]
                    except Exception as e:                     # noqa: BLE001
                        self.hf_state[gkey] = (
                            hs + [hs[-1]] * len(gl))[-(LEAD + 1):]
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
        return np.column_stack([pcm[k] for k in chans])


def probe_element(frames, limit=64):
    """Guess the element from the table of contents' channel_mode.
    -> "5_X" for 5.1, "pair" for stereo, None if not recognised."""
    from . import toc as M
    for f in frames[:limit]:
        try:
            st = M.parse(f)
        except Exception:                                      # noqa: BLE001
            continue
        cms = st.get("channel_modes") or []
        if not cms:
            continue
        cm = cms[-1]
        if cm == 1:
            return "pair"
        if cm == 4:
            return "5_X"
        return None
    return None


def decode(frames, element=None, channels=None, max_pass=1800):
    """Batch convenience: -> (pcm float64 (n, ch) at 48 kHz, decoder scale,
    Decoder). Decodes in passes of `max_pass` frames, exactly as a
    streaming caller would."""
    from .resample import StreamResampler
    element = element or probe_element(frames)
    if element is None:
        raise ValueError("cannot tell the element from the stream; pass "
                         "element='5_X' or 'pair'")
    dec = Decoder(element, channels)
    rs = StreamResampler()
    out = []
    for i in range(0, len(frames), max(1, max_pass)):
        y = dec.decode_pass(frames[i:i + max_pass])
        if len(y):
            out.append(rs.push(y))
    n_ch = len(dec.chans)
    pcm = np.vstack(out) if out else np.zeros((0, n_ch))
    return pcm, dec
