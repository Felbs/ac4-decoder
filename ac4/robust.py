# SPDX-License-Identifier: Apache-2.0
"""Damaged-input handling: what the bitstream lets a decoder check, and the
last-resort output guard.

AC-4 carried in MP4 has no frame CRC, and every Huffman codebook is a
complete prefix code, so random bits always "decode". A damaged frame is
caught only by structure the standard fixes. The checks, all raising
`Damaged` with a short reason code (the frame is then concealed):

TABLE OF CONTENTS (TS 103 190-1 4.2.3, -2 6.2.1)
  toc.parse       the TOC itself does not parse
  toc.version     bitstream_version differs from the stream's
  toc.seq_range   sequence_counter > 1020 (4.3.3.2.2: never written)
  toc.rate        fs_index / frame_rate_index differ from the stream's
  toc.shape       presentation/substream layout (channel modes, number of
                  substreams) differs from the stream's
  toc.sizes       the substreams do not fit in the frame:
                  toc + payload_base + sum(substream_size) > frame length,
                  or a substream of size 0. (Equality does not hold on air:
                  frames carry trailing bytes, up to ~190 measured.)
  mp4.size        an MP4 sample too short to hold a TOC, or larger than any
                  AC-4 frame this decoder accepts (MAX_FRAME_BYTES)

AUDIO SUBSTREAM (4.2.4.2, 4.3.4.1)
  audio_size.overrun   audio_size points past the substream
  audio_size.closure   the parse does not end where audio_size says:
                       past the end, or (when every block of audio_data()
                       was parsed) more than 7 bits before it (byte_align)
  audio_size.late      the same, for a frame from before the stream's
                       first aspx_config, checked once an I-frame later in
                       the same decode pass brings it (Ac4Stream.
                       _late_closure; only when >= 90 % of such frames close)
  element.mode         the element's codec mode differs from the stream's,
                       other than on an I-frame that continues the sequence
                       counter or marks a splice (a foreign frame)
  element.parse        the element does not parse (ran out of bits, a
                       mode this decoder does not implement)

ASF CORE (4.2.8, 5.1.2, 5.1.3)
  asf.sect_cb     a section codebook of 12..15 ("shall not be used")
  asf.sect_len    sections that do not tile 0..max_sfb exactly
  asf.max_sfb     max_sfb above the number of bands for the transform
  asf.quant       a quantised line above 8191 (5.1.3.1 NOTE 1; i.e. an
                  ext_code with more than 8 leading ones)
  asf.sf_range    a scale factor outside 0..255 (5.1.3.2 NOTE)
  asf.level       a dequantised line above MAX_SPECTRAL_LINE (a decoder
                  sanity bound, 37 dB above anything measured on air)

A-SPX (4.2.10, 4.3.10, 5.7.6)
  aspx.config     aspx_config gives no valid master subband table
  aspx.num_env    more envelopes than Table 127 allows (4 FIXFIX, 5 others)
  aspx.borders    envelope borders not strictly increasing inside the
                  interval
  aspx.parse      the A-SPX data does not parse

A-CPL (4.2.13; ASPX_ACPL_3 frames)
  acpl.parse      acpl_data_2ch() does not parse. It is read for bit
                  accounting only (the upmix is not built), so that these
                  frames close exactly on audio_size as well

SEQUENCE (4.3.3.2.2): a counter that is neither prev+1, nor the 1020 -> 1
wrap, nor a frame after a splice (prev 0) marks a source change or lost
frames. That alone is not damage (frames do go missing on air; the frame
decodes, counted in n_seq_jumps). But with one frame of lookahead:
  seq.outlier     the NEXT frame continues the counter from BEFORE this
                  one, and not from this one: this frame is swapped,
                  repeated, or from another stream, and is concealed.
                  (Not decidable for the last frame of a decode pass.)

LEVEL CHECKS. Some damage keeps every rule (a flipped bit in an ext_code
value or a fixed-width field changes a value, not a bit count), and
spec-valid values still allow absurd levels (a scale factor of 255 is a
gain of 2**38.75). Two checks on the decoded signal catch what is left;
both are tuned so that they do not fire on any clean material measured
(see the README):
  aspx.level      the A-SPX output of a channel pair is more than
                  MAX_HF_RATIO x the RMS of its core (the frame plays its
                  core band; Decoder / core_for_bursts)
  guard.level     the frame's peak is far above a running reference of
                  accepted frames (muted with short ramps; OutputGuard)
"""
from __future__ import annotations

import collections

import numpy as np

# 5.1.3.1 NOTE 1: "The maximum allowed value of the elements of quant_spec
# is 8 191."  ext_decode returns 2**(N+4) + val with val < 2**(N+4), so
# N_ext <= 8.
MAX_QUANT = 8191
MAX_EXT_N = 8
SF_MIN, SF_MAX = 0, 255                    # 5.1.3.2 NOTE
# Table 127: the most envelopes an A-SPX interval may carry
MAX_NUM_ENV = {0: 4, 1: 5, 2: 5, 3: 5}     # FIXFIX, FIXVAR, VARFIX, VARVAR
# A sanity bound on one MP4 sample. AC-4 at 48 kHz / 29.97 fps never needs
# more than a few kB per frame; 64 kB is far above any real bitrate (this
# is a decoder limit, not a figure from the standard).
MAX_FRAME_BYTES = 65536
MIN_FRAME_BYTES = 4
# A sanity bound on one dequantised spectral line, in decoder units. The
# standard allows far more (8191**(4/3) * 2**(155/4) ~ 7e16); the largest
# line measured on clean material from both broadcasters is 1.4e7, at an
# output peak near the decoder's working full scale. 1e9 is 37 dB above
# that: a line this large cannot come from a sane encoder (MEASURED bound,
# not a figure from the standard).
MAX_SPECTRAL_LINE = 1e9

# A-SPX level check: a frame whose A-SPX output RMS (one channel of a pair)
# is above this many times the RMS of the core it was regenerated from plays
# its core band instead (the A-SPX envelope state is delta coded in time,
# so one damaged frame that slips through can leave the high band wrong
# for several frames). Frames whose A-SPX output peaks below
# HF_MIN_FRAC x the element's reference level are not judged (over a
# near-silent core the ratio means nothing: up to 7.5e6 measured on clean
# frames, all with an A-SPX peak below 0.012 x the reference). MEASURED on
# clean material (about 92 min, both broadcasters): at most 7.43 above
# 0.03 x the reference; 16 leaves a 2.15x margin.
MAX_HF_RATIO = 16.0
HF_MIN_FRAC = 0.03

# See Ac4Stream._late_closure.
LATE_MAJORITY = 0.9


class TrackingState(dict):
    """A-SPX framing state that notes when a previous_stop_pos it was asked
    for is unknown (aspx_parse.var_borders reads it with .get)."""

    missing = False

    def get(self, key, default=None):
        if key not in self:
            self.missing = True
        return super().get(key, default)


# The bitstream checks can be switched off -- ONLY so tests can show that
# their gates fail without them (negative controls). With CHECKS False the
# decoder behaves as it did before these checks existed.
CHECKS = True


class _Concealed(tuple):
    """The marker a concealed frame leaves in the A-SPX group lists (an
    empty tuple, so code that skips frames without A-SPX data skips it)."""

    def __repr__(self):
        return "CONCEALED"


CONCEALED = _Concealed()


class Damaged(ValueError):
    """A frame that breaks a rule of the bitstream. `reason` is a short
    code (see the module docstring)."""

    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def reason_of(exc):
    """Reason code for any exception raised while decoding one frame."""
    if isinstance(exc, Damaged):
        return exc.reason
    return "element.parse"


# ------------------------------------------------------------ sequence
def seq_continues(cur, prev):
    """4.3.3.2.2 'normal operation' -> True; False = source change."""
    if prev is None:
        return True
    if cur == prev + 1:
        return True
    if cur == 1 and prev == 1020:
        return True
    if cur != 0 and prev == 0:
        return True
    return False


def seq_next(c):
    """The counter that follows c in normal operation."""
    if c is None:
        return None
    return 1 if c == 1020 else c + 1


def seq_prev_of(c):
    """The counter c follows in normal operation (0 for a counter of 1 is
    ambiguous with the wrap; 1020 is chosen, which seq_continues accepts
    either way)."""
    if c is None:
        return None
    return 1020 if c <= 1 else c - 1


def peek_counter(frame):
    """sequence_counter of a frame without decoding it (None if the TOC
    cannot be read). Bits 2..11 when bitstream_version < 3."""
    if len(frame) < 2:
        return None
    v = (frame[0] << 8) | frame[1]
    if (v >> 14) == 3:
        return None             # extended bitstream_version: not peeked
    return (v >> 4) & 0x3FF


# ------------------------------------------------------------ TOC
def toc_shape(st):
    """What must stay the same from frame to frame of one stream."""
    return (st.get("bitstream_version"), st.get("fs_index"),
            st.get("frame_rate_index"), st.get("n_presentations"),
            tuple(st.get("channel_modes") or ()), st.get("n_substreams"))


def check_toc(st, frame, ref_shape):
    """Raise Damaged when the TOC cannot belong to this stream."""
    if st["sequence_counter"] > 1020:
        raise Damaged("toc.seq_range", str(st["sequence_counter"]))
    sizes = st.get("substream_sizes") or []
    if len(sizes) < 2 or len(sizes) != st.get("n_substreams"):
        raise Damaged("toc.shape", f"substream sizes {sizes}")
    total = st["toc_bytes"] + st.get("payload_base", 0) + sum(sizes)
    if total > len(frame) or min(sizes) < 1:
        raise Damaged("toc.sizes", f"{total} > {len(frame)}")
    if ref_shape is not None:
        shape = toc_shape(st)
        if shape[0] != ref_shape[0]:
            raise Damaged("toc.version")
        if shape[1:3] != ref_shape[1:3]:
            raise Damaged("toc.rate")
        if shape[3:] != ref_shape[3:]:
            raise Damaged("toc.shape")


def check_frame_size(frame):
    n = len(frame)
    if n < MIN_FRAME_BYTES or n > MAX_FRAME_BYTES:
        raise Damaged("mp4.size", str(n))


# ------------------------------------------------------------ closure
def check_closure(r, sub_len, complete):
    """audio_size against where the parse ended.

    r["audio_start"] is the bit where audio_data() starts, r["audio_size"]
    its length in bytes, r["end"] the bit where our parse stopped.
    `complete` = every block of audio_data() was parsed (otherwise only the
    upper bound holds)."""
    start, size = r["audio_start"], r["audio_size"]
    stop = start + 8 * size
    if stop > 8 * sub_len:
        raise Damaged("audio_size.overrun", f"{stop} > {8 * sub_len}")
    end = r["end"]
    if end > stop:
        raise Damaged("audio_size.closure", f"parse ended {end - stop} "
                      "bits past audio_size")
    if complete and stop - end > 7:
        raise Damaged("audio_size.closure", f"{stop - end} bits left "
                      "before audio_size")


# ------------------------------------------------------------ output guard
class OutputGuard:
    """Last-resort level check on the decoded PCM, per frame.

    ref     a running reference: the largest per-frame peak accepted so far,
            released slowly (RELEASE per frame), never below FLOOR_FRAC of
            the element's start level. It starts at the larger of
            INIT_FRAC x the element's start level and the KTH largest of the
            first WARM frame peaks of the first pass (so a burst of up to
            KTH - 1 frames there cannot raise it), and releases from there.
    fire    a frame whose peak exceeds RATIO x ref.

    A frame that fires is muted: a short linear ramp from the last output
    sample down to zero at its start, zeros, and the next frame that does
    not fire ramps back in over RAMP samples. Frames that fire, and frames
    concealed upstream, do not raise the reference.

    WHY AN ABSOLUTE START. A decoder that starts during a quiet passage has
    no idea how loud the programme gets: simulated starts at every 250th
    frame of the clean material reached 3.6 x a purely relative reference
    when loud audio followed 10 s later. Starting the reference at
    INIT_FRAC of the element's level, and letting it release, removes that.

    MEASURED on clean material (136,000 frames, about 76 min, both
    broadcasters, 5.1 and stereo; 549 simulated stream starts; see the
    README): no frame peak above 1.76 x this reference. RATIO 2.5 leaves
    1.42x (3.1 dB) margin over that: a trade between never touching clean
    audio and catching damage the bitstream cannot show (for example a
    flipped bit in an ext_code value, which changes a line's size but no
    bit count). These are choices from those measurements, not figures
    from the standard.
    """

    RATIO = 2.5
    RELEASE = 0.9999          # per frame: about -0.03 dB per second
    FLOOR_FRAC = 0.05
    INIT_FRAC = 0.75
    WARM = 30
    KTH = 5
    RAMP = 256

    def __init__(self, start_level, frame=1536):
        self.frame = frame
        self.floor = self.FLOOR_FRAC * float(start_level)
        self.init = self.INIT_FRAC * float(start_level)
        self.ref = None
        self.last = None          # last output sample per channel
        self.muted = False        # was the previous frame muted?
        self.n_fired = 0

    def limit(self):
        return self.RATIO * max(self.ref or 0.0, self.floor)

    def process(self, y, force=None):
        """y: (n_frames * frame, ch), modified in place where frames fire.
        force: optional bool per frame, frames concealed upstream (they do
        not move the reference).
        -> bool array, True where this guard muted a frame."""
        F = self.frame
        n = len(y) // F
        fired = np.zeros(n, dtype=bool)
        if n == 0:
            return fired
        with np.errstate(invalid="ignore"):
            pk = np.abs(y[:n * F]).reshape(n, F, -1).max(axis=(1, 2))
        pk = np.where(np.isfinite(pk), pk, np.inf)
        if self.ref is None:
            ok = pk[:self.WARM]
            if force is not None:
                ok = ok[~np.asarray(force[:self.WARM], dtype=bool)]
            ok = np.sort(ok[np.isfinite(ok)])[::-1]
            seed = float(ok[min(self.KTH, len(ok)) - 1]) if len(ok) else 0.0
            self.ref = max(seed, self.init, self.floor)
        ramp_in = np.linspace(0.0, 1.0, self.RAMP, endpoint=False)[:, None]
        for f in range(n):
            seg = y[f * F:(f + 1) * F]
            forced = bool(force[f]) if force is not None else False
            if pk[f] > self.limit():
                fired[f] = True
                self.n_fired += 1
                seg[:] = 0.0
                if not self.muted and self.last is not None:
                    m = min(self.RAMP, F)
                    seg[:m] = (self.last[None, :]
                               * np.linspace(1.0, 0.0, m)[:, None])
                self.muted = True
            else:
                if self.muted:
                    m = min(self.RAMP, F)
                    seg[:m] *= ramp_in[:m]
                    self.muted = False
                if forced:
                    self.ref = max(self.ref * self.RELEASE, self.floor)
                else:
                    self.ref = max(self.ref * self.RELEASE, float(pk[f]),
                                   self.floor)
            self.last = np.array(seg[-1], dtype=float)
        return fired


class DamageLog:
    """Counts of concealed frames by reason."""

    def __init__(self):
        self.reasons = collections.Counter()

    def add(self, reason):
        self.reasons[reason] += 1
