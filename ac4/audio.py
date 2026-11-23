# SPDX-License-Identifier: Apache-2.0
"""M29 -- programme audio from broadcast AC-4: the five full-band channels.

M26 made the LFE audible: twelve spectral lines, a rumble, proof the chain from
RF to samples closes.  This is the actual programme -- centre, front pair,
surround pair -- reconstructed from the MDCT core that M28 decodes.

WHAT IS AND IS NOT RECONSTRUCTED HERE
--------------------------------------
  * **0 .. 12 kHz only.**  `max_sfb = 43` means the entropy-coded spectrum
    stops at line 768 of 1536 (M27).  Everything above is A-SPX's job -- a
    parametric high band that is NOT decoded here -- so the output is genuinely
    band-limited at 12 kHz.  That is not a defect in the reconstruction, it is
    the half of the signal this decoder does not yet cover.
  * **Long frames only.**  35 % of frames set `b_long_frame = 0` and need the
    short-block branch (Table 108 grouping).  Those come out as silence, so a
    straight render is choppy.  The longest unbroken run is written separately
    for an honest listen.
  * **No companding, no A-CPL** (A-CPL is not signalled in this stream).

MID/SIDE IS NOT OPTIONAL
-------------------------
Table 113: `sap_mode = 2` means M/S in ALL scale factor bands, and that is what
this stream uses in 98 % of frames.  So the two `sf_data` blocks of a pair are
sum and difference, not left and right:

    L = M + S      R = M - S

Skipping this step does not produce "slightly wrong stereo" -- it produces the
mid signal in one ear and the side signal in the other, which is why the two
blocks' energies are ANTI-correlated (M28 measures r = -0.59).

THE GATE THAT CANNOT BE FUDGED
-------------------------------
Only lines 0..767 were decoded, so the rendered audio must contain essentially
NO energy above 12 kHz -- and substantial energy below it.  That is a physical
consequence of what was decoded, checkable on the output alone, and it fails
loudly if the IMDCT size, the line mapping, or the overlap-add is wrong.  A
spectrum that leaks above 12 kHz means the reconstruction is broken even if the
file plays.

Second gate: L and R must be strongly correlated (same programme) but NOT
identical (real stereo separation).  A broken M/S step gives one or the other.
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from . import toc as M                                          # noqa: E402
from . import mdct as T22                                            # noqa: E402
from . import pcm as P                                               # noqa: E402
from . import sfb as B                                               # noqa: E402
from . import channels as C                                          # noqa: E402
from . import filterbank as FB                                       # noqa: E402

N = 1536
SF_OFFSET = P.SF_OFFSET            # 100, Pseudocode 21
QUANT_EXP = P.QUANT_EXP            # 4/3
FS = 48000
CODED_HZ = 12000.0                 # sfb_offset[43] / 1536 * 24000


def packed_spectrum(ch, fr, rng=None):
    """Dequantise into the PACKED (bitstream-order) spectrum; with `rng`
    (a SnfRandom) the spectral noise fill is applied (clause 5.1.4).

    For a long frame this is the 1536-line spectrum and nothing else happens.
    For a short frame the groups are laid out contiguously by
    `fr.sect_sfb_offset(g)`, several windows deep, and the per-window spectra
    only appear after ungrouping (Pseudocode 25).
    """
    offs = [fr.sect_sfb_offset(g) for g in range(fr.num_groups)]
    total = max(o[min(fr.max_sfb_g(g), len(o) - 1)]
                for g, o in enumerate(offs))
    spec = np.zeros(total)
    lines = ch["lines"]
    sfs_all = ch.get("sfs_all") or [ch["sfs"]]
    for g in range(fr.num_groups):
        sfs = sfs_all[g] if g < len(sfs_all) else None
        if sfs is None:
            continue
        for sfb in range(min(fr.max_sfb_g(g), len(sfs))):
            if sfs[sfb] is None:
                continue
            lo, hi = offs[g][sfb], offs[g][sfb + 1]
            hi = min(hi, len(lines), total)
            if hi <= lo:
                continue
            q = lines[lo:hi].astype(float)
            rec = np.sign(q) * np.abs(q) ** QUANT_EXP
            spec[lo:hi] = rec * (2.0 ** (0.25 * (sfs[sfb] - SF_OFFSET)))
    if rng is not None:
        noise_fill(spec, ch, fr, rng)
    return spec


def ungroup(packed, fr):
    """Pseudocode 25 -- bitstream order -> one spectrum per transform window.

    Grouped short blocks interleave their windows band by band, so a group of
    `num_win_in_group` windows stores each scale factor band `num_win` times in
    a row.  Undoing that is what turns the packed vector into the per-window
    spectra the IMDCT actually consumes.
    """
    lengths = [fr.length_g(fr.w2g[w]) for w in range(fr.num_windows)]
    win_off, acc = [], 0
    for L in lengths:
        win_off.append(acc)
        acc += L
    out = [np.zeros(L) for L in lengths]
    k = 0
    win = 0
    for g in range(fr.num_groups):
        base = C.SFB_TABLE[fr.length_g(g)]
        nwin = fr.nwin[g]
        for sfb in range(min(fr.max_sfb_g(g), len(base) - 1)):
            lo, hi = base[sfb], base[sfb + 1]
            for w in range(nwin):
                n = hi - lo
                if k + n > len(packed):
                    return out, lengths
                out[win + w][lo:hi] = packed[k:k + n]
                k += n
        win += nwin
    return out, lengths


# ---------------------------------------------------------------- noise fill
_RNT = None


def random_noise_table():
    """Table C.10, RANDOM_NOISE_TABLE: 256 float32 values."""
    global _RNT
    if _RNT is None:
        import json
        path = os.path.join(HERE, "tables", "ac4_random_noise.json")
        with open(path, encoding="utf-8") as fh:
            vals = json.load(fh)["values"]
        if len(vals) != 256:
            raise ValueError("RANDOM_NOISE_TABLE must have 256 entries")
        _RNT = np.array(vals, dtype=np.float32)
    return _RNT


class SnfRandom:
    """nRndStateSnf: the ssf_rndgen_state of the spectral noise fill.

    reset (Pseudocode 24, ResetRandGenStateSnf) from sequence_counter at the
    start of each frame; next() is GetRandomNoiseValue (Pseudocode 57): the
    float32 sum of two RANDOM_NOISE_TABLE entries, then the state update
    with every field a uint8 (the spec's "implicit modulo 256"). The
    pseudocode's `x = x++` statements are read as increments."""

    def __init__(self, sequence_counter):
        self.tab = random_noise_table()
        start = 255 * (sequence_counter % 256)
        a = start % 255
        b = (start // 255) % 256
        state = ((a + 1) * ((a + 2) + 2 * b) // 2
                 + 255 * b * (255 + b) // 2)
        if start % 130560 >= 65280:
            state += 128
        self.a, self.b = a & 255, b & 255
        self.state = state % 256
        self.cur = (a * (a + 1) // 2 + b * 32386) % 256

    def next(self):
        v = self.tab[self.cur] + self.tab[self.state]   # float32 + float32
        self.a = (self.a + 1) & 255
        self.state = (self.state + 1) & 255
        if self.a == 255:
            self.cur = (self.cur + 1) & 255
            self.b = (self.b + 1) & 255
            self.a = 0
        self.cur = (self.cur + self.a) & 255
        self.state = (self.state + self.b + self.a) & 255
        return float(v)

    def skip(self, n):
        for _ in range(n):
            self.next()


def _snf_bands(ch, fr):
    """(g, sfb, lo, hi, delta) of every band in bitstream order of the
    packed spectrum, delta = dpcm_snf - 17 or None where none was sent."""
    snf = ch.get("snf") or []
    for g in range(fr.num_groups):
        offs = fr.sect_sfb_offset(g)
        d = snf[g] if g < len(snf) else []
        for sfb in range(min(fr.max_sfb_g(g), len(offs) - 1)):
            yield (g, sfb, offs[sfb], offs[sfb + 1],
                   d[sfb] if sfb < len(d) else None)


def noise_fill(spec, ch, fr, rnd):
    """TS 103 190-1 5.1.4, Pseudocodes 22 and 23, on the PACKED spectrum
    (bitstream order: band sfb of group g holds its num_win_in_group[g]
    windows back to back, which is how both pseudocodes walk k).

    Bands with no energy (sfb_cb 0 or max_quant_idx 0: exactly the bands
    that carry a dpcm_snf) are replaced by noise at previous_rms + delta,
    previous_rms being log2 of the mean square of the last band with
    energy, or of the last noise band (Pseudocode 23); the first band with
    energy sets the initial reference (Pseudocode 22, -1000 if none).
    delta -17 is the escape: no noise, level not updated. With spec=None
    only the generator advances (a channel that is not rendered)."""
    if not ch.get("snf"):
        return
    bands = list(_snf_bands(ch, fr))
    prev = -1000.0
    if spec is not None:
        for _g, _sfb, lo, hi, _d in bands:
            e = float(np.sum(spec[lo:hi] ** 2))
            if e > 0:
                prev = 1.44269504 * math.log(e / (hi - lo))
                break
    for _g, _sfb, lo, hi, d in bands:
        if hi <= lo:
            continue
        if spec is not None:
            e = float(np.sum(spec[lo:hi] ** 2))
            if e > 0:
                prev = 1.44269504 * math.log(e / (hi - lo))
                continue
        elif d is None:
            continue                    # a band with energy
        if d is None or d == -17:       # -17: no noise, level held
            continue
        prev = prev + d
        if spec is None:
            rnd.skip(hi - lo)
            continue
        amp = 2.0 ** (0.5 * prev)
        for k in range(lo, hi):
            spec[k] = rnd.next() * amp


# the element's sf_data blocks in bitstream order (Table 22 / Table 25):
# the noise fill generator runs through them in this order every frame
BITSTREAM_ORDER = {
    "pair": ("L", "R"),
    "5_X": ("lfe", "L", "R", "Ls", "Rs", "C"),
    "acpl3": ("lfe", "L", "R"),
}


def unmix(m, s, st, fr):
    """Undo MDCT stereo processing on the PACKED spectrum.  Table 113."""
    mode = st.get("sap_mode")
    if mode == 2:                                   # M/S in every band
        return m + s, m - s
    if mode == 1 and st.get("ms_used"):             # M/S in the flagged bands
        l, r = m.copy(), s.copy()
        used = st["ms_used"]                        # per window group
        for g in range(min(fr.num_groups, len(used))):
            off = fr.sect_sfb_offset(g)
            for sfb, on in enumerate(used[g]):
                if not on or sfb + 1 >= len(off):
                    continue
                lo, hi = off[sfb], min(off[sfb + 1], len(m))
                if hi > lo:
                    l[lo:hi] = m[lo:hi] + s[lo:hi]
                    r[lo:hi] = m[lo:hi] - s[lo:hi]
        return l, r
    if mode == 3 and st.get("sap"):                 # full SAP, clause 5.3.2
        l, r = m.copy(), s.copy()
        coeffs = sap_coeffs(st["sap"])
        for g in range(min(fr.num_groups, len(coeffs))):
            off = fr.sect_sfb_offset(g)
            for sfb, abcd in enumerate(coeffs[g]):
                if abcd is None or sfb + 1 >= len(off):
                    continue                        # a=d=1, b=c=0: identity
                lo, hi = off[sfb], min(off[sfb + 1], len(m))
                if hi > lo:
                    a, b, c, d = abcd
                    l[lo:hi] = a * m[lo:hi] + b * s[lo:hi]
                    r[lo:hi] = c * m[lo:hi] + d * s[lo:hi]
        return l, r
    # sap_mode 0 = no processing
    return m, s


def sap_alpha_q(sap):
    """Pseudocode 59, the sap_mode 3 branch: the quantised prediction
    coefficients alpha_q[g][sfb] (None where sap_used is 0).

    Even bands carry a delta (sap_hcw symbol - 60) against band sfb - 2 of
    the same group, or against the same band of group g - 1 when
    delta_code_time is set, g > 0 and max_sfb did not change from the
    previous group; band 0 without time coding takes the delta itself.
    Odd bands copy the even band below them.

    The pseudocode does not say what an unused band's alpha_q is when a
    later band (or the next group) refers to it. Here it is 0. That is an
    inference, not spec text. It does occur on air: about 4,000 such
    references in 18 minutes of one broadcaster's 5.1 (none on the other's).
    Of three candidate rules it gives the most continuous alpha from frame
    to frame: the median change against the same band in the neighbouring
    frames is 2 steps with 0, against 4 for "carry the last coded band" and
    4 for "keep last frame's value" (1 step on bands with no such reference).
    """
    used, dpcm = sap["coeff_used"], sap["dpcm"]
    max_sfb = sap["max_sfb"]
    dct = sap["delta_code_time"]
    out = []
    prev_q = None
    max_sfb_prev = max_sfb[0] if max_sfb else 0
    for g, m in enumerate(max_sfb):
        q = [0] * m
        o = [None] * m
        for sfb in range(m):
            if not used[g][sfb]:
                continue
            if sfb % 2:
                q[sfb] = q[sfb - 1]
            else:
                delta = dpcm[g][sfb] - 60
                code_delta = 0 if (g == 0 or m != max_sfb_prev) else dct
                if code_delta:
                    q[sfb] = prev_q[sfb] + delta
                elif sfb == 0:
                    q[sfb] = delta
                else:
                    q[sfb] = q[sfb - 2] + delta
            o[sfb] = q[sfb]
        out.append(o)
        prev_q = q
        max_sfb_prev = m
    return out


def sap_coeffs(sap):
    """Clause 5.3.2 -> per group, per band: (a, b, c, d) or None (identity).

    sap_gain = alpha_q * 0.1;  a = 1 + sap_gain, b = 1, c = 1 - sap_gain,
    d = -1, and clause 5.3.3.2 applies [O0; O1] = [a b; c d] x [I0; I1].
    """
    return [[None if q is None else (1.0 + 0.1 * q, 1.0, 1.0 - 0.1 * q, -1.0)
             for q in grp] for grp in sap_alpha_q(sap)]


def write_wav(path, chans, rate=FS):
    """Interleaved 16-bit PCM, one or more channels."""
    x = np.stack(chans, axis=1) if len(chans) > 1 else chans[0][:, None]
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2").tobytes()
    nch = x.shape[1]
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, nch, rate,
                            rate * 2 * nch, 2 * nch, 16))
        f.write(b"data" + struct.pack("<I", len(pcm)) + pcm)


def band_energy(x, lo, hi):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fq = np.fft.rfftfreq(len(x), 1.0 / FS)
    m = (fq >= lo) & (fq < hi)
    return float((X[m] ** 2).sum()), float((X ** 2).sum())
