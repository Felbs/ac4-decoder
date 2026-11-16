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
    """Dequantise into the PACKED (bitstream-order) spectrum.

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


def spectrum(ch, max_sfb, rng=None):
    """One decoded channel -> a 1536-line dequantised spectrum.

    Clause 5.1.3.2 / Pseudocode 21, both verified verbatim:
        rec_spec    = sign(quant_spec) * |quant_spec| ** (4/3)
        sf_gain     = pow(2.0, 0.25 * (scale_factor - 100))

    With `rng`, spectral noise fill (clause 5.1.4, Pseudocode 22/23) is applied
    on top.  Bands the encoder left empty are meant to be REPLACED by noise at
    a transmitted level, not left as digital silence -- silence leaves audible
    spectral holes where the codec intended noise.
    """
    spec = np.zeros(N)
    lines, sfs = ch["lines"], ch["sfs"]
    off = B.SFB_OFFSET_1536
    for sfb in range(min(max_sfb, len(sfs))):
        if sfs[sfb] is None:
            continue
        lo, hi = off[sfb], off[sfb + 1]
        if lo >= len(lines):
            break
        q = lines[lo:min(hi, len(lines))].astype(float)
        rec = np.sign(q) * np.abs(q) ** QUANT_EXP
        spec[lo:lo + len(rec)] = rec * (2.0 ** (0.25 * (sfs[sfb] - SF_OFFSET)))
    if rng is not None:
        noise_fill(spec, ch, max_sfb, rng)
    return spec


def noise_fill(spec, ch, max_sfb, rng):
    """Clause 5.1.4.  Pseudocode 22 finds the reference level, 23 inserts.

    previous_rms is log2 of the band's MEAN SQUARE (1.44269504 = 1/ln 2), the
    transmitted delta is added in that log domain, and the amplitude is
    2 ** (0.5 * noise_rms) -- i.e. an RMS, which is why the exponent is halved.

    HONEST DEVIATION: the spec's GetRandomNoiseValue() is a specific table-driven
    generator seeded from sequence_counter (Pseudocode 24), and that table lives
    in the speech front end, which is not implemented here.  The spec states its
    output is "normal distributed with unit variance and zero mean", so a normal
    RNG seeded from the same sequence_counter is used instead.  The statistics
    are as specified and the levels are exact; the particular noise realisation
    differs from a reference decoder.  This is noise by construction, so it is
    perceptually equivalent -- but it is NOT bit-exact and must not be described
    as conformant.
    """
    snf = ch.get("snf")
    if not snf:
        return
    snf0 = snf[0]
    off = B.SFB_OFFSET_1536
    n = min(max_sfb, len(snf0))

    # Pseudocode 22: the reference level is the first band with any energy
    prev = None
    for sfb in range(n):
        lo, hi = off[sfb], off[sfb + 1]
        ms = float((spec[lo:hi] ** 2).mean()) if hi > lo else 0.0
        if ms > 0:
            prev = 1.44269504 * np.log(ms)
            break
    if prev is None:
        return

    for sfb in range(n):
        lo, hi = off[sfb], off[sfb + 1]
        if hi <= lo:
            continue
        ms = float((spec[lo:hi] ** 2).mean())
        if ms > 0:
            prev = 1.44269504 * np.log(ms)
            continue
        d = snf0[sfb]
        if d is None or d == -17:          # -17 escapes: no fill, level held
            continue
        prev = prev + d
        spec[lo:hi] = rng.standard_normal(hi - lo) * (2.0 ** (0.5 * prev))


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
    # sap_mode 0 = no processing; sap_mode 3 = full SAP (alpha prediction,
    # not implemented -- passed through rather than faked, and counted)
    return m, s


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
