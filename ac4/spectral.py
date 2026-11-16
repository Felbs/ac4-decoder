# SPDX-License-Identifier: Apache-2.0
"""M24 -- decode the LFE's quantised spectral lines off the air.

Everything upstream is verified: the TOC parses 3480/3480, the substream
boundary is gated by `audio_size`, the tool list is A-SPX with no A-CPL, the
section data lands on the bit the entropy map predicted, all 60 Huffman
codebooks pass Kraft, and the MDCT reconstructs to 2e-14.  This is the piece
that turns bits into numbers.

THE LFE IS THE SMALLEST REAL TARGET
------------------------------------
It is mono, it comes FIRST in `5_X_channel_element`, and `sf_info_lfe()` is one
field.  Table B.4 gives `sfb_offset[0..3] = 0, 4, 8, 12` -- and all three
transform-length columns agree there, so the number is not a reading choice.
With `max_sfb = 3` the LFE is **12 spectral lines**, roughly 0..190 Hz at
N = 1536, which is what an LFE channel should be.

    12 lines / CB_DIM 2  =  6 Huffman codewords, plus sign bits

THE DEQUANTISER, from Annex A's own parameters
-----------------------------------------------
`ASF_HCB_11`: codebook_length 289 = 17^2, cb_mod 17, cb_off 0.  So a codeword
index splits into a pair:

    quant_spec_1 = idx // 17 - cb_off
    quant_spec_2 = idx  % 17 - cb_off

and because `cb_off == 0` the alphabet is unsigned, so each NON-ZERO value is
followed by a sign bit.  (That is the same rule AAC uses, and it is why
Annex A bothers to print cb_off at all.)

CODEBOOK 11 ESCAPES AT 16, AND THE HISTOGRAM SAID SO BEFORE THE SPEC DID
--------------------------------------------------------------------------
The first run piled 14 % of all lines at exactly +-16 -- the top of the
codebook's alphabet, which no real spectrum does.  That is an escape flag being
read as a value.  The spec confirms it: for `sect_cb == 11`, a magnitude of 16
is followed by `ext_decode()`, 5..21 bits, unary prefix then N+4 value bits,
returning `2**(N+4) + val` -- and those bits come AFTER the sign bits, so the
order in the loop is load-bearing.  With escapes handled the spike vanishes and
values expand to the hundreds, as they must.

WHAT ACTUALLY FALSIFIES THIS, AND WHAT DOES NOT
------------------------------------------------
My first gate demanded ">20 % zeros" and FAILED a correct decode.  A full-band
spectrum is zero-dominated; the LFE is 12 lines covering 0..188 Hz, precisely
the band where its energy lives, so those lines should be populated.  The
criterion was wrong, not the decode.

Two other tempting checks are worthless here, and it is worth knowing why:

  * "no decode failures" proves nothing.  These codebooks are COMPLETE prefix
    codes (Kraft == 1, gated in M23), so every bit sequence decodes to
    something.  Garbage in, symbols out.
  * the histogram shape proves nothing either.  Random bits pushed through a
    Huffman table reproduce that table's own implied distribution -- which is
    Laplacian, exactly what a real spectrum looks like.

What random bits cannot fake is TIME STRUCTURE.  Adjacent frames of real audio
are correlated; shuffled frames are not.  Measured: adjacent r = +0.273 over
2946 pairs, shuffled control r = +0.013, and the autocorrelation decays
monotonically with lag (0.27, 0.16, 0.12, 0.04, 0.03, 0.01).  Every one of the
12 lines correlates positively, mean +0.34.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from . import toc as M                                         # noqa: E402
from . import hcb as H                                              # noqa: E402
from .toc_v1 import Bits                                     # noqa: E402

from . import sfb as B                                               # noqa: E402

# Table B.4, the 1536@48 column, all 50 bands -- extracted and gated in M27.
# This file previously carried only the first four entries, which is all the
# LFE needs; those four are unchanged (0, 4, 8, 12), so nothing decoded here
# moves.  Keeping one copy means the full-band channels and the LFE cannot
# drift apart.
SFB_OFFSET = B.SFB_OFFSET_1536
CB_MOD = {1: 3, 2: 3, 3: 3, 4: 3, 5: 9, 6: 9, 7: 8, 8: 8, 9: 13, 10: 13,
          11: 17}
CB_OFF = {1: 1, 2: 1, 3: 0, 4: 0, 5: 4, 6: 4, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}
CB_DIM = {1: 4, 2: 4, 3: 4, 4: 4, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2}


class Huff:
    """One codebook, as a dict from (length, codeword) to index."""

    def __init__(self, lens, words):
        self.map = {}
        self.maxlen = 0
        for i, (L, w) in enumerate(zip(lens, words)):
            if L <= 0:
                continue
            self.map[(L, w & ((1 << L) - 1))] = i
            self.maxlen = max(self.maxlen, L)

    def decode(self, b):
        """Read one codeword.  Raises if no prefix matches inside maxlen."""
        code = 0
        for L in range(1, self.maxlen + 1):
            code = (code << 1) | b.u(1)
            hit = self.map.get((L, code))
            if hit is not None:
                return hit
        raise ValueError("no codeword matched")


def decode_lfe(sub, tables, verbose=False):
    """-> (quantised lines, bit position) for one frame's LFE."""
    b = Bits(sub)
    b.u(15)
    if b.u(1):
        b.vb(7)
    codec_mode = b.u(3)
    max_sfb = b.u(3)
    sect_cb = b.u(4)
    sect_len = 1 + b.u(5)
    if sect_cb not in CB_MOD:
        raise ValueError(f"section codebook {sect_cb} is not a spectrum book")
    hb = tables[sect_cb]
    dim, mod, off = CB_DIM[sect_cb], CB_MOD[sect_cb], CB_OFF[sect_cb]
    start, end = SFB_OFFSET[0], SFB_OFFSET[min(max_sfb, 3)]
    lines = []
    k = start
    while k < end:
        idx = hb.decode(b)
        if dim == 4:
            vals = [idx // (mod ** 3) - off,
                    (idx // (mod ** 2)) % mod - off,
                    (idx // mod) % mod - off,
                    idx % mod - off]
        else:
            vals = [idx // mod - off, idx % mod - off]
        if off == 0:                       # unsigned book: sign bit per nonzero
            vals = [(-v if (v and b.u(1)) else v) for v in vals]
        # CODEBOOK 11 USES 16 AS AN ESCAPE.  Pseudocode 20:
        #     N = 0; while get_bits(1): N += 1
        #     return 2**(N+4) + get_bits(N+4)
        # and the spec is explicit that these bits come AFTER the sign bits,
        # which is why the order here matters and cannot be rearranged.
        # Without this the histogram piles up at +-16 -- the escape flag being
        # read as a value -- and every subsequent bit is misaligned.  That is
        # exactly what the first run showed.
        if sect_cb == 11:
            for j, v in enumerate(vals):
                if abs(v) == 16:
                    n_ext = 0
                    while b.u(1):
                        n_ext += 1
                    mag = (1 << (n_ext + 4)) + b.u(n_ext + 4)
                    vals[j] = -mag if v < 0 else mag
        lines.extend(vals)
        k += dim
    return np.array(lines[:end - start]), b.p, dict(
        codec_mode=codec_mode, max_sfb=max_sfb, sect_cb=sect_cb,
        sect_len=sect_len)
