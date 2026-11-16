# SPDX-License-Identifier: Apache-2.0
"""M25 -- the LFE's scale factors: the exponents that make the lines real.

M24 decoded the quantised spectral lines -- integers.  They are not amplitudes
yet: each scale factor band carries an exponent, and the reconstruction is

    coefficient = sign(q) * |q|^(4/3) * 2^((scale_factor - offset)/4)

so without the scale factors the spectrum has the right shape and the wrong
loudness, per band.  This reads them.

THE SYNTAX (4.2.8.5)
---------------------
    reference_scale_factor         8 bits
    first_scf_found = 0
    for each band with sfb_cb != 0 AND max_quant_idx > 0:
        if first_scf_found:  dpcm_sf = huff_decode(ASF_HCB_SCALEFAC)
        else:                first_scf_found = 1     # this band USES the reference

So the first populated band spends no bits at all -- it takes the 8-bit
reference -- and every later band is a Huffman-coded DELTA from the running
value.  `ASF_HCB_SCALEFAC` has 121 entries, which is 2*60+1, so the delta is
`index - 60`.  That centring is not a guess: a DPCM alphabet of odd size is
symmetric about its middle, and 121 admits exactly one centre.

WHAT MAKES A DECODE BELIEVABLE HERE
-------------------------------------
The same trap as M24 applies -- ASF_HCB_SCALEFAC is a complete prefix code, so
any bits decode to some delta, and the delta histogram will look like the
codebook whether or not the bits are ours.  Static shape proves nothing.

Two things random bits cannot fake:

  * **Scale factors are loudness, and loudness is continuous in time.**  The
    reference scale factor of frame N should track frame N+1, with a shuffled
    control at zero.  This is M24's test applied to a different quantity, and
    it is stronger here because the reference is read as a plain 8-bit field
    rather than through a codebook -- so a bit-offset error moves it wildly.
  * **Scale factors and quantised magnitudes are coupled.**  A band whose lines
    are large does not usually also carry a large exponent; the encoder trades
    them off to hit a bitrate.  A non-zero rank correlation between the two is
    evidence they came from the same encoder decision, not from noise.
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
from . import spectral as S                                         # noqa: E402
from .toc_v1 import Bits                                     # noqa: E402

SF_CENTRE = 60          # ASF_HCB_SCALEFAC is 121 = 2*60+1 entries


def decode_lfe_full(sub, tables, sf_table):
    """-> (lines, scale_factors, bit position after asf_scalefac_data)."""
    b = Bits(sub)
    b.u(15)
    if b.u(1):
        b.vb(7)
    b.u(3)                                   # 5_X_codec_mode
    max_sfb = b.u(3)
    sect_cb = b.u(4)
    b.u(5)                                   # sect_len_incr
    if sect_cb != 11:
        raise ValueError("not codebook 11")
    hb = tables[sect_cb]
    mod, off = S.CB_MOD[sect_cb], S.CB_OFF[sect_cb]
    end = S.SFB_OFFSET[min(max_sfb, 3)]

    lines = []
    while len(lines) < end:
        idx = hb.decode(b)
        vals = [idx // mod - off, idx % mod - off]
        vals = [(-v if (v and b.u(1)) else v) for v in vals]
        for j, v in enumerate(vals):
            if abs(v) == 16:
                n = 0
                while b.u(1):
                    n += 1
                mag = (1 << (n + 4)) + b.u(n + 4)
                vals[j] = -mag if v < 0 else mag
        lines.extend(vals)
    lines = np.array(lines[:end])

    # --- asf_scalefac_data -------------------------------------------------
    ref = b.u(8)
    sfs, cur, first = [], ref, False
    for sfb in range(min(max_sfb, 3)):
        band = lines[S.SFB_OFFSET[sfb]:S.SFB_OFFSET[sfb + 1]]
        if sect_cb != 0 and np.abs(band).max() > 0:
            if first:
                cur += sf_table.decode(b) - SF_CENTRE
            else:
                first = True
            sfs.append(cur)
        else:
            sfs.append(None)
    return lines, ref, sfs, b.p
