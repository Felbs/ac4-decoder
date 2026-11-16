# SPDX-License-Identifier: Apache-2.0
"""M34 -- decoding the A-SPX signal envelopes, not just counting their bits.

M32 walks the A-SPX payload and proves it consumes exactly `audio_size`.  But
proving the bit COUNT is right is not the same as having the VALUES: the
envelopes are what actually shape the high band, and until they are
reconstructed nothing above 12 kHz can be synthesised.

PSEUDOCODE 80 -- RECONSTRUCTION
--------------------------------
    delta = 2 if (ch == 1 and aspx_balance == 1) else 1

    FREQ (aspx_sig_delta_dir == 0):
        qscf[sbg][atsg] = delta * sum(aspx_data_sig[atsg][0..sbg])
    TIME:
        qscf[sbg][atsg] = qscf_prev[sbg] + delta * aspx_data_sig[atsg][sbg]

where the TIME branch remaps sbg through sbg_idx_low2high / high2low when the
frequency resolution CHANGES between consecutive envelopes -- so a resolution
switch does not corrupt the running value.

**This is stateful across frames.**  `qscf_sig_sbg_prev` is the last envelope of
the PREVIOUS A-SPX interval, so a TIME-coded first envelope depends on the
frame before it.  Second structure in this decoder that cannot be evaluated on
one frame in isolation (the framing borders were the first).

THE GATE
---------
Envelopes are LEVELS, quantised in 1.5 or 3.0 dB steps, so the same test that
proved the MDCT scale factors applies: loudness is continuous in time, so the
mean envelope of frame N must track frame N+1, with a shuffled control at zero.
Random bits pushed through a complete prefix code produce plausible-looking
values; they do not produce time structure.

Second gate: envelope level should FALL with frequency across the A-SPX range.
That is a property of real programme material, not of the arithmetic.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from . import toc as M                                          # noqa: E402
from . import hcb as H                                               # noqa: E402
from . import spectral as S                                          # noqa: E402
from . import channels as C                                          # noqa: E402
from . import aspx_bands as A                                        # noqa: E402
from . import aspx_parse as P                                        # noqa: E402
from .toc_v1 import Bits                                      # noqa: E402


def sbg_maps(t):
    """Pseudocode 80's index maps between the high- and low-res tables."""
    hi, lo = t["hi"], t["lo"]
    high2low = [0] * t["n_hi"]
    low2high = [0] * (t["n_lo"] + 1)
    sbg_low = 0
    for sbg in range(t["n_hi"]):
        if sbg_low + 1 < len(lo) and lo[sbg_low + 1] == hi[sbg]:
            sbg_low += 1
            low2high[sbg_low] = sbg
        high2low[sbg] = sbg_low
    return high2low, low2high


def reconstruct(dat, t, maps, prev):
    """Pseudocode 80.  -> qscf[atsg][sbg], and the new `prev` state."""
    high2low, low2high = maps
    fr = dat["fr"]
    delta = 2 if (dat["ch"] == 1 and dat["balance"] == 1) else 1
    out = []
    prev_q, prev_res = prev
    for atsg, vals in enumerate(dat["sig"]):
        res = fr["freq_res"][atsg]
        n_sbg = t["n_hi"] if res else t["n_lo"]
        q = [0] * n_sbg
        if dat["dir_sig"][atsg] == 0:                     # FREQ
            acc = 0
            for sbg in range(min(n_sbg, len(vals))):
                acc += delta * vals[sbg]
                q[sbg] = acc
        else:                                             # TIME
            for sbg in range(min(n_sbg, len(vals))):
                if prev_q is None:
                    base = 0
                elif res == prev_res:
                    base = prev_q[sbg] if sbg < len(prev_q) else 0
                elif res == 0 and prev_res == 1:
                    i = low2high[sbg] if sbg < len(low2high) else 0
                    base = prev_q[i] if i < len(prev_q) else 0
                else:
                    i = high2low[sbg] if sbg < len(high2low) else 0
                    base = prev_q[i] if i < len(prev_q) else 0
                q[sbg] = base + delta * vals[sbg]
        out.append(q)
        prev_q, prev_res = q, res
    return out, (prev_q, prev_res)
