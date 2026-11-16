# SPDX-License-Identifier: Apache-2.0
"""M35 -- the A-SPX HF generator: the patch copy, and nothing after it.

This is deliberately ONE stage, not the whole high-band chain.  A-SPX
reconstructs 12..21 kHz in four steps:

    1. HF GENERATION   copy low QMF subbands up into the A-SPX range   <- here
    2. envelope adjustment   scale them to the transmitted envelopes
    3. noise addition        fill where the envelope says noise, not tone
    4. additional harmonics  insert sinusoids where flagged

Only step 1 is implemented.  Steps 2-4 are what make the result *correct*; the
patched band on its own has the right content at the WRONG LEVEL, typically far
too loud, because the whole point of the envelope stage is that a copy of
7.5-12 kHz is not what 12-21 kHz should sound like.

So this file **does not render audio**.  It gates the patch in the QMF domain,
where the claim is exact and checkable, and stops there.  Rendering a
half-finished high band would produce a file that plays and is wrong, which is
the failure mode this project has guarded against throughout.

WHAT THE PATCH IS (M31, Pseudocode 71)
----------------------------------------
    patch 0: 12 subbands from 20..32 (7500..12000 Hz) -> 12000..16500 Hz
    patch 1: 12 subbands from 20..32 (7500..12000 Hz) -> 16500..21000 Hz

Both source the same block -- the top 4.5 kHz of the MDCT-coded band, ending
exactly at the crossover.

THE GATES
----------
  1. after patching, each destination range holds EXACTLY the energy of its
     source range -- a copy is a copy, so this is equality, not similarity;
  2. the A-SPX range is empty BEFORE the patch (M33's integration gate, which
     is what makes step 1's "before" well defined);
  3. subbands above the A-SPX stop (56 = 21000 Hz) stay empty -- A-SPX does not
     reach there, so a correct patch must not put energy there either.
"""
from __future__ import annotations

import argparse
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from . import channels as C                                          # noqa: E402
from . import aspx_bands as A                                        # noqa: E402
from . import qmf as Q                                               # noqa: E402


def hf_generate(Qm, t):
    """Step 1 only: copy the source blocks into the A-SPX range.

    Q_high[sbx + k] = Q_low[patch_start_sb[p] + k] for each patch p.
    """
    out = Qm.copy()
    for p, (n, st) in enumerate(zip(t["patch_num_sb"], t["patch_start_sb"])):
        dst = t["patches"][p]
        out[dst:dst + n, :] = Qm[st:st + n, :]
    return out
