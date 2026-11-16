# SPDX-License-Identifier: Apache-2.0
"""ac4 -- an AC-4 audio decoder (ETSI TS 103 190-1/-2), written from the
published standard, in Python and NumPy.

    from ac4 import decode, mp4
    pcm, dec = decode(mp4.read_frames("audio.m4s"))   # 48 kHz, decoder scale

See README.md for what it decodes and what it does not yet decode.
"""
__version__ = "0.1.0"

from .decoder import Decoder, decode, probe_element   # noqa: E402,F401
from .stream import Ac4Stream                          # noqa: E402,F401
from . import mp4                                      # noqa: E402,F401
