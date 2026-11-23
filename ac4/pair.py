# SPDX-License-Identifier: Apache-2.0
"""M43 -- channel_pair_element: the STEREO AC-4 element (Table 22).

Built 8/08 for a stereo simulcast track: its audio substream is ~365 bytes
against the 5.1 track's ~796. `m28.decode_element` parses only the 5_X
(5.1) element and raised on every frame of that track: 118,361 frames,
118,361 bad, decoded at
10x real time into perfect silence. A decoder that fails fast on the wrong
element type LOOKS like a healthy fast decoder in the throughput column --
the bad-frame counter was the only honest number on that line.

Syntax (TS 103 190-1 Table 22, channel_pair_element):
    stereo_codec_mode : 2 bits   (0 SIMPLE, 1 ASPX, 2/3 ASPX_ACPL)
    if (b_iframe && mode uses A-SPX)  aspx_config()
    SIMPLE: stereo_data()
    ASPX:   companding_control(2); stereo_data(); aspx_data_2ch()

`stereo_data` (Table 23) is the 5_X pair block (4.2.6.7 two_channel_data)
plus one spec_frontend bit before each channel's sf_info when MDCT stereo
processing is off (channels.stereo_data).
The trailing aspx_data_2ch is the caller's to parse (m42 does it at
r["bits"], exactly as for 5_X).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

from .toc import Bits                                     # noqa: E402
from .channels import (aspx_config, audio_size,                # noqa: E402
                       companding_control, stereo_data)


def decode_pair(sub, T, b_iframe=False):
    """One channel_pair_element substream -> dict like decode_element's,
    with L/R only. Raises on ACPL modes (not seen on this air; counted by
    the caller as bad frames rather than mis-decoded)."""
    b = Bits(sub)
    asz, a0 = audio_size(b)                          # Table 16
    mode = b.u(2)                                    # stereo_codec_mode
    if mode >= 2:
        raise ValueError(f"stereo_codec_mode {mode} (ACPL) not implemented")
    aspx = None
    if b_iframe and mode == 1:
        aspx = aspx_config(b)
    comp = None
    if mode == 1:
        comp = companding_control(b, 2)
    l, r, st = stereo_data(b, T)                     # Table 23
    return dict(L=l, R=r, st_lr=st, aspx=aspx, codec_mode=mode, compand=comp,
                bits=b.p, nbits=len(sub) * 8,
                audio_size=asz, audio_start=a0, mode=("pair", mode))
