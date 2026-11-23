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
    with L/R only (aspx_blocks / acpl_blocks as there).

    stereo_codec_mode 2 / 3 (ASPX_ACPL_1 / _2, Table 22): the "m" track
    (A-SPX'd, aspx_data_1ch) on L and, in ASPX_ACPL_1, the residual "s"
    track on R (5.3.4.1); one acpl_data_1ch() (5.7.7.5)."""
    from .acpl_parse import acpl_config_1ch, PARTIAL, FULL
    from .channels import (chparam_info, Framing, frame_len, mono_data,
                           sf_data)
    b = Bits(sub)
    asz, a0 = audio_size(b)                          # Table 16
    mode = b.u(2)                                    # stereo_codec_mode
    aspx = acpl = None
    if b_iframe and mode >= 1:
        aspx = aspx_config(b)
        if mode >= 2:
            acpl = acpl_config_1ch(b, PARTIAL if mode == 2 else FULL)
    comp = None
    if mode == 1:
        comp = companding_control(b, 2)
    elif mode >= 2:
        comp = companding_control(b, 1)
    out = dict(aspx=aspx, codec_mode=mode, compand=comp, acpl=acpl,
               audio_size=asz, audio_start=a0, mode=("pair", mode),
               acpl_blocks=0)
    if mode <= 1:
        l, r, st = stereo_data(b, T)                 # Table 23
        out.update(L=l, R=r, st_lr=st,
                   aspx_blocks=((0, True),) if mode == 1 else ())
    elif mode == 2:                                  # ASPX_ACPL_1
        fl = frame_len(T)
        st = dict(sap_mode=None, ms_used=None, sap=None)
        if b.u(1):                                   # b_enable_mdct_stereo_proc
            fm = Framing(b, fl, b_dual_maxsfb=1)
            sap, ms_used, sap_d = chparam_info(b, fm, T["sf"])
            st.update(sap_mode=sap, ms_used=ms_used, sap=sap_d)
            fs = fm.for_side()
        else:
            if b.u(1) != 0:                          # spec_frontend_m
                raise ValueError("SSF (speech frontend) not implemented")
            fm = Framing(b, fl)
            if b.u(1) != 0:                          # spec_frontend_s
                raise ValueError("SSF (speech frontend) not implemented")
            fs = Framing(b, fl, b_side_limited=1).for_side()
        m = sf_data(b, fm, T)
        sd = sf_data(b, fs, T)
        st.update(max_sfb=fm.max_sfb_g(0), framing=fm)
        out.update(L=m, R=sd, st_lr=st, acpl1="pair",
                   aspx_blocks=((0, False),), acpl_blocks=1)
    else:                                            # ASPX_ACPL_2
        m = mono_data(b, T, b_lfe=False)             # spec_frontend, sf_info
        out.update(L=m, R=None, st_lr=dict(sap_mode=None), acpl1="pair",
                   aspx_blocks=((0, False),), acpl_blocks=1)
    out.update(bits=b.p, nbits=len(sub) * 8)
    return out
