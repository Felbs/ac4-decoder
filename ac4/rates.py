# SPDX-License-Identifier: Apache-2.0
"""Frame rates, frame lengths and the output resampler ratio
(TS 103 190-1 4.3.3.2.5, 4.3.3.2.6, Tables 82-84; 4.3.3.5, Table 87;
TS 103 190-2 6.2.1.4, 5.1.3).

    fs_index          0 = 44,1 kHz, 1 = 48 kHz base sampling frequency
    frame_rate_index  0..13 (14, 15 reserved)

Table 83 (48 kHz): every frame_rate_index has an internal frame length,
frame_len_base, and a "decoder resampling ratio" that takes the internal
rate to the external 48 kHz: 1001/1000 x 25/24 for the 1001-family rates
(23.976, 29.97, 47.95, 59.94, 119.88 fps), 25/24 for 24, 30, 48, 60, 120
fps, 15/16 for 25, 50, 100 fps, and 1 for index 13 (23.44 fps, 2048
samples: the only rate at which the internal and the external rate are
the same). Table 84 (44,1 kHz): only index 13 (2048 samples, ratio 1).

What a frame of this decoder is, and what it is not:
  * frame_rate_factor (Table 87, b_multiplier) > 1: one ac4_substream_info
    refers to 2 or 4 substreams decoded one after the other. Not decoded:
    such frames are concealed (toc.unsupported_rate).
  * frame_rate_fraction (TS 103 190-2 6.2.1.4, the efficient high frame rate
    mode, 5.1.3) > 1: a codec frame is spread over 2 or 4 transmission
    frames and must be reassembled. Not decoded: concealed
    (toc.unsupported_rate).
  * b_sf_multiplier (96 / 192 kHz substreams): concealed
    (toc.unsupported_rate).
"""
from __future__ import annotations

# Table 83: frame_rate_index -> (frame_len_base at 48 kHz, (up, down))
_T83 = {
    0: (1920, (1001, 960)),     # 23.976 fps: 1001/1000 x 25/24
    1: (1920, (25, 24)),        # 24
    2: (2048, (15, 16)),        # 25
    3: (1536, (1001, 960)),     # 29.97
    4: (1536, (25, 24)),        # 30
    5: (960, (1001, 960)),      # 47.95
    6: (960, (25, 24)),         # 48
    7: (1024, (15, 16)),        # 50
    8: (768, (1001, 960)),      # 59.94
    9: (768, (25, 24)),         # 60
    10: (512, (15, 16)),        # 100
    11: (384, (1001, 960)),     # 119.88
    12: (384, (25, 24)),        # 120
    13: (2048, (1, 1)),         # 23.44 (48 kHz) / 11025/512 (44,1 kHz)
}

# frames per second (for reporting), Table 83 / 84
FPS = {0: 24000 / 1001, 1: 24.0, 2: 25.0, 3: 30000 / 1001, 4: 30.0,
       5: 48000 / 1001, 6: 48.0, 7: 50.0, 8: 60000 / 1001, 9: 60.0,
       10: 100.0, 11: 120000 / 1001, 12: 120.0, 13: 48000 / 2048}

DEFAULT_FRAME = 1536            # frames whose rate is not known yet


def frame_len_base(fs_index, frame_rate_index):
    """-> the internal frame length (samples), or None when Tables 83/84
    define none (reserved index, or 44,1 kHz with an index other than 13)."""
    if fs_index == 1:
        e = _T83.get(frame_rate_index)
        return e[0] if e else None
    if fs_index == 0 and frame_rate_index == 13:
        return 2048
    return None


def ratio(fs_index, frame_rate_index):
    """-> (up, down), the decoder resampling ratio of Table 83/84, or
    None."""
    if frame_len_base(fs_index, frame_rate_index) is None:
        return None
    if fs_index == 0:
        return (1, 1)
    return _T83[frame_rate_index][1]


def out_rate(fs_index):
    """The external sampling frequency this decoder writes: 48 kHz, or
    44,1 kHz for fs_index 0 (Table 84; 4.3.3.2.6: for frame_rate_index 13,
    44,1 kHz is a permissible external sampling frequency)."""
    return 44100 if fs_index == 0 else 48000


def of_toc(st):
    """-> (fs_index, frame_rate_index) of a parsed TOC."""
    return (st.get("fs_index"), st.get("frame_rate_index"))


def unsupported(st, sub=None):
    """-> a reason string when this decoder cannot decode the frame's
    rate signalling (see the module docstring), else None. sub: the audio
    substream's info (its sf_multiplier)."""
    fs, fri = of_toc(st)
    if frame_len_base(fs, fri) is None:
        return f"fs_index {fs} frame_rate_index {fri}"
    for p in st.get("presentations") or []:
        if p.get("frame_rate_factor", 1) != 1:
            return f"frame_rate_factor {p['frame_rate_factor']}"
        if p.get("frame_rate_fraction", 1) != 1:
            return f"frame_rate_fraction {p['frame_rate_fraction']}"
    if sub is not None and sub.get("sf_multiplier") is not None:
        return "sf_multiplier (96/192 kHz substream)"
    return None
