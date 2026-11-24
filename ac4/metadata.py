# SPDX-License-Identifier: Apache-2.0
"""AC-4 metadata: loudness, DRC, dialogue enhancement, EMDF.

Bitstream version 2 (TS 103 190-2) splits it in two places:

  * the presentation substream (6.2.2.3, ac4_presentation_substream):
    dialnorm_bits, further_loudness_info, and drc_frame (the DRC decoder
    modes and their gains or compression curves, TS 103 190-1 4.2.14.5 -
    4.2.14.10);
  * the metadata() at the end of each audio substream (6.2.7): substream
    loudness and downmix information (basic_metadata), dialogue
    classification (extended_metadata), dialogue enhancement
    (dialog_enhancement, de_config, de_data) and EMDF payloads.

Both keep state across frames: the DRC configuration and the dialogue
enhancement configuration come with I-frames (b_pres_ndot / b_audio_ndot),
and DE parameters are coded against the previous frame's.

Where the presentation substream's DRC block or a substream's tools block
does not parse, its own size field (drc_metadata_size, tools_metadata_size)
puts the reader back on the next field, and the failure is reported in the
result ("error"), not raised.
"""
from __future__ import annotations

import math

from . import hcb as H
from . import spectral as S
from .toc_v1 import Bits

# cb_off of the DE and DRC codebooks (TS 103 190-1 Tables A.58 - A.62)
CB_OFF = {"DE_HCB_ABS_0": 0, "DE_HCB_DIFF_0": 31, "DE_HCB_ABS_1": 30,
          "DE_HCB_DIFF_1": 60, "DRC_HCB": 127}

_BOOKS = None


def books():
    global _BOOKS
    if _BOOKS is None:
        arr = H.parse_c(H.DEFAULT_C)
        _BOOKS = {n: S.Huff(arr[n + "_LEN"], arr[n + "_CW"]) for n in CB_OFF}
    return _BOOKS


def huff_diff(b, name):
    """huff_decode_diff (4.3.10.8.3): codeword index minus cb_off."""
    return books()[name].decode(b) - CB_OFF[name]


# Table 161 / 168 / 169
DRC_MODE_NAME = {0: "home theatre", 1: "flat panel TV",
                 2: "portable speakers", 3: "portable headphones"}
DRC_MODE_RANGE = {0: (-31, -27), 1: (-26, -17), 2: (-16, 0), 3: (-16, 0)}
DRC_SUBFRAMES = {384: 1, 512: 2, 768: 3, 960: 3, 1024: 4, 1536: 6,
                 1920: 6, 2048: 8}
DRC_EAC3_PROFILE = ("none", "film standard", "film light", "music standard",
                    "music light", "speech", "reserved", "reserved")


def nr_drc_channels(ch_mode):
    """Table 168: 1 for mono and stereo, 3 (front+LFE, centre, the rest)
    for the multichannel modes."""
    return 1 if ch_mode in (0, 1) else 3


def lkfs(v):
    """An 11-bit loudness field (4.3.12.3.8 ff.) -> LKFS."""
    return (v - 1024) / 10.0


# ---------------------------------------------------------------- loudness
def further_loudness_info(b, sus_ver, pres_ldn):
    """6.2.7.3 -> dict of the loudness values present (LKFS / dBTP)."""
    out = {}
    if pres_ldn or sus_ver == 0:
        v = b.u(2)
        if v == 3:
            v += b.u(4)
        out["loudness_version"] = v
        out["loud_prac_type"] = t = b.u(4)
        if t != 0:
            if b.u(1):
                b.u(3)
            b.u(1)
    else:
        b.u(1)                                   # b_loudcorr_dialgate
    if b.u(1):
        out["loudrelgat"] = lkfs(b.u(11))
    if b.u(1):
        out["loudspchgat"] = lkfs(b.u(11))
        b.u(3)
    if b.u(1):
        out["loudstrm3s"] = lkfs(b.u(11))
    if b.u(1):
        out["max_loudstrm3s"] = lkfs(b.u(11))
    if b.u(1):
        out["truepk"] = lkfs(b.u(11))
    if b.u(1):
        out["max_truepk"] = lkfs(b.u(11))
    if pres_ldn or sus_ver == 0:
        if b.u(1):                               # b_prgmbndy
            n = 0
            while b.u(1) == 0:
                n += 1
                if n > 32:
                    raise ValueError("prgmbndy")
            b.u(1)
            if b.u(1):
                b.u(11)
    if b.u(1):
        out["lra"] = b.u(10) / 10.0
        b.u(3)
    if b.u(1):
        out["loudmntry"] = lkfs(b.u(11))
    if b.u(1):
        out["max_loudmntry"] = lkfs(b.u(11))
    if sus_ver >= 1:
        if b.u(1):
            out["rtll_comp"] = b.u(8)
        if b.u(1):
            n = b.u(5)
            if n == 31:
                n += b.vb(4)
            b.u(n) if n <= 64 else [b.u(1) for _ in range(n)]
    else:
        if b.u(1):
            n = b.u(5)
            if n == 31:
                n += b.vb(4)
            if b.u(1):
                out["rtll_comp"] = b.u(8)
                n -= 9
            else:
                n -= 1
            for _ in range(max(0, n)):
                b.u(1)
    return out


# ---------------------------------------------------------------- DRC
class DrcState:
    """The DRC configuration (drc_config), kept from the last I-frame."""

    def __init__(self):
        self.config = None


def drc_compression_curve(b):
    c = dict(nullband_low=b.u(4), nullband_high=b.u(4),
             gain_max_boost=b.u(4))
    if c["gain_max_boost"] > 0:
        c["lev_max_boost"] = b.u(5)
        c["nr_boost_sections"] = b.u(1)
        if c["nr_boost_sections"]:
            c["gain_section_boost"] = b.u(4)
            c["lev_section_boost"] = b.u(5)
    c["gain_max_cut"] = b.u(5)
    if c["gain_max_cut"] > 0:
        c["lev_max_cut"] = b.u(6)
        c["nr_cut_sections"] = b.u(1)
        if c["nr_cut_sections"]:
            c["gain_section_cut"] = b.u(5)
            c["lev_section_cut"] = b.u(5)
    c["tc_default"] = b.u(1)
    if not c["tc_default"]:
        c["tc"] = (b.u(8), b.u(8), b.u(8), b.u(8))
        c["adaptive"] = b.u(1)
        if c["adaptive"]:
            c["thresholds"] = (b.u(5), b.u(5))
    return c


def drc_config(b):
    """4.2.14.6/7 -> dict: modes (in order: id, curve flag, gains_config,
    curve or default, output range), eac3_profile."""
    n = b.u(3) + 1
    modes = []
    by_id = {}
    for _ in range(n):
        mid = b.u(3)
        m = dict(id=mid, out_range=DRC_MODE_RANGE.get(mid))
        if mid > 3:
            m["out_range"] = (-b.u(5), -b.u(5))
        if b.u(1):                               # drc_repeat_profile_flag
            rid = b.u(3)
            src = by_id.get(rid)
            m["repeat"] = rid
            m["curve_flag"] = src["curve_flag"] if src else 1
            m["gains_config"] = src.get("gains_config") if src else None
            m["curve"] = src.get("curve") if src else None
            m["default_profile"] = src.get("default_profile") if src else 0
        elif not b.u(1):                         # !drc_default_profile_flag
            m["curve_flag"] = b.u(1)
            m["default_profile"] = 0
            if m["curve_flag"]:
                m["curve"] = drc_compression_curve(b)
            else:
                m["gains_config"] = b.u(2)
        else:
            m["curve_flag"] = 1
            m["default_profile"] = 1
        modes.append(m)
        by_id[mid] = m
    return dict(modes=modes, eac3_profile=b.u(3))


def drc_gains(b, gains_config, n_ch, n_sf):
    """4.2.14.10 -> gains[ch][sf][band] in dB (drc_gain_val - 64, then
    Huffman-coded differences)."""
    n_band = {0: 1, 1: 1, 2: 2, 3: 4}[gains_config]
    g0 = b.u(7) - 64
    if gains_config == 0:
        return [[[g0]]]
    g = [[[0] * n_band for _ in range(n_sf)] for _ in range(n_ch)]
    g[0][0][0] = g0
    ref = g0
    for ch in range(n_ch):
        for band in range(n_band):
            for sf in range(n_sf):
                if sf or band or ch:
                    g[ch][sf][band] = ref + huff_diff(b, "DRC_HCB")
                ref = g[ch][sf][band]
            ref = g[ch][0][band]
        ref = g[ch][0][0]
    return g


def drc_frame(b, iframe, state, ch_mode, frame_len):
    """4.2.14.5 -> dict or None (b_drc_present 0). Gains are given per
    mode id for modes with drc_compression_curve_flag 0."""
    if not b.u(1):
        return None
    if iframe:
        state.config = drc_config(b)
    if state.config is None:
        raise ValueError("drc_data before the first drc_config")
    out = dict(config=state.config, gains={}, reset=None)
    curve = False
    for m in state.config["modes"]:
        if m["curve_flag"] == 0:
            size = b.u(6)
            if b.u(1):
                size += b.vb(2) << 6
            start = b.p
            ver = b.u(2)
            if ver <= 1:
                out["gains"][m["id"]] = drc_gains(
                    b, m["gains_config"], nr_drc_channels(ch_mode),
                    DRC_SUBFRAMES.get(frame_len, 1))
            if ver >= 1:
                b.p = start + size                # drc2_bits skipped
        else:
            curve = True
    if curve:
        out["reset"] = b.u(1)
        b.u(2)
    return out


# ---------------------------------------------------------------- presentation substream
def presentation_substream(sub, b_alternative, b_pres_ndot, n_substreams,
                           drc, ch_mode, frame_len, n_groups=1):
    """ac4_presentation_substream (6.2.2.3) up to the associated audio
    mixing metadata -> dict: dialnorm (dBFS), loudness
    (further_loudness_info), drc (drc_frame or None), name (alternative
    presentations); with n_groups > 1 substream groups, "sg_gain" (a list
    of dB, -inf for 63, Table 70; "keep" when b_keep; absent when not
    sent) and "assoc" (b_associated: scale_main, scale_main_centre,
    scale_main_front in dB, pan_associated in degrees, each None when not
    sent; TS 103 190-1 4.3.12.4)."""
    b = Bits(sub)
    out = {}
    if b_alternative:
        if b.u(1):                               # b_name_present
            n = b.u(5) if b.u(1) else 32
            out["name"] = bytes(b.u(8) for _ in range(n)).decode(
                "utf-8", "replace")
        nt = b.u(2) + 1
        if nt == 4:
            nt += b.vb(2)
        for _ in range(nt):
            b.u(3)
            b.u(4)
            if b.u(1):
                b.u(4)
            if b.u(1):
                b.u(6)
            if b.u(1):
                b.u(5)
            for _ in range(n_substreams):
                if b.u(1) and b.u(1):
                    b.vb(2)
    if b.u(1):                                   # b_additional_data
        n = b.u(4) + 1
        if n == 16:
            n += b.vb(2)
        b.align()
        b.p += 8 * n                             # add_data (A-DE etc.)
    out["dialnorm_bits"] = b.u(7)
    out["dialnorm"] = -out["dialnorm_bits"] / 4.0
    if b.u(1):
        out["loudness"] = further_loudness_info(b, 1, 1)
    size = b.u(5)
    if b.u(1):
        size += b.vb(3) << 5
    start = b.p
    try:
        out["drc"] = drc_frame(b, b_pres_ndot, drc, ch_mode, frame_len)
        if b.p != start + size:
            out["drc_error"] = f"drc_frame read {b.p - start} of {size} bits"
    except (ValueError, EOFError, KeyError) as e:
        out["drc"] = None
        out["drc_error"] = str(e)[:60]
    b.p = start + size
    out["drc_metadata_size"] = size
    try:
        if n_groups > 1 and b.u(1):          # b_substream_group_gains_present
            if b.u(1):                       # b_keep
                out["sg_gain"] = "keep"
            else:
                out["sg_gain"] = [sg_gain_db(b.u(6)) for _ in range(n_groups)]
        if b.u(1):                           # b_associated
            a = {}
            a["scale_main"] = scale_db(b.u(8)) if b.u(1) else None
            a["scale_main_centre"] = scale_db(b.u(8)) if b.u(1) else None
            a["scale_main_front"] = scale_db(b.u(8)) if b.u(1) else None
            a["pan_associated"] = pan_deg(b.u(8)) if b.u(1) else None
            out["assoc"] = a
    except (ValueError, EOFError, KeyError) as e:
        out["mix_error"] = str(e)[:60]
    return out


def sg_gain_db(v):
    """Table 70: -0.25 dB steps, 63 = -inf."""
    return -math.inf if v == 63 else -0.25 * v


def scale_db(v):
    """scale_main / _centre / _front (TS 103 190-1 4.3.12.4.4 - .8): 0 to
    -76.2 dB in 0.3 dB steps, 0xff = mute."""
    return -math.inf if v == 0xFF else -0.3 * v


def pan_deg(v):
    """pan_associated / pan_dialog (4.3.12.4.9): 1.5 degree steps,
    clockwise from the front; 0xf0 .. 0xff are not used (read as 0)."""
    return 1.5 * v if v < 0xF0 else 0.0


# ---------------------------------------------------------------- substream metadata
class SubMetaState:
    """Dialogue enhancement state of one audio substream."""

    def __init__(self):
        self.de_config = None
        self.de_par_prev = None
        self.de_mix = None


DE_NR_CHANNELS = (0, 1, 1, 2, 1, 2, 2, 3)          # Table 171


def de_config(b):
    return dict(method=b.u(2), max_gain=b.u(2), channel_config=b.u(3))


def de_data(b, cfg, iframe, st, simulcast=False):
    """6.2.7.6 -> dict (par[ch][band] indices, mix indices, ms, contrib)."""
    method = cfg["method"]
    n_ch = DE_NR_CHANNELS[cfg["channel_config"]]
    out = dict(keep_data=None)
    if n_ch == 0:
        return out
    if method in (1, 3) and n_ch > 1 and not simulcast:
        keep_pos = 0 if iframe else b.u(1)
        if not keep_pos:
            st.de_mix = [b.u(5)] + ([b.u(5)] if n_ch == 3 else [])
    out["mix"] = st.de_mix
    keep = 0 if iframe else b.u(1)
    out["keep_data"] = keep
    prev = st.de_par_prev or [[0] * 8 for _ in range(3)]
    if keep:
        out["par"] = prev
        return out
    ms = b.u(1) if (method in (0, 2) and n_ch == 2) else 0
    t = method % 2
    par = [[0] * 8 for _ in range(3)]
    ref = 0
    for ch in range(n_ch - ms):
        if iframe and ch == 0:
            par[0][0] = huff_diff(b, f"DE_HCB_ABS_{t}")
            ref = par[0][0]
            for band in range(1, 8):
                par[0][band] = ref + huff_diff(b, f"DE_HCB_DIFF_{t}")
                ref = par[0][band]
        else:
            for band in range(8):
                if iframe:
                    par[ch][band] = ref + huff_diff(b, f"DE_HCB_DIFF_{t}")
                    ref = par[ch][band]
                else:
                    par[ch][band] = prev[ch][band] + huff_diff(
                        b, f"DE_HCB_DIFF_{t}")
        ref = par[ch][0]
    st.de_par_prev = par
    out.update(par=par, ms=ms)
    if method >= 2:
        out["signal_contribution"] = b.u(5)
    return out


def basic_metadata(b, ch_mode, sus_ver):
    out = {}
    if sus_ver == 0:
        out["dialnorm"] = -b.u(7) / 4.0
    if b.u(1):                                   # b_more_basic_metadata
        if sus_ver == 0:
            if b.u(1):
                out["loudness"] = further_loudness_info(b, sus_ver, 0)
        elif b.u(1):                             # b_substream_loudness_info
            out["substream_loudness_bits"] = b.u(8)
            if b.u(1):
                out["loudness"] = further_loudness_info(b, sus_ver, 0)
        if ch_mode == 1:
            if b.u(1):
                out["pre_dmixtyp_2ch"] = b.u(3)
                b.u(2)
        if ch_mode > 1:
            if sus_ver == 0 and b.u(1):          # b_stereo_dmx_coeff
                out["loro"] = (b.u(3), b.u(3))
                if b.u(1):
                    b.u(5)
                if b.u(1):
                    out["ltrt"] = (b.u(3), b.u(3))
                if b.u(1):
                    b.u(5)
                if ch_mode in (4, 6, 8, 10, 12, 14, 15) and b.u(1):
                    b.u(5)
                out["preferred_dmx_method"] = b.u(2)
            if ch_mode in (3, 4):                # 5_X
                if b.u(1):
                    b.u(3)
                if b.u(1):
                    b.u(4)
            if ch_mode >= 5:                     # 7_X (and above)
                if b.u(1):
                    if ch_mode in (5, 6) or ch_mode >= 11:
                        b.u(2)                   # 3/4/0
                    elif ch_mode in (9, 10):
                        b.u(1)                   # 3/2/2
            b.u(2)                               # phase90_info_mc
            b.u(1)
            b.u(1)
        if b.u(1):                               # b_dc_blocking
            b.u(1)
    return out


def extended_metadata(b, ch_mode, sus_ver, b_associated=0):
    out = {}
    b_dialog = 0
    if sus_ver >= 1:
        b_dialog = b.u(1)
    elif b_associated:
        for _ in range(3):
            if b.u(1):
                b.u(8)
        if ch_mode == 0:
            b.u(8)
    out["dialog"] = b_dialog
    if b_dialog:
        if b.u(1):
            out["dialog_max_gain"] = b.u(2)
        if b.u(1):
            b.u(8) if ch_mode == 0 else (b.u(8), b.u(8), b.u(2))
    if b.u(1):                                   # b_channels_classifier
        cls = {}
        if ch_mode == 0 or 2 <= ch_mode <= 15:
            cls["c"] = b.u(1) and (b.u(1) + 1)
        if 1 <= ch_mode <= 15:
            cls["l"] = b.u(1) and (b.u(1) + 1)
            cls["r"] = b.u(1) and (b.u(1) + 1)
        if 3 <= ch_mode <= 15:
            b.u(2)
        if ch_mode in (5, 6) or 11 <= ch_mode <= 15:
            b.u(2)
        if ch_mode in (7, 8, 15):
            b.u(2)
        if ch_mode in (9, 10):
            b.u(2)
        if ch_mode in (4, 6, 8, 10, 12, 14, 15):
            b.u(1)
        out["dialog_channels"] = cls
    if b.u(1):
        out["event_probability"] = b.u(4)
    return out


def substream_metadata(sub, audio_start, audio_size, ch_mode, iframe, st,
                       sus_ver=1, no_emdf=False):
    """metadata() of an audio substream (6.2.7.1), read at audio_start +
    8 * audio_size -> dict: basic, extended, de (config + data, or None),
    emdf (payload ids), and "error" when the tools block did not parse."""
    b = Bits(sub)
    b.p = audio_start + 8 * audio_size
    out = dict(basic=basic_metadata(b, ch_mode, sus_ver),
               extended=extended_metadata(b, ch_mode, sus_ver))
    size = b.u(7)
    if b.u(1):
        size += b.vb(3) << 7
    start = b.p
    out["de"] = None
    try:
        if sus_ver == 0:
            out["drc_v0"] = True                 # drc_frame here: not v2
            raise ValueError("sus_ver 0 tools metadata not parsed")
        if b.u(1):                               # b_de_data_present
            if iframe or b.u(1):
                st.de_config = de_config(b)
            if st.de_config is None:
                raise ValueError("de_data before the first de_config")
            de = dict(config=st.de_config,
                      data=de_data(b, st.de_config, iframe, st))
            if ch_mode in (13, 14) and b.u(1):
                de_data(b, st.de_config, iframe, SubMetaState(), True)
            out["de"] = de
        else:
            st.de_par_prev = None
        if b.p != start + size:
            out["error"] = f"tools read {b.p - start} of {size} bits"
    except (ValueError, EOFError, KeyError) as e:
        out["error"] = str(e)[:60]
    b.p = start + size
    if no_emdf:
        return out
    out["emdf"] = []
    if b.u(1):                                   # b_emdf_payloads_substream
        while True:
            pid = b.u(5)
            if pid == 31:
                pid += b.vb(5)
            if pid == 0:
                break
            # emdf_payload_config (TS 103 190-1 Table 79)
            smpoffst = b.u(1)
            if smpoffst:
                b.vb(11)
            if b.u(1):                           # b_duration
                b.vb(11)
            if b.u(1):                           # b_groupid
                b.vb(2)
            if b.u(1):                           # b_codecdata
                b.u(8)
            if not b.u(1):                       # !b_discard_unknown_payload
                aligned = 0
                if not smpoffst:
                    aligned = b.u(1)             # b_payload_frame_aligned
                    if aligned:
                        b.u(2)                   # create / remove duplicate
                if smpoffst or aligned:
                    b.u(5)                       # priority
                    b.u(2)                       # proc_allowed
            n = b.vb(8)
            b.p += 8 * n
            out["emdf"].append(pid)
    return out
