# SPDX-License-Identifier: Apache-2.0
"""M20 -- the AC-4 table of contents, from TS 103 190-2 (the RIGHT clause).

M19 implemented part 1's `ac4_presentation_info()` and the gate rejected it:
the parse decoded `presentation_config = 6`, which part 1 Table 85 calls
Reserved.  Part 2 clause 6.2.1 explains why -- presentation information is
`ac4_presentation_info()` only when `bitstream_version <= 1`, and
`ac4_presentation_v1_info()` otherwise.  Ours is 2.

Part 2's `ac4_toc()` is a different function, not a superset: after
`payload_base` it reads a program identifier, then v1 presentations, then a
loop of `ac4_substream_group_info()` that part 1 has no equivalent for.

TWO THINGS THIS CHANGES ABOUT WHAT WE EXPECT
---------------------------------------------
1. The 2-bit field after `wait_frames` is `br_code` here.  Part 1 calls the
   same bits `reserved`.  My memory said br_code, part 1 said I was wrong, and
   part 2 says memory was right -- for THIS document.  Both are correct about
   their own version, which is exactly the trap.
2. **Part 2's per-`frame_rate_factor` loop reads `b_audio_ndot`, not
   `b_iframe`.**  M18's entropy map found four bits identical to
   `b_iframe_global` at 23, 95, 108 and 147, and I predicted three of them were
   I-frame bits from that loop.  If this syntax is right, that prediction is
   WRONG and those bits are something else.  The gate is unchanged either way:
   it is the size sum that decides.

THE GATE
--------
`raw_ac4_frame()` is the TOC, byte-aligned, then one `ac4_substream_data()`
per substream, and the TOC declares each substream's size.  So

    sum(substream_size) + toc_bytes == frame length

on all 3480 frames, or the parse is wrong.  There is no partial credit: one
misread bit shifts everything after it.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

from .toc_v1 import Bits, PROT_PRIMARY, PROT_SECOND          # noqa: E402
from .toc_v1 import CH_MODE_NAME                             # noqa: E402


def channel_mode(b):
    """channel_mode -> ch_mode, TS 103 190-2 table 56 (presentation_version
    1; part 1's table 88 is the 7-bit subset).

    The prefix code: 0 / 10 / 1100..1110 / 1111000..1111101 (ch_mode
    5..10) / 11111100, 11111101 (7.0.4, 7.1.4: ch_mode 11, 12) /
    111111100..111111110 (9.0.4, 9.1.4, 22.2: 13..15) / 111111111 followed
    by variable_bits(2) (reserved, ch_mode 16 and up). A 7-bit 1111110 is
    NOT a code: it is the start of the two 8-bit ones."""
    if b.u(1) == 0:
        return 0
    if b.u(1) == 0:
        return 1
    two = b.u(2)                          # past '11'
    if two != 0b11:
        return {0b00: 2, 0b01: 3, 0b10: 4}[two]
    tail = b.u(3)                         # past '1111'
    if tail <= 0b101:
        return 5 + tail                   # 1111000..1111101 -> 5..10
    if tail == 0b110:
        return 11 + b.u(1)                # 11111100, 11111101 -> 11, 12
    last = b.u(2)                         # past '1111111'
    if last != 0b11:
        return 13 + last                  # 111111100..111111110 -> 13..15
    return 16 + b.vb(2)                   # 0b111111111 + variable_bits(2)


# ch_mode of the channel_mode codes 0b11111100, 0b11111101, 0b111111100
# and 0b111111101 (7.0.4, 7.1.4, 9.0.4, 9.1.4): ac4_substream_info_chan
# reads b_4_back_channels_present, b_centre_present and
# top_channels_present for them (TS 103 190-2 6.2.1.8)
CH_MODES_WITH_TOP = (11, 12, 13, 14)
# ch_mode of 0b1111010..0b1111101 (7.0/7.1 5/2/0 and 3/2/2): add_ch_base
CH_MODES_WITH_ADD_CH_BASE = (7, 8, 9, 10)


def content_type(b):
    """TS 103 190-1 4.2.3.7 (Table 10) -> (content_classifier, language).

    The language is the BCP 47 tag of n_language_tag_bytes bytes, or None.
    A serialized tag (b_serialized_language_tag) comes in 16-bit chunks
    spread over several frames; its chunk is returned as "~" + hex."""
    cc = b.u(3)                           # content_classifier, Table 91
    lang = None
    if b.u(1):                            # b_language_indicator
        if b.u(1):                        # b_serialized_language_tag
            b.u(1)                        # b_start_tag
            lang = "~%04x" % b.u(16)      # language_tag_chunk
        else:
            n = b.u(6)
            lang = bytes(b.u(8) for _ in range(n)).decode("ascii", "replace")
    return cc, lang


# Table 91 (TS 103 190-1 4.3.3.8.1)
CONTENT_CLASS = ("complete main", "music and effects", "visually impaired",
                 "hearing impaired", "dialogue", "commentary", "emergency",
                 "voice over")
# presentation_config, TS 103 190-2 6.2.1.3
PRES_CONFIG = {0: "M&E + dialogue", 1: "main + DE", 2: "main + associated",
               3: "M&E + dialogue + associated", 4: "main + DE + associated",
               5: "arbitrary", 6: "EMDF only"}


def emdf_info(b, st):
    v = b.u(2)
    if v == 3:
        v += b.vb(2)
    k = b.u(3)
    if k == 7:
        k += b.vb(3)
    sidx = None
    if b.u(1):                            # b_emdf_payloads_substream_info
        sidx = b.u(2)
        if sidx == 3:
            sidx += b.vb(2)
    prim, sec = b.u(2), b.u(2)
    if PROT_PRIMARY[prim] is None:
        raise ValueError("protection_length_primary == 0 (Reserved)")
    b.u(PROT_PRIMARY[prim])
    if PROT_SECOND[sec]:
        b.u(PROT_SECOND[sec])
    st["emdf"] += 1
    return dict(version=v, key_id=k, substream_index=sidx)


def frame_rate_multiply_info(b, fri, st):
    frf = 1
    if fri in (2, 3, 4):
        if b.u(1):
            frf = 4 if b.u(1) else 2
    elif fri in (0, 1, 7, 8, 9):
        if b.u(1):
            frf = 2
    st["frame_rate_factor"] = frf
    return frf


def frame_rate_fractions_info(b, fri, frf):
    """TS 103 190-2 6.2.1.4 -> frame_rate_fraction."""
    if fri in (5, 6, 7, 8, 9):
        if frf == 1 and b.u(1):
            return 2
    elif fri in (10, 11, 12):
        if b.u(1):
            return 4 if b.u(1) else 2
    return 1


def sgi_specifier(b, st=None):
    """6.2.1.7 -> the group index. bitstream_version 1 (a presentation of
    presentation_config 7, phase E): the ac4_substream_group_info itself,
    appended to st["_v1_groups"], whose index it returns."""
    if st is not None and st.get("bitstream_version") == 1 and \
            "_v1_groups" in st:
        st["_v1_groups"].append(substream_group_info(b, st))
        return len(st["_v1_groups"]) - 1
    g = b.u(3)
    if g == 7:
        g += b.vb(2)
    return g


def substream_index(b):
    i = b.u(2)
    if i == 3:
        i += b.vb(2)
    return i


def presentation_substream_info(b):
    """6.2.1.12 -> (b_alternative, b_pres_ndot, substream_index)."""
    alt = b.u(1)
    ndot = b.u(1)
    return alt, ndot, substream_index(b)


def presentation_config_ext_info(b, st):
    """6.2.1.5 (bitstream_version 2: the payload is skipped)."""
    n = b.u(5)
    if b.u(1):
        n += b.vb(2) << 5
    b.u(8 * n) if n <= 64 else [b.u(8) for _ in range(n)]


def presentation_v1_info(b, fri, st):
    """6.2.1.3 -> dict describing the presentation. Its "groups" are the
    ac4_sgi_specifier group indices it references, in order."""
    single = b.u(1)                       # b_single_substream_group
    cfg = 0
    if not single:
        cfg = b.u(3)
        if cfg == 7:
            cfg += b.vb(2)
    ver = 0
    while b.u(1):                         # presentation_version, unary
        ver += 1
    p = dict(config=None if single else cfg, version=ver, md_compat=None,
             presentation_id=None, frame_rate_factor=1,
             frame_rate_fraction=1, enabled=True, groups=[],
             n_substream_groups=0, multi_pid=0, pre_virtualized=0,
             pres_substream=None, b_alternative=0, b_pres_ndot=0,
             emdf=[])
    add_emdf = 0
    if (not single) and cfg == 6:
        add_emdf = 1
    else:
        p["md_compat"] = b.u(3)
        if b.u(1):                        # b_presentation_id
            p["presentation_id"] = b.vb(2)
        frf = frame_rate_multiply_info(b, fri, st)
        p["frame_rate_factor"] = frf
        p["frame_rate_fraction"] = frame_rate_fractions_info(b, fri, frf)
        p["emdf"].append(emdf_info(b, st))
        if b.u(1):                        # b_presentation_filter
            p["enabled"] = bool(b.u(1))   # b_enable_presentation
        if single:
            p["groups"] = [sgi_specifier(b, st)]
            p["n_substream_groups"] = 1
        else:
            p["multi_pid"] = b.u(1)
            table = {0: (2, 2), 1: (2, 1), 2: (2, 2), 3: (3, 3), 4: (3, 2)}
            if cfg in table:
                n_spec, p["n_substream_groups"] = table[cfg]
                p["groups"] = [sgi_specifier(b, st)
                               for _ in range(n_spec)]
            elif cfg == 5:
                n_sg = b.u(2) + 2
                if n_sg == 5:
                    n_sg += b.vb(2)
                p["n_substream_groups"] = n_sg
                p["groups"] = [sgi_specifier(b, st) for _ in range(n_sg)]
            else:
                presentation_config_ext_info(b, st)
        p["pre_virtualized"] = b.u(1)
        add_emdf = b.u(1)
        alt, ndot, sidx = presentation_substream_info(b)
        p.update(b_alternative=alt, b_pres_ndot=ndot, pres_substream=sidx)
    if add_emdf:
        n = b.u(2)
        if n == 0:
            n = b.vb(2) + 4
        for _ in range(n):
            p["emdf"].append(emdf_info(b, st))
    st["presentation_config"] = cfg
    st["single_group"] = single
    return p


def _sf_bitrate_ndot_index(b, present, frf, st, s):
    """The tail shared by the three substream info elements: b_sf_multiplier,
    b_bitrate_info, (add_ch_base for chan), b_audio_ndot x frame_rate_factor,
    substream_index."""
    s["sf_multiplier"] = None
    if st["fs_index"] == 1:
        if b.u(1):                        # b_sf_multiplier
            s["sf_multiplier"] = b.u(1)   # 0: 96 kHz, 1: 192 kHz
    if b.u(1):                            # b_bitrate_info
        if b.u(3) == 0b111:
            b.u(2)
    if s.get("channel_mode") in CH_MODES_WITH_ADD_CH_BASE:
        s["add_ch_base"] = b.u(1)
    s["ndot"] = []
    for _ in range(frf):
        st["ndot_bits"].append(b.p)
        s["ndot"].append(b.u(1))          # b_audio_ndot  (NOT b_iframe)
    s["index"] = substream_index(b) if present else None


def substream_info_chan(b, present, frf, st):
    cm = channel_mode(b)
    st["channel_modes"].append(cm)
    s = dict(kind="chan", channel_mode=cm)
    # the spec compares the channel_mode CODE with these bit patterns;
    # channel_mode() returns ch_mode, so compare with their ch_mode
    if cm in CH_MODES_WITH_TOP:
        s["b_4_back_channels_present"] = b.u(1)
        s["b_centre_present"] = b.u(1)
        s["top_channels_present"] = b.u(2)
    _sf_bitrate_ndot_index(b, present, frf, st, s)
    return s


# bed_dyn_obj_assignment / ac4_substream_info_obj tables (6.2.1.10, 6.2.1.11)
ISF_N = (4, 8, 10, 14, 15, 30)
BED_AJOC_N = (2, 3, 5, 7, 9, 7, 9, 11)
BED_OBJ_N = (2, 3, 6, 8, 10, 8, 10, 12)
STD_BED_N = (2, 1, 1, 2, 2, 2, 2, 2, 2, 1)


def _isf_n(b):
    c = b.u(3)
    if c >= len(ISF_N):
        raise ValueError(f"isf_config {c} reserved")
    return ISF_N[c]


def bed_dyn_obj_assignment(b, n_signals):
    """6.2.1.10 -> number of bed/ISF objects it assigns (the rest of the
    n_signals are dynamic)."""
    if b.u(1):                            # b_dyn_objects_only
        return 0
    if b.u(1):                            # b_isf
        return _isf_n(b)
    if b.u(1):                            # b_ch_assign_code
        return BED_AJOC_N[b.u(3)]
    if b.u(1):                            # b_channel_assignment_flags_present
        if b.u(1):                        # b_nonstd_..._flags_present
            fl = b.u(17)
            return sum(1 for i in range(17)
                       if (fl >> i) & 1 and i not in (3, 16))
        fl = b.u(10)
        return sum(STD_BED_N[i] for i in range(10)
                   if (fl >> (9 - i)) & 1 and i not in (2, 9))
    n_bed = 1
    if n_signals > 1:
        n_bed = b.u(max(1, (n_signals - 1).bit_length())) + 1
    return sum(1 for _ in range(n_bed) if b.u(4) != 3)


# Tables 63 (A-JOC coded beds), 64, 65 (TS 103 190-2 6.3.2.10.8)
BED_AJOC_CH = (("L", "R"), ("L", "R", "C"), ("L", "R", "C", "Ls", "Rs"),
               ("L", "R", "C", "Ls", "Rs", "Tsl", "Tsr"),
               ("L", "R", "C", "Ls", "Rs", "Tfl", "Tfr", "Tbl", "Tbr"),
               ("L", "R", "C", "Ls", "Rs", "Lb", "Rb"),
               ("L", "R", "C", "Ls", "Rs", "Lb", "Rb", "Tsl", "Tsr"),
               ("L", "R", "C", "Ls", "Rs", "Lb", "Rb", "Tfl", "Tfr", "Tbl",
                "Tbr"))
NONSTD_CH = ("L", "R", "C", "lfe", "Ls", "Rs", "Lb", "Rb", "Tfl", "Tfr",
             "Tsl", "Tsr", "Tbl", "Tbr", "Lw", "Rw", "lfe2")
STD_CH = (("L", "R"), ("C",), ("lfe",), ("Ls", "Rs"), ("Lb", "Rb"),
          ("Tfl", "Tfr"), ("Tsl", "Tsr"), ("Tbl", "Tbr"), ("Lw", "Rw"),
          ("lfe2",))


def obj_assignment(b, n_signals):
    """bed_dyn_obj_assignment (6.2.1.10) -> per signal its type: a bed
    channel name (Tables 63 - 66), "ISF", or "DYN" (the signals the
    assignment leaves; n_signals long)."""
    types = []
    if not b.u(1):                        # b_dyn_objects_only
        if b.u(1):                        # b_isf
            types = ["ISF"] * _isf_n(b)
        elif b.u(1):                      # b_ch_assign_code
            types = list(BED_AJOC_CH[b.u(3)])
        elif b.u(1):                      # b_channel_assignment_flags_present
            if b.u(1):
                fl = b.u(17)
                types = [NONSTD_CH[i] for i in range(17)
                         if (fl >> (16 - i)) & 1 and i not in (3, 16)]
            else:
                fl = b.u(10)
                for i in range(10):
                    if (fl >> (9 - i)) & 1 and i not in (2, 9):
                        types += list(STD_CH[i])
        else:
            n_bed = 1
            if n_signals > 1:
                n_bed = b.u(max(1, (n_signals - 1).bit_length())) + 1
            for _ in range(n_bed):
                v = b.u(4)
                if v != 3:
                    types.append(NONSTD_CH[v])
    return (types + ["DYN"] * n_signals)[:n_signals]


def oamd_common_data(b):
    """6.2.2.x: skipped by its own size fields."""
    if b.u(1) == 0:                       # b_default_screen_size_ratio
        b.u(5)
    b.u(1)                                # b_bed_object_chan_distribute
    if b.u(1):                            # b_additional_data
        n = b.u(1) + 1
        if n == 2:
            n += b.vb(2)
        for _ in range(n):
            b.u(8)


def substream_info_ajoc(b, present, frf, st):
    s = dict(kind="ajoc", b_lfe=b.u(1))
    s["static"] = b.u(1)                  # b_static_dmx
    if s["static"]:
        s["n_dmx"] = 5
        s["dmx_types"] = ["L", "R", "C", "Ls", "Rs"]
    else:
        s["n_dmx"] = b.u(4) + 1
        s["dmx_types"] = obj_assignment(b, s["n_dmx"])
    if b.u(1):                            # b_oamd_common_data_present
        oamd_common_data(b)
    n = b.u(4) + 1
    if n == 16:
        n += b.vb(3)
    s["n_upmix"] = n
    s["umx_types"] = obj_assignment(b, n)
    s["n_bed_isf"] = sum(1 for t in s["umx_types"] if t != "DYN")
    _sf_bitrate_ndot_index(b, present, frf, st, s)
    return s


# direct-coded beds (6.2.1.11): bed_chan_assign_code with the LFE
# (Table 63's channels, the LFE fourth)
BED_OBJ_CH = tuple(c[:3] + ("lfe",) + c[3:] if len(c) > 3 else c
                   for c in BED_AJOC_CH)


def substream_info_obj(b, present, frf, st):
    """ac4_substream_info_obj (6.2.1.11) -> dict: n_objects (the coded
    signals of audio_data_objs, Table 60 without the LFE), objects
    ("dynamic", "bed", "isf", "reserved"), b_lfe (dynamic objects), and
    assign: the bed's or ISF's object types when this substream starts
    them (b_bed_start / b_isf_start), None when it extends the group's
    previous one. obj_split() turns them into each substream's objects."""
    code = b.u(3)
    if code > 5:
        raise ValueError(f"n_objects_code {code} reserved")
    s = dict(kind="obj", n_objects=(0, 1, 2, 3, 5, 7)[code], b_lfe=0,
             assign=None)
    if b.u(1):                            # b_dynamic_objects
        s["b_lfe"] = b.u(1)
        s["objects"] = "dynamic"
    elif b.u(1):                          # b_bed_objects
        s["objects"] = "bed"
        if b.u(1):                        # b_bed_start
            if b.u(1):                    # b_ch_assign_code
                s["assign"] = list(BED_OBJ_CH[b.u(3)])
            elif b.u(1):                  # nonstd flags present
                fl = b.u(17)
                s["assign"] = [NONSTD_CH[i] for i in range(17)
                               if (fl >> (16 - i)) & 1]
            else:
                fl = b.u(10)
                s["assign"] = [c for i in range(10) if (fl >> (9 - i)) & 1
                               for c in STD_CH[i]]
    elif b.u(1):                          # b_isf
        s["objects"] = "isf"
        if b.u(1):                        # b_isf_start
            c = b.u(3)
            if c >= len(ISF_N):
                raise ValueError(f"isf_config {c} reserved")
            s["assign"] = [f"ISF:{c}:{i}" for i in range(ISF_N[c])]
    else:
        s["objects"] = "reserved"
        b.u(8 * b.u(4))
    _sf_bitrate_ndot_index(b, present, frf, st, s)
    return s


def obj_split(g):
    """The objects of a substream group's object substreams (interpreted:
    6.2.1.11 restarts n_objs per substream, while the group's one
    oamd_substream covers them all): each direct-coded substream s gets
    s["types"], its objects in OAMD order ("lfe" / "lfe2", a bed channel
    name, "DYN", or "ISF:<isf_config>:<i>"), s["lfe"] (audio_data_objs'
    b_lfe: one of its objects is an LFE, coded first as mono_data) and
    s["oamd_offset"], its first object's index in the group's
    oamd_dyndata_multi; g["obj_types"] lists (type, b_lfe, b_ajoc_coded)
    of the whole group (A-JOC substreams' upmix objects are A-JOC coded).

    Dynamic objects: the LFE (b_lfe) first, then n_objects of them
    (Table 60's count). A bed or ISF is shared out in its assignment
    order: a substream takes the next n_objects non-LFE objects and the
    first LFE not taken yet (6.3.2.10.6: one LFE per substream); one
    that does not start a bed (ISF) continues the group's last."""
    pending = []
    types = []
    bed = []
    for s in g["substreams"]:
        if s["kind"] == "obj" and s["objects"] == "bed" and s["assign"]:
            bed += [t for t in s["assign"] if t not in bed]
    for s in g["substreams"]:
        if s["kind"] == "ajoc":
            n = s["n_upmix"] + (1 if s["b_lfe"] else 0)
            types += [("DYN", 0, 1)] * n
            continue
        if s["kind"] != "obj":
            continue
        n = s["n_objects"]
        if s["objects"] == "dynamic":
            own = (["lfe"] if s["b_lfe"] else []) + ["DYN"] * n
        elif s["objects"] in ("bed", "isf"):
            if s["assign"] is not None:
                pending = list(s["assign"])
            own, lfe_taken, k = [], False, 0
            rest = []
            for t in pending:
                is_lfe = t in ("lfe", "lfe2")
                if is_lfe and not lfe_taken:
                    own.append(t)
                    lfe_taken = True
                elif not is_lfe and k < n:
                    own.append(t)
                    k += 1
                else:
                    rest.append(t)
            pending = rest
            own += ["DYN"] * (n - k)       # more signals than assigned
        else:
            own = ["DYN"] * n
        s["types"] = own
        s["bed"] = bed
        s["lfe"] = int(any(t in ("lfe", "lfe2") for t in own))
        s["oamd_offset"] = len(types)
        types += [(t if not t.startswith("ISF") else "ISF",
                   int(t in ("lfe", "lfe2")), 0) for t in own]
    g["obj_types"] = types


def substream_info_v0(b, frf, st):
    """ac4_substream_info (TS 103 190-1 Table 9, bitstream_version 0 / 1)
    -> (substream dict as substream_info_chan's, content_classifier,
    language). These substreams are channel-coded and always carry their
    substream_index; b_iframe takes the place of b_audio_ndot."""
    cm = channel_mode(b)
    st["channel_modes"].append(cm)
    s = dict(kind="chan", channel_mode=cm, sf_multiplier=None)
    if st["fs_index"] == 1:
        if b.u(1):                        # b_sf_multiplier
            s["sf_multiplier"] = b.u(1)
    if b.u(1):                            # b_bitrate_info
        if b.u(3) == 0b111:
            b.u(2)
    if cm in CH_MODES_WITH_ADD_CH_BASE:
        s["add_ch_base"] = b.u(1)
    cc, lang = (content_type(b) if b.u(1) else (None, None))
    s["ndot"] = []
    for _ in range(frf):
        st["ndot_bits"].append(b.p)
        s["ndot"].append(b.u(1))          # b_iframe
    s["index"] = substream_index(b)
    st["n_lf_substreams"] += 1
    st["channel_coded"] += 1
    return s, cc, lang


# substream infos per presentation_config (TS 103 190-1 Table 5)
V0_N_SUBSTREAMS = {0: 2, 1: 2, 2: 2, 3: 3, 4: 3, 5: 1}


def presentation_info_v0(b, fri, st, groups):
    """ac4_presentation_info (TS 103 190-1 Table 5; TS 103 190-2 6.2.1.2),
    bitstream_version 0 and 1 -> the dict presentation_v1_info gives. Each
    ac4_substream_info becomes a substream group of its own (appended to
    `groups`), with that substream's content type; there is no
    presentation substream (the metadata is in the audio substreams).
    presentation_config 7 of bitstream_version 1: the ac4_presentation_v1_
    info inside presentation_config_ext_info (with its substream groups)
    is the presentation (phase E)."""
    single = b.u(1)                       # b_single_substream
    cfg = 0
    if not single:
        cfg = b.u(3)
        if cfg == 7:
            cfg += b.vb(2)
    ver = 0
    while b.u(1):                         # presentation_version, unary
        ver += 1
    p = dict(config=None if single else cfg, version=ver, md_compat=None,
             presentation_id=None, frame_rate_factor=1,
             frame_rate_fraction=1, enabled=True, groups=[],
             n_substream_groups=0, multi_pid=0, pre_virtualized=0,
             pres_substream=None, b_alternative=0, b_pres_ndot=0,
             emdf=[], v0=True)
    if (not single) and cfg == 6:
        add_emdf = 1
    else:
        p["md_compat"] = b.u(3)
        if b.u(1):                        # b_presentation_id
            p["presentation_id"] = b.vb(2)
        frf = frame_rate_multiply_info(b, fri, st)
        p["frame_rate_factor"] = frf
        p["emdf"].append(emdf_info(b, st))
        n = 1 if single else V0_N_SUBSTREAMS.get(cfg, 0)
        hsf = 0 if single else b.u(1)    # b_hsf_ext
        if not single and cfg == 7 and st.get("bitstream_version") == 1:
            # presentation_config_ext_info (TS 103 190-2 6.2.1.5): an
            # ac4_presentation_v1_info, its substream groups inline
            # (ac4_sgi_specifier, 6.2.1.7), padded to a byte, inside
            # n_skip_bytes; it is the presentation decoded (phase E)
            nskip = b.u(5)
            if b.u(1):
                nskip += b.vb(2) << 5
            p0 = b.p
            st["_v1_groups"] = []
            pv1 = presentation_v1_info(b, fri, st)
            got = st.pop("_v1_groups")
            nread = b.p - p0
            if nread % 8:
                b.u(8 - nread % 8)
                nread += 8 - nread % 8
            nskip -= nread // 8
            if nskip < 0:
                raise ValueError("presentation_config_ext_info overrun")
            b.p += 8 * nskip
            if b.p > len(b.d) * 8:
                raise EOFError("presentation_config_ext_info")
            base = len(groups)
            groups.extend(got)
            pv1["groups"] = [base + g for g in pv1["groups"]]
            pv1["config_ext"] = cfg
            pv1["pre_virtualized"] = b.u(1)
            if b.u(1):                    # b_add_emdf_substreams
                k = b.u(2)
                if k == 0:
                    k = b.vb(2) + 4
                for _ in range(k):
                    pv1["emdf"].append(emdf_info(b, st))
            st["presentation_config"] = cfg
            st["single_group"] = single
            return pv1
        if not single and cfg not in V0_N_SUBSTREAMS:
            presentation_config_ext_info(b, st)
        for i in range(n):
            s, cc, lang = substream_info_v0(b, frf, st)
            if i == 0 and hsf:
                s["hsf_index"] = substream_index(b)
            p["groups"].append(len(groups))
            groups.append(dict(substreams_present=1, hsf_ext=int(
                i == 0 and hsf), substreams=[s], channel_coded=1,
                oamd=None, content_classifier=cc, language=lang))
        p["n_substream_groups"] = n
        p["pre_virtualized"] = b.u(1)
        add_emdf = b.u(1)
    if add_emdf:
        n = b.u(2)
        if n == 0:
            n = b.vb(2) + 4
        for _ in range(n):
            p["emdf"].append(emdf_info(b, st))
    st["presentation_config"] = cfg
    st["single_group"] = single
    return p


def hsf_ext_substream_info(b, present):
    return substream_index(b) if present else None


def substream_group_info(b, st):
    """6.2.1.6 -> dict for one substream group."""
    present = b.u(1)                      # b_substreams_present
    hsf = b.u(1)                          # b_hsf_ext
    single = b.u(1)                       # b_single_substream
    if single:
        n_lf = 1
    else:
        n_lf = b.u(2) + 2
        if n_lf == 5:
            n_lf += b.vb(2)
    st["n_lf_substreams"] += n_lf
    frf = st["frame_rate_factor"]
    g = dict(substreams_present=present, hsf_ext=hsf, substreams=[],
             channel_coded=b.u(1), oamd=None, content_classifier=None,
             language=None)
    if g["channel_coded"]:
        st["channel_coded"] += 1
        for _ in range(n_lf):
            sus_ver = (b.u(1) if st.get("bitstream_version") == 1
                       else 1)
            s = substream_info_chan(b, present, frf, st)
            s["sus_ver"] = sus_ver
            if hsf:
                s["hsf_index"] = hsf_ext_substream_info(b, present)
            g["substreams"].append(s)
    else:
        st["object_coded"] += 1
        if b.u(1):                        # b_oamd_substream
            ndot = b.u(1)                 # b_oamd_ndot
            g["oamd"] = dict(ndot=ndot, index=(substream_index(b)
                                              if present else None))
        for _ in range(n_lf):
            if b.u(1):                    # b_ajoc
                s = substream_info_ajoc(b, present, frf, st)
            else:
                s = substream_info_obj(b, present, frf, st)
            if hsf:
                s["hsf_index"] = hsf_ext_substream_info(b, present)
            g["substreams"].append(s)
        obj_split(g)
    if b.u(1):                            # b_content_type
        g["content_classifier"], g["language"] = content_type(b)
    return g


def parse(frame):
    """ac4_toc() of TS 103 190-2 6.2.1.1 -> dict.

    Besides the flat fields the damaged-input rules use, it holds
    "presentations" (a dict per ac4_presentation_v1_info, with the indices
    of the substream groups it references) and "groups" (a dict per
    ac4_substream_group_info, each with its substreams: channel_mode or
    object info, b_audio_ndot per frame_rate_factor, and the
    substream_index into substream_index_table)."""
    b = Bits(frame)
    st = collections.defaultdict(int)
    st["channel_modes"] = []
    st["ndot_bits"] = []
    bv = b.u(2)
    if bv == 3:
        bv += b.vb(2)
    st["bitstream_version"] = bv
    st["sequence_counter"] = b.u(10)
    if b.u(1):                            # b_wait_frames
        st["wait_frames"] = b.u(3)
        if st["wait_frames"] > 0:
            b.u(2)                        # br_code
    st["fs_index"] = b.u(1)
    fri = b.u(4)
    st["frame_rate_index"] = fri
    st["iframe_global_bit"] = b.p
    st["b_iframe_global"] = b.u(1)
    if b.u(1):                            # b_single_presentation
        n_pres = 1
    else:
        n_pres = b.vb(2) + 2 if b.u(1) else 0
    st["n_presentations"] = n_pres
    if b.u(1):                            # b_payload_base
        pb = b.u(5) + 1
        if pb == 0x20:
            pb += b.vb(3)
        st["payload_base"] = pb
    if bv <= 1:
        # TS 103 190-1 4.2.3 (bitstream_version 0; 1: TS 103 190-2
        # 6.2.1.2): ac4_presentation_info, substream infos inline
        groups = []
        pres = [presentation_info_v0(b, fri, st, groups)
                for _ in range(n_pres)]
        st["total_n_substream_groups"] = len(groups)
        return _index_table(b, st, pres, groups)
    if b.u(1):                            # b_program_id
        st["short_program_id"] = b.u(16)
        if b.u(1):                        # b_program_uuid_present
            b.u(128)
            st["uuid"] += 1
    pres = [presentation_v1_info(b, fri, st) for _ in range(n_pres)]
    total_sg = 1 + max((g for p in pres for g in p["groups"]), default=-1)
    st["total_n_substream_groups"] = total_sg
    groups = [substream_group_info(b, st) for _ in range(total_sg)]
    return _index_table(b, st, pres, groups)


def _index_table(b, st, pres, groups):
    """substream_index_table (TS 103 190-1 4.2.3.11) and the rest of
    ac4_toc()."""
    n_sub = b.u(2)                        # substream_index_table
    if n_sub == 0:
        n_sub = b.vb(2) + 4
    size_present = b.u(1) if n_sub == 1 else 1
    sizes = []
    if size_present:
        for _ in range(n_sub):
            more = b.u(1)
            s = b.u(10)
            if more:
                s += (b.vb(2) << 10)
            sizes.append(s)
    b.align()
    st["n_substreams"] = n_sub
    st["toc_bytes"] = b.p // 8
    if not size_present:
        # one substream without a size: the rest of the frame
        sizes = [max(0, len(b.d) - st["toc_bytes"]
                     - st.get("payload_base", 0))]
        st["size_implicit"] = 1
    st["substream_sizes"] = sizes
    st["presentations"] = pres
    st["groups"] = groups
    return st


def substream_offset(st, index):
    """Byte offset of substream `index` in the raw frame (TS 103 190-1
    4.3.3.12.4, Pseudocode 1: from the end of the byte-aligned ac4_toc,
    starting at payload_base)."""
    return (st["toc_bytes"] + st.get("payload_base", 0)
            + sum(st["substream_sizes"][:index]))


def substream_bytes(frame, st, index):
    o = substream_offset(st, index)
    return frame[o:o + st["substream_sizes"][index]]


# ---------------------------------------------------------------- selection
# TS 103 190-2 4.8.2: the decoder lists the presentations, picks one that
# is enabled and matches it, and decodes the substreams of every substream
# group the presentation references (substream_index into
# substream_index_table).

def presentation_substreams(st, pi):
    """-> [(group_index, group, substream)] of presentation pi, in the
    order its ac4_sgi_specifier elements reference the groups."""
    p = st["presentations"][pi]
    out = []
    for gi in p["groups"]:
        if gi < len(st["groups"]):
            g = st["groups"][gi]
            for s in g["substreams"]:
                out.append((gi, g, s))
    return out


def describe(st):
    """-> one dict per presentation: index, presentation_id, config (name),
    enabled, md_compat, version, language, content (classifiers of its
    groups), channel_modes, object_coded, and decodable (has a
    channel-coded audio substream this decoder can take)."""
    out = []
    for pi, p in enumerate(st.get("presentations") or []):
        subs = presentation_substreams(st, pi)
        langs = [g["language"] for _, g, _ in subs if g["language"]]
        cls = []
        for _, g, _ in subs:
            c = g["content_classifier"]
            if c is not None and CONTENT_CLASS[c] not in cls:
                cls.append(CONTENT_CLASS[c])
        cms = [s.get("channel_mode") for _, _, s in subs
               if s["kind"] == "chan"]
        cfg = p["config"]
        out.append(dict(
            index=pi, presentation_id=p["presentation_id"],
            config=("single" if cfg is None else PRES_CONFIG.get(cfg, cfg)),
            enabled=p["enabled"], md_compat=p["md_compat"],
            version=p["version"], language=langs[0] if langs else None,
            content=cls, channel_modes=cms,
            object_coded=any(s["kind"] != "chan" for _, _, s in subs),
            decodable=audio_substream(st, pi) is not None))
    return out


def audio_substreams(st, pi):
    """-> [(group_index, group, substream)] of the channel-coded
    substreams of presentation pi that have a substream_index, in TOC
    order, without dialogue enhancement substreams (presentation_config 1
    and 4: the second group, TS 103 190-2 Table 53; they are not mixed)."""
    if pi is None or not (0 <= pi < len(st.get("presentations") or [])):
        return []
    p = st["presentations"][pi]
    de_group = (p["groups"][1] if p.get("config") in (1, 4)
                and len(p["groups"]) > 1 else None)
    out = []
    for gi, g, s in presentation_substreams(st, pi):
        if gi == de_group:
            continue
        if (s["kind"] in ("chan", "obj") and s.get("index") is not None
                and s["index"] < len(st["substream_sizes"])):
            out.append((gi, g, s))
    if not out and AJOC_CORE:
        # no channel-coded substream: the A-JOC substreams, decoded in
        # core decoding mode (their downmix and its OAMD; elements.py)
        for gi, g, s in presentation_substreams(st, pi):
            if (gi != de_group and s["kind"] == "ajoc"
                    and s.get("index") is not None
                    and s["index"] < len(st["substream_sizes"])):
                out.append((gi, g, s))
    return out


# decode A-JOC substreams (core decoding) when a presentation has no
# channel-coded one; False: such presentations are not decodable
AJOC_CORE = True


def audio_substream(st, pi, k=0):
    """-> the substream dict of the k-th channel-coded substream of
    presentation pi (audio_substreams; k = 0: the one a single-substream
    decode takes), or None."""
    subs = audio_substreams(st, pi)
    return subs[k][2] if 0 <= k < len(subs) else None


# substream types (TS 103 190-2 4.8.3.2): per presentation_config the
# type of the 1st, 2nd and 3rd substream group (Table 53); for
# presentation_config 5 by content_classifier (Table 54)
TYPES_OF_CONFIG = {0: ("me", "dialogue"), 1: ("main", "de"),
                   2: ("main", "associated"),
                   3: ("me", "dialogue", "associated"),
                   4: ("main", "de", "associated")}
TYPE_OF_CLASS = ("main", "main", "associated", "associated", "dialogue",
                 "associated", "main", "main")


def substream_type(st, pi, gi):
    """-> "main" (main or music and effects), "dialogue", "associated" or
    "de" for substream group gi of presentation pi."""
    p = st["presentations"][pi]
    cfg = p.get("config")
    if cfg is None:
        return "main"                     # b_single_substream_group
    pos = p["groups"].index(gi) if gi in p["groups"] else 0
    if cfg in TYPES_OF_CONFIG:
        t = TYPES_OF_CONFIG[cfg]
        return ("main" if t[min(pos, len(t) - 1)] == "me"
                else t[min(pos, len(t) - 1)])
    cc = st["groups"][gi].get("content_classifier") if gi < len(
        st["groups"]) else None
    return TYPE_OF_CLASS[cc] if cc is not None else "main"


def choose(st, presentation=None, language=None):
    """-> the index of the presentation to decode, or None.

    presentation: an int selects by position in the TOC; "id:N" selects the
    presentation whose presentation_id is N. Otherwise the first enabled,
    decodable presentation, preferring one whose language matches
    `language` (a BCP 47 prefix, e.g. "fr"), then one whose content is
    "complete main" (not audio description, commentary, ...)."""
    pres = st.get("presentations") or []
    if presentation is not None:
        if isinstance(presentation, str) and presentation.startswith("id:"):
            want = int(presentation[3:])
            for pi, p in enumerate(pres):
                if p["presentation_id"] == want:
                    return pi
            return None
        pi = int(presentation)
        return pi if 0 <= pi < len(pres) else None
    cands = [d for d in describe(st) if d["enabled"] and d["decodable"]]
    if not cands:
        return None
    if language:
        lw = language.lower()
        hit = [d for d in cands if d["language"]
               and d["language"].lower().split("-")[0] == lw.split("-")[0]]
        if hit:
            cands = hit
    main = [d for d in cands if not d["content"]
            or "complete main" in d["content"]]
    return (main or cands)[0]["index"]
