# SPDX-License-Identifier: Apache-2.0
"""Object audio metadata (OAMD), TS 103 190-2 6.2.8 / 6.3.9: parsed.

oamd_common_data, oamd_timing_data, oamd_dyndata_single and
oamd_dyndata_multi with their object_info_block (object_basic_info,
object_render_info, add_per_object_md, ext_prec_pos), trim, bed_render_info
and headphone. Each object keeps its state between blocks (REUSE,
PART_REUSE and the differential position coding refer to the previous
obj_info_block): ObjState. The result per object is its gain (dB), its
room-anchored position (x left 0 .. right 1, y front 0 .. back 1, z floor
-1 .. ceiling 1; 6.3.9.8.4) and whether it is active.

Timing (oamd_timing_data: sample_offset, block offsets, ramp durations;
5.9) is parsed and reported, not applied: a renderer takes the last block
of a frame for the whole frame (render.py / objrender.py)."""
from __future__ import annotations

import math

DYN, BED, ISF = "DYN", "BED", "ISF"
NUM_TRIM_CONFIGS = 9                    # 6.3.9.10.4


def _clip3(lo, hi, v):
    return lo if v < lo else hi if v > hi else v


def _s3(v):
    """3-bit two's complement."""
    return v - 8 if v >= 4 else v


EXT_PREC = (1, 2, -1, -2)                # Tables 123 - 125


class ObjState:
    """One object's metadata as the last obj_info_block left it."""

    def __init__(self):
        self.gain = -math.inf            # Table 98 defaults (not active)
        self.priority = 0.0
        self.active = False
        self.zone_mask = 0
        self.width = 0
        self.default_render()

    def default_render(self):
        # Table 99 (as standard precision codes: 31 / 62 = 0.5)
        self.px, self.py, self.pz = 31, 31, 0
        self.x, self.y, self.z = 0.5, 0.5, 0.0

    def position(self):
        return (self.x, self.y, self.z)

    def set_pos(self, px, py, pz, ext=(0, 0, 0)):
        """6.3.9.8.4.5 - .8 (px, py in 0 .. 62, pz signed -15 .. 15)."""
        self.px, self.py, self.pz = px, py, pz
        ex, ey, ez = (_clip3(-2, 2, e) for e in ext)
        self.x = _clip3(0.0, 1.0, _clip3(0, 62, px) / 62 + ex / (62 * 5))
        self.y = _clip3(0.0, 1.0, _clip3(0, 62, py) / 62 + ey / (62 * 5))
        self.z = _clip3(-1.0, 1.0, _clip3(-15, 15, pz) / 15
                        + ez / (15 * 5))

    def copy(self):
        c = ObjState.__new__(ObjState)
        c.__dict__.update(self.__dict__)
        return c


def oamd_common_data(b):
    """6.2.8.1 -> dict."""
    out = {}
    if b.u(1) == 0:                      # b_default_screen_size_ratio
        out["master_screen_size_ratio_code"] = b.u(5)
    out["b_bed_object_chan_distribute"] = b.u(1)
    if b.u(1):                           # b_additional_data
        n = b.u(1) + 1
        if n == 2:
            n += b.vb(2)
        bits = 8 * n
        p0 = b.p
        trim(b)
        if bits - (b.p - p0) > 0:
            out["bed_render_info"] = bed_render_info(b)
        if bits - (b.p - p0) > 0:
            headphone(b)
        rest = bits - (b.p - p0)
        if rest < 0:
            raise ValueError("oamd_common_data: add_data overrun")
        b.p += rest
    return out


def trim(b):
    """6.2.8.9 (parsed, not applied)."""
    if b.u(1):                           # b_trim_present
        b.u(2)                           # warp_mode
        b.u(2)                           # reserved
        if b.u(2) == 0b10:               # global_trim_mode
            for _ in range(NUM_TRIM_CONFIGS):
                if b.u(1) == 0 and b.u(1) == 0:   # b_default_trim, disable
                    pres = b.u(5)        # trim_balance_presence[]
                    for i in (4, 3, 2):
                        if (pres >> i) & 1:
                            b.u(4)
                    for i in (1, 0):
                        if (pres >> i) & 1:
                            b.u(1)
                            b.u(4)


def stereo_dmx_coeff(b):
    """6.2.8.8a."""
    out = dict(loro_centre=b.u(3), loro_surround=b.u(3))
    if b.u(1):
        b.u(3)
        b.u(3)
    if b.u(1):
        out["lfe_mixgain"] = b.u(5)
    out["preferred_dmx_method"] = b.u(2)
    return out


def _tool_f_s(b, back, first, second, third):
    """tool_t*_to_f_s_b (back) / tool_t*_to_f_s: -> {gain: code}."""
    g = {}
    if b.u(1):                           # *_to_front
        g[first] = b.u(3)
        g[second] = 7
    elif back:
        if b.u(1):                       # *_to_side
            g[second] = b.u(3)
        else:
            g[third] = b.u(3)
            g[second] = 7
    else:
        g[second] = b.u(3)
    return g


def bed_render_info(b):
    """6.2.8.8 -> dict (gain codes)."""
    out = {}
    if not b.u(1):                       # b_bed_render_info
        return out
    if b.u(1):                           # b_stereo_dmx_coeff
        out["stereo"] = stereo_dmx_coeff(b)
    if b.u(1):                           # b_cdmx_data_present
        g = {}
        if b.u(1):
            g["gain_w_to_f"] = b.u(3)
        if b.u(1):
            g["gain_b"] = b.u(3)
        if b.u(1):                       # b_tm_ch_present
            if b.u(1):
                g.update(_tool_f_s(b, True, "t2a", "t2b", "t2c"))
            if b.u(1):
                g.update(_tool_f_s(b, False, "t2a", "t2b", "t2c"))
        tb = b.u(1)                      # b_tb_ch_present
        if tb:
            if b.u(1):
                g.update(_tool_f_s(b, True, "t2d", "t2e", "t2f"))
            if b.u(1):
                g.update(_tool_f_s(b, False, "t2d", "t2e", "t2f"))
        tf = b.u(1)                      # b_tf_ch_present
        if tf:
            if b.u(1):
                g.update(_tool_f_s(b, True, "t2a", "t2b", "t2c"))
            if b.u(1):
                g.update(_tool_f_s(b, False, "t2a", "t2b", "t2c"))
        if tb or tf:
            if b.u(1):
                g["gain_tfb_to_tm"] = b.u(3)
        out["cdmx"] = g
    return out


def headphone(b):
    """6.2.8.9a."""
    if b.u(1):
        if b.u(3) in (0b001, 0b010):
            b.u(1)


def oamd_timing_data(b):
    """6.2.8.2 -> dict(sample_offset, blocks=[(block_offset_factor,
    ramp_duration)]) (Tables 92, 93; ramp_duration_code 0 / 1 / 2: 0, 512,
    1536 samples; the table form is kept as its index)."""
    so = 0
    if b.u(1):                           # oa_sample_offset_type 0b1x
        if b.u(1) == 0:                  # 0b10: oa_sample_offset_code
            so = 16 if b.u(1) == 0 else (8 if b.u(1) == 0 else 24)
        else:                            # 0b11
            so = b.u(5)
    blocks = []
    for _ in range(b.u(3)):              # num_obj_info_blocks
        off = b.u(6)
        rc = b.u(2)
        if rc == 3:
            ramp = ("table", b.u(4)) if b.u(1) else b.u(11)
        else:
            ramp = (0, 512, 1536)[rc]
        blocks.append((off, ramp))
    return dict(sample_offset=so, blocks=blocks)


def object_basic_info(b, st):
    """6.2.8.6 (Tables 100 - 102)."""
    if b.u(1):                           # b_default_basic_info_md
        st.gain, st.priority = 0.0, 1.0
        return
    md = 0 if b.u(1) == 0 else (0b10 if b.u(1) == 0 else 0b11)
    if md in (0, 0b10):
        if b.u(1) == 0:                  # object_gain_code 0b0
            v = b.u(6)
            st.gain = float(15 - v if v <= 14 else 14 - v)
        elif b.u(1) == 0:                # 0b10
            st.gain = -math.inf
        # 0b11: the previous object's gain (kept)
    else:
        st.gain = 0.0
    st.priority = b.u(5) / 31.0 if md in (0b10, 0b11) else 1.0


def object_render_info(b, st, status, b_no_delta):
    """6.2.8.7 (6.3.9.8.4): position, zones, other properties."""
    if status == "ALL_NEW":
        other = zone = pos = 1
    else:
        other, zone, pos = b.u(1), b.u(1), b.u(1)
    if pos:
        diff = 0 if b_no_delta else b.u(1)
        if diff:
            dx, dy, dz = _s3(b.u(3)), _s3(b.u(3)), _s3(b.u(3))
            st.set_pos(_clip3(0, 62, st.px + dx), _clip3(0, 62, st.py + dy),
                       _clip3(-15, 15, st.pz + dz))
        else:
            px, py = b.u(6), b.u(6)
            sign = 1 if b.u(1) else -1
            pz = b.u(4)
            st.set_pos(px, py, sign * pz)
    if zone:
        if b.u(1) == 0:                  # b_grouped_zone_defaults
            fl = b.u(3)
            if fl & 0b100:
                st.zone_mask = b.u(3)
    if other:
        if b.u(1) == 0:                  # b_grouped_other_defaults
            m = b.u(4)
            if m & 0b0001:
                if b.u(1) == 0:          # object_width_mode
                    st.width = b.u(5)
                else:
                    st.width = (b.u(5), b.u(5), b.u(5))
            if m & 0b0010:
                b.u(3)
                b.u(2)
            if m & 0b0100:
                if b.u(1) == 0:          # b_obj_at_infinity
                    b.u(4)
            if m & 0b1000:
                dm = b.u(2)
                if dm == 0:
                    b.u(2)
                elif dm & 0b10:
                    b.u(6)


def ext_prec_pos(b):
    """6.2.8.11 -> (ex, ey, ez) (Tables 122 - 125)."""
    pres = b.u(3)
    e = [0, 0, 0]
    for k, bit in ((0, 2), (1, 1), (2, 0)):
        if (pres >> bit) & 1:
            e[k] = EXT_PREC[b.u(2)]
    return tuple(e)


def add_per_object_md(b, st, b_object_not_active, b_dynamic_object):
    """6.2.8.10 (the argument order is the call's, 6.2.8.5)."""
    b.u(1)                               # b_obj_trim_disable
    if not b_object_not_active and b_dynamic_object:
        if b.u(1):                       # b_ext_prec_pos
            st.set_pos(st.px, st.py, st.pz, ext_prec_pos(b))
    if b.u(1):                           # b_headphone
        b.u(2)
        b.u(1)


def object_info_block(b, st, b_no_delta, b_dynamic_object):
    """6.2.8.5: one update of one object (st updated in place)."""
    not_active = b.u(1)
    st.active = not not_active
    if not_active:
        basic = "DEFAULT"
    elif b_no_delta:
        basic = "ALL_NEW"
    else:
        basic = "REUSE" if b.u(1) else "ALL_NEW"
    if basic == "ALL_NEW":
        object_basic_info(b, st)
    elif basic == "DEFAULT":
        st.gain, st.priority = -math.inf, 0.0
    if not_active:
        render = "DEFAULT"
    elif b_dynamic_object:
        if b_no_delta:
            render = "ALL_NEW"
        elif b.u(1):                     # b_render_info_reuse
            render = "REUSE"
        else:
            render = "PART_REUSE" if b.u(1) else "ALL_NEW"
    else:
        render = "DEFAULT"
    if render in ("ALL_NEW", "PART_REUSE"):
        object_render_info(b, st, render, b_no_delta)
    elif render == "DEFAULT":
        st.default_render()
    if b.u(1):                           # b_add_table_data
        size = b.u(4) + 1
        p0 = b.p
        add_per_object_md(b, st, not_active, b_dynamic_object)
        rest = 8 * size - (b.p - p0)
        if rest < 0:
            raise ValueError("add_table_data overrun")
        b.p += rest


def oamd_dyndata_single(b, states, n_blocks, b_iframe, b_alternative,
                        obj_type, b_lfe):
    """6.2.8.3: n_blocks updates of every object (states: one ObjState
    per object). The alternative metadata of b_alternative is parsed and
    dropped."""
    n = len(states)
    for i in range(n):
        dyn = obj_type[i] == DYN and not b_lfe[i]
        for blk in range(n_blocks):
            object_info_block(b, states[i], bool(b_iframe) and blk == 0,
                              dyn)
    if b_alternative:
        b.u(1)                           # b_ducking_disabled
        c = b.u(2)
        if c == 3:
            b.vb(2)
        ns = b.u(2)
        if ns == 3:
            ns += b.vb(2)
        for _ in range(ns):
            keep = b.u(1)
            if not keep:
                n_dp = n
                if n and obj_type[0] == ISF:
                    n_dp = 1
                elif b.u(1):             # b_common_data
                    n_dp = 1
                for dp in range(n_dp):
                    t = obj_type[dp] if dp < n else DYN
                    if t in (BED, ISF):
                        if b.u(1):
                            b.u(6)
                    elif t == DYN:
                        if b.u(1):
                            b.u(6)
                        if not b_lfe[dp] and b.u(1):
                            b.u(6)
                            b.u(6)
                            b.u(1)
                            b.u(4)
            if b.u(1):                   # b_additional_data
                bits = (b.vb(2) + 1) * 8
                p0 = b.p
                if not keep:
                    for o in range(n):
                        if obj_type[o] == DYN and not b_lfe[o] and b.u(1):
                            ext_prec_pos(b)
                rest = bits - (b.p - p0)
                if rest < 0:
                    raise ValueError("ext_prec_alt_pos overrun")
                b.p += rest


def oamd_dyndata_multi(b, states, n_blocks, b_iframe, obj_type, b_lfe,
                       b_ajoc_coded):
    """6.2.8.4."""
    for i in range(len(states)):
        if b_ajoc_coded[i]:
            continue
        dyn = obj_type[i] == DYN and not b_lfe[i]
        for blk in range(n_blocks):
            object_info_block(b, states[i], bool(b_iframe) and blk == 0,
                              dyn)
