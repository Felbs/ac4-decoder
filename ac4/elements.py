# SPDX-License-Identifier: Apache-2.0
"""The channel elements other than the channel_pair_element and the
5_X_channel_element (TS 103 190-1 4.2.6.1, 4.2.6.5, 4.2.6.14, Table 19):
single_channel_element (mono), 3_0_channel_element (3.0) and
7_X_channel_element (7.0 / 7.1 in the three layouts of Table 88), codec
modes SIMPLE and ASPX. Their tracks are mapped to channels as Tables 182
and 183 say; 7_X's ASPX_ACPL_1 / _2 are concealed (element.parse: not
built).

Channel keys: L R C lfe Ls Rs, and for 7_X the two additional channels of
the layout (LAYOUT_7X): Lb Rb (3/4/0), Lw Rw (5/2/0), Tfl Tfr (3/2/2).
The A-SPX blocks of Table 213 go to the decoder's pair groups 0 (L/R),
1 (the second pair of the table), 2 (C) and 3 (the fourth)."""
from __future__ import annotations

from . import channels as C
from . import multichannel as MC
from .robust import Damaged

# ch_mode (Table 88) -> (layout, has_lfe)
CH_MODE_7X = {5: ("3/4/0", False), 6: ("3/4/0", True),
              7: ("5/2/0", False), 8: ("5/2/0", True),
              9: ("3/2/2", False), 10: ("3/2/2", True)}
# layout -> the additional channels, and the A-SPX pairs of Table 213 in
# bitstream order (block groups 0, 1, (2 = C), 3)
LAYOUT_7X = {
    "3/4/0": (("Lb", "Rb"), (("L", "R"), ("Ls", "Rs"), ("Lb", "Rb"))),
    "5/2/0": (("Lw", "Rw"), (("L", "R"), ("Lw", "Rw"), ("Ls", "Rs"))),
    "3/2/2": (("Tfl", "Tfr"), (("L", "R"), ("Ls", "Rs"), ("Tfl", "Tfr"))),
}


def channels_of(element, ch_mode=None):
    """The output channels of an element, in WAV order (L R C LFE Ls Rs,
    then the 7_X additional pair)."""
    if element == "mono":
        return ("C",)
    if element == "pair":
        return ("L", "R")
    if element == "3_0":
        return ("L", "R", "C")
    if element == "5_0":
        return ("L", "R", "C", "Ls", "Rs")
    if element == "5_X":
        return ("L", "R", "C", "lfe", "Ls", "Rs")
    if element == "ice":
        # the coded tracks (A'' .. G'' as channels, and the LFE)
        lfe = CH_MODE_ICE[ch_mode][0]
        return (("L", "R", "C") + (("lfe",) if lfe else ())
                + ("Ls", "Rs", "Tfl", "Tfr"))
    layout, lfe = CH_MODE_7X[ch_mode]
    extra = LAYOUT_7X[layout][0]
    return (("L", "R", "C") + (("lfe",) if lfe else ()) + ("Ls", "Rs")
            + extra)


def aspx_pairs(element, ch_mode=None):
    """The pair groups of the A-SPX blocks: {group: (ka, kb)}."""
    if element == "ice":
        return dict(ICE_PAIRS)
    if element == "7_X":
        p = LAYOUT_7X[CH_MODE_7X[ch_mode][0]][1]
        return {0: p[0], 1: p[1], 2: ("C", "C"), 3: p[2]}
    return {0: ("L", "R"), 1: ("Ls", "Rs"), 2: ("C", "C")}


def _head(sub):
    b = C.Bits(sub)
    asz, a0 = C.audio_size(b)
    return b, asz, a0


def _finish(out, b, sub, asz, a0, mode):
    out.update(bits=b.p, nbits=len(sub) * 8, audio_size=asz,
               audio_start=a0, mode=mode, acpl_blocks=0)
    return out


def single_channel_element(sub, T, b_iframe=False):
    """Table 20 (mono): the track on C; A-SPX on it (block group 2)."""
    b, asz, a0 = _head(sub)
    mode = b.u(1)                                    # mono_codec_mode
    aspx = C.aspx_config(b) if (b_iframe and mode == 1) else None
    comp = C.companding_control(b, 1) if mode == 1 else None
    out = dict(C=C.mono_data(b, T, b_lfe=False), aspx=aspx, compand=comp,
               codec_mode=mode, steps=[], order=("C",),
               aspx_blocks=((2, False),) if mode == 1 else (),
               compand_idx={2: (0, 0)})
    return _finish(out, b, sub, asz, a0, ("mono", mode))


def three_0_channel_element(sub, T, b_iframe=False):
    """Table 24 (3.0): stereo_data + mono_data (3_0_coding_config 0) or
    three_channel_data (1) onto L, R, C (5.3.4.2); A-SPX on (L, R) and C;
    companding_control(3): L, R, C (Table 212)."""
    b, asz, a0 = _head(sub)
    mode = b.u(1)                                    # 3_0_codec_mode
    aspx = comp = None
    if mode == 1:
        if b_iframe:
            aspx = C.aspx_config(b)
        comp = C.companding_control(b, 3)
    out = dict(aspx=aspx, compand=comp, codec_mode=mode)
    if b.u(1) == 0:                                  # 3_0_coding_config
        l, r, st = C.stereo_data(b, T)
        out.update(L=l, R=r, C=C.mono_data(b, T, b_lfe=False),
                   steps=[(("L", "R"), MC.two_channel_step(st))],
                   order=("L", "R", "C"))
    else:
        tr, s3 = MC.three_channel_data(b, T)
        out.update(L=tr[0], R=tr[1], C=tr[2],
                   steps=[(("L", "R", "C"), s3)], order=("L", "R", "C"))
    out.update(aspx_blocks=((0, True), (2, False)) if mode == 1 else (),
               compand_idx={0: (0, 1), 2: (2, 2)})
    return _finish(out, b, sub, asz, a0, ("3_0", mode))


def seven_x_channel_element(sub, T, ch_mode, b_iframe=False):
    """Table 33 (7.0 / 7.1), codec modes SIMPLE and ASPX: the tracks onto
    the preliminary channels A B C D E F G (Table 182, with 2ch_mode),
    then the final mapping of Table 183 with the b_use_sap_add_ch
    parameters (a_i = d_i = 1, b_i = c_i = 0 without them).

    Readings: the two chparam_info() of b_use_sap_add_ch come before the
    additional channels' sf_info; each is read with the framing of the
    track it pairs F (or G) with (D / E for 3/4/0, A / B otherwise), and
    the pair is processed per transform window (after ungrouping), which
    needs the two tracks to have the same windows: a frame where they
    differ is damage (asf.add_ch)."""
    layout, has_lfe = CH_MODE_7X[ch_mode]
    extra = LAYOUT_7X[layout][0]
    b, asz, a0 = _head(sub)
    mode = b.u(2)                                    # 7_X_codec_mode
    if mode >= 2:
        raise Damaged("element.parse",
                      "7_X ASPX_ACPL_1 / _2 not implemented")
    aspx = C.aspx_config(b) if (b_iframe and mode == 1) else None
    out = dict(aspx=aspx, compand=None, codec_mode=mode)
    order = []
    if has_lfe:
        out["lfe"] = C.mono_data(b, T, b_lfe=True)
        order.append("lfe")
    cc = b.u(2)                                      # coding_config
    steps = []
    pre = {}                                         # A .. G -> track key
    if cc == 0:
        mode2ch = b.u(1)
        k1, k2 = (("A", "B"), ("D", "E")) if mode2ch == 0 else \
            (("A", "D"), ("B", "E"))
        for kk in (k1, k2):
            x, y, st = C.two_channel_data(b, T)
            out[kk[0]], out[kk[1]] = x, y
            steps.append((kk, MC.two_channel_step(st)))
            order += list(kk)
    elif cc == 1:
        tr, s3 = MC.three_channel_data(b, T)
        out.update(A=tr[0], B=tr[1], C=tr[2])
        steps.append((("A", "B", "C"), s3))
        x, y, st = C.two_channel_data(b, T)
        out.update(D=x, E=y)
        steps.append((("D", "E"), MC.two_channel_step(st)))
        order += ["A", "B", "C", "D", "E"]
    elif cc == 2:
        tr, s4 = MC.four_channel_data(b, T)
        out.update(A=tr[0], B=tr[1], D=tr[2], E=tr[3])
        steps.append((("A", "B", "D", "E"), s4))
        order += ["A", "B", "D", "E"]
    else:
        tr, s5 = MC.five_channel_data(b, T)
        out.update(A=tr[0], B=tr[1], C=tr[2], D=tr[3], E=tr[4])
        steps.append((("A", "B", "C", "D", "E"), s5))
        order += ["A", "B", "C", "D", "E"]
    # the pairs of Table 183: (base, additional) -> (out0, out1)
    if layout == "3/4/0":
        fin = [(("D", "F"), ("Ls", "Lb")), (("E", "G"), ("Rs", "Rb"))]
        plain = {"A": "L", "B": "R"}
    else:
        fin = [(("A", "F"), ("L", extra[0])), (("B", "G"), ("R", extra[1]))]
        plain = {"D": "Ls", "E": "Rs"}
    sap = [None, None]
    if b.u(1):                                       # b_use_sap_add_ch
        sap = [MC.read_chparam(b, out[base]["framing"], T)
               for (base, _a), _o in fin]
    x, y, st = C.two_channel_data(b, T)              # additional channels
    out.update(F=x, G=y)
    steps.append((("F", "G"), MC.two_channel_step(st)))
    order += ["F", "G"]
    if cc in (0, 2):
        out["C"] = C.mono_data(b, T, b_lfe=False)
        order.append("C")
    # rename the preliminary keys to channels
    ren = dict(plain, C="C", lfe="lfe")
    side = []
    for i, ((base, add), (o0, o1)) in enumerate(fin):
        ren[base], ren[add] = o0, o1
        if sap[i] is not None:
            fb, fa = out[base]["framing"], out[add]["framing"]
            if (fb.num_windows != fa.num_windows
                    or [fb.length_g(fb.w2g[w]) for w in range(fb.num_windows)]
                    != [fa.length_g(fa.w2g[w])
                        for w in range(fa.num_windows)]):
                raise Damaged("asf.add_ch", "framings differ")
            side.append(((o0, o1), sap[i], fb))
    res = {ren[k]: v for k, v in out.items() if k in ren}
    out = {k: v for k, v in out.items()
           if k not in ("A", "B", "D", "E", "F", "G")}
    out.update(res)
    out["steps"] = [(tuple(ren[k] for k in kk), s) for kk, s in steps]
    out["side_steps"] = side
    out["order"] = tuple(ren[k] for k in order)
    pairs = LAYOUT_7X[layout][1]
    out["aspx_blocks"] = (((0, True), (1, True), (2, False), (3, True))
                          if mode == 1 else ())
    out["aspx_pairs"] = {0: pairs[0], 1: pairs[1], 2: ("C", "C"),
                         3: pairs[2]}
    return _finish(out, b, sub, asz, a0, ("7_X", mode))


# ---------------------------------------------------------------- immersive
# TS 103 190-2 6.2.4.1 (immersive_channel_element), 6.3.5 (Tables 73, 74),
# 5.2.3 (SMP, Tables 19, 20), 5.5.2 (A-CPL, Table 25, Pseudocode 2).
ICE_MODES = {0: "SCPL", 1: "ASPX_SCPL", 2: "ASPX_ACPL_1", 3: "ASPX_ACPL_2",
             4: "ASPX_AJCC"}
# ch_mode (Table 56) -> (b_lfe, b_5fronts) of audio_data_chan (6.2.3.1)
CH_MODE_ICE = {11: (False, False), 12: (True, False), 13: (False, True),
               14: (True, True)}
# The intermediate signals A'' .. G'' of a 7CH_STATIC core as the channels
# they carry (Table 23 for S-CPL, Table 25 for A-CPL): A L, B R, C C, D Ls,
# E Rs, F Tfl, G Tfr; the A-CPL modules of Pseudocode 2 (b_5fronts 0)
# rebuild (Ls, Lb), (Rs, Rb) from D'', E'' (decorrelator D0) and (Tfl,
# Tbl), (Tfr, Tbr) from F'', G'' (decorrelator D1).
ICE_TRACK = {"A": "L", "B": "R", "C": "C", "D": "Ls", "E": "Rs",
             "F": "Tfl", "G": "Tfr"}
# the full channel set of 7.X.4 in output order (WAV order: L R C LFE,
# the surrounds, the back pair, the four top channels)
ICE_FULL = ("L", "R", "C", "lfe", "Ls", "Rs", "Lb", "Rb", "Tfl", "Tfr",
            "Tbl", "Tbr")
# the A-SPX block groups of Table 8 (ASPX_ACPL_1 / _2: (A'', B''),
# (D'', E''), (F'', G''), C'')
ICE_PAIRS = {0: ("L", "R"), 1: ("Ls", "Rs"), 2: ("C", "C"),
             3: ("Tfl", "Tfr")}


def ice_channels(ch_mode, present=None):
    """The output channels of an immersive channel mode (ch_mode 11 / 12:
    7.0.4 / 7.1.4) given ac4_substream_info_chan's (b_4_back_channels_
    present, b_centre_present, top_channels_present) (6.3.2.7, Tables 57
    - 59): the back pair only when present (else 5.X.4), the top channels
    only when any are (top_channels_present 1: Tfl Tfr carry the original
    Tsl Tsr; 2: Tbl Tbr do; 3: all four). The centre is kept even when
    signalled silent (it is a channel of every layout of Table A.27)."""
    lfe = CH_MODE_ICE[ch_mode][0]
    back, _centre, top = present if present is not None else (1, 1, 3)
    out = ["L", "R", "C"] + (["lfe"] if lfe else []) + ["Ls", "Rs"]
    if back:
        out += ["Lb", "Rb"]
    if top in (1, 3):
        out += ["Tfl", "Tfr"]
    if top in (2, 3):
        out += ["Tbl", "Tbr"]
    return tuple(out)


def immersive_channel_element(sub, T, ch_mode, b_iframe=False):
    """6.2.4.1 for b_5fronts 0 (7.0.4 / 7.1.4), immersive_codec_mode
    ASPX_ACPL_2: the 7CH_STATIC core (Table 74) as tracks A .. G (Table
    19, 2ch_mode included), the b_use_sap_add_ch step on (D, F) and (E, G)
    (5.2.3.2 step 4), A-SPX on (A'', B''), (D'', E''), (F'', G''), C''
    (Table 8) and four acpl_data_1ch() (Pseudocode 2).

    The other codec modes (SCPL, ASPX_SCPL: S-CPL; ASPX_ACPL_1: the
    residual tracks after the A-SPX data; ASPX_AJCC: A-JCC) and b_5fronts
    1 (9.X.4) are concealed (element.parse: not built).

    Reading: 5.2.3.3 (ASPX_ACPL_2) assigns the tracks "as in step 2" of
    5.2.3.2 and does not repeat steps 3 / 4, but the bitstream carries
    b_use_sap_add_ch and its chparam_info for every 7CH_STATIC mode; they
    are applied as in 5.2.3.2 (with b_use_sap_add_ch 0, as on the public
    test stream, both readings agree)."""
    from .acpl_parse import acpl_config_1ch, PARTIAL, FULL
    b_lfe, b_5fronts = CH_MODE_ICE[ch_mode]
    b, asz, a0 = _head(sub)
    mode = 4 if b.u(1) else b.u(2)               # Table 73
    if b_5fronts:
        raise Damaged("element.parse", "9.X.4 (b_5fronts) not implemented")
    if mode != 3:
        raise Damaged("element.parse",
                      f"immersive_codec_mode {ICE_MODES[mode]} "
                      "not implemented")
    aspx = acpl = None
    if b_iframe:                                 # immers_cfg (6.2.4.2)
        aspx = C.aspx_config(b)
        acpl = acpl_config_1ch(b, PARTIAL if mode == 2 else FULL)
    out = dict(aspx=aspx, acpl=acpl, compand=None, codec_mode=mode)
    order = []
    if b_lfe:
        out["lfe"] = C.mono_data(b, T, b_lfe=True)
        order.append("lfe")
    grouping = b.u(2)                            # core_5ch_grouping
    steps = []
    mono_c = False
    if grouping == 0:
        mode2ch = b.u(1)
        k1, k2 = (("A", "B"), ("D", "E")) if mode2ch == 0 else \
            (("A", "D"), ("B", "E"))
        for kk in (k1, k2):
            x, y, st = C.two_channel_data(b, T)
            out[kk[0]], out[kk[1]] = x, y
            steps.append((kk, MC.two_channel_step(st)))
            order += list(kk)
        mono_c = True
    elif grouping == 1:
        tr, s3 = MC.three_channel_data(b, T)
        out.update(A=tr[0], B=tr[1], C=tr[2])
        steps.append((("A", "B", "C"), s3))
        x, y, st = C.two_channel_data(b, T)
        out.update(D=x, E=y)
        steps.append((("D", "E"), MC.two_channel_step(st)))
        order += ["A", "B", "C", "D", "E"]
    elif grouping == 2:
        tr, s4 = MC.four_channel_data(b, T)
        out.update(A=tr[0], B=tr[1], D=tr[2], E=tr[3])
        steps.append((("A", "B", "D", "E"), s4))
        order += ["A", "B", "D", "E"]
        mono_c = True
    else:
        tr, s5 = MC.five_channel_data(b, T)
        out.update(A=tr[0], B=tr[1], C=tr[2], D=tr[3], E=tr[4])
        steps.append((("A", "B", "C", "D", "E"), s5))
        order += ["A", "B", "C", "D", "E"]
    if mono_c:
        out["C"] = C.mono_data(b, T, b_lfe=False)
        order.append("C")
    # 7CH_STATIC: the additional tracks F, G
    fin = [(("D", "F"), ("Ls", "Tfl")), (("E", "G"), ("Rs", "Tfr"))]
    sap = [None, None]
    use_sap = b.u(1)                             # b_use_sap_add_ch
    if use_sap:
        sap = [MC.read_chparam(b, out[base]["framing"], T)
               for (base, _a), _o in fin]
    x, y, st = C.two_channel_data(b, T)
    out.update(F=x, G=y)
    steps.append((("F", "G"), MC.two_channel_step(st)))
    order += ["F", "G"]
    ren = dict(ICE_TRACK, lfe="lfe")
    side = []
    for i, ((base, add), (o0, o1)) in enumerate(fin):
        if sap[i] is not None:
            fb, fa = out[base]["framing"], out[add]["framing"]
            if (fb.num_windows != fa.num_windows
                    or [fb.length_g(fb.w2g[w]) for w in range(fb.num_windows)]
                    != [fa.length_g(fa.w2g[w])
                        for w in range(fa.num_windows)]):
                raise Damaged("asf.add_ch", "framings differ")
            side.append(((o0, o1), sap[i], fb))
    res = {ren[k]: v for k, v in out.items() if k in ren and k != "lfe"}
    out = {k: v for k, v in out.items()
           if k not in ("A", "B", "C", "D", "E", "F", "G")}
    out.update(res)
    out["steps"] = [(tuple(ren[k] for k in kk), s) for kk, s in steps]
    out["side_steps"] = side
    out["order"] = tuple(ren[k] for k in order)
    out["use_sap_add_ch"] = use_sap
    out["core_5ch_grouping"] = grouping
    out["aspx_blocks"] = ((0, True), (1, True), (3, True), (2, False))
    out["aspx_pairs"] = dict(ICE_PAIRS)
    out["acpl1"] = "ice"
    out["acpl_blocks"] = 4
    out.update(bits=b.p, nbits=len(sub) * 8, audio_size=asz,
               audio_start=a0, mode=("ice", mode))
    return out


# ---------------------------------------------------------------- A-JOC
# The downmix signals of a var_channel_element (TS 103 190-2 6.2.4.4) as
# the decoder's track slots, in bitstream order: the pairs on (L, R), (Ls,
# Rs), (Tfl, Tfr) (A-SPX block groups 0, 1, 3), an odd last signal on C
# (group 2). Up to seven full-band signals.
VAR_SLOTS = {1: ("C",), 2: ("L", "R"), 3: ("L", "R", "C"),
             4: ("L", "R", "Ls", "Rs"), 5: ("L", "R", "Ls", "Rs", "C"),
             6: ("L", "R", "Ls", "Rs", "Tfl", "Tfr"),
             7: ("L", "R", "Ls", "Rs", "Tfl", "Tfr", "C")}
VAR_PAIR_GROUPS = (0, 1, 3)


def ajoc_slots(info):
    """The decoder track slots of an A-JOC substream's downmix (core
    decoding): ("lfe" first when b_lfe) + VAR_SLOTS, or the 5_X
    element's for b_static_dmx."""
    lfe = ("lfe",) if info["b_lfe"] else ()
    if info["static"]:
        return ("L", "R", "C") + lfe + ("Ls", "Rs")
    return lfe + VAR_SLOTS[info["n_dmx"]]


def var_channel_element(b, T, b_iframe, n, b_lfe, out):
    """6.2.4.4 into out (the element dict): tracks on VAR_SLOTS[n], the
    processing steps, the A-SPX blocks (pairs then the odd signal) and
    companding_control's channel order."""
    slots = VAR_SLOTS[n]
    mode = b.u(1)                                   # var_codec_mode
    if mode == 1:
        if b_iframe:
            out["aspx"] = C.aspx_config(b)
        if n <= 5:
            out["compand"] = C.companding_control(b, n)
    order = []
    if b_lfe:
        out["lfe"] = C.mono_data(b, T, b_lfe=True)
        order.append("lfe")
    steps = []
    n_pairs = n // 2

    def pair(k0, k1):
        x, y, st = C.two_channel_data(b, T)
        out[k0], out[k1] = x, y
        steps.append(((k0, k1), MC.two_channel_step(st)))
        order.extend((k0, k1))

    if n % 2:
        if n == 1:
            out["C"] = C.mono_data(b, T, b_lfe=False)
            order.append("C")
        else:
            for p in range(n_pairs - 1):
                pair(slots[2 * p], slots[2 * p + 1])
            k0, k1, k2 = slots[2 * n_pairs - 2], slots[2 * n_pairs - 1], \
                slots[-1]
            if b.u(1) == 0:                         # var_coding_config
                pair(k0, k1)
                out[k2] = C.mono_data(b, T, b_lfe=False)
                order.append(k2)
            else:
                tr, s3 = MC.three_channel_data(b, T)
                out.update({k0: tr[0], k1: tr[1], k2: tr[2]})
                steps.append(((k0, k1, k2), s3))
                order += [k0, k1, k2]
    else:
        for p in range(n_pairs):
            pair(slots[2 * p], slots[2 * p + 1])
    blocks = []
    cidx = {}
    if mode == 1:
        for p in range(n_pairs):
            g = VAR_PAIR_GROUPS[p]
            blocks.append((g, True))
            cidx[g] = (2 * p, 2 * p + 1)
        if n % 2:
            blocks.append((2, False))
            cidx[2] = (n - 1, n - 1)
    out.update(steps=steps, order=tuple(order), aspx_blocks=tuple(blocks),
               compand_idx=cidx, codec_mode=mode)
    return mode


def ajoc_element(sub, T, info, b_iframe=False):
    """ac4_substream of an A-JOC substream (6.2.2.2, audio_data_ajoc
    6.2.3.4), up to its A-SPX blocks; out["tail"] parses the rest (after
    the A-SPX data): the downmix's OAMD (timing, oamd_dyndata_single, the
    extension), ajoc() side information, ajoc_dmx_de_data, the upmix's
    timing and oamd_dyndata_single.

    b_static_dmx: the downmix is a 5.0 / 5.1 audio_data_chan (5_X
    element); otherwise a var_channel_element of n_dmx signals (at most
    seven here; more are concealed, element.parse)."""
    if info["n_upmix"] > 64:
        raise Damaged("element.parse", "A-JOC with more than 64 upmix "
                      "objects not built")
    if info["static"]:
        out = C.decode_element(sub, T, b_iframe=b_iframe,
                               has_lfe=bool(info["b_lfe"]))
        out["mode"] = ("ajoc", "static") + tuple(out["mode"][1:])
    else:
        n = info["n_dmx"]
        if n not in VAR_SLOTS:
            raise Damaged("element.parse", f"A-JOC with {n} downmix "
                          "signals not built")
        b, asz, a0 = _head(sub)
        b_inactive = b.u(1)                         # b_some_signals_inactive
        if b_inactive:
            b.u(n)                                  # dmx_active_signals_mask
        out = dict(aspx=None, compand=None)
        mode = var_channel_element(b, T, b_iframe, n, info["b_lfe"], out)
        out.update(bits=b.p, nbits=len(sub) * 8, audio_size=asz,
                   audio_start=a0, mode=("ajoc", mode), acpl_blocks=0)
    out["ajoc_info"] = info
    out["tail"] = "ajoc"
    return out


def ajoc_tail(b, info, ifr, st):
    """The rest of audio_data_ajoc after the downmix's A-SPX blocks ->
    dict(dmx=[ObjState] (the downmix objects' metadata, core decoding),
    side=ajoc(), de=ajoc_dmx_de_data, umx=[ObjState]). st holds the
    objects' states and the A-JOC state between frames (Ac4Stream)."""
    from . import ajoc as AJ
    from . import oamd as O
    lfe = bool(info["b_lfe"])
    n_dmx = info["n_dmx"] + (1 if lfe else 0)
    n_umx = info["n_upmix"] + (1 if lfe else 0)
    dmx_t = (["lfe"] if lfe else []) + list(info["dmx_types"])
    umx_t = (["lfe"] if lfe else []) + list(info["umx_types"])

    def types(ts):
        return ([O.BED if t not in ("DYN", "ISF") else t for t in ts],
                [1 if t == "lfe" else 0 for t in ts])

    out = {}
    dmx_s = st.setdefault("dmx", [O.ObjState() for _ in range(n_dmx)])
    umx_s = st.setdefault("umx", [O.ObjState() for _ in range(n_umx)])
    nb = st.get("blocks", 1)
    if not info["static"]:
        if b.u(1):                                  # b_dmx_timing
            out["dmx_timing"] = O.oamd_timing_data(b)
            nb = len(out["dmx_timing"]["blocks"])
        ot, lf = types(dmx_t)
        O.oamd_dyndata_single(b, dmx_s, nb, ifr, 0, ot, lf)
        if b.u(1):                                  # b_oamd_extension
            bits = (b.vb(3) + 1) * 8
            p0 = b.p
            if b.u(1):                              # ajoc_bed_info
                b.u(3)
            rest = bits - (b.p - p0)
            if rest < 0:
                raise Damaged("element.parse", "oamd extension overrun")
            b.p += rest
    out["side"] = AJ.ajoc(b, info["n_dmx"], info["n_upmix"])
    out["de"] = AJ.dmx_de_data(b, info["n_dmx"], info["n_upmix"],
                               st.setdefault("de", {}))
    if b.u(1):                                      # b_umx_timing
        out["umx_timing"] = O.oamd_timing_data(b)
        nb = len(out["umx_timing"]["blocks"])
    else:
        b.u(1)                                      # b_derive_timing_from_dmx
    ot, lf = types(umx_t)
    O.oamd_dyndata_single(b, umx_s, nb, ifr, 0, ot, lf)
    st["blocks"] = nb
    out["dmx"] = [s.copy() for s in dmx_s]
    out["umx"] = [s.copy() for s in umx_s]
    return out
