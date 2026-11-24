# SPDX-License-Identifier: Apache-2.0
"""Bit-by-bit writers for synthetic channel elements (tests only): the
syntax of TS 103 190-1 4.2.6 - 4.2.8 written out, independent of the
decoder's parser. Long blocks at frame length 1536, one section of
codebook 11 per track, every scale factor 100 (so a line q dequantises to
sign(q) |q|^(4/3))."""
import numpy as np

from ac4 import channels as CH
from test_phase_a import BW, build_toc

SFB = CH.SFB_TABLE[1536]
# the reference scale factor every Track writes (100: lines dequantise to
# sign(q) |q|^(4/3)); the fuzz streams raise it to a realistic level
SF_REF = 100


def huff_put(w, book, sym):
    for (n, code), s in book.map.items():
        if s == sym:
            w.u(n, code)
            return
    raise KeyError(sym)


class Track:
    """One sf_data(): nbands scale factor bands of random lines."""

    def __init__(self, rng, nbands, zero=False):
        self.nbands = nbands
        n = SFB[nbands] if nbands else 0
        self.q = np.zeros(1536, int)
        if not zero and n:
            self.q[:n] = rng.integers(-5, 6, n)

    def deq(self):
        return np.sign(self.q) * np.abs(self.q) ** (4.0 / 3.0)

    def write(self, w, T):
        if self.nbands:
            w.u(4, 11).u(5, self.nbands - 1)        # one section, book 11
            for i in range(0, SFB[self.nbands], 2):
                a, b = int(self.q[i]), int(self.q[i + 1])
                huff_put(w, T["cb"][11], abs(a) * 17 + abs(b))
                for v in (a, b):
                    if v:
                        w.u(1, 1 if v < 0 else 0)  # sign
        w.u(8, SF_REF)                              # reference sf
        for _ in range(max(self.nbands - 1, 0)):
            huff_put(w, T["sf"], 60)                # unchanged
        w.u(1, 0)                                   # no noise fill


def sf_info_long(w, max_sfb):
    w.u(1, 1).u(6, max_sfb)                         # b_long_frame, max_sfb


class St:
    """chparam_info: sap_mode 0 (none), 1 (ms_used flags) or 2 (M/S in
    every band). abcd(sfb) -> the (a, b, c, d) of clause 5.3.2."""

    def __init__(self, rng, mode, max_sfb):
        self.mode = mode
        self.used = (rng.integers(0, 2, max_sfb) if mode == 1
                     else np.zeros(max_sfb, int))

    def write(self, w):
        w.u(2, self.mode)
        if self.mode == 1:
            for v in self.used:
                w.u(1, int(v))

    def abcd(self, sfb):
        if self.mode == 2 or (self.mode == 1 and sfb < len(self.used)
                              and self.used[sfb]):
            return (1.0, 1.0, 1.0, -1.0)
        return (1.0, 0.0, 0.0, 1.0)


def mono(w, T, track):
    w.u(1, 0)                                       # spec_frontend ASF
    sf_info_long(w, track.nbands)
    track.write(w, T)


def lfe(w, T, track):
    w.u(3, track.nbands)                            # n_msfbl_bits 3
    track.write(w, T)


def two_channel(w, T, tracks, st):
    """two_channel_data with MDCT stereo processing (one sf_info)."""
    w.u(1, 1)
    sf_info_long(w, max(t.nbands for t in tracks))
    st.write(w)
    for t in tracks:
        t.write(w, T)


def n_channel(w, T, tracks, sts, matsel=None):
    """three / four / five_channel_data."""
    sf_info_long(w, max(t.nbands for t in tracks))
    if matsel is not None:
        w.u(4, matsel)
    for s in sts:
        s.write(w)
    for t in tracks:
        t.write(w, T)


def element(body_bits):
    """audio_size + byte-aligned body."""
    body = body_bits.align().bytes()
    return BW().u(15, len(body)).u(1, 0).bytes() + body


def frame(sub, seq=5, ch_mode_code="1110", fri=3, iframe=1, top=None):
    if top is None and ch_mode_code in ("11111100", "11111101",
                                        "111111100", "111111101"):
        top = (1, 1, 3)
    t = dict(pres=[dict(groups=[0], pres_sub=0)],
             groups=[dict(ch_mode_code=ch_mode_code, index=1, ndot=iframe,
                          add_ch_base=(0 if ch_mode_code in (
                              "1111010", "1111011", "1111100", "1111101")
                              else None), top=top)],
             sizes=[2, len(sub)])
    return build_toc(**t, seq=seq, fri=fri, iframe=iframe) + bytes(2) + sub
