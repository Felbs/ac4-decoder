# SPDX-License-Identifier: Apache-2.0
"""Walk each frame's audio_data() to its end and check it against audio_size.

TS 103 190-1 clause 4.2.4.2 (Table 16):

    ac4_substream() {
        audio_size_value (15); b_more_bits (1) [+ variable_bits(7) << 15]
        byte_align
        audio_data(channel_mode, b_iframe)
        fill_bits
        byte_align
        metadata(b_iframe) ...
    }

and clause 4.3.4.1: audio_size is the size in bytes of audio_data()
"including any following fill_bits and byte_align bits". So after a correct
parse of every element of audio_data(), the reader must stand at most a few
bits short of byte offset audio_size counted from the aligned start, and
never past it. `slack` below is that gap in bits: 0..7 is byte_align only,
more means fill_bits (or an under-read), negative means an over-read.

This is a check on the PARSE (every syntax element of the frame accounted
for, A-SPX and A-CPL side data included), independent of the PCM path: it
keeps its own A-SPX / A-CPL configuration state and never changes output.
"""
from __future__ import annotations

from . import acpl_parse as AP
from . import aspx_bands as A
from . import aspx_parse as P
from . import aspx_render as R37
from . import channels as C
from . import hcb as H
from . import spectral as S
from . import toc as M
from .toc import Bits


class FrameWalker:
    def __init__(self, element):
        if element not in ("5_X", "pair"):
            raise ValueError(element)
        self.element = element
        self.T = C.load_tables()
        arrays = H.parse_c(H.DEFAULT_C)
        self.tabs = {n[:-4]: S.Huff(arrays[n], arrays[n[:-4] + "_CW"])
                     for n in arrays
                     if n.endswith("_LEN") and n.startswith("ASPX_")}
        for nm in self.tabs:
            P.CENTRE[nm] = 0 if nm.endswith("_F0") else \
                (len(arrays[nm + "_LEN"]) - 1) // 2
        self.acpl_books = AP.load_books(arrays)
        self.nats = A.num_aspx_timeslots()
        self.cfg = None              # A-SPX config (i-frames only)
        self.acpl_cfg = None         # A-CPL config (i-frames only)
        self.fstate = {}

    def walk(self, frame):
        """-> dict(status, sap=[sap_mode per pair], slack, ...).

        status "closed": audio_data parsed to its end, slack in bits.
        status "no_config": an A-SPX/A-CPL frame before the first i-frame
        (its side data cannot be parsed without the configuration).
        status "error": the parse raised (message in `error`)."""
        try:
            st = M.parse(frame)
            sel = M.audio_substream(st, M.choose(st))
            if sel is None:
                raise ValueError("no decodable audio substream")
            sub = M.substream_bytes(frame, st, sel["index"])
            ifr = bool(sel["ndot"][0])
        except Exception as e:                                 # noqa: BLE001
            return dict(status="error", sap=[], iframe=None, slack=None,
                        error=f"{type(e).__name__}: {str(e)[:80]}")
        return self.walk_sub(sub, ifr)

    def walk_sub(self, sub, ifr):
        """walk() on one audio substream; also reports compand (the
        companding_control result), flags (short blocks, noise fill) and
        aspx_cfg (this frame's aspx_config)."""
        out = dict(status="error", sap=[], iframe=ifr, slack=None,
                   flags=[])
        try:
            b = Bits(sub)
            size = b.u(15)
            if b.u(1):
                size += b.vb(7) << 15
            b.align()
            start = b.p
            out["audio_size"] = size
            if self.element == "pair":
                from . import pair as PAIR
                r = PAIR.decode_pair(sub, self.T, b_iframe=ifr)
            else:
                r = C.decode_element(sub, self.T, b_iframe=ifr)
            if r.get("steps") is not None:
                pairs = [st for _k, step in r["steps"]
                         for st in step["st"]]
            else:
                pairs = [r["st_lr"]] + ([r["st_sr"]] if "st_sr" in r
                                        else [])
            out["sap"] = [p["sap_mode"] for p in pairs]
            out["compand"] = r.get("compand")
            out["aspx_cfg"] = r.get("aspx")
            for k in ("L", "R", "C", "Ls", "Rs", "lfe"):
                ch = r.get(k)
                if not ch:
                    continue
                if not ch["framing"].long:
                    out["flags"].append(f"short blocks ({k})")
                if ch.get("snf") and any(any(v is not None for v in g)
                                         for g in ch["snf"] if g):
                    out["flags"].append(f"noise fill ({k})")
            out["stereo"] = pairs
            out["codec_mode"] = r.get("codec_mode")
            if r.get("aspx"):
                self.cfg = r["aspx"]
            if r.get("acpl_cfg"):
                self.acpl_cfg = r["acpl_cfg"]
            b.p = r["bits"]
            mode = r.get("codec_mode")
            blocks = list(r.get("aspx_blocks", ()))
            if blocks and self.cfg is None:
                out["status"] = "no_config"
                return out
            t = R37.t_for(self.cfg) if blocks else None
            for grp, two in blocks:
                P.aspx_data(b, self.tabs, t, self.cfg, self.nats, ifr, two,
                            self.fstate, grp)
            if mode == 4 and self.element != "pair":
                if self.acpl_cfg is None:
                    out["status"] = "no_config"
                    return out
                AP.acpl_data_2ch(b, self.acpl_books, self.acpl_cfg)
            out["end"] = b.p - start
            out["slack"] = size * 8 - (b.p - start)
            out["status"] = "closed"
        except Exception as e:                                 # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
        return out


def walk(frames, element):
    w = FrameWalker(element)
    return [w.walk(f) for f in frames]
