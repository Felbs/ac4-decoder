# SPDX-License-Identifier: Apache-2.0
"""Survey AC-4 streams: what the bitstream uses, frame by frame.

    python3 tools/survey.py [--out DIR] [--frames N] FILE ...

For each input (any container ac4dec reads) it parses every frame's table
of contents, the presentation substreams and every channel-coded audio
substream, and reports:

  * bitstream_version, fs_index, frame_rate_index (and frame_rate_factor),
    I-frame spacing (b_iframe_global and the substreams' b_audio_ndot);
  * presentations: config, presentation_version, md_compat, id, enabled,
    substream groups (channel- or object-coded, A-JOC, OAMD substream),
    content classifier and language;
  * per audio substream: channel_mode, the channel element's codec mode
    and coding_config, companding, stereo processing (sap_mode), short
    blocks, spectral noise fill, A-SPX (TNA modes, additional harmonics,
    FIC, TIC), A-CPL, and whether this decoder can decode it;
  * metadata: dialnorm, further loudness information, DRC (modes, profile,
    compression curves or gains), dialogue enhancement (method, channels),
    substream loudness, dialogue flags, EMDF payload ids;
  * parse problems (counted by message).

Writes one JSON per input and summary.txt to --out (default
~/.cache/ac4-cport/survey); prints the summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from ac4 import aspx_parse as P                            # noqa: E402
from ac4 import channels as C                              # noqa: E402
from ac4 import container                                  # noqa: E402
from ac4 import decoder as DEC                            # noqa: E402
from ac4 import metadata as MD                             # noqa: E402
from ac4 import rates as RT                                # noqa: E402
from ac4 import toc as M                                   # noqa: E402
from ac4.frameinfo import FrameWalker                      # noqa: E402
from ac4.toc_v1 import Bits, CH_MODE_NAME                  # noqa: E402

# Table 83: internal frame length at 48 kHz per frame_rate_index
FRAME_LEN = {0: 1920, 1: 1920, 2: 2048, 3: 1536, 4: 1536, 5: 960, 6: 960,
             7: 1024, 8: 768, 9: 768, 10: 512, 11: 384, 12: 384, 13: 2048}
FPS = {0: "23.976", 1: "24", 2: "25", 3: "29.97", 4: "30", 5: "47.95",
       6: "48", 7: "50", 8: "59.94", 9: "60", 10: "100", 11: "119.88",
       12: "120", 13: "23.44"}
X5_MODE = ("SIMPLE", "ASPX", "ASPX_ACPL_1", "ASPX_ACPL_2", "ASPX_ACPL_3")
PAIR_MODE = ("SIMPLE", "ASPX", "ASPX_ACPL_1", "ASPX_ACPL_2")


def ch_name(cm):
    names = dict(CH_MODE_NAME)
    names.update({11: "7.0.4", 12: "7.1.4", 13: "9.0.4", 14: "9.1.4",
                  15: "22.2"})
    return names.get(cm, f"ch_mode {cm}")


def element_header(sub, cm, ifr, T):
    """The channel element's codec mode and coding_config, read from the
    start of audio_data -> dict (element, codec_mode, coding_config,
    compand)."""
    b = Bits(sub)
    C.audio_size(b)
    if cm == 0:
        m = b.u(1)
        return dict(element="single_channel", codec_mode=("SIMPLE", "ASPX")[m])
    if cm == 1:
        m = b.u(2)
        return dict(element="channel_pair", codec_mode=PAIR_MODE[m])
    if cm == 2:
        m = b.u(1)
        out = dict(element="3_0", codec_mode=("SIMPLE", "ASPX")[m])
        if m:
            if ifr:
                C.aspx_config(b)
            out["compand"] = C.companding_control(b, 3)
        out["coding_config"] = b.u(1)
        return out
    if cm in (3, 4):
        m = b.u(3)
        out = dict(element="5_X", codec_mode=X5_MODE[m] if m < 5
                   else f"reserved {m}")
        if m in (2, 3):
            if ifr:
                C.aspx_config(b)
                b.u(2)
                b.u(1)
                if m == 2:
                    b.u(3)                       # acpl_qmf_band_minus1
            if cm == 4:
                C.mono_data(b, T, b_lfe=True)
            out["compand"] = C.companding_control(b, 3)
            out["coding_config"] = b.u(1)
        return out
    if 5 <= cm <= 10:
        m = b.u(2)
        return dict(element="7_X", codec_mode=("SIMPLE", "ASPX",
                                                "ASPX_ACPL_1",
                                                "ASPX_ACPL_2")[m])
    if 11 <= cm <= 14:
        from ac4 import elements as EL
        m = 4 if b.u(1) else b.u(2)
        return dict(element="immersive_channel",
                    codec_mode=EL.ICE_MODES[m])
    if cm == 15:
        return dict(element="22_2")
    return dict(element=f"ch_mode {cm}")


class FileSurvey:
    def __init__(self):
        self.c = collections.Counter()
        self.errors = collections.Counter()
        self.toc = collections.Counter()
        self.pres = collections.Counter()
        self.subs = {}            # substream index -> Counter
        self.meta = collections.Counter()
        self.drc = collections.Counter()
        self.de = collections.Counter()
        self.emdf = collections.Counter()
        self.loud = {}
        self.iframes = []
        self.ndot = []

    def sub(self, k):
        return self.subs.setdefault(k, collections.Counter())


def survey_file(path, max_frames=None):
    src = container.read(path)
    fr = src.frames[:max_frames] if max_frames else src.frames
    S = FileSurvey()
    T = C.load_tables()
    walkers = {}
    drc_state = {}
    meta_state = {}
    for fi, f in enumerate(fr):
        if not f:
            S.c["empty frames"] += 1
            continue
        try:
            st = M.parse(f)
        except Exception as e:                                 # noqa: BLE001
            S.errors[f"toc: {str(e)[:50]}"] += 1
            continue
        S.c["frames"] += 1
        fri = st["frame_rate_index"]
        S.toc[(st["bitstream_version"], st["fs_index"], fri,
               st.get("frame_rate_factor", 1), st["n_presentations"],
               st["n_substreams"], st.get("payload_base", 0) > 0)] += 1
        if st["b_iframe_global"]:
            S.iframes.append(fi)
        # the frame length the decoder uses (Ac4Stream._set_rate): the
        # element parse and the A-SPX time grid both depend on it
        flen = (RT.frame_len_base(st["fs_index"], fri)
                or FRAME_LEN.get(fri, 1536))
        Tf = T if flen == 1536 else dict(T, fl=flen)
        done = set()
        for pi, p in enumerate(st["presentations"]):
            subs = M.presentation_substreams(st, pi)
            groups = []
            for gi in p["groups"]:
                if gi >= len(st["groups"]):
                    groups.append("missing")
                    continue
                g = st["groups"][gi]
                kinds = ",".join(
                    (ch_name(s["channel_mode"]) if s["kind"] == "chan"
                     else s["kind"]) for s in g["substreams"])
                cc = (M.CONTENT_CLASS[g["content_classifier"]]
                      if g["content_classifier"] is not None else "-")
                groups.append(f"{kinds}/{cc}/{g['language'] or '-'}"
                              + ("/oamd" if g["oamd"] else "")
                              + ("/hsf" if g["hsf_ext"] else ""))
            S.pres[(pi, p["config"], p["version"], p["md_compat"],
                    p["presentation_id"], p["enabled"],
                    p["pre_virtualized"], p["b_alternative"],
                    " + ".join(groups))] += 1
            # the presentation substream (dialnorm, loudness, DRC)
            ps = p["pres_substream"]
            if ps is not None and ps < len(st["substream_sizes"]) \
                    and ("p", ps) not in done:
                done.add(("p", ps))
                cms = [s["channel_mode"] for _, _, s in subs
                       if s["kind"] == "chan"]
                try:
                    pm = MD.presentation_substream(
                        M.substream_bytes(f, st, ps), p["b_alternative"],
                        p["b_pres_ndot"], len(subs),
                        drc_state.setdefault(ps, MD.DrcState()),
                        cms[0] if cms else 1, flen)
                    S.meta[f"dialnorm {pm['dialnorm']:+.2f} dBFS"] += 1
                    for k, v in (pm.get("loudness") or {}).items():
                        S.loud.setdefault(k, set()).add(v)
                    if pm.get("drc_error"):
                        S.errors[f"drc: {pm['drc_error']}"] += 1
                    d = pm.get("drc")
                    if d is None:
                        S.drc["no drc_frame"] += 1
                    else:
                        for m in d["config"]["modes"]:
                            kind = ("default profile" if m.get(
                                "default_profile") else "curve" if
                                m["curve_flag"] else
                                f"gains cfg {m.get('gains_config')}")
                            S.drc[f"mode {m['id']} "
                                  f"({MD.DRC_MODE_NAME.get(m['id'], '?')})"
                                  f": {kind}"] += 1
                        S.drc["eac3_profile "
                              + MD.DRC_EAC3_PROFILE[
                                  d["config"]["eac3_profile"]]] += 1
                        for mid, g in d["gains"].items():
                            flat = [x for ch in g for sf in ch for x in sf]
                            S.drc[f"mode {mid} gains min/max"] = 0
                            S.loud.setdefault(f"drc gains mode {mid}",
                                              set()).update(flat)
                except Exception as e:                         # noqa: BLE001
                    S.errors[f"pres substream: {str(e)[:50]}"] += 1
            # every audio substream
            for gi, g, s in subs:
                k = s.get("index")
                if k is None or ("a", k) in done:
                    continue
                done.add(("a", k))
                c = S.sub(k)
                if s["kind"] != "chan":
                    c[f"kind {s['kind']}"] += 1
                    continue
                cm = s["channel_mode"]
                c[f"channel_mode {ch_name(cm)}"] += 1
                if (cm in DEC.IMS_CH_MODES and p["version"] == 2):
                    # immersive stereo: the substream is a channel pair
                    # and its metadata() is read with the stereo channel
                    # mode, as the decoder does (README, Interpretations)
                    c["IMS: read as channel_pair_element"] += 1
                    cm = DEC.IMS_STEREO
                if s.get("sf_multiplier") is not None:
                    c[f"sf_multiplier {s['sf_multiplier']}"] += 1
                ifr = bool(s["ndot"][0])
                if ifr:
                    c["b_audio_ndot frames"] += 1
                if k < len(st["substream_sizes"]):
                    sub = M.substream_bytes(f, st, k)
                else:
                    S.errors["substream index outside the table"] += 1
                    continue
                try:
                    h = element_header(sub, cm, ifr, Tf)
                    c[f"{h['element']} {h.get('codec_mode', '')}".strip()] += 1
                    if "coding_config" in h:
                        c[f"coding_config {h['coding_config']}"] += 1
                except Exception as e:                         # noqa: BLE001
                    S.errors[f"element header: {str(e)[:40]}"] += 1
                    continue
                # a full walk where this decoder has the element
                el = ("pair" if cm == 1 else "5_X" if cm == 4 else None)
                if el and h.get("codec_mode") in (PAIR_MODE if el == "pair"
                                                  else X5_MODE):
                    w = walkers.get((k, flen))
                    if w is None:
                        w = walkers[(k, flen)] = FrameWalker(el, flen)
                    P.STATS = {}
                    r = w.walk_sub(sub, ifr)
                    for kk, v in P.STATS.items():
                        c[f"aspx {kk} (channel-blocks)"] += v
                    P.STATS = None
                    c[f"walk {r['status']}"] += 1
                    if r.get("error"):
                        S.errors[f"walk: {r['error'][:50]}"] += 1
                    elif r["status"] == "closed" and r["slack"] < 0:
                        # read past audio_size: the walk is wrong somewhere
                        S.errors["walk: read past audio_size"] += 1
                    for sm in r.get("sap") or []:
                        c[f"sap_mode {sm}"] += 1
                    for kk in r.get("flags", ()):
                        c[kk] += 1
                    comp = r.get("compand")
                    if comp is not None:
                        if any(comp[1]):
                            c["companding on (some channel)"] += 1
                        if comp[2] is not None:
                            c["companding avg flag sent"] += 1
                    if r.get("aspx_cfg"):
                        c["aspx_config " + ",".join(
                            f"{a}={b_}" for a, b_ in
                            sorted(r["aspx_cfg"].items()))] += 1
                elif el is None or h.get("codec_mode"):
                    c["not decodable here"] += 1
                # metadata() after audio_data
                try:
                    b = Bits(sub)
                    asz, a0 = C.audio_size(b)
                    md = MD.substream_metadata(
                        sub, a0, asz, cm, ifr,
                        meta_state.setdefault(k, MD.SubMetaState()))
                    if md.get("error"):
                        S.errors[f"metadata: {md['error'][:50]}"] += 1
                    if md["extended"].get("dialog"):
                        S.meta["b_dialog"] += 1
                    if md["extended"].get("dialog_channels"):
                        S.meta["channels classifier"] += 1
                    if "substream_loudness_bits" in md["basic"]:
                        S.meta["substream loudness"] += 1
                    for kk in md["basic"]:
                        if kk.startswith("pre_") or kk in ("loro", "ltrt"):
                            S.meta[f"basic: {kk}"] += 1
                    de = md.get("de")
                    if de is None:
                        S.de["no DE data"] += 1
                    else:
                        cf = de["config"]
                        S.de[f"method {cf['method']} max_gain "
                             f"{(cf['max_gain'] + 1) * 3} dB channels "
                             f"{cf['channel_config']:03b}"] += 1
                    for pid in md["emdf"]:
                        S.emdf[pid] += 1
                except Exception as e:                         # noqa: BLE001
                    S.errors[f"metadata: {str(e)[:50]}"] += 1
    return src, S


def summarise(name, src, S):
    def top(cnt, n=12):
        return {str(k): v for k, v in cnt.most_common(n)}
    gaps = [b - a for a, b in zip(S.iframes, S.iframes[1:])]
    toc = {}
    for (bv, fs, fri, frf, npres, nsub, pb), n in S.toc.items():
        toc[f"v{bv} fs_index {fs} ({'48' if fs else '44.1'} kHz) "
            f"fri {fri} ({FPS.get(fri, '?')} fps, frame {FRAME_LEN.get(fri)})"
            f" frf {frf} presentations {npres} substreams {nsub}"
            f"{' payload_base' if pb else ''}"] = n
    pres = {}
    for (pi, cfg, ver, mdc, pid, en, pv, alt, groups), n in S.pres.items():
        pres[f"[{pi}] config {'single' if cfg is None else cfg} version "
             f"{ver} md_compat {mdc} id {pid}"
             f"{'' if en else ' DISABLED'}{' pre-virtualized' if pv else ''}"
             f"{' alternative' if alt else ''}: {groups}"] = n
    return dict(
        file=name, kind=src.kind, frames=len(src.frames),
        container=src.info, dsi=src.dsi, counts=dict(S.c),
        toc=toc, presentations=pres,
        iframe_gap=(collections.Counter(gaps).most_common(3)),
        substreams={str(k): dict(v) for k, v in sorted(S.subs.items())},
        metadata=top(S.meta, 20), drc=top(S.drc, 20), de=top(S.de),
        emdf={str(k): v for k, v in S.emdf.items()},
        loudness={k: sorted(v)[:8] if len(v) <= 8 else
                  [min(v), max(v), f"{len(v)} values"]
                  for k, v in S.loud.items()},
        errors=top(S.errors, 20))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/.cache/ac4-cport/survey"))
    ap.add_argument("--frames", type=int, default=None,
                    help="survey only the first N frames of each file")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    lines = []
    for path in a.files:
        name = os.path.basename(os.path.dirname(path)) + "__" + \
            os.path.basename(path)
        src, S = survey_file(path, a.frames)
        res = summarise(name, src, S)
        with open(os.path.join(a.out, name + ".json"), "w") as fh:
            json.dump(res, fh, indent=1, default=str)
        lines.append(f"== {name}: {src.kind}, {res['frames']} frames")
        for k in ("toc", "presentations", "substreams", "metadata", "drc",
                  "de", "emdf", "loudness", "errors"):
            if res[k]:
                lines.append(f"  {k}: {json.dumps(res[k], default=str)}")
        lines.append(f"  iframe gaps: {res['iframe_gap']}")
        print("\n".join(lines[-12:]), flush=True)
    with open(os.path.join(a.out, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
