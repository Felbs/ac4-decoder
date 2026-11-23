# SPDX-License-Identifier: Apache-2.0
"""ac4dec -- decode AC-4 audio to WAV.

    ac4dec IN OUT.wav [--element auto|5_X|pair] [--channels 2|6]
                      [--presentation N|id:N] [--language xx]
                      [--seconds S] [--max-pass N]
                      [--target-level L] [--drc auto|off|N] [--ref-peak X]
                      [--no-guard]
    ac4dec IN --list        list the presentations and exit

IN is detected from its content: MP4 (fragmented: init segment + moof/mdat
fragments, or bare fragments; or non-fragmented with a sample table), a
stream of AC-4 sync frames (.ac4, TS 103 190-1 Annex G; the CRC of 0xAC41
frames is checked, a frame that fails it is concealed), or an MPEG-2
transport stream carrying AC-4 (ETSI TS 101 154).

PRESENTATIONS. By default the first enabled presentation with a
channel-coded audio substream is decoded, preferring --language, then
complete main content (TS 103 190-2 4.8.2). --presentation N takes the
N-th (0-based) of the table of contents, id:N the one whose
presentation_id is N.

DAMAGED INPUT. Frames that break a rule of the bitstream are concealed
(faded to silence, exact length kept); the summary line counts them as
"concealed", with the reasons above it.

OUT is 16-bit PCM at 48 kHz (44,1 kHz for a 44,1 kHz stream), every
frame rate of TS 103 190-1 Table 83 resampled as the table says: 6 channels in WAV 5.1 order (L R C LFE Ls Rs)
for a 5.1 element, 2 for stereo, 1 for mono, L R C for 3.0, L R C Ls Rs for 5.0,
and L R C LFE Ls Rs plus the layout's pair (Lb Rb, Lw Rw or Tfl Tfr) for 7.x.

OUTPUT LEVEL. Loudness normalisation (TS 103 190-1 5.7.9): the stream's
reference level (dialnorm) is brought to --target-level (default -24 dB,
the "flat panel TV" output range), and the DRC decoder mode the stream
configures for that level is applied (--drc auto; off; or a mode ID 0-7).
Full scale is IMDCT_K * 2^14.5 decoder units (see README); samples beyond
it are clipped and counted. --ref-peak X restores the old receiver rule
instead (samples x 0.9 / ref_peak, ref_peak rising with the loudest
sample, capped at 2 X).
"""
from __future__ import annotations

import argparse
import sys
import time
import wave

import numpy as np

OUT_FS = 48000
FPS = 30000.0 / 1001.0


class LevelPolicy:
    """The receiver's output level rule (see the module docstring)."""

    def __init__(self, element, ref_peak=0.0):
        from .decoder import REF_PEAK
        start = ref_peak or REF_PEAK[element]
        self.burst = 2.0 * start
        self.ref_peak = min(start, self.burst)
        self.notes = []

    def apply(self, y):
        pk = float(np.abs(y).max())
        if pk > self.burst:
            n_over = int((np.abs(y) > self.burst).any(axis=1).sum())
            self.notes.append(f"{n_over} samples up to {pk:.4g} clipped to "
                              f"{self.ref_peak:.4g}")
            y = np.clip(y, -self.ref_peak, self.ref_peak)
            pk = float(np.abs(y).max())
        if pk > self.ref_peak:
            self.notes.append(f"peak {pk:.4g} above reference "
                              f"{self.ref_peak:.4g}: gain lowered "
                              f"{20 * np.log10(pk / self.ref_peak):.1f} dB")
            self.ref_peak = pk
        return y * (0.9 / self.ref_peak)


def to_int16(pcm):
    """Float in [-1, 1] -> little-endian int16 bytes (truncating, as the
    receiver's writer does)."""
    pcm = np.clip(pcm, -1.0, 1.0)
    return (pcm * 32767).astype("<i2").tobytes()


def list_presentations(src, out=print):
    """Print the presentations of the first frame whose TOC parses, and
    the dac4 box's when the input has one. -> 0, or 1 when none."""
    from . import container
    from . import toc as M
    from .toc_v1 import CH_MODE_NAME
    s = container.read(src)
    out(f"input: {s.kind}, {len(s.frames)} frames")
    st = None
    for f in s.frames[:64]:
        try:
            st = M.parse(f)
            break
        except Exception:                                      # noqa: BLE001
            continue
    if st is None:
        out("no frame with a table of contents found")
        return 1
    chosen = M.choose(st)
    out(f"bitstream_version {st['bitstream_version']}, fs_index "
        f"{st['fs_index']}, frame_rate_index {st['frame_rate_index']}")
    for d in M.describe(st):
        cms = ", ".join(CH_MODE_NAME.get(c, f"ch_mode {c}")
                        for c in d["channel_modes"]) or "objects"
        out(f"  [{d['index']}]{' *' if d['index'] == chosen else '  '} "
            f"id {d['presentation_id']}, {d['config']}, "
            f"language {d['language'] or '-'}, "
            f"content {', '.join(d['content']) or '-'}, {cms}"
            f"{'' if d['enabled'] else ', DISABLED'}"
            f"{'' if d['decodable'] else ', not decodable here'}")
    if s.dsi and s.dsi.get("presentations"):
        out("dac4:")
        for i, p in enumerate(s.dsi["presentations"]):
            langs = [g["language"] for g in p.get("groups", [])
                     if g.get("language")]
            out(f"  [{i}] version {p.get('version')}, id "
                f"{p.get('presentation_id')}, ch_mode "
                f"{p.get('channel_mode')}, language "
                f"{langs[0] if langs else '-'}"
                f"{', name ' + p['name'] if p.get('name') else ''}")
    out("(* = decoded by default)")
    return 0


def decode_file(src, dst, element="auto", channels=None, seconds=None,
                max_pass=1800, ref_peak=0.0, quiet=False, guard=True,
                presentation=None, language=None, target_level=-24.0,
                drc="auto", de_level=0.0):
    from . import container
    from . import rates as RT
    from .decoder import Decoder, probe_element, probe_rate
    from .resample import StreamResampler

    def say(*a):
        if not quiet:
            print(*a, flush=True)

    source = container.read(src)
    frames = source.frames
    if source.kind in ("sync", "ts"):
        sy = source.info.get("sync", source.info)
        if sy.get("n_crc_bad") or sy.get("n_skipped"):
            say(f"  {source.kind}: {sy.get('n_crc_bad', 0)} frames failed "
                f"their CRC, {sy.get('n_skipped', 0)} bytes skipped to "
                "regain sync")
    rate = probe_rate(frames)
    if rate is None or RT.ratio(*rate) is None:
        rate = (1, 3)
    fps = RT.FPS[rate[1]]
    out_fs = RT.out_rate(rate[0])
    if seconds is not None:
        frames = frames[:max(0, int(round(seconds * fps)))]
    if not frames:
        say(f"no AC-4 frames found in {src}")
        return 1
    if element == "auto":
        element = probe_element(frames, presentation=presentation,
                                language=language)
        if element is None:
            say("cannot tell the element (5.1 or stereo) from the stream; "
                "pass --element")
            return 1
    t0 = time.time()
    legacy = bool(ref_peak)
    from .decoder import probe_ch_mode
    ch_mode = (probe_ch_mode(frames, presentation=presentation,
                             language=language) if element == "7_X" else None)
    try:
        dec = Decoder(element, channels, guard=guard,
                      presentation=presentation, language=language,
                      target_level=None if legacy else target_level, drc=drc,
                      ch_mode=ch_mode, de_level=de_level)
    except ValueError as e:
        say(str(e))
        return 1
    lvl = LevelPolicy(element, ref_peak) if legacy else None
    n_clip = 0
    rs = StreamResampler(*RT.ratio(*rate))
    n_ch = len(dec.chans)
    n_out = 0
    w = wave.open(dst, "wb")
    try:
        w.setnchannels(n_ch)
        w.setsampwidth(2)
        w.setframerate(out_fs)
        step = max(8, int(max_pass))
        for i in range(0, len(frames), step):
            y = dec.decode_pass(frames[i:i + step])
            if len(y):
                y = rs.push(lvl.apply(y) if lvl else y)
                n_clip += int((np.abs(y) > 1.0).sum())
                w.writeframes(to_int16(y))
                n_out += len(y)
    finally:
        w.close()
    dt = time.time() - t0
    for line in dec.log + (lvl.notes if lvl else []):
        say("  " + line)
    if not lvl and dec.leveler is not None:
        lv = dec.leveler
        say(f"  loudness: dialnorm "
            f"{'-' if lv.lin is None else f'{lv.lin:g}'} dB -> "
            f"{lv.target:g} dB, DRC mode "
            f"{'off' if lv.mode_id is None else lv.mode_id}"
            f"{f', {n_clip} samples clipped' if n_clip else ''}")
    secs = n_out / out_fs
    if dec.n_concealed:
        why = ", ".join(f"{k} {v}" for k, v in
                        sorted(dec.concealed.reasons.items()))
        say(f"  concealed frames by reason: {why}")
        if dec.concealed.reasons.get("toc.unsupported_rate"):
            say("  frames with frame_rate_factor or frame_rate_fraction "
                "above 1, 96/192 kHz substreams or reserved rates are not "
                "decoded; they are silent")
    say(f"{len(frames)} frames ({dec.n_bad} bad, {dec.n_concealed} "
        f"concealed, {dec.n_seq_jumps} sequence jumps), element {element}, "
        f"{n_ch} ch, {secs:.3f} s -> {dst}  "
        f"[{dt:.1f} s, {secs / dt if dt > 0 else 0:.2f}x real time; "
        f"A-SPX bursts {dec.n_burst}, A-SPX failures {dec.n_hf_fail}]")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="ac4dec", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="MP4 (.mp4/.m4s), AC-4 sync frames (.ac4) "
                    "or MPEG-TS (.ts) carrying AC-4")
    ap.add_argument("dst", nargs="?", help="output .wav")
    ap.add_argument("--list", action="store_true",
                    help="list the presentations and exit")
    ap.add_argument("--presentation", default=None,
                    help="N: the N-th presentation (0-based); id:N: the "
                         "one with presentation_id N")
    ap.add_argument("--language", default=None,
                    help="preferred language (BCP 47, e.g. fr) for the "
                         "automatic choice")
    ap.add_argument("--element", default="auto",
                    choices=("auto", "5_X", "pair", "mono", "3_0", "5_0",
                             "7_X"))
    ap.add_argument("--channels", type=int, default=None,
                    help="default: all of the element's channels (6 for "
                    "5.1, 8 for 7.1, ...); 2: L, R only")
    ap.add_argument("--seconds", type=float, default=None,
                    help="decode only the first S seconds")
    ap.add_argument("--max-pass", type=int, default=1800,
                    help="frames per decode pass (bounds memory). The output "
                         "can differ slightly with it; see README, Known "
                         "issues")
    ap.add_argument("--target-level", type=float, default=-24.0,
                    help="output loudness: dialnorm is normalised to this "
                         "level, dB (default -24)")
    ap.add_argument("--drc", default="auto",
                    help="DRC decoder mode: auto (the stream's mode for "
                         "the target level), off, or a mode ID 0-7")
    ap.add_argument("--de-level", type=float, default=0.0,
                    help="dialogue enhancement gain in dB (0 = off, the "
                         "default; capped by the stream's maximum)")
    ap.add_argument("--ref-peak", type=float, default=0.0,
                    help="use the old fixed-reference level rule instead: "
                         "decoder-scale level written at 0.9 of full scale")
    ap.add_argument("--no-guard", action="store_true",
                    help="turn off the output level guard (damaged-frame "
                         "safety net; see README)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        return list_presentations(a.src)
    if not a.dst:
        ap.error("an output .wav is needed (or --list)")
    pres = a.presentation
    if pres is not None and not pres.startswith("id:"):
        pres = int(pres)
    return decode_file(a.src, a.dst, a.element, a.channels, a.seconds,
                       a.max_pass, a.ref_peak, a.quiet,
                       guard=not a.no_guard, presentation=pres,
                       language=a.language, target_level=a.target_level,
                       drc=a.drc if a.drc in ("auto", "off") else
                       int(a.drc), de_level=a.de_level)


if __name__ == "__main__":
    sys.exit(main())
