# SPDX-License-Identifier: Apache-2.0
"""ac4dec -- decode the AC-4 audio track of a fragmented MP4 file to WAV.

    ac4dec IN.m4s OUT.wav [--element auto|5_X|pair] [--channels 2|6]
                          [--seconds S] [--max-pass N] [--ref-peak X]
                          [--no-guard]

DAMAGED INPUT. Frames that break a rule of the bitstream are concealed
(faded to silence, exact length kept); the summary line counts them as
"concealed", with the reasons above it.

IN is an init segment followed by media fragments (moof + mdat), or bare
fragments. OUT is 16-bit PCM at 48 kHz: 6 channels in WAV 5.1 order
(L R C LFE Ls Rs) for a 5.1 element, 2 for stereo.

OUTPUT LEVEL. The decoder's true full scale is not established, so the
level policy of the receiver this decoder came from is kept, unchanged:
samples are multiplied by 0.9 / ref_peak, where ref_peak starts at a fixed
per-element reference (9e5 for 5.1, 2e5 for stereo, decoder units) and only
ever rises, to the loudest sample seen so far, capped at twice the start.
Anything above the cap is clipped. --ref-peak overrides the start value.
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


def decode_file(src, dst, element="auto", channels=None, seconds=None,
                max_pass=1800, ref_peak=0.0, quiet=False, guard=True):
    from . import mp4
    from .decoder import Decoder, probe_element
    from .resample import StreamResampler

    def say(*a):
        if not quiet:
            print(*a, flush=True)

    frames = mp4.read_frames(src)
    if seconds is not None:
        frames = frames[:max(0, int(round(seconds * FPS)))]
    if not frames:
        say(f"no AC-4 frames found in {src}")
        return 1
    if element == "auto":
        element = probe_element(frames)
        if element is None:
            say("cannot tell the element (5.1 or stereo) from the stream; "
                "pass --element")
            return 1
    t0 = time.time()
    dec = Decoder(element, channels, guard=guard)
    lvl = LevelPolicy(element, ref_peak)
    rs = StreamResampler()
    n_ch = len(dec.chans)
    n_out = 0
    w = wave.open(dst, "wb")
    try:
        w.setnchannels(n_ch)
        w.setsampwidth(2)
        w.setframerate(OUT_FS)
        step = max(8, int(max_pass))
        for i in range(0, len(frames), step):
            y = dec.decode_pass(frames[i:i + step])
            if len(y):
                y = rs.push(lvl.apply(y))
                w.writeframes(to_int16(y))
                n_out += len(y)
    finally:
        w.close()
    dt = time.time() - t0
    for line in dec.log + lvl.notes:
        say("  " + line)
    secs = n_out / OUT_FS
    if dec.n_concealed:
        why = ", ".join(f"{k} {v}" for k, v in
                        sorted(dec.concealed.reasons.items()))
        say(f"  concealed frames by reason: {why}")
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
    ap.add_argument("src", help="fragmented MP4 (.mp4/.m4s) with an AC-4 track")
    ap.add_argument("dst", help="output .wav")
    ap.add_argument("--element", default="auto", choices=("auto", "5_X", "pair"))
    ap.add_argument("--channels", type=int, choices=(2, 6), default=None,
                    help="default: 6 for a 5.1 element, 2 for stereo")
    ap.add_argument("--seconds", type=float, default=None,
                    help="decode only the first S seconds")
    ap.add_argument("--max-pass", type=int, default=1800,
                    help="frames per decode pass (bounds memory). The output "
                         "can differ slightly with it; see README, Known "
                         "issues")
    ap.add_argument("--ref-peak", type=float, default=0.0,
                    help="decoder-scale level written at 0.9 of full scale "
                         "(0 = the per-element default)")
    ap.add_argument("--no-guard", action="store_true",
                    help="turn off the output level guard (damaged-frame "
                         "safety net; see README)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    return decode_file(a.src, a.dst, a.element, a.channels, a.seconds,
                       a.max_pass, a.ref_peak, a.quiet,
                       guard=not a.no_guard)


if __name__ == "__main__":
    sys.exit(main())
