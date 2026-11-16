# SPDX-License-Identifier: Apache-2.0
"""Helpers for the integration tests: find banked captures, cut them short,
and (optionally) run the original receiver's audio worker on the same cut.

Captures are NEVER part of this repository. Point AC4_TEST_DIR at a folder
of fragmented-MP4 AC-4 files (*.m4s / *.mp4); the integration tests skip
when it is unset or holds none.

To also prove the copy is unchanged against the code it was copied from,
set AC4_REFERENCE_REPO to a checkout of that receiver (it must contain
tools/atsc3_audio.py and lab/); its Python is AC4_REFERENCE_PYTHON or the
current interpreter.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys


def capture_files():
    d = os.environ.get("AC4_TEST_DIR")
    if not d or not os.path.isdir(d):
        return []
    out = []
    for pat in ("*.m4s", "*.mp4"):
        out += glob.glob(os.path.join(d, "**", pat), recursive=True)
    return sorted(out)


def cut_fragments(src, dst, n_frag):
    """Copy everything before the first moof, plus the first n_frag
    complete moof+mdat pairs. -> number of fragments written."""
    d = open(src, "rb").read()
    o, k, head_end, end = 0, 0, None, None
    while o + 8 <= len(d):
        sz = int.from_bytes(d[o:o + 4], "big")
        if sz < 8 or o + sz > len(d):
            break
        t = d[o + 4:o + 8]
        if t == b"moof":
            if head_end is None:
                head_end = o
            m2 = o + sz
            if m2 + 8 > len(d):
                break
            sz2 = int.from_bytes(d[m2:m2 + 4], "big")
            if d[m2 + 4:m2 + 8] != b"mdat" or m2 + sz2 > len(d):
                break
            k += 1
            end = m2 + sz2
            o = end
            if k >= n_frag:
                break
            continue
        o += sz
    if end is None:
        return 0
    with open(dst, "wb") as fh:
        fh.write(d[:end])
    return k


def reference_repo():
    r = os.environ.get("AC4_REFERENCE_REPO")
    if r and os.path.exists(os.path.join(r, "tools", "atsc3_audio.py")):
        return r
    return None


def run_reference_worker(repo, lane_path, out_wav, element, channels,
                         max_pass, timeout=600):
    """Run the receiver's production audio worker once over `lane_path`,
    which it finds through a one-lane live.json beside it. -> (rc, log)."""
    d = os.path.dirname(os.path.abspath(lane_path))
    lane = dict(path=os.path.abspath(lane_path), kind="audio", pid=1,
                generation=0)
    with open(os.path.join(d, "live.json"), "w") as fh:
        json.dump({"lanes": {"audio_pid1": lane}}, fh)
    py = os.environ.get("AC4_REFERENCE_PYTHON", sys.executable)
    cmd = [py, os.path.join(repo, "tools", "atsc3_audio.py"),
           "--live-dir", d, "--pid", "1", "--once", "--wait", "5",
           "--element", element, "--channels", str(channels),
           "--max-pass", str(max_pass), "--out", out_wav]
    env = dict(os.environ)
    # the receiver names its kernel switches differently; keep them in step
    for k_new, k_old in (("AC4_QMF_KERNEL", "ATSC3_QMF_KERNEL"),
                         ("AC4_SPECTRAL_KERNEL", "ATSC3_SPECTRAL_KERNEL")):
        if k_new in os.environ:
            env[k_old] = os.environ[k_new]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def wav_data(path):
    """-> (channels, rate, raw PCM bytes)."""
    import wave
    with wave.open(path, "rb") as w:
        return w.getnchannels(), w.getframerate(), w.readframes(w.getnframes())
