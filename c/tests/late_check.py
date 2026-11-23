# SPDX-License-Identifier: Apache-2.0
"""A frame concealed twice: the late audio_size check (_late_closure) on a
frame that was already concealed for another reason. Both decoders.

    python3 c/tests/late_check.py [CAPTURE.m4s ...]

Captures: arguments, or the stereo captures under AC4_TEST_DIR whose first
aspx_config comes after frame 2. For each, frame 1 (before the first
aspx_config, so on the late list) is damaged twice over, in memory:
  * frame_rate_index is changed to another rate with the same
    frame_rate_factor: the frame is concealed as toc.rate (a rule judged
    after the frame went on the late list);
  * one bit of its A-SPX data is flipped: once the configuration is known,
    the late check finds that the frame does not close.
Required, Python (Decoder) and C (libac4dec.so): frame 1 keeps the reason
toc.rate, no frame is audio_size.late, and the Python counts every frame
once (n_bad = the frames concealed, n_ok = the rest). Before the fix the Python re-marked frame
1 as audio_size.late and counted it twice in n_bad (n_ok one short), and
the C re-marked it. Nothing is written.
"""
from __future__ import annotations

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import robust_compare as RC                           # noqa: E402
from ac4 import mp4                                   # noqa: E402
from ac4 import toc as M                              # noqa: E402
from ac4 import stream as ST                          # noqa: E402
from ac4.decoder import Decoder, probe_element        # noqa: E402

DAMAGED = 1


def flip(b, bit):
    b = bytearray(b)
    b[bit >> 3] ^= 0x80 >> (bit & 7)
    return bytes(b)


def rate_flip(f):
    """-> f with frame_rate_index changed (same factor, same TOC size)."""
    st0 = M.parse(f)
    for bit in range(48):
        g = flip(f, bit)
        try:
            st = M.parse(g)
        except Exception:                                  # noqa: BLE001
            continue
        if (st["frame_rate_index"] != st0["frame_rate_index"]
                and st["toc_bytes"] == st0["toc_bytes"]
                and st["frame_rate_factor"] == st0["frame_rate_factor"]
                and st["substream_sizes"] == st0["substream_sizes"]
                and all(st[k] == st0[k] for k in
                        ("sequence_counter", "fs_index", "b_iframe_global"))):
            return g
    return None


def first_cfg(frames, el):
    s = ST.Ac4Stream(el)
    _w, _l, groups, _c = s.decode_frames(frames, ("L", "R"))
    for i, g in enumerate(groups["lr"]):
        if g is not None:
            return i
    return None


def aspx_flip(frames, el, fi):
    """-> frame fi with one bit flipped in its A-SPX data such that, with
    the stream's configuration, its blocks no longer close."""
    s = ST.Ac4Stream(el)
    s.decode_frames(frames, ("L", "R"))
    cfg = s.cfg
    f = frames[fi]
    st = M.parse(f)
    o = st["toc_bytes"] + st.get("payload_base", 0) + st["substream_sizes"][0]
    from ac4 import pair as PAIR
    sub = f[o:o + st["substream_sizes"][1]]
    r = PAIR.decode_pair(sub, s.T, b_iframe=bool(st["b_iframe_global"]))
    stop = r["audio_start"] + 8 * r["audio_size"]
    t = ST.R37.t_for(cfg)
    for k in range(r["bits"], stop):
        g = flip(sub, k)
        b = M.Bits(g)
        b.p = r["bits"]
        try:
            ST.P.aspx_data(b, s.tabs, t, cfg, s.nats, False, True,
                           ST.RB.TrackingState(), 0)
            ok = stop - 7 <= b.p <= stop
        except Exception:                                  # noqa: BLE001
            ok = False
        if not ok:
            return f[:o] + g + f[o + len(sub):]
    return None


def check(path):
    frames = mp4.read_frames(path)
    el = probe_element(frames)
    if el != "pair":
        return None
    fc = first_cfg(frames, el)
    if fc is None or fc <= DAMAGED + 1:
        return None
    frames = frames[:fc + 8]
    g = aspx_flip(frames, el, DAMAGED)
    g = rate_flip(g) if g else None
    if g is None:
        print(f"{os.path.basename(path)}: no damage found, skipped")
        return None
    dam = list(frames)
    dam[DAMAGED] = g
    d = Decoder(el)
    d.decode_pass(dam)
    st_py = d.last_status
    _y, _c, st_c = RC.c_decode(dam, el, lookahead=0)
    ok = True
    for nm, st in (("py", st_py), ("c", st_c)):
        late = [i for i, s in enumerate(st) if s == "audio_size.late"]
        good = st[DAMAGED] == "toc.rate" and not late
        print(f"{os.path.basename(path)} {nm}: frame {DAMAGED} "
              f"{st[DAMAGED]}, audio_size.late at {late} "
              f"{'ok' if good else 'FAIL'}")
        ok &= good
    n = len(dam)
    n_conc = sum(s is not None for s in st_py)
    good = d.n_bad == n_conc and d.n_ok == n - n_conc
    print(f"  py n_ok {d.n_ok}, n_bad {d.n_bad}; {n_conc} of {n} frames "
          f"concealed {'ok' if good else 'FAIL'}")
    return ok and good


def main(argv):
    files = argv
    if not files:
        dd = os.environ.get("AC4_TEST_DIR")
        if not dd:
            sys.exit("pass capture files or set AC4_TEST_DIR")
        files = sorted(glob.glob(os.path.join(dd, "**", "*.m4s"),
                                 recursive=True))
    res = [r for r in (check(p) for p in files) if r is not None]
    if not res:
        sys.exit("no stereo capture with a late first aspx_config")
    print("PASS" if all(res) else "FAIL")
    return 0 if all(res) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
