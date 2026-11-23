# SPDX-License-Identifier: Apache-2.0
"""Check that the C input layer (c/src/input.c, mp4.c) reads the same
frames as ac4/container.py.

    python3 c/tests/check_inputs.py FILE ...

For each file both sides give the kind, the frame count, the CRC-failed
and skipped counts and, per frame, its length and FNV-1a hash; any
difference is printed. With --damage N, each file is also checked after
N random byte changes (same seed on both sides), which exercises the
resync, CRC and TS continuity paths."""
from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from ac4 import container                                  # noqa: E402

UNIT = os.path.join(REPO, "c", "tests", "unit_test")
KIND = {None: 0, "mp4": 1, "sync": 2, "ts": 3}


def fnv(b):
    h = 2166136261
    for x in b:
        h = ((h ^ x) * 16777619) & 0xFFFFFFFF
    return h


def python_side(data):
    s = container.read_bytes(data)
    sy = s.info.get("sync", s.info) if s.kind in ("sync", "ts") else {}
    head = (KIND[s.kind], len(s.frames), sy.get("n_crc_bad", 0),
            sy.get("n_skipped", 0), s.info.get("n_cc_errors", 0))
    return head, [(len(f), fnv(f)) for f in s.frames]


def c_side(path):
    out = subprocess.run([UNIT, "--dump-input", path], capture_output=True,
                         text=True, check=True).stdout.split("\n")
    head = tuple(int(x) for x in out[0].split())
    fr = [(int(a), int(b, 16)) for a, b in
          (ln.split() for ln in out[1:] if ln.strip())]
    return head, fr


def check(path, data, label):
    py = python_side(data)
    c = c_side(path)
    ok = py == c
    print(f"{'ok  ' if ok else 'DIFF'} {label}: kind {py[0][0]}, "
          f"{py[0][1]} frames, crc bad {py[0][2]}, skipped {py[0][3]}, "
          f"cc errors {py[0][4]}" + ("" if ok else f"  C: {c[0]}"))
    if not ok:
        for i, (a, b) in enumerate(zip(py[1], c[1])):
            if a != b:
                print(f"     first frame difference at {i}: py {a} c {b}")
                break
    return ok


def main(argv):
    n_dmg = 0
    if argv and argv[0] == "--damage":
        n_dmg = int(argv[1])
        argv = argv[2:]
    bad = 0
    for p in argv:
        data = open(p, "rb").read()
        bad += not check(p, data, os.path.basename(p))
        if n_dmg:
            rng = random.Random(fnv(os.path.basename(p).encode()))
            d = bytearray(data)
            for _ in range(n_dmg):
                d[rng.randrange(len(d))] = rng.randrange(256)
            with tempfile.NamedTemporaryFile(suffix=".bin") as tf:
                tf.write(d)
                tf.flush()
                bad += not check(tf.name, bytes(d),
                                 os.path.basename(p) + " (damaged)")
    print("ALL SAME" if not bad else f"{bad} DIFFERENT")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
