# SPDX-License-Identifier: Apache-2.0
"""Deterministic damage for AC-4 frame lists, and the measurements the
robustness gate is made of. Used by tests/test_robust.py and by
tools/errinject.py (the long run).

Every damage is a pure function of (frames, kind, seed): the same seed
always damages the same frames in the same way.

KINDS
  bitflip    1..8 random bits flipped anywhere in the frame (TOC included)
  burst      a run of 2..32 consecutive bytes XOR-ed with random bytes
  truncate   the frame cut to 0..90 % of its length
  zero       a run of 1..64 bytes set to zero
  swap       the frame exchanged with the one after it (both count as
             damaged)
  garbage    the frame replaced by random bytes of the same length
  foreign    the frame replaced by a frame from another stream (or, given
             none, from far away in the same stream)
  repeat     the frame replaced by a copy of the one before it
  extend     random bytes appended to the frame
  empty      a zero-length sample
  run        3..20 CONSECUTIVE frames, each with 1..3 flipped bits or a
             2..32-byte burst (reception damage comes in runs; the hits
             above are kept at least 6 frames apart)
  first      1..3 bits flipped in the first 8 bytes (the TOC) of frame 0,
             the frame a decoder learns the stream's layout from
"""
from __future__ import annotations

import numpy as np

KINDS = ("bitflip", "burst", "truncate", "zero", "swap", "garbage",
         "foreign", "repeat", "extend", "empty", "run", "first")
FRAME = 1536


def pick_positions(n_frames, n_hits, rng, first=8, gap=6):
    """n_hits frame indices in [first, n_frames - 2), at least `gap`
    apart, so the effect of one hit can be told from the next."""
    cand = list(range(first, max(first, n_frames - 2)))
    rng.shuffle(cand)
    out = []
    for c in cand:
        if all(abs(c - o) >= gap for o in out):
            out.append(c)
        if len(out) >= n_hits:
            break
    return sorted(out)


def damage(frames, kind, seed, n_hits=6, foreign=None):
    """-> (damaged frame list, sorted list of damaged indices)."""
    rng = np.random.default_rng(seed)
    out = [bytes(f) for f in frames]
    if kind == "run":
        return _run(out, rng)
    if kind == "first":
        f = bytearray(out[0])
        for _ in range(int(rng.integers(1, 4))):
            j = int(rng.integers(0, 8 * min(8, len(f))))
            f[j >> 3] ^= 0x80 >> (j & 7)
        out[0] = bytes(f)
        return out, [0]
    pos = pick_positions(len(out), n_hits, rng)
    hit = set()
    for i in pos:
        f = bytearray(out[i])
        n = len(f)
        if kind == "bitflip":
            for _ in range(int(rng.integers(1, 9))):
                if n:
                    j = int(rng.integers(0, 8 * n))
                    f[j >> 3] ^= 0x80 >> (j & 7)
            out[i] = bytes(f)
        elif kind == "burst":
            if n:
                m = int(rng.integers(2, 33))
                a = int(rng.integers(0, max(1, n - m)))
                for j in range(a, min(n, a + m)):
                    f[j] ^= int(rng.integers(1, 256))
            out[i] = bytes(f)
        elif kind == "truncate":
            out[i] = bytes(f[:int(n * rng.uniform(0.0, 0.9))])
        elif kind == "zero":
            if n:
                m = int(rng.integers(1, 65))
                a = int(rng.integers(0, max(1, n - m)))
                f[a:a + m] = bytes(len(f[a:a + m]))
            out[i] = bytes(f)
        elif kind == "swap":
            out[i], out[i + 1] = out[i + 1], out[i]
            hit.add(i + 1)
        elif kind == "garbage":
            out[i] = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        elif kind == "foreign":
            src = foreign if foreign else frames
            k = int(rng.integers(0, len(src)))
            if src is frames and abs(k - i) < 30:
                k = (i + len(src) // 2) % len(src)
            out[i] = bytes(src[k])
        elif kind == "repeat":
            out[i] = out[i - 1]
        elif kind == "extend":
            out[i] = bytes(f) + rng.integers(
                0, 256, int(rng.integers(1, 200)), dtype=np.uint8).tobytes()
        elif kind == "empty":
            out[i] = b""
        else:
            raise ValueError(kind)
        hit.add(i)
    return out, sorted(hit)


def _run(out, rng):
    n_run = int(rng.integers(3, 21))
    a = int(rng.integers(8, max(9, len(out) - n_run - 2)))
    hit = []
    for i in range(a, min(len(out) - 2, a + n_run)):
        f = bytearray(out[i])
        n = len(f)
        if n and rng.random() < 0.5:
            for _ in range(int(rng.integers(1, 4))):
                j = int(rng.integers(0, 8 * n))
                f[j >> 3] ^= 0x80 >> (j & 7)
        elif n:
            m = int(rng.integers(2, 33))
            b = int(rng.integers(0, max(1, n - m)))
            for j in range(b, min(n, b + m)):
                f[j] ^= int(rng.integers(1, 256))
        out[i] = bytes(f)
        hit.append(i)
    return out, hit


def frame_peaks(y, frame=FRAME):
    n = len(y) // frame
    if n == 0:
        return np.zeros(0)
    return np.abs(y[:n * frame]).reshape(n, frame, -1).max(axis=(1, 2))


def assess(clean, dmg, hits, concealed, window=(1, 8), limit=2.0):
    """Compare a damaged decode with the clean decode of the same frames.

    clean, dmg   (n * 1536, ch) arrays
    hits         damaged frame indices
    concealed    bool per frame: the decoder concealed it
    -> dict(peak_ratio = max|dmg| / max|clean|,
            per_hit = [(index, verdict, local ratio, level ratio)])

    For frames i-1 .. i+8 around a hit, `local ratio` is the damaged
    decode's peak over the clean decode's peak in the same frames, and
    `level ratio` the damaged decode's peak there over the clean decode's
    peak over the WHOLE stream. Verdicts:
        "concealed"  the decoder flagged the frame;
        "identical"  frames i-1 .. i+8 equal the clean decode;
        "bounded"    not flagged, the output differs, but its level ratio
                     is at most `limit` (no sample there above 2x the
                     clean stream's own peak -- the gate's level rule);
        "LOUD"       anything else.
    The local ratio is reported, not gated: clean programme has isolated
    transients up to 28x their neighbours (MEASURED), so a damaged click
    that stays inside the programme's own level cannot be told from one.
    """
    pc = frame_peaks(clean)
    pd = frame_peaks(dmg)
    cmax = float(np.abs(clean).max()) if clean.size else 0.0
    dmax = float(np.abs(dmg).max()) if dmg.size else 0.0
    per = []
    for i in hits:
        a, z = max(0, i - window[0]), min(len(pc), i + window[1])
        lc = float(pc[a:z].max()) if z > a else 0.0
        ld = float(pd[a:z].max()) if z > a else 0.0
        local = ld / lc if lc > 0 else (0.0 if ld == 0 else np.inf)
        level = ld / cmax if cmax > 0 else (0.0 if ld == 0 else np.inf)
        if concealed[i]:
            v = "concealed"
        elif np.array_equal(clean[a * FRAME:z * FRAME],
                            dmg[a * FRAME:z * FRAME]):
            v = "identical"
        elif level <= limit:
            v = "bounded"
        else:
            v = "LOUD"
        per.append((i, v, local, level))
    return dict(peak_ratio=(dmax / cmax if cmax > 0 else
                            (0.0 if dmax == 0 else np.inf)),
                per_hit=per)
