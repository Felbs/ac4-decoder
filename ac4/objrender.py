# SPDX-License-Identifier: Apache-2.0
"""A simple object renderer: object positions (OAMD, oamd.py) to the
loudspeakers of an output layout (render.LAYOUTS).

NOT DOLBY'S RENDERER, and not specified by TS 103 190-2 (which leaves
object rendering to the device): a plain, documented amplitude panner,
for listening to object content at all.

* Positions: the room-anchored coordinates of 6.3.9.8.4 (x left 0 ..
  right 1, y front 0 .. back 1, z floor -1 .. ceiling 1) are taken to
  the cartesian frame of TS 103 190-2 Table A.30 (X = 2 x - 1 to the
  right, Y = 1 - 2 y to the front, Z = z).
* Loudspeakers: the layout's channels at the azimuths of Table A.30 (L /
  R +-45, C 0, the surrounds +-110 in 5.X layouts and +-90 in 7.X ones,
  Lb / Rb +-135, Tfl / Tfr +-45, Tbl / Tbr +-135, Tsl / Tsr +-90), in
  two rings: the ear layer (Z = 0) and the top layer (Z = 1).
* Within a ring: pairwise constant-power panning (the two-dimensional
  form of vector base amplitude panning) between the two loudspeakers
  around the object's azimuth atan2(-X, Y); an object at the centre of
  the room (X = Y = 0) feeds every loudspeaker of the ring equally (at
  1 / sqrt(n) each).
* Between rings: z > 0 crossfades to the top ring with constant power
  (cos / sin of z * 90 degrees); z <= 0, or a layout without a top
  ring, renders to the ear ring alone.
* An LFE object goes to the LFE channel (dropped when the layout has
  none). Gains are per object (object_gain), per frame: the last
  metadata block of a frame holds for the whole frame (no ramps)."""
from __future__ import annotations

import math

import numpy as np

AZ = {"L": 45.0, "R": -45.0, "C": 0.0, "Lb": 135.0, "Rb": -135.0,
      "Tfl": 45.0, "Tfr": -45.0, "Tbl": 135.0, "Tbr": -135.0,
      "Tsl": 90.0, "Tsr": -90.0, "Lw": 60.0, "Rw": -60.0}
TOP = {"Tfl", "Tfr", "Tbl", "Tbr", "Tsl", "Tsr"}


def speaker_azimuths(chans):
    """{channel: azimuth} of a layout's loudspeakers (no LFE)."""
    seven = "Lb" in chans
    out = {}
    for k in chans:
        if k == "lfe":
            continue
        if k in ("Ls", "Rs"):
            a = 90.0 if seven else 110.0
            out[k] = a if k == "Ls" else -a
        else:
            out[k] = AZ[k]
    return out


def _ring_gains(az, ring):
    """Pairwise constant-power panning of azimuth az (degrees, None for
    the centre of the room) over ring = [(channel, azimuth)] -> {channel:
    gain}."""
    if not ring:
        return {}
    if az is None or len(ring) == 1:
        g = 1.0 / math.sqrt(len(ring))
        return {k: g for k, _a in ring}
    sp = sorted(ring, key=lambda t: t[1])
    a = (az + 180.0) % 360.0 - 180.0
    n = len(sp)
    for i in range(n):
        k0, a0 = sp[i]
        k1, a1 = sp[(i + 1) % n]
        hi = a1 if i + 1 < n else a1 + 360.0
        x = a if a >= a0 else a + 360.0
        if a0 <= x <= hi:
            t = (x - a0) / (hi - a0) if hi > a0 else 0.0
            return {k0: math.cos(t * math.pi / 2),
                    k1: math.sin(t * math.pi / 2)}
    return {sp[0][0]: 1.0}


def gains(pos, chans, lfe=False):
    """-> array (len(chans),): the gains of one object at pos = (x, y, z)
    (6.3.9.8.4) on the layout's channels."""
    g = np.zeros(len(chans))
    idx = {k: i for i, k in enumerate(chans)}
    if lfe:
        if "lfe" in idx:
            g[idx["lfe"]] = 1.0
        return g
    x, y, z = pos
    X, Y, Z = 2.0 * x - 1.0, 1.0 - 2.0 * y, max(-1.0, min(1.0, z))
    az = (None if abs(X) < 1e-9 and abs(Y) < 1e-9
          else math.degrees(math.atan2(-X, Y)))
    sa = speaker_azimuths(chans)
    ear = [(k, a) for k, a in sa.items() if k not in TOP]
    top = [(k, a) for k, a in sa.items() if k in TOP]
    wt = math.sin(Z * math.pi / 2) if (Z > 0 and top) else 0.0
    we = math.cos(Z * math.pi / 2) if (Z > 0 and top) else 1.0
    for k, v in _ring_gains(az, ear).items():
        g[idx[k]] += we * v
    if wt:
        for k, v in _ring_gains(az, top).items():
            g[idx[k]] += wt * v
    return g


def render(signals, objects, chans):
    """signals (n, n_obj) of one frame, objects [(pos, gain_db, lfe)] ->
    (n, len(chans))."""
    M = np.zeros((len(chans), signals.shape[1]))
    for j, (pos, gdb, lfe) in enumerate(objects):
        lin = 0.0 if gdb == -math.inf else 10.0 ** (gdb / 20.0)
        M[:, j] = lin * gains(pos, chans, lfe)
    return signals @ M.T
