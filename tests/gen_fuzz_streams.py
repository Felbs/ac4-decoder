# SPDX-License-Identifier: Apache-2.0
"""Writes the synthetic streams of c/tests/data (AC-4 sync frames, .syn:
not captures, which the repository never holds) that
`make -C c test` feeds the damaged-input gate (fuzz_test): elements no
real stream at hand uses, written bit by bit by the test writers. They
hold no captured audio (random spectra).

    python tests/gen_fuzz_streams.py c/tests/data"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]


def main(out):
    import test_phase_d as D
    from test_inputs import syncframe
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(2026)
    tm = [(0, 0, 0), (12, 1, 0), (30, 3, 700)]
    upos = [[(9 * k % 63, (5 + 11 * k + 20 * b) % 63, (3 * k + b) % 16)
             for k in range(6)] for b in range(3)]
    for name, kw in (("ajoc_full_aspx", dict(n_dmx=5, aspx=True)),
                     ("ajoc_static", dict(n_dmx=5, static=True))):
        n = 48
        sides = [D._side(rng, kw["n_dmx"], 6, nd=3, start=(5 * i) % 20,
                         ramp=1 + (7 * i) % 40, nb_code=i % 8,
                         qs=i % 2)[0] for i in range(n)]
        frames, _ = D.ajoc_frames(rng, n_umx=6, n=n, sides=sides,
                                  upos=upos, timing=tm,
                                  ugains=[3, 9, 0, 20, 5, 1, 7],
                                  iframe_every=8, **kw)
        with open(os.path.join(out, name + ".syn"), "wb") as fh:
            fh.write(b"".join(syncframe(f) for f in frames))
    # phase E: direct-coded objects (a bed over two substreams and two
    # dynamic objects; ISF over two substreams)
    import test_phase_e as E
    import elem_writer as EW
    rng = np.random.default_rng(2027)
    EW.SF_REF = 160                      # tracks at a realistic level
    for name, subs, objs in (
            ("objs_bed_dyn",
             [dict(kind="bed", n=3, blfe=1, start=("code", 2)),
              dict(kind="bed", n=2, start=None),
              dict(kind="dyn", n=2, lfe=0)],
             [(False, None, None)] * 6 + [(True, (5, 50, 3), -3),
                                          (True, (60, 10, -2), None)]),
            ("objs_isf",
             [dict(kind="isf", n=5, start=1), dict(kind="isf", n=3,
                                                   start=None)],
             [(False, None, None)] * 8)):
        frames, _ = E.objs_frames(rng, subs, objs, n=48, iframe_every=8)
        with open(os.path.join(out, name + ".syn"), "wb") as fh:
            fh.write(b"".join(syncframe(f) for f in frames))
    # phase E: bitstream_version 0 main + associated audio with their
    # metadata (associated gains, DRC); bitstream_version 1 with a
    # presentation of presentation_config 7
    from test_bitstream_v0 import toc_v0
    from test_mix import pair_sub
    rng = np.random.default_rng(2028)
    frames = []
    for i in range(48):
        ifr = int(i % 8 == 0)
        a = E.v0_sub(rng, "pair", dialnorm=92,
                     drc="config" if ifr else "data")
        d = E.v0_sub(rng, "mono", dialnorm=80, assoc=(10, 3, 20, 200))
        frames.append(toc_v0([("10", None), ("0", None)], [len(a), len(d)],
                             config=2, seq=5 + i, iframe=ifr) + a + d)
    with open(os.path.join(out, "v0_assoc.syn"), "wb") as fh:
        fh.write(b"".join(syncframe(f) for f in frames))
    frames = []
    for i in range(48):
        a = pair_sub(rng)
        frames.append(E.toc_v1_cfg7(len(a), 5 + i, int(i % 8 == 0),
                                    skip_extra=1) + bytes(2) + a)
    with open(os.path.join(out, "v1_cfg7.syn"), "wb") as fh:
        fh.write(b"".join(syncframe(f) for f in frames))
    EW.SF_REF = 100


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "c/tests/data")
