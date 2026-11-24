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


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "c/tests/data")
