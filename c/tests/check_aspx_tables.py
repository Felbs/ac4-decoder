# SPDX-License-Identifier: Apache-2.0
"""The C A-SPX tables against the Python's, for every aspx_config.

    python3 c/tests/check_aspx_tables.py      (after make -C c test)

For each (start_freq, stop_freq, master_freq_scale, noise_sbg) the C
(tests/unit_test --dump-aspx) and the Python (aspx_bands.sbg_tables,
aspx_envadj.sbg_lim, aspx_env.sbg_maps, and the aspx.config rule of
ac4/stream.py) must agree: both reject the configuration, or both give
the same master, high/low-resolution and noise tables, patches, limiter
bands and index maps. The C also rejects configurations whose patches
would index outside the 64 QMF subbands (the Python fails on those later,
in its renderer); those are listed.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from ac4 import aspx_bands as A          # noqa: E402
from ac4 import aspx_env as V            # noqa: E402
from ac4 import aspx_envadj as E         # noqa: E402


def py_tables(st, sp, ms, ns):
    cfg = dict(start_freq=st, stop_freq=sp, master_freq_scale=ms,
               noise_sbg=ns)
    try:
        t = A.sbg_tables(cfg, 0)
        if not (t["n_hi"] >= 1 and t["n_lo"] >= 1 and t["num_sb_aspx"] >= 1):
            return None
    except Exception:                                          # noqa: BLE001
        return None
    h2l, l2h = V.sbg_maps(t)
    return dict(master=t["master"], hi=t["hi"], lo=t["lo"], noise=t["noise"],
                pnum=t["patch_num_sb"], pstart=t["patch_start_sb"],
                patches=t["patches"], lim=E.sbg_lim(t), h2l=h2l, l2h=l2h)


def main():
    out = subprocess.run([os.path.join(HERE, "unit_test"), "--dump-aspx"],
                         capture_output=True, text=True, check=True).stdout
    bad = n = n_ok = 0
    outside = []
    for line in out.splitlines():
        f = line.split()
        st, sp, ms, ns = map(int, f[:4])
        py = py_tables(st, sp, ms, ns)
        n += 1
        if f[4] == "fail":
            if py is not None:
                ok_geom = all(0 <= s and s + k <= 64 and d + k <= 64
                              for s, k, d in zip(py["pstart"], py["pnum"],
                                                 py["patches"]))
                if ok_geom:
                    print("C rejects, Python accepts:", st, sp, ms, ns, py)
                    bad += 1
                else:
                    outside.append((st, sp, ms, ns))
            continue
        c = {}
        for kv in f[4:]:
            k, v = kv.split("=")
            c[k] = [int(x) for x in v.split(",")] if v else []
        if py is None:
            print("C accepts, Python rejects:", st, sp, ms, ns)
            bad += 1
            continue
        n_ok += 1
        for k in py:
            if list(py[k]) != c[k]:
                print(f"{st} {sp} {ms} {ns} {k}: py {py[k]} c {c[k]}")
                bad += 1
    print(f"{n} configurations, {n_ok} with tables, identical in C and "
          f"Python; {len(outside)} rejected by the C only (patches outside "
          f"the 64 subbands): {outside}; {bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
