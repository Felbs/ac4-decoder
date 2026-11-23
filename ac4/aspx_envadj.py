# SPDX-License-Identifier: Apache-2.0
"""M36 -- A-SPX envelope adjustment: scaling the patched high band.

M35 copied low subbands into 12..21 kHz.  They are at the WRONG LEVEL until
this stage scales them to the transmitted envelopes.  That is what makes the
high band correct rather than merely present.

THE CHAIN, clause 5.7.6.4.2
-----------------------------
    Pseudocode 90   est_sig_sb   measure what the patched band actually has
    Pseudocode 82   scf_sig_sbg  = num_qmf_subbands * 2**(qscf/a)
    Pseudocode 83   scf_noise_sbg = 2**(NOISE_FLOOR_OFFSET - qscf_noise)
    Pseudocode 91   map both onto QMF subbands
    Pseudocode 95   sig_gain_sb  = sqrt(scf_sig / (1 + est_sig) / (1+scf_noise))
    Pseudocode 96   max_sig_gain per limiter subband group
    Pseudocode 98   limit the gain
    Pseudocode 99   boost factor -- recover the energy the limiter removed
    Pseudocode 100  cap the boost at 1.584893192
    Pseudocode 101  apply the boost to the gain
    Pseudocode 94   noise_lev_sb = sqrt(scf_sig/(1+scf_noise) * scf_noise)
    Pseudocode 97   trim the noise in proportion to the gain limiting
    Pseudocode 83   scf_noise_sbg = 2 ** (NOISE_FLOOR_OFFSET - qscf_noise)
                    -- note the SIGN: a larger index means LESS noise
    Table D.2       ASPX_NOISE[512][2], verified: mean |z|^2 = 1.000000
                    exactly, every entry unit magnitude, phase spread 1.787
                    rad against uniform 1.814

`aspx_interpolation = 1` in this stream, which selects Pseudocode 90's simpler
per-subband estimate rather than the per-subband-group average.

THE GATE
---------
The whole point of the stage is that the adjusted band carries the TRANSMITTED
envelope.  So after adjustment, measure the band's energy again and compare it
to the target `scf_sig_sb`.  If the gain chain is right the two agree; if the
estimate, the dequantisation, the mapping or the limiter is wrong they do not.
That is a closed loop with no free parameters, and it needs no reference
decoder.

SCOPE -- STATED BEFORE THE RESULTS
------------------------------------
Not implemented, and counted rather than hidden:

  * **additional harmonics.**  Signalled in under 1.5 % of channel-frames
    (ah_left 50 of 6960, ah_right 11, ah_present 44 of 3480); those frames lose
    a tone.  The sine branch of Pseudocode 95 is therefore never taken here.
  * **the limiter subband table** (Pseudocode 72) is implemented in its main
    form -- merge sbg_sig_lowres with the patch borders, sort, and drop borders
    closer than 0.245 octaves preferring to keep patch borders.  The spec's
    full tie-breaking is more elaborate.

frequency- and time-interleaved coding are NOT scope gaps: `fic_present` and
`tic_present` are 0 in every frame of this capture.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from . import hcb as H                                               # noqa: E402
from . import spectral as S                                          # noqa: E402
from . import channels as C                                          # noqa: E402
from . import aspx_bands as A                                        # noqa: E402
from . import aspx_parse as P                                        # noqa: E402
from . import qmf as Q                                               # noqa: E402
from . import aspx_env as V                                          # noqa: E402
from . import aspx_hfgen as G                                             # noqa: E402

NOISE_FLOOR_OFFSET = 6
EPSILON = 1.0
EPSILON0 = 1e-12
LIM_GAIN = 1.41254
MAX_SIG_GAIN = 1e5
MAX_BOOST_FACT = 1.584893192


def sbg_lim(t):
    """Pseudocode 72, main form: lowres borders merged with patch borders."""
    cand = sorted(set(t["lo"]) | set(t["patches"]))
    patch = set(t["patches"])
    out = [cand[0]]
    for b in cand[1:]:
        if math.log2(b / out[-1]) < 0.245:
            # too close: keep whichever is a patch border
            if b in patch and out[-1] not in patch:
                out[-1] = b
            continue
        out.append(b)
    return out


def estimate(Qh, t, borders, sl0, nts_in_ats, ts_off=0):
    """Pseudocode 90, the aspx_interpolation == 1 branch."""
    n_sb, n_env = t["num_sb_aspx"], len(borders) - 1
    est = np.zeros((n_sb, n_env))
    for e in range(n_env):
        ta = int(borders[e]) * nts_in_ats + ts_off + sl0
        tz = int(borders[e + 1]) * nts_in_ats + ts_off + sl0
        ta, tz = max(ta, 0), min(tz, Qh.shape[1])
        if tz <= ta:
            continue
        seg = np.abs(Qh[t["sbx"]:t["sbx"] + n_sb, ta:tz]) ** 2
        est[:, e] = seg.mean(axis=1)
    return est


def estimate_groups(Qh, t, borders, freq_res, sl0, nts_in_ats, ts_off=0):
    """Pseudocode 90, the aspx_interpolation == 0 branch: the mean energy
    over each subband group (of the envelope's resolution) and the
    envelope's slots, for every subband of the group. (Normalised per QMF
    slot, as estimate() is; see the README, Interpretations.)"""
    n_sb, n_env = t["num_sb_aspx"], len(borders) - 1
    sbx = t["sbx"]
    est = np.zeros((n_sb, n_env))
    for e in range(n_env):
        ta = int(borders[e]) * nts_in_ats + ts_off + sl0
        tz = int(borders[e + 1]) * nts_in_ats + ts_off + sl0
        ta, tz = max(ta, 0), min(tz, Qh.shape[1])
        if tz <= ta:
            continue
        tab = t["hi"] if freq_res[e] else t["lo"]
        for g in range(len(tab) - 1):
            seg = np.abs(Qh[tab[g]:tab[g + 1], ta:tz]) ** 2
            est[tab[g] - sbx:tab[g + 1] - sbx, e] = seg.mean()
    return est


# THE SCALE OF THE ENVELOPES. scf_sig_sbg is an absolute QMF-domain energy
# on the standard's scale (Pseudocode 82: num_qmf_subbands * 2^(q / a)).
# This decoder's samples are the standard's times filterbank.IMDCT_K
# (sqrt(2 * 1536)), so the energies it compares them with (est) are
# IMDCT_K^2 larger: the dequantised envelopes take the same factor.
# (Until the 1.5 dB envelopes were dequantised with a = 2, an a = 1 that
# inflated the envelopes of FIXFIX one-envelope intervals stood in for
# this factor on average; the frames it overshot were the "A-SPX bursts".)
ENV_SCALE = 64.0 * 2.0 * 1536.0


def env_a(qmode):
    """Pseudocodes 82, 84: a = 2 for aspx_qmode_env 0 (1.5 dB steps), 1 for
    aspx_qmode_env 1 (3.0 dB). aspx_data forces qmode 0 for a FIXFIX
    interval with one envelope, whatever aspx_quant_mode_env says."""
    return 2 if qmode == 0 else 1


def map_scf(qscf_env, t, fr, n_sb, a=1):
    """Pseudocode 82 + 91: dequantise the signal envelope onto QMF subbands.
    a: env_a(the channel's aspx_qmode_env)."""
    n_env = len(qscf_env)
    out = np.zeros((n_sb, n_env))
    for e, q in enumerate(qscf_env):
        tbl = t["hi"] if fr["freq_res"][e] else t["lo"]
        for sbg, val in enumerate(q):
            if sbg + 1 >= len(tbl):
                break
            lo = tbl[sbg] - t["sbx"]
            hi = min(tbl[sbg + 1] - t["sbx"], n_sb)
            out[max(lo, 0):hi, e] = ENV_SCALE * (2.0 ** (val / a))
    return out


PAN_OFFSET = 12


def map_scf_joint(qa, qb, t, fr, n_sb, a=1):
    """Pseudocode 84 -- JOINT dequantisation when aspx_balance = 1.

    With balance coding the pair is a SUM channel and a BALANCE channel, and
    channel B's transmitted values are a pan ratio, not a level.  Pseudocode 82
    covers only balance = 0 and says so; running it on channel B treats a pan
    ratio as an absolute level, which is how the right channel ended up 20x too
    loud and uncorrelated with its own core.

        nom     = 2 ** (qscf_a/a + 1) * num_qmf_subbands
        denom_a = 1 + 2 ** (PAN_OFFSET - qscf_b/a)
        denom_b = 1 + 2 ** (qscf_b/a - PAN_OFFSET)
    """
    n_env = min(len(qa), len(qb))
    A = np.zeros((n_sb, n_env))
    B = np.zeros((n_sb, n_env))
    for e in range(n_env):
        tbl = t["hi"] if fr["freq_res"][e] else t["lo"]
        for sbg in range(min(len(qa[e]), len(qb[e]))):
            if sbg + 1 >= len(tbl):
                break
            lo = max(tbl[sbg] - t["sbx"], 0)
            hi = min(tbl[sbg + 1] - t["sbx"], n_sb)
            if hi <= lo:
                continue
            va, vb = qa[e][sbg], qb[e][sbg]
            nom = (2.0 ** (va / a + 1)) * ENV_SCALE
            A[lo:hi, e] = nom / (1.0 + 2.0 ** (PAN_OFFSET - vb / a))
            B[lo:hi, e] = nom / (1.0 + 2.0 ** (vb / a - PAN_OFFSET))
    return A, B


def map_noise_joint(qa, qb, t, n_sb, n_env):
    """Pseudocode 84, the noise half."""
    A = np.zeros((n_sb, n_env))
    B = np.zeros((n_sb, n_env))
    for e in range(n_env):
        ia = min(e, len(qa) - 1)
        ib = min(e, len(qb) - 1)
        for sbg in range(min(t["n_noise"], len(qa[ia]), len(qb[ib]))):
            lo = max(t["noise"][sbg] - t["sbx"], 0)
            hi = min(t["noise"][sbg + 1] - t["sbx"], n_sb)
            if hi <= lo:
                continue
            va, vb = qa[ia][sbg], qb[ib][sbg]
            nom = 2.0 ** (NOISE_FLOOR_OFFSET - va + 1)
            A[lo:hi, e] = nom / (1.0 + 2.0 ** (PAN_OFFSET - vb))
            B[lo:hi, e] = nom / (1.0 + 2.0 ** (vb - PAN_OFFSET))
    return A, B


def noise_qscf(dat, t, prev):
    """Reconstruct the noise envelope INDICES (before dequantisation)."""
    delta = 2 if (dat["ch"] == 1 and dat["balance"] == 1) else 1
    dirs = dat.get("dir_noise") or [0] * len(dat["noise"])
    out, prev_q = [], prev
    for e, vals in enumerate(dat["noise"]):
        n = t["n_noise"]
        q = [0] * n
        if e < len(dirs) and dirs[e] == 0:
            acc = 0
            for sbg in range(min(n, len(vals))):
                acc += delta * vals[sbg]
                q[sbg] = acc
        else:
            for sbg in range(min(n, len(vals))):
                base = prev_q[sbg] if (prev_q and sbg < len(prev_q)) else 0
                q[sbg] = base + delta * vals[sbg]
        out.append(q)
        prev_q = q
    return out, prev_q


def gain_unlimited(sig_sb, est):
    """Pseudocode 95, non-sine branch, before the limiter touches it."""
    return np.sqrt(sig_sb / (EPSILON + est))


def transient_envelopes(fr, n_env):
    """The envelopes Pseudocodes 95 and 99 treat apart: atsg ==
    aspx_tsg_ptr (the envelope that starts at a transient) and atsg ==
    p_sine_at_end (Pseudocode 92: envelope 0 when the previous interval's
    pointer was at its end, num_atsg_sig_prev). -> bool per envelope.

    FIXFIX intervals carry no aspx_tsg_ptr (Table 53); aspx_parse gives
    them -1, the value the field's own coding (tmp - 1) uses for "none"."""
    ptr = fr.get("tsg_ptr", -1)
    end = fr.get("sine_at_end", -1)
    return np.array([e == ptr or e == end for e in range(n_env)], bool)


def adjust(Qh, t, sig_sb, est, lim, noise_sb=None, transient=None,
           sine_idx=None, sine_area=None, with_sine=False):
    """Pseudocodes 94..101.  -> (gain, adjusted noise level) per subband/env,
    and with with_sine the adjusted sine level too.

    transient (bool per envelope, transient_envelopes()): in those
    envelopes the gain leaves out the noise factor (Pseudocode 95: denom
    is not multiplied by 1 + scf_noise_sb) and the boost's denominator
    leaves out the noise level (Pseudocode 99). sine_idx / sine_area
    (aspx_tools.sine_idx_sb / sine_area, bool n_sb x n_env): the
    additional sinusoids (Pseudocodes 94, 95, 99, 101); None: none."""
    n_sb, n_env = est.shape
    if noise_sb is None:
        noise_sb = np.zeros_like(est)
    if transient is None:
        transient = np.zeros(n_env, bool)
    transient = np.asarray(transient, bool)[:n_env]
    gain = np.sqrt(sig_sb / (EPSILON + est) / (1.0 + noise_sb))
    if transient.any():
        gain[:, transient] = np.sqrt(sig_sb[:, transient]
                                     / (EPSILON + est[:, transient]))
    # Pseudocode 94: the noise level shares the band energy with the signal
    sig_noise_fact = sig_sb / (1.0 + noise_sb)
    noise_lev = np.sqrt(sig_noise_fact * noise_sb)
    sine_lev = np.zeros_like(est)
    if sine_idx is not None and sine_idx.any() or (
            sine_area is not None and sine_area.any()):
        sidx = np.asarray(sine_idx[:, :n_env], float)
        sine_lev = np.sqrt(sig_noise_fact * sidx)
        area = np.asarray(sine_area[:, :n_env], bool)
        # Pseudocode 95, the sine-area branch
        den = (EPSILON + est) * (1.0 + noise_sb)
        gain = np.where(area, np.sqrt(sig_sb * noise_sb / den), gain)

    # Pseudocode 96: per limiter subband group ceiling. Group i is the
    # subbands lim[i] .. lim[i+1] - 1 (the mapping loop's sb == sbg_lim
    # [sbg+1] - sbx rule); V1.3.1 sums to lim[i+1] - 2 only, which leaves
    # the top subband of each group out of its sums and unlimited --
    # corrected in V1.4.1 (2025), whose loops run to sbg_lim[sbg+1] - sbx
    maxg = np.zeros_like(gain)
    for i in range(len(lim) - 1):
        lo = max(lim[i] - t["sbx"], 0)
        hi = min(lim[i + 1] - t["sbx"], n_sb)
        if hi <= lo:
            continue
        nom = sig_sb[lo:hi].sum(axis=0)
        den = EPSILON0 + est[lo:hi].sum(axis=0)
        maxg[lo:hi] = np.minimum(np.sqrt(nom / den) * LIM_GAIN, MAX_SIG_GAIN)
    maxg[maxg == 0] = MAX_SIG_GAIN

    lim_gain = np.minimum(gain, maxg)            # Pseudocode 98
    # Pseudocode 97: the noise is trimmed in proportion to the gain limiting
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(gain > 0, maxg / np.maximum(gain, 1e-30), 1.0)
    noise_lim = np.minimum(noise_lev, noise_lev * ratio)

    boost = np.ones_like(gain)                   # Pseudocode 99 / 100
    # the noise level counts where there is no sinusoid and the envelope
    # is not a transient one
    use_noise = (sine_lev == 0) & ~transient[None, :]
    for i in range(len(lim) - 1):
        lo = max(lim[i] - t["sbx"], 0)
        hi = min(lim[i + 1] - t["sbx"], n_sb)
        if hi <= lo:
            continue
        nom = EPSILON0 + sig_sb[lo:hi].sum(axis=0)
        den = EPSILON0 + (est[lo:hi] * lim_gain[lo:hi] ** 2).sum(axis=0)
        if with_sine:
            den = den + (sine_lev[lo:hi] ** 2).sum(axis=0)
            den = den + np.where(use_noise[lo:hi], noise_lim[lo:hi] ** 2,
                                 0.0).sum(axis=0)
        else:
            den = den + np.where(transient, 0.0,
                                 (noise_lim[lo:hi] ** 2).sum(axis=0))
        boost[lo:hi] = np.minimum(np.sqrt(nom / den), MAX_BOOST_FACT)
    if with_sine:                                # Pseudocode 101
        return lim_gain * boost, noise_lim * boost, sine_lev * boost
    return lim_gain * boost, noise_lim * boost


def noise_table(path=None):
    """Table D.2 -- ASPX_NOISE[512][2] in the spec C file, 512 complex.

    Each entry has unit magnitude with random phase ("average energy of 1"),
    which is checkable and is the gate on the transcription.
    """
    import re
    # E108: fall back to the banked coefficients when the ETSI source is
    # absent -- which is every clone, since the document is not ours to
    # redistribute. Without this the A-SPX high-frequency regeneration raised
    # FileNotFoundError, the receiver's audio worker counted a silent `self_hf_fail`, and the
    # listener got core-band-only audio that decoded and played and was
    # quietly missing its top octave.
    bank = os.path.join(HERE, "tables", "ac4_aspx_noise.json")
    if path is None and not os.path.exists(H.DEFAULT_C)             and os.path.exists(bank):
        import json
        d = json.load(open(bank, encoding="utf-8"))
        a = np.asarray(d["re"], float) + 1j * np.asarray(d["im"], float)
        if a.size != 512:
            raise ValueError(f"banked ASPX_NOISE has {a.size}, expected 512")
        return a
    src = open(path or H.DEFAULT_C, encoding="utf-8",
               errors="replace").read()
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    m = re.search(r"ASPX_NOISE\s*\[\s*512\s*\]\s*\[\s*2\s*\]"
                  r"\s*=\s*\{(.*?)\}\s*;", src, re.S)
    v = [float(x) for x in
         re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))]
    a = np.asarray(v, float).reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1]


def noise_envelopes(dat, t, prev):
    """Reconstruct + dequantise the NOISE envelopes.

    Same delta scheme as the signal envelopes (Pseudocode 80) but over
    num_sbg_noise groups, then Pseudocode 83:

        scf_noise_sbg = 2 ** (NOISE_FLOOR_OFFSET - qscf_noise_sbg)

    Note the SIGN: a larger transmitted index means LESS noise.
    """
    delta = 2 if (dat["ch"] == 1 and dat["balance"] == 1) else 1
    dirs = dat.get("dir_noise") or [0] * len(dat["noise"])
    out, prev_q = [], prev
    for e, vals in enumerate(dat["noise"]):
        n = t["n_noise"]
        q = [0] * n
        if e < len(dirs) and dirs[e] == 0:            # FREQ
            acc = 0
            for sbg in range(min(n, len(vals))):
                acc += delta * vals[sbg]
                q[sbg] = acc
        else:                                          # TIME
            for sbg in range(min(n, len(vals))):
                base = prev_q[sbg] if (prev_q and sbg < len(prev_q)) else 0
                q[sbg] = base + delta * vals[sbg]
        out.append([2.0 ** (NOISE_FLOOR_OFFSET - x) for x in q])
        prev_q = q
    return out, prev_q


def map_noise(scf_noise_env, t, n_sb, n_env):
    """Pseudocode 91, the noise half: subband groups -> QMF subbands."""
    out = np.zeros((n_sb, n_env))
    for e in range(n_env):
        src = scf_noise_env[min(e, len(scf_noise_env) - 1)]
        for sbg in range(min(t["n_noise"], len(src))):
            lo = max(t["noise"][sbg] - t["sbx"], 0)
            hi = min(t["noise"][sbg + 1] - t["sbx"], n_sb)
            if hi > lo:
                out[lo:hi, e] = src[sbg]
    return out
