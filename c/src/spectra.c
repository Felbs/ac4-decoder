/* SPDX-License-Identifier: Apache-2.0 */
/* Dequantisation into the packed spectrum, stereo processing (sap_mode 1,
 * 2 and full SAP, sap_mode 3) and ungrouping. Port of ac4/audio.py
 * (packed_spectrum, unmix, sap_alpha_q, sap_coeffs, ungroup);
 * TS 103 190-1 5.1.3, 5.3, Pseudocode 21, 25, 59.
 *
 * Noise fill: the Python decoder parses asf_snf_data but its stream path
 * never applies it (packed_spectrum does not use its rng argument), so
 * neither does this port. */
#include <math.h>
#include <string.h>

#include "ac4_internal.h"

#define SF_OFFSET 100
#define MAX_SPECTRAL_LINE 1e9

static double POW43[8192];
static double SF_GAIN[256];
static int tables_ready;

static void init_tables(void)
{
    int i;
    if (tables_ready)
        return;
    for (i = 0; i < 8192; i++)
        POW43[i] = pow((double)i, 4.0 / 3.0);
    for (i = 0; i < 256; i++)
        SF_GAIN[i] = pow(2.0, 0.25 * (double)(i - SF_OFFSET));
    tables_ready = 1;
}

/* called once from ac4dec_open (not thread-racy in practice: the values
   written are identical on every call) */
void ac4_spectra_init(void)
{
    init_tables();
}

int ac4_packed_spectrum(const ac4_chan *ch, double *spec, jmp_buf *jb)
{
    const ac4_framing *f = ch->fr;
    int total = ch->n_lines, g, sfb, k;
    for (k = 0; k < total; k++)
        spec[k] = 0.0;
    for (g = 0; g < f->num_groups; g++) {
        const ac4t_sfb *t = f->gtab[g];
        int nw = f->nwin[g], base = f->gbase[g];
        int m = f->gmax[g];
        for (sfb = 0; sfb < m; sfb++) {
            int sf = ch->sfs[g][sfb], lo, hi;
            double gain;
            if (sf == AC4_SF_NONE || sfb + 1 > t->num_sfb)
                continue;
            lo = base + t->off[sfb] * nw;
            hi = base + t->off[sfb + 1] * nw;
            if (hi > total)
                hi = total;
            if (hi <= lo)
                continue;
            gain = SF_GAIN[sf & 255];
            for (k = lo; k < hi; k++) {
                int32_t q = ch->lines[k];
                double v;
                if (q == 0) {
                    spec[k] = 0.0;
                    continue;
                }
                v = q > 0 ? POW43[q] : -POW43[-q];
                v = v * gain;
                if (!(fabs(v) <= MAX_SPECTRAL_LINE))
                    longjmp(*jb, AC4E_LEVEL);
                spec[k] = v;
            }
        }
    }
    return total;
}

/* sap_alpha_q for one group sequence: alpha_q[g][sfb] or AC4_SF_NONE. */
static void sap_alpha_q(const ac4_stereo *st, const ac4_framing *f,
                        int out[AC4_MAX_WIN][AC4_MAX_SFB])
{
    int prev_q[AC4_MAX_SFB], q[AC4_MAX_SFB];
    int g, sfb, max_sfb_prev = f->num_groups ? f->gmax[0] : 0;
    for (g = 0; g < f->num_groups; g++) {
        int m = f->gmax[g];
        for (sfb = 0; sfb < m; sfb++) {
            q[sfb] = 0;
            out[g][sfb] = AC4_SF_NONE;
        }
        for (sfb = 0; sfb < m; sfb++) {
            if (!st->coeff_used[g][sfb])
                continue;
            if (sfb % 2) {
                q[sfb] = q[sfb - 1];
            } else {
                int delta = st->dpcm[g][sfb] - 60;
                int code_delta = (g == 0 || m != max_sfb_prev)
                    ? 0 : st->delta_code_time;
                if (code_delta)
                    q[sfb] = prev_q[sfb] + delta;
                else if (sfb == 0)
                    q[sfb] = delta;
                else
                    q[sfb] = q[sfb - 2] + delta;
            }
            out[g][sfb] = q[sfb];
        }
        for (sfb = 0; sfb < m; sfb++)
            prev_q[sfb] = q[sfb];
        max_sfb_prev = m;
    }
}

void ac4_unmix(double *m, double *s, int n, const ac4_stereo *st,
               const ac4_framing *f)
{
    int g, sfb, k;
    if (st->sap_mode == 2) {
        for (k = 0; k < n; k++) {
            double a = m[k], b = s[k];
            m[k] = a + b;
            s[k] = a - b;
        }
        return;
    }
    if (st->sap_mode == 1 || st->sap_mode == 3) {
        int alpha[AC4_MAX_WIN][AC4_MAX_SFB];
        if (st->sap_mode == 3)
            sap_alpha_q(st, f, alpha);
        for (g = 0; g < f->num_groups; g++) {
            const ac4t_sfb *t = f->gtab[g];
            int nw = f->nwin[g], base = f->gbase[g];
            for (sfb = 0; sfb < f->gmax[g]; sfb++) {
                int lo, hi;
                double a, b, c, d;
                if (sfb + 1 >= t->num_sfb + 1)
                    continue;
                if (st->sap_mode == 1) {
                    if (!st->ms_used[g][sfb])
                        continue;
                    a = 1.0; b = 1.0; c = 1.0; d = -1.0;
                } else {
                    int q = alpha[g][sfb];
                    if (q == AC4_SF_NONE)
                        continue;          /* identity */
                    a = 1.0 + 0.1 * q;
                    b = 1.0;
                    c = 1.0 - 0.1 * q;
                    d = -1.0;
                }
                lo = base + t->off[sfb] * nw;
                hi = base + t->off[sfb + 1] * nw;
                if (hi > n)
                    hi = n;
                for (k = lo; k < hi; k++) {
                    double x = m[k], y = s[k];
                    if (st->sap_mode == 1) {
                        m[k] = x + y;
                        s[k] = x - y;
                    } else {
                        m[k] = a * x + b * y;
                        s[k] = c * x + d * y;
                    }
                }
            }
        }
    }
    /* sap_mode 0 or stereo processing off: unchanged */
}

void ac4_ungroup(const double *packed, int n, const ac4_framing *f,
                 double *win_buf, int *lens)
{
    int w, g, sfb, k = 0, win = 0, off[AC4_MAX_WIN + 1];
    off[0] = 0;
    for (w = 0; w < f->num_windows; w++) {
        lens[w] = f->glen[f->w2g[w]];
        off[w + 1] = off[w] + lens[w];
    }
    memset(win_buf, 0, sizeof(double) * (size_t)off[f->num_windows]);
    for (g = 0; g < f->num_groups; g++) {
        const ac4t_sfb *t = f->gtab[g];
        int nw = f->nwin[g];
        int m = f->gmax[g] < t->num_sfb ? f->gmax[g] : t->num_sfb;
        for (sfb = 0; sfb < m; sfb++) {
            int lo = t->off[sfb], hi = t->off[sfb + 1];
            for (w = 0; w < nw; w++) {
                int len = hi - lo;
                if (k + len > n)
                    return;
                memcpy(win_buf + off[win + w] + lo, packed + k,
                       sizeof(double) * (size_t)len);
                k += len;
            }
        }
        win += nw;
    }
}
