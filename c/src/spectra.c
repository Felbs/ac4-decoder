/* SPDX-License-Identifier: Apache-2.0 */
/* Dequantisation into the packed spectrum, stereo processing (sap_mode 1,
 * 2 and full SAP, sap_mode 3) and ungrouping. Port of ac4/audio.py
 * (packed_spectrum, noise_fill, SnfRandom, unmix, sap_alpha_q, sap_coeffs,
 * ungroup); TS 103 190-1 5.1.3, 5.1.4, 5.3, Pseudocode 21-25, 57, 59. */
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

/* ------------------------------------------------------------ noise fill */
void ac4_snf_reset(ac4_snf_rng *r, int sequence_counter)
{
    long start = 255L * (long)(sequence_counter % 256);
    long a = start % 255, b = (start / 255) % 256, state, cur;
    state = (a + 1) * ((a + 2) + 2 * b) / 2 + 255 * b * (255 + b) / 2;
    if (start % 130560 >= 65280)
        state += 128;
    cur = a * (a + 1) / 2 + b * 32386;
    r->a = (unsigned)(a & 255);
    r->b = (unsigned)(b & 255);
    r->state = (unsigned)(state % 256);
    r->cur = (unsigned)(cur % 256);
}

/* Pseudocode 57; the spec's `x = x++` statements are increments */
double ac4_snf_next(ac4_snf_rng *r)
{
    float v = AC4T_RANDOM_NOISE[r->cur] + AC4T_RANDOM_NOISE[r->state];
    r->a = (r->a + 1) & 255;
    r->state = (r->state + 1) & 255;
    if (r->a == 255) {
        r->cur = (r->cur + 1) & 255;
        r->b = (r->b + 1) & 255;
        r->a = 0;
    }
    r->cur = (r->cur + r->a) & 255;
    r->state = (r->state + r->b + r->a) & 255;
    return (double)v;
}

/* mean-square level of a band, log2 (Pseudocodes 22, 23): numpy's sum of
   the squares, as the Python computes it */
static int band_level(const double *spec, int lo, int hi, double *lev)
{
    double sq[AC4_MAX_FRAME], e;
    int k;
    for (k = lo; k < hi; k++)
        sq[k - lo] = spec[k] * spec[k];
    e = ac4_np_sum(sq, hi - lo);
    if (!(e > 0))
        return 0;
    *lev = 1.44269504 * log(e / (double)(hi - lo));
    return 1;
}

void ac4_noise_fill(const ac4_chan *ch, double *spec, int n,
                    ac4_snf_rng *r, jmp_buf *jb)
{
    const ac4_framing *f = ch->fr;
    double prev = -1000.0;
    int g, sfb, k, found = 0;
    if (!ch->has_snf)
        return;
    /* Pseudocode 22: the first band with energy sets the reference */
    for (g = 0; spec && g < f->num_groups && !found; g++) {
        const ac4t_sfb *t = f->gtab[g];
        int nw = f->nwin[g], base = f->gbase[g];
        int m = f->gmax[g] < t->num_sfb ? f->gmax[g] : t->num_sfb;
        for (sfb = 0; sfb < m; sfb++) {
            int lo = base + t->off[sfb] * nw, hi = base + t->off[sfb + 1] * nw;
            if (hi > n)
                hi = n;
            if (hi > lo && band_level(spec, lo, hi, &prev)) {
                found = 1;
                break;
            }
        }
    }
    /* Pseudocode 23 */
    for (g = 0; g < f->num_groups; g++) {
        const ac4t_sfb *t = f->gtab[g];
        int nw = f->nwin[g], base = f->gbase[g];
        int m = f->gmax[g] < t->num_sfb ? f->gmax[g] : t->num_sfb;
        for (sfb = 0; sfb < m; sfb++) {
            int lo = base + t->off[sfb] * nw, hi = base + t->off[sfb + 1] * nw;
            int d = ch->snf[g][sfb];
            double amp;
            if (spec && hi > n)
                hi = n;
            if (hi <= lo)
                continue;
            if (spec) {
                if (band_level(spec, lo, hi, &prev))
                    continue;
            } else if (d == AC4_SF_NONE) {
                continue;               /* a band with energy */
            }
            if (d == AC4_SF_NONE || d == -17)
                continue;               /* -17: no noise, level held */
            prev = prev + (double)d;
            if (!spec) {
                for (k = lo; k < hi; k++)
                    ac4_snf_next(r);
                continue;
            }
            amp = pow(2.0, 0.5 * prev);
            for (k = lo; k < hi; k++) {
                double v = ac4_snf_next(r) * amp;
                if (!(fabs(v) <= MAX_SPECTRAL_LINE))
                    longjmp(*jb, AC4E_LEVEL);
                spec[k] = v;
            }
        }
    }
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

/* ------------------------------------------- channel data elements */
/* chel_matsel -> pair 0 inputs, pair 1 inputs, outputs O0..O2 (as
   ac4/multichannel.py MATSEL3): 0..2 the input tracks, 3 P, 4 Q (pair
   0's outputs), 5 R, 6 S (pair 1's) */
enum { MP = 3, MQ = 4, MR = 5, MS = 6 };
static const int MATSEL3[12][7] = {
    { 0, 1, MP, 2, MR, MQ, MS }, { 1, 0, MP, 2, MQ, MR, MS },
    { 0, 2, MP, 1, MR, MS, MQ }, { 1, 2, 0, MQ, MR, MP, MS },
    { 0, 2, 1, MQ, MP, MR, MS }, { 2, 1, 0, MQ, MR, MS, MP },
    { 1, 0, 2, MQ, MS, MP, MR }, { 0, 1, 2, MQ, MP, MS, MR },
    { 2, 0, 1, MQ, MS, MR, MP }, { 2, 1, MP, 0, MS, MQ, MR },
    { 2, 0, MP, 1, MQ, MS, MR }, { 1, 2, MP, 0, MS, MR, MQ },
};

/* the three-track matrix in place on tr[0..2] (Table 178): each pair
   works in place, so a name is the buffer that holds it */
static void matrix3(double **tr, int n, int matsel, const ac4_stereo *st,
                    const ac4_framing *f, double *tmp)
{
    const int *m = MATSEL3[matsel];
    double *buf[7];
    int i;
    buf[0] = tr[0];
    buf[1] = tr[1];
    buf[2] = tr[2];
    ac4_unmix(buf[m[0]], buf[m[1]], n, &st[0], f);
    buf[MP] = buf[m[0]];
    buf[MQ] = buf[m[1]];
    ac4_unmix(buf[m[2]], buf[m[3]], n, &st[1], f);
    buf[MR] = buf[m[2]];
    buf[MS] = buf[m[3]];
    for (i = 0; i < 3; i++)
        memcpy(tmp + (size_t)i * AC4_MAX_FRAME, buf[m[4 + i]],
               sizeof(double) * (size_t)n);
    for (i = 0; i < 3; i++)
        memcpy(tr[i], tmp + (size_t)i * AC4_MAX_FRAME,
               sizeof(double) * (size_t)n);
}

void ac4_matrix(double **tr, int n, int kind, int matsel,
                const ac4_stereo *st, const ac4_framing *f)
{
    double tmp[3 * AC4_MAX_FRAME];
    if (kind == 3) {
        matrix3(tr, n, matsel, st, f, tmp);
    } else if (kind == 4) {
        /* (P0, Q0) = pair0(I0, I1), (P1, Q1) = pair1(I2, I3),
           (O0, O2) = pair2(P0, P1), (O1, O3) = pair3(Q0, Q1) */
        ac4_unmix(tr[0], tr[1], n, &st[0], f);
        ac4_unmix(tr[2], tr[3], n, &st[1], f);
        ac4_unmix(tr[0], tr[2], n, &st[2], f);
        ac4_unmix(tr[1], tr[3], n, &st[3], f);
    } else if (kind == 5) {
        /* (X, Y, Z) = the three-track matrix, (U, V) = pair2(I3, I4),
           (O0, O3) = pair3(X, U), (O1, O4) = pair4(Y, V), O2 = Z */
        matrix3(tr, n, matsel, st, f, tmp);
        ac4_unmix(tr[3], tr[4], n, &st[2], f);
        ac4_unmix(tr[0], tr[3], n, &st[3], f);
        ac4_unmix(tr[1], tr[4], n, &st[4], f);
    } else {
        ac4_unmix(tr[0], tr[1], n, &st[0], f);
    }
}

void ac4_unmix_windows(double *wa, double *wb, const int *lens, int nwin,
                       const ac4_stereo *st, const ac4_framing *f)
{
    int alpha[AC4_MAX_WIN][AC4_MAX_SFB];
    int w, off = 0, k, sfb;
    if (st->sap_mode <= 0)
        return;
    if (st->sap_mode == 3)
        sap_alpha_q(st, f, alpha);
    for (w = 0; w < nwin; off += lens[w], w++) {
        double *x = wa + off, *y = wb + off;
        int g = f->w2g[w];
        const ac4t_sfb *t;
        if (st->sap_mode == 2) {
            for (k = 0; k < lens[w]; k++) {
                double a = x[k], b = y[k];
                x[k] = a + b;
                y[k] = a - b;
            }
            continue;
        }
        if (g >= f->num_groups)
            continue;
        t = f->gtab[g];
        for (sfb = 0; sfb < f->gmax[g] && sfb < t->num_sfb; sfb++) {
            double a, b, c, d;
            int lo = t->off[sfb], hi = t->off[sfb + 1];
            if (st->sap_mode == 1) {
                if (!st->ms_used[g][sfb])
                    continue;
                a = 1.0; b = 1.0; c = 1.0; d = -1.0;
            } else {
                int q = alpha[g][sfb];
                if (q == AC4_SF_NONE)
                    continue;
                a = 1.0 + 0.1 * q;
                b = 1.0;
                c = 1.0 - 0.1 * q;
                d = -1.0;
            }
            if (hi > lens[w])
                hi = lens[w];
            for (k = lo; k < hi; k++) {
                double u = x[k], v = y[k];
                x[k] = a * u + b * v;
                y[k] = c * u + d * v;
            }
        }
    }
}
