/* SPDX-License-Identifier: Apache-2.0 */
/* A-SPX rendering of one channel pair, streaming: the 64-band complex QMF
 * bank (ac4/qmf.py and its C kernel ac4/qmf_kernel.c, same loop order),
 * HF generation (ac4/aspx_hfgen.py), envelope reconstruction
 * (ac4/aspx_env.py), dequantisation, estimation and adjustment
 * (ac4/aspx_envadj.py), and the per-frame loop of
 * ac4/aspx_render.apply_hf_pair; TS 103 190-1 5.7.3, 5.7.4, 5.7.6.
 *
 * The Python renders a whole decode pass at once; this renders frame by
 * frame with the same arithmetic. With nts = num_qmf_timeslots, tia =
 * num_ts_in_ats and TS_OFFSET = nts - ts_offset_hfgen (aspx_bands.
 * geometry; 24, 2 and 18 at frame length 1536), a frame's envelopes cover
 * QMF slots [nts j + TS_OFFSET + tia b0, nts j + TS_OFFSET + tia bn) with
 * 0 <= b0 and bn <= num_aspx_timeslots + 3 (the aspx.borders rule), i.e.
 * at most to the end of frame j + 1 at every frame length of Table 192,
 * so frame j is adjusted and synthesised once core frame j + 1 has been
 * analysed.
 *
 * A slot is HF-generated only when the frame its envelope belongs to
 * (frame floor((slot - TS_OFFSET) / nts)) has A-SPX data or was concealed
 * with a configuration known: the slots of frames from before the
 * stream's first aspx_config keep the core alone (the Python does the
 * same). */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ac4_internal.h"

#define NSB 64
#define AC4_PI 3.14159265358979323846   /* np.pi */
#define NWIN 640
#define NOISE_FLOOR_OFFSET 6
#define PAN_OFFSET 12
#define EPSILON 1.0
#define EPSILON0 1e-12
#define LIM_GAIN 1.41254
#define MAX_SIG_GAIN 1e5
#define MAX_BOOST_FACT 1.584893192

/* Stored transposed (analysis [n][sb], synthesis [sb][n]) so the matrix
   products run over contiguous output lanes; each output still sums its
   terms in the kernel's order, so the results are bit-identical. */
struct ac4_qmf_tables {
    double mre[2 * NSB][NSB], mim[2 * NSB][NSB];     /* analysis M^T */
    double nre[NSB][2 * NSB], nim[NSB][2 * NSB];     /* synthesis N^T */
};

ac4_qmf_tables *ac4_qmf_tables_new(void)
{
    ac4_qmf_tables *q = (ac4_qmf_tables *)malloc(sizeof(*q));
    int sb, n;
    if (!q)
        return NULL;
    /* qmf.analysis_matrix: exp(j pi/128 (sb + 0.5)(2n - 1)) */
    for (sb = 0; sb < NSB; sb++)
        for (n = 0; n < 2 * NSB; n++) {
            double ph = (AC4_PI / 128.0 * ((double)sb + 0.5))
                        * (double)(2 * n - 1);
            q->mre[n][sb] = cos(ph);
            q->mim[n][sb] = sin(ph);
        }
    /* qmf.synthesis_matrix(255): exp(j pi/128 (sb + 0.5)(2n - 255)) / 64 */
    for (n = 0; n < 2 * NSB; n++)
        for (sb = 0; sb < NSB; sb++) {
            double ph = (AC4_PI / 128.0 * ((double)sb + 0.5))
                        * (double)(2 * n - 255);
            q->nre[sb][n] = cos(ph) / 64.0;
            q->nim[sb][n] = sin(ph) / 64.0;
        }
    return q;
}

void ac4_qmf_tables_free(ac4_qmf_tables *q)
{
    free(q);
}

/* numpy's pairwise summation (np.add.reduce over a contiguous run) */
double ac4_np_sum(const double *a, long n)
{
    if (n < 8) {
        double r = 0.0;
        long i;
        for (i = 0; i < n; i++)
            r += a[i];
        return r;
    } else if (n <= 128) {
        double r[8], res;
        long i;
        int j;
        for (j = 0; j < 8; j++)
            r[j] = a[j];
        for (i = 8; i < n - (n % 8); i += 8)
            for (j = 0; j < 8; j++)
                r[j] += a[i + j];
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++)
            res += a[i];
        return res;
    } else {
        long n2 = n / 2;
        n2 -= n2 % 8;
        return ac4_np_sum(a, n2) + ac4_np_sum(a + n2, n - n2);
    }
}

/* numpy minimum / maximum: NaN propagates */
static double np_min(double a, double b)
{
    if (a != a)
        return a;
    if (b != b)
        return b;
    return a <= b ? a : b;
}

static double np_max(double a, double b)
{
    if (a != a)
        return a;
    if (b != b)
        return b;
    return a >= b ? a : b;
}

/* ------------------------------------------------------------ QMF */
/* one slot of analysis (qmf_kernel.qmf_analyse_f64) */
static void qmf_analyse_slot(const ac4_qmf_tables *q, double *filt,
                             const double *x, double *ore, double *oim)
{
    const double *w = AC4T_QWIN;
    double z[NWIN], u[2 * NSB];
    int i, k, sb;
    memmove(filt + NSB, filt, sizeof(double) * (NWIN - NSB));
    for (i = 0; i < NSB; i++)
        filt[i] = x[NSB - 1 - i];
    for (i = 0; i < NWIN; i++)
        z[i] = filt[i] * w[i];
    for (i = 0; i < 2 * NSB; i++)
        u[i] = z[i];
    for (k = 1; k < 5; k++)
        for (i = 0; i < 2 * NSB; i++)
            u[i] += z[k * 2 * NSB + i];
    {
        double ar[NSB], ai[NSB];
        for (sb = 0; sb < NSB; sb++)
            ar[sb] = ai[sb] = 0.0;
        for (i = 0; i < 2 * NSB; i++) {
            const double *mr = q->mre[i], *mi = q->mim[i];
            double ui = u[i];
            for (sb = 0; sb < NSB; sb++)
                ar[sb] += mr[sb] * ui;
            for (sb = 0; sb < NSB; sb++)
                ai[sb] += mi[sb] * ui;
        }
        memcpy(ore, ar, sizeof(ar));
        memcpy(oim, ai, sizeof(ai));
    }
}

/* one slot of synthesis (qmf_kernel.qmf_synthesise_f64) */
static void qmf_synth_slot(const ac4_qmf_tables *q, double *filt,
                           const double *qre, const double *qim, double *o)
{
    const double *w = AC4T_QWIN;
    double g[NWIN];
    int n, i, r;
    memmove(filt + 2 * NSB, filt, sizeof(double) * (10 * 2 * NSB - 2 * NSB));
    {
        double acc[2 * NSB];
        int sb;
        for (n = 0; n < 2 * NSB; n++)
            acc[n] = 0.0;
        for (sb = 0; sb < NSB; sb++) {
            const double *nr = q->nre[sb], *ni = q->nim[sb];
            double a = qre[sb], b = qim[sb];
            for (n = 0; n < 2 * NSB; n++)
                acc[n] += nr[n] * a - ni[n] * b;
        }
        memcpy(filt, acc, sizeof(acc));
    }
    for (n = 0; n < 5; n++) {
        for (i = 0; i < NSB; i++)
            g[128 * n + i] = filt[256 * n + i];
        for (i = 0; i < NSB; i++)
            g[128 * n + NSB + i] = filt[256 * n + 192 + i];
    }
    for (i = 0; i < NSB; i++)
        o[i] = g[i] * w[i];
    for (r = 1; r < 10; r++)
        for (i = 0; i < NSB; i++)
            o[i] += g[r * NSB + i] * w[r * NSB + i];
}

/* ------------------------------------------------------------ state */
void ac4_aspx_pair_reset(ac4_aspx_pair *p)
{
    ac4_aspx_pair_init(p, AC4_DEFAULT_FRAME);
}

/* Table 192 / Pseudocode 75a: the time grid at frame length fl */
void ac4_aspx_pair_init(ac4_aspx_pair *p, int fl)
{
    int hf = fl >= 1536 ? 6 : 3;
    memset(p, 0, sizeof(*p));
    p->sine_idx[0] = p->sine_idx[1] = -1;
    p->fl = fl;
    p->nts = fl / 64;
    p->tia = fl >= 1536 ? 2 : 1;
    p->nats = p->nts / p->tia;
    p->tso = p->nts - hf;
}

#define QI(ts) ((int)((ts) % AC4_QRING))

static int effective(const ac4_aspx_frame *f)
{
    return f->kind != AC4_GRP_NONE && f->has_tab;
}

/* ------------------------------------------------------------ tools
 * aspx_tools.py: TNA (Pseudocodes 86-88, Table 195), pre-flattening
 * (Pseudocode 85), HF signal creation (Pseudocode 89), sinusoids
 * (Pseudocodes 92-94, 104, 105), interleaved waveform coding (5.7.6.5.3).
 * An interval's slot ts is analysis slot A0 + ts, A0 = nts f + TS_OFFSET. */
static const double NEW_CHIRP[4][4] = {
    { 0.0, 0.6, 0.9, 0.98 }, { 0.6, 0.75, 0.9, 0.98 },
    { 0.0, 0.75, 0.9, 0.98 }, { 0.0, 0.75, 0.9, 0.98 } };
static const double SINE_RE[4] = { 1.0, 0.0, -1.0, 0.0 };
static const double SINE_IM[4] = { 0.0, 1.0, 0.0, -1.0 };

/* analysis slot ts of channel c, 0 outside [0, nts) */
static void ana_at(const ac4_aspx_pair *p, int c, long long ts,
                   long long nts, int sb, double *re, double *im)
{
    if (ts < 0 || ts >= nts) {
        *re = *im = 0.0;
        return;
    }
    *re = p->are[c][QI(ts)][sb];
    *im = p->aim[c][QI(ts)][sb];
}

/* Pseudocodes 86, 87 for subbands below sba */
static void tna_alpha(const ac4_aspx_pair *p, int c, int sba, long long a0,
                      long long nts, double *a0r, double *a0i, double *a1r,
                      double *a1i)
{
    int sb, hf = p->fl >= 1536 ? 6 : 3;
    for (sb = 0; sb < sba; sb++) {
        double cr[3][3], ci[3][3], dr, di, nr, ni, m12;
        int i, j, m;
        for (i = 0; i < 3; i++)
            for (j = 0; j < 3; j++)
                cr[i][j] = ci[i][j] = 0.0;
        for (m = 0; m < p->nts + hf; m += 2) {
            double xr[3], xi[3];
            for (i = 0; i < 3; i++)
                ana_at(p, c, a0 + m - 2 * i, nts, sb, &xr[i], &xi[i]);
            /* cov[i][j] += x[i] * conj(x[j]) for (0,1) (0,2) (1,1) (1,2)
               (2,2) */
            cr[0][1] += xr[0] * xr[1] + xi[0] * xi[1];
            ci[0][1] += xi[0] * xr[1] - xr[0] * xi[1];
            cr[0][2] += xr[0] * xr[2] + xi[0] * xi[2];
            ci[0][2] += xi[0] * xr[2] - xr[0] * xi[2];
            cr[1][1] += xr[1] * xr[1] + xi[1] * xi[1];
            ci[1][1] += xi[1] * xr[1] - xr[1] * xi[1];
            cr[1][2] += xr[1] * xr[2] + xi[1] * xi[2];
            ci[1][2] += xi[1] * xr[2] - xr[1] * xi[2];
            cr[2][2] += xr[2] * xr[2] + xi[2] * xi[2];
            ci[2][2] += xi[2] * xr[2] - xr[2] * xi[2];
        }
        /* denom = c22 c11 - |c12|^2 / (1 + EPSILON_INV) */
        m12 = hypot(cr[1][2], ci[1][2]);
        dr = cr[2][2] * cr[1][1] - ci[2][2] * ci[1][1];
        di = cr[2][2] * ci[1][1] + ci[2][2] * cr[1][1];
        dr = dr - m12 * m12 * (1.0 / (1.0 + 9.5367431640625e-07));
        a1r[sb] = a1i[sb] = 0.0;
        if (!(dr == 0.0 && di == 0.0)) {
            /* (c01 c12 - c02 c11) / denom */
            double pr = (cr[0][1] * cr[1][2] - ci[0][1] * ci[1][2])
                        - (cr[0][2] * cr[1][1] - ci[0][2] * ci[1][1]);
            double pi = (cr[0][1] * ci[1][2] + ci[0][1] * cr[1][2])
                        - (cr[0][2] * ci[1][1] + ci[0][2] * cr[1][1]);
            double den = dr * dr + di * di;
            a1r[sb] = (pr * dr + pi * di) / den;
            a1i[sb] = (pi * dr - pr * di) / den;
        }
        a0r[sb] = a0i[sb] = 0.0;
        if (!(cr[1][1] == 0.0 && ci[1][1] == 0.0)) {
            /* (-c01 + a1 conj(c12)) / c11 */
            double tr_ = a1r[sb] * cr[1][2] + a1i[sb] * ci[1][2];
            double ti = a1i[sb] * cr[1][2] - a1r[sb] * ci[1][2];
            double den = cr[1][1] * cr[1][1] + ci[1][1] * ci[1][1];
            nr = -cr[0][1] + tr_;
            ni = -ci[0][1] + ti;
            a0r[sb] = (nr * cr[1][1] + ni * ci[1][1]) / den;
            a0i[sb] = (ni * cr[1][1] - nr * ci[1][1]) / den;
        }
        if (hypot(a0r[sb], a0i[sb]) >= 4.0 || hypot(a1r[sb], a1i[sb]) >= 4.0)
            a0r[sb] = a0i[sb] = a1r[sb] = a1i[sb] = 0.0;
    }
}

/* Pseudocode 88 */
static void chirps(ac4_aspx_pair *p, int c, const uint8_t *tna, int n,
                   double *out)
{
    int g;
    if (p->n_tna[c] != n) {
        for (g = 0; g < n; g++) {
            p->chirp[c][g] = 0.0;
            p->tna_prev[c][g] = 0;
        }
    }
    for (g = 0; g < n; g++) {
        double nc = NEW_CHIRP[p->tna_prev[c][g]][tna[g]];
        if (nc < p->chirp[c][g])
            nc = 0.75 * nc + 0.25 * p->chirp[c][g];
        else
            nc = 0.90625 * nc + 0.09375 * p->chirp[c][g];
        out[g] = nc < 0.015625 ? 0.0 : nc;
    }
    for (g = 0; g < n; g++) {
        p->chirp[c][g] = out[g];
        p->tna_prev[c][g] = tna[g];
    }
    p->n_tna[c] = n;
}

/* aspx_tools.polyfit3: the normal equations, Gaussian elimination with
   partial pivoting; -> coef[0..3], highest power first */
static void polyfit3(int n, const double *x, const double *y, double *coef)
{
    double A[4][4], v[4], cf[4];
    int r, c, i, k;
    for (r = 0; r < 4; r++) {
        for (c = 0; c < 4; c++) {
            double s = 0.0;
            for (i = 0; i < n; i++)
                s += pow(x[i], (double)(r + c));
            A[r][c] = s;
        }
        {
            double s = 0.0;
            for (i = 0; i < n; i++)
                s += pow(x[i], (double)r) * y[i];
            v[r] = s;
        }
    }
    for (c = 0; c < 4; c++) {
        int pv = c;
        double best = fabs(A[c][c]);
        for (r = c + 1; r < 4; r++)
            if (fabs(A[r][c]) > best) {
                best = fabs(A[r][c]);
                pv = r;
            }
        if (pv != c) {
            for (k = 0; k < 4; k++) {
                double tmp = A[c][k];
                A[c][k] = A[pv][k];
                A[pv][k] = tmp;
            }
            {
                double tmp = v[c];
                v[c] = v[pv];
                v[pv] = tmp;
            }
        }
        if (A[c][c] == 0.0)
            continue;
        for (r = c + 1; r < 4; r++) {
            double f = A[r][c] / A[c][c];
            for (k = c; k < 4; k++)
                A[r][k] = A[r][k] - f * A[c][k];
            v[r] = v[r] - f * v[c];
        }
    }
    for (r = 3; r >= 0; r--) {
        double s = v[r];
        for (k = r + 1; k < 4; k++)
            s = s - A[r][k] * cf[k];
        cf[r] = A[r][r] != 0.0 ? s / A[r][r] : 0.0;
    }
    for (r = 0; r < 4; r++)
        coef[r] = cf[3 - r];
}

/* Pseudocode 85 over analysis slots [ta, tz) */
static void preflat_gain(const ac4_aspx_pair *p, int c, int sbx,
                         long long ta, long long tz, double *gv)
{
    double pw[64], x[64], poly[4], mean = 0.0;
    int sb, k;
    long long ts;
    for (sb = 0; sb < sbx; sb++) {
        double e = 0.0;
        for (ts = ta; ts < tz; ts++) {
            double re = p->are[c][QI(ts)][sb], im = p->aim[c][QI(ts)][sb];
            e += re * re;
            e += im * im;
        }
        pw[sb] = 10.0 * log10(e / (double)(tz - ta > 1 ? tz - ta : 1) + 1.0);
        x[sb] = (double)sb;
    }
    for (sb = 0; sb < sbx; sb++)
        mean += pw[sb];
    mean /= sbx;
    polyfit3(sbx, x, pw, poly);
    for (sb = 0; sb < sbx; sb++) {
        double sl = 0.0;
        for (k = 3; k >= 0; k--)
            sl += pow(x[sb], (double)k) * poly[3 - k];
        gv[sb] = pow(10.0, (mean - sl) / 20.0);
    }
}

/* Pseudocode 89 over analysis slots [ta, tz) into the output ring */
static void generate(ac4_aspx_pair *p, int c, const ac4_aspx_tab *t,
                     long long ta, long long tz, long long nts,
                     const double *a0r, const double *a0i, const double *a1r,
                     const double *a1i, const double *chirp, const double *gv)
{
    int i, sb, sum_sb = 0, g = 0;
    long long ts;
    for (i = 0; i < t->n_patches; i++) {
        for (sb = 0; sb < t->patch_num_sb[i]; sb++) {
            int hi = t->sbx + sum_sb + sb, pp = t->patch_start_sb[i] + sb;
            double ch;
            if (g + 1 <= t->n_noise && t->noise[g + 1] == hi)
                g++;
            ch = g < t->n_noise ? chirp[g] : 0.0;
            for (ts = ta; ts < tz; ts++) {
                double xr = p->are[c][QI(ts)][pp], xi = p->aim[c][QI(ts)][pp];
                if (ch != 0.0) {
                    double r2, i2, r4, i4, cr, ci;
                    ana_at(p, c, ts - 2, nts, pp, &r2, &i2);
                    ana_at(p, c, ts - 4, nts, pp, &r4, &i4);
                    if (ts - 2 >= nts)
                        r2 = i2 = 0.0;
                    cr = ch * a0r[pp];
                    ci = ch * a0i[pp];
                    xr = xr + (cr * r2 - ci * i2);
                    xi = xi + (cr * i2 + ci * r2);
                    cr = ch * ch * a1r[pp];
                    ci = ch * ch * a1i[pp];
                    xr = xr + (cr * r4 - ci * i4);
                    xi = xi + (cr * i4 + ci * r4);
                }
                if (gv) {
                    double gg = 1.0 / gv[pp];
                    xr = xr * gg;
                    xi = xi * gg;
                }
                p->qre[c][QI(ts)][hi] = xr;
                p->qim[c][QI(ts)][hi] = xi;
            }
        }
        sum_sb += t->patch_num_sb[i];
    }
}

/* ------------------------------------------------------------ envelopes */
/* Pseudocode 80 (aspx_env.reconstruct) -> q[env][sbg], n[env] */
static void reconstruct(ac4_aspx_pair *p, int c, const ac4_aspx_chd *d,
                        const ac4_aspx_tab *t,
                        long long q[AC4_ASPX_MAX_ENV][AC4_ASPX_MAX_SBG],
                        int *nq)
{
    int delta = (d->ch == 1 && d->balance == 1) ? 2 : 1;
    int e, s;
    for (e = 0; e < d->fr.num_env; e++) {
        int res = d->fr.freq_res[e];
        int n_sbg = res ? t->n_hi : t->n_lo;
        int m = n_sbg < d->n_sig[e] ? n_sbg : d->n_sig[e];
        for (s = 0; s < n_sbg; s++)
            q[e][s] = 0;
        if (d->dir_sig[e] == 0) {
            long long acc = 0;
            for (s = 0; s < m; s++) {
                acc += (long long)delta * d->sig[e][s];
                q[e][s] = acc;
            }
        } else {
            for (s = 0; s < m; s++) {
                long long base = 0;
                if (!p->sprev_valid[c]) {
                    base = 0;
                } else if (res == p->sprev_res[c]) {
                    base = s < p->sprev_n[c] ? p->sprev[c][s] : 0;
                } else if (res == 0 && p->sprev_res[c] == 1) {
                    int i = s < t->n_lo + 1 ? t->low2high[s] : 0;
                    base = i < p->sprev_n[c] ? p->sprev[c][i] : 0;
                } else {
                    int i = s < t->n_hi ? t->high2low[s] : 0;
                    base = i < p->sprev_n[c] ? p->sprev[c][i] : 0;
                }
                q[e][s] = base + (long long)delta * d->sig[e][s];
            }
        }
        nq[e] = n_sbg;
        for (s = 0; s < n_sbg; s++)
            p->sprev[c][s] = q[e][s];
        p->sprev_n[c] = n_sbg;
        p->sprev_res[c] = res;
        p->sprev_valid[c] = 1;
    }
}

/* aspx_envadj.noise_qscf -> q[env][sbg] (n_noise each) */
static void noise_qscf(ac4_aspx_pair *p, int c, const ac4_aspx_chd *d,
                       const ac4_aspx_tab *t,
                       long long q[2][AC4_ASPX_MAX_NOISE])
{
    int delta = (d->ch == 1 && d->balance == 1) ? 2 : 1;
    int n = t->n_noise, e, s;
    for (e = 0; e < d->fr.num_noise; e++) {
        int m = n < d->n_noi[e] ? n : d->n_noi[e];
        for (s = 0; s < n; s++)
            q[e][s] = 0;
        if (d->dir_noise[e] == 0) {
            long long acc = 0;
            for (s = 0; s < m; s++) {
                acc += (long long)delta * d->noi[e][s];
                q[e][s] = acc;
            }
        } else {
            for (s = 0; s < m; s++) {
                long long base = 0;
                if (p->nprev_valid[c] && s < p->nprev_n[c])
                    base = p->nprev[c][s];
                q[e][s] = base + (long long)delta * d->noi[e][s];
            }
        }
        for (s = 0; s < n; s++)
            p->nprev[c][s] = q[e][s];
        p->nprev_n[c] = n;
        p->nprev_valid[c] = 1;
    }
}

/* Python's float ** : OverflowError when the result overflows */
static int pow2(double x, double *out)
{
    double r = pow(2.0, x);
    if (isinf(r))
        return -1;
    *out = r;
    return 0;
}

typedef double mat_t[NSB][AC4_ASPX_MAX_ENV];

/* the envelopes' scale: num_qmf_subbands * IMDCT_K^2 (aspx_envadj.
   ENV_SCALE: the standard's absolute energies in this decoder's units) */
#define ENV_SCALE (64.0 * 2.0 * 1536.0)

/* Pseudocodes 82, 84: a = 2 for 1.5 dB steps (aspx_qmode_env 0) */
static double env_a(int qmode)
{
    return qmode == 0 ? 2.0 : 1.0;
}

/* Pseudocode 82 + 91 (map_scf) */
static int map_scf(long long q[AC4_ASPX_MAX_ENV][AC4_ASPX_MAX_SBG],
                   const int *nq, int n_env, const ac4_aspx_tab *t,
                   const ac4_aspx_framing *fr, int n_sb, double a, mat_t out)
{
    int e, s, k;
    for (k = 0; k < n_sb; k++)
        for (e = 0; e < n_env; e++)
            out[k][e] = 0.0;
    for (e = 0; e < n_env; e++) {
        const int *tbl = fr->freq_res[e] ? t->hi : t->lo;
        int len = (fr->freq_res[e] ? t->n_hi : t->n_lo) + 1;
        for (s = 0; s < nq[e]; s++) {
            int lo, hi;
            double v;
            if (s + 1 >= len)
                break;
            lo = tbl[s] - t->sbx;
            hi = tbl[s + 1] - t->sbx;
            if (hi > n_sb)
                hi = n_sb;
            if (pow2((double)q[e][s] / a, &v) != 0)
                return -1;
            v = ENV_SCALE * v;
            for (k = lo < 0 ? 0 : lo; k < hi; k++)
                out[k][e] = v;
        }
    }
    return 0;
}

/* Pseudocode 84 (map_scf_joint) */
static int map_scf_joint(long long qa[AC4_ASPX_MAX_ENV][AC4_ASPX_MAX_SBG],
                         const int *na,
                         long long qb[AC4_ASPX_MAX_ENV][AC4_ASPX_MAX_SBG],
                         const int *nb, int n_env, const ac4_aspx_tab *t,
                         const ac4_aspx_framing *fr, int n_sb, double a,
                         mat_t A, mat_t B)
{
    int e, s, k;
    for (k = 0; k < n_sb; k++)
        for (e = 0; e < n_env; e++)
            A[k][e] = B[k][e] = 0.0;
    for (e = 0; e < n_env; e++) {
        const int *tbl = fr->freq_res[e] ? t->hi : t->lo;
        int len = (fr->freq_res[e] ? t->n_hi : t->n_lo) + 1;
        int m = na[e] < nb[e] ? na[e] : nb[e];
        for (s = 0; s < m; s++) {
            int lo, hi;
            double nom, pa, pb, va, vb;
            if (s + 1 >= len)
                break;
            lo = tbl[s] - t->sbx;
            if (lo < 0)
                lo = 0;
            hi = tbl[s + 1] - t->sbx;
            if (hi > n_sb)
                hi = n_sb;
            if (hi <= lo)
                continue;
            va = (double)qa[e][s];
            vb = (double)qb[e][s];
            if (pow2(va / a + 1.0, &nom) != 0)
                return -1;
            nom = nom * ENV_SCALE;
            if (pow2((double)PAN_OFFSET - vb / a, &pa) != 0)
                return -1;
            if (pow2(vb / a - (double)PAN_OFFSET, &pb) != 0)
                return -1;
            for (k = lo; k < hi; k++) {
                A[k][e] = nom / (1.0 + pa);
                B[k][e] = nom / (1.0 + pb);
            }
        }
    }
    return 0;
}

/* Pseudocode 84, the noise half (map_noise_joint) */
static int map_noise_joint(long long qa[2][AC4_ASPX_MAX_NOISE], int ne_a,
                           long long qb[2][AC4_ASPX_MAX_NOISE], int ne_b,
                           const ac4_aspx_tab *t, int n_sb, int n_env,
                           mat_t A, mat_t B)
{
    int e, s, k;
    for (k = 0; k < n_sb; k++)
        for (e = 0; e < n_env; e++)
            A[k][e] = B[k][e] = 0.0;
    for (e = 0; e < n_env; e++) {
        int ia = e < ne_a - 1 ? e : ne_a - 1;
        int ib = e < ne_b - 1 ? e : ne_b - 1;
        for (s = 0; s < t->n_noise; s++) {
            int lo = t->noise[s] - t->sbx, hi = t->noise[s + 1] - t->sbx;
            double nom, pa, pb;
            long long va = qa[ia][s], vb = qb[ib][s];
            if (lo < 0)
                lo = 0;
            if (hi > n_sb)
                hi = n_sb;
            if (hi <= lo)
                continue;
            if (pow2((double)(NOISE_FLOOR_OFFSET - va + 1), &nom) != 0
                || pow2((double)(PAN_OFFSET - vb), &pa) != 0
                || pow2((double)(vb - PAN_OFFSET), &pb) != 0)
                return -1;
            for (k = lo; k < hi; k++) {
                A[k][e] = nom / (1.0 + pa);
                B[k][e] = nom / (1.0 + pb);
            }
        }
    }
    return 0;
}

/* Pseudocode 83 + map_noise */
static int map_noise(long long q[2][AC4_ASPX_MAX_NOISE], int ne,
                     const ac4_aspx_tab *t, int n_sb, int n_env, mat_t out)
{
    double dq[2][AC4_ASPX_MAX_NOISE];
    int e, s, k;
    for (e = 0; e < ne; e++)
        for (s = 0; s < t->n_noise; s++)
            if (pow2((double)(NOISE_FLOOR_OFFSET - q[e][s]), &dq[e][s]) != 0)
                return -1;
    for (k = 0; k < n_sb; k++)
        for (e = 0; e < n_env; e++)
            out[k][e] = 0.0;
    for (e = 0; e < n_env; e++) {
        int src = e < ne - 1 ? e : ne - 1;
        for (s = 0; s < t->n_noise; s++) {
            int lo = t->noise[s] - t->sbx, hi = t->noise[s + 1] - t->sbx;
            if (lo < 0)
                lo = 0;
            if (hi > n_sb)
                hi = n_sb;
            for (k = lo; k < hi; k++)
                out[k][e] = dq[src][s];
        }
    }
    return 0;
}

/* Pseudocode 90, aspx_interpolation 1 branch (aspx_envadj.estimate) */
static void estimate(const ac4_aspx_pair *p, int c, const ac4_aspx_tab *t,
                     const int *bd, int n_env, long long slot,
                     long long nts, mat_t est)
{
    int n_sb = t->num_sb, e, k;
    double row[64];
    for (k = 0; k < n_sb; k++)
        for (e = 0; e < n_env; e++)
            est[k][e] = 0.0;
    for (e = 0; e < n_env; e++) {
        long long ta = (long long)bd[e] * p->tia + p->tso + slot;
        long long tz = (long long)bd[e + 1] * p->tia + p->tso + slot, ts;
        int L;
        if (ta < 0)
            ta = 0;
        if (tz > nts)
            tz = nts;
        if (tz <= ta)
            continue;
        L = (int)(tz - ta);
        for (k = 0; k < n_sb; k++) {
            int sb = t->sbx + k;
            for (ts = ta; ts < tz; ts++) {
                double h = hypot(p->qre[c][QI(ts)][sb], p->qim[c][QI(ts)][sb]);
                row[ts - ta] = h * h;
            }
            est[k][e] = ac4_np_sum(row, L) / (double)L;
        }
    }
}

/* Pseudocodes 94..101 (aspx_envadj.adjust) */
static void adjust(const ac4_aspx_tab *t, int n_sb, int n_env, mat_t sig,
                   mat_t est, mat_t noise, const ac4_aspx_framing *fr,
                   const uint8_t (*sidx)[AC4_ASPX_MAX_ENV],
                   const uint8_t (*area)[AC4_ASPX_MAX_ENV],
                   mat_t gain_out, mat_t noise_out, mat_t sine_out)
{
    /* Pseudocodes 95 and 99: the envelope at aspx_tsg_ptr (and at
       p_sine_at_end) has no noise factor in its gain and no noise term in
       its boost (aspx_envadj.transient_envelopes) */
    int tr[AC4_ASPX_MAX_ENV];
    static const double ZERO = 0.0;
    double gain[NSB][AC4_ASPX_MAX_ENV], noise_lev[NSB][AC4_ASPX_MAX_ENV];
    double maxg[NSB][AC4_ASPX_MAX_ENV], lim_gain[NSB][AC4_ASPX_MAX_ENV];
    double noise_lim[NSB][AC4_ASPX_MAX_ENV], boost[NSB][AC4_ASPX_MAX_ENV];
    double sine_lev[NSB][AC4_ASPX_MAX_ENV];
    int k, e, i;
    (void)ZERO;
    for (e = 0; e < n_env; e++)
        tr[e] = e == fr->tsg_ptr || e == fr->sine_at_end;
    for (k = 0; k < n_sb; k++)
        for (e = 0; e < n_env; e++) {
            double snf;
            if (tr[e])
                gain[k][e] = sqrt(sig[k][e] / (EPSILON + est[k][e]));
            else
                gain[k][e] = sqrt(sig[k][e] / (EPSILON + est[k][e])
                                  / (1.0 + noise[k][e]));
            snf = sig[k][e] / (1.0 + noise[k][e]);
            noise_lev[k][e] = sqrt(snf * noise[k][e]);
            /* Pseudocodes 94, 95: the sinusoids */
            sine_lev[k][e] = sqrt(snf * (double)sidx[k][e]);
            if (area[k][e])
                gain[k][e] = sqrt(sig[k][e] * noise[k][e]
                                  / ((EPSILON + est[k][e])
                                     * (1.0 + noise[k][e])));
            maxg[k][e] = 0.0;
            boost[k][e] = 1.0;
        }
    /* Pseudocode 96: limiter group i is subbands lim[i] .. lim[i+1] - 1
       (V1.4.1; V1.3.1's loops stop at lim[i+1] - 2, leaving the top
       subband of each group out) */
    for (i = 0; i < t->n_lim - 1; i++) {
        int lo = t->lim[i] - t->sbx, hi = t->lim[i + 1] - t->sbx;
        if (lo < 0)
            lo = 0;
        if (hi > n_sb)
            hi = n_sb;
        if (hi <= lo)
            continue;
        for (e = 0; e < n_env; e++) {
            double nom = 0.0, den = 0.0, v;
            for (k = lo; k < hi; k++)
                nom += sig[k][e];
            for (k = lo; k < hi; k++)
                den += est[k][e];
            den = EPSILON0 + den;
            v = np_min(sqrt(nom / den) * LIM_GAIN, MAX_SIG_GAIN);
            for (k = lo; k < hi; k++)
                maxg[k][e] = v;
        }
    }
    for (k = 0; k < n_sb; k++)
        for (e = 0; e < n_env; e++) {
            double ratio;
            if (maxg[k][e] == 0.0)
                maxg[k][e] = MAX_SIG_GAIN;
            lim_gain[k][e] = np_min(gain[k][e], maxg[k][e]);
            if (gain[k][e] > 0.0)
                ratio = maxg[k][e] / np_max(gain[k][e], 1e-30);
            else
                ratio = 1.0;
            noise_lim[k][e] = np_min(noise_lev[k][e],
                                     noise_lev[k][e] * ratio);
        }
    for (i = 0; i < t->n_lim - 1; i++) {
        int lo = t->lim[i] - t->sbx, hi = t->lim[i + 1] - t->sbx;
        if (lo < 0)
            lo = 0;
        if (hi > n_sb)
            hi = n_sb;
        if (hi <= lo)
            continue;
        for (e = 0; e < n_env; e++) {
            double nom = 0.0, den = 0.0, dn = 0.0, ds = 0.0, v;
            for (k = lo; k < hi; k++)
                nom += sig[k][e];
            nom = EPSILON0 + nom;
            for (k = lo; k < hi; k++)
                den += est[k][e] * (lim_gain[k][e] * lim_gain[k][e]);
            den = EPSILON0 + den;
            for (k = lo; k < hi; k++)
                ds += sine_lev[k][e] * sine_lev[k][e];
            den = den + ds;
            /* Pseudocode 99: the noise counts where there is no sinusoid
               and the envelope is not a transient one */
            for (k = lo; k < hi; k++)
                dn += (sine_lev[k][e] == 0.0 && !tr[e])
                      ? noise_lim[k][e] * noise_lim[k][e] : 0.0;
            den = den + dn;
            v = np_min(sqrt(nom / den), MAX_BOOST_FACT);
            for (k = lo; k < hi; k++)
                boost[k][e] = v;
        }
    }
    for (k = 0; k < n_sb; k++)
        for (e = 0; e < n_env; e++) {
            gain_out[k][e] = lim_gain[k][e] * boost[k][e];
            noise_out[k][e] = noise_lim[k][e] * boost[k][e];
            sine_out[k][e] = sine_lev[k][e] * boost[k][e];
        }
}

/* Pseudocode 90, aspx_interpolation 0 (aspx_envadj.estimate_groups):
   the mean energy over each subband group of the envelope's resolution */
static void estimate_groups(const ac4_aspx_pair *p, int c,
                            const ac4_aspx_tab *t,
                            const ac4_aspx_framing *fr, const int *bd,
                            int n_env, long long slot, long long nts,
                            mat_t est)
{
    int n_sb = t->num_sb, e, k, g;
    static double buf[64 * 2 * AC4_MAX_NTS];
    for (k = 0; k < n_sb; k++)
        for (e = 0; e < n_env; e++)
            est[k][e] = 0.0;
    for (e = 0; e < n_env; e++) {
        long long ta = (long long)bd[e] * p->tia + p->tso + slot;
        long long tz = (long long)bd[e + 1] * p->tia + p->tso + slot, ts;
        const int *tab = fr->freq_res[e] ? t->hi : t->lo;
        int ng = fr->freq_res[e] ? t->n_hi : t->n_lo;
        if (ta < 0)
            ta = 0;
        if (tz > nts)
            tz = nts;
        if (tz <= ta)
            continue;
        for (g = 0; g < ng; g++) {
            int n = 0, sb;
            double m;
            for (sb = tab[g]; sb < tab[g + 1]; sb++)
                for (ts = ta; ts < tz; ts++) {
                    double h = hypot(p->qre[c][QI(ts)][sb],
                                     p->qim[c][QI(ts)][sb]);
                    buf[n++] = h * h;
                }
            m = ac4_np_sum(buf, n) / (double)n;
            for (sb = tab[g]; sb < tab[g + 1]; sb++)
                est[sb - t->sbx][e] = m;
        }
    }
}

/* Pseudocodes 92, 93 for one channel: sine_idx_sb and sine_area_sb */
static void sines(const ac4_aspx_pair *p, int c, const ac4_aspx_tab *t,
                  const ac4_aspx_chd *d, int n_env,
                  uint8_t (*sidx)[AC4_ASPX_MAX_ENV],
                  uint8_t (*area)[AC4_ASPX_MAX_ENV])
{
    int e, g, k, n_sb = t->num_sb;
    for (k = 0; k < n_sb; k++)
        for (e = 0; e < n_env; e++)
            sidx[k][e] = area[k][e] = 0;
    for (e = 0; e < n_env; e++)
        for (g = 0; g < t->n_hi; g++) {
            int a = t->hi[g] - t->sbx, z = t->hi[g + 1] - t->sbx, mid;
            if (!d->ah[g])
                continue;
            mid = (int)(0.5 * (double)(z + a));
            if (e >= d->fr.tsg_ptr || d->fr.sine_at_end == 0
                || (mid < p->n_sine_sb[c] && p->sine_sb[c][mid]))
                sidx[mid][e] = 1;
        }
    for (e = 0; e < n_env; e++) {
        const int *tab = d->fr.freq_res[e] ? t->hi : t->lo;
        int ng = d->fr.freq_res[e] ? t->n_hi : t->n_lo;
        for (g = 0; g < ng; g++) {
            int a = tab[g] - t->sbx, z = tab[g + 1] - t->sbx, any = 0;
            for (k = a; k < z; k++)
                any |= sidx[k][e];
            for (k = a; k < z; k++)
                area[k][e] = (uint8_t)any;
        }
    }
}

/* the body of apply_hf_pair's loop for one frame at QMF slot `slot`;
   nts = slots analysed (the Python's Qh.shape[1]) */
static void apply_frame(ac4_aspx_pair *p, const ac4_aspx_frame *f,
                        long long slot, long long nts)
{
    const ac4_aspx_tab *t = &f->tab;
    int n_sb = t->num_sb, sbx = t->sbx, c, e, k;
    long long qa[AC4_ASPX_MAX_ENV][AC4_ASPX_MAX_SBG];
    long long qb[AC4_ASPX_MAX_ENV][AC4_ASPX_MAX_SBG];
    long long na[2][AC4_ASPX_MAX_NOISE], nb[2][AC4_ASPX_MAX_NOISE];
    int nqa[AC4_ASPX_MAX_ENV], nqb[AC4_ASPX_MAX_ENV];
    mat_t sigA, sigB, nsA, nsB, est, g, nl;
    const ac4_aspx_chd *d0, *d1;
    const int *b0, *b1;
    int ne0, ne1, joint;
    long long t0;

    if (!effective(f))
        return;
    if (f->kind == AC4_GRP_CONCEALED) {
        long long ta = slot + p->tso, tz = slot + p->nts + p->tso, ts;
        if (ta < 0)
            ta = 0;
        if (tz > nts)
            tz = nts;
        for (c = 0; c < 2; c++)
            for (ts = ta; ts < tz; ts++)
                for (k = sbx; k < sbx + n_sb; k++) {
                    p->qre[c][QI(ts)][k] = 0.0;
                    p->qim[c][QI(ts)][k] = 0.0;
                }
        p->idx_noise = (p->idx_noise + n_sb * p->nts) % 512;
        return;
    }
    d0 = &f->g.c[0];
    d1 = &f->g.c[1];
    reconstruct(p, 0, d0, t, qa, nqa);
    reconstruct(p, 1, d1, t, qb, nqb);
    noise_qscf(p, 0, d0, t, na);
    noise_qscf(p, 1, d1, t, nb);
    joint = d0->balance == 1;
    ne0 = d0->fr.num_env;
    ne1 = d1->fr.num_env;
    if (joint) {
        int ne = ne0 < ne1 ? ne0 : ne1;
        if (map_scf_joint(qa, nqa, qb, nqb, ne, t, &d0->fr, n_sb,
                          env_a(d0->qmode), sigA, sigB) != 0)
            return;
        if (map_noise_joint(na, d0->fr.num_noise, nb, d1->fr.num_noise, t,
                            n_sb, ne, nsA, nsB) != 0)
            return;
        ne0 = ne1 = ne;
    } else {
        if (map_scf(qa, nqa, ne0, t, &d0->fr, n_sb, env_a(d0->qmode),
                    sigA) != 0)
            return;
        if (map_scf(qb, nqb, ne1, t, &d1->fr, n_sb, env_a(d1->qmode),
                    sigB) != 0)
            return;
        if (map_noise(na, d0->fr.num_noise, t, n_sb, ne0, nsA) != 0)
            return;
        if (map_noise(nb, d1->fr.num_noise, t, n_sb, ne1, nsB) != 0)
            return;
    }
    b0 = d0->fr.borders;
    b1 = joint ? b0 : d1->fr.borders;
    for (c = 0; c < 2; c++) {
        const int *bd = c ? b1 : b0;
        int ne = c ? ne1 : ne0, si, sbase, last_si;
        const ac4_aspx_chd *dc = c ? d1 : d0;
        long long a0 = slot + p->tso, ta_i, tz_i, ts;
        double ch[AC4_ASPX_MAX_NOISE], a0r[64], a0i[64], a1r[64], a1i[64];
        double gv[64];
        int any = 0, i;
        uint8_t sidx[NSB][AC4_ASPX_MAX_ENV], area[NSB][AC4_ASPX_MAX_ENV];
        mat_t sl;
        ta_i = a0 + (long long)bd[0] * p->tia;
        tz_i = a0 + (long long)bd[ne] * p->tia;
        if (ta_i < 0)
            ta_i = 0;
        if (tz_i > nts)
            tz_i = nts;
        /* HF generation over the interval (Pseudocodes 85-89) */
        chirps(p, c, dc->tna, t->n_noise, ch);
        for (i = 0; i < t->n_noise; i++)
            any |= ch[i] != 0.0;
        if (any)
            tna_alpha(p, c, t->sba, a0, nts, a0r, a0i, a1r, a1i);
        else
            for (i = 0; i < t->sba; i++)
                a0r[i] = a0i[i] = a1r[i] = a1i[i] = 0.0;
        if (f->preflat && tz_i > ta_i)
            preflat_gain(p, c, sbx, ta_i, tz_i, gv);
        if (tz_i > ta_i)
            generate(p, c, t, ta_i, tz_i, nts, a0r, a0i, a1r, a1i, ch,
                     f->preflat ? gv : NULL);
        if (f->interpolation == 0)
            estimate_groups(p, c, t, &dc->fr, bd, ne, slot, nts, est);
        else
            estimate(p, c, t, bd, ne, slot, nts, est);
        sines(p, c, t, dc, ne, sidx, area);
        adjust(t, n_sb, ne, c ? sigB : sigA, est, c ? nsB : nsA,
               c ? &d1->fr : &d0->fr, (const uint8_t (*)[AC4_ASPX_MAX_ENV])
               sidx, (const uint8_t (*)[AC4_ASPX_MAX_ENV])area, g, nl, sl);
        /* atsg_sig[0] in QMF slots: Pseudocode 103 counts the noise index
           from the start of the frame's first envelope */
        t0 = slot + (long long)bd[0] * p->tia + p->tso;
        si = p->sine_idx[c];
        sbase = si < 0 ? 1 : (si + 1) % 4;
        last_si = si;
        for (e = 0; e < ne; e++) {
            long long ta = slot + (long long)bd[e] * p->tia + p->tso;
            long long tz = slot + (long long)bd[e + 1] * p->tia + p->tso;
            int has_sine = 0;
            if (ta < 0)
                ta = 0;
            if (tz > nts)
                tz = nts;
            if (tz <= ta)
                continue;
            for (k = 0; k < n_sb; k++)
                has_sine |= sl[k][e] != 0.0;
            for (ts = ta; ts < tz; ts++) {
                double *qr = p->qre[c][QI(ts)], *qi = p->qim[c][QI(ts)];
                for (k = 0; k < n_sb; k++) {
                    /* complex * (g + 0j), as numpy */
                    double ar = qr[sbx + k], ai = qi[sbx + k], gg = g[k][e];
                    qr[sbx + k] = ar * gg - ai * 0.0;
                    qi[sbx + k] = ar * 0.0 + ai * gg;
                }
            }
            for (ts = ta; ts < tz; ts++) {
                double *qr = p->qre[c][QI(ts)], *qi = p->qim[c][QI(ts)];
                long long base = (long long)p->idx_noise
                                 + (long long)n_sb * (ts - t0);
                int ix = (int)(((sbase + ts - t0) % 4 + 4) % 4);
                for (k = 0; k < n_sb; k++) {
                    int kk = (int)((base + k + 1) % 512);
                    double lv = nl[k][e];
                    double nr = AC4T_ASPX_NOISE[kk][0];
                    double ni = AC4T_ASPX_NOISE[kk][1];
                    qr[sbx + k] += lv * nr - 0.0 * ni;
                    qi[sbx + k] += lv * ni + 0.0 * nr;
                }
                if (has_sine)
                    for (k = 0; k < n_sb; k++) {
                        /* Pseudocodes 104, 105 */
                        double sgn = (k + sbx) % 2 ? -1.0 : 1.0;
                        qr[sbx + k] += sl[k][e] * SINE_RE[ix];
                        qi[sbx + k] += sl[k][e] * sgn * SINE_IM[ix];
                    }
                last_si = ix;
            }
        }
        p->sine_idx[c] = last_si;
        for (k = 0; k < n_sb; k++)
            p->sine_sb[c][k] = ne > 0 ? sidx[k][ne - 1] : 0;
        p->n_sine_sb[c] = n_sb;
        /* interleaved waveform coding (5.7.6.5.3): the core above sbx is
           added in the FIC groups and replaces the band in the TIC slots */
        if (dc->has_fic)
            for (i = 0; i < t->n_hi; i++) {
                int sb;
                if (!dc->fic[i])
                    continue;
                for (sb = t->hi[i]; sb < t->hi[i + 1]; sb++)
                    for (ts = ta_i; ts < tz_i; ts++) {
                        p->qre[c][QI(ts)][sb] += p->are[c][QI(ts)][sb];
                        p->qim[c][QI(ts)][sb] += p->aim[c][QI(ts)][sb];
                    }
            }
        if (dc->has_tic)
            for (i = 0; i < p->nats; i++) {
                long long a_ = a0 + (long long)i * p->tia, z_ = a_ + p->tia;
                int sb;
                if (!dc->tic[i])
                    continue;
                if (a_ < 0)
                    a_ = 0;
                if (z_ > nts)
                    z_ = nts;
                for (ts = a_; ts < z_; ts++)
                    for (sb = sbx; sb < sbx + n_sb; sb++) {
                        p->qre[c][QI(ts)][sb] = p->are[c][QI(ts)][sb];
                        p->qim[c][QI(ts)][sb] = p->aim[c][QI(ts)][sb];
                    }
            }
    }
    p->n_applied++;
    {
        long long span = 0;
        for (e = 0; e < d0->fr.num_env; e++) {
            long long a = slot + (long long)b0[e] * p->tia + p->tso;
            long long z = slot + (long long)b0[e + 1] * p->tia + p->tso;
            if (z > nts)
                z = nts;
            if (a < 0)
                a = 0;
            if (z - a > 0)
                span += z - a;
        }
        p->idx_noise = (int)(((long long)p->idx_noise
                              + (long long)n_sb * span) % 512);
    }
}

/* ------------------------------------------------------------ A-CPL */
/* acpl.render for one frame: the A-CPL of frame f (its parameters in
   fr->acpl, or none) on slots [nts f + TS_OFFSET, + nts) clipped to nts,
   from the pair's A-SPX output into the upmix ring */
static void acpl_frame(ac4_aspx_pair *p, const ac4_aspx_frame *fr,
                       long long f, long long nts)
{
    ac4_acpl_work *w = &p->acpl_work;
    long long a = (long long)p->nts * f + p->tso, z = a + p->nts, ts;
    const double (*xr[2])[64], (*xi[2])[64];
    double (*zr[5])[64], (*zi[5])[64];
    int c, n, i;
    if (z > nts)
        z = nts;
    n = (int)(z - a);
    if (n < 0)
        n = 0;
    for (c = 0; c < 2; c++)
        for (i = 0; i < n; i++) {
            memcpy(w->xr[c][i], p->qre[c][QI(a + i)], sizeof(w->xr[c][i]));
            memcpy(w->xi[c][i], p->qim[c][QI(a + i)], sizeof(w->xi[c][i]));
        }
    for (c = 0; c < 2; c++) {
        xr[c] = (const double (*)[64])w->xr[c];
        xi[c] = (const double (*)[64])w->xi[c];
    }
    for (c = 0; c < 5; c++) {
        zr[c] = w->zr[c];
        zi[c] = w->zi[c];
    }
    ac4_acpl_run(&p->acpl, w, fr && fr->acpl.kind != AC4_GRP_NONE
                              ? &fr->acpl : NULL, n, p->nts, xr, xi, zr, zi);
    for (c = 0; c < 5; c++)
        for (i = 0; i < n; i++) {
            ts = a + i;
            memcpy(p->ure[c][QI(ts)], w->zr[c][i], sizeof(w->zr[c][i]));
            memcpy(p->uim[c][QI(ts)], w->zi[c][i], sizeof(w->zi[c][i]));
        }
}

/* acpl.render_1ch for one frame (ASPX_ACPL_1 / _2): the modules of
   frame f (fr->acpl1, or none) on slots [nts f + TS_OFFSET, + nts)
   clipped to nts; module m takes the A-SPX output of channel m and the
   residual m; outputs to the upmix ring (L, R, -, Ls, Rs; a pair: L, R) */
static void acpl1_frame(ac4_aspx_pair *p, const ac4_aspx_frame *fr,
                        long long f, long long nts)
{
    ac4_acpl_work *w = &p->acpl_work;
    long long a = (long long)p->nts * f + p->tso, z = a + p->nts, ts;
    const double (*x0r[2])[64], (*x0i[2])[64], (*x1r[2])[64], (*x1i[2])[64];
    double (*z0r[2])[64], (*z0i[2])[64], (*z1r[2])[64], (*z1i[2])[64];
    int c, n, i, nm = p->acpl_on == 3 ? 2 : 1, kind;
    const double sq2 = sqrt(2.0);
    if (z > nts)
        z = nts;
    n = (int)(z - a);
    if (n < 0)
        n = 0;
    /* gather: x0 into xr / xi, the residual into zr / zi[0..1]; the
       outputs into zr / zi[2..3] (z0) and [4], ip-free scratch below */
    for (c = 0; c < nm; c++)
        for (i = 0; i < n; i++) {
            memcpy(w->xr[c][i], p->qre[c][QI(a + i)], sizeof(w->xr[c][i]));
            memcpy(w->xi[c][i], p->qim[c][QI(a + i)], sizeof(w->xi[c][i]));
            memcpy(w->zr[c][i], p->rre[c][QI(a + i)], sizeof(w->zr[c][i]));
            memcpy(w->zi[c][i], p->rim[c][QI(a + i)], sizeof(w->zi[c][i]));
        }
    for (c = 0; c < 2; c++) {
        x0r[c] = (const double (*)[64])w->xr[c];
        x0i[c] = (const double (*)[64])w->xi[c];
        x1r[c] = (const double (*)[64])w->zr[c];
        x1i[c] = (const double (*)[64])w->zi[c];
    }
    z0r[0] = w->zr[2];
    z0i[0] = w->zi[2];
    z0r[1] = w->zr[3];
    z0i[1] = w->zi[3];
    z1r[0] = p->tmp1r[0];
    z1i[0] = p->tmp1i[0];
    z1r[1] = p->tmp1r[1];
    z1i[1] = p->tmp1i[1];
    kind = fr ? fr->acpl1_kind : AC4_GRP_NONE;
    ac4_acpl1_run(&p->acpl, w, fr ? fr->acpl1 : NULL, kind, nm, n, p->nts,
                  p->acpl_on == 2, x0r, x0i, x1r, x1i, z0r, z0i, z1r, z1i);
    for (i = 0; i < n; i++) {
        ts = a + i;
        if (nm == 1) {
            memcpy(p->ure[0][QI(ts)], z0r[0][i], sizeof(z0r[0][i]));
            memcpy(p->uim[0][QI(ts)], z0i[0][i], sizeof(z0i[0][i]));
            memcpy(p->ure[1][QI(ts)], z1r[0][i], sizeof(z1r[0][i]));
            memcpy(p->uim[1][QI(ts)], z1i[0][i], sizeof(z1i[0][i]));
        } else {
            int k;
            for (c = 0; c < 2; c++) {
                memcpy(p->ure[c][QI(ts)], z0r[c][i], sizeof(z0r[c][i]));
                memcpy(p->uim[c][QI(ts)], z0i[c][i], sizeof(z0i[c][i]));
                for (k = 0; k < 64; k++) {
                    p->ure[3 + c][QI(ts)][k] = z1r[c][i][k] * sq2;
                    p->uim[3 + c][QI(ts)][k] = z1i[c][i][k] * sq2;
                }
            }
        }
    }
}

/* the TS_OFFSET slots before the first frame: no frame, so the core
   passes through (acpl.render leaves them as they are) */
static void acpl_lead(ac4_aspx_pair *p)
{
    long long ts;
    int c;
    if (p->acpl_on >= 2) {
        /* render_1ch's passthrough: split (a pair) x0 + x1, x0 - x1;
           else x0 on z0, nothing on z1 */
        for (ts = 0; ts < p->tso; ts++) {
            int k, nm = p->acpl_on == 3 ? 2 : 1;
            for (c = 0; c < nm; c++)
                for (k = 0; k < 64; k++) {
                    double ar = p->qre[c][QI(ts)][k], ai = p->qim[c][QI(ts)][k];
                    double br = p->rre[c][QI(ts)][k], bi = p->rim[c][QI(ts)][k];
                    int o0 = c, o1 = nm == 1 ? 1 : 3 + c;
                    if (p->acpl_on == 2) {
                        p->ure[o0][QI(ts)][k] = ar + br;
                        p->uim[o0][QI(ts)][k] = ai + bi;
                        p->ure[o1][QI(ts)][k] = ar - br;
                        p->uim[o1][QI(ts)][k] = ai - bi;
                    } else {
                        p->ure[o0][QI(ts)][k] = ar;
                        p->uim[o0][QI(ts)][k] = ai;
                        p->ure[o1][QI(ts)][k] = 0.0;
                        p->uim[o1][QI(ts)][k] = 0.0;
                    }
                }
        }
        return;
    }
    for (ts = 0; ts < p->tso; ts++)
        for (c = 0; c < 5; c++) {
            if (c < 2) {
                memcpy(p->ure[c][QI(ts)], p->qre[c][QI(ts)],
                       sizeof(p->ure[c][0]));
                memcpy(p->uim[c][QI(ts)], p->qim[c][QI(ts)],
                       sizeof(p->uim[c][0]));
            } else {
                memset(p->ure[c][QI(ts)], 0, sizeof(p->ure[c][0]));
                memset(p->uim[c][QI(ts)], 0, sizeof(p->uim[c][0]));
            }
        }
}

/* ------------------------------------------------------------ streaming */
static void synth_frame(ac4_aspx_pair *p, const ac4_qmf_tables *q,
                        long long slot0, double *const y[2])
{
    int c, i;
    for (c = 0; c < 2; c++)
        for (i = 0; i < p->nts; i++) {
            long long ts = slot0 + i;
            qmf_synth_slot(q, p->syn[c], p->qre[c][QI(ts)],
                           p->qim[c][QI(ts)], y[c] + 64 * i);
        }
    if (p->acpl_on)
        for (c = 0; c < 5; c++) {
            if ((p->acpl_on == 2 && c >= 2) || (p->acpl_on == 3 && c == 2))
                continue;
            for (i = 0; i < p->nts; i++) {
                long long ts = slot0 + i;
                qmf_synth_slot(q, p->usyn[c], p->ure[c][QI(ts)],
                               p->uim[c][QI(ts)], p->upout[c] + 64 * i);
            }
        }
}

void ac4_aspx_pair_residual(ac4_aspx_pair *p, const ac4_qmf_tables *q,
                            const double *const res[2])
{
    long long s0 = p->n_in * p->nts;
    int c, i;
    for (c = 0; c < 2; c++)
        for (i = 0; i < p->nts; i++) {
            long long ts = s0 + i;
            if (res[c])
                qmf_analyse_slot(q, p->rana[c], res[c] + 64 * i,
                                 p->rre[c][QI(ts)], p->rim[c][QI(ts)]);
            else {
                memset(p->rre[c][QI(ts)], 0, sizeof(p->rre[c][0]));
                memset(p->rim[c][QI(ts)], 0, sizeof(p->rim[c][0]));
            }
        }
}

static void delayed_core(const ac4_aspx_pair *p, double *const dc[2])
{
    int c;
    if (!dc)
        return;
    for (c = 0; c < 2; c++)
        if (dc[c])
            memcpy(dc[c], p->core[c], sizeof(double) * (size_t)p->fl);
}

int ac4_aspx_pair_push(ac4_aspx_pair *p, const ac4_qmf_tables *q,
                       const double *const x[2], const ac4_aspx_frame *f,
                       double *const y[2], double *const dc[2])
{
    long long j = p->n_in, s0 = j * p->nts;
    int c, i, out = 0, nt = p->nts, fl = p->fl;
    /* analyse core frame j, HF-generating the slots whose frame has
       A-SPX (slots s0 .. s0 + TS_OFFSET - 1 belong to frame j - 1's
       envelopes) */
    for (c = 0; c < 2; c++)
        for (i = 0; i < nt; i++) {
            long long ts = s0 + i;
            double *re = p->are[c][QI(ts)], *im = p->aim[c][QI(ts)];
            qmf_analyse_slot(q, p->ana[c], x[c] + 64 * i, re, im);
            /* the output starts as the core; apply_frame generates the
               high band over each interval */
            memcpy(p->qre[c][QI(ts)], re, sizeof(p->qre[c][0]));
            memcpy(p->qim[c][QI(ts)], im, sizeof(p->qim[c][0]));
        }
    if (j >= 1) {
        apply_frame(p, &p->prev, s0 - nt, s0 + nt);
        if (p->acpl_on) {
            if (j == 1)
                acpl_lead(p);
            if (p->acpl_on >= 2)
                acpl1_frame(p, &p->prev, j - 1, s0 + nt);
            else
                acpl_frame(p, &p->prev, j - 1, s0 + nt);
        }
        synth_frame(p, q, s0 - nt, y);
        delayed_core(p, dc);
        out = 1;
    }
    /* keep the last fl + 577 core samples (the delayed core of the next
       output frame starts 577 samples before this frame's end) */
    for (c = 0; c < 2; c++) {
        memmove(p->core[c], p->core[c] + fl, sizeof(double) * 577);
        memcpy(p->core[c] + 577, x[c], sizeof(double) * (size_t)fl);
    }
    p->prev = *f;
    p->n_in++;
    return out;
}

int ac4_aspx_pair_finish(ac4_aspx_pair *p, const ac4_qmf_tables *q,
                         double *const y[2], double *const dc[2])
{
    long long s0 = p->n_in * p->nts;
    if (p->n_in < 1)
        return 0;
    apply_frame(p, &p->prev, s0 - p->nts, s0);
    if (p->acpl_on) {
        if (p->n_in == 1)
            acpl_lead(p);
        if (p->acpl_on >= 2)
            acpl1_frame(p, &p->prev, p->n_in - 1, s0);
        else
            acpl_frame(p, &p->prev, p->n_in - 1, s0);
    }
    synth_frame(p, q, s0 - p->nts, y);
    delayed_core(p, dc);
    return 1;
}
