/* SPDX-License-Identifier: Apache-2.0 */
/* A-SPX rendering of one channel pair, streaming: the 64-band complex QMF
 * bank (ac4/qmf.py and its C kernel ac4/qmf_kernel.c, same loop order),
 * HF generation (ac4/aspx_hfgen.py), envelope reconstruction
 * (ac4/aspx_env.py), dequantisation, estimation and adjustment
 * (ac4/aspx_envadj.py), and the per-frame loop of
 * ac4/aspx_render.apply_hf_pair; TS 103 190-1 5.7.3, 5.7.4, 5.7.6.
 *
 * The Python renders a whole decode pass at once; this renders frame by
 * frame with the same arithmetic. A frame's envelopes cover QMF slots
 * [24 j + 18 + 2 b0, 24 j + 18 + 2 bn) with 0 <= b0 and bn <= 15 (the
 * aspx.borders rule; TS_OFFSET 18, see aspx_render.TS_OFFSET), i.e. at
 * most to the end of frame j + 1, so frame j is adjusted and synthesised
 * once core frame j + 1 has been analysed.
 *
 * A slot is HF-generated only when the frame its envelope belongs to
 * (frame floor((slot - 18) / 24)) has A-SPX data or was concealed with a
 * configuration known: the slots of frames from before the stream's first
 * aspx_config keep the core alone (the Python does the same). */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ac4_internal.h"

#define NSB 64
#define AC4_PI 3.14159265358979323846   /* np.pi */
#define NWIN 640
#define TS_OFFSET 18            /* 24 - ts_offset_hfgen: aspx_render.py */
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
    memset(p, 0, sizeof(*p));
}

#define QI(ts) ((int)((ts) % AC4_QRING))

static int effective(const ac4_aspx_frame *f)
{
    return f->kind != AC4_GRP_NONE && f->has_tab;
}

/* HF generation for one slot (aspx_hfgen.hf_generate): the patches copy
   the slot's analysed low subbands up */
static void hf_generate(const ac4_aspx_tab *t, double *re, double *im)
{
    double r0[NSB], i0[NSB];
    int p, k;
    memcpy(r0, re, sizeof(r0));
    memcpy(i0, im, sizeof(i0));
    for (p = 0; p < t->n_patches; p++) {
        int n = t->patch_num_sb[p], st = t->patch_start_sb[p];
        int dst = t->patches[p];
        for (k = 0; k < n; k++) {
            re[dst + k] = r0[st + k];
            im[dst + k] = i0[st + k];
        }
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
        long long ta = (long long)bd[e] * 2 + TS_OFFSET + slot;
        long long tz = (long long)bd[e + 1] * 2 + TS_OFFSET + slot, ts;
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
                   mat_t gain_out, mat_t noise_out)
{
    /* Pseudocodes 95 and 99: the envelope at aspx_tsg_ptr (and at
       p_sine_at_end) has no noise factor in its gain and no noise term in
       its boost (aspx_envadj.transient_envelopes) */
    int tr[AC4_ASPX_MAX_ENV];
    static const double ZERO = 0.0;
    double gain[NSB][AC4_ASPX_MAX_ENV], noise_lev[NSB][AC4_ASPX_MAX_ENV];
    double maxg[NSB][AC4_ASPX_MAX_ENV], lim_gain[NSB][AC4_ASPX_MAX_ENV];
    double noise_lim[NSB][AC4_ASPX_MAX_ENV], boost[NSB][AC4_ASPX_MAX_ENV];
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
            double nom = 0.0, den = 0.0, dn = 0.0, v;
            for (k = lo; k < hi; k++)
                nom += sig[k][e];
            nom = EPSILON0 + nom;
            for (k = lo; k < hi; k++)
                den += est[k][e] * (lim_gain[k][e] * lim_gain[k][e]);
            den = EPSILON0 + den;
            for (k = lo; k < hi; k++)
                dn += noise_lim[k][e] * noise_lim[k][e];
            den = den + (tr[e] ? 0.0 : dn);
            v = np_min(sqrt(nom / den), MAX_BOOST_FACT);
            for (k = lo; k < hi; k++)
                boost[k][e] = v;
        }
    }
    for (k = 0; k < n_sb; k++)
        for (e = 0; e < n_env; e++) {
            gain_out[k][e] = lim_gain[k][e] * boost[k][e];
            noise_out[k][e] = noise_lim[k][e] * boost[k][e];
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
        long long ta = slot + TS_OFFSET, tz = slot + 24 + TS_OFFSET, ts;
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
        p->idx_noise = (p->idx_noise + n_sb * 24) % 512;
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
        int ne = c ? ne1 : ne0;
        estimate(p, c, t, bd, ne, slot, nts, est);
        adjust(t, n_sb, ne, c ? sigB : sigA, est, c ? nsB : nsA,
               c ? &d1->fr : &d0->fr, g, nl);
        /* atsg_sig[0] in QMF slots: Pseudocode 103 counts the noise index
           from the start of the frame's first envelope */
        t0 = slot + (long long)bd[0] * 2 + TS_OFFSET;
        for (e = 0; e < ne; e++) {
            long long ta = slot + (long long)bd[e] * 2 + TS_OFFSET;
            long long tz = slot + (long long)bd[e + 1] * 2 + TS_OFFSET, ts;
            if (ta < 0)
                ta = 0;
            if (tz > nts)
                tz = nts;
            if (tz <= ta)
                continue;
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
                for (k = 0; k < n_sb; k++) {
                    int kk = (int)((base + k + 1) % 512);
                    double lv = nl[k][e];
                    double nr = AC4T_ASPX_NOISE[kk][0];
                    double ni = AC4T_ASPX_NOISE[kk][1];
                    qr[sbx + k] += lv * nr - 0.0 * ni;
                    qi[sbx + k] += lv * ni + 0.0 * nr;
                }
            }
        }
    }
    p->n_applied++;
    {
        long long span = 0;
        for (e = 0; e < d0->fr.num_env; e++) {
            long long a = slot + (long long)b0[e] * 2 + TS_OFFSET;
            long long z = slot + (long long)b0[e + 1] * 2 + TS_OFFSET;
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
   fr->acpl, or none) on slots [24 f + TS_OFFSET, + 24) clipped to nts,
   from the pair's A-SPX output into the upmix ring */
static void acpl_frame(ac4_aspx_pair *p, const ac4_aspx_frame *fr,
                       long long f, long long nts)
{
    ac4_acpl_work *w = &p->acpl_work;
    long long a = 24 * f + TS_OFFSET, z = a + 24, ts;
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
                              ? &fr->acpl : NULL, n, xr, xi, zr, zi);
    for (c = 0; c < 5; c++)
        for (i = 0; i < n; i++) {
            ts = a + i;
            memcpy(p->ure[c][QI(ts)], w->zr[c][i], sizeof(w->zr[c][i]));
            memcpy(p->uim[c][QI(ts)], w->zi[c][i], sizeof(w->zi[c][i]));
        }
}

/* the TS_OFFSET slots before the first frame: no frame, so the core
   passes through (acpl.render leaves them as they are) */
static void acpl_lead(ac4_aspx_pair *p)
{
    long long ts;
    int c;
    for (ts = 0; ts < TS_OFFSET; ts++)
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
        for (i = 0; i < 24; i++) {
            long long ts = slot0 + i;
            qmf_synth_slot(q, p->syn[c], p->qre[c][QI(ts)],
                           p->qim[c][QI(ts)], y[c] + 64 * i);
        }
    if (p->acpl_on)
        for (c = 0; c < 5; c++)
            for (i = 0; i < 24; i++) {
                long long ts = slot0 + i;
                qmf_synth_slot(q, p->usyn[c], p->ure[c][QI(ts)],
                               p->uim[c][QI(ts)], p->upout[c] + 64 * i);
            }
}

static void delayed_core(const ac4_aspx_pair *p, double *const dc[2])
{
    int c;
    if (!dc)
        return;
    for (c = 0; c < 2; c++)
        if (dc[c])
            memcpy(dc[c], p->core[c] + 1536 - 577, sizeof(double) * 1536);
}

int ac4_aspx_pair_push(ac4_aspx_pair *p, const ac4_qmf_tables *q,
                       const double *const x[2], const ac4_aspx_frame *f,
                       double *const y[2], double *const dc[2])
{
    long long j = p->n_in, s0 = j * 24;
    int c, i, out = 0;
    /* analyse core frame j, HF-generating the slots whose frame has
       A-SPX (slots s0 .. s0+17 belong to frame j - 1's envelopes) */
    for (c = 0; c < 2; c++)
        for (i = 0; i < 24; i++) {
            long long ts = s0 + i;
            const ac4_aspx_frame *own = i < TS_OFFSET ? &p->prev : f;
            double *re = p->qre[c][QI(ts)], *im = p->qim[c][QI(ts)];
            qmf_analyse_slot(q, p->ana[c], x[c] + 64 * i, re, im);
            if (j == 0 && i < TS_OFFSET)
                own = NULL;               /* no frame before the first */
            if (own && effective(own))
                hf_generate(&own->tab, re, im);
        }
    if (j >= 1) {
        apply_frame(p, &p->prev, s0 - 24, s0 + 24);
        if (p->acpl_on) {
            if (j == 1)
                acpl_lead(p);
            acpl_frame(p, &p->prev, j - 1, s0 + 24);
        }
        synth_frame(p, q, s0 - 24, y);
        delayed_core(p, dc);
        out = 1;
    }
    /* keep the last two core frames */
    for (c = 0; c < 2; c++) {
        memmove(p->core[c], p->core[c] + 1536, sizeof(double) * 1536);
        memcpy(p->core[c] + 1536, x[c], sizeof(double) * 1536);
    }
    p->prev = *f;
    p->n_in++;
    return out;
}

int ac4_aspx_pair_finish(ac4_aspx_pair *p, const ac4_qmf_tables *q,
                         double *const y[2], double *const dc[2])
{
    long long s0 = p->n_in * 24;
    if (p->n_in < 1)
        return 0;
    apply_frame(p, &p->prev, s0 - 24, s0);
    if (p->acpl_on) {
        if (p->n_in == 1)
            acpl_lead(p);
        acpl_frame(p, &p->prev, p->n_in - 1, s0);
    }
    synth_frame(p, q, s0 - 24, y);
    delayed_core(p, dc);
    return 1;
}
