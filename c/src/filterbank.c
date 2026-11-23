/* SPDX-License-Identifier: Apache-2.0 */
/* IMDCT, KBD windows with block switching, overlap-add.
 * Port of ac4/filterbank.py (synthesise, "reversed" reading of step 6) and
 * ac4/mdct.py; TS 103 190-1 5.5.2, 5.5.3, Table 185.
 *
 * The IMDCT is a DCT-IV computed with an N/2-point complex FFT (N/2 =
 * 3 * 2^k for every AC-4 transform length used at 48 kHz), scaled as the
 * Python's imdct_raw: dct4 (2 * sum) / N * IMDCT_K / 2, IMDCT_K =
 * sqrt(2 * 1536): the standard's (1/N) * sum (TS 103 190-1 Pseudocodes
 * 60-63) for every transform length, times one global constant. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ac4_internal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N_LEN 5
static const int LENS[N_LEN] = { 1536, 768, 384, 192, 96 };

static int len_index(int n)
{
    int i;
    for (i = 0; i < N_LEN; i++)
        if (LENS[i] == n)
            return i;
    return -1;
}

static double alpha_for(int n)
{
    /* Table 185, 44.1 / 48 kHz column */
    switch (n) {
    case 2048: case 1920: case 1536: return 3.0;
    case 1024: case 960: case 768: return 4.0;
    case 512: case 480: case 384: return 4.5;
    case 256: case 240: case 192: return 5.0;
    case 128: case 120: case 96: return 6.0;
    default: return 4.0;
    }
}

/* modified Bessel function of the first kind, order 0 (power series) */
static double bessel_i0(double x)
{
    double sum = 1.0, term = 1.0, q = 0.25 * x * x;
    int k;
    for (k = 1; k < 500; k++) {
        term *= q / ((double)k * (double)k);
        sum += term;
        if (term < sum * 1e-17)
            break;
    }
    return sum;
}

/* KBD_LEFT(n, alpha): the rising half, length n (clause 5.5.3) */
static void kbd_left(int n, double alpha, double *w)
{
    double *c = (double *)malloc(sizeof(double) * (size_t)(n + 1));
    double i0a = bessel_i0(M_PI * alpha), acc = 0.0;
    int p;
    if (!c)
        return;
    for (p = 0; p <= n; p++) {
        double r = 2.0 * p / n - 1.0, v = 1.0 - r * r;
        if (v < 0.0)
            v = 0.0;
        acc += bessel_i0(M_PI * alpha * sqrt(v)) / i0a;
        c[p] = acc;
    }
    for (p = 0; p < n; p++)
        w[p] = sqrt(c[p] / c[n]);
    free(c);
}

/* ------------------------------------------------------------ FFT
 * Complex FFT of size M = 3 * P, P a power of two: three radix-2 FFTs of
 * the decimated sequences, combined with radix-3 butterflies. */
typedef struct {
    int n;                  /* DCT-IV length N */
    int M, P, logP;
    double *pre_re, *pre_im;    /* exp(-i pi (4n+1) / (4N)), n < M */
    double *post_re, *post_im;  /* exp(-i pi k / N), k < M */
    double *twP_re, *twP_im;    /* exp(-2 pi i j / P), j < P/2 */
    double *twM_re, *twM_im;    /* exp(-2 pi i k / M), k < M */
    int *rev;                   /* bit reversal, P */
} dct4_plan;

struct ac4_fb_tables {
    dct4_plan plan[N_LEN];
    /* left_window(n, n_prev) for every pair: lw[i][j] has length LENS[i],
       n = LENS[i], n_prev = LENS[j] */
    double *lw[N_LEN][N_LEN];
};

static int plan_init(dct4_plan *pl, int n)
{
    int k, P, logP = 0;
    memset(pl, 0, sizeof(*pl));
    pl->n = n;
    pl->M = n / 2;
    if (pl->M % 3)
        return -1;
    P = pl->M / 3;
    while ((1 << logP) < P)
        logP++;
    if ((1 << logP) != P)
        return -1;
    pl->P = P;
    pl->logP = logP;
    pl->pre_re = (double *)malloc(sizeof(double) * (size_t)pl->M * 6);
    pl->twP_re = (double *)malloc(sizeof(double) * (size_t)P);
    pl->rev = (int *)malloc(sizeof(int) * (size_t)P);
    if (!pl->pre_re || !pl->twP_re || !pl->rev)
        return -1;
    pl->pre_im = pl->pre_re + pl->M;
    pl->post_re = pl->pre_re + 2 * pl->M;
    pl->post_im = pl->pre_re + 3 * pl->M;
    pl->twM_re = pl->pre_re + 4 * pl->M;
    pl->twM_im = pl->pre_re + 5 * pl->M;
    pl->twP_im = pl->twP_re + P / 2;
    for (k = 0; k < pl->M; k++) {
        double a = -M_PI * (4.0 * k + 1.0) / (4.0 * n);
        double b = -M_PI * (double)k / (double)n;
        double c = -2.0 * M_PI * (double)k / (double)pl->M;
        pl->pre_re[k] = cos(a);
        pl->pre_im[k] = sin(a);
        pl->post_re[k] = cos(b);
        pl->post_im[k] = sin(b);
        pl->twM_re[k] = cos(c);
        pl->twM_im[k] = sin(c);
    }
    for (k = 0; k < P / 2; k++) {
        double a = -2.0 * M_PI * (double)k / (double)P;
        pl->twP_re[k] = cos(a);
        pl->twP_im[k] = sin(a);
    }
    for (k = 0; k < P; k++) {
        int r = 0, x = k, i;
        for (i = 0; i < logP; i++) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        pl->rev[k] = r;
    }
    return 0;
}

static void plan_free(dct4_plan *pl)
{
    free(pl->pre_re);
    free(pl->twP_re);
    free(pl->rev);
}

/* in-place radix-2 FFT of size P on (re, im) with stride-1 arrays,
   input already in bit-reversed order */
static void fft2(const dct4_plan *pl, double *re, double *im)
{
    int P = pl->P, len, i, j;
    for (len = 2; len <= P; len <<= 1) {
        int half = len >> 1, step = P / len;
        for (i = 0; i < P; i += len) {
            for (j = 0; j < half; j++) {
                double wr = pl->twP_re[j * step], wi = pl->twP_im[j * step];
                double *ar = re + i + j, *ai = im + i + j;
                double *br = re + i + j + half, *bi = im + i + j + half;
                double tr = *br * wr - *bi * wi;
                double ti = *br * wi + *bi * wr;
                *br = *ar - tr;
                *bi = *ai - ti;
                *ar += tr;
                *ai += ti;
            }
        }
    }
}

/* DCT-IV: X[k] = sum x[n] cos(pi/N (n+1/2)(k+1/2)); y receives N values.
   tmp: >= 4*M doubles. */
static void dct4(const dct4_plan *pl, const double *x, double *y,
                 double *tmp)
{
    int N = pl->n, M = pl->M, P = pl->P, n, r, k;
    double *vr = tmp, *vi = tmp + M;          /* decimated, per phase r */
    double *zr = tmp + 2 * M, *zi = tmp + 3 * M;
    /* v[n] = (x[2n] + i x[N-1-2n]) * pre[n]; split by n mod 3 into three
       bit-reversed P-point sequences */
    for (r = 0; r < 3; r++) {
        double *br = vr + r * P, *bi = vi + r * P;
        for (n = 0; n < P; n++) {
            int m = 3 * n + r;
            double a = x[2 * m], b = x[N - 1 - 2 * m];
            double pr = pl->pre_re[m], pi = pl->pre_im[m];
            int d = pl->rev[n];
            br[d] = a * pr - b * pi;
            bi[d] = a * pi + b * pr;
        }
        fft2(pl, br, bi);
    }
    /* radix-3 combine: Z[k + P q] = sum_r W_M^{r(k+Pq)} F_r[k] */
    {
        const double c3 = -0.5, s3 = -0.86602540378443864676; /* W_3 */
        for (k = 0; k < P; k++) {
            double f0r = vr[k], f0i = vi[k];
            double w1r = pl->twM_re[k], w1i = pl->twM_im[k];
            double w2r = pl->twM_re[(2 * k) % M], w2i = pl->twM_im[(2 * k) % M];
            double f1r = vr[P + k] * w1r - vi[P + k] * w1i;
            double f1i = vr[P + k] * w1i + vi[P + k] * w1r;
            double f2r = vr[2 * P + k] * w2r - vi[2 * P + k] * w2i;
            double f2i = vr[2 * P + k] * w2i + vi[2 * P + k] * w2r;
            double sr = f1r + f2r, si = f1i + f2i;
            double dr = f1r - f2r, di = f1i - f2i;
            zr[k] = f0r + sr;
            zi[k] = f0i + si;
            /* q = 1: W3 = c3 + i s3; q = 2: conj */
            zr[k + P] = f0r + c3 * sr - s3 * di;
            zi[k + P] = f0i + c3 * si + s3 * dr;
            zr[k + 2 * P] = f0r + c3 * sr + s3 * di;
            zi[k + 2 * P] = f0i + c3 * si - s3 * dr;
        }
    }
    for (k = 0; k < M; k++) {
        double cr = zr[k] * pl->post_re[k] - zi[k] * pl->post_im[k];
        double ci = zr[k] * pl->post_im[k] + zi[k] * pl->post_re[k];
        y[2 * k] = cr;
        y[N - 1 - 2 * k] = -ci;
    }
}

ac4_fb_tables *ac4_fb_tables_new(void)
{
    ac4_fb_tables *t = (ac4_fb_tables *)calloc(1, sizeof(*t));
    int i, j;
    if (!t)
        return NULL;
    for (i = 0; i < N_LEN; i++) {
        if (plan_init(&t->plan[i], LENS[i]) != 0) {
            ac4_fb_tables_free(t);
            return NULL;
        }
    }
    for (i = 0; i < N_LEN; i++) {
        for (j = 0; j < N_LEN; j++) {
            int n = LENS[i], np = LENS[j];
            int nw = n < np ? n : np, nskip = (n - nw) / 2, k;
            double *w = (double *)malloc(sizeof(double) * (size_t)n);
            if (!w) {
                ac4_fb_tables_free(t);
                return NULL;
            }
            for (k = 0; k < n; k++)
                w[k] = 1.0;
            for (k = 0; k < nskip; k++)
                w[k] = 0.0;
            kbd_left(nw, alpha_for(nw), w + nskip);
            t->lw[i][j] = w;
        }
    }
    return t;
}

void ac4_fb_tables_free(ac4_fb_tables *t)
{
    int i, j;
    if (!t)
        return;
    for (i = 0; i < N_LEN; i++) {
        plan_free(&t->plan[i]);
        for (j = 0; j < N_LEN; j++)
            free(t->lw[i][j]);
    }
    free(t);
}

/* One block: Python synthesise(), mode "reversed". tmp: >= 3*1536. */
void ac4_fb_block(const ac4_fb_tables *t, ac4_fb_state *s, const double *X,
                  int n, double *out, double *tmp)
{
    int i = len_index(n), ip, k, h = n / 2, n_prev, nskip, nskip_prev;
    double *folded = tmp, *work = tmp + 1536;
    double *ov = s->overlap, scale;
    const double *w, *wrv;
    if (i < 0) {                         /* not an AC-4 length: silence */
        memset(out, 0, sizeof(double) * (size_t)(n > 0 ? n : 0));
        return;
    }
    if (s->n_prev == 0)
        s->n_prev = n;
    n_prev = s->n_prev;
    ip = len_index(n_prev);
    dct4(&t->plan[i], X, folded, work);
    scale = sqrt(2.0 * 1536.0) / 2.0;     /* IMDCT_K / 2 */
    for (k = 0; k < n; k++)
        folded[k] = 2.0 * folded[k] / n * scale;
    w = t->lw[i][ip];                    /* left_window(n, n_prev) */
    wrv = t->lw[ip][i];                  /* left_window(n_prev, n), reversed */
    nskip = (1536 - n) / 2;
    nskip_prev = (1536 - n_prev) / 2;
    for (k = 0; k < n_prev; k++)
        ov[nskip_prev + k] *= wrv[n_prev - 1 - k];
    /* first = raw[:n] * w, raw[:n] = [a_b, -a_b reversed], a_b = folded[h:] */
    for (k = 0; k < h; k++)
        ov[nskip + k] += folded[h + k] * w[k];
    for (k = 0; k < h; k++)
        ov[nskip + h + k] += -folded[n - 1 - k] * w[h + k];
    memcpy(out, ov, sizeof(double) * (size_t)n);
    memmove(ov, ov + n, sizeof(double) * (size_t)(1536 - n));
    memset(ov + 1536 - n, 0, sizeof(double) * (size_t)n);
    /* second = raw[n:] = [-c_d reversed, -c_d], c_d = folded[:h] */
    for (k = 0; k < h; k++)
        ov[nskip + k] = -folded[h - 1 - k];
    for (k = 0; k < h; k++)
        ov[nskip + h + k] = -folded[k];
    s->n_prev = n;
}
