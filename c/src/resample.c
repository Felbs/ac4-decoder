/* SPDX-License-Identifier: Apache-2.0 */
/* The 1001/960 output resampler (TS 103 190-1 Table 83: 1001/1000 x 25/24
 * for 29.97 fps at 48 kHz). Port of ac4/resample.py StreamResampler, which
 * is scipy.signal.resample_poly(x, 1001, 960) (Kaiser window, beta 5,
 * 20021 taps, cutoff 1/1001) on the input delayed by 16 samples, fed in
 * pieces. Written out as the direct convolution it amounts to:
 *
 *     y[m] = sum_N xd[N] h[960 m + 10010 - 1001 N],  0 <= index < 20021
 *
 * with xd the input delayed by DELAY = 16 samples (zeros before). After n
 * input samples in total exactly ceil(n * 1001 / 960) outputs have been
 * given, as in the Python. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ac4_internal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define UP 1001
#define DOWN 960
#define HALF (10 * UP)          /* 10010 */
#define NTAPS (2 * HALF + 1)    /* 20021 */
#define DELAY 16

struct ac4_rs_filter {
    double h[NTAPS];
};

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

ac4_rs_filter *ac4_rs_filter_new(void)
{
    ac4_rs_filter *f = (ac4_rs_filter *)malloc(sizeof(*f));
    double fc = 1.0 / UP, alpha = 0.5 * (NTAPS - 1), s = 0.0;
    double beta = 5.0, i0b = bessel_i0(beta);
    int n;
    if (!f)
        return NULL;
    /* firwin(NTAPS, 1/1001, window=('kaiser', 5.0)), scaled to unit DC
       gain, times UP (scipy resample_poly) */
    for (n = 0; n < NTAPS; n++) {
        double m = n - alpha, x = fc * m, sinc, r, win;
        sinc = x == 0.0 ? 1.0 : sin(M_PI * x) / (M_PI * x);
        r = (n - alpha) / alpha;
        win = bessel_i0(beta * sqrt(1.0 - r * r > 0.0 ? 1.0 - r * r : 0.0))
            / i0b;
        f->h[n] = fc * sinc * win;
    }
    for (n = 0; n < NTAPS; n++)
        s += f->h[n];
    for (n = 0; n < NTAPS; n++)
        f->h[n] = f->h[n] / s * UP;
    return f;
}

void ac4_rs_filter_free(ac4_rs_filter *f)
{
    free(f);
}

void ac4_rs_init(ac4_rs_state *s)
{
    memset(s, 0, sizeof(*s));
    s->n_in = DELAY;            /* the delay's zeros are already "in" */
}

static long long floor_div(long long a, long long b)
{
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0)))
        q--;
    return q;
}

/* x: n new input samples. out: room for ceil(n * 1001 / 960) + 1. */
int ac4_rs_push(const ac4_rs_filter *f, ac4_rs_state *s, const double *x,
                int n, double *out)
{
    /* working buffer: the history, then the new samples */
    double buf[AC4_RS_HIST + 4096];
    long long base, m, m_hi, n_real;
    int nout = 0, done = 0;
    while (done < n) {
        int chunk = n - done > 4096 ? 4096 : n - done, i;
        memcpy(buf, s->hist, sizeof(s->hist));
        memcpy(buf + AC4_RS_HIST, x + done, sizeof(double) * (size_t)chunk);
        base = s->n_in - AC4_RS_HIST;    /* xd index of buf[0] */
        s->n_in += chunk;
        n_real = s->n_in - DELAY;
        m_hi = (n_real * UP + DOWN - 1) / DOWN;
        for (m = s->n_out; m < m_hi; m++) {
            long long c = m * DOWN + HALF;
            long long n_lo = floor_div(c - (NTAPS - 1) + UP - 1, UP);
            long long n_hi = floor_div(c, UP), N;
            double acc = 0.0;
            if (n_lo < 0)
                n_lo = 0;
            for (N = n_lo; N <= n_hi; N++) {
                long long bi = N - base;
                if (bi < 0 || bi >= AC4_RS_HIST + chunk)
                    continue;            /* outside: zero (never reached) */
                acc += buf[bi] * f->h[c - N * UP];
            }
            out[nout++] = acc;
        }
        s->n_out = m_hi;
        for (i = 0; i < AC4_RS_HIST; i++)
            s->hist[i] = buf[chunk + i];
        done += chunk;
    }
    return nout;
}

/* ------------------------------------------------------------ public */
#include "ac4dec.h"

struct ac4dec_resampler {
    ac4_rs_filter *f;
    int ch;
    ac4_rs_state st[AC4DEC_MAX_CHANNELS];
};

ac4dec_resampler *ac4dec_resampler_open(int channels)
{
    ac4dec_resampler *r;
    int c;
    if (channels < 1 || channels > AC4DEC_MAX_CHANNELS)
        return NULL;
    r = (ac4dec_resampler *)calloc(1, sizeof(*r));
    if (!r)
        return NULL;
    r->f = ac4_rs_filter_new();
    if (!r->f) {
        free(r);
        return NULL;
    }
    r->ch = channels;
    for (c = 0; c < channels; c++)
        ac4_rs_init(&r->st[c]);
    return r;
}

void ac4dec_resampler_close(ac4dec_resampler *r)
{
    if (!r)
        return;
    ac4_rs_filter_free(r->f);
    free(r);
}

int ac4dec_resampler_push(ac4dec_resampler *r, const double *const *in,
                          int n, double *const *out)
{
    int c, k = 0;
    if (!r || n <= 0)
        return 0;
    for (c = 0; c < r->ch; c++)
        k = ac4_rs_push(r->f, &r->st[c], in[c], n, out[c]);
    return k;
}
