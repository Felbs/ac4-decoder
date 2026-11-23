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

#define NPH 21               /* taps per phase, zero padded */

struct ac4_rs_filter {
    double h[NTAPS];
    /* polyphase: ph[phi][j] = h[phi + UP * j] (0 past the end) */
    double ph[UP][NPH];
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
    for (n = 0; n < UP; n++) {
        int j;
        for (j = 0; j < NPH; j++)
            f->ph[n][j] = n + UP * j < NTAPS ? f->h[n + UP * j] : 0.0;
    }
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

/* x: n new input samples. out: room for n * 1001 / 960 + 2.
 *
 * y[m] = sum_j h[phi + 1001 j] xd[N_hi - j], where c = 960 m + 10010,
 * N_hi = c / 1001 and phi = c % 1001 (taps past the filter are zero). */
int ac4_rs_push(const ac4_rs_filter *f, ac4_rs_state *s, const double *x,
                int n, double *out)
{
    /* working buffer: the history, then the new samples */
    double buf[AC4_RS_HIST + 4096];
    long long base, m, m_hi, n_real;
    int nout = 0, done = 0;
    while (done < n) {
        int chunk = n - done > 4096 ? 4096 : n - done, i, silent = 1;
        memcpy(buf, s->hist, sizeof(s->hist));
        memcpy(buf + AC4_RS_HIST, x + done, sizeof(double) * (size_t)chunk);
        for (i = 0; i < AC4_RS_HIST + chunk && silent; i++)
            if (buf[i] != 0.0)
                silent = 0;
        base = s->n_in - AC4_RS_HIST;    /* xd index of buf[0] */
        s->n_in += chunk;
        n_real = s->n_in - DELAY;
        m_hi = (n_real * UP + DOWN - 1) / DOWN;
        if (silent) {
            for (m = s->n_out; m < m_hi; m++)
                out[nout++] = 0.0;
        } else if (s->n_out < m_hi) {
            long long c = s->n_out * DOWN + HALF;
            long long nh = c / UP;
            int phi = (int)(c % UP);
            for (m = s->n_out; m < m_hi; m++) {
                const double *hp = f->ph[phi];
                const double *xp = buf + (nh - base);
                double a0 = 0.0, a1 = 0.0, a2 = 0.0;
                int j;
                /* nh - base - j >= 0 always: the history reaches back
                   further than the filter. Three partial sums (7 taps
                   each) shorten the dependency chain; the order differs
                   from scipy's in the last bits only. */
                for (j = 0; j < NPH; j += 3) {
                    a0 += hp[j] * xp[-j];
                    a1 += hp[j + 1] * xp[-j - 1];
                    a2 += hp[j + 2] * xp[-j - 2];
                }
                out[nout++] = (a0 + a1) + a2;
                phi += DOWN;
                if (phi >= UP) {
                    phi -= UP;
                    nh++;
                }
            }
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
