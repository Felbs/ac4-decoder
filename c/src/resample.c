/* SPDX-License-Identifier: Apache-2.0 */
/* The output resampler (TS 103 190-1 Tables 83 / 84: the decoder
 * resampling ratio up / down = 1001/960 (1001/1000 x 25/24), 25/24 or
 * 15/16; 1 is no resampling). Port of ac4/resample.py StreamResampler,
 * which is scipy.signal.resample_poly(x, up, down) (Kaiser window, beta 5,
 * 2 * half + 1 taps with half = 10 max(up, down), cutoff 1 / max(up,
 * down)) on the input delayed by 16 samples, fed in pieces. Written out as
 * the direct convolution it amounts to:
 *
 *     y[m] = sum_N xd[N] h[down m + half - up N],  0 <= index < 2 half + 1
 *
 * with xd the input delayed by DELAY = 16 samples (zeros before). After n
 * input samples in total exactly ceil(n * up / down) outputs have been
 * given, as in the Python. With up == down the samples pass through. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ac4_internal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DELAY 16

struct ac4_rs_filter {
    int up, down, half, ntaps, nph;
    double *h;
    /* polyphase: ph[phi * nph + j] = h[phi + up * j] (0 past the end) */
    double *ph;
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

static int gcd(int a, int b)
{
    while (b) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

ac4_rs_filter *ac4_rs_filter_new_ratio(int up, int down)
{
    ac4_rs_filter *f;
    double fc, alpha, s = 0.0, beta = 5.0, i0b = bessel_i0(beta);
    int n, g, mx;
    if (up < 1 || down < 1)
        return NULL;
    g = gcd(up, down);
    up /= g;
    down /= g;
    f = (ac4_rs_filter *)calloc(1, sizeof(*f));
    if (!f)
        return NULL;
    f->up = up;
    f->down = down;
    if (up == down)
        return f;                       /* ratio 1: pass through */
    mx = up > down ? up : down;
    f->half = 10 * mx;
    f->ntaps = 2 * f->half + 1;
    f->nph = (f->ntaps + up - 1) / up;
    f->nph = (f->nph + 2) / 3 * 3;      /* three partial sums */
    f->h = (double *)malloc(sizeof(double) * (size_t)f->ntaps);
    f->ph = (double *)calloc((size_t)up * (size_t)f->nph, sizeof(double));
    if (!f->h || !f->ph) {
        ac4_rs_filter_free(f);
        return NULL;
    }
    fc = 1.0 / mx;
    alpha = 0.5 * (f->ntaps - 1);
    /* firwin(ntaps, 1/max, window=('kaiser', 5.0)), scaled to unit DC
       gain, times up (scipy resample_poly) */
    for (n = 0; n < f->ntaps; n++) {
        double m = n - alpha, x = fc * m, sinc, r, win;
        sinc = x == 0.0 ? 1.0 : sin(M_PI * x) / (M_PI * x);
        r = (n - alpha) / alpha;
        win = bessel_i0(beta * sqrt(1.0 - r * r > 0.0 ? 1.0 - r * r : 0.0))
            / i0b;
        f->h[n] = fc * sinc * win;
    }
    for (n = 0; n < f->ntaps; n++)
        s += f->h[n];
    for (n = 0; n < f->ntaps; n++)
        f->h[n] = f->h[n] / s * up;
    for (n = 0; n < up; n++) {
        int j;
        for (j = 0; j < f->nph; j++)
            f->ph[n * f->nph + j] = n + up * j < f->ntaps
                                    ? f->h[n + up * j] : 0.0;
    }
    return f;
}

ac4_rs_filter *ac4_rs_filter_new(void)
{
    return ac4_rs_filter_new_ratio(1001, 960);
}

int ac4_rs_ratio(const ac4_rs_filter *f, int *up, int *down)
{
    *up = f->up;
    *down = f->down;
    return f->up == f->down;
}

void ac4_rs_filter_free(ac4_rs_filter *f)
{
    if (!f)
        return;
    free(f->h);
    free(f->ph);
    free(f);
}

void ac4_rs_init(ac4_rs_state *s)
{
    memset(s, 0, sizeof(*s));
    s->n_in = DELAY;            /* the delay's zeros are already "in" */
}

/* x: n new input samples. out: room for n * up / down + 2.
 *
 * y[m] = sum_j h[phi + up j] xd[N_hi - j], where c = down m + half,
 * N_hi = c / up and phi = c % up (taps past the filter are zero). */
int ac4_rs_push(const ac4_rs_filter *f, ac4_rs_state *s, const double *x,
                int n, double *out)
{
    /* working buffer: the history, then the new samples */
    double buf[AC4_RS_HIST + 4096];
    long long base, m, m_hi, n_real;
    int nout = 0, done = 0;
    const int UP = f->up, DOWN = f->down, NPH = f->nph;
    if (UP == DOWN) {
        if (n > 0)
            memmove(out, x, sizeof(double) * (size_t)n);
        s->n_out += n;
        return n > 0 ? n : 0;
    }
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
            long long c = s->n_out * DOWN + f->half;
            long long nh = c / UP;
            int phi = (int)(c % UP);
            for (m = s->n_out; m < m_hi; m++) {
                const double *hp = f->ph + (size_t)phi * (size_t)NPH;
                const double *xp = buf + (nh - base);
                double a0 = 0.0, a1 = 0.0, a2 = 0.0;
                int j;
                /* nh - base - j >= 0 always: the history reaches back
                   further than the filter. Three partial sums shorten the
                   dependency chain; the order differs from scipy's in the
                   last bits only. */
                for (j = 0; j < NPH; j += 3) {
                    a0 += hp[j] * xp[-j];
                    a1 += hp[j + 1] * xp[-j - 1];
                    a2 += hp[j + 2] * xp[-j - 2];
                }
                out[nout++] = (a0 + a1) + a2;
                phi += DOWN;
                while (phi >= UP) {
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
    return ac4dec_resampler_open_ratio(channels, 1001, 960);
}

ac4dec_resampler *ac4dec_resampler_open_ratio(int channels, int up, int down)
{
    ac4dec_resampler *r;
    int c;
    if (channels < 1 || channels > AC4DEC_MAX_CHANNELS)
        return NULL;
    r = (ac4dec_resampler *)calloc(1, sizeof(*r));
    if (!r)
        return NULL;
    r->f = ac4_rs_filter_new_ratio(up, down);
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
