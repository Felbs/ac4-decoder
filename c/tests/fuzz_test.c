/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Damaged-input gate for libac4dec.
 *
 *   tests/fuzz_test [CAPTURE.m4s ...]
 *
 * Without captures: garbage, empty, NULL and truncated frames, and damaged
 * MP4 boxes. With captures (arguments, or AC4_TEST_FILES as a colon-
 * separated list): each capture is decoded clean, then damaged
 * deterministically (bit flips, byte bursts, truncation, zeroed bytes,
 * garbage, empty, swapped, repeated and extended frames, runs of damaged
 * frames, a damaged first-frame TOC) over several seeds, and the MP4
 * container itself is damaged.
 *
 * Required: no crash (build the _asan variant to check memory errors too),
 * every frame gives exactly the samples the timeline owes it, every sample
 * is finite, every frame comes out exactly once and in order (the decoder
 * is a delay line; ac4dec_flush drains it). Reported (the gate on levels is
 * c/tests/robust_compare.py): concealed counts and the worst peak against
 * the clean decode's own peak.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac4dec.h"
#include "ac4dec_input.h"
#include "ac4dec_mp4.h"

static unsigned long long rng_s = 88172645463325252ull;

static unsigned rnd(void)
{
    rng_s ^= rng_s << 13;
    rng_s ^= rng_s >> 7;
    rng_s ^= rng_s << 17;
    return (unsigned)(rng_s >> 11);
}

static int fails;

#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

typedef struct {
    const uint8_t *p;
    size_t n;
} frame_t;

/* one output frame: timing, order, finiteness */
typedef struct {
    long next;                 /* the index expected next */
    long total;                /* samples given so far */
    int internal;
    long n_conc;
    double pk;
    const double *clean_pk;
    double *worst;
} acc_t;

/* garbage TOCs of bitstream_version 0 / 1 parse often enough that two in
   a row can agree on a rate other than 29.97 fps: the sample count per
   frame then follows that rate */
static int g_any_rate;
/* the stream's frame length and resampling ratio (capture mode: from
   its first frame; 29.97 fps otherwise) */
static long g_fl = 1536, g_up = 1001, g_down = 960;

static void take(acc_t *a, const ac4dec_frame *o)
{
    long want;
    int c, k;
    double fpk = 0.0;
    CHECK(o->status == AC4DEC_OK || o->status == AC4DEC_CONCEALED
          || o->status == AC4DEC_BAD, "bad status %d", (int)o->status);
    CHECK(o->index == a->next, "frame %lld out, %ld expected", o->index,
          a->next);
    if (o->status != AC4DEC_OK)
        a->n_conc++;
    if (a->internal) {
        want = g_fl;
    } else {
        long long tot = (long long)(a->next + 1) * g_fl;
        want = (long)((tot * g_up + g_down - 1) / g_down) - a->total;
    }
    if (g_any_rate)                     /* garbage may set any rate */
        want = o->n_samples > 0 && o->n_samples <= AC4DEC_MAX_OUT
               ? o->n_samples : -1;
    CHECK(o->n_samples == want, "frame %ld: %d samples, want %ld", a->next,
          o->n_samples, want);
    a->total += o->n_samples;
    a->next++;
    CHECK(o->n_channels >= 1 && o->n_channels <= AC4DEC_MAX_CHANNELS,
          "channels %d",
          o->n_channels);
    for (c = 0; c < o->n_channels; c++)
        for (k = 0; k < o->n_samples; k++) {
            double v = o->pcm[c][k];
            if (!isfinite(v)) {
                CHECK(0, "non-finite sample");
                break;
            }
            if (fabs(v) > fpk)
                fpk = fabs(v);
        }
    if (fpk > a->pk)
        a->pk = fpk;
    if (a->clean_pk && a->worst && *a->clean_pk > 0) {
        double r = fpk / *a->clean_pk;
        if (r > *a->worst)
            *a->worst = r;
    }
}

/* decode a frame list; -> peak, checks timing, order and finiteness */
static double run_la(frame_t *fr, long n, ac4dec_element el, int internal,
                     int lookahead, long *n_conc, double *clean_pk,
                     double *worst_ratio)
{
    ac4dec_config cfg;
    ac4dec *d;
    ac4dec_frame o;
    long i;
    acc_t a;
    memset(&cfg, 0, sizeof(cfg));
    memset(&a, 0, sizeof(a));
    cfg.element = el;
    cfg.internal_rate = internal;
    cfg.lookahead = lookahead;
    d = ac4dec_open(&cfg);
    if (!d) {
        CHECK(0, "open failed");
        return 0.0;
    }
    a.internal = internal;
    a.clean_pk = clean_pk;
    a.worst = worst_ratio;
    for (i = 0; i < n; i++)
        if (ac4dec_decode_frame(d, fr[i].p, fr[i].n, &o))
            take(&a, &o);
    while (ac4dec_flush(d, &o))
        take(&a, &o);
    CHECK(a.next == n, "%ld frames in, %ld out", n, a.next);
    *n_conc = a.n_conc;
    ac4dec_close(d);
    return a.pk;
}

static double run(frame_t *fr, long n, ac4dec_element el, int internal,
                  long *n_conc, double *clean_pk, double *worst_ratio)
{
    return run_la(fr, n, el, internal, 0, n_conc, clean_pk, worst_ratio);
}

static void synthetic(void)
{
    static uint8_t buf[300][4000];
    frame_t fr[300];
    long n_conc, i;
    int el, internal;
    for (i = 0; i < 300; i++) {
        size_t n = rnd() % 4000, k;
        for (k = 0; k < n; k++)
            buf[i][k] = (uint8_t)rnd();
        if (i % 7 == 0)
            n = 0;
        if (i % 11 == 0)
            n = rnd() % 6;
        if (i % 13 == 0) {
            /* a plausible TOC head: version 2, counter, 48 kHz, 29.97 */
            buf[i][0] = 0x80;
            buf[i][1] = (uint8_t)(i & 0x0f);
            buf[i][2] = 0x0d;
        }
        fr[i].p = (i % 17 == 0) ? NULL : buf[i];
        fr[i].n = fr[i].p ? n : 0;
    }
    /* garbage of any bitstream_version can set any rate: the per-frame
       sample count is checked against the rate the decoder took */
    g_any_rate = 1;
    /* auto, 5_X, pair, and the immersive element on garbage */
    for (el = 0; el <= 3; el++)
        for (internal = 0; internal <= 1; internal++) {
            ac4dec_element e = el == 3 ? AC4DEC_ELEMENT_ICE
                                       : (ac4dec_element)el;
            run(fr, 300, e, internal, &n_conc, NULL, NULL);
            run_la(fr, 300, e, internal, AC4DEC_MIN_LOOKAHEAD, &n_conc, NULL,
                   NULL);
            run_la(fr, 3, e, internal, 0, &n_conc, NULL, NULL);
            run_la(fr, 0, e, internal, 0, &n_conc, NULL, NULL);
        }
    /* the MP4 reader on garbage */
    for (i = 0; i < 200; i++) {
        ac4mp4_sample *s = NULL;
        size_t n = rnd() % 4000, k;
        long cnt;
        for (k = 0; k < n; k++)
            buf[0][k] = (uint8_t)rnd();
        if (n >= 16 && i % 2) {
            memcpy(buf[0] + 4, "moof", 4);
            buf[0][0] = 0;
            buf[0][1] = 0;
        }
        cnt = ac4mp4_samples(buf[0], n, &s);
        CHECK(cnt >= 0, "mp4 reader failed on garbage");
        free(s);
    }
    g_any_rate = 0;
    printf("synthetic garbage: done\n");
}

enum { K_FLIP, K_BURST, K_TRUNC, K_ZERO, K_GARBAGE, K_EMPTY, K_SWAP,
       K_REPEAT, K_EXTEND, K_RUN, K_TOC0, K_COUNT };
static const char *KNAME[K_COUNT] = { "bitflip", "burst", "truncate",
    "zeroed", "garbage", "empty", "swap", "repeat", "extend", "run",
    "toc0" };

static void capture(const char *path, int seeds)
{
    FILE *f = fopen(path, "rb");
    uint8_t *data;
    long sz, n, i, n_conc, k;
    ac4mp4_sample *smp = NULL;
    frame_t *fr, *dm;
    uint8_t **own;
    double clean_pk, worst = 0.0;
    long conc_total = 0, dmg_total = 0;
    ac4dec_element el;
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path);
        fails++;
        return;
    }
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    data = (uint8_t *)malloc((size_t)sz);
    if (!data || fread(data, 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        fails++;
        return;
    }
    fclose(f);
    n = ac4mp4_samples(data, (size_t)sz, &smp);
    if (n > 600)
        n = 600;                       /* 20 s per capture is plenty */
    if (n <= 0) {
        fprintf(stderr, "%s: no frames\n", path);
        fails++;
        free(data);
        return;
    }
    fr = (frame_t *)malloc(sizeof(frame_t) * (size_t)n);
    dm = (frame_t *)malloc(sizeof(frame_t) * (size_t)(n + 32));
    own = (uint8_t **)calloc((size_t)(n + 32), sizeof(uint8_t *));
    for (i = 0; i < n; i++) {
        fr[i].p = data + smp[i].off;
        fr[i].n = smp[i].len;
    }
    el = AC4DEC_ELEMENT_AUTO;
    for (i = 0; i < n && el == AC4DEC_ELEMENT_AUTO; i++)
        el = ac4dec_probe(fr[i].p, fr[i].n);
    {
        int fs, fri, fl, up, dn, orate;
        double fps;
        g_fl = 1536;
        g_up = 1001;
        g_down = 960;
        for (i = 0; i < n; i++)
            if (ac4dec_probe_rate(fr[i].p, fr[i].n, &fs, &fri) == 0) {
                if (ac4dec_rate_info(fs, fri, &fl, &up, &dn, &orate,
                                     &fps) == 0) {
                    g_fl = fl;
                    g_up = up;
                    g_down = dn;
                }
                break;
            }
    }
    clean_pk = run(fr, n, el, 0, &n_conc, NULL, NULL);
    CHECK(n_conc == 0, "%s: %ld clean frames concealed", path, n_conc);
    {
        long n2;
        run_la(fr, n, el, 1, AC4DEC_MIN_LOOKAHEAD, &n2, NULL, NULL);
        CHECK(n2 == 0, "%s: %ld clean frames concealed (lookahead %d)", path,
              n2, AC4DEC_MIN_LOOKAHEAD);
    }
    for (k = 0; k < K_COUNT; k++) {
        int s;
        for (s = 0; s < seeds; s++) {
            long m = n, j, hits = 0;
            double wr = 0.0;
            rng_s = 0x9E3779B97F4A7C15ull * (unsigned long long)(k * 101 + s + 1);
            for (i = 0; i < n; i++)
                dm[i] = fr[i];
            for (i = 0; i < n + 32; i++) {
                free(own[i]);
                own[i] = NULL;
            }
            for (j = 0; j < 6; j++) {
                long at = k == K_TOC0 ? 0
                    : 1 + (long)(rnd() % (unsigned)(n - 2));
                uint8_t *cp;
                size_t L = dm[at].n, q;
                hits++;
                switch (k) {
                case K_SWAP:
                    if (at + 1 < m) {
                        frame_t t = dm[at];
                        dm[at] = dm[at + 1];
                        dm[at + 1] = t;
                    }
                    continue;
                case K_REPEAT:
                    if (at > 0)
                        dm[at] = dm[at - 1];
                    continue;
                case K_EMPTY:
                    dm[at].n = 0;
                    continue;
                default:
                    break;
                }
                cp = (uint8_t *)malloc(L + 64);
                if (!cp)
                    continue;
                memcpy(cp, dm[at].p, L);
                free(own[at]);
                own[at] = cp;
                dm[at].p = cp;
                switch (k) {
                case K_FLIP:
                    for (q = 0; q < 3 && L; q++)
                        cp[rnd() % L] ^= (uint8_t)(1u << (rnd() % 8));
                    break;
                case K_BURST:
                    if (L > 8) {
                        size_t o = rnd() % (L - 8);
                        for (q = 0; q < 8; q++)
                            cp[o + q] = (uint8_t)rnd();
                    }
                    break;
                case K_TRUNC:
                    dm[at].n = L ? rnd() % L : 0;
                    break;
                case K_ZERO:
                    if (L > 16) {
                        size_t o = rnd() % (L - 16);
                        memset(cp + o, 0, 16);
                    }
                    break;
                case K_GARBAGE:
                    for (q = 0; q < L; q++)
                        cp[q] = (uint8_t)rnd();
                    break;
                case K_EXTEND:
                    for (q = 0; q < 64; q++)
                        cp[L + q] = (uint8_t)rnd();
                    dm[at].n = L + 64;
                    break;
                case K_RUN: {
                    long r, len = 3 + (long)(rnd() % 18);
                    for (r = at; r < at + len && r < m; r++) {
                        if (r != at) {
                            uint8_t *c2 = (uint8_t *)malloc(dm[r].n + 1);
                            if (!c2)
                                break;
                            memcpy(c2, dm[r].p, dm[r].n);
                            free(own[r]);
                            own[r] = c2;
                            dm[r].p = c2;
                        }
                        for (q = 0; q < 4 && dm[r].n; q++)
                            ((uint8_t *)dm[r].p)[rnd() % dm[r].n] ^= 0xff;
                    }
                    break;
                }
                case K_TOC0:
                    if (L > 4)
                        cp[1 + rnd() % 4] ^= (uint8_t)(1u << (rnd() % 8));
                    j = 6;
                    break;
                default:
                    break;
                }
            }
            {
                double pk1 = clean_pk;
                run(dm, m, el, 0, &n_conc, &pk1, &wr);
                if (wr > worst)
                    worst = wr;
                conc_total += n_conc;
                dmg_total += hits;
            }
        }
        printf("  %-9s ok\n", KNAME[k]);
        fflush(stdout);
    }
    /* damaged container */
    for (k = 0; k < 40; k++) {
        uint8_t *cp = (uint8_t *)malloc((size_t)sz);
        ac4mp4_sample *s2 = NULL;
        long cnt, q;
        memcpy(cp, data, (size_t)sz);
        rng_s = 1234567ull + (unsigned long long)k;
        for (q = 0; q < 8; q++)
            cp[rnd() % (unsigned long)(sz > 4096 ? 4096 : sz)] ^= (uint8_t)rnd();
        for (q = 0; q < 8; q++)
            cp[rnd() % (unsigned long)sz] ^= (uint8_t)rnd();
        cnt = ac4mp4_samples(cp, (size_t)(k % 2 ? sz : sz - (long)(rnd() % 1000)),
                             &s2);
        CHECK(cnt >= 0, "mp4 reader failed");
        for (q = 0; q < cnt; q++)
            CHECK(s2[q].len == 0 || s2[q].off + s2[q].len <= (size_t)sz,
                  "sample outside the input");
        free(s2);
        free(cp);
    }
    printf("%s: %ld frames, clean peak %.4g; %ld damage hits -> %ld frames "
           "not OK; worst frame peak %.2fx the clean peak\n", path, n,
           clean_pk, dmg_total, conc_total, worst);
    for (i = 0; i < n + 32; i++)
        free(own[i]);
    free(own);
    free(fr);
    free(dm);
    free(smp);
    free(data);
}

/* the input readers (sync frames, TS, MP4) on random and mutated buffers:
   no crash, every frame inside the input or the reader's own buffer */
static void inputs(void)
{
    static const uint8_t seeds[][12] = {
        { 0xAC, 0x41, 0x00, 0x08 }, { 0xAC, 0x40, 0xFF, 0xFF, 0, 0, 9 },
        { 'I', 'D', '3', 4, 0, 0, 0, 0, 0, 3 },
        { 0x47, 0x40, 0x00, 0x10 }, { 0x47, 0x41, 0x01, 0x10, 0, 0, 1, 0xBD },
        { 0, 0, 0, 24, 'f', 't', 'y', 'p' }, { 0, 0, 0, 40, 'm', 'o', 'o', 'v' },
        { 0, 0, 0, 20, 's', 't', 's', 'z' }, { 0, 0, 0, 16, 's', 't', 's', 'c' },
        { 0, 0, 0, 28, 'd', 'a', 'c', '4' },
    };
    int it;
    for (it = 0; it < 3000; it++) {
        size_t n = 1 + rnd() % 4000, i, k;
        uint8_t *d = (uint8_t *)malloc(n);
        ac4in_frame *fr;
        ac4in_info info;
        long m;
        int pk = (int)(rnd() % 5);
        for (i = 0; i < n; i++)
            d[i] = (uint8_t)rnd();
        /* sprinkle structure: sync words, TS packets 188 apart, boxes */
        for (k = 0; k < 40; k++) {
            const uint8_t *sd = seeds[rnd() % (sizeof(seeds) / sizeof(seeds[0]))];
            size_t at = pk == 0 ? (rnd() % (n / 188 + 1)) * 188 : rnd() % n;
            for (i = 0; i < 12 && at + i < n; i++)
                d[at + i] = sd[i];
        }
        if (pk == 1 && n > 4) {
            d[0] = 0xAC;
            d[1] = 0x41;
        }
        m = ac4in_read(d, n, &fr, &info);
        CHECK(m >= 0, "ac4in_read failed on %zu bytes", n);
        for (i = 0; m > 0 && i < (size_t)m; i++) {
            const uint8_t *lo = d, *hi = d + n;
            if (!fr[i].len)
                continue;
            if (info.buf && fr[i].p >= info.buf
                && fr[i].p < info.buf + (size_t)1e9) {
                continue;               /* inside the reader's buffer */
            }
            CHECK(fr[i].p >= lo && fr[i].p + fr[i].len <= hi,
                  "frame %zu outside the input", i);
        }
        free(fr);
        ac4in_free(&info);
        free(d);
    }
    printf("input readers: done\n");
}

int main(int argc, char **argv)
{
    int i, seeds = 3;
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *env = getenv("AC4_TEST_FILES");
    if (getenv("AC4_FUZZ_SEEDS"))
        seeds = atoi(getenv("AC4_FUZZ_SEEDS"));
    synthetic();
    inputs();
    for (i = 1; i < argc; i++)
        capture(argv[i], seeds);
    if (argc < 2 && env && *env) {
        char *list = (char *)malloc(strlen(env) + 1), *p, *save = NULL;
        strcpy(list, env);
        for (p = strtok_r(list, ":", &save); p; p = strtok_r(NULL, ":", &save))
            capture(p, seeds);
        free(list);
    }
    if (fails) {
        printf("FUZZ GATE FAILED (%d)\n", fails);
        return 1;
    }
    printf("FUZZ GATE PASS\n");
    return 0;
}
