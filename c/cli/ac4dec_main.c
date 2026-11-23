/* SPDX-License-Identifier: Apache-2.0 */
/*
 * ac4dec-c -- decode the AC-4 audio track of a fragmented MP4 file to WAV.
 * The C port of `ac4dec` (ac4/cli.py), milestone 1 (core only: no A-SPX,
 * no A-CPL; see c/README.md).
 *
 *   ac4dec-c IN.m4s OUT.wav [--element auto|5_X|pair] [--channels 2|6]
 *            [--seconds S] [--max-pass N] [--ref-peak X] [--f64 RAW]
 *            [--internal] [-q]
 *
 * OUT is 16-bit PCM at 48 kHz, written with the Python CLI's level rule:
 * samples times 0.9 / ref_peak, ref_peak starting at the element's fixed
 * reference (9e5 for 5.1, 2e5 stereo), rising to the loudest sample of a
 * pass, capped at twice the start (anything above it is clipped), applied
 * per decode pass of --max-pass frames before resampling, as there.
 *
 * --f64 RAW also writes the decoder-scale samples (before the level rule)
 * as interleaved little-endian float64 at 48 kHz (or the internal rate
 * with --internal): the comparison output.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ac4dec.h"
#include "ac4dec_mp4.h"

#define FPS (30000.0 / 1001.0)

static void usage(void)
{
    fprintf(stderr,
            "usage: ac4dec-c IN.m4s OUT.wav [--element auto|5_X|pair] "
            "[--channels 2|6]\n"
            "                [--seconds S] [--max-pass N] [--ref-peak X] "
            "[--f64 RAW] [--internal] [-q]\n");
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void put32(unsigned char *p, unsigned v)
{
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}

static void put16(unsigned char *p, unsigned v)
{
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
}

static int wav_header(FILE *f, int ch, unsigned long long data_bytes)
{
    unsigned char h[44];
    unsigned rate = 48000;
    if (data_bytes > 0xFFFFFFFFull - 36)
        data_bytes = 0xFFFFFFFFull - 36;
    memcpy(h, "RIFF", 4);
    put32(h + 4, (unsigned)(36 + data_bytes));
    memcpy(h + 8, "WAVEfmt ", 8);
    put32(h + 16, 16);
    put16(h + 20, 1);
    put16(h + 22, (unsigned)ch);
    put32(h + 24, rate);
    put32(h + 28, rate * 2u * (unsigned)ch);
    put16(h + 32, 2u * (unsigned)ch);
    put16(h + 34, 16);
    memcpy(h + 36, "data", 4);
    put32(h + 40, (unsigned)data_bytes);
    return fwrite(h, 1, 44, f) == 44 ? 0 : -1;
}

/* the Python CLI's LevelPolicy */
typedef struct {
    double burst, ref_peak;
} level_t;

static void level_apply(level_t *lv, double *y, long n, int quiet)
{
    double pk = 0.0;
    long i;
    for (i = 0; i < n; i++)
        if (fabs(y[i]) > pk)
            pk = fabs(y[i]);
    if (pk > lv->burst) {
        if (!quiet)
            fprintf(stdout, "  samples up to %.4g clipped to %.4g\n", pk,
                    lv->ref_peak);
        for (i = 0; i < n; i++) {
            if (y[i] > lv->ref_peak)
                y[i] = lv->ref_peak;
            else if (y[i] < -lv->ref_peak)
                y[i] = -lv->ref_peak;
        }
        pk = 0.0;
        for (i = 0; i < n; i++)
            if (fabs(y[i]) > pk)
                pk = fabs(y[i]);
    }
    if (pk > lv->ref_peak) {
        if (!quiet)
            fprintf(stdout, "  peak %.4g above reference %.4g: gain lowered "
                    "%.1f dB\n", pk, lv->ref_peak,
                    20.0 * log10(pk / lv->ref_peak));
        lv->ref_peak = pk;
    }
    for (i = 0; i < n; i++)
        y[i] = y[i] * (0.9 / lv->ref_peak);
}

static unsigned char *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    unsigned char *d = NULL;
    size_t cap = 0, n = 0;
    if (!f)
        return NULL;
    for (;;) {
        size_t r;
        if (n == cap) {
            size_t nc = cap ? cap * 2 : (1u << 20);
            unsigned char *nd = (unsigned char *)realloc(d, nc);
            if (!nd) {
                free(d);
                fclose(f);
                return NULL;
            }
            d = nd;
            cap = nc;
        }
        r = fread(d + n, 1, cap - n, f);
        n += r;
        if (r == 0)
            break;
    }
    fclose(f);
    *len = n;
    return d;
}

int main(int argc, char **argv)
{
    const char *src = NULL, *dst = NULL, *f64 = NULL;
    const char *el_arg = "auto";
    int channels = 0, quiet = 0, internal = 0, i, c;
    long max_pass = 1800;
    double seconds = -1.0, ref_peak = 0.0, t0, dt;
    unsigned char *data;
    size_t len;
    ac4mp4_sample *smp = NULL;
    long n_frames, f0;
    ac4dec_element element = AC4DEC_ELEMENT_AUTO;
    ac4dec_config cfg;
    ac4dec *dec;
    ac4dec_resampler *rs_wav = NULL, *rs_raw = NULL;
    ac4dec_stats stats;
    FILE *fw, *fr = NULL;
    level_t lv;
    double *pass = NULL, *planar[AC4DEC_MAX_CHANNELS], *rout[AC4DEC_MAX_CHANNELS];
    short *pcm16 = NULL;
    double *ibuf = NULL;
    unsigned long long n_out = 0;
    int n_ch;

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--element") && i + 1 < argc)
            el_arg = argv[++i];
        else if (!strcmp(a, "--channels") && i + 1 < argc)
            channels = atoi(argv[++i]);
        else if (!strcmp(a, "--seconds") && i + 1 < argc)
            seconds = atof(argv[++i]);
        else if (!strcmp(a, "--max-pass") && i + 1 < argc)
            max_pass = atol(argv[++i]);
        else if (!strcmp(a, "--ref-peak") && i + 1 < argc)
            ref_peak = atof(argv[++i]);
        else if (!strcmp(a, "--f64") && i + 1 < argc)
            f64 = argv[++i];
        else if (!strcmp(a, "--internal"))
            internal = 1;
        else if (!strcmp(a, "-q") || !strcmp(a, "--quiet"))
            quiet = 1;
        else if (a[0] == '-' && a[1]) {
            usage();
            return 2;
        } else if (!src)
            src = a;
        else if (!dst)
            dst = a;
        else {
            usage();
            return 2;
        }
    }
    if (!src || !dst || (channels != 0 && channels != 2 && channels != 6)) {
        usage();
        return 2;
    }
    if (!strcmp(el_arg, "5_X"))
        element = AC4DEC_ELEMENT_5X;
    else if (!strcmp(el_arg, "pair"))
        element = AC4DEC_ELEMENT_PAIR;
    else if (strcmp(el_arg, "auto")) {
        usage();
        return 2;
    }

    data = read_file(src, &len);
    if (!data) {
        fprintf(stderr, "cannot read %s\n", src);
        return 1;
    }
    n_frames = ac4mp4_samples(data, len, &smp);
    if (n_frames < 0) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    if (seconds >= 0.0) {
        long lim = (long)floor(seconds * FPS + 0.5);
        if (lim < n_frames)
            n_frames = lim < 0 ? 0 : lim;
    }
    if (n_frames == 0) {
        if (!quiet)
            printf("no AC-4 frames found in %s\n", src);
        return 1;
    }
    if (element == AC4DEC_ELEMENT_AUTO) {
        /* as the Python probe_element: the first TOC that parses decides */
        for (f0 = 0; f0 < n_frames && f0 < 64; f0++) {
            const uint8_t *p = data + smp[f0].off;
            if (!smp[f0].len)
                continue;
            element = ac4dec_probe(p, smp[f0].len);
            if (element != AC4DEC_ELEMENT_AUTO)
                break;
        }
        if (element == AC4DEC_ELEMENT_AUTO) {
            if (!quiet)
                printf("cannot tell the element (5.1 or stereo) from the "
                       "stream; pass --element\n");
            return 1;
        }
    }
    if (max_pass < 8)
        max_pass = 8;

    t0 = now();
    memset(&cfg, 0, sizeof(cfg));
    cfg.element = element;
    cfg.channels = channels;
    cfg.internal_rate = 1;          /* the level rule runs before resampling */
    dec = ac4dec_open(&cfg);
    if (!dec) {
        fprintf(stderr, "cannot open the decoder\n");
        return 1;
    }
    n_ch = (element == AC4DEC_ELEMENT_PAIR) ? 2 : (channels ? channels : 6);
    lv.ref_peak = ref_peak > 0.0 ? ref_peak : ac4dec_ref_peak(element);
    lv.burst = 2.0 * lv.ref_peak;
    if (lv.ref_peak > lv.burst)
        lv.ref_peak = lv.burst;
    rs_wav = ac4dec_resampler_open(n_ch);
    if (f64 && !internal)
        rs_raw = ac4dec_resampler_open(n_ch);
    pass = (double *)malloc(sizeof(double) * (size_t)max_pass * 1536
                            * (size_t)n_ch);
    ibuf = (double *)malloc(sizeof(double) * (size_t)max_pass * 1536
                            * (size_t)n_ch);
    for (c = 0; c < n_ch; c++)
        rout[c] = (double *)malloc(sizeof(double)
                                   * ((size_t)max_pass * 1602 + 8));
    pcm16 = (short *)malloc(sizeof(short) * ((size_t)max_pass * 1602 + 8)
                            * (size_t)n_ch);
    if (!rs_wav || (f64 && !internal && !rs_raw) || !pass || !ibuf || !pcm16) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    for (c = 0; c < n_ch; c++)
        if (!rout[c]) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
    fw = fopen(dst, "wb");
    if (!fw || wav_header(fw, n_ch, 0) != 0) {
        fprintf(stderr, "cannot write %s\n", dst);
        return 1;
    }
    if (f64) {
        fr = fopen(f64, "wb");
        if (!fr) {
            fprintf(stderr, "cannot write %s\n", f64);
            return 1;
        }
    }

    for (f0 = 0; f0 < n_frames; f0 += max_pass) {
        long nf = n_frames - f0 < max_pass ? n_frames - f0 : max_pass, k;
        long ns = nf * 1536, s;
        int no;
        /* decode a pass: planar per channel */
        for (c = 0; c < n_ch; c++)
            planar[c] = pass + (size_t)c * (size_t)ns;
        for (k = 0; k < nf; k++) {
            ac4dec_frame fo;
            const ac4mp4_sample *sm = &smp[f0 + k];
            ac4dec_decode_frame(dec, sm->len ? data + sm->off : NULL, sm->len,
                                &fo);
            for (c = 0; c < n_ch; c++) {
                if (c < fo.n_channels && fo.n_samples == 1536)
                    memcpy(planar[c] + k * 1536, fo.pcm[c],
                           sizeof(double) * 1536);
                else
                    memset(planar[c] + k * 1536, 0, sizeof(double) * 1536);
            }
        }
        if (fr) {
            /* decoder-scale comparison output */
            if (internal) {
                for (s = 0; s < ns; s++)
                    for (c = 0; c < n_ch; c++)
                        ibuf[s * n_ch + c] = planar[c][s];
                fwrite(ibuf, sizeof(double), (size_t)(ns * n_ch), fr);
            } else {
                no = ac4dec_resampler_push(rs_raw,
                                           (const double *const *)planar,
                                           (int)ns, rout);
                for (s = 0; s < no; s++)
                    for (c = 0; c < n_ch; c++)
                        ibuf[s * n_ch + c] = rout[c][s];
                fwrite(ibuf, sizeof(double), (size_t)no * (size_t)n_ch, fr);
            }
        }
        /* the level rule sees the whole pass (all channels) at once */
        level_apply(&lv, pass, ns * n_ch, quiet);
        no = ac4dec_resampler_push(rs_wav, (const double *const *)planar,
                                   (int)ns, rout);
        for (s = 0; s < no; s++) {
            for (c = 0; c < n_ch; c++) {
                double v = rout[c][s];
                if (v > 1.0)
                    v = 1.0;
                else if (v < -1.0)
                    v = -1.0;
                {
                    double x = v * 32767.0;
                    short q = (short)x;          /* truncation, as numpy */
                    unsigned char *b = (unsigned char *)&pcm16[s * n_ch + c];
                    b[0] = (unsigned char)((unsigned short)q & 0xff);
                    b[1] = (unsigned char)((unsigned short)q >> 8);
                }
            }
        }
        fwrite(pcm16, 2, (size_t)no * (size_t)n_ch, fw);
        n_out += (unsigned long long)no;
    }
    fseek(fw, 0, SEEK_SET);
    wav_header(fw, n_ch, n_out * 2ull * (unsigned long long)n_ch);
    fclose(fw);
    if (fr)
        fclose(fr);
    dt = now() - t0;
    ac4dec_get_stats(dec, &stats);
    if (!quiet) {
        long cnt;
        const char *r;
        int first = 1;
        for (i = 0; (r = ac4dec_reason(dec, i, &cnt)) != NULL; i++) {
            printf(first ? "  concealed frames by reason: %s %ld"
                         : ", %s %ld", r, cnt);
            first = 0;
        }
        if (!first)
            printf("\n");
        if (stats.acpl3)
            printf("  stream is ASPX_ACPL_3 (2-ch core + A-CPL): rendering "
                   "L/R; the other channels are written silent\n");
        printf("%ld frames (%ld bad, %ld concealed, %ld sequence jumps), "
               "element %s, %d ch, %.3f s -> %s  [%.2f s, %.2fx real time; "
               "core only: A-SPX/A-CPL not rendered]\n",
               n_frames, stats.n_bad + stats.n_concealed, stats.n_concealed,
               stats.n_seq_jumps,
               element == AC4DEC_ELEMENT_PAIR ? "pair" : "5_X", n_ch,
               (double)n_out / 48000.0, dst, dt,
               dt > 0 ? ((double)n_out / 48000.0) / dt : 0.0);
    }
    ac4dec_close(dec);
    ac4dec_resampler_close(rs_wav);
    ac4dec_resampler_close(rs_raw);
    free(pass);
    free(ibuf);
    free(pcm16);
    for (c = 0; c < n_ch; c++)
        free(rout[c]);
    free(smp);
    free(data);
    return 0;
}
