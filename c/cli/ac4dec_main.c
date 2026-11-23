/* SPDX-License-Identifier: Apache-2.0 */
/*
 * ac4dec-c -- decode the AC-4 audio track of a fragmented MP4 file to WAV.
 * The C port of `ac4dec` (ac4/cli.py): A-SPX rendered, output guard on,
 * as the Python CLI (see include/ac4dec.h).
 *
 *   ac4dec-c IN.m4s OUT.wav [--element auto|5_X|pair] [--channels 2|6]
 *            [--seconds S] [--max-pass N] [--ref-peak X] [--no-guard]
 *            [--lookahead N] [--f64 RAW] [--internal] [--status FILE] [-q]
 *
 * OUT is 16-bit PCM at 48 kHz, written with the Python CLI's level rule:
 * samples times 0.9 / ref_peak, ref_peak starting at the element's fixed
 * reference (9e5 for 5.1, 2e5 stereo), rising to the loudest sample of a
 * pass, capped at twice the start (anything above it is clipped), applied
 * per pass of --max-pass output frames before resampling, as there. The
 * decoder itself is one stream (no pass boundaries); --lookahead sets its
 * delay (default AC4DEC_DEFAULT_LOOKAHEAD).
 *
 * --status FILE writes one line per frame: index, status (0 OK,
 * 1 concealed, 2 bad) and the reason.
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
            "[--no-guard]\n"
            "                [--lookahead N] [--f64 RAW] [--internal] "
            "[--status FILE] [-q]\n");
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

static void level_apply(level_t *lv, double *const *y, int n_ch, long n,
                        int quiet)
{
    double pk = 0.0;
    long i;
    int c;
    for (c = 0; c < n_ch; c++)
        for (i = 0; i < n; i++)
            if (fabs(y[c][i]) > pk)
                pk = fabs(y[c][i]);
    if (pk > lv->burst) {
        if (!quiet)
            fprintf(stdout, "  samples up to %.4g clipped to %.4g\n", pk,
                    lv->ref_peak);
        for (c = 0; c < n_ch; c++)
            for (i = 0; i < n; i++) {
                if (y[c][i] > lv->ref_peak)
                    y[c][i] = lv->ref_peak;
                else if (y[c][i] < -lv->ref_peak)
                    y[c][i] = -lv->ref_peak;
            }
        pk = 0.0;
        for (c = 0; c < n_ch; c++)
            for (i = 0; i < n; i++)
                if (fabs(y[c][i]) > pk)
                    pk = fabs(y[c][i]);
    }
    if (pk > lv->ref_peak) {
        if (!quiet)
            fprintf(stdout, "  peak %.4g above reference %.4g: gain lowered "
                    "%.1f dB\n", pk, lv->ref_peak,
                    20.0 * log10(pk / lv->ref_peak));
        lv->ref_peak = pk;
    }
    for (c = 0; c < n_ch; c++)
        for (i = 0; i < n; i++)
            y[c][i] = y[c][i] * (0.9 / lv->ref_peak);
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

/* the output of one pass (planar), level rule + resampler + writers */
typedef struct {
    int n_ch, internal, quiet;
    long max_pass, nf;
    double *pass, *ibuf, *rout[AC4DEC_MAX_CHANNELS];
    short *pcm16;
    ac4dec_resampler *rs_wav, *rs_raw;
    level_t lv;
    FILE *fw, *fr;
    unsigned long long n_out;
} sink_t;

static void sink_flush(sink_t *k)
{
    double *planar[AC4DEC_MAX_CHANNELS];
    long ns = k->nf * 1536, s;
    int c, no;
    if (k->nf == 0)
        return;
    for (c = 0; c < k->n_ch; c++)
        planar[c] = k->pass + (size_t)c * (size_t)k->max_pass * 1536;
    if (k->fr) {
        /* decoder-scale comparison output */
        if (k->internal) {
            for (s = 0; s < ns; s++)
                for (c = 0; c < k->n_ch; c++)
                    k->ibuf[s * k->n_ch + c] = planar[c][s];
            fwrite(k->ibuf, sizeof(double), (size_t)(ns * k->n_ch), k->fr);
        } else {
            no = ac4dec_resampler_push(k->rs_raw,
                                       (const double *const *)planar,
                                       (int)ns, k->rout);
            for (s = 0; s < no; s++)
                for (c = 0; c < k->n_ch; c++)
                    k->ibuf[s * k->n_ch + c] = k->rout[c][s];
            fwrite(k->ibuf, sizeof(double), (size_t)no * (size_t)k->n_ch,
                   k->fr);
        }
    }
    /* the level rule sees the whole pass (all channels) at once */
    level_apply(&k->lv, planar, k->n_ch, ns, k->quiet);
    no = ac4dec_resampler_push(k->rs_wav, (const double *const *)planar,
                               (int)ns, k->rout);
    for (s = 0; s < no; s++) {
        for (c = 0; c < k->n_ch; c++) {
            double v = k->rout[c][s];
            if (v > 1.0)
                v = 1.0;
            else if (v < -1.0)
                v = -1.0;
            {
                double x = v * 32767.0;
                short q = (short)x;              /* truncation, as numpy */
                unsigned char *b = (unsigned char *)&k->pcm16[s * k->n_ch + c];
                b[0] = (unsigned char)((unsigned short)q & 0xff);
                b[1] = (unsigned char)((unsigned short)q >> 8);
            }
        }
    }
    fwrite(k->pcm16, 2, (size_t)no * (size_t)k->n_ch, k->fw);
    k->n_out += (unsigned long long)no;
    k->nf = 0;
}

static void sink_frame(sink_t *k, const ac4dec_frame *fo, FILE *st)
{
    int c;
    long off = k->nf * 1536;
    for (c = 0; c < k->n_ch; c++) {
        double *dst = k->pass + (size_t)c * (size_t)k->max_pass * 1536 + off;
        if (c < fo->n_channels && fo->n_samples == 1536)
            memcpy(dst, fo->pcm[c], sizeof(double) * 1536);
        else
            memset(dst, 0, sizeof(double) * 1536);
    }
    if (st)
        fprintf(st, "%lld %d %s\n", fo->index, (int)fo->status,
                fo->reason ? fo->reason : "-");
    if (++k->nf == k->max_pass)
        sink_flush(k);
}

int main(int argc, char **argv)
{
    const char *src = NULL, *dst = NULL, *f64 = NULL, *status = NULL;
    const char *el_arg = "auto";
    int channels = 0, quiet = 0, internal = 0, no_guard = 0, i, c;
    long max_pass = 1800, lookahead = 0;
    double seconds = -1.0, ref_peak = 0.0, t0, dt;
    unsigned char *data;
    size_t len;
    ac4mp4_sample *smp = NULL;
    long n_frames, f0;
    ac4dec_element element = AC4DEC_ELEMENT_AUTO;
    ac4dec_config cfg;
    ac4dec *dec;
    ac4dec_stats stats;
    FILE *fst = NULL;
    sink_t k;
    int n_ch;

    memset(&k, 0, sizeof(k));
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
        else if (!strcmp(a, "--lookahead") && i + 1 < argc)
            lookahead = atol(argv[++i]);
        else if (!strcmp(a, "--f64") && i + 1 < argc)
            f64 = argv[++i];
        else if (!strcmp(a, "--status") && i + 1 < argc)
            status = argv[++i];
        else if (!strcmp(a, "--internal"))
            internal = 1;
        else if (!strcmp(a, "--no-guard"))
            no_guard = 1;
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
    cfg.lookahead = (int)lookahead;
    cfg.no_guard = no_guard;
    dec = ac4dec_open(&cfg);
    if (!dec) {
        fprintf(stderr, "cannot open the decoder (bad --lookahead?)\n");
        return 1;
    }
    n_ch = (element == AC4DEC_ELEMENT_PAIR) ? 2 : (channels ? channels : 6);
    k.n_ch = n_ch;
    k.internal = internal;
    k.quiet = quiet;
    k.max_pass = max_pass;
    k.lv.ref_peak = ref_peak > 0.0 ? ref_peak : ac4dec_ref_peak(element);
    k.lv.burst = 2.0 * k.lv.ref_peak;
    if (k.lv.ref_peak > k.lv.burst)
        k.lv.ref_peak = k.lv.burst;
    k.rs_wav = ac4dec_resampler_open(n_ch);
    if (f64 && !internal)
        k.rs_raw = ac4dec_resampler_open(n_ch);
    k.pass = (double *)malloc(sizeof(double) * (size_t)max_pass * 1536
                              * (size_t)n_ch);
    k.ibuf = (double *)malloc(sizeof(double) * (size_t)max_pass * 1602
                              * (size_t)n_ch);
    for (c = 0; c < n_ch; c++)
        k.rout[c] = (double *)malloc(sizeof(double)
                                     * ((size_t)max_pass * 1602 + 8));
    k.pcm16 = (short *)malloc(sizeof(short) * ((size_t)max_pass * 1602 + 8)
                              * (size_t)n_ch);
    if (!k.rs_wav || (f64 && !internal && !k.rs_raw) || !k.pass || !k.ibuf
        || !k.pcm16) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    for (c = 0; c < n_ch; c++)
        if (!k.rout[c]) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
    k.fw = fopen(dst, "wb");
    if (!k.fw || wav_header(k.fw, n_ch, 0) != 0) {
        fprintf(stderr, "cannot write %s\n", dst);
        return 1;
    }
    if (f64) {
        k.fr = fopen(f64, "wb");
        if (!k.fr) {
            fprintf(stderr, "cannot write %s\n", f64);
            return 1;
        }
    }
    if (status) {
        fst = fopen(status, "w");
        if (!fst) {
            fprintf(stderr, "cannot write %s\n", status);
            return 1;
        }
    }

    for (f0 = 0; f0 < n_frames; f0++) {
        ac4dec_frame fo;
        const ac4mp4_sample *sm = &smp[f0];
        if (ac4dec_decode_frame(dec, sm->len ? data + sm->off : NULL,
                                sm->len, &fo))
            sink_frame(&k, &fo, fst);
    }
    {
        ac4dec_frame fo;
        while (ac4dec_flush(dec, &fo))
            sink_frame(&k, &fo, fst);
    }
    sink_flush(&k);
    fseek(k.fw, 0, SEEK_SET);
    wav_header(k.fw, n_ch, k.n_out * 2ull * (unsigned long long)n_ch);
    fclose(k.fw);
    if (k.fr)
        fclose(k.fr);
    if (fst)
        fclose(fst);
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
            printf("  stream is ASPX_ACPL_3 (2-ch core + A-CPL): %s\n",
                   n_ch == 6 ? "5.1 from the A-CPL upmix"
                                         : "its stereo core");
        printf("%ld frames (%ld bad, %ld concealed, %ld sequence jumps), "
               "element %s, %d ch, %.3f s -> %s  [%.2f s, %.2fx real time; "
               "A-SPX bursts %ld]\n",
               n_frames, stats.n_bad, stats.n_concealed, stats.n_seq_jumps,
               element == AC4DEC_ELEMENT_PAIR ? "pair" : "5_X", n_ch,
               (double)k.n_out / 48000.0, dst, dt,
               dt > 0 ? ((double)k.n_out / 48000.0) / dt : 0.0,
               stats.n_burst);
    }
    ac4dec_close(dec);
    ac4dec_resampler_close(k.rs_wav);
    ac4dec_resampler_close(k.rs_raw);
    free(k.pass);
    free(k.ibuf);
    free(k.pcm16);
    for (c = 0; c < n_ch; c++)
        free(k.rout[c]);
    free(smp);
    free(data);
    return 0;
}
