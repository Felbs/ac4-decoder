/* SPDX-License-Identifier: Apache-2.0 */
/*
 * ac4dec-c -- decode AC-4 audio to WAV.
 * The C port of `ac4dec` (ac4/cli.py): A-SPX rendered, output guard on,
 * as the Python CLI (see include/ac4dec.h).
 *
 *   ac4dec-c IN OUT.wav [--element auto|5_X|pair|mono|3_0|5_0|7_X] [--channels N]
 *            [--presentation N|id:N] [--language xx]
 *            [--seconds S] [--max-pass N] [--target-level L]
 *            [--drc auto|off|N] [--legacy-level] [--ref-peak X] [--no-guard]
 *            [--lookahead N] [--f64 RAW] [--internal] [--status FILE] [-q]
 *   ac4dec-c IN --list
 *
 * IN is detected from its content (include/ac4dec_input.h): MP4
 * (fragmented or not), AC-4 sync frames, or an MPEG-2 transport stream.
 *
 * OUT is 16-bit PCM at the stream's external rate (48 kHz, or 44,1 kHz for
 * fs_index 0) through the resampler of TS 103 190-1 Table 83 for its frame
 * rate, loudness-normalised as the Python CLI: dialnorm brought to
 * --target-level (default -24 dB) and the stream's DRC mode for that
 * level applied (--drc auto, off, or a mode ID), in the decoder
 * (ac4dec_config.loudness); samples beyond full scale are clipped and
 * counted. --legacy-level (or --ref-peak X) writes with the old level rule
 * instead: samples times 0.9 / ref_peak, ref_peak starting at the
 * element's fixed reference (9e5 for 5.1, 2e5 stereo, or X), rising to
 * the loudest sample of a pass, capped at twice the start, applied per
 * pass of --max-pass output frames before resampling. The decoder itself
 * is one stream (no pass boundaries); --lookahead sets its delay (default
 * AC4DEC_DEFAULT_LOOKAHEAD).
 *
 * --status FILE writes one line per frame: index, status (0 OK,
 * 1 concealed, 2 bad) and the reason.
 *
 * --f64 RAW also writes the decoder's samples (before the legacy level
 * rule; normalised unless --legacy-level)
 * as interleaved little-endian float64 at the external rate (or the
 * internal rate with --internal): the comparison output.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ac4dec.h"
#include "ac4dec_input.h"


static void usage(void)
{
    fprintf(stderr,
            "usage: ac4dec-c IN OUT.wav "
            "[--element auto|5_X|pair|mono|3_0|5_0|7_X|ice]\n"
            "                [--channels N] "
            "[--layout native|2.0|5.1|5.1.2|5.1.4|7.1|7.1.2|7.1.4]\n"
            "                [--presentation N|id:N] [--language xx]\n"
            "                [--seconds S] [--max-pass N] [--target-level L]\n"
            "                [--drc auto|off|N] [--de-level dB] [--legacy-level]"
            "\n                [--no-mix] [--assoc-gain dB]"
            "\n                [--ref-peak X] "
            "[--no-guard]\n"
            "                [--lookahead N] [--f64 RAW] [--internal] "
            "[--status FILE] [-q]\n"
            "       ac4dec-c IN --list\n");
}

static const char *const KIND[4] = { "unknown", "mp4", "sync", "ts" };
static const char *const EL_NAME[9] = { "auto", "5_X", "pair", "mono",
                                        "3_0", "5_0", "7_X", "ice",
                                        "ajoc" };
static const char *const CMODE[11] = {
    "Mono", "Stereo", "3.0", "5.0", "5.1", "7.0 (3/4/0)", "7.1 (3/4/0.1)",
    "7.0 (5/2/0)", "7.1 (5/2/0.1)", "7.0 (3/2/2)", "7.1 (3/2/2.1)"
};

static int list_presentations(const ac4in_frame *fr, long n,
                              const ac4in_info *info)
{
    long f;
    printf("input: %s, %ld frames\n", KIND[info->kind], n);
    for (f = 0; f < n && f < 64; f++) {
        ac4dec_presentation p[16];
        int k, m;
        if (!fr[f].len)
            continue;
        m = ac4dec_presentations(fr[f].p, fr[f].len, p, 16);
        if (m < 0)
            continue;
        for (k = 0; k < m && k < 16; k++) {
            char cm[32];
            if (p[k].channel_mode >= 0 && p[k].channel_mode <= 10)
                snprintf(cm, sizeof(cm), "%s", CMODE[p[k].channel_mode]);
            else if (p[k].channel_mode >= 0)
                snprintf(cm, sizeof(cm), "ch_mode %d", p[k].channel_mode);
            else
                snprintf(cm, sizeof(cm), "objects");
            printf("  [%d]%s id %ld, config %d, language %s, content %s, "
                   "%s%s%s\n", p[k].index, p[k].is_default ? " *" : "  ",
                   p[k].presentation_id, p[k].config,
                   p[k].language[0] ? p[k].language : "-",
                   p[k].content[0] ? p[k].content : "-", cm,
                   p[k].enabled ? "" : ", DISABLED",
                   p[k].decodable ? "" : ", not decodable here");
        }
        if (info->has_dsi)
            printf("dac4: bitstream_version %d, fs_index %d, "
                   "frame_rate_index %d, %d presentations\n",
                   info->dsi_bitstream_version, info->dsi_fs_index,
                   info->dsi_frame_rate_index, info->dsi_n_presentations);
        printf("(* = decoded by default)\n");
        return 0;
    }
    printf("no frame with a table of contents found\n");
    return 1;
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

static int wav_header(FILE *f, int ch, unsigned rate,
                      unsigned long long data_bytes)
{
    unsigned char h[44];
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
    long max_pass, nf, ns;
    double *pass, *ibuf, *rout[AC4DEC_MAX_CHANNELS];
    short *pcm16;
    ac4dec_resampler *rs_wav, *rs_raw;
    level_t lv;
    int legacy;
    unsigned long long n_clip;
    FILE *fw, *fr;
    unsigned long long n_out;
} sink_t;

static void sink_flush(sink_t *k)
{
    double *planar[AC4DEC_MAX_CHANNELS];
    long ns = k->ns, s;
    int c, no;
    if (k->nf == 0)
        return;
    for (c = 0; c < k->n_ch; c++)
        planar[c] = k->pass + (size_t)c * (size_t)k->max_pass
                              * AC4DEC_MAX_FRAME;
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
    if (k->legacy)
        level_apply(&k->lv, planar, k->n_ch, ns, k->quiet);
    no = ac4dec_resampler_push(k->rs_wav, (const double *const *)planar,
                               (int)ns, k->rout);
    for (s = 0; s < no; s++) {
        for (c = 0; c < k->n_ch; c++) {
            double v = k->rout[c][s];
            if (fabs(v) > 1.0)
                k->n_clip++;
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
    k->ns = 0;
}

static void sink_frame(sink_t *k, const ac4dec_frame *fo, FILE *st)
{
    int c, n = fo->n_samples;
    long off = k->ns;
    if (n < 0 || n > AC4DEC_MAX_FRAME)
        n = 0;
    for (c = 0; c < k->n_ch; c++) {
        double *dst = k->pass + (size_t)c * (size_t)k->max_pass
                                * AC4DEC_MAX_FRAME + off;
        if (c < fo->n_channels)
            memcpy(dst, fo->pcm[c], sizeof(double) * (size_t)n);
        else
            memset(dst, 0, sizeof(double) * (size_t)n);
    }
    k->ns += n;
    if (st)
        fprintf(st, "%lld %d %s\n", fo->index, (int)fo->status,
                fo->reason ? fo->reason : "-");
    if (++k->nf == k->max_pass)
        sink_flush(k);
}

int main(int argc, char **argv)
{
    const char *src = NULL, *dst = NULL, *f64 = NULL, *status = NULL;
    const char *el_arg = "auto", *pres = NULL, *lang = NULL;
    int channels = 0, quiet = 0, internal = 0, no_guard = 0, i, c, list = 0;
    int layout = 0;
    long max_pass = 1800, lookahead = 0;
    double seconds = -1.0, ref_peak = 0.0, t0, dt, target = -24.0, de_level = 0.0;
    double assoc_gain = 0.0;
    int no_mix = 0;
    int legacy = 0, drc = 0;
    unsigned char *data;
    size_t len;
    ac4in_frame *smp = NULL;
    ac4in_info info;
    long n_frames, f0;
    ac4dec_element element = AC4DEC_ELEMENT_AUTO;
    ac4dec_config cfg;
    ac4dec *dec;
    ac4dec_stats stats;
    FILE *fst = NULL;
    sink_t k;
    int n_ch, r_fs = 1, r_fri = 3, r_fl = 1536, r_up = 1001, r_down = 960;
    int out_fs = 48000;
    double fps = 30000.0 / 1001.0;

    memset(&k, 0, sizeof(k));
    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--element") && i + 1 < argc)
            el_arg = argv[++i];
        else if (!strcmp(a, "--layout") && i + 1 < argc) {
            static const char *const LN[8] = { "native", "2.0", "5.1",
                "5.1.2", "5.1.4", "7.1", "7.1.2", "7.1.4" };
            int q;
            const char *v = argv[++i];
            layout = -1;
            for (q = 0; q < 8; q++)
                if (!strcmp(v, LN[q]))
                    layout = q;
            if (layout < 0) {
                usage();
                return 2;
            }
        }
        else if (!strcmp(a, "--channels") && i + 1 < argc)
            channels = atoi(argv[++i]);
        else if (!strcmp(a, "--seconds") && i + 1 < argc)
            seconds = atof(argv[++i]);
        else if (!strcmp(a, "--max-pass") && i + 1 < argc)
            max_pass = atol(argv[++i]);
        else if (!strcmp(a, "--ref-peak") && i + 1 < argc) {
            ref_peak = atof(argv[++i]);
            legacy = 1;
        } else if (!strcmp(a, "--legacy-level"))
            legacy = 1;
        else if (!strcmp(a, "--target-level") && i + 1 < argc)
            target = atof(argv[++i]);
        else if (!strcmp(a, "--de-level") && i + 1 < argc)
            de_level = atof(argv[++i]);
        else if (!strcmp(a, "--assoc-gain") && i + 1 < argc)
            assoc_gain = atof(argv[++i]);
        else if (!strcmp(a, "--no-mix"))
            no_mix = 1;
        else if (!strcmp(a, "--drc") && i + 1 < argc) {
            const char *v = argv[++i];
            drc = !strcmp(v, "auto") ? 0 : !strcmp(v, "off") ? -1
                  : atoi(v) + 1;
        }
        else if (!strcmp(a, "--lookahead") && i + 1 < argc)
            lookahead = atol(argv[++i]);
        else if (!strcmp(a, "--f64") && i + 1 < argc)
            f64 = argv[++i];
        else if (!strcmp(a, "--status") && i + 1 < argc)
            status = argv[++i];
        else if (!strcmp(a, "--presentation") && i + 1 < argc)
            pres = argv[++i];
        else if (!strcmp(a, "--language") && i + 1 < argc)
            lang = argv[++i];
        else if (!strcmp(a, "--list"))
            list = 1;
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
    if (!src || (!dst && !list)
        || channels < 0 || channels > AC4DEC_MAX_CHANNELS) {
        usage();
        return 2;
    }
    if (!strcmp(el_arg, "5_X"))
        element = AC4DEC_ELEMENT_5X;
    else if (!strcmp(el_arg, "pair"))
        element = AC4DEC_ELEMENT_PAIR;
    else if (!strcmp(el_arg, "mono"))
        element = AC4DEC_ELEMENT_MONO;
    else if (!strcmp(el_arg, "3_0"))
        element = AC4DEC_ELEMENT_3_0;
    else if (!strcmp(el_arg, "5_0"))
        element = AC4DEC_ELEMENT_5_0;
    else if (!strcmp(el_arg, "7_X"))
        element = AC4DEC_ELEMENT_7X;
    else if (!strcmp(el_arg, "ice"))
        element = AC4DEC_ELEMENT_ICE;
    else if (strcmp(el_arg, "auto")) {
        usage();
        return 2;
    }

    data = read_file(src, &len);
    if (!data) {
        fprintf(stderr, "cannot read %s\n", src);
        return 1;
    }
    n_frames = ac4in_read(data, len, &smp, &info);
    if (n_frames < 0) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    if (list)
        return list_presentations(smp, n_frames, &info);
    if (!quiet && (info.n_crc_bad || info.n_skipped))
        printf("  %s: %ld frames failed their CRC, %ld bytes skipped to "
               "regain sync\n", KIND[info.kind], info.n_crc_bad,
               info.n_skipped);
    memset(&cfg, 0, sizeof(cfg));
    if (pres) {
        if (!strncmp(pres, "id:", 3))
            cfg.presentation_id = atol(pres + 3) + 1;
        else
            cfg.presentation = atoi(pres) + 1;
    }
    if (lang) {
        strncpy(cfg.language, lang, sizeof(cfg.language) - 1);
        cfg.language[sizeof(cfg.language) - 1] = 0;
    }
    {
        /* the stream's rate, as the Python probe_rate: the first frame
           whose TOC parses and whose next frame agrees (Table 83 / 84;
           29.97 fps at 48 kHz when none or not decodable) */
        int pf = -1, pr = -1, have = 0;
        for (f0 = 0; f0 < n_frames && f0 < 64; f0++) {
            int fs, fri;
            if (!smp[f0].len
                || ac4dec_probe_rate(smp[f0].p, smp[f0].len, &fs, &fri)) {
                pf = pr = -1;
                continue;
            }
            if (fs == pf && fri == pr) {
                have = 1;
                break;
            }
            pf = fs;
            pr = fri;
        }
        if ((have || pf >= 0)
            && ac4dec_rate_info(pf, pr, &r_fl, &r_up, &r_down, &out_fs,
                                &fps) == 0) {
            r_fs = pf;
            r_fri = pr;
        }
    }
    if (seconds >= 0.0) {
        long lim = (long)floor(seconds * fps + 0.5);
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
            if (!smp[f0].len)
                continue;
            element = ac4dec_probe_config(&cfg, smp[f0].p, smp[f0].len);
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
    cfg.element = element;
    if (element == AC4DEC_ELEMENT_7X || element == AC4DEC_ELEMENT_ICE)
        for (long f = 0; f < n_frames && f < 64 && !cfg.ch_mode; f++)
            if (smp[f].len)
                cfg.ch_mode = ac4dec_probe_ch_mode(&cfg, smp[f].p,
                                                   smp[f].len);
    if (element == AC4DEC_ELEMENT_ICE)
        for (long f = 0; f < n_frames && f < 64 && !cfg.present; f++)
            if (smp[f].len)
                cfg.present = ac4dec_probe_present(&cfg, smp[f].p,
                                                   smp[f].len);
    cfg.channels = channels;
    cfg.layout = layout;
    cfg.internal_rate = 1;          /* the level rule runs before resampling */
    cfg.lookahead = (int)lookahead;
    cfg.no_guard = no_guard;
    cfg.loudness = !legacy;
    cfg.target_level = target;
    cfg.drc = drc;
    cfg.de_level = de_level;
    cfg.no_mix = no_mix;
    cfg.assoc_gain = assoc_gain;
    dec = ac4dec_open(&cfg);
    if (!dec) {
        fprintf(stderr, "cannot open the decoder (bad --lookahead?)\n");
        return 1;
    }
    n_ch = ac4dec_channels(dec);
    k.n_ch = n_ch;
    k.internal = internal;
    k.quiet = quiet;
    k.max_pass = max_pass;
    k.legacy = legacy;
    k.lv.ref_peak = ref_peak > 0.0 ? ref_peak : ac4dec_ref_peak(element);
    k.lv.burst = 2.0 * k.lv.ref_peak;
    if (k.lv.ref_peak > k.lv.burst)
        k.lv.ref_peak = k.lv.burst;
    (void)r_fs;
    (void)r_fri;
    (void)r_fl;
    k.rs_wav = ac4dec_resampler_open_ratio(n_ch, r_up, r_down);
    if (f64 && !internal)
        k.rs_raw = ac4dec_resampler_open_ratio(n_ch, r_up, r_down);
    k.pass = (double *)malloc(sizeof(double) * (size_t)max_pass
                              * AC4DEC_MAX_FRAME * (size_t)n_ch);
    k.ibuf = (double *)malloc(sizeof(double) * (size_t)max_pass
                              * AC4DEC_MAX_OUT * (size_t)n_ch);
    for (c = 0; c < n_ch; c++)
        k.rout[c] = (double *)malloc(sizeof(double)
                                     * ((size_t)max_pass * AC4DEC_MAX_OUT
                                        + 8));
    k.pcm16 = (short *)malloc(sizeof(short)
                              * ((size_t)max_pass * AC4DEC_MAX_OUT + 8)
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
    if (!k.fw || wav_header(k.fw, n_ch, (unsigned)out_fs, 0) != 0) {
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
        const ac4in_frame *sm = &smp[f0];
        if (ac4dec_decode_frame(dec, sm->len ? sm->p : NULL, sm->len, &fo))
            sink_frame(&k, &fo, fst);
    }
    {
        ac4dec_frame fo;
        while (ac4dec_flush(dec, &fo))
            sink_frame(&k, &fo, fst);
    }
    sink_flush(&k);
    fseek(k.fw, 0, SEEK_SET);
    wav_header(k.fw, n_ch, (unsigned)out_fs,
               k.n_out * 2ull * (unsigned long long)n_ch);
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
        for (i = 0; (r = ac4dec_reason(dec, i, &cnt)) != NULL; i++)
            if (!strcmp(r, "toc.unsupported_rate"))
                printf("  frames with frame_rate_factor or "
                       "frame_rate_fraction above 1, 96/192 kHz substreams "
                       "or reserved rates are not decoded; they are "
                       "silent\n");
        if (stats.acpl3)
            printf("  stream is ASPX_ACPL_3 (2-ch core + A-CPL): %s\n",
                   n_ch == 6 ? "5.1 from the A-CPL upmix"
                                         : "its stereo core");
        if (!legacy)
            printf("  loudness: dialnorm -> %g dB, DRC %s%s%llu samples "
                   "clipped\n", target,
                   drc == 0 ? "auto" : drc < 0 ? "off" : "mode",
                   ", ", k.n_clip);
        printf("%ld frames (%ld bad, %ld concealed, %ld sequence jumps), "
               "element %s, %d ch, %.3f s -> %s  [%.2f s, %.2fx real time; "
               "A-SPX bursts %ld]\n",
               n_frames, stats.n_bad, stats.n_concealed, stats.n_seq_jumps,
               EL_NAME[element], n_ch,
               (double)k.n_out / out_fs, dst, dt,
               dt > 0 ? ((double)k.n_out / out_fs) / dt : 0.0,
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
    ac4in_free(&info);
    free(data);
    return 0;
}
