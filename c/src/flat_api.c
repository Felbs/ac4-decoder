/* SPDX-License-Identifier: Apache-2.0 */
/*
 * The flat entry points (ac4dec.h, "RUN-TIME LOADING"): ac4dec_open with
 * the settings as text and decoding into an interleaved float buffer, for
 * programs that load libac4dec at run time (dlopen) and must not depend on
 * the layout of ac4dec_config or ac4dec_frame, which may grow between
 * versions.
 */
#include "ac4dec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *ac4dec_version(void) { return AC4DEC_VERSION; }

static void set_err(char *err, size_t errlen, const char *what,
                    const char *tok, size_t n)
{
    if (err && errlen)
        snprintf(err, errlen, "%s: %.*s", what, (int)n, tok);
}

static int parse_long(const char *v, long *out)
{
    char *end;
    long x;
    if (!*v)
        return -1;
    x = strtol(v, &end, 10);
    if (*end)
        return -1;
    *out = x;
    return 0;
}

static int parse_double(const char *v, double *out)
{
    char *end;
    double x;
    if (!*v)
        return -1;
    x = strtod(v, &end);
    if (*end)
        return -1;
    *out = x;
    return 0;
}

static int parse_layout(const char *v)
{
    static const char *const names[] = {"native", "2.0", "5.1", "5.1.2",
                                        "5.1.4", "7.1", "7.1.2", "7.1.4"};
    int i;
    for (i = 0; i < (int)(sizeof(names) / sizeof(names[0])); i++)
        if (!strcmp(v, names[i]))
            return i;
    return -1;
}

/* One key=value pair into cfg. -> 0, or -1 for an unknown key or a bad
   value. */
static int apply(ac4dec_config *c, const char *k, const char *v)
{
    long n;
    double x;
    if (!strcmp(k, "lookahead")) {
        if (parse_long(v, &n) || n < 0 || n > 100000)
            return -1;
        c->lookahead = (int)n;
    } else if (!strcmp(k, "layout")) {
        int l = parse_layout(v);
        if (l < 0)
            return -1;
        c->layout = l;
    } else if (!strcmp(k, "channels")) {
        if (parse_long(v, &n) || n < 0 || n > AC4DEC_MAX_CHANNELS)
            return -1;
        c->channels = (int)n;
    } else if (!strcmp(k, "loudness")) {
        if (parse_long(v, &n) || (n != 0 && n != 1))
            return -1;
        c->loudness = (int)n;
    } else if (!strcmp(k, "target_level")) {
        if (parse_double(v, &x) || x > 0 || x < -70)
            return -1;
        c->target_level = x;
    } else if (!strcmp(k, "drc")) {
        if (!strcmp(v, "auto"))
            c->drc = 0;
        else if (!strcmp(v, "off"))
            c->drc = -1;
        else if (!parse_long(v, &n) && n >= 0 && n < 8)
            c->drc = (int)n + 1;
        else
            return -1;
    } else if (!strcmp(k, "de_level")) {
        if (parse_double(v, &x) || x < 0 || x > 30)
            return -1;
        c->de_level = x;
    } else if (!strcmp(k, "presentation")) {
        if (parse_long(v, &n) || n < 0 || n > 512)
            return -1;
        c->presentation = (int)n;
    } else if (!strcmp(k, "language")) {
        if (strlen(v) >= sizeof(c->language))
            return -1;
        strcpy(c->language, v);
    } else if (!strcmp(k, "stereo_dmx")) {
        if (parse_long(v, &n) || n < 0 || n > 3)
            return -1;
        c->stereo_dmx = (int)n;
    } else if (!strcmp(k, "guard")) {
        if (parse_long(v, &n) || (n != 0 && n != 1))
            return -1;
        c->no_guard = !n;
    } else {
        return -1;
    }
    return 0;
}

ac4dec *ac4dec_open_options(const char *options, char *err, size_t errlen)
{
    ac4dec_config c;
    const char *p = options ? options : "";
    ac4dec *d;

    memset(&c, 0, sizeof(c));
    c.target_level = -24.0;
    if (err && errlen)
        err[0] = 0;
    while (*p) {
        const char *tok, *eq;
        size_t n;
        char key[32], val[64];
        while (*p == ' ' || *p == ',' || *p == '\t')
            p++;
        if (!*p)
            break;
        tok = p;
        while (*p && *p != ' ' && *p != ',' && *p != '\t')
            p++;
        n = (size_t)(p - tok);
        eq = memchr(tok, '=', n);
        if (!eq || eq == tok || (size_t)(eq - tok) >= sizeof(key)
            || n - (size_t)(eq - tok) - 1 >= sizeof(val)) {
            set_err(err, errlen, "not key=value", tok, n);
            return NULL;
        }
        memcpy(key, tok, (size_t)(eq - tok));
        key[eq - tok] = 0;
        memcpy(val, eq + 1, n - (size_t)(eq - tok) - 1);
        val[n - (size_t)(eq - tok) - 1] = 0;
        if (apply(&c, key, val)) {
            set_err(err, errlen, "unknown option or bad value", tok, n);
            return NULL;
        }
    }
    d = ac4dec_open(&c);
    if (!d)
        set_err(err, errlen, "the decoder would not open with", options ? options : "", strlen(options ? options : ""));
    return d;
}

static int interleave(const ac4dec_frame *f, float *out, int max_samples,
                      int *channels, int *sample_rate, int *status)
{
    int c, i, n = f->n_samples, ch = f->n_channels;
    if (ch < 1 || ch > AC4DEC_MAX_CHANNELS || n < 0)
        return 0;   /* nothing playable: element not known yet */
    if (n > max_samples)
        n = max_samples;
    for (i = 0; i < n; i++)
        for (c = 0; c < ch; c++)
            out[(size_t)i * ch + c] = f->pcm[c] ? (float)f->pcm[c][i] : 0.f;
    if (channels)
        *channels = ch;
    if (sample_rate)
        *sample_rate = f->sample_rate;
    if (status)
        *status = (int)f->status;
    return n;
}

int ac4dec_decode_interleaved(ac4dec *d, const uint8_t *data, size_t len,
                              float *out, int max_samples, int *channels,
                              int *sample_rate, int *status)
{
    ac4dec_frame f;
    if (!d || !out || max_samples <= 0)
        return -1;
    memset(&f, 0, sizeof(f));
    if (!ac4dec_decode_frame(d, data, len, &f))
        return 0;
    return interleave(&f, out, max_samples, channels, sample_rate, status);
}

int ac4dec_flush_interleaved(ac4dec *d, float *out, int max_samples,
                             int *channels, int *sample_rate, int *status)
{
    ac4dec_frame f;
    if (!d || !out || max_samples <= 0)
        return -1;
    memset(&f, 0, sizeof(f));
    if (!ac4dec_flush(d, &f))
        return 0;
    return interleave(&f, out, max_samples, channels, sample_rate, status);
}
