/* SPDX-License-Identifier: Apache-2.0 */
/* Frames in, PCM out: the public API of libac4dec.
 *
 * Per frame (port of ac4/stream.py Ac4Stream.decode_frames and
 * ac4/decoder.py Decoder, core path):
 *   frame size check -> TOC -> TOC checks -> audio substream -> element
 *   parse -> codec mode check -> audio_size checks -> dequantise (level
 *   check) -> source-change rule -> stereo processing -> ungroup ->
 *   IMDCT / overlap-add per channel -> 1001/960 resampler.
 * Any rule the frame breaks conceals it: every channel gets one zero
 * 1536-line window, so the previous frame's overlap fades out through its
 * window and the next good frame fades in. The output length never
 * changes.
 *
 * Concealment rules ported from ac4/robust.py: mp4.size, toc.parse,
 * toc.seq_range, toc.sizes, toc.version/rate/shape and element.mode (with
 * the re-learn rule), element.parse, audio_size.overrun/closure (upper
 * bound; the lower bound only where no A-SPX/A-CPL block follows the
 * core), asf.sect_cb, asf.sect_len, asf.max_sfb, asf.quant, asf.sf_range,
 * asf.level. Not ported in milestone 1 (they need lookahead or A-SPX):
 * seq.outlier, aspx.*, acpl.parse, audio_size.late, the A-SPX level check
 * and the output guard. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ac4dec.h"
#include "ac4_internal.h"

void ac4_spectra_init(void);

#define MIN_FRAME_BYTES 4
#define MAX_FRAME_BYTES 65536
#define RELEARN 3

enum { CH_L, CH_R, CH_C, CH_LFE, CH_LS, CH_RS };

static const char *REASONS[AC4E_COUNT] = {
    "none", "toc.parse", "toc.seq_range", "toc.sizes", "toc.version",
    "toc.rate", "toc.shape", "mp4.size", "audio_size.overrun",
    "audio_size.closure", "element.mode", "element.parse", "asf.sect_cb",
    "asf.sect_len", "asf.max_sfb", "asf.quant", "asf.sf_range", "asf.level",
    "decoder.internal", "element.unknown"
};

const char *ac4_reason_name(int code)
{
    if (code <= 0 || code >= AC4E_COUNT)
        return "decoder.internal";
    return REASONS[code];
}

typedef struct {
    int bv, fs, fri;
    long n_pres;
    int n_cmodes;
    long cmodes[AC4_MAX_CMODES];
    long n_sub;
} toc_shape;

typedef struct {
    int valid;
    int reason;
    toc_shape shape;
    int mode;
    int n;
} relearn_t;

struct ac4dec {
    ac4dec_config cfg;
    int element;                /* resolved ac4dec_element, 0 = not yet */
    int n_ch;                   /* output channels */
    int acpl3;
    ac4_huff huff[AC4T_N_BOOKS];
    ac4_books books;
    ac4_fb_tables *fbt;
    ac4_rs_filter *rsf;
    ac4_fb_state fb[6];
    ac4_rs_state rs[6];
    ac4_element el;
    double spec[6][1536];
    int spec_n[6];
    double win[6][1536];
    int wlen[6][AC4_MAX_WIN];
    int nwin[6];
    double pcm_int[6][1536];
    double pcm_out[6][AC4DEC_MAX_OUT + 2];
    double tmp[4 * 1536];
    /* robustness state */
    int have_shape;
    toc_shape shape;
    int mode;                   /* -1 = none yet */
    int seq_prev;               /* -1 = none */
    relearn_t relearn;
    /* stats */
    long n_frames, n_ok, n_concealed, n_bad, n_seq_jumps;
    long reasons[AC4E_COUNT];
};

/* ------------------------------------------------------------ helpers */
static int seq_continues(int cur, int prev)
{
    if (prev < 0)
        return 1;
    if (cur == prev + 1)
        return 1;
    if (cur == 1 && prev == 1020)
        return 1;
    if (cur != 0 && prev == 0)
        return 1;
    return 0;
}

static int seq_next(int c)
{
    if (c < 0)
        return -1;
    return c == 1020 ? 1 : c + 1;
}

static void shape_of(const ac4_toc *t, toc_shape *s)
{
    int i, n = t->n_cmodes < AC4_MAX_CMODES ? t->n_cmodes : AC4_MAX_CMODES;
    memset(s, 0, sizeof(*s));
    s->bv = t->bitstream_version;
    s->fs = t->fs_index;
    s->fri = t->frame_rate_index;
    s->n_pres = t->n_presentations;
    s->n_cmodes = t->n_cmodes;
    for (i = 0; i < n; i++)
        s->cmodes[i] = t->cmodes[i];
    s->n_sub = t->n_substreams;
}

static int shape_eq_tail(const toc_shape *a, const toc_shape *b)
{
    int i, n;
    if (a->n_pres != b->n_pres || a->n_cmodes != b->n_cmodes
        || a->n_sub != b->n_sub)
        return 0;
    n = a->n_cmodes < AC4_MAX_CMODES ? a->n_cmodes : AC4_MAX_CMODES;
    for (i = 0; i < n; i++)
        if (a->cmodes[i] != b->cmodes[i])
            return 0;
    return 1;
}

static int shape_eq(const toc_shape *a, const toc_shape *b)
{
    return a->bv == b->bv && a->fs == b->fs && a->fri == b->fri
        && shape_eq_tail(a, b);
}

static int relearn_accept(relearn_t *r, int reason, const toc_shape *s,
                          int mode, int at_iframe)
{
    if (r->valid && r->reason == reason && r->mode == mode
        && shape_eq(&r->shape, s)) {
        r->n++;
    } else {
        r->valid = 1;
        r->reason = reason;
        r->shape = *s;
        r->mode = mode;
        r->n = 1;
    }
    if (at_iframe || r->n >= RELEARN) {
        r->valid = 0;
        r->n = 0;
        return 1;
    }
    return 0;
}

static ac4dec_element element_of_toc(const ac4_toc *t)
{
    long cm;
    if (t->n_cmodes < 1)
        return AC4DEC_ELEMENT_AUTO;
    cm = t->n_cmodes <= AC4_MAX_CMODES ? t->cmodes[t->n_cmodes - 1]
                                       : t->cmodes[AC4_MAX_CMODES - 1];
    if (cm == 1)
        return AC4DEC_ELEMENT_PAIR;
    if (cm == 4)
        return AC4DEC_ELEMENT_5X;
    return AC4DEC_ELEMENT_AUTO;
}

/* parse a TOC without side effects -> 0 ok */
static int toc_try(const uint8_t *data, size_t len, ac4_toc *t)
{
    jmp_buf jb;
    ac4_bits b;
    if (!data || len == 0)
        return -1;
    b.d = data;
    b.nbits = len * 8;
    b.p = 0;
    b.jb = &jb;
    b.eof_code = AC4E_TOC_PARSE;
    if (setjmp(jb) != 0)
        return -1;
    ac4_toc_parse(&b, t);
    return 0;
}

ac4dec_element ac4dec_probe(const uint8_t *data, size_t len)
{
    ac4_toc t;
    if (len > MAX_FRAME_BYTES || toc_try(data, len, &t) != 0)
        return AC4DEC_ELEMENT_AUTO;
    return element_of_toc(&t);
}

double ac4dec_ref_peak(ac4dec_element e)
{
    return e == AC4DEC_ELEMENT_PAIR ? 2.0e5 : 9.0e5;
}

static void set_element(ac4dec *d, int element)
{
    d->element = element;
    if (element == AC4DEC_ELEMENT_PAIR)
        d->n_ch = 2;
    else
        d->n_ch = d->cfg.channels ? d->cfg.channels : 6;
}

/* ------------------------------------------------------------ open/close */
ac4dec *ac4dec_open(const ac4dec_config *cfg)
{
    ac4dec *d;
    int i;
    ac4dec_config c;
    memset(&c, 0, sizeof(c));
    if (cfg)
        c = *cfg;
    if (c.channels != 0 && c.channels != 2 && c.channels != 6)
        return NULL;
    if (c.element != AC4DEC_ELEMENT_AUTO && c.element != AC4DEC_ELEMENT_5X
        && c.element != AC4DEC_ELEMENT_PAIR)
        return NULL;
    d = (ac4dec *)calloc(1, sizeof(*d));
    if (!d)
        return NULL;
    d->cfg = c;
    ac4_spectra_init();
    for (i = 0; i < AC4T_N_BOOKS; i++) {
        if (ac4_huff_build(&d->huff[i], &AC4T_BOOKS[i]) != 0) {
            ac4dec_close(d);
            return NULL;
        }
    }
    for (i = 1; i <= 11; i++)
        d->books.cb[i] = &d->huff[i - 1];
    d->books.sf = &d->huff[AC4T_BOOK_SF];
    d->books.snf = &d->huff[AC4T_BOOK_SNF];
    d->fbt = ac4_fb_tables_new();
    d->rsf = ac4_rs_filter_new();
    if (!d->fbt || !d->rsf) {
        ac4dec_close(d);
        return NULL;
    }
    for (i = 0; i < 6; i++)
        ac4_rs_init(&d->rs[i]);
    d->mode = -1;
    d->seq_prev = -1;
    if (c.element != AC4DEC_ELEMENT_AUTO)
        set_element(d, c.element);
    else
        d->n_ch = c.channels ? c.channels : 2;
    return d;
}

void ac4dec_close(ac4dec *d)
{
    int i;
    if (!d)
        return;
    for (i = 0; i < AC4T_N_BOOKS; i++)
        ac4_huff_free(&d->huff[i]);
    ac4_fb_tables_free(d->fbt);
    ac4_rs_filter_free(d->rsf);
    free(d);
}

void ac4dec_get_stats(const ac4dec *d, ac4dec_stats *s)
{
    memset(s, 0, sizeof(*s));
    if (!d)
        return;
    s->n_frames = d->n_frames;
    s->n_ok = d->n_ok;
    s->n_concealed = d->n_concealed;
    s->n_bad = d->n_bad;
    s->n_seq_jumps = d->n_seq_jumps;
    s->element = d->element;
    s->acpl3 = d->acpl3;
}

const char *ac4dec_reason(const ac4dec *d, int i, long *count)
{
    int c, k = 0;
    if (!d)
        return NULL;
    for (c = 1; c < AC4E_COUNT; c++) {
        if (!d->reasons[c])
            continue;
        if (k++ == i) {
            if (count)
                *count = d->reasons[c];
            return REASONS[c];
        }
    }
    return NULL;
}

/* ------------------------------------------------------------ decoding */
/* which of the six channels this frame synthesises */
static int chan_mask(const ac4dec *d, int acpl3)
{
    if (d->element == AC4DEC_ELEMENT_PAIR || acpl3 || d->n_ch == 2)
        return (1 << CH_L) | (1 << CH_R);
    return 0x3f;
}

static const ac4_chan *chan_src(const ac4_element *e, int k)
{
    switch (k) {
    case CH_L: return &e->lr.ch[0];
    case CH_R: return &e->lr.ch[1];
    case CH_C: return &e->c.ch;
    case CH_LFE: return &e->lfe.ch;
    case CH_LS: return &e->sr.ch[0];
    default: return &e->sr.ch[1];
    }
}

/* Parse and check one frame; fills d->spec for the channels in *mask.
   Returns AC4E_NONE or the reason code. */
static int parse_frame(ac4dec *d, const uint8_t *data, size_t len,
                       int *mask, ac4dec_frame *out)
{
    jmp_buf jb;
    ac4_bits b;
    ac4_toc t;
    volatile int changed = AC4E_NONE;
    int code, k, ifr;
    size_t o, sub_len;
    toc_shape sh;
    ac4_element *e = &d->el;

    if (len < MIN_FRAME_BYTES || len > MAX_FRAME_BYTES || !data)
        return AC4E_MP4_SIZE;
    if (toc_try(data, len, &t) != 0)
        return AC4E_TOC_PARSE;
    out->sequence_counter = t.sequence_counter;
    out->iframe = t.b_iframe_global;
    /* check_toc */
    if (t.sequence_counter > 1020)
        return AC4E_TOC_SEQ_RANGE;
    if (t.n_sizes < 2 || (long)t.n_sizes != t.n_substreams)
        return AC4E_TOC_SIZES;
    if ((double)t.toc_bytes + (double)t.payload_base + t.sizes_sum
        > (double)len || t.sizes_min < 1)
        return AC4E_TOC_SIZES;
    shape_of(&t, &sh);
    if (d->have_shape) {
        if (sh.bv != d->shape.bv)
            changed = AC4E_TOC_VERSION;
        else if (sh.fs != d->shape.fs || sh.fri != d->shape.fri)
            changed = AC4E_TOC_RATE;
        else if (!shape_eq_tail(&sh, &d->shape))
            changed = AC4E_TOC_SHAPE;
    }
    out->seq_jump = !seq_continues(t.sequence_counter, d->seq_prev);
    /* the audio substream (substream 1) */
    o = t.toc_bytes + (size_t)t.payload_base + (size_t)t.sizes[0];
    if (o > len)
        o = len;
    sub_len = (size_t)t.sizes[1];
    if (sub_len > len - o)
        sub_len = len - o;
    ifr = t.b_iframe_global;
    b.d = data + o;
    b.nbits = sub_len * 8;
    b.p = 0;
    b.jb = &jb;
    b.eof_code = AC4E_ELEMENT_PARSE;
    code = setjmp(jb);
    if (code != 0)
        return code;
    if (d->element == AC4DEC_ELEMENT_PAIR)
        ac4_parse_pair(&b, &d->books, ifr, e);
    else
        ac4_parse_5x(&b, &d->books, ifr, e);
    if (changed == AC4E_NONE && d->mode >= 0 && e->mode != d->mode)
        changed = AC4E_ELEMENT_MODE;
    /* audio_size closure */
    {
        size_t stop = e->audio_start + 8 * e->audio_size;
        int complete = d->element == AC4DEC_ELEMENT_PAIR && e->mode == 0;
        if (e->audio_size > sub_len || stop > 8 * sub_len)
            return AC4E_AUDIO_OVERRUN;
        if (e->bits_end > stop)
            return AC4E_AUDIO_CLOSURE;
        if (complete && stop - e->bits_end > 7)
            return AC4E_AUDIO_CLOSURE;
    }
    /* dequantise (with the level check) */
    *mask = chan_mask(d, e->acpl3);
    for (k = 0; k < 6; k++)
        if (*mask & (1 << k))
            d->spec_n[k] = ac4_packed_spectrum(chan_src(e, k), d->spec[k],
                                               &jb);
    if (changed != AC4E_NONE) {
        if (!relearn_accept(&d->relearn, changed, &sh, e->mode, ifr))
            return changed;
        d->shape = sh;
        d->have_shape = 1;
    } else {
        d->relearn.valid = 0;
        d->relearn.n = 0;
    }
    /* accepted: commit */
    if (!d->have_shape) {
        d->shape = sh;
        d->have_shape = 1;
    }
    d->mode = e->mode;
    if (e->acpl3)
        d->acpl3 = 1;
    if (out->seq_jump)
        d->n_seq_jumps++;
    d->seq_prev = t.sequence_counter;
    return AC4E_NONE;
}

static void zero_out(ac4dec *d, ac4dec_frame *out, int n)
{
    int k;
    for (k = 0; k < d->n_ch; k++) {
        memset(d->pcm_out[k], 0, sizeof(double) * (size_t)n);
        out->pcm[k] = d->pcm_out[k];
    }
}

int ac4dec_decode_frame(ac4dec *d, const uint8_t *data, size_t len,
                        ac4dec_frame *out)
{
    int code, mask = 0, k, i, nout = 0;
    ac4dec_frame dummy;
    if (!out)
        out = &dummy;
    memset(out, 0, sizeof(*out));
    out->sequence_counter = -1;
    if (!d)
        return AC4DEC_BAD;
    d->n_frames++;
    if (!d->element) {
        ac4dec_element el = (data && len) ? ac4dec_probe(data, len)
                                          : AC4DEC_ELEMENT_AUTO;
        if (el != AC4DEC_ELEMENT_AUTO) {
            set_element(d, el);
        } else {
            /* nothing known yet: silence, timeline kept */
            static const double zeros[1536];
            out->status = AC4DEC_BAD;
            out->reason = REASONS[AC4E_UNKNOWN_ELEMENT];
            out->n_channels = d->n_ch;
            if (d->cfg.internal_rate) {
                out->n_samples = AC4DEC_FRAME;
                zero_out(d, out, AC4DEC_FRAME);
            } else {
                for (k = 0; k < 6; k++)
                    nout = ac4_rs_push(d->rsf, &d->rs[k], zeros, 1536,
                                       d->pcm_out[k]);
                out->n_samples = nout;
                zero_out(d, out, nout);
            }
            d->n_bad++;
            d->reasons[AC4E_UNKNOWN_ELEMENT]++;
            return AC4DEC_BAD;
        }
    }
    code = parse_frame(d, data, len, &mask, out);
    if (code == AC4E_NONE) {
        ac4_element *e = &d->el;
        /* stereo processing on the packed spectra */
        if ((mask & 3) == 3)
            ac4_unmix(d->spec[CH_L], d->spec[CH_R],
                      d->spec_n[CH_L] < d->spec_n[CH_R] ? d->spec_n[CH_L]
                                                        : d->spec_n[CH_R],
                      &e->lr.st, &e->lr.fr[0]);
        if ((mask & 0x30) == 0x30)
            ac4_unmix(d->spec[CH_LS], d->spec[CH_RS],
                      d->spec_n[CH_LS] < d->spec_n[CH_RS] ? d->spec_n[CH_LS]
                                                          : d->spec_n[CH_RS],
                      &e->sr.st, &e->sr.fr[0]);
        for (k = 0; k < 6 && code == AC4E_NONE; k++) {
            const ac4_framing *fr;
            int w, sum = 0;
            if (!(mask & (1 << k)))
                continue;
            fr = chan_src(e, k)->fr;
            ac4_ungroup(d->spec[k], d->spec_n[k], fr, d->win[k], d->wlen[k]);
            d->nwin[k] = fr->num_windows;
            for (w = 0; w < fr->num_windows; w++)
                sum += d->wlen[k][w];
            if (sum != 1536)            /* cannot happen; never trust it */
                code = AC4E_INTERNAL;
        }
        for (k = 0; k < 6 && code == AC4E_NONE; k++) {
            int w, pos = 0;
            if (!(mask & (1 << k)))
                continue;
            for (w = 0; w < d->nwin[k]; w++) {
                ac4_fb_block(d->fbt, &d->fb[k], d->win[k] + pos,
                             d->wlen[k][w], d->pcm_int[k] + pos, d->tmp);
                pos += d->wlen[k][w];
            }
        }
        if (code == AC4E_NONE) {
            out->status = AC4DEC_OK;
            d->n_ok++;
        }
    }
    if (code != AC4E_NONE) {
        /* conceal: one zero 1536-line window per channel */
        memset(d->win[0], 0, sizeof(double) * 1536);
        mask = chan_mask(d, d->acpl3);
        for (k = 0; k < 6; k++)
            if (mask & (1 << k))
                ac4_fb_block(d->fbt, &d->fb[k], d->win[0], 1536,
                             d->pcm_int[k], d->tmp);
        out->status = AC4DEC_CONCEALED;
        out->reason = ac4_reason_name(code);
        out->seq_jump = 0;
        d->n_concealed++;
        d->reasons[code < AC4E_COUNT && code > 0 ? code : AC4E_INTERNAL]++;
        if (d->seq_prev >= 0)
            d->seq_prev = seq_next(d->seq_prev);
    }
    /* channels not synthesised (ASPX_ACPL_3 surrounds, or a 2-channel
       request) are silent */
    for (k = 0; k < 6; k++) {
        if (!(mask & (1 << k)))
            memset(d->pcm_int[k], 0, sizeof(double) * 1536);
        for (i = 0; i < 1536; i++)
            if (!isfinite(d->pcm_int[k][i]))
                d->pcm_int[k][i] = 0.0;
    }
    out->n_channels = d->n_ch;
    if (d->cfg.internal_rate) {
        for (k = 0; k < d->n_ch; k++)
            out->pcm[k] = d->pcm_int[k];
        out->n_samples = AC4DEC_FRAME;
    } else {
        for (k = 0; k < d->n_ch; k++)
            nout = ac4_rs_push(d->rsf, &d->rs[k], d->pcm_int[k], 1536,
                               d->pcm_out[k]);
        for (k = 0; k < d->n_ch; k++)
            out->pcm[k] = d->pcm_out[k];
        out->n_samples = nout;
    }
    return out->status;
}
