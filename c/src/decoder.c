/* SPDX-License-Identifier: Apache-2.0 */
/* Frames in, PCM out: the public API of libac4dec.
 *
 * A port of ac4/stream.py (Ac4Stream.decode_frames, _late_closure),
 * ac4/decoder.py (Decoder.decode_pass, core_for_bursts) and ac4/robust.py
 * (every rule, the output guard), as a delay line. The Python decodes a
 * pass of frames at once; this decodes one frame per call and gives the
 * output of the frame fed `lookahead` calls earlier, which is what the
 * rules that look ahead need:
 *
 *   parse     frame j when frame j + 1 has arrived (seq.outlier and the
 *             TOC-change rule look at the next frame's TOC)
 *   hold      L parsed frames before synthesis, so that audio_size.late
 *             can still conceal frames from before the first aspx_config
 *             when the I-frame that brings it arrives
 *   synthesis IMDCT / overlap-add of frame j
 *   A-SPX     frame j once core frame j + 1 exists (its envelopes reach
 *             into the next frame); the L/R and Ls/Rs pairs go through
 *             the QMF bank (577 samples later than C and LFE, as in the
 *             Python), with the burst guard and the aspx.level check
 *   guard     the output guard judges frame f with G frames after it
 *             (sustained-rise exception: 6; the seed: the first 30)
 *
 * lookahead N = L + G + 2, G = min(N - 2, 30). See include/ac4dec.h.
 *
 * Per frame: frame size -> TOC -> TOC checks -> seq.outlier -> element
 * parse -> codec mode -> aspx_config check -> A-SPX blocks -> A-CPL block
 * -> audio_size closure (both bounds once every block can be sized) ->
 * dequantise (asf.level) -> TOC / codec-mode change rule -> commit ->
 * stereo processing -> ungroup. Any rule the frame breaks conceals it:
 * every channel gets one zero 1536-line window (the previous frame's
 * overlap fades out), its A-SPX high band is muted and the A-SPX framing
 * state is held. The output length never changes. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ac4dec.h"
#include "ac4_internal.h"

void ac4_spectra_init(void);

#define MIN_FRAME_BYTES 4
#define MAX_FRAME_BYTES 65536
#define RELEARN 3
#define LATE_MAJORITY 0.9
#define MAX_PENDING 4096
#define FRAME 1536
#define HF_DELAY 577         /* QMF analysis + synthesis delay, samples */

/* robust.py: A-SPX level check, output guard */
#define MAX_HF_RATIO 16.0
#define HF_MIN_FRAC 0.03
#define HF_TAIL 2
#define G_RATIO 3.0
#define G_RELEASE 0.9999
#define G_INIT_RELEASE 0.9995
#define G_UP 2.0
#define G_FLOOR_FRAC 0.05
#define G_INIT_FRAC 0.75
#define G_WARM 30
#define G_KTH 5
#define G_L 6
#define G_S 4
#define G_SPREAD 4.0
#define G_NEAR 3
#define G_NEAR_RATIO 2.0
#define G_RAMP 256

enum { CH_L, CH_R, CH_C, CH_LFE, CH_LS, CH_RS };

static const char *REASONS[AC4E_COUNT] = {
    "none", "toc.parse", "toc.seq_range", "toc.sizes", "toc.version",
    "toc.rate", "toc.shape", "mp4.size", "audio_size.overrun",
    "audio_size.closure", "element.mode", "element.parse", "asf.sect_cb",
    "asf.sect_len", "asf.max_sfb", "asf.quant", "asf.sf_range", "asf.level",
    "decoder.internal", "element.unknown", "seq.outlier", "aspx.config",
    "aspx.parse", "aspx.num_env", "aspx.borders", "acpl.parse",
    "audio_size.late", "guard.level", "aspx.level"
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

/* a frame between parse and synthesis */
typedef struct {
    long long index;
    int code;                   /* AC4E_NONE or the reason */
    int unknown;                /* element not known: BAD, silence */
    int seq, iframe, seq_jump;
    int mask;                   /* channels synthesised */
    int nwin[6];
    int wlen[6][AC4_MAX_WIN];
    double *win;                /* 6 x 1536 */
    ac4_aspx_frame af[2];       /* L/R, Ls/Rs */
} rec_t;

/* a frame from before the first aspx_config (audio_size.late) */
typedef struct {
    long long index;
    uint8_t *sub;
    size_t sub_len;
    size_t bits, start, size;
    int ifr, acpl3, five;
} pending_t;

/* an output frame waiting for the guard */
typedef struct {
    long long index;
    int code, unknown, seq, iframe, seq_jump;
    int forced;                 /* concealed upstream */
    int aspx_flag;              /* aspx.level */
    double pk;                  /* peak over all output channels */
    int ok;
    double pcm[6][FRAME];
} outf_t;

typedef struct {
    double floor, init, ref;
    int seeded;
    double last[6];
    int has_last, muted;
    int forced_hist[G_NEAR];    /* forced flags of the last frames judged */
    int n_hist;
    long n_fired, n_sustained;
} guard_t;

struct ac4dec {
    ac4dec_config cfg;
    int N, L, G;
    int element;                /* resolved ac4dec_element, 0 = not yet */
    int n_ch;
    ac4_huff huff[AC4T_N_BOOKS];
    ac4_books books;
    ac4_aspx_books abk;
    ac4_fb_tables *fbt;
    ac4_rs_filter *rsf;
    ac4_qmf_tables *qmf;
    ac4_element el;
    double spec[6][FRAME];
    int spec_n[6];
    double tmp[4 * FRAME];
    double zeros[FRAME];

    /* ---- stream state (Ac4Stream) */
    int acpl3;
    int have_shape;
    toc_shape shape;
    int mode;                   /* -1 = none yet */
    int seq_prev;               /* -1 = none */
    relearn_t relearn;
    int has_cfg;
    ac4_aspx_cfg acfg;
    ac4_aspx_tab tab;
    int has_pcfg;
    ac4_acpl_cfg pcfg;
    ac4_aspx_state fstate;
    ac4_acpl_qstate aq;         /* A-CPL q_prev (Pseudocode 121) */
    pending_t *pend;
    int n_pend;
    int late_done;

    /* ---- input */
    uint8_t *raw;               /* the frame waiting for its successor */
    size_t raw_len, raw_cap;
    int raw_null, has_raw;
    long long n_in, n_parsed;

    /* ---- held records (ring of L + 1) */
    rec_t *recs;
    int rec_cap;
    long long rec_first;        /* index of the oldest held record */
    int rec_n;

    /* ---- synthesis + A-SPX */
    ac4_fb_state fb[6];
    long long n_synth;
    double core[6][FRAME];
    double core_prev[6][FRAME];
    double core_tail[6][HF_DELAY];   /* delay line: last 577 core samples */
    rec_t prev_info;            /* frame n_synth - 1 (no windows) */
    ac4_aspx_pair pair[2];
    int pair_active[2];
    long long pair_start[2];    /* first frame the pair took */
    int path_on[2];
    int finished;
    double hf[2][2][FRAME];     /* the pairs' A-SPX output frames */
    double dcore[2][2][FRAME];  /* and their cores, 577 samples late */

    /* ---- guard (ring of G + 2) */
    outf_t *ring;
    int ring_cap, ring_n;
    long long ring_first;
    guard_t guard;

    /* ---- output */
    ac4_rs_state rs[6];
    double pcm_out[6][AC4DEC_MAX_OUT + 2];
    outf_t cur;                 /* the frame last given out */
    int flushing;

    /* ---- stats */
    long n_frames, n_out, n_ok, n_concealed, n_bad, n_seq_jumps;
    long n_guard, n_aspx_level, n_burst, n_late;
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

static int seq_prev_of(int c)
{
    if (c < 0)
        return -1;
    return c <= 1 ? 1020 : c - 1;
}

/* robust.peek_counter: -1 when not peekable */
static int peek_counter(const uint8_t *f, size_t n)
{
    unsigned v;
    if (!f || n < 2)
        return -1;
    v = ((unsigned)f[0] << 8) | f[1];
    if ((v >> 14) == 3)
        return -1;
    return (int)((v >> 4) & 0x3FF);
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
    double ref;
    d->element = element;
    if (element == AC4DEC_ELEMENT_PAIR)
        d->n_ch = 2;
    else
        d->n_ch = d->cfg.channels ? d->cfg.channels : 6;
    ref = ac4dec_ref_peak((ac4dec_element)element);
    d->guard.floor = G_FLOOR_FRAC * ref;
    d->guard.init = G_INIT_FRAC * ref;
}

/* which of the six channels a frame synthesises (ASPX_ACPL_3 in 5.1:
   the core pair and the LFE; C, Ls, Rs come from the A-CPL upmix) */
static int chan_mask(const ac4dec *d, int acpl3)
{
    if (d->element == AC4DEC_ELEMENT_PAIR || d->n_ch == 2)
        return (1 << CH_L) | (1 << CH_R);
    if (acpl3)
        return (1 << CH_L) | (1 << CH_R) | (1 << CH_LFE);
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

/* ------------------------------------------------------------ state */
static void free_pending(ac4dec *d)
{
    int i;
    for (i = 0; i < d->n_pend; i++)
        free(d->pend[i].sub);
    d->n_pend = 0;
}

/* everything that belongs to one stream (not the config or tables) */
static void stream_reset(ac4dec *d)
{
    int i;
    free_pending(d);
    d->acpl3 = 0;
    d->have_shape = 0;
    d->mode = -1;
    d->seq_prev = -1;
    memset(&d->relearn, 0, sizeof(d->relearn));
    d->has_cfg = d->has_pcfg = 0;
    memset(&d->fstate, 0, sizeof(d->fstate));
    memset(&d->aq, 0, sizeof(d->aq));
    d->late_done = 0;
    d->has_raw = 0;
    d->n_in = d->n_parsed = 0;
    d->rec_first = 0;
    d->rec_n = 0;
    memset(d->fb, 0, sizeof(d->fb));
    d->n_synth = 0;
    memset(d->core_prev, 0, sizeof(d->core_prev));
    memset(d->core_tail, 0, sizeof(d->core_tail));
    for (i = 0; i < 2; i++) {
        ac4_aspx_pair_reset(&d->pair[i]);
        d->pair_active[i] = 0;
        d->path_on[i] = 0;
        d->pair_start[i] = 0;
    }
    d->finished = 0;
    d->ring_n = 0;
    d->ring_first = 0;
    {
        double fl = d->guard.floor, in = d->guard.init;
        memset(&d->guard, 0, sizeof(d->guard));
        d->guard.floor = fl;
        d->guard.init = in;
    }
    if (d->element)
        set_element(d, d->element);
    for (i = 0; i < 6; i++)
        ac4_rs_init(&d->rs[i]);
    d->flushing = 0;
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
    if (c.lookahead == 0)
        c.lookahead = AC4DEC_DEFAULT_LOOKAHEAD;
    if (c.lookahead < AC4DEC_MIN_LOOKAHEAD || c.lookahead > 100000)
        return NULL;
    d = (ac4dec *)calloc(1, sizeof(*d));
    if (!d)
        return NULL;
    d->cfg = c;
    d->N = c.lookahead;
    d->G = d->N - 2 < G_WARM ? d->N - 2 : G_WARM;
    d->L = d->N - 2 - d->G;
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
    if (ac4_aspx_books_init(&d->abk) != 0) {
        ac4dec_close(d);
        return NULL;
    }
    d->fbt = ac4_fb_tables_new();
    d->rsf = ac4_rs_filter_new();
    d->qmf = ac4_qmf_tables_new();
    d->rec_cap = d->L + 2;
    d->recs = (rec_t *)calloc((size_t)d->rec_cap, sizeof(rec_t));
    d->ring_cap = d->G + 3;
    d->ring = (outf_t *)calloc((size_t)d->ring_cap, sizeof(outf_t));
    d->pend = (pending_t *)calloc(MAX_PENDING, sizeof(pending_t));
    if (!d->fbt || !d->rsf || !d->qmf || !d->recs || !d->ring || !d->pend) {
        ac4dec_close(d);
        return NULL;
    }
    for (i = 0; i < d->rec_cap; i++) {
        d->recs[i].win = (double *)calloc(6 * FRAME, sizeof(double));
        if (!d->recs[i].win) {
            ac4dec_close(d);
            return NULL;
        }
    }
    if (c.element != AC4DEC_ELEMENT_AUTO)
        set_element(d, c.element);
    else
        d->n_ch = c.channels ? c.channels : 2;
    stream_reset(d);
    return d;
}

void ac4dec_close(ac4dec *d)
{
    int i;
    if (!d)
        return;
    for (i = 0; i < AC4T_N_BOOKS; i++)
        ac4_huff_free(&d->huff[i]);
    ac4_aspx_books_free(&d->abk);
    ac4_fb_tables_free(d->fbt);
    ac4_rs_filter_free(d->rsf);
    ac4_qmf_tables_free(d->qmf);
    if (d->recs)
        for (i = 0; i < d->rec_cap; i++)
            free(d->recs[i].win);
    free(d->recs);
    free(d->ring);
    if (d->pend)
        free_pending(d);
    free(d->pend);
    free(d->raw);
    free(d);
}

int ac4dec_delay(const ac4dec *d)
{
    return d ? d->N : 0;
}

void ac4dec_get_stats(const ac4dec *d, ac4dec_stats *s)
{
    memset(s, 0, sizeof(*s));
    if (!d)
        return;
    s->n_frames = d->n_frames;
    s->n_out = d->n_out;
    s->n_ok = d->n_ok;
    s->n_concealed = d->n_concealed;
    s->n_bad = d->n_bad;
    s->n_seq_jumps = d->n_seq_jumps;
    s->n_guard = d->n_guard;
    s->n_aspx_level = d->n_aspx_level;
    s->n_burst = d->n_burst;
    s->n_late = d->n_late;
    s->element = d->element;
    s->acpl3 = d->acpl3;
    s->aspx = (d->path_on[0] || d->path_on[1]) && d->has_cfg;
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

/* ------------------------------------------------------------ records */
static rec_t *rec_at(ac4dec *d, long long index)
{
    long long k = index - d->rec_first;
    if (k < 0 || k >= d->rec_n)
        return NULL;
    return &d->recs[(int)(index % d->rec_cap)];
}

static rec_t *rec_new(ac4dec *d, long long index)
{
    rec_t *r = &d->recs[(int)(index % d->rec_cap)];
    double *w = r->win;
    memset(r, 0, sizeof(*r));
    r->win = w;
    r->index = index;
    r->seq = -1;
    if (d->rec_n == 0)
        d->rec_first = index;
    d->rec_n++;
    return r;
}

/* a concealed frame: one zero window per channel, A-SPX muted */
static void rec_conceal(ac4dec *d, rec_t *r, int code)
{
    int k;
    r->code = code;
    r->mask = chan_mask(d, d->acpl3);
    for (k = 0; k < 6; k++) {
        r->nwin[k] = 1;
        r->wlen[k][0] = FRAME;
        memset(r->win + k * FRAME, 0, sizeof(double) * FRAME);
    }
    r->af[0].kind = r->af[1].kind = AC4_GRP_CONCEALED;
    r->af[0].has_tab = r->af[1].has_tab = 0;
    r->af[0].acpl.kind = r->af[1].acpl.kind = AC4_GRP_CONCEALED;
    r->seq_jump = 0;
}

/* ------------------------------------------------------------ late */
static void add_pending(ac4dec *d, long long index, const uint8_t *sub,
                        size_t sub_len, const ac4_element *e, int ifr,
                        int five)
{
    pending_t *p;
    if (d->n_pend >= MAX_PENDING)
        return;
    p = &d->pend[d->n_pend];
    p->sub = (uint8_t *)malloc(sub_len ? sub_len : 1);
    if (!p->sub)
        return;
    memcpy(p->sub, sub, sub_len);
    p->sub_len = sub_len;
    p->index = index;
    p->bits = e->bits_end;
    p->start = e->audio_start;
    p->size = e->audio_size;
    p->ifr = ifr;
    p->acpl3 = e->acpl3;
    p->five = five;
    d->n_pend++;
}

/* one frame of the late check: -> 1 closes, 0 not, -1 not judged (an
   ASPX_ACPL_3 frame with no acpl_config_2ch known) */
static int late_one(ac4dec *d, const pending_t *pd, ac4_aspx_state *st)
{
    jmp_buf jb;
    ac4_bits b;
    size_t stop;
    b.d = pd->sub;
    b.nbits = pd->sub_len * 8;
    b.p = pd->bits;
    b.jb = &jb;
    b.eof_code = AC4E_ASPX_PARSE;
    if (setjmp(jb) != 0)
        return 0;
    ac4_aspx_data(&b, &d->abk, &d->tab, &d->acfg, pd->ifr, 1, st, 0, NULL);
    if (pd->five) {
        ac4_aspx_data(&b, &d->abk, &d->tab, &d->acfg, pd->ifr, 1, st, 1,
                      NULL);
        ac4_aspx_data(&b, &d->abk, &d->tab, &d->acfg, pd->ifr, 0, st, 2,
                      NULL);
    }
    if (pd->acpl3) {
        if (!d->has_pcfg)
            return -1;
        b.eof_code = AC4E_ACPL_PARSE;
        ac4_acpl_data_2ch(&b, &d->abk, &d->pcfg, NULL, NULL);
    }
    stop = pd->start + 8 * pd->size;
    return b.p <= stop && b.p + 7 >= stop;
}

/* Ac4Stream._late_closure, run when the first aspx_config is accepted:
   the frames from before it are re-checked with it, and those that do not
   close are concealed if they are still held, provided at least 90 % of
   the judged frames close */
static void late_closure(ac4dec *d)
{
    ac4_aspx_state st;
    int i, n_ver = 0, n_ok = 0;
    signed char *ver = (signed char *)malloc((size_t)(d->n_pend + 1));
    if (!ver) {
        free_pending(d);
        return;
    }
    memset(&st, 0, sizeof(st));
    for (i = 0; i < d->n_pend; i++) {
        int ok;
        ver[i] = -1;
        st.missing = 0;
        ok = late_one(d, &d->pend[i], &st);
        if (ok < 0)
            continue;
        if (!st.missing) {
            ver[i] = (signed char)ok;
            n_ver++;
            n_ok += ok;
        }
        if (!ok)
            memset(&st, 0, sizeof(st));
    }
    if (n_ver && !((double)n_ok < LATE_MAJORITY * (double)n_ver)) {
        for (i = 0; i < d->n_pend; i++) {
            rec_t *r;
            int brk;
            if (ver[i] != 0)
                continue;
            r = rec_at(d, d->pend[i].index);
            if (!r || r->code != AC4E_NONE)
                continue;           /* already synthesised: too late; or
                                       concealed already */
            brk = r->seq_jump;      /* (the Python keeps seq_break) */
            rec_conceal(d, r, AC4E_AUDIO_LATE);
            r->seq_jump = brk;
        }
    }
    free(ver);
    free_pending(d);
}

/* ------------------------------------------------------------ parse */
/* Parse and check frame `index` (Ac4Stream.decode_frames, one frame);
   nx = the next frame (has_next 0 at the end of the stream). */
static void parse_frame(ac4dec *d, const uint8_t *data, size_t len,
                        const uint8_t *nx, size_t nx_len, int has_next,
                        rec_t *r)
{
    jmp_buf jb;
    ac4_bits b;
    ac4_toc t;
    volatile int changed = AC4E_NONE, resume = -1, seq = -1, brk = 0;
    volatile int first_cfg = 0;
    ac4_aspx_state fstate0 = d->fstate;
    ac4_acpl_qstate aq0 = d->aq;
    int code = AC4E_NONE, k, ifr = 0, five, has_aspx, acpl3, complete;
    size_t o, sub_len = 0;
    const uint8_t *sub = NULL;
    toc_shape sh;
    ac4_element *e = &d->el;
    ac4_aspx_tab tab_new;
    const ac4_aspx_cfg *volatile cfg = NULL;
    const ac4_aspx_tab *volatile tab = NULL;
    size_t end;

    memset(&sh, 0, sizeof(sh));
    if (len < MIN_FRAME_BYTES || len > MAX_FRAME_BYTES || !data) {
        code = AC4E_MP4_SIZE;
        goto fail;
    }
    if (toc_try(data, len, &t) != 0) {
        code = AC4E_TOC_PARSE;
        goto fail;
    }
    r->seq = t.sequence_counter;
    r->iframe = t.b_iframe_global;
    /* check_toc */
    if (t.sequence_counter > 1020) {
        code = AC4E_TOC_SEQ_RANGE;
        goto fail;
    }
    if (t.n_sizes < 2 || (long)t.n_sizes != t.n_substreams
        || (double)t.toc_bytes + (double)t.payload_base + t.sizes_sum
           > (double)len || t.sizes_min < 1) {
        code = AC4E_TOC_SIZES;
        goto fail;
    }
    shape_of(&t, &sh);
    if (d->have_shape) {
        if (sh.bv != d->shape.bv)
            changed = AC4E_TOC_VERSION;
        else if (sh.fs != d->shape.fs || sh.fri != d->shape.fri)
            changed = AC4E_TOC_RATE;
        else if (!shape_eq_tail(&sh, &d->shape))
            changed = AC4E_TOC_SHAPE;
    }
    seq = t.sequence_counter;
    brk = !seq_continues(seq, d->seq_prev);
    if (brk && seq != 0 && has_next) {
        /* the NEXT frame carries on from the frame before this one (and
           not from this one): an outlier */
        int nxt = peek_counter(nx, nx_len), prev = d->seq_prev;
        if (nxt >= 0 && !seq_continues(nxt, seq)
            && (seq_continues(nxt, prev) || nxt == seq_next(seq_next(prev)))) {
            resume = seq_prev_of(nxt);
            code = AC4E_SEQ_OUTLIER;
            goto fail;
        }
    }
    /* the audio substream (substream 1) */
    o = t.toc_bytes + (size_t)t.payload_base + (size_t)t.sizes[0];
    if (o > len)
        o = len;
    sub_len = (size_t)t.sizes[1];
    if (sub_len > len - o)
        sub_len = len - o;
    sub = data + o;
    ifr = t.b_iframe_global;
    b.d = sub;
    b.nbits = sub_len * 8;
    b.p = 0;
    b.jb = &jb;
    b.eof_code = AC4E_ELEMENT_PARSE;
    code = setjmp(jb);
    if (code != 0)
        goto fail;
    if (d->element == AC4DEC_ELEMENT_PAIR)
        ac4_parse_pair(&b, &d->books, ifr, e);
    else
        ac4_parse_5x(&b, &d->books, ifr, e);
    if (changed == AC4E_NONE && d->mode >= 0 && e->mode != d->mode)
        changed = AC4E_ELEMENT_MODE;
    if (e->has_aspx_cfg) {
        if (ac4_aspx_tables(&e->aspx_cfg, &tab_new) != 0)
            ac4_fail(&b, AC4E_ASPX_CONFIG);
        cfg = &e->aspx_cfg;
        tab = &tab_new;
    } else if (d->has_cfg) {
        cfg = &d->acfg;
        tab = &d->tab;
    }
    acpl3 = e->acpl3;
    five = d->element != AC4DEC_ELEMENT_PAIR && !acpl3;
    has_aspx = !(d->element == AC4DEC_ELEMENT_PAIR && e->mode == 0);
    end = e->bits_end;
    r->af[0].kind = r->af[1].kind = AC4_GRP_NONE;
    r->af[0].acpl.kind = r->af[1].acpl.kind = AC4_GRP_NONE;
    if (cfg && has_aspx) {
        const ac4_acpl_cfg *pc = e->has_acpl_cfg ? &e->acpl_cfg
                                 : (d->has_pcfg ? &d->pcfg : NULL);
        b.p = e->bits_end;
        b.eof_code = AC4E_ASPX_PARSE;
        ac4_aspx_data(&b, &d->abk, tab, cfg, ifr, 1, &d->fstate, 0,
                      &r->af[0].g);
        r->af[0].kind = AC4_GRP_DATA;
        if (five) {
            ac4_aspx_data(&b, &d->abk, tab, cfg, ifr, 1, &d->fstate, 1,
                          &r->af[1].g);
            r->af[1].kind = AC4_GRP_DATA;
            ac4_aspx_data(&b, &d->abk, tab, cfg, ifr, 0, &d->fstate, 2,
                          NULL);
        }
        end = b.p;
        if (acpl3 && pc) {
            b.eof_code = AC4E_ACPL_PARSE;
            /* parsed, and differentially decoded (Pseudocode 121) */
            ac4_acpl_data_2ch(&b, &d->abk, pc, &d->aq, &r->af[0].acpl);
            end = b.p;
        }
    }
    complete = (cfg || !has_aspx)
               && (!acpl3 || e->has_acpl_cfg || d->has_pcfg);
    /* audio_size closure */
    {
        size_t stop = e->audio_start + 8 * e->audio_size;
        if (e->audio_size > sub_len || stop > 8 * sub_len)
            ac4_fail(&b, AC4E_AUDIO_OVERRUN);
        if (end > stop)
            ac4_fail(&b, AC4E_AUDIO_CLOSURE);
        if (complete && stop - end > 7)
            ac4_fail(&b, AC4E_AUDIO_CLOSURE);
    }
    if (!cfg && has_aspx)
        add_pending(d, r->index, sub, sub_len, e, ifr, five);
    /* dequantise (with the level check) */
    r->mask = chan_mask(d, acpl3);
    for (k = 0; k < 6; k++)
        if (r->mask & (1 << k))
            d->spec_n[k] = ac4_packed_spectrum(chan_src(e, k), d->spec[k],
                                               &jb);
    /* spectral noise fill: one generator per frame, reset from
       sequence_counter, through the element's sf_data blocks in bitstream
       order (pair: L R; 5_X: LFE L R Ls Rs C; ASPX_ACPL_3: LFE L R) */
    {
        static const int ORDER_PAIR[] = { CH_L, CH_R };
        static const int ORDER_5X[] = { CH_LFE, CH_L, CH_R, CH_LS, CH_RS,
                                        CH_C };
        const int *ord = ORDER_5X;
        int n_ord = acpl3 ? 3 : 6, i;
        ac4_snf_rng rng;
        if (d->element == AC4DEC_ELEMENT_PAIR) {
            ord = ORDER_PAIR;
            n_ord = 2;
        }
        ac4_snf_reset(&rng, t.sequence_counter);
        for (i = 0; i < n_ord; i++) {
            k = ord[i];
            if (r->mask & (1 << k))
                ac4_noise_fill(chan_src(e, k), d->spec[k], d->spec_n[k],
                               &rng, &jb);
            else
                ac4_noise_fill(chan_src(e, k), NULL, 0, &rng, &jb);
        }
    }
    if (changed != AC4E_NONE) {
        int at_i = ifr;
        if (ifr && changed != AC4E_ELEMENT_MODE) {
            /* a TOC change at an I-frame: the next frame's TOC must
               agree (robust.next_toc_agrees) */
            if (has_next) {
                ac4_toc tn;
                toc_shape shn;
                at_i = 0;
                if (toc_try(nx, nx_len, &tn) == 0) {
                    shape_of(&tn, &shn);
                    at_i = shape_eq(&shn, &sh);
                }
            }
        }
        if (!relearn_accept(&d->relearn, changed, &sh, e->mode, at_i))
            ac4_fail(&b, changed);
        d->shape = sh;
        d->have_shape = 1;
    } else {
        d->relearn.valid = 0;
        d->relearn.n = 0;
    }
    /* ---- accepted: commit */
    if (e->has_aspx_cfg) {
        first_cfg = !d->has_cfg;
        d->acfg = e->aspx_cfg;
        d->tab = tab_new;
        d->has_cfg = 1;
    }
    if (e->has_acpl_cfg) {
        d->pcfg = e->acpl_cfg;
        d->has_pcfg = 1;
    }
    if (!d->have_shape) {
        d->shape = sh;
        d->have_shape = 1;
    }
    d->mode = e->mode;
    if (acpl3)
        d->acpl3 = 1;
    /* stereo processing, ungroup */
    if ((r->mask & 3) == 3)
        ac4_unmix(d->spec[CH_L], d->spec[CH_R],
                  d->spec_n[CH_L] < d->spec_n[CH_R] ? d->spec_n[CH_L]
                                                    : d->spec_n[CH_R],
                  &e->lr.st, &e->lr.fr[0]);
    if ((r->mask & 0x30) == 0x30)
        ac4_unmix(d->spec[CH_LS], d->spec[CH_RS],
                  d->spec_n[CH_LS] < d->spec_n[CH_RS] ? d->spec_n[CH_LS]
                                                      : d->spec_n[CH_RS],
                  &e->sr.st, &e->sr.fr[0]);
    for (k = 0; k < 6; k++) {
        const ac4_framing *fr;
        int w, sum = 0;
        if (!(r->mask & (1 << k)))
            continue;
        fr = chan_src(e, k)->fr;
        ac4_ungroup(d->spec[k], d->spec_n[k], fr, r->win + k * FRAME,
                    r->wlen[k]);
        r->nwin[k] = fr->num_windows;
        for (w = 0; w < fr->num_windows; w++)
            sum += r->wlen[k][w];
        if (sum != FRAME) {         /* cannot happen; never trust it */
            code = AC4E_INTERNAL;
            goto fail;
        }
    }
    for (k = 0; k < 2; k++)
        if (r->af[k].kind == AC4_GRP_DATA) {
            r->af[k].has_tab = 1;
            r->af[k].tab = *tab;
        }
    if (!(r->mask & (1 << CH_LS)))
        r->af[1].kind = AC4_GRP_NONE;
    r->code = AC4E_NONE;
    r->seq_jump = brk;
    if (brk)
        d->n_seq_jumps++;
    d->seq_prev = seq;
    if (first_cfg && !d->late_done) {
        d->late_done = 1;
        if (d->n_pend)
            late_closure(d);
    }
    return;

fail:
    d->fstate = fstate0;
    d->aq = aq0;
    /* a frame concealed after it was put on the late list (a rule checked
       after add_pending) is not judged by the late check: it keeps its
       reason, is counted once, and its A-SPX does not feed the check's
       tracking state */
    if (d->n_pend && d->pend[d->n_pend - 1].index == r->index) {
        free(d->pend[d->n_pend - 1].sub);
        d->n_pend--;
    }
    rec_conceal(d, r, code);
    {
        int nxt = has_next ? peek_counter(nx, nx_len) : -1;
        if (resume >= 0)
            d->seq_prev = resume;
        else if (seq >= 0 && nxt >= 0 && seq_continues(nxt, seq))
            d->seq_prev = seq;
        else if (d->seq_prev >= 0)
            d->seq_prev = seq_next(d->seq_prev);
    }
}

/* parse the held raw frame (its successor nx, or none) into a record */
static void parse_held(ac4dec *d, const uint8_t *nx, size_t nx_len,
                       int has_next)
{
    long long index = d->n_parsed++;
    rec_t *r = rec_new(d, index);
    const uint8_t *data = d->raw_null ? NULL : d->raw;
    if (!d->element) {
        ac4dec_element el = (data && d->raw_len)
                            ? ac4dec_probe(data, d->raw_len)
                            : AC4DEC_ELEMENT_AUTO;
        if (el != AC4DEC_ELEMENT_AUTO)
            set_element(d, el);
    }
    if (!d->element) {
        /* nothing known yet: silence, timeline kept */
        rec_conceal(d, r, AC4E_UNKNOWN_ELEMENT);
        r->unknown = 1;
        r->mask = 0;
        r->af[0].kind = r->af[1].kind = AC4_GRP_NONE;
        r->af[0].acpl.kind = r->af[1].acpl.kind = AC4_GRP_NONE;
        return;
    }
    parse_frame(d, data, d->raw_len, nx, nx_len, has_next, r);
}

/* ------------------------------------------------------------ A-SPX stage */
static int pair_capable(const ac4dec *d, int p)
{
    if (d->mode < 0)
        return 0;
    if (d->element == AC4DEC_ELEMENT_PAIR)
        return p == 0 && d->mode == 1;
    if (p == 0)
        return 1;                   /* 5_X codec mode 1 or 4 */
    return !d->acpl3 && d->n_ch == 6;
}

static void np_frame_stats(const double *x, double *rms, double *amax)
{
    double sq[FRAME], m = 0.0;
    int i, nan = 0;
    for (i = 0; i < FRAME; i++) {
        double a = fabs(x[i]);
        sq[i] = x[i] * x[i];
        if (a != a)
            nan = 1;
        else if (a > m)
            m = a;
    }
    *rms = sqrt(ac4_np_sum(sq, FRAME) / (double)FRAME);
    *amax = nan ? NAN : m;
}

/* core_for_bursts for one frame of one pair: the burst guard and the
   aspx.level check. -> 1 when aspx.level flags the frame; *bad_out: the
   frame plays its core */
static int bursts(ac4dec *d, int p, double *y[2], double *dc[2],
                  int *bad_out)
{
    double limit = 2.0 * ac4dec_ref_peak((ac4dec_element)d->element);
    double floor_ = HF_MIN_FRAC * ac4dec_ref_peak((ac4dec_element)d->element);
    double pk = 0.0;
    int c, over = 0, hot, bad = 0;
    for (c = 0; c < 2; c++) {
        double rh, rc, hm, tmp;
        np_frame_stats(y[c], &rh, &hm);
        np_frame_stats(dc[c], &rc, &tmp);
        /* np.maximum of the two channels' peaks (NaN propagates) */
        pk = c == 0 ? hm : (pk != pk ? pk : (hm != hm ? hm
                                             : (hm > pk ? hm : pk)));
        if (!(rh <= MAX_HF_RATIO * rc) && !(hm <= floor_))
            over = 1;
    }
    hot = pk > limit || over;
    if (hot) {
        bad = 1;
        d->pair[p].hot_left = HF_TAIL;
    } else if (d->pair[p].hot_left) {
        bad = 1;
        d->pair[p].hot_left--;
    }
    if (bad) {
        d->n_burst++;
        for (c = 0; c < 2; c++)
            memcpy(y[c], dc[c], sizeof(double) * FRAME);
    }
    *bad_out = bad;
    return over;
}

/* frame peak over the output channels (np max: NaN propagates; then
   non-finite -> inf) */
static double frame_peak(const outf_t *o, int n_ch)
{
    double m = 0.0;
    int c, i;
    for (c = 0; c < n_ch; c++)
        for (i = 0; i < FRAME; i++) {
            double a = fabs(o->pcm[c][i]);
            if (!(a <= m)) {
                if (a != a)
                    return INFINITY;
                m = a;
            }
        }
    return isfinite(m) ? m : INFINITY;
}

/* assemble output frame (n_synth - 1) into the guard ring; hf[p] are the
   pairs' outputs (NULL when the pair gave none) */
static void assemble(ac4dec *d, double hf[2][2][FRAME], double dcore[2][2][FRAME],
                     const int *have)
{
    outf_t *o;
    const rec_t *ri = &d->prev_info;
    int k, p, bad;
    o = &d->ring[(int)((d->ring_first + d->ring_n) % d->ring_cap)];
    if (d->ring_n == 0)
        d->ring_first = ri->index;
    d->ring_n++;
    o->index = ri->index;
    o->code = ri->code;
    o->unknown = ri->unknown;
    o->seq = ri->seq;
    o->iframe = ri->iframe;
    o->seq_jump = ri->seq_jump;
    o->forced = ri->code != AC4E_NONE;
    o->aspx_flag = 0;
    /* every channel 577 samples late: the delay line, unless its pair
       is on the QMF path (below) */
    for (k = 0; k < 6; k++) {
        memcpy(o->pcm[k], d->core_tail[k], sizeof(double) * HF_DELAY);
        memcpy(o->pcm[k] + HF_DELAY, d->core_prev[k],
               sizeof(double) * (FRAME - HF_DELAY));
        memcpy(d->core_tail[k], d->core_prev[k] + FRAME - HF_DELAY,
               sizeof(double) * HF_DELAY);
    }
    for (p = 0; p < 2; p++) {
        double *y[2], *dc[2];
        int a = p == 0 ? CH_L : CH_LS;
        if (!d->path_on[p] && d->pair_active[p] && have[p])
            d->path_on[p] = 1;
        if (!d->path_on[p] || !have[p])
            continue;
        y[0] = hf[p][0];
        y[1] = hf[p][1];
        dc[0] = dcore[p][0];
        dc[1] = dcore[p][1];
        if (bursts(d, p, y, dc, &bad))
            o->aspx_flag = 1;
        memcpy(o->pcm[a], y[0], sizeof(o->pcm[a]));
        memcpy(o->pcm[a + 1], y[1], sizeof(o->pcm[a + 1]));
        if (p == 0 && d->pair[0].acpl_on) {
            /* the A-CPL upmix (decoder.py): a frame the checks above gave
               its core plays the core (the stereo downmix) on L / R and
               nothing on C / Ls / Rs */
            static const int CH5[5] = { CH_L, CH_R, CH_C, CH_LS, CH_RS };
            int c;
            for (c = 0; c < 5; c++) {
                if (bad && c < 2)
                    continue;
                if (bad)
                    memset(o->pcm[CH5[c]], 0, sizeof(o->pcm[0]));
                else
                    memcpy(o->pcm[CH5[c]], d->pair[0].upout[c],
                           sizeof(o->pcm[0]));
            }
        }
    }
    /* channels a frame does not have are silent */
    for (k = d->n_ch; k < 6; k++)
        memset(o->pcm[k], 0, sizeof(o->pcm[k]));
    o->pk = frame_peak(o, d->n_ch);
    o->ok = !o->forced && isfinite(o->pk);
}

/* synthesise the oldest held record */
static void synth_one(ac4dec *d)
{
    rec_t *r = &d->recs[(int)(d->rec_first % d->rec_cap)];
    double (*hf)[2][FRAME] = d->hf, (*dcore)[2][FRAME] = d->dcore;
    int have[2] = { 0, 0 }, k, w, p, i;
    /* IMDCT / overlap-add */
    for (k = 0; k < 6; k++) {
        int pos = 0;
        if (!(r->mask & (1 << k))) {
            memset(d->core[k], 0, sizeof(d->core[k]));
            continue;
        }
        for (w = 0; w < r->nwin[k]; w++) {
            ac4_fb_block(d->fbt, &d->fb[k], r->win + k * FRAME + pos,
                         r->wlen[k][w], d->core[k] + pos, d->tmp);
            pos += r->wlen[k][w];
        }
        for (i = 0; i < FRAME; i++)
            if (!isfinite(d->core[k][i]))
                d->core[k][i] = 0.0;
    }
    /* A-SPX: concealed frames take the stream's configuration */
    for (p = 0; p < 2; p++) {
        ac4_aspx_frame *f = &r->af[p];
        if (f->kind == AC4_GRP_CONCEALED) {
            f->has_tab = d->has_cfg;
            if (d->has_cfg)
                f->tab = d->tab;
        }
        if (!d->pair_active[p] && pair_capable(d, p)) {
            d->pair_active[p] = 1;
            d->pair_start[p] = r->index;
            /* ASPX_ACPL_3 decoded to 5.1: the A-CPL upmix on the core
               pair, from its first frame */
            if (p == 0 && d->acpl3 && d->n_ch == 6
                && d->element == AC4DEC_ELEMENT_5X)
                d->pair[0].acpl_on = 1;
        }
        if (d->pair_active[p]) {
            int a = p == 0 ? CH_L : CH_LS;
            const double *x[2];
            double *y[2], *dc[2];
            x[0] = d->core[a];
            x[1] = d->core[a + 1];
            y[0] = hf[p][0];
            y[1] = hf[p][1];
            dc[0] = dcore[p][0];
            dc[1] = dcore[p][1];
            have[p] = ac4_aspx_pair_push(&d->pair[p], d->qmf, x, f, y, dc);
        }
    }
    if (d->n_synth >= 1)
        assemble(d, hf, dcore, have);
    memcpy(d->core_prev, d->core, sizeof(d->core));
    d->prev_info = *r;
    d->prev_info.win = NULL;
    d->n_synth++;
    d->rec_first++;
    d->rec_n--;
}

/* end of stream: the last frame */
static void finish(ac4dec *d)
{
    double (*hf)[2][FRAME] = d->hf, (*dcore)[2][FRAME] = d->dcore;
    int have[2] = { 0, 0 }, p;
    if (d->finished)
        return;
    d->finished = 1;
    if (d->n_synth < 1)
        return;
    for (p = 0; p < 2; p++)
        if (d->pair_active[p]) {
            double *y[2], *dc[2];
            y[0] = hf[p][0];
            y[1] = hf[p][1];
            dc[0] = dcore[p][0];
            dc[1] = dcore[p][1];
            have[p] = ac4_aspx_pair_finish(&d->pair[p], d->qmf, y, dc);
        }
    assemble(d, hf, dcore, have);
}

/* ------------------------------------------------------------ guard */
static int sustained(const ac4dec *d, int f, double ref)
{
    const outf_t *o = &d->ring[(int)((d->ring_first + f) % d->ring_cap)];
    double thr = ref > o->pk / G_SPREAD ? ref : o->pk / G_SPREAD;
    int i, n = 0;
    if (f + G_L >= d->ring_n)
        return 0;
    for (i = 1; i <= G_L; i++) {
        const outf_t *a = &d->ring[(int)((d->ring_first + f + i)
                                         % d->ring_cap)];
        if (a->pk > thr && a->ok)
            n++;
    }
    return n >= G_S;
}

static int cmp_desc(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) - (x > y);
}

/* robust.OutputGuard.process for the oldest frame in the ring */
static int guard_one(ac4dec *d, outf_t *o)
{
    guard_t *g = &d->guard;
    double ref, lv, down;
    int near = 0, loud, i, c;
    if (!g->seeded) {
        double w[G_WARM];
        int n = 0;
        for (i = 0; i < d->ring_n && i < G_WARM; i++) {
            const outf_t *a = &d->ring[(int)((d->ring_first + i)
                                             % d->ring_cap)];
            if (a->ok)
                w[n++] = a->pk;
        }
        qsort(w, (size_t)n, sizeof(double), cmp_desc);
        g->ref = n ? w[(n < G_KTH ? n : G_KTH) - 1] : 0.0;
        if (g->ref < g->floor)
            g->ref = g->floor;
        g->seeded = 1;
    }
    lv = g->ref;
    if (g->init > lv)
        lv = g->init;
    if (g->floor > lv)
        lv = g->floor;
    ref = lv;
    g->init *= G_INIT_RELEASE;
    for (i = 0; i < g->n_hist; i++)
        if (g->forced_hist[i])
            near = 1;
    for (i = 0; i <= G_NEAR && i < d->ring_n; i++) {
        const outf_t *a = &d->ring[(int)((d->ring_first + i) % d->ring_cap)];
        if (a->forced)
            near = 1;
    }
    if (near) {
        double m = g->ref > g->floor ? g->ref : g->floor;
        loud = o->pk > G_NEAR_RATIO * m;
    } else {
        loud = o->pk > G_RATIO * ref;
    }
    if (loud && o->ok && !near && sustained(d, 0, ref)) {
        loud = 0;
        g->n_sustained++;
    }
    if (loud) {
        g->n_fired++;
        for (c = 0; c < d->n_ch; c++)
            memset(o->pcm[c], 0, sizeof(o->pcm[c]));
        if (!g->muted && g->has_last)
            for (c = 0; c < d->n_ch; c++)
                for (i = 0; i < G_RAMP; i++) {
                    down = i == G_RAMP - 1 ? 0.0
                           : (double)i * (-1.0 / (double)(G_RAMP - 1)) + 1.0;
                    o->pcm[c][i] = g->last[c] * down;
                }
        g->muted = 1;
    } else {
        double rise = 0.0, nr;
        if (g->muted) {
            for (c = 0; c < d->n_ch; c++)
                for (i = 0; i < G_RAMP; i++)
                    o->pcm[c][i] *= (double)i * (1.0 / (double)G_RAMP);
            g->muted = 0;
        }
        if (o->ok && !near)
            rise = o->pk < G_UP * g->ref ? o->pk : G_UP * g->ref;
        nr = g->ref * G_RELEASE;
        if (rise > nr)
            nr = rise;
        if (g->floor > nr)
            nr = g->floor;
        g->ref = nr;
    }
    for (c = 0; c < d->n_ch; c++)
        g->last[c] = o->pcm[c][FRAME - 1];
    g->has_last = 1;
    /* history of forced flags (and of frames muted by the plain rule, not
       by the near rule, so it cannot chain) for the frames after this one:
       every channel is 577 samples late, so a damaged frame's core still
       reaches into the second output frame after it */
    if (g->n_hist < G_NEAR) {
        g->forced_hist[g->n_hist++] = o->forced || (loud && !near);
    } else {
        memmove(g->forced_hist, g->forced_hist + 1,
                sizeof(int) * (G_NEAR - 1));
        g->forced_hist[G_NEAR - 1] = o->forced || (loud && !near);
    }
    return loud;
}

/* judge and give out the oldest ring frame */
static void emit(ac4dec *d, ac4dec_frame *out)
{
    outf_t *o = &d->ring[(int)(d->ring_first % d->ring_cap)];
    int fired = 0, k, nout = 0, code;
    if (!d->cfg.no_guard) {
        fired = guard_one(d, o);
    } else {
        for (k = 0; k < d->n_ch; k++) {
            int i;
            for (i = 0; i < FRAME; i++)
                if (!isfinite(o->pcm[k][i]))
                    o->pcm[k][i] = 0.0;
        }
    }
    d->cur = *o;
    d->ring_first++;
    d->ring_n--;
    o = &d->cur;
    code = o->code;
    if (code == AC4E_NONE && fired)
        code = AC4E_GUARD_LEVEL;
    else if (code == AC4E_NONE && o->aspx_flag)
        code = AC4E_ASPX_LEVEL;
    memset(out, 0, sizeof(*out));
    out->index = o->index;
    out->sequence_counter = o->seq;
    out->iframe = o->iframe;
    out->seq_jump = o->seq_jump;
    d->n_out++;
    if (o->unknown) {
        out->status = AC4DEC_BAD;
        out->reason = REASONS[AC4E_UNKNOWN_ELEMENT];
        d->n_bad++;
        d->n_concealed++;
        d->reasons[AC4E_UNKNOWN_ELEMENT]++;
    } else if (code != AC4E_NONE) {
        out->status = AC4DEC_CONCEALED;
        out->reason = ac4_reason_name(code);
        d->n_concealed++;
        d->reasons[code > 0 && code < AC4E_COUNT ? code : AC4E_INTERNAL]++;
        if (code == AC4E_GUARD_LEVEL)
            d->n_guard++;
        else if (code == AC4E_ASPX_LEVEL)
            d->n_aspx_level++;
        else
            d->n_bad++;
        if (code == AC4E_AUDIO_LATE)
            d->n_late++;
    } else {
        out->status = AC4DEC_OK;
        d->n_ok++;
    }
    out->n_channels = d->n_ch;
    if (d->cfg.internal_rate) {
        for (k = 0; k < d->n_ch; k++)
            out->pcm[k] = o->pcm[k];
        out->n_samples = FRAME;
    } else {
        for (k = 0; k < d->n_ch; k++)
            nout = ac4_rs_push(d->rsf, &d->rs[k], o->pcm[k], FRAME,
                               d->pcm_out[k]);
        for (k = 0; k < d->n_ch; k++)
            out->pcm[k] = d->pcm_out[k];
        out->n_samples = nout;
    }
}

/* ------------------------------------------------------------ API */
static int hold_raw(ac4dec *d, const uint8_t *data, size_t len)
{
    if (len > d->raw_cap) {
        size_t cap = len < 4096 ? 4096 : len;
        uint8_t *n = (uint8_t *)realloc(d->raw, cap);
        if (!n)
            return -1;
        d->raw = n;
        d->raw_cap = cap;
    }
    if (len)
        memcpy(d->raw, data, len);
    d->raw_len = len;
    d->raw_null = data == NULL;
    d->has_raw = 1;
    return 0;
}

int ac4dec_decode_frame(ac4dec *d, const uint8_t *data, size_t len,
                        ac4dec_frame *out)
{
    ac4dec_frame dummy;
    if (!out)
        out = &dummy;
    memset(out, 0, sizeof(*out));
    out->sequence_counter = -1;
    out->index = -1;
    if (!d)
        return 0;
    if (d->flushing)
        stream_reset(d);
    if (!data)
        len = 0;
    if (len > MAX_FRAME_BYTES + 1)
        len = MAX_FRAME_BYTES + 1;      /* too large anyway: mp4.size */
    d->n_frames++;
    d->n_in++;
    if (d->has_raw) {
        /* parse the held frame now that its successor is here */
        parse_held(d, data, len, 1);
    }
    if (hold_raw(d, data, len) != 0)
        hold_raw(d, NULL, 0);           /* out of memory: a lost frame */
    while (d->rec_n > d->L)
        synth_one(d);
    if (d->ring_n > d->G) {
        emit(d, out);
        return 1;
    }
    return 0;
}

int ac4dec_flush(ac4dec *d, ac4dec_frame *out)
{
    ac4dec_frame dummy;
    if (!out)
        out = &dummy;
    memset(out, 0, sizeof(*out));
    out->sequence_counter = -1;
    out->index = -1;
    if (!d)
        return 0;
    d->flushing = 1;
    for (;;) {
        int upstream = d->has_raw || d->rec_n > 0 || !d->finished;
        if (d->ring_n > 0 && (d->ring_n > d->G || !upstream)) {
            emit(d, out);
            return 1;
        }
        if (d->has_raw) {
            parse_held(d, NULL, 0, 0);
            d->has_raw = 0;
        } else if (d->rec_n > 0) {
            synth_one(d);
        } else if (!d->finished) {
            finish(d);
        } else {
            return 0;
        }
    }
}
