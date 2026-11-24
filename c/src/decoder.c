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
#define FRAME AC4_MAX_FRAME   /* buffer stride; a frame is d->fl long */
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
#define G_TP_WIN 3            /* truepk window: f - 3 .. f + 3 */
#define G_TP_MARGIN_DB 20.0
#define G_TP_NEAR_DB 12.0
#define G_TP_STEP 0.25

enum { CH_L, CH_R, CH_C, CH_LFE, CH_LS, CH_RS, CH_X1, CH_X2,
       CH_LB, CH_RB, CH_TBL, CH_TBR };
#define NCH AC4_NCH                /* coded track slots (CH_L .. CH_X2) */
#define NOUT AC4DEC_MAX_CHANNELS   /* output slots: the tracks, then the
                                      immersive A-CPL's Lb Rb Tbl Tbr */

/* the QMF-path "pairs" (the A-SPX block groups): L/R, Ls/Rs, the centre
   as C/C (its aspx_data_1ch rendered as the same channel twice,
   decoder.py), and 7_X's fourth pair; d->pair_ch by the element's
   layout (Table 213) */
#define NPAIR 4

static const char *REASONS[AC4E_COUNT] = {
    "none", "toc.parse", "toc.seq_range", "toc.sizes", "toc.version",
    "toc.rate", "toc.shape", "mp4.size", "audio_size.overrun",
    "audio_size.closure", "element.mode", "element.parse", "asf.sect_cb",
    "asf.sect_len", "asf.max_sfb", "asf.quant", "asf.sf_range", "asf.level",
    "decoder.internal", "element.unknown", "seq.outlier", "aspx.config",
    "aspx.parse", "aspx.num_env", "aspx.borders", "acpl.parse",
    "audio_size.late", "guard.level", "aspx.level", "toc.presentation",
    "toc.unsupported_rate", "asf.transf_length", "asf.chel_matsel",
    "asf.add_ch", "guard.truepk"
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
    int nwin[NCH];
    int wlen[NCH][AC4_MAX_WIN];
    double *win;                /* NCH x FRAME */
    ac4_aspx_frame af[NPAIR];   /* L/R, Ls/Rs, C (as C/C) */
    ac4_meta meta;              /* presentation substream (valid 0: none) */
    ac4_de_entry de;            /* dialogue enhancement data */
    int has_obj;                /* AJOC: the frame's data parsed (in the
                                   decoder's ajf ring) */
} rec_t;

/* a frame from before the first aspx_config (audio_size.late) */
typedef struct {
    long long index;
    uint8_t *sub;
    size_t sub_len;
    size_t bits, start, size;
    int ifr, acpl3, n_acpl1;
    int n_blocks, blk_grp[4], blk_two[4];
} pending_t;

/* an output frame waiting for the guard */
typedef struct {
    long long index;
    int code, unknown, seq, iframe, seq_jump;
    int forced;                 /* concealed upstream */
    int aspx_flag;              /* aspx.level */
    int tp_fired;               /* guard.truepk */
    double pk;                  /* peak over all output channels */
    int ok;
    int n;                      /* samples (the frame length) */
    ac4_meta meta;
    int has_obj;                /* AJOC: the frame's downmix objects' block
                                   updates (core decoding) */
    int n_dev, dev[AC4_OAMD_NB][2];
    ac4_obj dblk[AC4_OAMD_NB][17];
    double pcm[NOUT][FRAME];
} outf_t;

typedef struct {
    double floor, init, ref;
    int seeded;
    double last[NOUT];
    int has_last, muted;
    int forced_hist[G_NEAR];    /* forced flags of the last frames judged */
    double tp_hist[G_TP_WIN];   /* truepk of the last frames judged */
    int tp_has[G_TP_WIN], n_tp;
    double tp_off;              /* decoded peak re truepk, dB (tracked) */
    int n_hist;
    long n_fired, n_sustained, n_truepk;
} guard_t;

struct ac4dec {
    ac4dec_config cfg;
    int N, L, G;
    int element;                /* resolved ac4dec_element, 0 = not yet */
    int n_ch;
    int ch_mode;                /* 7_X: Table 88 ch_mode; ICE: 11 / 12 */
    int ice_back4, ice_top;     /* ICE: b_4_back_channels_present,
                                   top_channels_present */
    int ice_known;              /* ... read from the stream's TOC */
    /* AJOC: the substream's info (ac4_toc_sub, from the TOC), the
       objects' state (oamd.c), the A-JOC codebooks, the downmix objects
       held for rendering */
    ac4_toc_sub ajoc_info;
    int ajoc_known;
    ac4_obj_state obj, obj_bak;
    ac4_huff ajoc_huff[AC4T_N_AJOC_BOOKS];
    /* the frames' A-JOC data (a ring indexed by frame), full decoding
       (ajoc.c) and its output frame; core decoding's rendering ramp
       (objrender.SampleRamp) */
    ac4_ajoc_frame *ajf;
    int ajf_cap;
    int ajoc_full;
    ac4_ajoc_full *ajx;
    double ajout[NOUT][FRAME];
    double ajin_r[AC4_AJOC_ND][AC4_MAX_NTS][64];
    double ajin_i[AC4_AJOC_ND][AC4_MAX_NTS][64];
    int or_init, or_n, or_k;
    /* the downmix loudness correction held (render.Renderer.lc) and
       whether the layout is a downmix */
    unsigned char lc_has[AC4_LC_N];
    double lc[AC4_LC_N];
    int is_dmx;
    double or_cur[NOUT * NOUT], or_start[NOUT * NOUT], or_tgt[NOUT * NOUT];
    int out_ch[NOUT];           /* output channel c is slot out_ch[c] */
    /* an output layout (cfg.layout, render.c): n_lay channels from the
       n_ch decoded ones through rmat (n_lay x n_ch), with the custom
       downmix data held from the frames */
    int n_lay;
    double rmat[NOUT * NOUT];
    ac4_cdmx cdmx_held;
    int el_mask;                /* the element's slots */
    int pair_ch[NPAIR][2];
    /* the stream's rate (rates.py): (fs_index, frame_rate_index) of its
       first decodable frame whose next frame agrees; fl = frame_len_base
       (AC4_DEFAULT_FRAME until then); the output resampler is fixed at
       the first frame given out */
    int rate_known, fs, fri, fl, nats;
    int rs_fixed, rs_up, rs_down, out_rate;
    ac4_huff huff[AC4T_N_BOOKS];
    ac4_huff drc_hcb;           /* DRC_HCB */
    ac4_huff de_hcb[4];         /* DE_HCB_ABS_0, DIFF_0, ABS_1, DIFF_1 */
    ac4_de_state de_state;
    ac4_drc_config drc_state;   /* from the last I-frame */
    ac4_leveler *lev;           /* cfg.loudness: made at the first emit */
    ac4_books books;
    ac4_aspx_books abk;
    ac4_fb_tables *fbt;
    ac4_rs_filter *rsf;
    ac4_qmf_tables *qmf;
    ac4_element el;
    double spec[NCH][FRAME];
    int spec_n[NCH];
    double tmp[4 * FRAME];
    double zeros[FRAME];

    /* ---- stream state (Ac4Stream) */
    int acpl3;
    int acpl1;                  /* AC4_ACPL1_* once seen */
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
    int aq1[4][2][AC4_ACPL_NB]; /* the 1ch modules' alpha1 / beta1 q_prev */
    int aq1n[4][2];
    int has_pcfg1;
    ac4_acpl1_cfg pcfg1;
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
    ac4_fb_state fb[NCH];
    long long n_synth;
    double core[NCH][FRAME];
    double core_prev[NCH][FRAME];
    double core_tail[NCH][HF_DELAY];   /* delay line: last 577 core samples */
    rec_t prev_info;            /* frame n_synth - 1 (no windows) */
    ac4_aspx_pair pair[NPAIR];
    int pair_active[NPAIR];
    ac4_aspx_tab syn_tab[NPAIR];    /* tables of the last frame with data
                                       synthesised, per pair */
    int has_syn_tab[NPAIR];
    long long pair_start[NPAIR];    /* first frame the pair took */
    int path_on[NPAIR];
    int finished;
    double hf[NPAIR][2][FRAME];     /* the pairs' A-SPX output frames */
    double reorder[NOUT][FRAME];    /* assemble: output order scratch */
    double dcore[NPAIR][2][FRAME];  /* and their cores, 577 samples late */

    /* ---- guard (ring of G + 2) */
    outf_t *ring;
    int ring_cap, ring_n;
    long long ring_first;
    guard_t guard;

    /* ---- output */
    ac4_rs_state rs[NOUT];
    double pcm_out[NOUT][AC4DEC_MAX_OUT + 2];
    outf_t cur;                 /* the frame last given out */
    int flushing;

    /* ---- stats */
    long n_frames, n_out, n_ok, n_concealed, n_bad, n_seq_jumps;
    long n_guard, n_aspx_level, n_burst, n_late;
    int v0;                     /* the frame's presentation is
                                   bitstream_version 0 / 1 */
    /* ---- mixing (mix.py): the other substreams' decoders */
    int mix_decided, n_kid, mix_n, mix_version;
    /* an object substream as the main one (phase E): its objects are
       rendered to its layout before the mix, which then runs on the
       layout's channels (lay_slot: their slots by name) */
    int objmix, lay_slot[NOUT];
    double mix_gprev[AC4_MIX_MAX][NOUT][NOUT];  /* last frame's matrices */
    int mix_gn[AC4_MIX_MAX], mix_ghas[AC4_MIX_MAX];
    ac4dec *kid[AC4_MIX_MAX - 1];
    ac4dec_frame kid_out[AC4_MIX_MAX - 1];
    int kid_have[AC4_MIX_MAX - 1];
    int mix_type[AC4_MIX_MAX], mix_sg[AC4_MIX_MAX];  /* [0]: this one */
    int mix_n_sg;
    double mix_sgdb[AC4_MAX_SG];
    int mix_has_assoc, mix_has_sc[4];
    double mix_sc[4];
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

/* the presentation this decoder takes (config; TS 103 190-2 4.8.2) */
static int choose_pres(const ac4dec_config *c, const ac4_toc *t)
{
    return ac4_toc_choose(t, c->presentation - 1, c->presentation_id - 1,
                          c->language);
}

/* the audio substream this decoder takes: the cfg->substream-th
   channel-coded substream of the chosen presentation */
static const ac4_toc_sub *chosen_sub(const ac4dec_config *c,
                                     const ac4_toc *t)
{
    const ac4_toc_sub *s[AC4_MIX_MAX];
    int k = c->substream > 0 ? c->substream : 0;
    int n = ac4_toc_audio_substreams(t, choose_pres(c, t), s, NULL,
                                     AC4_MIX_MAX);
    return k < n ? s[k] : NULL;
}

/* the element of the audio substream of the chosen presentation */
static ac4dec_element element_of_toc(const ac4dec_config *c, const ac4_toc *t)
{
    const ac4_toc_sub *s = chosen_sub(c, t);
    long cm;
    if (!s)
        return AC4DEC_ELEMENT_AUTO;
    if (s->kind == 1 || s->kind == 2)
        return AC4DEC_ELEMENT_AJOC;
    cm = s->channel_mode;
    switch (cm) {
    case 0: return AC4DEC_ELEMENT_MONO;
    case 1: return AC4DEC_ELEMENT_PAIR;
    case 2: return AC4DEC_ELEMENT_3_0;
    case 3: return AC4DEC_ELEMENT_5_0;
    case 4: return AC4DEC_ELEMENT_5X;
    default: break;
    }
    if (cm >= 5 && cm <= 10) {
        /* presentation_version 2 with a 7.x mode (the kit's "IMS"): a
           channel_pair_element (decoder.py IMS_STEREO) */
        int pi = choose_pres(c, t);
        if (pi >= 0 && pi < t->n_pres && t->pres[pi].version == 2)
            return AC4DEC_ELEMENT_PAIR;
        return AC4DEC_ELEMENT_7X;
    }
    if (cm >= 11 && cm <= 14)
        return AC4DEC_ELEMENT_ICE;
    return AC4DEC_ELEMENT_AUTO;
}

/* the 7_X ch_mode of the chosen presentation's audio substream, or 0 */
static int ch_mode_of_toc(const ac4dec_config *c, const ac4_toc *t)
{
    const ac4_toc_sub *s = chosen_sub(c, t);
    return s && s->channel_mode >= 5 && s->channel_mode <= 14
           ? (int)s->channel_mode : 0;
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
    ac4dec_config c;
    memset(&c, 0, sizeof(c));
    if (len > MAX_FRAME_BYTES || toc_try(data, len, &t) != 0)
        return AC4DEC_ELEMENT_AUTO;
    return element_of_toc(&c, &t);
}

ac4dec_element ac4dec_probe_config(const ac4dec_config *cfg,
                                   const uint8_t *data, size_t len)
{
    ac4_toc t;
    ac4dec_config c;
    memset(&c, 0, sizeof(c));
    if (cfg)
        c = *cfg;
    if (len > MAX_FRAME_BYTES || toc_try(data, len, &t) != 0)
        return AC4DEC_ELEMENT_AUTO;
    return element_of_toc(&c, &t);
}

int ac4dec_probe_ch_mode(const ac4dec_config *cfg, const uint8_t *data,
                         size_t len)
{
    ac4_toc t;
    ac4dec_config c;
    memset(&c, 0, sizeof(c));
    if (cfg)
        c = *cfg;
    if (len > MAX_FRAME_BYTES || toc_try(data, len, &t) != 0)
        return 0;
    if (element_of_toc(&c, &t) != AC4DEC_ELEMENT_7X
        && element_of_toc(&c, &t) != AC4DEC_ELEMENT_ICE)
        return 0;
    return ch_mode_of_toc(&c, &t);
}

int ac4dec_probe_present(const ac4dec_config *cfg, const uint8_t *data,
                         size_t len)
{
    ac4_toc t;
    ac4dec_config c;
    const ac4_toc_sub *s;
    memset(&c, 0, sizeof(c));
    if (cfg)
        c = *cfg;
    if (len > MAX_FRAME_BYTES || toc_try(data, len, &t) != 0)
        return 0;
    s = chosen_sub(&c, &t);
    if (!s || s->channel_mode < 11 || s->channel_mode > 14)
        return 0;
    return 16 | (s->back4 ? 1 : 0) | (s->centre ? 2 : 0) | (s->top << 2);
}

int ac4dec_channels(const ac4dec *d)
{
    return d->element ? d->n_lay : 0;
}

/* the render ids (render.c RN_*) of the decoded output channels */
static void render_ids(const ac4dec *d, int *ids)
{
    int k;
    for (k = 0; k < d->n_ch; k++) {
        int sl = d->out_ch[k];
        switch (sl) {
        case CH_L: ids[k] = RN_L; break;
        case CH_R: ids[k] = RN_R; break;
        case CH_C: ids[k] = RN_C; break;
        case CH_LFE: ids[k] = RN_LFE; break;
        case CH_LS: ids[k] = RN_LS; break;
        case CH_RS: ids[k] = RN_RS; break;
        case CH_LB: ids[k] = RN_LB; break;
        case CH_RB: ids[k] = RN_RB; break;
        case CH_TBL: ids[k] = RN_TBL; break;
        case CH_TBR: ids[k] = RN_TBR; break;
        case CH_X1: case CH_X2: {
            int r = sl == CH_X2;
            if (d->element == AC4DEC_ELEMENT_ICE
                || (d->element == AC4DEC_ELEMENT_7X && d->ch_mode >= 9))
                ids[k] = r ? RN_TFR : RN_TFL;
            else if (d->element == AC4DEC_ELEMENT_7X && d->ch_mode >= 7)
                ids[k] = r ? RN_RW : RN_LW;
            else
                ids[k] = r ? RN_RB : RN_LB;
            break;
        }
        default: ids[k] = -1; break;
        }
    }
}

static int ajoc_layout(const ac4dec *d);

/* decoder.ObjectRenderer._matrix: the downmix objects of an AJOC
   substream (states st, or NULL: unknown) to its layout -> M (n_lay x
   n_ch). The decoded slot j holds A-JOC input (object) slot_obj[j]
   (Pseudocode 14a; the LFE first). */
static void obj_matrix(const ac4dec *d, const ac4_obj *st, double *M)
{
    const ac4_toc_sub *ai = &d->ajoc_info;
    int lay = ajoc_layout(d), j, lfe = ai->lfe;
    int ids[NOUT], pos[AC4_AJOC_ND], slot_obj[NOUT];
    signed char ty[NOUT];
    ac4_obj ss[NOUT];
    render_ids(d, ids);
    if (d->ajoc_known && ai->kind == 2) {
        /* direct-coded objects: the LFE slot first, then the element's
           tracks in order (toc.obj_split's OAMD order) */
        int nl = 0, q;
        for (q = 0; q < ai->n_types; q++)
            if (ai->types[q] == RN_LFE)
                slot_obj[nl++] = q;
        for (q = 0; q < ai->n_types; q++)
            if (ai->types[q] != RN_LFE)
                slot_obj[nl++] = q;
        for (j = 0; j < d->n_ch; j++) {
            int k = j < nl ? slot_obj[j] : -1;
            ty[j] = k >= 0 ? ai->types[k] : AC4_OBJ_DYN;
            if (st && k >= 0)
                ss[j] = st[k];
            else
                ac4_obj_init(&ss[j]);
        }
        ac4_obj_matrix_ex(ty, st ? ss : NULL, d->n_ch, lay,
                          ac4_bed_of(ai->obj_bed), ai->isf_cfg, M);
        return;
    }
    if (!d->ajoc_known || ai->static_dmx) {
        ac4_render_matrix(ids, d->n_ch, lay, NULL, M);
        return;
    }
    for (j = 0; j < d->n_ch; j++)
        slot_obj[j] = j;
    ac4_ajoc_input_order(d->n_ch - lfe, pos);
    for (j = 0; j < d->n_ch - lfe; j++)
        slot_obj[pos[j] + lfe] = j + lfe;
    for (j = 0; j < d->n_ch; j++) {
        int k = slot_obj[j];
        ty[j] = lfe ? (k == 0 ? RN_LFE : ai->dmx_type[k - 1])
                    : ai->dmx_type[k];
        if (st)
            ss[j] = st[k];
    }
    ac4_obj_matrix(ty, st ? ss : NULL, d->n_ch, lay, M);
}

/* objrender.SampleRamp.event */
static void or_event(ac4dec *d, const ac4_obj *st, int n)
{
    int nm = d->n_lay * d->n_ch, i;
    obj_matrix(d, st, d->or_tgt);
    if (!d->or_init || n <= 0) {
        memcpy(d->or_cur, d->or_tgt, sizeof(double) * (size_t)nm);
        d->or_n = d->or_k = 0;
        d->or_init = 1;
        return;
    }
    for (i = 0; i < nm; i++)
        d->or_start[i] = d->or_cur[i];
    d->or_n = n;
    d->or_k = 0;
}

/* objrender.SampleRamp.apply on samples [i0, i1) of o into tmp */
static void or_apply(ac4dec *d, const outf_t *o, int i0, int i1,
                     double (*tmp)[FRAME])
{
    int no = d->n_lay, nc = d->n_ch, nm = no * nc, i = i0, c, j, m;
    double M[NOUT * NOUT];
    while (i < i1) {
        if (d->or_k < d->or_n) {
            int mm = i1 - i < d->or_n - d->or_k ? i1 - i : d->or_n - d->or_k;
            for (j = 0; j < mm; j++) {
                double t = (double)d->or_k / d->or_n;
                for (m = 0; m < nm; m++)
                    M[m] = d->or_start[m] + (d->or_tgt[m] - d->or_start[m])
                                            * t;
                for (c = 0; c < no; c++) {
                    double acc = 0.0;
                    int k;
                    for (k = 0; k < nc; k++)
                        acc = acc + M[c * nc + k] * o->pcm[k][i + j];
                    tmp[c][i + j] = acc;
                }
                d->or_k++;
            }
            if (d->or_k >= d->or_n) {
                memcpy(d->or_cur, d->or_tgt, sizeof(double) * (size_t)nm);
            } else {
                double t = (double)d->or_k / d->or_n;
                for (m = 0; m < nm; m++)
                    d->or_cur[m] = d->or_start[m]
                                   + (d->or_tgt[m] - d->or_start[m]) * t;
            }
            i += mm;
        } else {
            for (c = 0; c < no; c++)
                for (j = i; j < i1; j++) {
                    double acc = 0.0;
                    int k;
                    for (k = 0; k < nc; k++)
                        acc = acc + o->pcm[k][j] * d->or_cur[c * nc + k];
                    tmp[c][j] = acc;
                }
            break;
        }
    }
}

/* decoder.ObjectRenderer.process for one frame (core decoding) */
static void obj_render_frame(ac4dec *d, const outf_t *o,
                             double (*tmp)[FRAME])
{
    int n = o->n, pos = 0, k, i, ord[AC4_OAMD_NB], ne = 0;
    if (!d->or_init)
        or_event(d, NULL, 0);
    if (o->has_obj && d->ajoc_known && !d->ajoc_info.static_dmx) {
        /* the block updates in order of their (clamped) samples, stable */
        int sm[AC4_OAMD_NB];
        ne = o->n_dev;
        for (k = 0; k < ne; k++) {
            int s_ = o->dev[k][0] < 0 ? 0 : o->dev[k][0];
            sm[k] = s_ > n ? n : s_;
            ord[k] = k;
        }
        for (k = 1; k < ne; k++)
            for (i = k; i > 0 && sm[ord[i - 1]] > sm[ord[i]]; i--) {
                int t = ord[i];
                ord[i] = ord[i - 1];
                ord[i - 1] = t;
            }
        for (k = 0; k < ne; k++) {
            int at = sm[ord[k]];
            if (at > pos) {
                or_apply(d, o, pos, at, tmp);
                pos = at;
            }
            or_event(d, o->dblk[ord[k]], o->dev[ord[k]][1]);
        }
    }
    if (n > pos)
        or_apply(d, o, pos, n, tmp);
}

static void render_update(ac4dec *d)
{
    int ids[NOUT];
    if (d->element == AC4DEC_ELEMENT_AJOC)
        return;                 /* obj_render_frame / ajoc.c */
    if (!d->cfg.layout)
        return;
    render_ids(d, ids);
    ac4_render_matrix_dmx(ids, d->n_ch, d->cfg.layout,
                          d->cdmx_held.present ? &d->cdmx_held : NULL,
                          d->cfg.stereo_dmx, d->rmat);
    {
        /* render.is_downmix: a decoded channel the layout lacks */
        const int *lo = ac4_layout_ids(d->cfg.layout);
        int nl = ac4_layout_channels(d->cfg.layout), k, j;
        d->is_dmx = 0;
        for (k = 0; k < d->n_ch; k++) {
            int hit = ids[k] == RN_LFE;
            for (j = 0; j < nl && !hit; j++)
                hit = lo[j] == ids[k];
            if (!hit)
                d->is_dmx = 1;
        }
    }
}

int ac4dec_probe_rate(const uint8_t *data, size_t len, int *fs_index,
                      int *frame_rate_index)
{
    ac4_toc t;
    if (len > MAX_FRAME_BYTES || toc_try(data, len, &t) != 0)
        return -1;
    if (fs_index)
        *fs_index = t.fs_index;
    if (frame_rate_index)
        *frame_rate_index = t.frame_rate_index;
    return 0;
}

int ac4dec_rate_info(int fs_index, int frame_rate_index, int *frame_len,
                     int *up, int *down, int *out_rate, double *fps)
{
    static const double FPS[14] = {
        24000.0 / 1001.0, 24.0, 25.0, 30000.0 / 1001.0, 30.0,
        48000.0 / 1001.0, 48.0, 50.0, 60000.0 / 1001.0, 60.0, 100.0,
        120000.0 / 1001.0, 120.0, 48000.0 / 2048.0 };
    int fl = ac4_frame_len_base(fs_index, frame_rate_index), u, dn;
    if (!fl)
        return -1;
    ac4_rate_ratio(fs_index, frame_rate_index, &u, &dn);
    if (frame_len)
        *frame_len = fl;
    if (up)
        *up = u;
    if (down)
        *down = dn;
    if (out_rate)
        *out_rate = fs_index == 0 ? 44100 : 48000;
    if (fps)
        *fps = FPS[frame_rate_index];
    return 0;
}

static const char *const CONTENT[8] = {
    "complete main", "music and effects", "visually impaired",
    "hearing impaired", "dialogue", "commentary", "emergency", "voice over"
};

int ac4dec_presentations(const uint8_t *data, size_t len,
                         ac4dec_presentation *out, int max)
{
    ac4_toc t;
    ac4dec_config c;
    int pi, k, n = 0, chosen;
    memset(&c, 0, sizeof(c));
    if (len > MAX_FRAME_BYTES || toc_try(data, len, &t) != 0)
        return -1;
    chosen = choose_pres(&c, &t);
    for (pi = 0; pi < t.n_pres; pi++) {
        const ac4_toc_pres *p = &t.pres[pi];
        const ac4_toc_sub *s = ac4_toc_audio_substream(&t, pi);
        ac4dec_presentation *o;
        if (n >= max || !out) {
            n++;
            continue;
        }
        o = &out[n++];
        memset(o, 0, sizeof(*o));
        o->index = pi;
        o->presentation_id = p->presentation_id;
        o->config = p->config;
        o->enabled = p->enabled;
        o->md_compat = p->md_compat;
        o->version = p->version;
        o->channel_mode = s ? (int)s->channel_mode : -1;
        o->decodable = s != NULL;
        o->is_default = pi == chosen;
        o->content = "";
        for (k = 0; k < p->n_groups; k++) {
            long gi = p->groups[k];
            const ac4_toc_group *g;
            int j;
            if (gi < 0 || gi >= t.n_groups)
                continue;
            g = &t.groups[gi];
            if (!o->language[0] && g->language[0])
                memcpy(o->language, g->language, sizeof(o->language));
            if (!o->content[0] && g->content_classifier >= 0)
                o->content = CONTENT[g->content_classifier & 7];
            for (j = 0; j < g->n_subs; j++)
                if (g->subs[j].kind != 0)
                    o->object_coded = 1;
        }
    }
    return n;
}

double ac4dec_ref_peak(ac4dec_element e)
{
    return e == AC4DEC_ELEMENT_PAIR || e == AC4DEC_ELEMENT_MONO ? 2.0e5
                                                                : 9.0e5;
}

/* elements.ice_channels: the output slots of an immersive channel mode
   (ch_mode 11 / 12) given b_4_back_channels_present and
   top_channels_present: L R C [LFE] Ls Rs [Lb Rb] [Tfl Tfr] [Tbl Tbr] */
static int ice_slots(int ch_mode, int back4, int top, int *slot)
{
    int n = 0;
    slot[n++] = CH_L;
    slot[n++] = CH_R;
    slot[n++] = CH_C;
    if (ch_mode == 12 || ch_mode == 14)
        slot[n++] = CH_LFE;
    slot[n++] = CH_LS;
    slot[n++] = CH_RS;
    if (back4) {
        slot[n++] = CH_LB;
        slot[n++] = CH_RB;
    }
    if (top == 1 || top == 3) {
        slot[n++] = CH_X1;
        slot[n++] = CH_X2;
    }
    if (top == 2 || top == 3) {
        slot[n++] = CH_TBL;
        slot[n++] = CH_TBR;
    }
    return n;
}

/* the element's output channels (slots, in output order) -> count */
static int element_slots(int element, int ch_mode, int *slot)
{
    static const int S51[6] = { CH_L, CH_R, CH_C, CH_LFE, CH_LS, CH_RS };
    static const int S50[5] = { CH_L, CH_R, CH_C, CH_LS, CH_RS };
    static const int S71[8] = { CH_L, CH_R, CH_C, CH_LFE, CH_LS, CH_RS,
                                CH_X1, CH_X2 };
    static const int S70[7] = { CH_L, CH_R, CH_C, CH_LS, CH_RS, CH_X1,
                                CH_X2 };
    const int *src = S51;
    int n = 6, i;
    switch (element) {
    case AC4DEC_ELEMENT_PAIR: n = 2; break;
    case AC4DEC_ELEMENT_MONO: src = S51 + 2; n = 1; break;
    case AC4DEC_ELEMENT_3_0: n = 3; break;
    case AC4DEC_ELEMENT_5_0: src = S50; n = 5; break;
    case AC4DEC_ELEMENT_7X:
        if (ch_mode % 2 == 0) {
            src = S71;
            n = 8;
        } else {
            src = S70;
            n = 7;
        }
        break;
    case AC4DEC_ELEMENT_ICE:
        return ice_slots(ch_mode, 1, 3, slot);
    default: break;
    }
    for (i = 0; i < n; i++)
        slot[i] = src[i];
    return n;
}

/* the layout an AJOC decoder renders to (decoder.py: 5.1 unless one is
   asked for; channels 2: 2.0) */
/* a layout's channels as decoder slots, by name (for the leveler's
   weights and the mixer) */
static int lay_slots(int lay, int *slot)
{
    const int *ids = ac4_layout_ids(lay);
    int i, n = ac4_layout_channels(lay);
    for (i = 0; i < n; i++) {
        int sl = CH_X1;
        switch (ids[i]) {
        case RN_L: sl = CH_L; break;
        case RN_R: sl = CH_R; break;
        case RN_C: sl = CH_C; break;
        case RN_LFE: sl = CH_LFE; break;
        case RN_LS: sl = CH_LS; break;
        case RN_RS: sl = CH_RS; break;
        case RN_LB: sl = CH_LB; break;
        case RN_RB: sl = CH_RB; break;
        case RN_TBL: sl = CH_TBL; break;
        case RN_TBR: sl = CH_TBR; break;
        case RN_TFR: case RN_TSR: case RN_RW: sl = CH_X2; break;
        default: break;
        }
        slot[i] = sl;
    }
    return n;
}

static int ajoc_layout(const ac4dec *d)
{
    if (d->cfg.layout)
        return d->cfg.layout;
    return d->cfg.channels == 2 ? AC4DEC_LAYOUT_2_0 : AC4DEC_LAYOUT_5_1;
}

static void set_element(ac4dec *d, int element)
{
    double ref;
    int slot[NOUT], n, i;
    d->element = element;
    if (element == AC4DEC_ELEMENT_7X && !d->ch_mode)
        d->ch_mode = d->cfg.ch_mode ? d->cfg.ch_mode : 6;
    if (element == AC4DEC_ELEMENT_ICE && !d->ch_mode)
        d->ch_mode = d->cfg.ch_mode ? d->cfg.ch_mode : 12;
    if (element == AC4DEC_ELEMENT_AJOC) {
        /* elements.ajoc_slots: the downmix tracks ([LFE] then the
           var_channel_element's slots, or the 5_X element's) */
        static const int SL[8][7] = {
            { 0 }, { CH_C }, { CH_L, CH_R }, { CH_L, CH_R, CH_C },
            { CH_L, CH_R, CH_LS, CH_RS }, { CH_L, CH_R, CH_LS, CH_RS, CH_C },
            { CH_L, CH_R, CH_LS, CH_RS, CH_X1, CH_X2 },
            { CH_L, CH_R, CH_LS, CH_RS, CH_X1, CH_X2, CH_C } };
        const ac4_toc_sub *ai = &d->ajoc_info;
        int nd = d->ajoc_known ? ai->n_dmx : 5;
        n = 0;
        if (d->ajoc_known && ai->kind == 2) {
            /* audio_data_objs: the LFE, then the element's tracks */
            static const int OB[6][5] = {
                { 0 }, { CH_C }, { CH_L, CH_R }, { CH_L, CH_R, CH_C },
                { 0 }, { CH_L, CH_R, CH_C, CH_LS, CH_RS } };
            static const int ON[6] = { 0, 1, 2, 3, 0, 5 };
            int no = ai->n_objects >= 0 && ai->n_objects < 6
                     ? ON[ai->n_objects] : 0;
            if (ai->lfe)
                slot[n++] = CH_LFE;
            for (i = 0; i < no; i++)
                slot[n++] = OB[ai->n_objects][i];
        } else if (!d->ajoc_known || ai->static_dmx) {
            slot[n++] = CH_L;
            slot[n++] = CH_R;
            slot[n++] = CH_C;
            if (!d->ajoc_known || ai->lfe)
                slot[n++] = CH_LFE;
            slot[n++] = CH_LS;
            slot[n++] = CH_RS;
        } else {
            if (ai->lfe)
                slot[n++] = CH_LFE;
            for (i = 0; i < nd && i < 7; i++)
                slot[n++] = SL[nd < 8 ? nd : 7][i];
        }
        d->el_mask = 0;
        for (i = 0; i < n; i++)
            d->el_mask |= 1 << slot[i];
    } else if (element == AC4DEC_ELEMENT_ICE) {
        /* the coded tracks (L R C [LFE] Ls Rs Tfl Tfr); the outputs by
           what the substream signals present */
        int pr = d->cfg.present;
        if (pr & 16) {
            d->ice_back4 = pr & 1;
            d->ice_top = (pr >> 2) & 3;
        } else if (!d->ice_known) {
            d->ice_back4 = 1;
            d->ice_top = 3;
        }
        n = ice_slots(d->ch_mode, d->ice_back4, d->ice_top, slot);
        d->el_mask = (1 << CH_L) | (1 << CH_R) | (1 << CH_C) | (1 << CH_LS)
                     | (1 << CH_RS) | (1 << CH_X1) | (1 << CH_X2);
        if (d->ch_mode == 12 || d->ch_mode == 14)
            d->el_mask |= 1 << CH_LFE;
    } else {
        n = element_slots(element, d->ch_mode, slot);
        d->el_mask = 0;
        for (i = 0; i < n; i++)
            d->el_mask |= 1 << slot[i];
    }
    if (d->cfg.channels == 2 && n > 2 && (d->el_mask & 3) == 3
        && element != AC4DEC_ELEMENT_AJOC) {
        n = 2;
        slot[0] = CH_L;
        slot[1] = CH_R;
    }
    d->n_ch = n;
    for (i = 0; i < NOUT; i++)
        d->out_ch[i] = i < n ? slot[i] : i;
    d->n_lay = d->cfg.layout ? ac4_layout_channels(d->cfg.layout) : n;
    if (element == AC4DEC_ELEMENT_AJOC)
        d->n_lay = ac4_layout_channels(ajoc_layout(d));
    d->ajoc_full = 0;
    if (element == AC4DEC_ELEMENT_AJOC && d->ajoc_known && !d->cfg.ajoc_core
        && d->ajoc_info.kind == 1) {
        /* full decoding (decoder.AjocFull): the decoded channels are the
           layout's, rendered from the upmix objects in the QMF domain;
           their slots (for the leveler's weights) by name */
        const int *ids = ac4_layout_ids(ajoc_layout(d));
        d->ajoc_full = 1;
        if (!d->ajx)
            d->ajx = ac4_ajoc_full_new(&d->ajoc_info, ajoc_layout(d));
        d->n_ch = d->n_lay;
        (void)ids;
        lay_slots(ajoc_layout(d), d->out_ch);
    }
    if (element == AC4DEC_ELEMENT_AJOC && !d->ajf) {
        d->ajf_cap = d->rec_cap + 2;
        d->ajf = (ac4_ajoc_frame *)calloc((size_t)d->ajf_cap,
                                          sizeof(ac4_ajoc_frame));
        if (!d->ajf)
            d->ajf_cap = 0;
    }
    render_update(d);
    /* A-SPX block groups -> channel pairs (Table 213) */
    d->pair_ch[0][0] = CH_L;
    d->pair_ch[0][1] = CH_R;
    d->pair_ch[1][0] = CH_LS;
    d->pair_ch[1][1] = CH_RS;
    d->pair_ch[2][0] = d->pair_ch[2][1] = CH_C;
    d->pair_ch[3][0] = CH_X1;
    d->pair_ch[3][1] = CH_X2;
    if (element == AC4DEC_ELEMENT_7X && (d->ch_mode == 7 || d->ch_mode == 8)) {
        d->pair_ch[1][0] = CH_X1;         /* 5/2/0: (Lw, Rw) second */
        d->pair_ch[1][1] = CH_X2;
        d->pair_ch[3][0] = CH_LS;
        d->pair_ch[3][1] = CH_RS;
    }
    ref = ac4dec_ref_peak((ac4dec_element)element);
    d->guard.floor = G_FLOOR_FRAC * ref;
    d->guard.init = G_INIT_FRAC * ref;
}

/* adopt the stream's rate: frame length, num_aspx_timeslots, the
   filterbank and the pairs' time grid (rates.py; Decoder._adopt_frame) */
static void set_rate(ac4dec *d, int fs, int fri)
{
    int fl = ac4_frame_len_base(fs, fri), k;
    d->rate_known = 1;
    d->fs = fs;
    d->fri = fri;
    d->books.fl = fl;
    d->nats = fl >= 1536 ? fl / 128 : fl / 64;
    if (fl != d->fl) {
        long long j;
        d->fl = fl;
        for (k = 0; k < NCH; k++) {
            memset(&d->fb[k], 0, sizeof(d->fb[k]));
            d->fb[k].n_full = fl;
        }
        for (k = 0; k < NPAIR; k++) {
            ac4_aspx_pair_init(&d->pair[k], fl);
            d->pair_active[k] = 0;
        }
        if (d->ajx)
            ac4_ajoc_full_reset(d->ajx);
        /* frames concealed before the rate was known and not synthesised
           yet take the stream's frame length (one zero window) */
        for (j = d->rec_first; j < d->rec_first + d->rec_n; j++) {
            rec_t *r = &d->recs[(int)(j % d->rec_cap)];
            if (r->code != AC4E_NONE)
                for (k = 0; k < NCH; k++) {
                    r->nwin[k] = 1;
                    r->wlen[k][0] = fl;
                    memset(r->win + k * FRAME, 0, sizeof(double) * FRAME);
                }
        }
    }
}

/* which of the six channels a frame synthesises (ASPX_ACPL_3 in 5.1:
   the core pair and the LFE; C, Ls, Rs come from the A-CPL upmix) */
static int chan_mask(const ac4dec *d, int acpl3)
{
    if (d->element == AC4DEC_ELEMENT_AJOC)
        return d->el_mask;
    if (d->element == AC4DEC_ELEMENT_PAIR || d->n_ch == 2)
        return (1 << CH_L) | (1 << CH_R);
    if (acpl3)
        return (1 << CH_L) | (1 << CH_R) | (1 << CH_LFE);
    return d->el_mask;
}

/* the channels of the mask the element does not code (ASPX_ACPL_2: R of
   a pair; ASPX_ACPL_1 / _2 in 5_X: Ls, Rs): one silent long block */
static int silent_mask(const ac4_element *e, int mask)
{
    int m = 0;
    if (e->acpl1 == AC4_ACPL1_PAIR && e->r_silent)
        m = 1 << CH_R;
    else if (e->acpl1 == AC4_ACPL1_5X && !e->n_side)
        m = (1 << CH_LS) | (1 << CH_RS);   /* ASPX_ACPL_2 */
    return m & mask;
}

static const ac4_chan *chan_src(const ac4_element *e, int k)
{
    return &e->ch[k];
}

/* the tracks to dequantise: the channels of the mask, and every track a
   processing step (or an ASPX_ACPL_1 side step) mixes into one of them */
static int need_mask(const ac4_element *e, int mask)
{
    int i, j, m = mask;
    for (i = 0; i < e->n_steps; i++) {
        int km = 0;
        for (j = 0; j < e->steps[i].nk; j++)
            km |= 1 << e->steps[i].key[j];
        if (km & mask)
            m |= km;
    }
    for (i = 0; i < e->n_side; i++) {
        int km = (1 << e->side[i].ka) | (1 << e->side[i].kb);
        if (km & m)
            m |= km;
    }
    return m;
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
    d->rate_known = 0;
    d->rs_fixed = 0;
    memset(&d->cdmx_held, 0, sizeof(d->cdmx_held));
    memset(&d->obj, 0, sizeof(d->obj));
    d->or_init = 0;
    d->fs = d->fri = -1;
    d->fl = AC4_DEFAULT_FRAME;
    d->books.fl = AC4_DEFAULT_FRAME;
    d->nats = AC4_DEFAULT_FRAME / 128;
    memset(d->has_syn_tab, 0, sizeof(d->has_syn_tab));
    d->acpl3 = 0;
    d->acpl1 = 0;
    d->has_pcfg1 = 0;
    memset(d->aq1, 0, sizeof(d->aq1));
    memset(d->aq1n, 0, sizeof(d->aq1n));
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
    for (i = 0; i < NCH; i++)
        d->fb[i].n_full = AC4_DEFAULT_FRAME;
    d->n_synth = 0;
    memset(d->core_prev, 0, sizeof(d->core_prev));
    memset(d->core_tail, 0, sizeof(d->core_tail));
    for (i = 0; i < NPAIR; i++) {
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
    for (i = 0; i < NOUT; i++)
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
    if (c.channels < 0 || c.channels > NOUT)
        return NULL;
    if ((int)c.element < AC4DEC_ELEMENT_AUTO
        || (int)c.element > AC4DEC_ELEMENT_AJOC)
        return NULL;
    if (c.ch_mode != 0 && (c.ch_mode < 5 || c.ch_mode > 14))
        return NULL;
    if (c.layout < 0 || c.layout > AC4DEC_LAYOUT_7_1_4 || c.stereo_dmx < 0
        || c.stereo_dmx > 3)
        return NULL;
    if (c.element != AC4DEC_ELEMENT_AUTO && c.element != AC4DEC_ELEMENT_AJOC
        && c.channels) {
        int slot[NOUT], n;
        if (c.element == AC4DEC_ELEMENT_ICE)
            n = (c.present & 16)
                ? ice_slots(c.ch_mode ? c.ch_mode : 12, c.present & 1,
                            (c.present >> 2) & 3, slot)
                : ice_slots(c.ch_mode ? c.ch_mode : 12, 1, 3, slot);
        else
            n = element_slots(c.element, c.ch_mode ? c.ch_mode : 6, slot);
        if (c.channels != n && !(c.channels == 2 && n > 2
                                 && c.element != AC4DEC_ELEMENT_MONO))
            return NULL;
    }
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
    if (ac4_huff_build(&d->drc_hcb, &AC4T_MISC_BOOKS[0]) != 0) {
        ac4dec_close(d);
        return NULL;
    }
    for (i = 0; i < 4; i++)
        if (ac4_huff_build(&d->de_hcb[i], &AC4T_MISC_BOOKS[1 + i]) != 0) {
            ac4dec_close(d);
            return NULL;
        }
    for (i = 0; i < AC4T_N_AJOC_BOOKS; i++)
        if (ac4_huff_build(&d->ajoc_huff[i], &AC4T_AJOC_BOOKS[i]) != 0) {
            ac4dec_close(d);
            return NULL;
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
        d->recs[i].win = (double *)calloc(NCH * FRAME, sizeof(double));
        if (!d->recs[i].win) {
            ac4dec_close(d);
            return NULL;
        }
    }
    if (c.element != AC4DEC_ELEMENT_AUTO)
        set_element(d, c.element);
    else
        d->n_ch = c.channels ? c.channels : 2;
    if (!d->element)
        d->n_lay = c.layout ? ac4_layout_channels(c.layout) : d->n_ch;
    stream_reset(d);
    return d;
}

void ac4dec_close(ac4dec *d)
{
    int i;
    if (!d)
        return;
    for (i = 0; i < d->n_kid; i++)
        ac4dec_close(d->kid[i]);
    for (i = 0; i < AC4T_N_BOOKS; i++)
        ac4_huff_free(&d->huff[i]);
    ac4_huff_free(&d->drc_hcb);
    for (i = 0; i < 4; i++)
        ac4_huff_free(&d->de_hcb[i]);
    for (i = 0; i < AC4T_N_AJOC_BOOKS; i++)
        ac4_huff_free(&d->ajoc_huff[i]);
    free(d->lev);
    free(d->ajf);
    ac4_ajoc_full_free(d->ajx);
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
    s->frame_len = d->rate_known ? d->fl : 0;
    s->out_rate = d->rate_known ? (d->fs == 0 ? 44100 : 48000) : 0;
    s->aspx = (d->path_on[0] || d->path_on[1] || d->path_on[2])
              && d->has_cfg;
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

/* companding_control, per channel (stream.py): L R C Ls Rs for 5_X codec
   mode 1; L R C for ASPX_ACPL_1 / _2 and 3.0; L R for a pair and
   ASPX_ACPL_3; the "m" track of a pair in ASPX_ACPL_1 / _2; C for mono */
static void set_comp(ac4_aspx_frame *f, int g, const ac4_element *e,
                     int acpl1)
{
    int ch;
    for (ch = 0; ch < 2; ch++) {
        ac4_aspx_chd *cd = &f->g.c[ch];
        int i = g == 0 ? (ch == 0 || acpl1 == AC4_ACPL1_PAIR ? 0 : 1)
                : g == 1 ? 3 + ch : e->comp_c;
        if (e->comp_var)                /* A-JOC var_channel_element */
            i = g == 0 ? ch : g == 1 ? 2 + ch : e->comp_c;
        cd->comp_set = 1;
        cd->comp_sync = e->comp_sync;
        cd->comp_avg = e->comp_avg;
        cd->comp_on = e->comp_sync ? e->comp_on[0]
                      : (i < e->comp_n ? e->comp_on[i] : 0);
    }
}

/* every pair group of a record: A-SPX, A-CPL and 1ch A-CPL kinds */
static void af_kinds(rec_t *r, int kind)
{
    int p;
    for (p = 0; p < NPAIR; p++)
        r->af[p].kind = r->af[p].acpl.kind = r->af[p].acpl1_kind = kind;
}

/* a concealed frame: one zero window per channel, A-SPX muted */
static void rec_conceal(ac4dec *d, rec_t *r, int code)
{
    int k;
    r->code = code;
    r->mask = chan_mask(d, d->acpl3);
    for (k = 0; k < NCH; k++) {
        r->nwin[k] = 1;
        r->wlen[k][0] = d->fl;
        memset(r->win + k * FRAME, 0, sizeof(double) * (size_t)d->fl);
    }
    af_kinds(r, AC4_GRP_CONCEALED);
    r->has_obj = 0;
    for (k = 0; k < NPAIR; k++)
        r->af[k].has_tab = 0;
    r->de.kind = AC4_DE_HOLD;
    r->seq_jump = 0;
}

/* ------------------------------------------------------------ late */
static void add_pending(ac4dec *d, long long index, const uint8_t *sub,
                        size_t sub_len, const ac4_element *e, int ifr)
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
    p->n_acpl1 = e->n_acpl1;
    p->n_blocks = e->n_blocks;
    memcpy(p->blk_grp, e->blk_grp, sizeof(p->blk_grp));
    memcpy(p->blk_two, e->blk_two, sizeof(p->blk_two));
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
    {
        int i;
        for (i = 0; i < pd->n_blocks; i++)
            ac4_aspx_data(&b, &d->abk, &d->acfg, d->nats, pd->ifr,
                          pd->blk_two[i], st, pd->blk_grp[i], NULL, NULL);
    }
    if (pd->acpl3 || pd->n_acpl1) {
        int i;
        if (pd->acpl3 ? !d->has_pcfg : !d->has_pcfg1)
            return -1;
        b.eof_code = AC4E_ACPL_PARSE;
        if (pd->acpl3)
            ac4_acpl_data_2ch(&b, &d->abk, &d->pcfg, NULL, NULL);
        for (i = 0; i < pd->n_acpl1; i++)
            ac4_acpl_data_1ch(&b, &d->abk, &d->pcfg1, NULL, NULL, NULL);
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
    /* the frames are checked at the aspx_xover_subband_offsets the
       configuration's I-frame brought (Ac4Stream.xover0) */
    memcpy(st.xover, d->fstate.xover, sizeof(st.xover));
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
        if (!ok) {
            memset(&st, 0, sizeof(st));
            memcpy(st.xover, d->fstate.xover, sizeof(st.xover));
        }
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
    int aq10[4][2][AC4_ACPL_NB], aq1n0[4][2];
    int code = AC4E_NONE, k, ifr = 0, has_aspx, acpl3, acpl1, complete;
    int silent, need;
    size_t o, sub_len = 0;
    const uint8_t *sub = NULL;
    toc_shape sh;
    ac4_element *e = &d->el;
    ac4_aspx_tab tab_new, gtab[NPAIR];
    const ac4_aspx_cfg *volatile cfg = NULL;
    size_t end;
    if (d->element == AC4DEC_ELEMENT_AJOC)
        d->obj_bak = d->obj;

    memset(&sh, 0, sizeof(sh));
    memcpy(aq10, d->aq1, sizeof(aq10));
    memcpy(aq1n0, d->aq1n, sizeof(aq1n0));
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
    /* the rate (rates.py): every frame_rate_index of Table 83 and index
       13 of Table 84; not frame_rate_factor / frame_rate_fraction > 1 */
    if (!ac4_frame_len_base(t.fs_index, t.frame_rate_index)
        || t.rate_multi) {
        code = AC4E_TOC_UNSUPPORTED;
        goto fail;
    }
    if (!d->rate_known) {
        /* the stream's rate is the first decodable frame's, when the next
           frame (if any) agrees */
        if (has_next) {
            ac4_toc tn;
            if (toc_try(nx, nx_len, &tn) != 0
                || tn.fs_index != t.fs_index
                || tn.frame_rate_index != t.frame_rate_index) {
                code = AC4E_TOC_RATE;
                goto fail;
            }
        }
        set_rate(d, t.fs_index, t.frame_rate_index);
    } else if (t.fs_index != d->fs || t.frame_rate_index != d->fri) {
        /* a change of rate is not followed */
        code = AC4E_TOC_RATE;
        goto fail;
    }
    /* check_toc */
    if (t.sequence_counter > 1020) {
        code = AC4E_TOC_SEQ_RANGE;
        goto fail;
    }
    if (t.n_sizes < (t.bitstream_version >= 2 ? 2 : 1)
        || (long)t.n_sizes != t.n_substreams
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
    /* the audio substream of the chosen presentation (TS 103 190-2
       4.8.2), located by its substream_index; ITS b_audio_ndot says
       whether it is an I-frame */
    {
        const ac4_toc_sub *s = chosen_sub(&d->cfg, &t);
        long i;
        if (!s) {
            code = AC4E_TOC_PRESENTATION;
            goto fail;
        }
        o = t.toc_bytes + (size_t)t.payload_base;
        for (i = 0; i < s->index; i++)
            o += (size_t)t.sizes[i];
        if (o > len)
            o = len;
        sub_len = (size_t)t.sizes[s->index];
        if (sub_len > len - o)
            sub_len = len - o;
        sub = data + o;
        ifr = s->ndot;
        if (d->element == AC4DEC_ELEMENT_AJOC && !d->ajoc_known) {
            /* the substream's info, from the first frame that has it
               (decoder.probe_ajoc_info); the decoded slots follow it */
            if (s->kind != 1 && s->kind != 2) {
                code = AC4E_ELEMENT_PARSE;
                goto fail;
            }
            d->ajoc_info = *s;
            d->ajoc_known = 1;
            set_element(d, d->element);
        }
        if (s->sf_multiplier >= 0) {     /* 96 / 192 kHz substream */
            code = AC4E_TOC_UNSUPPORTED;
            goto fail;
        }
    }
    b.d = sub;
    b.nbits = sub_len * 8;
    b.p = 0;
    b.jb = &jb;
    b.eof_code = AC4E_ELEMENT_PARSE;
    code = setjmp(jb);
    if (code != 0)
        goto fail;
    e->objs_lfe = NULL;
    switch (d->element) {
    case AC4DEC_ELEMENT_PAIR: ac4_parse_pair(&b, &d->books, ifr, e); break;
    case AC4DEC_ELEMENT_MONO: ac4_parse_mono(&b, &d->books, ifr, e); break;
    case AC4DEC_ELEMENT_3_0: ac4_parse_3_0(&b, &d->books, ifr, e); break;
    case AC4DEC_ELEMENT_5_0:
        ac4_parse_5x_lfe(&b, &d->books, ifr, 0, e);
        break;
    case AC4DEC_ELEMENT_7X:
        ac4_parse_7x(&b, &d->books, ifr, d->ch_mode, e);
        break;
    case AC4DEC_ELEMENT_ICE:
        ac4_parse_ice(&b, &d->books, ifr, d->ch_mode, e);
        break;
    case AC4DEC_ELEMENT_AJOC:
        if (!d->ajoc_known)
            ac4_fail(&b, AC4E_ELEMENT_PARSE);
        if (d->ajoc_info.kind == 2) {
            ac4_parse_objs(&b, &d->books, ifr, d->ajoc_info.n_objects,
                           d->ajoc_info.lfe, e);
            break;
        }
        ac4_parse_ajoc(&b, &d->books, ifr, d->ajoc_info.lfe,
                       d->ajoc_info.static_dmx, d->ajoc_info.n_dmx,
                       d->ajoc_info.n_umx, e);
        break;
    default: ac4_parse_5x(&b, &d->books, ifr, e); break;
    }
    if (changed == AC4E_NONE && d->mode >= 0 && e->mode != d->mode)
        changed = AC4E_ELEMENT_MODE;
    if (e->has_aspx_cfg) {
        if (ac4_aspx_tables(&e->aspx_cfg, 0, &tab_new) != 0)
            ac4_fail(&b, AC4E_ASPX_CONFIG);
        cfg = &e->aspx_cfg;
    } else if (d->has_cfg) {
        cfg = &d->acfg;
    }
    acpl3 = e->acpl3;
    acpl1 = e->acpl1;
    has_aspx = e->n_blocks > 0;
    end = e->bits_end;
    af_kinds(r, AC4_GRP_NONE);
    if (cfg && has_aspx) {
        const ac4_acpl_cfg *pc = e->has_acpl_cfg ? &e->acpl_cfg
                                 : (d->has_pcfg ? &d->pcfg : NULL);
        const ac4_acpl1_cfg *pc1 = e->has_acpl_cfg ? &e->acpl1_cfg
                                   : (d->has_pcfg1 ? &d->pcfg1 : NULL);
        int i;
        b.p = e->bits_end;
        b.eof_code = AC4E_ASPX_PARSE;
        for (i = 0; i < e->n_blocks; i++) {
            int g = e->blk_grp[i];
            ac4_aspx_data(&b, &d->abk, cfg, d->nats, ifr, e->blk_two[i],
                          &d->fstate, g, &r->af[g].g, &gtab[g]);
            /* a one-channel block (the centre; the "m" track of a pair in
               ASPX_ACPL_1 / _2) is rendered as the same channel twice
               (decoder.py): with balance 0 each half decodes alone */
            if (!e->blk_two[i])
                r->af[g].g.c[1] = r->af[g].g.c[0];
            r->af[g].kind = AC4_GRP_DATA;
            /* companding_control, per channel (stream.py): L R C Ls Rs
               for 5_X codec mode 1; L R C for ASPX_ACPL_1 / _2; L R for
               a pair and ASPX_ACPL_3; the "m" track of a pair in
               ASPX_ACPL_1 / _2 */
            if (e->comp_present)
                set_comp(&r->af[g], g, e, acpl1);
        }
        end = b.p;
        if (acpl3 && pc) {
            b.eof_code = AC4E_ACPL_PARSE;
            /* parsed, and differentially decoded (Pseudocode 121) */
            ac4_acpl_data_2ch(&b, &d->abk, pc, &d->aq, &r->af[0].acpl);
            end = b.p;
        }
        if (acpl1 && pc1) {
            b.eof_code = AC4E_ACPL_PARSE;
            for (i = 0; i < e->n_acpl1; i++) {
                /* the immersive element: modules 0, 1 to the Ls / Rs
                   pair, 2, 3 to the Tfl / Tfr pair */
                int pg = acpl1 == AC4_ACPL1_ICE ? (i < 2 ? 1 : 3) : 0;
                ac4_acpl_data_1ch(&b, &d->abk, pc1, d->aq1[i], d->aq1n[i],
                                  &r->af[pg].acpl1[i & 1]);
                r->af[pg].acpl1_kind = AC4_GRP_DATA;
            }
            end = b.p;
        }
    }
    complete = (cfg || !has_aspx)
               && (!(acpl3 || acpl1) || e->has_acpl_cfg
                   || (acpl3 ? d->has_pcfg : d->has_pcfg1));
    r->has_obj = 0;
    if (e->tail == 2) {
        /* the objects' metadata: the group's oamd_substream
           (elements.objs_tail) */
        const ac4_toc_sub *ss[AC4_MIX_MAX];
        long gg[AC4_MIX_MAX], oi;
        int ns = ac4_toc_audio_substreams(&t, choose_pres(&d->cfg, &t), ss,
                                          gg, AC4_MIX_MAX);
        int k = d->cfg.substream > 0 ? d->cfg.substream : 0;
        const ac4_toc_group *g;
        size_t oo;
        ac4_bits ob;
        long i;
        if (k >= ns || !d->ajf_cap)
            ac4_fail(&b, AC4E_ELEMENT_PARSE);
        g = &t.groups[gg[k]];
        oi = g->has_oamd ? g->oamd_index : -1;
        if (oi < 0 || oi >= t.n_sizes || oi >= AC4_MAX_SIZES)
            ac4_fail(&b, AC4E_ELEMENT_PARSE);
        oo = t.toc_bytes + (size_t)t.payload_base;
        for (i = 0; i < oi; i++)
            oo += (size_t)t.sizes[i];
        if (oo > len)
            oo = len;
        ob.d = data + oo;
        ob.nbits = ((size_t)t.sizes[oi] > len - oo ? len - oo
                    : (size_t)t.sizes[oi]) * 8;
        ob.p = 0;
        ob.jb = &jb;
        ob.eof_code = AC4E_ELEMENT_PARSE;
        ac4_objs_tail(&ob, g, ss[k], g->oamd_ndot,
                      t.pres[choose_pres(&d->cfg, &t)].b_alternative,
                      &d->obj, &d->ajf[(int)(r->index % d->ajf_cap)]);
        r->has_obj = 1;
    } else if (e->tail && (cfg || !has_aspx)) {
        /* the rest of audio_data_ajoc (elements.ajoc_tail) */
        b.p = end;
        b.eof_code = AC4E_ELEMENT_PARSE;
        if (!d->ajf_cap)
            ac4_fail(&b, AC4E_INTERNAL);
        ac4_ajoc_tail(&b, d->ajoc_huff, &d->ajoc_info, ifr, &d->obj,
                      &d->ajf[(int)(r->index % d->ajf_cap)]);
        end = b.p;
        r->has_obj = 1;
    } else if (e->tail == 1) {
        complete = 0;
    }
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
    if (!cfg && has_aspx && !e->tail)
        add_pending(d, r->index, sub, sub_len, e, ifr);
    /* dequantise (with the level check) */
    r->mask = chan_mask(d, acpl3);
    silent = silent_mask(e, r->mask);
    need = need_mask(e, r->mask) & ~silent;
    for (k = 0; k < NCH; k++)
        if ((need & (1 << k)) && e->has_ch[k])
            d->spec_n[k] = ac4_packed_spectrum(chan_src(e, k), d->spec[k],
                                               &jb);
    /* spectral noise fill: one generator per frame, reset from
       sequence_counter, through the element's sf_data blocks in bitstream
       order (e->order) */
    {
        int i;
        ac4_snf_rng rng;
        ac4_snf_reset(&rng, t.sequence_counter);
        for (i = 0; i < e->n_order; i++) {
            k = e->order[i];
            if ((need & (1 << k)) && e->has_ch[k])
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
    if (e->has_acpl_cfg && acpl3) {
        d->pcfg = e->acpl_cfg;
        d->has_pcfg = 1;
    }
    if (e->has_acpl_cfg && acpl1) {
        d->pcfg1 = e->acpl1_cfg;
        d->has_pcfg1 = 1;
    }
    if (!d->have_shape) {
        d->shape = sh;
        d->have_shape = 1;
    }
    d->mode = e->mode;
    if (acpl3)
        d->acpl3 = 1;
    if (acpl1)
        d->acpl1 = acpl1;
    /* the silent channels: zero */
    for (k = 0; k < NCH; k++)
        if (silent & (1 << k))
            d->spec_n[k] = 0;
    /* stereo processing, ungroup (ASPX_ACPL_1's side track can be
       shorter (max_sfb_side): both are zero-padded to the longer, as the
       Python) */
    if (acpl1 == AC4_ACPL1_PAIR) {
        if ((r->mask & 3) == 3 && !(silent & (1 << CH_R))) {
            int nl = d->spec_n[CH_L], nr = d->spec_n[CH_R];
            if (nl != nr) {
                int nm = nl > nr ? nl : nr;
                for (k = nl; k < nm; k++)
                    d->spec[CH_L][k] = 0.0;
                for (k = nr; k < nm; k++)
                    d->spec[CH_R][k] = 0.0;
                d->spec_n[CH_L] = d->spec_n[CH_R] = nm;
                nl = nr = nm;
            }
            ac4_unmix(d->spec[CH_L], d->spec[CH_R], nl, &e->st[0],
                      &e->fr[CH_L]);
        }
    } else {
        /* the channel data elements' processing (5.3.3); the keys carry
           the track to channel mapping (5.3.4) */
        int i, j;
        for (i = 0; i < e->n_steps; i++) {
            double *tr[5];
            int nk = e->steps[i].nk, n = 0, all = 1;
            for (j = 0; j < nk; j++) {
                k = e->steps[i].key[j];
                if (!(need & (1 << k)))
                    all = 0;
                tr[j] = d->spec[k];
                if (d->spec_n[k] > n)
                    n = d->spec_n[k];
            }
            if (!all)
                continue;
            for (j = 0; j < nk; j++) {
                int q;
                k = e->steps[i].key[j];
                for (q = d->spec_n[k]; q < n; q++)
                    d->spec[k][q] = 0.0;
                d->spec_n[k] = n;
            }
            ac4_matrix(tr, n, e->steps[i].kind, e->steps[i].matsel,
                       &e->st[e->steps[i].st0], &e->fr[e->steps[i].key[0]]);
        }
    }
    for (k = 0; k < NCH; k++) {
        const ac4_framing *fr;
        int w, sum = 0;
        if (!((r->mask | need) & (1 << k)))
            continue;
        if (silent & (1 << k)) {
            r->nwin[k] = 1;
            r->wlen[k][0] = d->fl;
            memset(r->win + k * FRAME, 0, sizeof(double) * (size_t)d->fl);
            continue;
        }
        fr = chan_src(e, k)->fr;
        ac4_ungroup(d->spec[k], d->spec_n[k], fr, r->win + k * FRAME,
                    r->wlen[k]);
        r->nwin[k] = fr->num_windows;
        for (w = 0; w < fr->num_windows; w++)
            sum += r->wlen[k][w];
        if (sum != d->fl) {         /* cannot happen; never trust it */
            code = AC4E_INTERNAL;
            goto fail;
        }
    }
    /* ASPX_ACPL_1 in 5_X (5.3.4.3.2): each side track with its A / B
       track, per window (their max_sfb differ) */
    for (k = 0; k < e->n_side; k++) {
        int ka = e->side[k].ka, kb = e->side[k].kb;
        if ((need & (1 << ka)) && (need & (1 << kb)))
            ac4_unmix_windows(r->win + ka * FRAME, r->win + kb * FRAME,
                              r->wlen[ka], r->nwin[ka], &e->st[e->side[k].st],
                              &e->fr[ka]);
    }
    for (k = 0; k < NPAIR; k++)
        if (r->af[k].kind == AC4_GRP_DATA) {
            r->af[k].has_tab = 1;
            r->af[k].tab = gtab[k];
            r->af[k].preflat = cfg->preflat;
            r->af[k].interpolation = cfg->interpolation;
        }
    if (!(r->mask & (1 << CH_LS)))
        r->af[1].kind = AC4_GRP_NONE;
    if (!(r->mask & (1 << CH_C)))
        r->af[2].kind = AC4_GRP_NONE;
    /* loudness / DRC metadata of the presentation substream (stream.py
       _pres_metadata) */
    {
        int pi = choose_pres(&d->cfg, &t);
        const ac4_toc_sub *s = chosen_sub(&d->cfg, &t);
        const ac4_toc_pres *pp = pi >= 0 && pi < t.n_pres ? &t.pres[pi]
                                                          : NULL;
        r->meta.valid = 0;
        d->v0 = pp && pp->v0;
        if (pp && s && pp->v0) {
            /* bitstream_version 0 / 1 (stream._v0_metadata) */
            if (ac4_v0_metadata(sub, sub_len,
                                e->audio_start + 8 * e->audio_size,
                                &r->meta) != 0)
                r->meta.valid = 0;
        } else if (pp && s && pp->pres_substream >= 0
            && pp->pres_substream < t.n_substreams
            && pp->pres_substream < t.n_sizes) {
            size_t po = t.toc_bytes + (size_t)t.payload_base, pl;
            long i;
            for (i = 0; i < pp->pres_substream; i++)
                po += (size_t)t.sizes[i];
            if (po > len)
                po = len;
            pl = (size_t)t.sizes[pp->pres_substream];
            if (pl > len - po)
                pl = len - po;
            int pm[5], have_pm = 0;
            {
                /* metadata.pres_modes_of: the presentation's channel
                   modes for custom_dmx_data */
                const ac4_toc_sub *ss[8];
                int ns = ac4_toc_audio_substreams(&t, pi, ss, NULL, 8), j;
                pm[0] = -1;
                pm[2] = pm[3] = 0;
                for (j = 0; j < ns; j++) {
                    const ac4_toc_sub *q = ss[j];
                    if (q->kind != 0)
                        continue;
                    if (q->channel_mode > pm[0])
                        pm[0] = (int)q->channel_mode;
                    if (q->channel_mode >= 11 && q->channel_mode <= 14) {
                        if (q->back4)
                            pm[2] = 1;
                        if (q->top == 3)
                            pm[3] = 2;
                        else if ((q->top == 1 || q->top == 2) && pm[3] < 1)
                            pm[3] = 1;
                    }
                }
                if (ns == 0 && s->kind == 0) {
                    pm[0] = (int)s->channel_mode;
                    if (pm[0] >= 11 && pm[0] <= 14) {
                        pm[2] = s->back4;
                        pm[3] = s->top == 3 ? 2 : (s->top ? 1 : 0);
                    }
                }
                /* Table 71, Pseudocode 26: 5.0.2 / 5.1.2 cores */
                pm[1] = pm[0] == 11 || pm[0] == 13 ? 5
                        : pm[0] == 12 || pm[0] == 14 ? 6 : -1;
                pm[4] = pm[0] == 4 || pm[0] == 6 || pm[0] == 8
                        || pm[0] == 10 || pm[0] == 12 || pm[0] == 14;
                have_pm = pm[0] >= 0;
            }
            if (ac4_pres_metadata(data + po, pl, pp->b_alternative,
                                  pp->b_pres_ndot, t.n_substreams,
                                  &d->drc_state, &d->drc_hcb,
                                  (int)s->channel_mode, d->fl, pp->n_sg,
                                  have_pm ? pm : NULL, &r->meta) != 0)
                r->meta.valid = 0;
        }
    }
    /* dialogue enhancement data of the audio substream's metadata()
       (stream.py _sub_metadata), read with the coded element's channel
       mode */
    r->de.kind = AC4_DE_NONE;
    if (d->cfg.de_level != 0.0 && !d->v0
        && d->element != AC4DEC_ELEMENT_AJOC) {
        static const int CM[8] = { 4, 4, 1, 0, 2, 3, 6, 12 };
        int cm = d->element == AC4DEC_ELEMENT_7X
                 || d->element == AC4DEC_ELEMENT_ICE ? d->ch_mode
                                                     : CM[d->element];
        const ac4_huff *bks[4];
        for (k = 0; k < 4; k++)
            bks[k] = &d->de_hcb[k];
        ac4_de_metadata(sub, sub_len,
                        e->audio_start + 8 * e->audio_size, cm, ifr,
                        &d->de_state, bks, &r->de);
    }
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
    if (d->element == AC4DEC_ELEMENT_AJOC)
        d->obj = d->obj_bak;
    d->aq = aq0;
    memcpy(d->aq1, aq10, sizeof(aq10));
    memcpy(d->aq1n, aq1n0, sizeof(aq1n0));
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
        ac4_toc tt;
        int ok = data && d->raw_len && d->raw_len <= MAX_FRAME_BYTES
                 && toc_try(data, d->raw_len, &tt) == 0;
        ac4dec_element el = ok ? element_of_toc(&d->cfg, &tt)
                               : AC4DEC_ELEMENT_AUTO;
        if ((el == AC4DEC_ELEMENT_7X || el == AC4DEC_ELEMENT_ICE)
            && !d->cfg.ch_mode)
            d->ch_mode = ch_mode_of_toc(&d->cfg, &tt);
        if (el == AC4DEC_ELEMENT_ICE) {
            const ac4_toc_sub *s = chosen_sub(&d->cfg, &tt);
            if (s) {
                d->ice_back4 = s->back4;
                d->ice_top = s->top;
                d->ice_known = 1;
            }
        }
        if (el == AC4DEC_ELEMENT_AJOC) {
            const ac4_toc_sub *s = chosen_sub(&d->cfg, &tt);
            if (s) {
                d->ajoc_info = *s;
                d->ajoc_known = 1;
            }
        }
        if (el != AC4DEC_ELEMENT_AUTO)
            set_element(d, el);
    }
    if (!d->element) {
        /* nothing known yet: silence, timeline kept */
        rec_conceal(d, r, AC4E_UNKNOWN_ELEMENT);
        r->unknown = 1;
        r->mask = 0;
        af_kinds(r, AC4_GRP_NONE);
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
        return p == 0 && d->mode >= 1;
    if (p == 0)
        return (d->el_mask & 1) != 0;   /* L/R: every element but mono */
    if (p == 1 && d->acpl1 && d->acpl1 != AC4_ACPL1_ICE)
        return 0;                   /* ASPX_ACPL_1 / _2: no Ls/Rs A-SPX */
    if (d->element == AC4DEC_ELEMENT_AJOC)
        return (d->el_mask >> d->pair_ch[p][0]) & 1;
    if (d->acpl3 || d->n_ch <= 2)
        return 0;
    return (d->el_mask >> d->pair_ch[p][0]) & 1;  /* the element has it */
}

static void np_frame_stats(const double *x, int n, double *rms,
                           double *amax)
{
    double sq[FRAME], m = 0.0;
    int i, nan = 0;
    for (i = 0; i < n; i++) {
        double a = fabs(x[i]);
        sq[i] = x[i] * x[i];
        if (a != a)
            nan = 1;
        else if (a > m)
            m = a;
    }
    *rms = sqrt(ac4_np_sum(sq, n) / (double)n);
    *amax = nan ? NAN : m;
}

/* core_for_bursts for one frame of one pair: the burst guard and the
   aspx.level check. -> 1 when aspx.level flags the frame; *bad_out: the
   frame plays its core */
static int bursts(ac4dec *d, int p, double *y[2], double *dc[2],
                  int *bad_out)
{
    double limit = 2.0 * AC4_FULL_SCALE;   /* decoder.BURST_LIMIT */
    double floor_ = HF_MIN_FRAC * ac4dec_ref_peak((ac4dec_element)d->element);
    double pk = 0.0;
    int c, over = 0, hot, bad = 0;
    for (c = 0; c < 2; c++) {
        double rh, rc, hm, tmp;
        np_frame_stats(y[c], d->fl, &rh, &hm);
        np_frame_stats(dc[c], d->fl, &rc, &tmp);
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
            memcpy(y[c], dc[c], sizeof(double) * (size_t)d->fl);
    }
    *bad_out = bad;
    return over;
}

/* frame peak over the output channels (np max: NaN propagates; then
   non-finite -> inf) */
static double frame_peak(const outf_t *o, int n_ch, int fl)
{
    double m = 0.0;
    int c, i;
    for (c = 0; c < n_ch; c++)
        for (i = 0; i < fl; i++) {
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
static void assemble(ac4dec *d, double hf[NPAIR][2][FRAME],
                     double dcore[NPAIR][2][FRAME], const int *have)
{
    outf_t *o;
    const rec_t *ri = &d->prev_info;
    int k, p, bad, bad_any = 0;
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
    o->meta = ri->meta;
    o->has_obj = ri->has_obj && d->ajf_cap;
    if (o->has_obj) {
        const ac4_ajoc_frame *fr = &d->ajf[(int)(ri->index % d->ajf_cap)];
        o->n_dev = fr->n_dev;
        memcpy(o->dev, fr->dev, sizeof(o->dev));
        memcpy(o->dblk, fr->dblk, sizeof(o->dblk));
    }
    o->forced = ri->code != AC4E_NONE;
    o->aspx_flag = 0;
    /* every channel 577 samples late: the delay line, unless its pair
       is on the QMF path (below) */
    for (k = 0; k < NCH; k++) {
        /* (tail ++ core)[:fl] out, its last HF_DELAY kept: the frame can
           be shorter than the delay (512, 384) */
        double x[HF_DELAY + FRAME];
        memcpy(x, d->core_tail[k], sizeof(double) * HF_DELAY);
        memcpy(x + HF_DELAY, d->core_prev[k], sizeof(double) * (size_t)d->fl);
        memcpy(o->pcm[k], x, sizeof(double) * (size_t)d->fl);
        memcpy(d->core_tail[k], x + d->fl, sizeof(double) * HF_DELAY);
    }
    /* the immersive A-CPL's own channels (Lb Rb Tbl Tbr): silent until
       the upmix runs */
    for (k = NCH; k < NOUT; k++)
        memset(o->pcm[k], 0, sizeof(o->pcm[k]));
    for (p = 0; p < NPAIR; p++) {
        double *y[2], *dc[2];
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
        if (bad)
            bad_any = 1;
        if (d->pair[p].de_on == 1 && !bad) {
            /* dialogue enhancement: the frames the checks kept */
            y[0] = d->pair[p].deout[0];
            y[1] = d->pair[p].deout[1];
        }
        memcpy(o->pcm[d->pair_ch[p][0]], y[0], sizeof(o->pcm[0]));
        memcpy(o->pcm[d->pair_ch[p][1]], y[1], sizeof(o->pcm[0]));
        if (p == 0 && d->pair[0].acpl_on) {
            /* the A-CPL upmix (decoder.py): a frame the checks above gave
               its core plays the core (the stereo downmix) on L / R and
               nothing on C / Ls / Rs (a pair element in ASPX_ACPL_1 / _2:
               its mono core on both) */
            static const int CH5[5] = { CH_L, CH_R, CH_C, CH_LS, CH_RS };
            int c, on = d->pair[0].acpl_on;
            if (on == 2 && bad)
                memcpy(o->pcm[CH_R], o->pcm[CH_L], sizeof(o->pcm[0]));
            for (c = 0; c < 5; c++) {
                if ((on == 2 && c >= 2) || (on == 3 && c == 2))
                    continue;
                if (bad && c < 2)
                    continue;
                if (bad)
                    memset(o->pcm[CH5[c]], 0, sizeof(o->pcm[0]));
                else
                    memcpy(o->pcm[CH5[c]], d->pair[0].upout[c],
                           sizeof(o->pcm[0]));
            }
        }
        if (d->pair[p].acpl_on == 4) {
            /* the immersive element (decoder.py ice_up): z0, z1 of the
               pair's two modules; a frame the checks gave its core plays
               the core at gain 2 (core decoding mode) on Ls / Rs (Tfl /
               Tfr) and nothing on Lb / Rb (Tbl / Tbr) */
            int sl[4], c, i;
            sl[0] = d->pair_ch[p][0];
            sl[2] = d->pair_ch[p][1];
            sl[1] = p == 1 ? CH_LB : CH_TBL;
            sl[3] = p == 1 ? CH_RB : CH_TBR;
            for (c = 0; c < 4; c++) {
                if (bad && (c == 1 || c == 3))
                    memset(o->pcm[sl[c]], 0, sizeof(o->pcm[0]));
                else if (bad)
                    for (i = 0; i < d->fl; i++)
                        o->pcm[sl[c]][i] = 2.0 * o->pcm[sl[c]][i];
                else
                    memcpy(o->pcm[sl[c]], d->pair[p].upout[c],
                           sizeof(o->pcm[0]));
            }
        }
    }
    if (d->ajoc_full && d->ajx) {
        /* A-JOC full decoding (decoder.AjocFull): the downmix tracks'
           A-SPX output of this frame's slots, in A-JOC input order, to
           the layout; a frame the checks gave its core plays silence */
        static const int SL[8][7] = {
            { 0 }, { CH_C }, { CH_L, CH_R }, { CH_L, CH_R, CH_C },
            { CH_L, CH_R, CH_LS, CH_RS }, { CH_L, CH_R, CH_LS, CH_RS, CH_C },
            { CH_L, CH_R, CH_LS, CH_RS, CH_X1, CH_X2 },
            { CH_L, CH_R, CH_LS, CH_RS, CH_X1, CH_X2, CH_C } };
        static const int S5[5] = { CH_L, CH_R, CH_C, CH_LS, CH_RS };
        const double (*xr[AC4_AJOC_ND])[64], (*xi[AC4_AJOC_ND])[64];
        const int *pos = ac4_ajoc_full_in_pos(d->ajx);
        int n_in = ac4_ajoc_full_n_in(d->ajx), nts = d->pair[0].nts, i, c;
        int nd = d->ajoc_info.n_dmx < 8 ? d->ajoc_info.n_dmx : 7;
        const ac4_ajoc_frame *fr = NULL;
        double lfe[FRAME];
        for (i = 0; i < n_in; i++) {
            int sl = d->ajoc_info.static_dmx ? S5[pos[i]] : SL[nd][pos[i]];
            xr[i] = xi[i] = NULL;
            for (p = 0; p < NPAIR; p++) {
                int h = d->pair_ch[p][1] == sl ? 1
                        : d->pair_ch[p][0] == sl ? 0 : -1, t;
                long long ts0;
                if (h < 0 || !d->path_on[p] || !have[p])
                    continue;
                /* the slots the pair synthesised last (push: frame
                   n_in - 2; finish: n_in - 1) */
                ts0 = (d->pair[p].n_in - (d->finished ? 1 : 2))
                      * (long long)nts;
                for (t = 0; t < nts; t++) {
                    int q = (int)((ts0 + t) % AC4_QRING);
                    memcpy(d->ajin_r[i][t], d->pair[p].qre[h][q],
                           sizeof(d->ajin_r[i][t]));
                    memcpy(d->ajin_i[i][t], d->pair[p].qim[h][q],
                           sizeof(d->ajin_i[i][t]));
                }
                xr[i] = (const double (*)[64])d->ajin_r[i];
                xi[i] = (const double (*)[64])d->ajin_i[i];
            }
        }
        if (o->has_obj)
            fr = &d->ajf[(int)(ri->index % d->ajf_cap)];
        memcpy(lfe, o->pcm[CH_LFE], sizeof(lfe));
        ac4_ajoc_full_frame(d->ajx, d->qmf, xr, xi, nts, d->pair[0].tso, fr,
                            d->ajoc_info.lfe ? lfe : NULL, d->ajout);
        for (c = 0; c < d->n_lay; c++) {
            if (bad_any)
                memset(o->pcm[c], 0, sizeof(o->pcm[c]));
            else
                memcpy(o->pcm[c], d->ajout[c], sizeof(o->pcm[c]));
        }
        goto ordered;
    }
    if (d->element == AC4DEC_ELEMENT_ICE) {
        /* Pseudocode 2: z0, z2, z4 = 2 x (L, R, C); the core decoding
           mode's gain 2 on the others where there is no upmix */
        int i, up = d->pair[1].acpl_on == 4 || d->pair[3].acpl_on == 4;
        for (k = 0; k < NCH; k++) {
            if (k == CH_LFE)
                continue;
            if (k == CH_L || k == CH_R || k == CH_C || (!up && d->n_ch > 2))
                for (i = 0; i < d->fl; i++)
                    o->pcm[k][i] = 2.0 * o->pcm[k][i];
        }
    }
    /* the output channels in order (output c is slot out_ch[c]); through
       a copy when a slot an earlier output overwrote is still needed */
    {
        int need_copy = 0;
        for (k = 0; k < d->n_ch; k++)
            if (d->out_ch[k] < k)
                need_copy = 1;
        if (need_copy) {
            double (*tmp)[FRAME] = d->reorder;
            for (k = 0; k < d->n_ch; k++)
                memcpy(tmp[k], o->pcm[d->out_ch[k]], sizeof(tmp[k]));
            for (k = 0; k < d->n_ch; k++)
                memcpy(o->pcm[k], tmp[k], sizeof(tmp[k]));
        } else {
            for (k = 0; k < d->n_ch; k++)
                if (d->out_ch[k] != k)
                    memcpy(o->pcm[k], o->pcm[d->out_ch[k]],
                           sizeof(o->pcm[k]));
        }
    }
ordered:
    /* channels a frame does not have are silent */
    for (k = d->n_ch; k < NOUT; k++)
        memset(o->pcm[k], 0, sizeof(o->pcm[k]));
    o->n = d->fl;
    o->pk = frame_peak(o, d->n_ch, d->fl);
    o->ok = !o->forced && isfinite(o->pk);
}

/* synthesise the oldest held record */
static void synth_one(ac4dec *d)
{
    rec_t *r = &d->recs[(int)(d->rec_first % d->rec_cap)];
    double (*hf)[2][FRAME] = d->hf, (*dcore)[2][FRAME] = d->dcore;
    int have[NPAIR] = { 0, 0, 0, 0 }, k, w, p, i;
    /* IMDCT / overlap-add */
    for (k = 0; k < NCH; k++) {
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
        for (i = 0; i < d->fl; i++)
            if (!isfinite(d->core[k][i]))
                d->core[k][i] = 0.0;
    }
    /* A-SPX: concealed frames take the tables of the pair's last frame
       with data (its aspx_xover_subband_offset), or the stream's
       configuration at offset 0 */
    for (p = 0; p < NPAIR; p++) {
        ac4_aspx_frame *f = &r->af[p];
        if (f->kind == AC4_GRP_CONCEALED) {
            f->has_tab = d->has_cfg;
            if (d->has_cfg)
                f->tab = d->has_syn_tab[p] ? d->syn_tab[p] : d->tab;
        } else if (f->kind == AC4_GRP_DATA) {
            d->syn_tab[p] = f->tab;
            d->has_syn_tab[p] = 1;
        }
        if (!d->pair_active[p] && pair_capable(d, p)) {
            d->pair_active[p] = 1;
            d->pair_start[p] = r->index;
            /* ASPX_ACPL_3 decoded to 5.1: the A-CPL upmix on the core
               pair, from its first frame */
            if (p == 0 && d->acpl3 && d->n_ch == 6
                && d->element == AC4DEC_ELEMENT_5X)
                d->pair[0].acpl_on = 1;
            if (p == 0 && d->acpl1 == AC4_ACPL1_PAIR)
                d->pair[0].acpl_on = 2;
            if (p == 0 && d->acpl1 == AC4_ACPL1_5X && d->n_ch == 6
                && d->element == AC4DEC_ELEMENT_5X)
                d->pair[0].acpl_on = 3;
            /* the immersive element's A-CPL (decoder.py ice_up): two
               modules on Ls / Rs (D0) and on Tfl / Tfr (D1) */
            if ((p == 1 || p == 3) && d->acpl1 == AC4_ACPL1_ICE
                && d->n_ch > 2) {
                d->pair[p].acpl_on = 4;
                d->pair[p].acpl_dec = p == 1 ? 0 : 1;
            }
            /* dialogue enhancement (decoder.py de_arg): on the upmix's
               L / R / C, or on a pair of L, R or C */
            if (d->cfg.de_level != 0.0) {
                ac4_aspx_pair *pr = &d->pair[p];
                pr->de_level = d->cfg.de_level;
                if (p == 0 && pr->acpl_on) {
                    pr->de_on = 2;
                } else if (d->pair_ch[p][0] <= CH_C
                           && d->pair_ch[p][1] <= CH_C) {
                    pr->de_on = 1;
                    pr->de_ch[0] = d->pair_ch[p][0];
                    pr->de_ch[1] = d->pair_ch[p][1];
                }
            }
        }
        if (d->pair_active[p]) {
            const double *x[2];
            double *y[2], *dc[2];
            x[0] = d->core[d->pair_ch[p][0]];
            x[1] = d->core[d->pair_ch[p][1]];
            if (p == 0 && d->acpl1 == AC4_ACPL1_PAIR)
                x[1] = d->core[CH_L];       /* the "m" track twice */
            if (p == 0 && d->pair[0].acpl_on >= 2) {
                const double *res[2];
                res[0] = d->pair[0].acpl_on == 2 ? d->core[CH_R]
                                                 : d->core[CH_LS];
                res[1] = d->pair[0].acpl_on == 2 ? NULL : d->core[CH_RS];
                ac4_aspx_pair_residual(&d->pair[0], d->qmf, res);
            }
            y[0] = hf[p][0];
            y[1] = hf[p][1];
            dc[0] = dcore[p][0];
            dc[1] = dcore[p][1];
            f->de = r->de;
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
    int have[NPAIR] = { 0, 0, 0, 0 }, p;
    if (d->finished)
        return;
    d->finished = 1;
    if (d->n_synth < 1)
        return;
    for (p = 0; p < NPAIR; p++)
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

static void mix_decide(ac4dec *d, const uint8_t *data, size_t len);
static void mix_frame(ac4dec *d, outf_t *o, ac4dec_frame *out);

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

/* FULL_SCALE x 10^((T + min(offset, 0) + margin) / 20) */
static double tp_limit(const guard_t *g, double t, double margin)
{
    double off = g->tp_off < 0.0 ? g->tp_off : 0.0;
    return AC4_FULL_SCALE * pow(10.0, (t + off + margin) / 20.0);
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
    double ref, lv, down, tmax = 0.0;
    int near = 0, loud, i, c, have = 0;
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
    /* the true peak the stream sent for frames f - 3 .. f + 3
       (robust.OutputGuard.truepk_limits) */
    for (i = 0; i < g->n_tp; i++)
        if (g->tp_has[i] && (!have || g->tp_hist[i] > tmax)) {
            tmax = g->tp_hist[i];
            have = 1;
        }
    for (i = 0; i <= G_TP_WIN && i < d->ring_n; i++) {
        const outf_t *a = &d->ring[(int)((d->ring_first + i) % d->ring_cap)];
        if (a->meta.valid && a->meta.has_truepk
            && (!have || a->meta.truepk > tmax)) {
            tmax = a->meta.truepk;
            have = 1;
        }
    }
    if (near) {
        double m = g->ref > g->floor ? g->ref : g->floor;
        loud = o->pk > G_NEAR_RATIO * m
               && (!have || o->pk > tp_limit(g, tmax, G_TP_NEAR_DB));
    } else {
        loud = o->pk > G_RATIO * ref;
    }
    if (loud && o->ok && !near && sustained(d, 0, ref)) {
        loud = 0;
        g->n_sustained++;
    }
    o->tp_fired = 0;
    if (!loud && have && o->pk > tp_limit(g, tmax, G_TP_MARGIN_DB)) {
        loud = 1;
        o->tp_fired = 1;
        g->n_truepk++;
    }
    if (g->n_tp < G_TP_WIN) {
        g->n_tp++;
    } else {
        memmove(g->tp_hist, g->tp_hist + 1, sizeof(double) * (G_TP_WIN - 1));
        memmove(g->tp_has, g->tp_has + 1, sizeof(int) * (G_TP_WIN - 1));
    }
    g->tp_has[g->n_tp - 1] = o->meta.valid && o->meta.has_truepk;
    g->tp_hist[g->n_tp - 1] = o->meta.truepk;
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
        if (o->ok && !near && have && o->pk > 0.0) {
            /* the stream's level against its truepk: a median tracker */
            double r = 20.0 * log10(o->pk / AC4_FULL_SCALE) - tmax;
            if (r > g->tp_off)
                g->tp_off += G_TP_STEP;
            else if (r < g->tp_off)
                g->tp_off -= G_TP_STEP;
        }
        nr = g->ref * G_RELEASE;
        if (rise > nr)
            nr = rise;
        if (g->floor > nr)
            nr = g->floor;
        g->ref = nr;
    }
    for (c = 0; c < d->n_ch; c++)
        g->last[c] = o->pcm[c][o->n - 1];
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
            for (i = 0; i < o->n; i++)
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
        code = o->tp_fired ? AC4E_GUARD_TRUEPK : AC4E_GUARD_LEVEL;
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
        if (code == AC4E_GUARD_LEVEL || code == AC4E_GUARD_TRUEPK)
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
    if (d->n_kid && d->objmix && d->element == AC4DEC_ELEMENT_AJOC
        && !d->ajoc_full) {
        /* mix.MixDecoder: an object main substream is rendered by its
           own decoder, then mixed on the layout's channels */
        double (*tmp)[FRAME] = d->reorder;
        int c;
        obj_render_frame(d, o, tmp);
        for (c = 0; c < d->n_lay; c++)
            memcpy(o->pcm[c], tmp[c], sizeof(tmp[c]));
    }
    if (d->n_kid)
        mix_frame(d, o, out);
    if (!d->rs_fixed) {
        /* the output resampler: the stream's ratio (Table 83 / 84), fixed
           at the first frame given out (1001 / 960 if no rate is known
           by then) */
        int up = 1001, dn = 960;
        if (d->rate_known)
            ac4_rate_ratio(d->fs, d->fri, &up, &dn);
        ac4_rs_filter_free(d->rsf);
        d->rsf = ac4_rs_filter_new_ratio(up, dn);
        d->out_rate = d->rate_known && d->fs == 0 ? 44100 : 48000;
        d->rs_up = up;
        d->rs_down = dn;
        d->rs_fixed = 1;
    }
    if (d->cfg.loudness) {
        double *pp[NOUT];
        if (!d->lev) {
            int drc = d->cfg.drc == 0 ? AC4_DRC_AUTO
                      : d->cfg.drc < 0 ? AC4_DRC_OFF : d->cfg.drc - 1;
            d->lev = (ac4_leveler *)malloc(sizeof(ac4_leveler));
            if (d->lev && d->objmix && d->n_kid)
                ac4_leveler_init(d->lev, d->n_lay, d->lay_slot, 0,
                                 d->cfg.target_level, drc,
                                 (double)d->out_rate * d->rs_down / d->rs_up);
            else if (d->lev)
                ac4_leveler_init(d->lev, d->n_ch, d->out_ch,
                                 d->element == AC4DEC_ELEMENT_7X
                                 && d->ch_mode <= 6,
                                 d->cfg.target_level, drc,
                                 (double)d->out_rate * d->rs_down / d->rs_up);
        }
        for (k = 0; k < (d->objmix && d->n_kid ? d->n_lay : d->n_ch); k++)
            pp[k] = o->pcm[k];
        if (d->lev)
            ac4_leveler_frame(d->lev, o->meta.valid ? &o->meta : NULL, pp,
                              o->n);
    }
    if (d->element == AC4DEC_ELEMENT_AJOC && !d->ajoc_full
        && !(d->objmix && d->n_kid)) {
        /* ObjectRenderer (core decoding): the frame's downmix objects'
           block updates */
        double (*tmp)[FRAME] = d->reorder;
        int c;
        obj_render_frame(d, o, tmp);
        for (c = 0; c < d->n_lay; c++)
            memcpy(o->pcm[c], tmp[c], sizeof(tmp[c]));
    } else if (d->cfg.layout && d->element != AC4DEC_ELEMENT_AJOC) {
        /* render.Renderer: the frame's custom downmix data held, then
           the matrix */
        double (*tmp)[FRAME] = d->reorder;
        int c, j, i;
        double lg = 1.0;
        if (o->meta.has_lc)
            for (c = 0; c < AC4_LC_N; c++)
                if (o->meta.lc_has[c]) {
                    d->lc_has[c] = 1;
                    d->lc[c] = o->meta.lc[c];
                }
        if (o->meta.cdmx.present) {
            ac4_cdmx_hold(&d->cdmx_held, &o->meta.cdmx);
            render_update(d);
        }
        if (d->is_dmx) {
            /* render.Renderer._gain: the downmix loudness correction of
               the layout (4.8.5.3), held */
            static const int LC[8] = { 0, 0, AC4_LC_5X, AC4_LC_5X2,
                                       AC4_LC_5X4, AC4_LC_7X, AC4_LC_7X2,
                                       AC4_LC_7X4 };
            int lk = d->cfg.layout == AC4DEC_LAYOUT_2_0
                     ? (ac4_stereo_method(d->cfg.stereo_dmx,
                                          d->cdmx_held.present
                                          ? &d->cdmx_held : NULL) == 0
                        ? AC4_LC_LORO : AC4_LC_LTRT)
                     : LC[d->cfg.layout];
            if (d->lc_has[lk])
                lg = pow(10.0, d->lc[lk] / 20.0);
        }
        for (c = 0; c < d->n_lay; c++) {
            memset(tmp[c], 0, sizeof(tmp[c]));
            for (j = 0; j < d->n_ch; j++) {
                double g = lg == 1.0 ? d->rmat[c * d->n_ch + j]
                                     : d->rmat[c * d->n_ch + j] * lg;
                for (i = 0; i < o->n; i++)
                    tmp[c][i] = tmp[c][i] + o->pcm[j][i] * g;
            }
        }
        for (c = 0; c < d->n_lay; c++)
            memcpy(o->pcm[c], tmp[c], sizeof(tmp[c]));
    }
    out->n_channels = d->n_lay;
    if (d->cfg.internal_rate) {
        for (k = 0; k < d->n_lay; k++)
            out->pcm[k] = o->pcm[k];
        out->n_samples = o->n;
        out->sample_rate = (int)floor((double)d->out_rate * d->rs_down
                                      / d->rs_up + 0.5);
    } else if (!d->rsf) {               /* out of memory: silence */
        for (k = 0; k < d->n_lay; k++) {
            memset(d->pcm_out[k], 0, sizeof(d->pcm_out[k]));
            out->pcm[k] = d->pcm_out[k];
        }
        out->n_samples = 0;
    } else {
        out->sample_rate = d->out_rate;
        for (k = 0; k < d->n_lay; k++)
            nout = ac4_rs_push(d->rsf, &d->rs[k], o->pcm[k], o->n,
                               d->pcm_out[k]);
        for (k = 0; k < d->n_lay; k++)
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
    if (!d->mix_decided && !d->cfg.no_mix && d->cfg.substream == 0)
        mix_decide(d, data, len);
    {
        int k;
        for (k = 0; k < d->n_kid; k++)
            d->kid_have[k] = ac4dec_decode_frame(d->kid[k], data, len,
                                                 &d->kid_out[k]);
    }
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

/* ------------------------------------------------------------ mixing */
static ac4dec_element element_of_cm(long cm, int version)
{
    switch (cm) {
    case 0: return AC4DEC_ELEMENT_MONO;
    case 1: return AC4DEC_ELEMENT_PAIR;
    case 2: return AC4DEC_ELEMENT_3_0;
    case 3: return AC4DEC_ELEMENT_5_0;
    case 4: return AC4DEC_ELEMENT_5X;
    default: break;
    }
    if (cm >= 5 && cm <= 10)
        return version == 2 ? AC4DEC_ELEMENT_PAIR : AC4DEC_ELEMENT_7X;
    if (cm >= 11 && cm <= 14)
        return AC4DEC_ELEMENT_ICE;
    return AC4DEC_ELEMENT_AUTO;
}

/* mix.probe / MixDecoder: at the first frame whose TOC parses, open a
   decoder for every other channel-coded substream of the presentation
   and bring it up to the frames fed so far (as lost frames) */
static void mix_decide(ac4dec *d, const uint8_t *data, size_t len)
{
    ac4_toc t;
    const ac4_toc_sub *s[AC4_MIX_MAX];
    long grp[AC4_MIX_MAX], order[AC4_MIX_MAX];
    int pi, n, k, i, n_order = 0, kmain = 0, obj_lay;
    if (toc_try(data, len, &t) != 0)
        return;
    d->mix_decided = 1;
    pi = choose_pres(&d->cfg, &t);
    n = ac4_toc_audio_substreams(&t, pi, s, grp, AC4_MIX_MAX);
    if (n < 2)
        return;
    for (k = 0; k < n; k++) {
        for (i = 0; i < n_order; i++)
            if (order[i] == grp[k])
                break;
        if (i == n_order)
            order[n_order++] = grp[k];
    }
    for (k = n - 1; k >= 0; k--)
        if (ac4_toc_substream_type(&t, pi, grp[k]) == 0)
            kmain = k;
    d->mix_version = t.pres[pi].version;
    d->cfg.substream = kmain;
    d->mix_n = 0;
    /* mix.MixDecoder: object substreams in core decoding, rendered to
       the main substream's layout (when it is a channel substream of one
       of the layouts), else to the layout asked for, or 5.1 */
    {
        const ac4_toc_sub *m = s[kmain];
        obj_lay = d->cfg.layout ? d->cfg.layout : AC4DEC_LAYOUT_5_1;
        if (m->kind == 0) {
            int el = element_of_cm(m->channel_mode, t.pres[pi].version);
            if (el == AC4DEC_ELEMENT_PAIR)
                obj_lay = AC4DEC_LAYOUT_2_0;
            else if (el == AC4DEC_ELEMENT_5X)
                obj_lay = AC4DEC_LAYOUT_5_1;
            else if (el == AC4DEC_ELEMENT_7X && m->channel_mode == 6)
                obj_lay = AC4DEC_LAYOUT_7_1;
            else if (el == AC4DEC_ELEMENT_ICE
                     && (m->channel_mode == 12 || m->channel_mode == 14)
                     && m->centre && m->top == 3)
                obj_lay = m->back4 ? AC4DEC_LAYOUT_7_1_4
                                   : AC4DEC_LAYOUT_5_1_4;
        } else {
            d->cfg.ajoc_core = 1;
            d->objmix = 1;
            lay_slots(ajoc_layout(d), d->lay_slot);
        }
    }
    for (k = 0; k < n; k++) {
        int slot = k == kmain ? 0 : ++d->n_kid;
        d->mix_type[slot] = ac4_toc_substream_type(&t, pi, grp[k]);
        for (i = 0; i < n_order; i++)
            if (order[i] == grp[k])
                d->mix_sg[slot] = i;
        if (slot > 0) {
            ac4dec_config c = d->cfg;
            long long j;
            c.element = s[k]->kind != 0 ? AC4DEC_ELEMENT_AJOC
                        : element_of_cm(s[k]->channel_mode,
                                        t.pres[pi].version);
            c.ch_mode = c.element == AC4DEC_ELEMENT_7X
                        || c.element == AC4DEC_ELEMENT_ICE
                        ? (int)s[k]->channel_mode : 0;
            c.present = c.element == AC4DEC_ELEMENT_ICE
                        ? 16 | (s[k]->back4 ? 1 : 0) | (s[k]->centre ? 2 : 0)
                          | (s[k]->top << 2) : 0;
            c.layout = c.element == AC4DEC_ELEMENT_AJOC ? obj_lay : 0;
            c.ajoc_core = 1;
            c.channels = 0;
            c.internal_rate = 1;
            c.loudness = 0;
            c.substream = k;
            c.no_mix = 1;
            d->kid[slot - 1] = c.element == AC4DEC_ELEMENT_AUTO ? NULL
                               : ac4dec_open(&c);
            if (!d->kid[slot - 1]) {
                /* not decodable: the main substream only */
                for (i = 0; i < slot - 1; i++)
                    ac4dec_close(d->kid[i]);
                d->n_kid = 0;
                d->cfg.substream = 0;
                return;
            }
            for (j = 0; j < d->n_in - 1; j++)
                ac4dec_decode_frame(d->kid[slot - 1], NULL, 0,
                                    &d->kid_out[slot - 1]);
        }
    }
    d->mix_n = n;
    d->mix_n_sg = n_order < AC4_MAX_SG ? n_order : AC4_MAX_SG;
}

static double mix_db(double x)
{
    return isinf(x) && x < 0 ? 0.0 : pow(10.0, x / 20.0);
}

/* mix.pan_gains over the element's loudspeakers (slot mask) */
static void pan_gains(double angle, int mask, double *g)
{
    static const int SL[5] = { CH_LS, CH_L, CH_C, CH_R, CH_RS };
    static const double AZ[5] = { -110.0, -30.0, 0.0, 30.0, 110.0 };
    int sp[6], n = 0, i;
    double az[6], a;
    for (i = 0; i < NOUT; i++)
        g[i] = 0.0;
    for (i = 0; i < 5; i++)
        if (mask & (1 << SL[i])) {
            sp[n] = SL[i];
            az[n++] = AZ[i];
        }
    if (n == 0)
        return;
    if (n == 1) {
        g[sp[0]] = 1.0;
        return;
    }
    a = fmod(angle + 180.0, 360.0) - 180.0;
    sp[n] = sp[0];
    az[n] = az[0] + 360.0;
    for (i = 0; i < n; i++) {
        double x = a >= az[i] ? a : a + 360.0, t;
        if (az[i] <= x && x <= az[i + 1]) {
            t = az[i + 1] > az[i] ? (x - az[i]) / (az[i + 1] - az[i]) : 0.0;
            g[sp[i]] = 1.0 - t;
            g[sp[i + 1]] += t;
            return;
        }
    }
    g[sp[0]] = 1.0;
}

/* MixDecoder.gains + decode_pass for one output frame: o->pcm (this
   decoder's substream, after its guard) becomes the mix */
static void mix_frame(ac4dec *d, outf_t *o, ac4dec_frame *out)
{
    double y[NOUT][FRAME];
    int s, c, j, i, has_assoc = 0, conc = 0;
    const ac4_meta *m = &o->meta;
    /* the mix's channels: the main decoder's, or with an object main the
       layout's (objmix) */
    int nd = d->objmix ? d->n_lay : d->n_ch, dmask = 0;
    const int *dslot = d->objmix ? d->lay_slot : d->out_ch;
    for (c = 0; c < nd; c++)
        dmask |= 1 << dslot[c];
    if (m->valid) {
        if (m->sg_state == 2)
            for (i = 0; i < d->mix_n_sg; i++)
                d->mix_sgdb[i] = i < m->n_sg ? m->sg[i] : 0.0;
        if (m->has_assoc) {
            d->mix_has_assoc = 1;
            memcpy(d->mix_sc, m->sc, sizeof(d->mix_sc));
            memcpy(d->mix_has_sc, m->has_sc, sizeof(d->mix_has_sc));
        }
    }
    for (s = 0; s < d->mix_n; s++)
        if (d->mix_type[s] == 2)
            has_assoc = 1;
    memset(y, 0, sizeof(y));
    for (s = 0; s < d->mix_n; s++) {
        double gm[NOUT][NOUT];        /* [output c][source j] */
        int n_src;
        const double *src[NOUT];
        int src_slot[NOUT];
        if (s == 0) {
            n_src = nd;
            for (j = 0; j < n_src; j++) {
                src[j] = o->pcm[j];
                src_slot[j] = dslot[j];
            }
        } else {
            const ac4dec *k = d->kid[s - 1];
            if (!d->kid_have[s - 1] || d->kid_out[s - 1].index != o->index)
                continue;
            if (d->kid_out[s - 1].status != AC4DEC_OK)
                conc = s;
            n_src = d->kid_out[s - 1].n_channels;
            for (j = 0; j < n_src; j++) {
                src[j] = d->kid_out[s - 1].pcm[j];
                src_slot[j] = k->out_ch[j];
            }
            if (k->element == AC4DEC_ELEMENT_AJOC && !k->ajoc_full)
                lay_slots(ajoc_layout(k), src_slot);   /* rendered */
        }
        memset(gm, 0, sizeof(gm));
        if (n_src == 1 && src_slot[0] == CH_C
            && ((d->mix_type[s] == 2 && d->mix_has_assoc && d->mix_has_sc[3])
                || !(dmask & (1 << CH_C)))) {
            double pg[NOUT];
            double ang = d->mix_type[s] == 2 && d->mix_has_assoc
                         && d->mix_has_sc[3] ? d->mix_sc[3] : 0.0;
            pan_gains(ang, dmask, pg);
            for (c = 0; c < nd; c++)
                gm[c][0] = pg[dslot[c]];
        } else {
            for (c = 0; c < nd; c++)
                for (j = 0; j < n_src; j++)
                    if (src_slot[j] == dslot[c])
                        gm[c][j] = 1.0;
        }
        {
            double g = mix_db(d->mix_sgdb[d->mix_sg[s]]);
            for (c = 0; c < nd; c++)
                for (j = 0; j < n_src; j++)
                    gm[c][j] *= g;
            if (d->mix_type[s] == 2) {
                double ga = mix_db(d->cfg.assoc_gain < 0.0
                                   ? d->cfg.assoc_gain : 0.0);
                for (c = 0; c < nd; c++)
                    for (j = 0; j < n_src; j++)
                        gm[c][j] *= ga;
            }
        }
        if (d->mix_has_assoc && has_assoc
            && (d->mix_type[s] == 0 || d->mix_type[s] == 1)) {
            /* 4.8.3.17: centre (sc[1]), then L / R (sc[2]), then every
               channel (sc[0], scale_main) */
            static const int ORD[3] = { 1, 2, 0 };
            int q;
            for (q = 0; q < 3; q++) {
                double g;
                i = ORD[q];
                if (!d->mix_has_sc[i])
                    continue;
                g = mix_db(d->mix_sc[i]);
                for (c = 0; c < nd; c++) {
                    int sl = dslot[c];
                    if (i == 0 || (i == 2 && (sl == CH_L || sl == CH_R))
                        || (i == 1 && sl == CH_C))
                        for (j = 0; j < n_src; j++)
                            gm[c][j] *= g;
                }
            }
        }
        if (d->mix_version >= 1)
            for (c = 0; c < nd; c++)
                for (j = 0; j < n_src; j++)
                    gm[c][j] /= (double)d->mix_n;
        {
            /* mix.py: a changed matrix crossfades over the frame */
            int same = d->mix_gn[s] == n_src;
            for (c = 0; c < nd && same; c++)
                for (j = 0; j < n_src; j++)
                    if (gm[c][j] != d->mix_gprev[s][c][j])
                        same = 0;
            if (!d->mix_ghas[s] || same || d->mix_gn[s] != n_src) {
                for (c = 0; c < nd; c++)
                    for (j = 0; j < n_src; j++)
                        if (gm[c][j] != 0.0)
                            for (i = 0; i < o->n; i++)
                                y[c][i] += src[j][i] * gm[c][j];
            } else {
                for (c = 0; c < nd; c++)
                    for (j = 0; j < n_src; j++) {
                        double p = j < d->mix_gn[s]
                                   ? d->mix_gprev[s][c][j] : 0.0;
                        if (gm[c][j] == 0.0 && p == 0.0)
                            continue;
                        for (i = 0; i < o->n; i++) {
                            double w = ((double)i + 0.5) / o->n;
                            y[c][i] += src[j][i]
                                       * (p + (gm[c][j] - p) * w);
                        }
                    }
            }
            memcpy(d->mix_gprev[s], gm, sizeof(gm));
            d->mix_gn[s] = n_src;
            d->mix_ghas[s] = 1;
        }
    }
    for (c = 0; c < nd; c++)
        memcpy(o->pcm[c], y[c], sizeof(double) * (size_t)o->n);
    if (conc && out->status == AC4DEC_OK) {
        out->status = AC4DEC_CONCEALED;
        out->reason = d->kid_out[conc - 1].reason;
        d->n_ok--;
        d->n_concealed++;
    }
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
            int k;
            for (k = 0; k < d->n_kid; k++)
                d->kid_have[k] = ac4dec_flush(d->kid[k], &d->kid_out[k]);
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
