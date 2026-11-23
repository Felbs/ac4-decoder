/* SPDX-License-Identifier: Apache-2.0 */
/* Internal declarations of libac4dec. */
#ifndef AC4_INTERNAL_H
#define AC4_INTERNAL_H

#include <setjmp.h>
#include <stddef.h>
#include <stdint.h>

#include "ac4_tables.h"

/* ------------------------------------------------------------ errors
 * Any rule a frame breaks ends its parse with a longjmp to the frame's
 * handler, carrying one of these codes (the reason strings match the
 * Python decoder's robust.py codes). */
enum {
    AC4E_NONE = 0,
    AC4E_TOC_PARSE,        /* toc.parse */
    AC4E_TOC_SEQ_RANGE,    /* toc.seq_range */
    AC4E_TOC_SIZES,        /* toc.sizes */
    AC4E_TOC_VERSION,      /* toc.version */
    AC4E_TOC_RATE,         /* toc.rate */
    AC4E_TOC_SHAPE,        /* toc.shape */
    AC4E_MP4_SIZE,         /* mp4.size */
    AC4E_AUDIO_OVERRUN,    /* audio_size.overrun */
    AC4E_AUDIO_CLOSURE,    /* audio_size.closure */
    AC4E_ELEMENT_MODE,     /* element.mode */
    AC4E_ELEMENT_PARSE,    /* element.parse (ran out of bits, unsupported) */
    AC4E_SECT_CB,          /* asf.sect_cb */
    AC4E_SECT_LEN,         /* asf.sect_len */
    AC4E_MAX_SFB,          /* asf.max_sfb */
    AC4E_QUANT,            /* asf.quant */
    AC4E_SF_RANGE,         /* asf.sf_range */
    AC4E_LEVEL,            /* asf.level */
    AC4E_INTERNAL,         /* decoder.internal */
    AC4E_UNKNOWN_ELEMENT,  /* element.unknown (AUTO not resolved yet) */
    AC4E_SEQ_OUTLIER,      /* seq.outlier */
    AC4E_ASPX_CONFIG,      /* aspx.config */
    AC4E_ASPX_PARSE,       /* aspx.parse */
    AC4E_ASPX_NUM_ENV,     /* aspx.num_env */
    AC4E_ASPX_BORDERS,     /* aspx.borders */
    AC4E_ACPL_PARSE,       /* acpl.parse */
    AC4E_AUDIO_LATE,       /* audio_size.late */
    AC4E_GUARD_LEVEL,      /* guard.level (output guard) */
    AC4E_ASPX_LEVEL,       /* aspx.level (A-SPX level check) */
    AC4E_TOC_PRESENTATION, /* toc.presentation (no decodable substream) */
    AC4E_TOC_UNSUPPORTED,  /* toc.unsupported_rate (rates.py) */
    AC4E_TRANSF_LEN,       /* asf.transf_length (Table 103: not defined) */
    AC4E_COUNT
};
const char *ac4_reason_name(int code);

/* ------------------------------------------------------------ bits */
typedef struct {
    const uint8_t *d;
    size_t nbits;
    size_t p;
    jmp_buf *jb;
    int eof_code;          /* code to raise when reading past the end */
} ac4_bits;

static inline void ac4_fail(ac4_bits *b, int code)
{
    longjmp(*b->jb, code);
}

/* u(n), 0 <= n <= 32, MSB first. Raises past the end. */
static inline uint32_t ac4_u(ac4_bits *b, int n)
{
    size_t p = b->p, q;
    uint64_t acc;
    int sh;
    if (n == 0) {
        if (p > b->nbits)
            ac4_fail(b, b->eof_code);
        return 0;
    }
    if (p + (size_t)n > b->nbits)
        ac4_fail(b, b->eof_code);
    q = p >> 3;
    sh = (int)(p & 7);
    /* gather the bytes covering bits p .. p+n-1 (at most 5) */
    acc = 0;
    {
        int nb = (sh + n + 7) >> 3, i;
        for (i = 0; i < nb; i++)
            acc = (acc << 8) | b->d[q + (size_t)i];
        acc >>= (nb * 8 - sh - n);
    }
    b->p = p + (size_t)n;
    if (n == 32)
        return (uint32_t)acc;
    return (uint32_t)(acc & ((1u << n) - 1u));
}

static inline uint32_t ac4_bit(ac4_bits *b)
{
    size_t p = b->p;
    if (p >= b->nbits)
        ac4_fail(b, b->eof_code);
    b->p = p + 1;
    return (b->d[p >> 3] >> (7 - (p & 7))) & 1u;
}

/* skip n bits (any n), raising past the end */
static inline void ac4_skip(ac4_bits *b, size_t n)
{
    if (b->p + n > b->nbits)
        ac4_fail(b, b->eof_code);
    b->p += n;
}

static inline void ac4_align(ac4_bits *b)
{
    b->p = (b->p + 7) & ~(size_t)7;
}

/* variable_bits(n): Table 3. Raises (element/TOC parse) above 32 bits. */
uint64_t ac4_vb(ac4_bits *b, int n);

/* ------------------------------------------------------------ Huffman */
typedef struct {
    int32_t *node;         /* 2 entries per node: >0 child, <0 -(sym+1) */
    int n_nodes;
    int ok;
} ac4_huff;

int ac4_huff_build(ac4_huff *h, const ac4t_book *bk);
void ac4_huff_free(ac4_huff *h);
static inline int ac4_huff_decode(const ac4_huff *h, ac4_bits *b)
{
    int node = 0;
    for (;;) {
        int32_t nx = h->node[2 * node + (int)ac4_bit(b)];
        if (nx < 0)
            return -nx - 1;
        if (nx == 0)               /* incomplete code: no match */
            ac4_fail(b, b->eof_code);
        node = nx;
    }
}

/* ------------------------------------------------------------ TOC */
#define AC4_MAX_CMODES 32
#define AC4_MAX_PRES 8         /* presentations kept (more are parsed) */
#define AC4_MAX_PGROUPS 8      /* group references kept per presentation */
#define AC4_MAX_GROUPS 16      /* substream groups kept */
#define AC4_MAX_GSUBS 8        /* substreams kept per group */
#define AC4_MAX_SIZES 32       /* substream sizes kept */

typedef struct {
    int kind;              /* 0 channel-coded, 1 A-JOC, 2 objects */
    long channel_mode;     /* ch_mode (kind 0), else -1 */
    int ndot;              /* the first b_audio_ndot */
    long index;            /* substream_index, -1 when not present */
    int sf_multiplier;     /* -1: none; 0: 96 kHz; 1: 192 kHz */
} ac4_toc_sub;

typedef struct {
    int substreams_present, hsf_ext, channel_coded;
    int n_subs;            /* kept (at most AC4_MAX_GSUBS) */
    long n_subs_all;
    ac4_toc_sub subs[AC4_MAX_GSUBS];
    int content_classifier;   /* -1 when no content_type */
    char language[16];        /* "" when none */
} ac4_toc_group;

typedef struct {
    int config;            /* -1: b_single_substream_group */
    int version, md_compat;
    long presentation_id;  /* -1 when none */
    int enabled;
    int n_groups;          /* kept references */
    long groups[AC4_MAX_PGROUPS];
    long pres_substream;   /* -1 when none */
} ac4_toc_pres;

typedef struct {
    int bitstream_version;
    int sequence_counter;
    int fs_index;
    int frame_rate_index;
    int rate_multi;        /* a presentation with frame_rate_factor or
                              frame_rate_fraction other than 1 */
    int b_iframe_global;
    long n_presentations;
    long payload_base;
    int n_cmodes;          /* channel modes seen (capped list below) */
    long cmodes[AC4_MAX_CMODES];
    long n_substreams;
    int n_sizes;           /* 0 when the sizes are not present */
    long sizes[AC4_MAX_SIZES]; /* first AC4_MAX_SIZES substream sizes */
    double sizes_sum;      /* sum of all sizes */
    long sizes_min;
    size_t toc_bytes;
    int n_pres;            /* kept presentations */
    ac4_toc_pres pres[AC4_MAX_PRES];
    int n_groups;          /* kept substream groups */
    long n_groups_all;     /* total_n_substream_groups */
    ac4_toc_group groups[AC4_MAX_GROUPS];
} ac4_toc;

/* Presentation choice (TS 103 190-2 4.8.2), the rule of ac4/toc.py
   choose(): index >= 0 selects by position, id >= 0 by presentation_id;
   otherwise the first enabled presentation with a decodable audio
   substream, preferring language (a BCP 47 prefix, NULL/"" for none),
   then "complete main" content. -> presentation index or -1. */
int ac4_toc_choose(const ac4_toc *t, int index, long id, const char *language);
/* The audio substream decoded for presentation pi (the first
   channel-coded one with a substream_index inside the size table), or
   NULL. */
const ac4_toc_sub *ac4_toc_audio_substream(const ac4_toc *t, int pi);

/* Parse the TOC (TS 103 190-2 6.2.1, bitstream_version >= 2). Raises
   AC4E_TOC_PARSE through b->jb. */
void ac4_toc_parse(ac4_bits *b, ac4_toc *t);
/* channel_mode -> ch_mode (TS 103 190-2 table 56), and
   ac4_substream_info_chan (6.2.1.8); exposed for the unit tests */
long ac4_toc_channel_mode(ac4_bits *b);
void ac4_toc_substream_info_chan(ac4_bits *b, int present, int frf,
                                 ac4_toc *t);

/* ------------------------------------------------------------ ASF */
#define AC4_MAX_WIN 17
#define AC4_MAX_SFB 64
#define AC4_MAX_FRAME 2048      /* the longest frame_len_base (Table 83) */
#define AC4_DEFAULT_FRAME 1536  /* frames before the stream's rate is known */

/* rates.py: Table 83 / 84. -> frame_len_base, or 0 when not defined */
int ac4_frame_len_base(int fs_index, int frame_rate_index);
/* the decoder resampling ratio (up, down); 1, 1 for frame_rate_index 13 */
void ac4_rate_ratio(int fs_index, int frame_rate_index, int *up, int *down);

typedef struct {
    int fl;                     /* frame_len_base */
    int is_long;
    int tl[2];
    int lens[2];                /* transform length of each half */
    int different;
    int max_sfb[2];
    int n_max_sfb;
    int max_sfb_side[2];        /* ASPX_ACPL_1 (b_dual_maxsfb /
                                   b_side_limited) */
    int n_max_sfb_side;
    int side;                   /* the groups use max_sfb_side */
    int num_windows, num_groups;
    int w2g[AC4_MAX_WIN];
    int nwin[AC4_MAX_WIN];
    /* derived per group: */
    int glen[AC4_MAX_WIN];      /* transform length */
    int gmax[AC4_MAX_WIN];      /* max_sfb */
    int gbase[AC4_MAX_WIN];     /* packed offset base (Pseudocode 4) */
    const ac4t_sfb *gtab[AC4_MAX_WIN];
} ac4_framing;

#define AC4_SF_NONE (-32768)

typedef struct {
    const ac4_framing *fr;
    int32_t lines[AC4_MAX_FRAME];
    int n_lines;                /* packed length ("total") */
    int16_t sfs[AC4_MAX_WIN][AC4_MAX_SFB];   /* AC4_SF_NONE = not sent */
    int has_snf;
    int16_t snf[AC4_MAX_WIN][AC4_MAX_SFB];   /* AC4_SF_NONE = not sent */
} ac4_chan;

typedef struct {
    int sap_mode;               /* -1 = stereo processing off */
    uint8_t ms_used[AC4_MAX_WIN][AC4_MAX_SFB];
    /* sap_mode 3 */
    int coeff_all;
    uint8_t coeff_used[AC4_MAX_WIN][AC4_MAX_SFB];
    int delta_code_time;
    int16_t dpcm[AC4_MAX_WIN][AC4_MAX_SFB];
} ac4_stereo;

typedef struct {
    ac4_framing fr[2];          /* fr[0] shared when stereo proc is on */
    ac4_chan ch[2];
    ac4_stereo st;
} ac4_pair;

typedef struct {
    ac4_framing fr;
    ac4_chan ch;
} ac4_mono;

typedef struct {
    const ac4_huff *cb[12];     /* 1..11 */
    const ac4_huff *sf;
    const ac4_huff *snf;
    int fl;                     /* the stream's frame_len_base */
} ac4_books;

/* ------------------------------------------------------------ A-SPX config */
typedef struct {                /* aspx_config(), Table 50 */
    int quant_mode_env, start_freq, stop_freq, master_freq_scale,
        interpolation, preflat, limiter, noise_sbg, num_env_bits_fixfix,
        freq_res_mode;
} ac4_aspx_cfg;

typedef struct {                /* acpl_config_2ch(), Table 60 */
    int num_param_bands, param_band, quant_mode_0, quant_mode_1;
} ac4_acpl_cfg;

typedef struct {                /* acpl_config_1ch(), Table 59 */
    int num_param_bands, param_band, qmf_band, quant_mode;
} ac4_acpl1_cfg;

/* The parsed element. */
#define AC4_ACPL1_NONE 0
#define AC4_ACPL1_5X 1          /* 5_X ASPX_ACPL_1 / _2 (two modules) */
#define AC4_ACPL1_PAIR 2        /* channel pair ASPX_ACPL_1 / _2 */
typedef struct {
    int mode;                   /* 5_X_codec_mode or stereo_codec_mode */
    int acpl3;
    int acpl1;                  /* AC4_ACPL1_*: acpl_data_1ch() blocks */
    int has_aspx_cfg, has_acpl_cfg;
    ac4_aspx_cfg aspx_cfg;      /* valid when has_aspx_cfg */
    ac4_acpl_cfg acpl_cfg;      /* valid when has_acpl_cfg (2ch) */
    ac4_acpl1_cfg acpl1_cfg;    /* valid when has_acpl_cfg (1ch) */
    /* the A-SPX blocks after the channel data, in bitstream order: block
       group (0 L/R, 1 Ls/Rs, 2 C) and two-channel flag; then n_acpl1
       acpl_data_1ch() (acpl1) or one acpl_data_2ch() (acpl3) */
    int n_blocks, blk_grp[3], blk_two[3], n_acpl1;
    int r_silent;               /* the pair's R not coded (ASPX_ACPL_2) */
    size_t audio_size, audio_start;
    size_t bits_end;            /* where the core parse ended */
    ac4_mono lfe;
    ac4_pair lr, sr;
    ac4_mono c;
} ac4_element;

/* Element parse: 5_X (codec modes 1 and 4) and channel_pair_element
   (stereo_codec_mode 0 and 1). Raises through b->jb. */
void ac4_parse_5x(ac4_bits *b, const ac4_books *bk, int b_iframe,
                  ac4_element *e);
void ac4_parse_pair(ac4_bits *b, const ac4_books *bk, int b_iframe,
                    ac4_element *e);

/* ------------------------------------------------------------ A-SPX
 * Port of ac4/aspx_bands.py, aspx_parse.py, aspx_env.py, aspx_envadj.py,
 * aspx_hfgen.py, aspx_render.py, qmf.py and acpl_parse.py. */
#define AC4_ASPX_MAX_NATS 16    /* num_aspx_timeslots (Pseudocode 75a) */
#define AC4_MAX_NTS 32          /* num_qmf_timeslots at frame length 2048 */
#define AC4_ASPX_MAX_ENV 8      /* FIXFIX: 1 << u(2) */
#define AC4_ASPX_MAX_SBG 24
#define AC4_ASPX_MAX_NOISE 16
#define AC4_ASPX_MAX_PATCH 64
#define AC4_ASPX_MAX_LIM 96

typedef struct {                /* Pseudocodes 67-72 (aspx_bands.sbg_tables) */
    int n_master, master[AC4_ASPX_MAX_SBG];
    int n_hi, hi[AC4_ASPX_MAX_SBG];
    int n_lo, lo[AC4_ASPX_MAX_SBG];
    int n_noise, noise[AC4_ASPX_MAX_NOISE + 1];
    int sba, sbz, sbx, num_sb;
    int n_patches;
    int patch_num_sb[AC4_ASPX_MAX_PATCH], patch_start_sb[AC4_ASPX_MAX_PATCH];
    int patches[AC4_ASPX_MAX_PATCH + 1];
    int n_lim, lim[AC4_ASPX_MAX_LIM];   /* aspx_envadj.sbg_lim (n_lim
                                           borders) */
    int high2low[AC4_ASPX_MAX_SBG], low2high[AC4_ASPX_MAX_SBG + 1];
} ac4_aspx_tab;

/* -> 0, or -1 when the configuration gives no usable tables (the
   Python's aspx.config rule, plus patches that would index outside the
   64 QMF subbands, which the Python would fail on later). */
int ac4_aspx_tables(const ac4_aspx_cfg *c, int xover, ac4_aspx_tab *t);

typedef struct {
    int int_class, num_env, num_noise;
    int freq_res[AC4_ASPX_MAX_ENV];
    int borders[AC4_ASPX_MAX_ENV + 1];
    int tsg_ptr;                /* aspx_tsg_ptr; -1 for FIXFIX (none) */
    int sine_at_end;            /* Pseudocode 92's p_sine_at_end: 0 or -1 */
} ac4_aspx_framing;

typedef struct {                /* one channel of aspx_data_1ch/2ch */
    ac4_aspx_framing fr;
    int balance, ch;
    /* aspx_hfgen_iwc (Tables 55, 56): TNA modes per noise group, add-
       harmonic flags per high-resolution group, FIC flags (has_fic) per
       high-resolution group, TIC flags (has_tic) per A-SPX timeslot */
    uint8_t tna[AC4_ASPX_MAX_NOISE], ah[AC4_ASPX_MAX_SBG];
    int has_fic, has_tic;
    uint8_t fic[AC4_ASPX_MAX_SBG], tic[AC4_ASPX_MAX_NATS];
    int qmode;                  /* aspx_qmode_env: 0 = 1.5 dB, 1 = 3 dB */
    int dir_sig[AC4_ASPX_MAX_ENV], dir_noise[2];
    int n_sig[AC4_ASPX_MAX_ENV];
    int sig[AC4_ASPX_MAX_ENV][AC4_ASPX_MAX_SBG];
    int n_noi[2];
    int noi[2][AC4_ASPX_MAX_NOISE];
} ac4_aspx_chd;

typedef struct {
    ac4_aspx_chd c[2];
    int xover;                  /* aspx_xover_subband_offset in force */
} ac4_aspx_grp;

/* previous_stop_pos per (channel group 0..2, channel 0..1) */
typedef struct {
    int stop[3][2];
    uint8_t has[3][2];
    int missing;                /* a missing previous_stop_pos was read */
    /* aspx_tsg_ptr and num_atsg_sig of the previous interval, per channel
       (Pseudocode 92), and whether there was one */
    int ptr[3][2], nenv[3][2];
    uint8_t has_ptr[3][2];
    int xover[3];               /* aspx_xover_subband_offset per block */
} ac4_aspx_state;

typedef struct {
    ac4_huff h[AC4T_N_ASPX_BOOKS];
    int centre[AC4T_N_ASPX_BOOKS];
    ac4_huff acpl[AC4T_N_ACPL_BOOKS];
} ac4_aspx_books;

int ac4_aspx_books_init(ac4_aspx_books *bk);
void ac4_aspx_books_free(ac4_aspx_books *bk);

/* aspx_data_2ch (two_ch) or aspx_data_1ch into *out (may be NULL), for
   num_aspx_timeslots nats; tab_out (may be NULL) receives the block's
   tables (cfg at its aspx_xover_subband_offset). Raises AC4E_ASPX_PARSE
   (out of bits, no codeword), AC4E_ASPX_NUM_ENV, AC4E_ASPX_BORDERS,
   AC4E_ASPX_CONFIG. */
void ac4_aspx_data(ac4_bits *b, const ac4_aspx_books *bk,
                   const ac4_aspx_cfg *cfg, int nats, int b_iframe,
                   int two_ch, ac4_aspx_state *st, int grp,
                   ac4_aspx_grp *out, ac4_aspx_tab *tab_out);
/* ------------------------------------------------------------ A-CPL
 * Port of ac4/acpl_parse.py (quantised_2ch) and ac4/acpl.py: the upmix
 * of 5_X_codec_mode ASPX_ACPL_3, TS 103 190-1 5.7.7. */
#define AC4_ACPL_NB 15          /* acpl_max_num_param_bands */
#define AC4_ACPL_NQ 19          /* interpolated quantities (acpl.QUANTITIES) */

/* acpl_<SET>_q_prev of the eleven sets (Pseudocode 121); n[k] = 0: none */
typedef struct {
    int q[11][AC4_ACPL_NB];
    int n[11];
} ac4_acpl_qstate;

/* one frame's A-CPL parameters (acpl.render's frame dict) */
typedef struct {
    int kind;                   /* AC4_GRP_NONE / _CONCEALED / _DATA */
    ac4_acpl_cfg cfg;
    int interp, n_sets, ts[2];
    int q[11][2][AC4_ACPL_NB];  /* acpl_<SET>_q, Pseudocode 121 */
} ac4_acpl_frame;

/* acpl_data_2ch. Raises b->eof_code. With out not NULL, the parameters
   are differentially decoded into it against qs (updated). */
void ac4_acpl_data_2ch(ac4_bits *b, const ac4_aspx_books *bk,
                       const ac4_acpl_cfg *cfg, ac4_acpl_qstate *qs,
                       ac4_acpl_frame *out);

/* the tool's state between frames (acpl.new_state) */
typedef struct {
    int warm;
    int qb[2];                  /* 1ch: acpl_qmf_band of the last data */
    double prev[AC4_ACPL_NQ][64];
    double xr[3][14][64], xi[3][14][64];    /* decorrelator input history */
    double yr[3][7][64], yi[3][7][64];      /* and output history */
    double peak[3][AC4_ACPL_NB], smooth[3][AC4_ACPL_NB],
           spd[3][AC4_ACPL_NB];
} ac4_acpl_state;

/* scratch space of ac4_acpl_run (no state) */
typedef struct {
    double ip[AC4_ACPL_NQ][AC4_MAX_NTS][64];
    double sets[AC4_ACPL_NQ][2][64];
    double vr[3][14 + AC4_MAX_NTS][64], vi[3][14 + AC4_MAX_NTS][64];
    double ur[3][7 + AC4_MAX_NTS][64], ui[3][7 + AC4_MAX_NTS][64];
    /* the renderer's gather / scatter buffers */
    double xr[2][AC4_MAX_NTS][64], xi[2][AC4_MAX_NTS][64];
    double zr[5][AC4_MAX_NTS][64], zi[5][AC4_MAX_NTS][64];
} ac4_acpl_work;

/* one frame's parameters of the acpl_data_1ch() modules */
typedef struct {
    int kind;                   /* AC4_GRP_* */
    ac4_acpl1_cfg cfg;
    int interp, n_sets, ts[2];
    int q[2][2][AC4_ACPL_NB];   /* alpha1, beta1 per set (Pseudocode 121) */
} ac4_acpl1_frame;

/* acpl_alpha1 / beta1 q_prev of the two modules */
typedef struct {
    int q[2][2][AC4_ACPL_NB];
    int n[2][2];
} ac4_acpl1_qstate;

/* Table 197: the parameter band of QMF subband sb at num_param_bands */
int ac4_acpl_sb_to_pb(int num_param_bands, int sb);

/* acpl_data_1ch (Table 61). With out not NULL, differentially decoded
   against qs[0..1] (updated). Raises b->eof_code. */
void ac4_acpl_data_1ch(ac4_bits *b, const ac4_aspx_books *bk,
                       const ac4_acpl1_cfg *cfg, int (*qs)[AC4_ACPL_NB],
                       int *qn, ac4_acpl1_frame *out);

/* acpl.render_1ch for one frame of n (<= nts) slots and nm (1 or 2)
   modules: x0[m] the A-SPX output of module m's core channel, x1[m] its
   residual (NULL: none); z0[m], z1[m] receive the outputs (unscaled).
   frames NULL = no data (passthrough: split ? (x0 + x1, x0 - x1) :
   (x0, 0); state cleared), kind CONCEALED = hold. The state is the 2ch
   tool's (prev[2 m], prev[2 m + 1] = module m's alpha, beta). */
void ac4_acpl1_run(ac4_acpl_state *s, ac4_acpl_work *w,
                   const ac4_acpl1_frame *frames, int kind, int nm, int n,
                   int nts, int split,
                   const double (*x0r[2])[64], const double (*x0i[2])[64],
                   const double (*x1r[2])[64], const double (*x1i[2])[64],
                   double (*z0r[2])[64], double (*z0i[2])[64],
                   double (*z1r[2])[64], double (*z1i[2])[64]);

/* acpl.upmix_frame / render for one frame of n (<= nts) slots, nts =
   num_qmf_timeslots: x[c] the pair's QMF slots (re, im: [n][64]); z[ch]
   receives L, R, C, Ls, Rs. f NULL = no A-CPL data (passthrough, state
   cleared). */
void ac4_acpl_run(ac4_acpl_state *s, ac4_acpl_work *w,
                  const ac4_acpl_frame *f, int n, int nts,
                  const double (*xr[2])[64], const double (*xi[2])[64],
                  double (*zr[5])[64], double (*zi[5])[64]);

/* ------------------------------------------------------------ A-SPX render
 * One channel pair, streaming. Core frames go in one at a time; the
 * output for frame j comes out when core frame j + 1 has gone in (the
 * frame's envelopes reach 10 QMF slots into the next frame). The QMF bank
 * delays the pair by 577 samples. */
enum { AC4_GRP_NONE = 0, AC4_GRP_CONCEALED, AC4_GRP_DATA };

#define AC4_QRING 128           /* QMF slots kept (4 frames of 2048) */

typedef struct {
    int kind;                   /* AC4_GRP_* */
    int has_tab;
    ac4_aspx_tab tab;           /* the tables this frame's data is for */
    int preflat, interpolation; /* aspx_config's, for this frame */
    ac4_aspx_grp g;
    ac4_acpl_frame acpl;        /* ASPX_ACPL_3 (pair 0 only) */
    int acpl1_kind;             /* ASPX_ACPL_1 / _2 (pair 0 only) */
    ac4_acpl1_frame acpl1[2];
} ac4_aspx_frame;

typedef struct ac4_qmf_tables ac4_qmf_tables;
ac4_qmf_tables *ac4_qmf_tables_new(void);
void ac4_qmf_tables_free(ac4_qmf_tables *q);

typedef struct {
    int fl, nts, tia, nats, tso;    /* the time grid (Table 192) */
    long long n_in;             /* core frames taken */
    double ana[2][640];
    /* the analysed core (QinASPX), kept apart from the output ring
       qre / qim (TNA, pre-flattening and interleaved waveform coding
       read it) */
    double are[2][AC4_QRING][64], aim[2][AC4_QRING][64];
    /* the A-SPX tools' state per channel (aspx_tools.new_state): chirp
       factors and TNA modes of the previous interval, sine_idx_sb of its
       last envelope, the tone generator's index (-1: first frame) */
    double chirp[2][AC4_ASPX_MAX_NOISE];
    int tna_prev[2][AC4_ASPX_MAX_NOISE], n_tna[2];
    uint8_t sine_sb[2][64];
    int n_sine_sb[2], sine_idx[2];
    double syn[2][1280];
    double qre[2][AC4_QRING][64], qim[2][AC4_QRING][64];
    /* envelope state (aspx_render.apply_hf_pair) */
    long long sprev[2][AC4_ASPX_MAX_SBG];
    int sprev_n[2], sprev_res[2], sprev_valid[2];
    long long nprev[2][AC4_ASPX_MAX_NOISE];
    int nprev_n[2], nprev_valid[2];
    int idx_noise;
    /* the frame whose envelopes are applied next (frame n_in - 1) */
    ac4_aspx_frame prev;
    ac4_aspx_tab tab;           /* tables of the frame being analysed */
    int has_tab;
    double core[2][AC4_MAX_FRAME + 577];    /* the last fl + 577 core
                                               samples */
    int hot_left;               /* burst guard: frames still owed */
    long n_applied;
    /* A-CPL (ASPX_ACPL_3 decoded to 5.1; acpl_on 1), or the 1ch modules
       of ASPX_ACPL_1 / _2 (acpl_on 2: a pair element, outputs L, R; 3: a
       5_X element, outputs L, R, -, Ls, Rs): on, its state, the residuals'
       analysis, the upmixed channels' QMF slots (L, R, C, Ls, Rs) and
       their synthesis banks */
    int acpl_on;
    double rana[2][640];
    double rre[2][AC4_QRING][64], rim[2][AC4_QRING][64];
    ac4_acpl_state acpl;
    ac4_acpl_work acpl_work;
    double ure[5][AC4_QRING][64], uim[5][AC4_QRING][64];
    double tmp1r[2][AC4_MAX_NTS][64], tmp1i[2][AC4_MAX_NTS][64];
    double usyn[5][1280];
    double upout[5][AC4_MAX_FRAME];   /* the last output frame's upmix */
} ac4_aspx_pair;

/* reset to the time grid of frame length 1536 / fl */
void ac4_aspx_pair_reset(ac4_aspx_pair *p);
void ac4_aspx_pair_init(ac4_aspx_pair *p, int fl);
/* Take core frame j (x[0], x[1]: p->fl samples each) with its A-SPX
   frame f. When j >= 1, writes the pair's output frame j - 1 to y[0],
   y[1], and (dc not NULL) the core delayed by the QMF bank's 577 samples
   aligned with it to dc[0], dc[1]; returns 1. Else returns 0. */
int ac4_aspx_pair_push(ac4_aspx_pair *p, const ac4_qmf_tables *q,
                       const double *const x[2], const ac4_aspx_frame *f,
                       double *const y[2], double *const dc[2]);
/* With p->acpl_on, the push / finish that writes output frame j - 1 also
   writes its A-CPL upmix (L, R, C, Ls, Rs) to p->upout. With acpl_on 2 or
   3, ac4_aspx_pair_residual analyses core frame j of the residuals
   (res[m], NULL = silent) before ac4_aspx_pair_push of frame j. */
void ac4_aspx_pair_residual(ac4_aspx_pair *p, const ac4_qmf_tables *q,
                            const double *const res[2]);
/* End of stream: the last frame's output (its envelopes clipped at the
   end of the analysed signal, as at the end of a Python decode pass).
   -> 1 if a frame was written. */
int ac4_aspx_pair_finish(ac4_aspx_pair *p, const ac4_qmf_tables *q,
                         double *const y[2], double *const dc[2]);

/* numpy-compatible helpers (np.add.reduce over a contiguous run) */
double ac4_np_sum(const double *x, long n);

/* ------------------------------------------------------------ spectra */
/* Dequantise into the packed spectrum (returns its length). Raises
   AC4E_LEVEL through jb when a line exceeds the sanity bound. */
int ac4_packed_spectrum(const ac4_chan *ch, double *spec, jmp_buf *jb);

/* Spectral noise fill, TS 103 190-1 5.1.4. nRndStateSnf: reset from
   sequence_counter at the start of every frame (Pseudocode 24), then
   GetRandomNoiseValue (Pseudocode 57) through the element's channels in
   bitstream order. ac4_noise_fill fills the packed spectrum spec (n lines)
   of channel ch (Pseudocodes 22, 23), raising AC4E_LEVEL through jb on a
   line above the level limit; spec NULL only advances the generator by the
   lines the channel would fill (a channel that is not rendered). */
typedef struct {
    unsigned a, b, state, cur;  /* uiOffsetA, uiOffsetB, uiStateIdx,
                                   uiCurrentIdx (uint8 in the spec) */
} ac4_snf_rng;
void ac4_snf_reset(ac4_snf_rng *r, int sequence_counter);
double ac4_snf_next(ac4_snf_rng *r);
void ac4_noise_fill(const ac4_chan *ch, double *spec, int n,
                    ac4_snf_rng *r, jmp_buf *jb);
/* sf_info (asf_transform_info + asf_psy_info) at frame length fl */
void ac4_framing_parse(ac4_bits *b, ac4_framing *f, int fl);
void ac4_framing_parse_ex(ac4_bits *b, ac4_framing *f, int fl,
                          int dual_maxsfb, int side_limited);
void ac4_framing_side(const ac4_framing *src, ac4_framing *f);
void ac4_unmix(double *m, double *s, int n, const ac4_stereo *st,
               const ac4_framing *fr);
/* Packed -> per-window spectra. win[w] (w < num_windows) must hold at
   least the window's length; *lens receives the lengths. */
void ac4_ungroup(const double *packed, int n, const ac4_framing *fr,
                 double *win_buf, int *lens);

/* ------------------------------------------------------------ filterbank */
typedef struct ac4_fb_tables ac4_fb_tables;
ac4_fb_tables *ac4_fb_tables_new(void);
void ac4_fb_tables_free(ac4_fb_tables *t);

typedef struct {
    double overlap[AC4_MAX_FRAME];
    int n_prev;                 /* 0 = no block yet */
    int n_full;                 /* the frame length (0: 1536) */
} ac4_fb_state;

/* One block of n spectral lines -> n output samples. tmp: >= 3 *
   AC4_MAX_FRAME. */
void ac4_fb_block(const ac4_fb_tables *t, ac4_fb_state *s, const double *X,
                  int n, double *out, double *tmp);

/* ------------------------------------------------------------ resampler */
typedef struct ac4_rs_filter ac4_rs_filter;
ac4_rs_filter *ac4_rs_filter_new(void);                 /* 1001 / 960 */
ac4_rs_filter *ac4_rs_filter_new_ratio(int up, int down);
/* -> 1 when the ratio is 1 (samples pass through) */
int ac4_rs_ratio(const ac4_rs_filter *f, int *up, int *down);
void ac4_rs_filter_free(ac4_rs_filter *f);

#define AC4_RS_HIST 64
typedef struct {
    double hist[AC4_RS_HIST];   /* the last inputs (delayed stream) */
    long long n_in;             /* delayed-stream samples taken */
    long long n_out;            /* output samples given */
} ac4_rs_state;

void ac4_rs_init(ac4_rs_state *s);
/* Push n input samples; writes the outputs to out, returns their count. */
int ac4_rs_push(const ac4_rs_filter *f, ac4_rs_state *s, const double *x,
                int n, double *out);

#endif
