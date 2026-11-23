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
typedef struct {
    int bitstream_version;
    int sequence_counter;
    int fs_index;
    int frame_rate_index;
    int b_iframe_global;
    long n_presentations;
    long payload_base;
    int n_cmodes;          /* channel modes seen (capped list below) */
    long cmodes[AC4_MAX_CMODES];
    long n_substreams;
    int n_sizes;           /* 0 when the sizes are not present */
    long sizes[8];         /* first 8 substream sizes */
    double sizes_sum;      /* sum of all sizes */
    long sizes_min;
    size_t toc_bytes;
} ac4_toc;

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

typedef struct {
    int is_long;
    int tl[2];
    int different;
    int max_sfb[2];
    int n_max_sfb;
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
    int32_t lines[1536];
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

/* The parsed element. */
typedef struct {
    int mode;                   /* 5_X_codec_mode or stereo_codec_mode */
    int acpl3;
    int has_aspx_cfg, has_acpl_cfg;
    ac4_aspx_cfg aspx_cfg;      /* valid when has_aspx_cfg */
    ac4_acpl_cfg acpl_cfg;      /* valid when has_acpl_cfg */
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
#define AC4_ASPX_NATS 12        /* num_aspx_timeslots (1536 / 64 / 2) */
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
int ac4_aspx_tables(const ac4_aspx_cfg *c, ac4_aspx_tab *t);

typedef struct {
    int int_class, num_env, num_noise;
    int freq_res[AC4_ASPX_MAX_ENV];
    int borders[AC4_ASPX_MAX_ENV + 1];
} ac4_aspx_framing;

typedef struct {                /* one channel of aspx_data_1ch/2ch */
    ac4_aspx_framing fr;
    int balance, ch;
    int dir_sig[AC4_ASPX_MAX_ENV], dir_noise[2];
    int n_sig[AC4_ASPX_MAX_ENV];
    int sig[AC4_ASPX_MAX_ENV][AC4_ASPX_MAX_SBG];
    int n_noi[2];
    int noi[2][AC4_ASPX_MAX_NOISE];
} ac4_aspx_chd;

typedef struct {
    ac4_aspx_chd c[2];
} ac4_aspx_grp;

/* previous_stop_pos per (channel group 0..2, channel 0..1) */
typedef struct {
    int stop[3][2];
    uint8_t has[3][2];
    int missing;                /* a missing previous_stop_pos was read */
} ac4_aspx_state;

typedef struct {
    ac4_huff h[AC4T_N_ASPX_BOOKS];
    int centre[AC4T_N_ASPX_BOOKS];
    ac4_huff acpl[AC4T_N_ACPL_BOOKS];
} ac4_aspx_books;

int ac4_aspx_books_init(ac4_aspx_books *bk);
void ac4_aspx_books_free(ac4_aspx_books *bk);

/* aspx_data_2ch (two_ch) or aspx_data_1ch into *out (may be NULL).
   Raises AC4E_ASPX_PARSE (out of bits, no codeword), AC4E_ASPX_NUM_ENV,
   AC4E_ASPX_BORDERS. */
void ac4_aspx_data(ac4_bits *b, const ac4_aspx_books *bk,
                   const ac4_aspx_tab *t, const ac4_aspx_cfg *cfg,
                   int b_iframe, int two_ch, ac4_aspx_state *st, int grp,
                   ac4_aspx_grp *out);
/* acpl_data_2ch, parsed for bit accounting only. Raises b->eof_code. */
void ac4_acpl_data_2ch(ac4_bits *b, const ac4_aspx_books *bk,
                       const ac4_acpl_cfg *cfg);

/* ------------------------------------------------------------ A-SPX render
 * One channel pair, streaming. Core frames go in one at a time; the
 * output for frame j comes out when core frame j + 1 has gone in (the
 * frame's envelopes reach 10 QMF slots into the next frame). The QMF bank
 * delays the pair by 577 samples. */
enum { AC4_GRP_NONE = 0, AC4_GRP_CONCEALED, AC4_GRP_DATA };

#define AC4_QRING 96            /* QMF slots kept (4 frames) */

typedef struct {
    int kind;                   /* AC4_GRP_* */
    int has_tab;
    ac4_aspx_tab tab;           /* the tables this frame's data is for */
    ac4_aspx_grp g;
} ac4_aspx_frame;

typedef struct ac4_qmf_tables ac4_qmf_tables;
ac4_qmf_tables *ac4_qmf_tables_new(void);
void ac4_qmf_tables_free(ac4_qmf_tables *q);

typedef struct {
    long long n_in;             /* core frames taken */
    double ana[2][640];
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
    double core[2][2 * 1536];   /* the last two core frames */
    int hot_left;               /* burst guard: frames still owed */
    long n_applied;
} ac4_aspx_pair;

void ac4_aspx_pair_reset(ac4_aspx_pair *p);
/* Take core frame j (x[0], x[1]: 1536 samples each) with its A-SPX
   frame f. When j >= 1, writes the pair's output frame j - 1 to y[0],
   y[1], and (dc not NULL) the core delayed by the QMF bank's 577 samples
   aligned with it to dc[0], dc[1]; returns 1. Else returns 0. */
int ac4_aspx_pair_push(ac4_aspx_pair *p, const ac4_qmf_tables *q,
                       const double *const x[2], const ac4_aspx_frame *f,
                       double *const y[2], double *const dc[2]);
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
    double overlap[1536];
    int n_prev;                 /* 0 = no block yet */
} ac4_fb_state;

/* One block of n spectral lines -> n output samples. tmp: >= 3*1536. */
void ac4_fb_block(const ac4_fb_tables *t, ac4_fb_state *s, const double *X,
                  int n, double *out, double *tmp);

/* ------------------------------------------------------------ resampler */
typedef struct ac4_rs_filter ac4_rs_filter;
ac4_rs_filter *ac4_rs_filter_new(void);
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
