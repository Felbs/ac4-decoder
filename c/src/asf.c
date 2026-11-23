/* SPDX-License-Identifier: Apache-2.0 */
/* ASF (MDCT core) bitstream parse: framing (asf_transform_info /
 * asf_psy_info), section, spectral, scale factor and noise fill data,
 * stereo side information (all sap_modes), and the 5_X and
 * channel_pair_element walks. Port of ac4/channels.py and ac4/pair.py
 * (TS 103 190-1 4.2.6 - 4.2.10). */
#include <stdlib.h>
#include <string.h>

#include "ac4_internal.h"

#define MAX_QUANT 8191
#define MAX_EXT_N 8
#define SF_CENTRE 60
#define SNF_CENTRE 17

/* TS 103 190-1 4.3.6.1-4.3.6.2, 44,1 / 48 kHz: Table 100 (partial
   blocks, frame_len_base >= 1536), Table 103 (frame_len_base < 1536; 0 =
   not defined), Table 109 / 110 (n_grp_bits), Table 106 (n_msfb_bits,
   n_side_bits, n_msfbl_bits) */
static int transf_len_long(int fl, int t)
{
    static const int L2048[4] = { 128, 256, 512, 1024 };
    static const int L1920[4] = { 120, 240, 480, 960 };
    static const int L1536[4] = { 96, 192, 384, 768 };
    return fl == 2048 ? L2048[t] : fl == 1920 ? L1920[t] : L1536[t];
}

static int transf_len_short(int fl, int t)
{
    switch (fl) {
    case 1024: return 128 << t;
    case 960: return 120 << t;
    case 768: return 96 << t;
    case 512: return t < 3 ? 128 << t : 0;
    case 384: return t < 3 ? 96 << t : 0;
    }
    return 0;
}

static const int N_GRP_BITS[4][4] = {
    { 15, 10, 8, 7 }, { 10, 7, 4, 3 }, { 8, 4, 3, 1 }, { 7, 3, 1, 1 }
};

static int n_grp_bits_short(int fl, int t)
{
    static const int A[4] = { 7, 3, 1, 0 }, B[3] = { 3, 1, 0 };
    return fl >= 768 ? A[t] : B[t];
}

static const int CB_MOD[12] = { 0, 3, 3, 3, 3, 9, 9, 8, 8, 13, 13, 17 };
static const int CB_OFF[12] = { 0, 1, 1, 0, 0, 4, 4, 0, 0, 0, 0, 0 };
static const int CB_DIM[12] = { 0, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2 };

static const ac4t_sfb *sfb_for(int length)
{
    int i;
    for (i = 0; i < AC4T_N_SFB; i++)
        if (AC4T_SFB[i].length == length)
            return &AC4T_SFB[i];
    return NULL;
}

static int n_msfb_bits(int length)
{
    if (length >= 384)
        return 6;
    if (length >= 192)
        return 5;
    return 4;
}

static int n_side_bits(int length)
{
    if (length >= 480)
        return 5;
    if (length >= 240 || length == 384)
        return 4;
    return 3;
}

static int n_msfbl_bits(int fl)
{
    return fl >= 1536 ? 3 : 2;
}

/* ------------------------------------------------------------ Huffman */
int ac4_huff_build(ac4_huff *h, const ac4t_book *bk)
{
    int i, cap = 1, total = 0;
    memset(h, 0, sizeof(*h));
    for (i = 0; i < bk->n; i++)
        total += bk->len[i];
    cap = total + 2;
    h->node = (int32_t *)calloc((size_t)cap * 2, sizeof(int32_t));
    if (!h->node)
        return -1;
    h->n_nodes = 1;
    h->ok = 1;
    for (i = 0; i < bk->n; i++) {
        int L = bk->len[i], k, node = 0;
        uint32_t code = bk->cw[i];
        if (L <= 0)
            continue;
        for (k = 0; k < L; k++) {
            int bit = (int)((code >> (L - 1 - k)) & 1u);
            int32_t *slot = &h->node[2 * node + bit];
            if (k == L - 1) {
                if (*slot != 0)
                    h->ok = 0;
                *slot = -(i + 1);
            } else {
                if (*slot < 0) {
                    h->ok = 0;
                    break;
                }
                if (*slot == 0) {
                    if (h->n_nodes >= cap) {
                        h->ok = 0;
                        break;
                    }
                    *slot = h->n_nodes++;
                }
                node = *slot;
            }
        }
    }
    return h->ok ? 0 : -1;
}

void ac4_huff_free(ac4_huff *h)
{
    free(h->node);
    h->node = NULL;
}

/* ------------------------------------------------------------ framing */
static int nw0_of(const ac4_framing *f)
{
    return 1 << (3 - f->tl[0]);     /* num_windows_0, Pseudocodes 2, 3 */
}

static void framing_derive(ac4_framing *f)
{
    int g, base = 0, nw0 = 0;
    if (f->different)
        nw0 = nw0_of(f);
    for (g = 0; g < f->num_groups; g++) {
        int half = 0;
        const int *ms = (f->side && f->n_max_sfb_side) ? f->max_sfb_side
                                                        : f->max_sfb;
        int nm = (f->side && f->n_max_sfb_side) ? f->n_max_sfb_side
                                                 : f->n_max_sfb;
        if (f->different)
            half = g >= f->w2g[nw0] ? 1 : 0;
        f->glen[g] = f->lens[half];
        f->gmax[g] = ms[half < nm ? half : nm - 1];
        f->gtab[g] = sfb_for(f->glen[g]);
        f->gbase[g] = base;
        {
            const ac4t_sfb *t = f->gtab[g];
            int m = f->gmax[g] < t->num_sfb ? f->gmax[g] : t->num_sfb;
            base += t->off[m] * f->nwin[g];
        }
    }
}

/* Pseudocode 6: n_sect_bits = 3 when get_transf_length(g) <= 2 */
static int n_sect_bits(const ac4_framing *f, int g)
{
    int half = 0, tlg;
    if (f->fl >= 1536) {
        if (f->is_long)
            return 5;
        if (f->different)
            half = g >= f->w2g[nw0_of(f)] ? 1 : 0;
        tlg = f->tl[half];
    } else {
        tlg = f->tl[0];
    }
    return tlg <= 2 ? 3 : 5;
}

void ac4_framing_parse(ac4_bits *b, ac4_framing *f, int fl)
{
    ac4_framing_parse_ex(b, f, fl, 0, 0);
}

/* asf_transform_info + asf_psy_info (Tables 37, 38; Pseudocodes 2-5)
   for frame length fl. b_dual_maxsfb / b_side_limited: the max_sfb_side
   fields of ASPX_ACPL_1. */
void ac4_framing_parse_ex(ac4_bits *b, ac4_framing *f, int fl,
                          int dual_maxsfb, int side_limited)
{
    int n_grp, i, sfg[AC4_MAX_WIN + 1];
    memset(f, 0, sizeof(*f));
    f->fl = fl;
    if (fl >= 1536) {
        f->is_long = (int)ac4_bit(b);
        if (!f->is_long) {
            f->tl[0] = (int)ac4_u(b, 2);
            f->tl[1] = (int)ac4_u(b, 2);
        }
        f->different = !f->is_long && f->tl[0] != f->tl[1];
        f->lens[0] = f->is_long ? fl : transf_len_long(fl, f->tl[0]);
        f->lens[1] = f->is_long ? fl : transf_len_long(fl, f->tl[1]);
    } else {
        f->tl[0] = (int)ac4_u(b, 2);
        f->lens[0] = f->lens[1] = transf_len_short(fl, f->tl[0]);
        if (!f->lens[0])
            ac4_fail(b, AC4E_TRANSF_LEN);
        f->is_long = f->lens[0] == fl;
        f->different = 0;
    }
    {
        int nh = f->different ? 2 : 1;
        for (i = 0; i < nh; i++) {
            int len = f->lens[i];
            if (side_limited) {
                f->max_sfb_side[i] = (int)ac4_u(b, n_side_bits(len));
                f->max_sfb[i] = f->max_sfb_side[i];
                f->n_max_sfb_side++;
            } else {
                f->max_sfb[i] = (int)ac4_u(b, n_msfb_bits(len));
                if (dual_maxsfb) {
                    f->max_sfb_side[i] = (int)ac4_u(b, n_msfb_bits(len));
                    f->n_max_sfb_side++;
                }
            }
            f->n_max_sfb++;
        }
        for (i = 0; i < f->n_max_sfb; i++)
            if (f->max_sfb[i] > sfb_for(f->lens[i])->num_sfb)
                ac4_fail(b, AC4E_MAX_SFB);
        for (i = 0; i < f->n_max_sfb_side; i++)
            if (f->max_sfb_side[i] > sfb_for(f->lens[i])->num_sfb)
                ac4_fail(b, AC4E_MAX_SFB);
    }
    if (fl >= 1536)
        n_grp = f->is_long ? 0 : N_GRP_BITS[f->tl[0]][f->tl[1]];
    else
        n_grp = n_grp_bits_short(fl, f->tl[0]);
    for (i = 0; i < n_grp; i++)
        sfg[i] = (int)ac4_bit(b);
    /* Pseudocode 3 */
    f->num_windows = 1;
    f->num_groups = 1;
    f->w2g[0] = 0;
    if (!f->is_long) {
        f->num_windows = n_grp + 1;
        if (f->different) {
            int nw0 = nw0_of(f);
            sfg[n_grp] = 0;
            for (i = n_grp; i > nw0 - 1; i--)
                sfg[i] = sfg[i - 1];
            sfg[nw0 - 1] = 0;
            f->num_windows += 1;
        }
        for (i = 0; i < f->num_windows; i++)
            f->w2g[i] = 0;
        for (i = 0; i < f->num_windows - 1; i++) {
            if (sfg[i] == 0)
                f->num_groups += 1;
            f->w2g[i + 1] = f->num_groups - 1;
        }
    }
    for (i = 0; i < f->num_groups; i++)
        f->nwin[i] = 0;
    for (i = 0; i < f->num_windows; i++)
        f->nwin[f->w2g[i]]++;
    framing_derive(f);
}

/* the same framing read for the side channel (Pseudocode 5,
   b_side_channel = 1: its groups use max_sfb_side) */
void ac4_framing_side(const ac4_framing *src, ac4_framing *f)
{
    *f = *src;
    f->side = 1;
    framing_derive(f);
}

static void framing_lfe(ac4_bits *b, ac4_framing *f, int fl)
{
    memset(f, 0, sizeof(*f));
    f->fl = fl;
    f->is_long = 1;
    f->tl[0] = 3;
    f->lens[0] = f->lens[1] = fl;
    f->max_sfb[0] = (int)ac4_u(b, n_msfbl_bits(fl));
    f->n_max_sfb = 1;
    f->num_windows = 1;
    f->num_groups = 1;
    f->nwin[0] = 1;
    framing_derive(f);
}

/* ------------------------------------------------------------ sf_data */
typedef struct {
    int cb;
    int s0, s1;
} sect_t;

static int section_data(ac4_bits *b, int max_sfb, int nbits, uint8_t *sfb_cb,
                        sect_t *sects)
{
    int esc = (1 << nbits) - 1, k = 0, n = 0;
    memset(sfb_cb, 0, AC4_MAX_SFB);
    while (k < max_sfb) {
        int cb = (int)ac4_u(b, 4), sect_len = 1, incr, end, s;
        if (cb > 11)
            ac4_fail(b, AC4E_SECT_CB);
        for (;;) {
            incr = (int)ac4_u(b, nbits);
            if (incr != esc)
                break;
            sect_len += esc;
        }
        sect_len += incr;
        end = k + sect_len;
        if (end > max_sfb)
            ac4_fail(b, AC4E_SECT_LEN);
        for (s = k; s < end; s++)
            sfb_cb[s] = (uint8_t)cb;
        sects[n].cb = cb;
        sects[n].s0 = k;
        sects[n].s1 = end;
        n++;
        k = end;
    }
    return n;
}

static int read_ext(ac4_bits *b)
{
    int n = 0;
    while (ac4_bit(b)) {
        n++;
        if (n > MAX_EXT_N)
            ac4_fail(b, AC4E_QUANT);
    }
    return (1 << (n + 4)) + (int)ac4_u(b, n + 4);
}

/* One group's spectral data into lines[] (absolute packed offsets). */
static void spectral_data(ac4_bits *b, const ac4_books *bk,
                          const sect_t *sects, int n_sect,
                          const ac4_framing *f, int g, int32_t *lines,
                          int total)
{
    const ac4t_sfb *t = f->gtab[g];
    int nw = f->nwin[g], base = f->gbase[g], s, gend;
    /* the group's own end: Python's per-group array clips writes there */
    gend = n_sect ? base + t->off[sects[n_sect - 1].s1 < t->num_sfb
                                  ? sects[n_sect - 1].s1 : t->num_sfb] * nw
                  : 0;
    if (gend > total)
        gend = total;
    for (s = 0; s < n_sect; s++) {
        int cb = sects[s].cb, dim, mod, off, k, end, s0, s1;
        const ac4_huff *h;
        if (cb == 0)
            continue;
        dim = CB_DIM[cb];
        mod = CB_MOD[cb];
        off = CB_OFF[cb];
        h = bk->cb[cb];
        s0 = sects[s].s0 < t->num_sfb ? sects[s].s0 : t->num_sfb;
        s1 = sects[s].s1 < t->num_sfb ? sects[s].s1 : t->num_sfb;
        k = base + t->off[s0] * nw;
        end = base + t->off[s1] * nw;
        while (k < end) {
            int idx = ac4_huff_decode(h, b), v[4], j;
            if (dim == 4) {
                v[0] = idx / (mod * mod * mod) - off;
                v[1] = (idx / (mod * mod)) % mod - off;
                v[2] = (idx / mod) % mod - off;
                v[3] = idx % mod - off;
            } else {
                v[0] = idx / mod - off;
                v[1] = idx % mod - off;
            }
            if (off == 0) {
                for (j = 0; j < dim; j++)
                    if (v[j] && ac4_bit(b))
                        v[j] = -v[j];
            }
            if (cb == 11) {
                for (j = 0; j < dim; j++) {
                    if (v[j] == 16 || v[j] == -16) {
                        int mag = read_ext(b);
                        v[j] = v[j] < 0 ? -mag : mag;
                    }
                }
            }
            for (j = 0; j < dim && k + j < gend; j++) {
                if (v[j] > MAX_QUANT || v[j] < -MAX_QUANT)
                    ac4_fail(b, AC4E_QUANT);
                lines[k + j] = v[j];
            }
            k += dim;
        }
    }
}

/* max |line| of band sfb of group g is non-zero? (band_max > 0) */
static int band_nonzero(const ac4_chan *c, const ac4_framing *f, int g,
                        int sfb)
{
    const ac4t_sfb *t = f->gtab[g];
    int nw = f->nwin[g], base = f->gbase[g], lo, hi, k;
    if (sfb + 1 >= t->num_sfb + 1)
        return 0;
    lo = base + t->off[sfb] * nw;
    hi = base + t->off[sfb + 1] * nw;
    if (hi > c->n_lines)
        hi = c->n_lines;
    for (k = lo; k < hi; k++)
        if (c->lines[k])
            return 1;
    return 0;
}

static void sf_data(ac4_bits *b, const ac4_books *bk, const ac4_framing *f,
                    ac4_chan *c)
{
    uint8_t sfb_cb[AC4_MAX_WIN][AC4_MAX_SFB];
    sect_t sects[AC4_MAX_WIN][AC4_MAX_SFB];
    int n_sect[AC4_MAX_WIN];
    int G = f->num_groups, g, total = 0, cur, first = 0, sfb;
    c->fr = f;
    for (g = 0; g < G; g++)
        n_sect[g] = section_data(b, f->gmax[g], n_sect_bits(f, g),
                                 sfb_cb[g], sects[g]);
    for (g = 0; g < G; g++) {
        const ac4t_sfb *t = f->gtab[g];
        int m = f->gmax[g] < t->num_sfb ? f->gmax[g] : t->num_sfb;
        int e = f->gbase[g] + t->off[m] * f->nwin[g];
        if (e > total)
            total = e;
    }
    if (total > AC4_MAX_FRAME)
        total = AC4_MAX_FRAME;
    c->n_lines = total;
    memset(c->lines, 0, sizeof(int32_t) * (size_t)total);
    for (g = 0; g < G; g++)
        spectral_data(b, bk, sects[g], n_sect[g], f, g, c->lines, total);
    cur = (int)ac4_u(b, 8);                /* reference scale factor */
    for (g = 0; g < G; g++) {
        for (sfb = 0; sfb < f->gmax[g]; sfb++) {
            c->sfs[g][sfb] = AC4_SF_NONE;
            if (sfb_cb[g][sfb] != 0 && band_nonzero(c, f, g, sfb)) {
                if (first) {
                    cur += ac4_huff_decode(bk->sf, b) - SF_CENTRE;
                    if (cur < 0 || cur > 255)
                        ac4_fail(b, AC4E_SF_RANGE);
                } else {
                    first = 1;
                }
                c->sfs[g][sfb] = (int16_t)cur;
            }
        }
    }
    c->has_snf = (int)ac4_bit(b);
    if (c->has_snf) {
        for (g = 0; g < G; g++) {
            for (sfb = 0; sfb < f->gmax[g]; sfb++) {
                c->snf[g][sfb] = AC4_SF_NONE;
                if (sfb_cb[g][sfb] == 0 || !band_nonzero(c, f, g, sfb))
                    c->snf[g][sfb] = (int16_t)(ac4_huff_decode(bk->snf, b)
                                               - SNF_CENTRE);
            }
        }
    }
}

/* ------------------------------------------------------------ stereo */
static void sap_data(ac4_bits *b, const ac4_books *bk, const ac4_framing *f,
                     ac4_stereo *st)
{
    int G = f->num_groups, g, sfb;
    st->coeff_all = (int)ac4_bit(b);
    for (g = 0; g < G; g++) {
        int m = f->gmax[g];
        if (st->coeff_all) {
            for (sfb = 0; sfb < m; sfb++)
                st->coeff_used[g][sfb] = 1;
            continue;
        }
        for (sfb = 0; sfb < m; sfb += 2) {
            uint8_t v = (uint8_t)ac4_bit(b);
            st->coeff_used[g][sfb] = v;
            if (sfb + 1 < m)
                st->coeff_used[g][sfb + 1] = v;
        }
    }
    st->delta_code_time = G != 1 ? (int)ac4_bit(b) : 0;
    for (g = 0; g < G; g++) {
        int m = f->gmax[g];
        for (sfb = 0; sfb < m; sfb++)
            st->dpcm[g][sfb] = AC4_SF_NONE;
        for (sfb = 0; sfb < m; sfb += 2)
            if (st->coeff_used[g][sfb])
                st->dpcm[g][sfb] = (int16_t)ac4_huff_decode(bk->sf, b);
    }
}

static void chparam_info(ac4_bits *b, const ac4_books *bk,
                         const ac4_framing *f, ac4_stereo *st)
{
    int g, sfb;
    st->sap_mode = (int)ac4_u(b, 2);
    if (st->sap_mode == 1) {
        for (g = 0; g < f->num_groups; g++)
            for (sfb = 0; sfb < f->gmax[g]; sfb++)
                st->ms_used[g][sfb] = (uint8_t)ac4_bit(b);
    } else if (st->sap_mode == 3) {
        sap_data(b, bk, f, st);
    }
}

/* Tables B.8 - B.19: max_sfb_master at the largest signalled transform
   length -> the max_sfb of a block of another length */
typedef struct {
    int lmax, n_rows, n_cols, len[4];
    uint8_t v[4][32];
} msfb_master_t;

static const msfb_master_t MSFB_MASTER[] = {
    { 2048, 32, 4, { 1024, 512, 256, 128 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12,
          13, 13, 14, 15, 15, 16, 17, 18, 19, 19, 20, 21 },
        { 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
          9, 10, 10, 11, 12, 13, 14, 15, 16, 17 },
        { 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
          5, 5, 5, 6, 6, 7, 7, 8, 8, 8 },
        { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
          3, 3, 3, 3, 3, 4, 4, 4, 5, 5 } } },
    { 1024, 32, 3, { 512, 256, 128, 0 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 },
        { 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8,
          9, 9, 10, 11, 11, 12, 12, 13, 13, 14 },
        { 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5,
          5, 6, 6, 6, 7, 7, 8, 8, 8, 9 } } },
    { 512, 32, 2, { 256, 128, 0, 0 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
          11, 11, 12, 12, 13, 13, 14, 15, 16, 17, 17 },
        { 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 6,
          7, 7, 8, 8, 9, 9, 10, 10, 11, 12 } } },
    { 256, 16, 1, { 128, 0, 0, 0 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10 } } },
    { 1920, 32, 4, { 960, 480, 240, 120 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12,
          13, 13, 14, 15, 15, 16, 17, 18, 19, 19, 20, 21 },
        { 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
          9, 10, 10, 11, 12, 13, 14, 15, 16, 17 },
        { 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
          5, 5, 5, 6, 6, 7, 7, 8, 8, 8 },
        { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
          3, 3, 3, 3, 3, 4, 4, 4, 5, 5 } } },
    { 960, 32, 3, { 480, 240, 120, 0 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 },
        { 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8,
          9, 9, 10, 11, 11, 12, 12, 13, 13, 14 },
        { 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5,
          5, 6, 6, 6, 7, 7, 8, 8, 8, 9 } } },
    { 480, 32, 2, { 240, 120, 0, 0 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
          11, 11, 12, 12, 13, 13, 14, 15, 16, 17, 17 },
        { 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 6,
          7, 7, 8, 8, 9, 9, 10, 10, 11, 12 } } },
    { 240, 16, 1, { 120, 0, 0, 0 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10 } } },
    { 1536, 32, 4, { 768, 384, 192, 96 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12,
          13, 13, 14, 15, 15, 16, 17, 18, 19, 19, 20, 21 },
        { 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
          9, 10, 10, 11, 12, 13, 14, 15, 16, 17 },
        { 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
          5, 5, 5, 6, 6, 7, 7, 8, 8, 8 },
        { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
          3, 3, 3, 3, 3, 4, 4, 4, 5, 5 } } },
    { 768, 32, 3, { 384, 192, 96, 0 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 },
        { 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 8,
          9, 9, 10, 11, 11, 12, 12, 13, 13, 14 },
        { 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5,
          5, 6, 6, 6, 7, 7, 8, 8, 8, 9 } } },
    { 384, 16, 2, { 192, 96, 0, 0 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 },
        { 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4 } } },
    { 192, 8, 1, { 96, 0, 0, 0 },
      {
        { 0, 1, 1, 2, 2, 3, 3, 4 } } },
};

enum { K_L, K_R, K_C, K_LFE, K_LS, K_RS, K_X1, K_X2 };  /* decoder.c CH_* */

static ac4_stereo *new_st(ac4_bits *b, ac4_element *e)
{
    ac4_stereo *st;
    if (e->n_st >= AC4_MAX_ST)
        ac4_fail(b, AC4E_ELEMENT_PARSE);
    st = &e->st[e->n_st++];
    st->sap_mode = -1;
    return st;
}

static void track(ac4_bits *b, const ac4_books *bk, ac4_element *e, int k)
{
    sf_data(b, bk, &e->fr[k], &e->ch[k]);
    e->has_ch[k] = 1;
    e->order[e->n_order++] = k;
}

static void add_step(ac4_element *e, int kind, int matsel, int st0, int nk,
                     const int *keys)
{
    int i, s;
    if (e->n_steps >= 4)
        return;                           /* cannot happen (3 at most) */
    s = e->n_steps++;
    e->steps[s].kind = kind;
    e->steps[s].nk = nk;
    e->steps[s].matsel = matsel;
    e->steps[s].st0 = st0;
    for (i = 0; i < nk; i++)
        e->steps[s].key[i] = keys[i];
}

/* two_channel_data (Table 26) onto channels k0, k1; stereo_data (Table
   23, spec_frontend bits without stereo processing) when sd */
static void two_ch(ac4_bits *b, const ac4_books *bk, ac4_element *e,
                   int k0, int k1, int sd)
{
    int st0 = e->n_st, keys[2];
    ac4_stereo *st = new_st(b, e);
    keys[0] = k0;
    keys[1] = k1;
    if (ac4_bit(b)) {                     /* b_enable_mdct_stereo_proc */
        ac4_framing_parse(b, &e->fr[k0], bk->fl);
        chparam_info(b, bk, &e->fr[k0], st);
        e->fr[k1] = e->fr[k0];
    } else {
        if (sd && ac4_bit(b) != 0)        /* spec_frontend_l: SSF */
            ac4_fail(b, AC4E_ELEMENT_PARSE);
        ac4_framing_parse(b, &e->fr[k0], bk->fl);
        if (sd && ac4_bit(b) != 0)        /* spec_frontend_r: SSF */
            ac4_fail(b, AC4E_ELEMENT_PARSE);
        ac4_framing_parse(b, &e->fr[k1], bk->fl);
    }
    track(b, bk, e, k0);
    track(b, bk, e, k1);
    add_step(e, 2, 0, st0, 2, keys);
}

/* three / four / five_channel_data (Tables 27 - 32) onto keys[0..n-1] */
static void n_ch(ac4_bits *b, const ac4_books *bk, ac4_element *e, int n,
                 const int *keys)
{
    int i, st0 = e->n_st, matsel = 0;
    ac4_framing_parse(b, &e->fr[keys[0]], bk->fl);
    for (i = 1; i < n; i++)
        e->fr[keys[i]] = e->fr[keys[0]];
    if (n != 4) {
        matsel = (int)ac4_u(b, 4);        /* chel_matsel */
        if (matsel > 11)
            ac4_fail(b, AC4E_CHEL_MATSEL);
    }
    for (i = 0; i < (n == 3 ? 2 : n); i++)
        chparam_info(b, bk, &e->fr[keys[0]], new_st(b, e));
    for (i = 0; i < n; i++)
        track(b, bk, e, keys[i]);
    add_step(e, n, matsel, st0, n, keys);
}

static void mono_data(ac4_bits *b, const ac4_books *bk, int lfe,
                      ac4_element *e, int k)
{
    if (lfe) {
        framing_lfe(b, &e->fr[k], bk->fl);
    } else {
        if (ac4_bit(b) != 0)              /* spec_frontend: SSF */
            ac4_fail(b, AC4E_ELEMENT_PARSE);
        ac4_framing_parse(b, &e->fr[k], bk->fl);
    }
    track(b, bk, e, k);
}

/* the framing of a max_sfb_master track (4.3.5.13): src's transform
   lengths and grouping, each block's max_sfb from max_sfb_master (Tables
   B.8 - B.19) */
static void master_framing(ac4_bits *b, const ac4_framing *src,
                           ac4_framing *f, int master, int lmax)
{
    const msfb_master_t *t = NULL;
    int i, h, nh = src->different ? 2 : 1;
    *f = *src;
    for (i = 0; i < (int)(sizeof(MSFB_MASTER) / sizeof(MSFB_MASTER[0]));
         i++)
        if (MSFB_MASTER[i].lmax == lmax)
            t = &MSFB_MASTER[i];
    f->n_max_sfb_side = nh;
    for (h = 0; h < nh; h++) {
        int len = src->lens[h], m = -1;
        if (len == lmax) {
            m = master;
        } else if (t) {
            for (i = 0; i < t->n_cols; i++)
                if (t->len[i] == len && master < t->n_rows)
                    m = t->v[i][master];
        }
        if (m < 0)
            ac4_fail(b, AC4E_MAX_SFB);
        if (m > sfb_for(len)->num_sfb)
            ac4_fail(b, AC4E_MAX_SFB);
        f->max_sfb_side[h] = m;
    }
    f->side = 1;
    framing_derive(f);
}

static int largest_len(const ac4_framing *a, const ac4_framing *b)
{
    int m = a->lens[0];
    if (a->lens[1] > m) m = a->lens[1];
    if (b->lens[0] > m) m = b->lens[0];
    if (b->lens[1] > m) m = b->lens[1];
    return m;
}
/* Table 49 (aspx_render: flags per channel, sync_flag, b_compand_avg) */
static void companding_control(ac4_bits *b, int num_chan, ac4_element *e)
{
    int sync = num_chan > 1 ? (int)ac4_bit(b) : 0;
    int nc = sync ? 1 : num_chan, i, all = 1;
    e->comp_present = 1;
    e->comp_sync = sync;
    e->comp_n = nc;
    e->comp_avg = 0;
    for (i = 0; i < nc; i++) {
        e->comp_on[i] = (int)ac4_bit(b);
        if (!e->comp_on[i])
            all = 0;
    }
    if (!all)
        e->comp_avg = (int)ac4_bit(b);    /* b_compand_avg */
}

static void audio_size(ac4_bits *b, ac4_element *e)
{
    uint64_t asz = ac4_u(b, 15);
    if (ac4_bit(b)) {
        asz += ac4_vb(b, 7) << 15;
        ac4_align(b);
    }
    e->audio_size = (size_t)(asz > ((uint64_t)1 << 40)
                             ? ((uint64_t)1 << 40) : asz);
    e->audio_start = b->p;
}

static void aspx_config(ac4_bits *b, ac4_aspx_cfg *c)
{
    /* Table 50: 15 fixed bits */
    c->quant_mode_env = (int)ac4_u(b, 1);
    c->start_freq = (int)ac4_u(b, 3);
    c->stop_freq = (int)ac4_u(b, 2);
    c->master_freq_scale = (int)ac4_u(b, 1);
    c->interpolation = (int)ac4_u(b, 1);
    c->preflat = (int)ac4_u(b, 1);
    c->limiter = (int)ac4_u(b, 1);
    c->noise_sbg = (int)ac4_u(b, 2);
    c->num_env_bits_fixfix = (int)ac4_u(b, 1);
    c->freq_res_mode = (int)ac4_u(b, 2);
}

static void acpl_config_2ch(ac4_bits *b, ac4_acpl_cfg *c)
{
    static const int NPB[4] = { 15, 12, 9, 7 };   /* Table 142 */
    c->num_param_bands = NPB[ac4_u(b, 2)];
    c->param_band = 0;
    c->quant_mode_0 = (int)ac4_u(b, 1);
    c->quant_mode_1 = (int)ac4_u(b, 1);
}

/* Table 59; partial = ASPX_ACPL_1 */
static void acpl_config_1ch(ac4_bits *b, int partial, ac4_acpl1_cfg *c)
{
    static const int NPB[4] = { 15, 12, 9, 7 };
    c->num_param_bands = NPB[ac4_u(b, 2)];
    c->quant_mode = (int)ac4_u(b, 1);
    c->qmf_band = c->param_band = 0;
    if (partial) {
        c->qmf_band = (int)ac4_u(b, 3) + 1;
        c->param_band = ac4_acpl_sb_to_pb(c->num_param_bands, c->qmf_band);
    }
}

static void set_blocks(ac4_element *e, int n, int g0, int t0, int g1,
                       int t1, int g2, int t2)
{
    e->blk_grp[3] = 3;
    e->blk_two[3] = 1;
    e->n_blocks = n;
    e->blk_grp[0] = g0;
    e->blk_two[0] = t0;
    e->blk_grp[1] = g1;
    e->blk_two[1] = t1;
    e->blk_grp[2] = g2;
    e->blk_two[2] = t2;
}

static void element_reset(ac4_element *e)
{
    int k;
    e->acpl3 = 0;
    e->acpl1 = AC4_ACPL1_NONE;
    e->n_acpl1 = 0;
    e->r_silent = 0;
    e->comp_present = 0;
    e->has_aspx_cfg = e->has_acpl_cfg = 0;
    e->n_st = e->n_steps = e->n_side = e->n_order = 0;
    e->comp_c = 2;
    for (k = 0; k < AC4_NCH; k++)
        e->has_ch[k] = 0;
}

/* 5_X_channel_element (Table 25): every codec mode and coding_config;
   the track to channel mapping of Tables 180 / 181 */
void ac4_parse_5x(ac4_bits *b, const ac4_books *bk, int b_iframe,
                  ac4_element *e)
{
    ac4_parse_5x_lfe(b, bk, b_iframe, 1, e);
}

void ac4_parse_5x_lfe(ac4_bits *b, const ac4_books *bk, int b_iframe,
                      int has_lfe, ac4_element *e)
{
    element_reset(e);
    audio_size(b, e);
    e->mode = (int)ac4_u(b, 3);
    if (e->mode == 4) {                   /* ASPX_ACPL_3: stereo core */
        e->acpl3 = 1;
        if (b_iframe) {
            aspx_config(b, &e->aspx_cfg);
            acpl_config_2ch(b, &e->acpl_cfg);   /* Table 60 */
            e->has_aspx_cfg = e->has_acpl_cfg = 1;
        }
        if (has_lfe)
            mono_data(b, bk, 1, e, K_LFE);
        companding_control(b, 2, e);
        two_ch(b, bk, e, K_L, K_R, 1);    /* stereo_data */
        set_blocks(e, 1, 0, 1, 0, 0, 0, 0);
        e->bits_end = b->p;
        return;
    }
    if (e->mode == 2 || e->mode == 3) {   /* ASPX_ACPL_1 / _2 */
        int cc;
        e->acpl1 = AC4_ACPL1_5X;
        if (b_iframe) {
            aspx_config(b, &e->aspx_cfg);
            acpl_config_1ch(b, e->mode == 2, &e->acpl1_cfg);
            e->has_aspx_cfg = e->has_acpl_cfg = 1;
        }
        if (has_lfe)
            mono_data(b, bk, 1, e, K_LFE);
        companding_control(b, 3, e);
        cc = (int)ac4_bit(b);             /* coding_config */
        if (cc) {
            static const int K3[3] = { K_L, K_R, K_C };
            n_ch(b, bk, e, 3, K3);        /* A, B, C (Table 181) */
        } else {
            two_ch(b, bk, e, K_L, K_R, 0);    /* A, B */
        }
        if (e->mode == 2) {
            /* max_sfb_master, chparam_info x 2, sf_data x 2: the side
               tracks of A and B (5.3.4.3.2), read with A's / B's framing;
               their chparam_info with A's / B's max_sfb */
            int lmax = largest_len(&e->fr[K_L], &e->fr[K_R]), master, s0;
            master = (int)ac4_u(b, n_side_bits(lmax));
            s0 = e->n_st;
            chparam_info(b, bk, &e->fr[K_L], new_st(b, e));
            chparam_info(b, bk, &e->fr[K_R], new_st(b, e));
            master_framing(b, &e->fr[K_L], &e->fr[K_LS], master, lmax);
            master_framing(b, &e->fr[K_R], &e->fr[K_RS], master, lmax);
            track(b, bk, e, K_LS);
            track(b, bk, e, K_RS);
            e->n_side = 2;
            e->side[0].ka = K_L;
            e->side[0].kb = K_LS;
            e->side[0].st = s0;
            e->side[1].ka = K_R;
            e->side[1].kb = K_RS;
            e->side[1].st = s0 + 1;
        }
        if (!cc)
            mono_data(b, bk, 0, e, K_C);
        set_blocks(e, 2, 0, 1, 2, 0, 0, 0);
        e->n_acpl1 = 2;
        e->bits_end = b->p;
        return;
    }
    if (e->mode > 1)
        ac4_fail(b, AC4E_ELEMENT_PARSE);
    if (b_iframe && e->mode == 1) {
        aspx_config(b, &e->aspx_cfg);
        e->has_aspx_cfg = 1;
    }
    if (has_lfe)
        mono_data(b, bk, 1, e, K_LFE);
    if (e->mode == 1)
        companding_control(b, 5, e);
    switch (ac4_u(b, 2)) {                /* coding_config */
    case 0:
        if (ac4_bit(b)) {                 /* 2ch_mode 1: (L, Ls), (R, Rs) */
            two_ch(b, bk, e, K_L, K_LS, 0);
            two_ch(b, bk, e, K_R, K_RS, 0);
        } else {
            two_ch(b, bk, e, K_L, K_R, 0);
            two_ch(b, bk, e, K_LS, K_RS, 0);
        }
        mono_data(b, bk, 0, e, K_C);
        break;
    case 1: {
        static const int K3[3] = { K_L, K_R, K_C };
        n_ch(b, bk, e, 3, K3);
        two_ch(b, bk, e, K_LS, K_RS, 0);
        break;
    }
    case 2: {
        static const int K4[4] = { K_L, K_R, K_LS, K_RS };
        n_ch(b, bk, e, 4, K4);
        mono_data(b, bk, 0, e, K_C);
        break;
    }
    default: {
        static const int K5[5] = { K_L, K_R, K_C, K_LS, K_RS };
        n_ch(b, bk, e, 5, K5);
        break;
    }
    }
    if (e->mode == 1)
        set_blocks(e, 3, 0, 1, 1, 1, 2, 0);
    else
        set_blocks(e, 0, 0, 0, 0, 0, 0, 0);
    e->bits_end = b->p;
}

void ac4_parse_pair(ac4_bits *b, const ac4_books *bk, int b_iframe,
                    ac4_element *e)
{
    element_reset(e);
    audio_size(b, e);
    e->mode = (int)ac4_u(b, 2);           /* stereo_codec_mode */
    if (b_iframe && e->mode >= 1) {
        aspx_config(b, &e->aspx_cfg);
        e->has_aspx_cfg = 1;
        if (e->mode >= 2) {
            acpl_config_1ch(b, e->mode == 2, &e->acpl1_cfg);
            e->has_acpl_cfg = 1;
        }
    }
    if (e->mode == 1)
        companding_control(b, 2, e);
    else if (e->mode >= 2)
        companding_control(b, 1, e);
    if (e->mode <= 1) {
        two_ch(b, bk, e, K_L, K_R, 1);    /* stereo_data */
        set_blocks(e, e->mode == 1 ? 1 : 0, 0, 1, 0, 0, 0, 0);
    } else if (e->mode == 2) {
        /* ASPX_ACPL_1 (Table 22): the "m" and "s" tracks; their stereo
           processing is done by the decoder (zero-padded, as the Python) */
        ac4_stereo *st = new_st(b, e);
        if (ac4_bit(b)) {                 /* b_enable_mdct_stereo_proc */
            ac4_framing_parse_ex(b, &e->fr[K_L], bk->fl, 1, 0);
            chparam_info(b, bk, &e->fr[K_L], st);
            ac4_framing_side(&e->fr[K_L], &e->fr[K_R]);
        } else {
            ac4_framing f;
            if (ac4_bit(b) != 0)          /* spec_frontend_m: SSF */
                ac4_fail(b, AC4E_ELEMENT_PARSE);
            ac4_framing_parse(b, &e->fr[K_L], bk->fl);
            if (ac4_bit(b) != 0)          /* spec_frontend_s: SSF */
                ac4_fail(b, AC4E_ELEMENT_PARSE);
            ac4_framing_parse_ex(b, &f, bk->fl, 0, 1);
            ac4_framing_side(&f, &e->fr[K_R]);
        }
        track(b, bk, e, K_L);
        track(b, bk, e, K_R);
        set_blocks(e, 1, 0, 0, 0, 0, 0, 0);
        e->acpl1 = AC4_ACPL1_PAIR;
        e->n_acpl1 = 1;
    } else {                              /* ASPX_ACPL_2: one track */
        mono_data(b, bk, 0, e, K_L);
        e->r_silent = 1;
        set_blocks(e, 1, 0, 0, 0, 0, 0, 0);
        e->acpl1 = AC4_ACPL1_PAIR;
        e->n_acpl1 = 1;
    }
    e->bits_end = b->p;
}

/* single_channel_element (Table 20): the track on C */
void ac4_parse_mono(ac4_bits *b, const ac4_books *bk, int b_iframe,
                    ac4_element *e)
{
    element_reset(e);
    audio_size(b, e);
    e->mode = (int)ac4_bit(b);            /* mono_codec_mode */
    if (b_iframe && e->mode == 1) {
        aspx_config(b, &e->aspx_cfg);
        e->has_aspx_cfg = 1;
    }
    if (e->mode == 1)
        companding_control(b, 1, e);
    mono_data(b, bk, 0, e, K_C);
    e->comp_c = 0;
    set_blocks(e, e->mode == 1 ? 1 : 0, 2, 0, 0, 0, 0, 0);
    e->bits_end = b->p;
}

/* 3_0_channel_element (Table 24, 5.3.4.2) */
void ac4_parse_3_0(ac4_bits *b, const ac4_books *bk, int b_iframe,
                   ac4_element *e)
{
    element_reset(e);
    audio_size(b, e);
    e->mode = (int)ac4_bit(b);            /* 3_0_codec_mode */
    if (e->mode == 1) {
        if (b_iframe) {
            aspx_config(b, &e->aspx_cfg);
            e->has_aspx_cfg = 1;
        }
        companding_control(b, 3, e);
    }
    if (ac4_bit(b) == 0) {                /* 3_0_coding_config */
        two_ch(b, bk, e, K_L, K_R, 1);    /* stereo_data */
        mono_data(b, bk, 0, e, K_C);
    } else {
        static const int K3[3] = { K_L, K_R, K_C };
        n_ch(b, bk, e, 3, K3);
    }
    set_blocks(e, e->mode == 1 ? 2 : 0, 0, 1, 2, 0, 0, 0);
    e->bits_end = b->p;
}

/* 7_X_channel_element (Table 33), codec modes SIMPLE and ASPX, ch_mode 5
   - 10: the tracks of A..G (Table 182) go straight to their channels (A
   L, B R, D Ls, E Rs, F X1, G X2); Table 183's pairs, (D, F) / (E, G) for
   3/4/0 and (A, F) / (B, G) otherwise, are the side steps (per window,
   with the b_use_sap_add_ch parameters, read with the base track's
   framing; the two tracks must have the same windows: asf.add_ch) */
void ac4_parse_7x(ac4_bits *b, const ac4_books *bk, int b_iframe,
                  int ch_mode, ac4_element *e)
{
    int has_lfe = ch_mode % 2 == 0, l340 = ch_mode <= 6, cc, i, s0 = -1;
    int base[2];
    element_reset(e);
    audio_size(b, e);
    e->mode = (int)ac4_u(b, 2);           /* 7_X_codec_mode */
    if (e->mode >= 2)                     /* ASPX_ACPL_1 / _2: not built */
        ac4_fail(b, AC4E_ELEMENT_PARSE);
    if (b_iframe && e->mode == 1) {
        aspx_config(b, &e->aspx_cfg);
        e->has_aspx_cfg = 1;
    }
    if (has_lfe)
        mono_data(b, bk, 1, e, K_LFE);
    cc = (int)ac4_u(b, 2);                /* coding_config */
    if (cc == 0) {
        if (ac4_bit(b)) {                 /* 2ch_mode 1: (A, D), (B, E) */
            two_ch(b, bk, e, K_L, K_LS, 0);
            two_ch(b, bk, e, K_R, K_RS, 0);
        } else {
            two_ch(b, bk, e, K_L, K_R, 0);
            two_ch(b, bk, e, K_LS, K_RS, 0);
        }
    } else if (cc == 1) {
        static const int K3[3] = { K_L, K_R, K_C };
        n_ch(b, bk, e, 3, K3);
        two_ch(b, bk, e, K_LS, K_RS, 0);
    } else if (cc == 2) {
        static const int K4[4] = { K_L, K_R, K_LS, K_RS };
        n_ch(b, bk, e, 4, K4);
    } else {
        static const int K5[5] = { K_L, K_R, K_C, K_LS, K_RS };
        n_ch(b, bk, e, 5, K5);
    }
    base[0] = l340 ? K_LS : K_L;
    base[1] = l340 ? K_RS : K_R;
    if (ac4_bit(b)) {                     /* b_use_sap_add_ch */
        s0 = e->n_st;
        for (i = 0; i < 2; i++)
            chparam_info(b, bk, &e->fr[base[i]], new_st(b, e));
    }
    two_ch(b, bk, e, K_X1, K_X2, 0);      /* the additional channels */
    if (cc == 0 || cc == 2)
        mono_data(b, bk, 0, e, K_C);
    if (s0 >= 0) {
        for (i = 0; i < 2; i++) {
            const ac4_framing *fb = &e->fr[base[i]];
            const ac4_framing *fa = &e->fr[i ? K_X2 : K_X1];
            int w;
            if (fb->num_windows != fa->num_windows)
                ac4_fail(b, AC4E_ADD_CH);
            for (w = 0; w < fb->num_windows; w++)
                if (fb->glen[fb->w2g[w]] != fa->glen[fa->w2g[w]])
                    ac4_fail(b, AC4E_ADD_CH);
            e->side[i].ka = base[i];
            e->side[i].kb = i ? K_X2 : K_X1;
            e->side[i].st = s0 + i;
        }
        e->n_side = 2;
    }
    if (e->mode == 1) {
        set_blocks(e, 4, 0, 1, 1, 1, 2, 0);
        e->blk_grp[3] = 3;
        e->blk_two[3] = 1;
    } else {
        set_blocks(e, 0, 0, 0, 0, 0, 0, 0);
    }
    e->bits_end = b->p;
}
