/* SPDX-License-Identifier: Apache-2.0 */
/* The AC-4 table of contents (TS 103 190-2 6.2.1; bitstream_version 0 / 1:
 * TS 103 190-1 4.2.3, presentation_info_v0).
 * Port of ac4/toc.py. Every read is bounds-checked by the bit reader; any
 * failure raises AC4E_TOC_PARSE. */
#include <string.h>

#include "ac4_internal.h"

static const int PROT_PRIMARY[4] = { -1, 8, 32, 128 };
static const int PROT_SECOND[4] = { 0, 8, 32, 128 };

uint64_t ac4_vb(ac4_bits *b, int n)
{
    uint64_t val = 0;
    for (;;) {
        val += ac4_u(b, n);
        if (!ac4_bit(b))
            return val;
        if (val >> 32)
            ac4_fail(b, b->eof_code);
        val <<= n;
        val += (uint64_t)1 << n;
    }
}

/* channel_mode -> ch_mode, TS 103 190-2 table 56: 0 / 10 / 1100..1110 /
   1111000..1111101 (5..10) / 11111100, 11111101 (7.0.4, 7.1.4: 11, 12) /
   111111100..111111110 (9.0.4, 9.1.4, 22.2: 13..15) / 111111111 +
   variable_bits(2) (reserved, 16 and up). 1111110 is not a code: it
   starts the two 8-bit ones. */
long ac4_toc_channel_mode(ac4_bits *b)
{
    uint32_t two, tail, last;
    if (ac4_bit(b) == 0)
        return 0;
    if (ac4_bit(b) == 0)
        return 1;
    two = ac4_u(b, 2);
    if (two != 3)
        return two == 0 ? 2 : two == 1 ? 3 : 4;
    tail = ac4_u(b, 3);
    if (tail <= 5)
        return 5 + (long)tail;
    if (tail == 6)
        return 11 + (long)ac4_bit(b);
    last = ac4_u(b, 2);
    if (last != 3)
        return 13 + (long)last;
    return 16 + (long)ac4_vb(b, 2);
}

/* content_type (TS 103 190-1 Table 10): classifier and language */
static void content_type(ac4_bits *b, ac4_toc_group *g)
{
    g->content_classifier = (int)ac4_u(b, 3);
    if (ac4_bit(b)) {                   /* b_language_indicator */
        if (ac4_bit(b)) {               /* b_serialized_language_tag */
            ac4_bit(b);                 /* b_start_tag */
            ac4_u(b, 16);               /* language_tag_chunk */
            strcpy(g->language, "~");
        } else {
            uint32_t n = ac4_u(b, 6), i;
            for (i = 0; i < n; i++) {
                uint32_t c = ac4_u(b, 8);
                if (i < sizeof(g->language) - 1)
                    g->language[i] = (char)(c >= 32 && c < 127 ? c : '?');
            }
        }
    }
}

static void emdf_info(ac4_bits *b)
{
    uint32_t prim, sec;
    if (ac4_u(b, 2) == 3)               /* emdf_version */
        ac4_vb(b, 2);
    if (ac4_u(b, 3) == 7)               /* key_id */
        ac4_vb(b, 3);
    if (ac4_bit(b)) {                   /* b_emdf_payloads_substream_info */
        if (ac4_u(b, 2) == 3)
            ac4_vb(b, 2);
    }
    prim = ac4_u(b, 2);
    sec = ac4_u(b, 2);
    if (PROT_PRIMARY[prim] < 0)
        ac4_fail(b, b->eof_code);       /* Reserved */
    ac4_skip(b, PROT_PRIMARY[prim]);
    if (PROT_SECOND[sec])
        ac4_skip(b, PROT_SECOND[sec]);
}

static int frame_rate_multiply_info(ac4_bits *b, int fri)
{
    int frf = 1;
    if (fri >= 2 && fri <= 4) {
        if (ac4_bit(b))
            frf = ac4_bit(b) ? 4 : 2;
    } else if (fri == 0 || fri == 1 || (fri >= 7 && fri <= 9)) {
        if (ac4_bit(b))
            frf = 2;
    }
    return frf;
}

/* TS 103 190-2 6.2.1.4 -> frame_rate_fraction */
static int frame_rate_fractions_info(ac4_bits *b, int fri, int frf)
{
    if (fri >= 5 && fri <= 9 && frf == 1) {
        if (ac4_bit(b))
            return 2;
    } else if (fri >= 10 && fri <= 12) {
        if (ac4_bit(b))
            return ac4_bit(b) ? 4 : 2;
    }
    return 1;
}

static void substream_group_info(ac4_bits *b, int frf, ac4_toc *t,
                                 ac4_toc_group *g);

/* 6.2.1.7; bitstream_version 1 (presentation_config 7, phase E): the
   ac4_substream_group_info itself, appended to t's groups */
static long sgi_specifier_t(ac4_bits *b, ac4_toc *t, int frf)
{
    if (t && t->bitstream_version == 1) {
        ac4_toc_group g;
        long gi = t->n_groups_all;
        substream_group_info(b, frf, t, &g);
        if (t->n_groups < AC4_MAX_GROUPS)
            t->groups[t->n_groups++] = g;
        t->n_groups_all++;
        return gi;
    }
    {
        uint64_t g = ac4_u(b, 3);
        if (g == 7)
            g += ac4_vb(b, 2);
        return (long)g;
    }
}

static long substream_index(ac4_bits *b)
{
    uint64_t i = ac4_u(b, 2);
    if (i == 3)
        i += ac4_vb(b, 2);
    return (long)(i > 1000000 ? 1000000 : i);
}

static void presentation_config_ext_info(ac4_bits *b)
{
    uint64_t n = ac4_u(b, 5);
    if (ac4_bit(b))
        n += ac4_vb(b, 2) << 5;
    if (n > b->nbits)
        ac4_fail(b, b->eof_code);
    ac4_skip(b, (int)(8 * n));
}

/* 6.2.1.3; *frf_out receives frame_rate_factor, *multi is set when
   frame_rate_factor or frame_rate_fraction is not 1 */
static void presentation_v1_info_t(ac4_bits *b, int fri, int *frf_out,
                                   int *multi, ac4_toc_pres *p,
                                   long *max_group, ac4_toc *t);

static void presentation_v1_info(ac4_bits *b, int fri, int *frf_out,
                                 int *multi, ac4_toc_pres *p,
                                 long *max_group)
{
    presentation_v1_info_t(b, fri, frf_out, multi, p, max_group, NULL);
}

static void presentation_v1_info_t(ac4_bits *b, int fri, int *frf_out,
                                   int *multi, ac4_toc_pres *p,
                                   long *max_group, ac4_toc *t)
{
    int single = (int)ac4_bit(b);
    uint64_t cfg = 0;
    int add_emdf;
    memset(p, 0, sizeof(*p));
    p->config = -1;
    p->presentation_id = -1;
    p->pres_substream = -1;
    p->enabled = 1;
    p->md_compat = -1;
    if (!single) {
        cfg = ac4_u(b, 3);
        if (cfg == 7)
            cfg += ac4_vb(b, 2);
        p->config = (int)(cfg > 1000 ? 1000 : cfg);
    }
    while (ac4_bit(b))                  /* presentation_version, unary */
        p->version++;
    if (!single && cfg == 6) {
        add_emdf = 1;
    } else {
        int frf;
        uint64_t i, n_spec = 0;
        p->md_compat = (int)ac4_u(b, 3);
        if (ac4_bit(b))                 /* b_presentation_id */
            p->presentation_id = (long)ac4_vb(b, 2);
        frf = frame_rate_multiply_info(b, fri);
        *frf_out = frf;
        if (frame_rate_fractions_info(b, fri, frf) != 1 || frf != 1)
            *multi = 1;
        emdf_info(b);
        if (ac4_bit(b))                 /* b_presentation_filter */
            p->enabled = (int)ac4_bit(b);   /* b_enable_presentation */
        if (single) {
            n_spec = 1;
        } else {
            ac4_bit(b);                 /* b_multi_pid */
            switch (cfg) {
            case 0: case 1: case 2: n_spec = 2; break;
            case 3: case 4: n_spec = 3; break;
            case 5:
                n_spec = ac4_u(b, 2) + 2;
                if (n_spec == 5)
                    n_spec += ac4_vb(b, 2);
                break;
            default:
                presentation_config_ext_info(b);
                break;
            }
        }
        {
            /* n_substream_groups (toc.py presentation_v1_info) */
            static const int NSG[5] = { 2, 1, 2, 3, 2 };
            p->n_sg = single ? 1 : cfg <= 4 ? NSG[cfg]
                      : cfg == 5 ? (long)n_spec : 0;
        }
        for (i = 0; i < n_spec; i++) {
            long g = sgi_specifier_t(b, t, frf);
            if (p->n_groups < AC4_MAX_PGROUPS)
                p->groups[p->n_groups++] = g;
            if (g > *max_group)
                *max_group = g;
        }
        ac4_bit(b);                     /* b_pre_virtualized */
        add_emdf = (int)ac4_bit(b);
        p->b_alternative = (int)ac4_bit(b);
        p->b_pres_ndot = (int)ac4_bit(b);
        p->pres_substream = substream_index(b);
    }
    if (add_emdf) {
        uint64_t n = ac4_u(b, 2), i;
        if (n == 0)
            n = ac4_vb(b, 2) + 4;
        for (i = 0; i < n; i++)
            emdf_info(b);
    }
}

/* the tail of the three substream info elements */
static void sub_tail(ac4_bits *b, int present, int frf, int fs_index,
                     ac4_toc_sub *s)
{
    int i;
    s->sf_multiplier = -1;
    if (fs_index == 1) {
        if (ac4_bit(b))                 /* b_sf_multiplier */
            s->sf_multiplier = (int)ac4_bit(b);
    }
    if (ac4_bit(b)) {                   /* b_bitrate_info */
        if (ac4_u(b, 3) == 7)
            ac4_u(b, 2);
    }
    if (s->kind == 0 && s->channel_mode >= 7 && s->channel_mode <= 10)
        ac4_bit(b);                     /* add_ch_base */
    for (i = 0; i < frf; i++) {
        int v = (int)ac4_bit(b);        /* b_audio_ndot */
        if (i == 0)
            s->ndot = v;
    }
    s->index = present ? substream_index(b) : -1;
}

static void info_chan(ac4_bits *b, int present, int frf, ac4_toc *t,
                      ac4_toc_sub *s)
{
    long cm = ac4_toc_channel_mode(b);
    memset(s, 0, sizeof(*s));
    s->kind = 0;
    s->channel_mode = cm;
    s->back4 = s->centre = 1;
    s->top = 3;
    if (t->n_cmodes < AC4_MAX_CMODES)
        t->cmodes[t->n_cmodes] = cm;
    if (t->n_cmodes < 1 << 30)
        t->n_cmodes++;
    /* the spec compares the channel_mode code with 0b11111100,
       0b11111101, 0b111111100 and 0b111111101: ch_mode 11 .. 14 */
    if (cm >= 11 && cm <= 14) {
        s->back4 = (int)ac4_bit(b);     /* b_4_back_channels_present */
        s->centre = (int)ac4_bit(b);    /* b_centre_present */
        s->top = (int)ac4_u(b, 2);      /* top_channels_present */
    }
    sub_tail(b, present, frf, t->fs_index, s);
}

void ac4_toc_substream_info_chan(ac4_bits *b, int present, int frf,
                                 ac4_toc *t)
{
    ac4_toc_sub s;
    info_chan(b, present, frf, t, &s);
}

static const int ISF_N[6] = { 4, 8, 10, 14, 15, 30 };
static const int BED_AJOC_N[8] = { 2, 3, 5, 7, 9, 7, 9, 11 };

/* toc.obj_assignment: the type of each of n_signals signals (types[]:
   max entries kept) */
static void obj_assignment(ac4_bits *b, long n_signals, signed char *types,
                           int max)
{
    /* Tables 63 - 66 as render ids (render.c RN_*) */
    static const signed char BED[8][11] = {
        { 0, 1 }, { 0, 1, 2 }, { 0, 1, 2, 4, 5 },
        { 0, 1, 2, 4, 5, 12, 13 }, { 0, 1, 2, 4, 5, 8, 9, 10, 11 },
        { 0, 1, 2, 4, 5, 6, 7 }, { 0, 1, 2, 4, 5, 6, 7, 12, 13 },
        { 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11 } };
    /* Table 64 / 66 order: L R C LFE Ls Rs Lb Rb Tfl Tfr Tsl Tsr Tbl Tbr
       Lw Rw LFE2 (RN_LFE for the LFEs) */
    static const signed char NONSTD[17] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                            12, 13, 10, 11, 14, 15, 3 };
    /* Table 65 */
    static const signed char STD[10][2] = { { 0, 1 }, { 2, -9 }, { 3, -9 },
        { 4, 5 }, { 6, 7 }, { 8, 9 }, { 12, 13 }, { 10, 11 }, { 14, 15 },
        { 3, -9 } };
    long n = 0, i;
#define PUT(t) do { if (n < max) types[n] = (signed char)(t); n++; } while (0)
    if (!ac4_bit(b)) {                  /* b_dyn_objects_only */
        if (ac4_bit(b)) {               /* b_isf */
            int c = (int)ac4_u(b, 3);
            if (c > 5)
                ac4_fail(b, b->eof_code);
            for (i = 0; i < ISF_N[c]; i++)
                PUT(AC4_OBJ_ISF);
        } else if (ac4_bit(b)) {        /* b_ch_assign_code */
            int c = (int)ac4_u(b, 3);
            for (i = 0; i < BED_AJOC_N[c]; i++)
                PUT(BED[c][i]);
        } else if (ac4_bit(b)) {        /* flags present */
            if (ac4_bit(b)) {
                uint32_t fl = ac4_u(b, 17);
                for (i = 0; i < 17; i++)
                    if ((fl >> (16 - i)) & 1 && i != 3 && i != 16)
                        PUT(NONSTD[i]);
            } else {
                uint32_t fl = ac4_u(b, 10);
                for (i = 0; i < 10; i++)
                    if ((fl >> (9 - i)) & 1 && i != 2 && i != 9) {
                        PUT(STD[i][0]);
                        if (STD[i][1] >= 0)
                            PUT(STD[i][1]);
                    }
            }
        } else {
            long n_bed = 1;
            if (n_signals > 1) {
                int nb = 0;
                while ((1L << nb) < n_signals)
                    nb++;
                n_bed = (long)ac4_u(b, nb) + 1;
            }
            for (i = 0; i < n_bed; i++) {
                int v = (int)ac4_u(b, 4);
                if (v != 3)
                    PUT(NONSTD[v]);
            }
        }
    }
    for (; n < n_signals && n < max; n++)
        types[n] = AC4_OBJ_DYN;
#undef PUT
}

static void oamd_common_data(ac4_bits *b)
{
    if (ac4_bit(b) == 0)                /* b_default_screen_size_ratio */
        ac4_u(b, 5);
    ac4_bit(b);                         /* b_bed_object_chan_distribute */
    if (ac4_bit(b)) {                   /* b_additional_data */
        uint64_t n = ac4_u(b, 1) + 1;
        if (n == 2)
            n += ac4_vb(b, 2);
        if (n > b->nbits)
            ac4_fail(b, b->eof_code);
        ac4_skip(b, (int)(8 * n));
    }
}

static void info_ajoc(ac4_bits *b, int present, int frf, ac4_toc *t,
                      ac4_toc_sub *s)
{
    uint64_t n;
    memset(s, 0, sizeof(*s));
    s->kind = 1;
    s->channel_mode = -1;
    s->lfe = (int)ac4_bit(b);           /* b_lfe */
    s->static_dmx = (int)ac4_bit(b);    /* b_static_dmx */
    if (!s->static_dmx) {
        n = ac4_u(b, 4) + 1;
        s->n_dmx = (int)n;
        obj_assignment(b, (long)n, s->dmx_type, 16);
    } else {
        static const signed char ST[5] = { 0, 1, 2, 4, 5 };
        s->n_dmx = 5;
        memcpy(s->dmx_type, ST, 5);
    }
    if (ac4_bit(b))                     /* b_oamd_common_data_present */
        oamd_common_data(b);
    n = ac4_u(b, 4) + 1;
    if (n == 16)
        n += ac4_vb(b, 3);
    s->n_umx = (long)(n > 1000 ? 1000 : n);
    obj_assignment(b, s->n_umx, s->umx_type, 64);
    sub_tail(b, present, frf, t->fs_index, s);
}

/* the bed channels of a direct-coded bed (Tables 63 - 66 with the LFE,
   render ids) */
static const signed char OBJ_NONSTD[17] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                            12, 13, 10, 11, 14, 15, 3 };
static const signed char OBJ_STD[10][2] = { { 0, 1 }, { 2, -9 }, { 3, -9 },
    { 4, 5 }, { 6, 7 }, { 8, 9 }, { 12, 13 }, { 10, 11 }, { 14, 15 },
    { 3, -9 } };

static void info_obj(ac4_bits *b, int present, int frf, ac4_toc *t,
                     ac4_toc_sub *s)
{
    static const int NOBJ[6] = { 0, 1, 2, 3, 5, 7 };
    static const signed char BED[8][12] = {
        { 0, 1 }, { 0, 1, 2 }, { 0, 1, 2, 3, 4, 5 },
        { 0, 1, 2, 3, 4, 5, 12, 13 }, { 0, 1, 2, 3, 4, 5, 8, 9, 10, 11 },
        { 0, 1, 2, 3, 4, 5, 6, 7 }, { 0, 1, 2, 3, 4, 5, 6, 7, 12, 13 },
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 } };
    static const int BED_N[8] = { 2, 3, 6, 8, 10, 8, 10, 12 };
    int code, i;
    memset(s, 0, sizeof(*s));
    s->kind = 2;
    s->channel_mode = -1;
    s->n_assign = -1;
    code = (int)ac4_u(b, 3);            /* n_objects_code */
    if (code > 5)
        ac4_fail(b, b->eof_code);
    s->n_objects = NOBJ[code];
    if (ac4_bit(b)) {                   /* b_dynamic_objects */
        s->objects = 0;
        s->dyn_lfe = (int)ac4_bit(b);   /* b_lfe */
    } else if (ac4_bit(b)) {            /* b_bed_objects */
        s->objects = 1;
        if (ac4_bit(b)) {               /* b_bed_start */
            s->n_assign = 0;
            if (ac4_bit(b)) {           /* b_ch_assign_code */
                int c = (int)ac4_u(b, 3);
                for (i = 0; i < BED_N[c]; i++)
                    s->assign[s->n_assign++] = BED[c][i];
            } else if (ac4_bit(b)) {
                uint32_t fl = ac4_u(b, 17);
                for (i = 0; i < 17; i++)
                    if ((fl >> (16 - i)) & 1)
                        s->assign[s->n_assign++] = OBJ_NONSTD[i];
            } else {
                uint32_t fl = ac4_u(b, 10);
                for (i = 0; i < 10; i++)
                    if ((fl >> (9 - i)) & 1) {
                        s->assign[s->n_assign++] = OBJ_STD[i][0];
                        if (OBJ_STD[i][1] >= 0)
                            s->assign[s->n_assign++] = OBJ_STD[i][1];
                    }
            }
        }
    } else if (ac4_bit(b)) {            /* b_isf */
        s->objects = 2;
        if (ac4_bit(b)) {               /* b_isf_start */
            int c = (int)ac4_u(b, 3);
            if (c > 5)
                ac4_fail(b, b->eof_code);
            s->isf_cfg = c;
            s->n_assign = ISF_N[c];
            for (i = 0; i < ISF_N[c]; i++)
                s->assign[i] = (signed char)AC4_OBJ_ISF_I(i);
        }
    } else {
        s->objects = 3;
        ac4_skip(b, 8 * (int)ac4_u(b, 4));   /* res_bytes */
    }
    sub_tail(b, present, frf, t->fs_index, s);
}

/* toc.obj_split: each direct-coded substream's objects, the group's */
static void obj_split(ac4_toc_group *g)
{
    signed char pend[32];
    int np = 0, j, i, isf_cfg = 0;
    g->n_obj = 0;
    g->bed_mask = 0;
    for (j = 0; j < g->n_subs; j++) {
        const ac4_toc_sub *s = &g->subs[j];
        if (s->kind == 2 && s->objects == 1)
            for (i = 0; i < s->n_assign; i++)
                g->bed_mask |= 1u << s->assign[i];
    }
#define PUTO(t, l, a) do { if (g->n_obj < 64) { \
        g->obj_type[g->n_obj] = (signed char)(t); \
        g->obj_lfe[g->n_obj] = (unsigned char)(l); \
        g->obj_ajoc[g->n_obj] = (unsigned char)(a); } g->n_obj++; } while (0)
    for (j = 0; j < g->n_subs; j++) {
        ac4_toc_sub *s = &g->subs[j];
        int n = s->n_objects, k = 0;
        if (s->kind == 1) {
            long m = s->n_umx + (s->lfe ? 1 : 0), q;
            for (q = 0; q < m; q++)
                PUTO(AC4_OBJ_DYN, 0, 1);
            continue;
        }
        if (s->kind != 2)
            continue;
        s->n_types = 0;
#define OWN(t) do { if (s->n_types < 8) s->types[s->n_types] = \
        (signed char)(t); s->n_types++; } while (0)
        if (s->objects == 0) {
            if (s->dyn_lfe)
                OWN(RN_LFE);
            for (i = 0; i < n; i++)
                OWN(AC4_OBJ_DYN);
        } else if (s->objects == 1 || s->objects == 2) {
            signed char rest[32];
            int nr = 0, lfe_taken = 0;
            if (s->n_assign >= 0) {
                np = s->n_assign;
                memcpy(pend, s->assign, (size_t)np);
                if (s->objects == 2)
                    isf_cfg = s->isf_cfg;
            }
            for (i = 0; i < np; i++) {
                int t = pend[i], is_lfe = t == RN_LFE;
                if (is_lfe && !lfe_taken) {
                    OWN(t);
                    lfe_taken = 1;
                } else if (!is_lfe && k < n) {
                    OWN(t);
                    k++;
                } else {
                    rest[nr++] = (signed char)t;
                }
            }
            memcpy(pend, rest, (size_t)nr);
            np = nr;
            for (; k < n; k++)
                OWN(AC4_OBJ_DYN);
            if (s->objects == 2)
                s->isf_cfg = isf_cfg;
        } else {
            for (i = 0; i < n; i++)
                OWN(AC4_OBJ_DYN);
        }
#undef OWN
        if (s->n_types > 8)
            s->n_types = 8;      /* more than a substream codes: unused */
        s->lfe = 0;
        for (i = 0; i < s->n_types; i++)
            if (s->types[i] == RN_LFE)
                s->lfe = 1;
        s->oamd_offset = g->n_obj;
        s->obj_bed = g->bed_mask;
        for (i = 0; i < s->n_types; i++) {
            int t = s->types[i];
            PUTO(t <= AC4_OBJ_ISF_I(0) ? AC4_OBJ_ISF : t, t == RN_LFE, 0);
        }
    }
#undef PUTO
}

static void substream_group_info(ac4_bits *b, int frf, ac4_toc *t,
                                 ac4_toc_group *g)
{
    int present = (int)ac4_bit(b);      /* b_substreams_present */
    int hsf = (int)ac4_bit(b);          /* b_hsf_ext */
    int single = (int)ac4_bit(b);       /* b_single_substream */
    uint64_t n_lf, i;
    memset(g, 0, sizeof(*g));
    g->substreams_present = present;
    g->hsf_ext = hsf;
    g->content_classifier = -1;
    if (single) {
        n_lf = 1;
    } else {
        n_lf = ac4_u(b, 2) + 2;
        if (n_lf == 5)
            n_lf += ac4_vb(b, 2);
    }
    g->n_subs_all = (long)(n_lf > 1000000 ? 1000000 : n_lf);
    g->channel_coded = (int)ac4_bit(b);
    if (!g->channel_coded) {
        g->oamd_index = -1;
        if (ac4_bit(b)) {               /* b_oamd_substream */
            g->has_oamd = 1;
            g->oamd_ndot = (int)ac4_bit(b);   /* b_oamd_ndot */
            if (present)
                g->oamd_index = substream_index(b);
        }
    }
    for (i = 0; i < n_lf; i++) {
        ac4_toc_sub s;
        if (g->channel_coded) {
            int sv = t->bitstream_version == 1 ? (int)ac4_bit(b) : 1;
            info_chan(b, present, frf, t, &s);
            s.sus_ver = sv;
        }
        else if (ac4_bit(b))            /* b_ajoc */
            info_ajoc(b, present, frf, t, &s);
        else
            info_obj(b, present, frf, t, &s);
        if (hsf && present)
            substream_index(b);         /* ac4_hsf_ext_substream_info */
        if (g->n_subs < AC4_MAX_GSUBS)
            g->subs[g->n_subs++] = s;
    }
    if (!g->channel_coded)
        obj_split(g);
    if (ac4_bit(b))                     /* b_content_type */
        content_type(b, g);
}

/* ac4_substream_info (TS 103 190-1 Table 9; toc.substream_info_v0):
   bitstream_version 0 / 1, one substream group of its own */
static void substream_info_v0(ac4_bits *b, int frf, ac4_toc *t,
                              ac4_toc_group *g)
{
    ac4_toc_sub *s = &g->subs[0];
    long cm = ac4_toc_channel_mode(b);
    int i;
    memset(g, 0, sizeof(*g));
    g->substreams_present = 1;
    g->channel_coded = 1;
    g->content_classifier = -1;
    g->n_subs = 1;
    g->n_subs_all = 1;
    s->kind = 0;
    s->channel_mode = cm;
    if (t->n_cmodes < AC4_MAX_CMODES)
        t->cmodes[t->n_cmodes] = cm;
    if (t->n_cmodes < 1 << 30)
        t->n_cmodes++;
    s->sf_multiplier = -1;
    if (t->fs_index == 1 && ac4_bit(b))         /* b_sf_multiplier */
        s->sf_multiplier = (int)ac4_bit(b);
    if (ac4_bit(b)) {                   /* b_bitrate_info */
        if (ac4_u(b, 3) == 7)
            ac4_u(b, 2);
    }
    if (cm >= 7 && cm <= 10)
        ac4_bit(b);                     /* add_ch_base */
    if (ac4_bit(b))                     /* b_content_type */
        content_type(b, g);
    for (i = 0; i < frf; i++) {
        int v = (int)ac4_bit(b);        /* b_iframe */
        if (i == 0)
            s->ndot = v;
    }
    s->index = substream_index(b);
}

/* ac4_presentation_info (TS 103 190-1 Table 5; TS 103 190-2 6.2.1.2;
   toc.presentation_info_v0) */
static void presentation_info_v0(ac4_bits *b, int fri, int *frf_out,
                                 int *multi, ac4_toc *t, ac4_toc_pres *p)
{
    static const int NSUB[6] = { 2, 2, 2, 3, 3, 1 };
    int single = (int)ac4_bit(b), add_emdf;
    uint64_t cfg = 0;
    memset(p, 0, sizeof(*p));
    p->config = -1;
    p->presentation_id = -1;
    p->pres_substream = -1;
    p->enabled = 1;
    p->md_compat = -1;
    p->v0 = 1;
    if (!single) {
        cfg = ac4_u(b, 3);
        if (cfg == 7)
            cfg += ac4_vb(b, 2);
        p->config = (int)(cfg > 1000 ? 1000 : cfg);
    }
    while (ac4_bit(b))                  /* presentation_version */
        p->version++;
    if (!single && cfg == 6) {
        add_emdf = 1;
    } else {
        int frf, n, hsf = 0, i;
        p->md_compat = (int)ac4_u(b, 3);
        if (ac4_bit(b))                 /* b_presentation_id */
            p->presentation_id = (long)ac4_vb(b, 2);
        frf = frame_rate_multiply_info(b, fri);
        *frf_out = frf;
        if (frf != 1)
            *multi = 1;
        emdf_info(b);
        n = single ? 1 : cfg <= 5 ? NSUB[cfg] : 0;
        if (!single)
            hsf = (int)ac4_bit(b);      /* b_hsf_ext */
        if (!single && cfg == 7 && t->bitstream_version == 1) {
            /* presentation_config_ext_info (6.2.1.5): the
               ac4_presentation_v1_info it holds is the presentation
               (toc.presentation_info_v0, phase E) */
            uint64_t nskip = ac4_u(b, 5);
            size_t p0, nread;
            long mg = -1;
            int frf1 = 1;
            if (ac4_bit(b))
                nskip += ac4_vb(b, 2) << 5;
            p0 = b->p;
            presentation_v1_info_t(b, fri, &frf1, multi, p, &mg, t);
            *frf_out = frf1;
            nread = b->p - p0;
            if (nread % 8) {
                ac4_skip(b, (int)(8 - nread % 8));
                nread += 8 - nread % 8;
            }
            if ((uint64_t)(nread / 8) > nskip)
                ac4_fail(b, b->eof_code);
            nskip -= nread / 8;
            if (nskip > b->nbits)
                ac4_fail(b, b->eof_code);
            ac4_skip(b, (int)(8 * nskip));
            ac4_bit(b);                 /* b_pre_virtualized */
            if (ac4_bit(b)) {           /* b_add_emdf_substreams */
                uint64_t k = ac4_u(b, 2), j;
                if (k == 0)
                    k = ac4_vb(b, 2) + 4;
                for (j = 0; j < k; j++)
                    emdf_info(b);
            }
            return;
        }
        if (!single && cfg > 5)
            presentation_config_ext_info(b);
        for (i = 0; i < n; i++) {
            ac4_toc_group g;
            substream_info_v0(b, frf, t, &g);
            if (i == 0 && hsf) {
                substream_index(b);     /* ac4_hsf_ext_substream_info */
                g.hsf_ext = 1;
            }
            if (p->n_groups < AC4_MAX_PGROUPS)
                p->groups[p->n_groups++] = t->n_groups_all;
            if (t->n_groups < AC4_MAX_GROUPS)
                t->groups[t->n_groups++] = g;
            t->n_groups_all++;
        }
        p->n_sg = n;
        ac4_bit(b);                     /* b_pre_virtualized */
        add_emdf = (int)ac4_bit(b);
    }
    if (add_emdf) {
        uint64_t n = ac4_u(b, 2), i;
        if (n == 0)
            n = ac4_vb(b, 2) + 4;
        for (i = 0; i < n; i++)
            emdf_info(b);
    }
}

void ac4_toc_parse(ac4_bits *b, ac4_toc *t)
{
    uint64_t bv, n_pres, i, n_sub;
    long max_group = -1;
    int fri, frf = 1, size_present;
    memset(t, 0, sizeof(*t));
    t->sequence_counter = -1;
    bv = ac4_u(b, 2);
    if (bv == 3)
        bv += ac4_vb(b, 2);
    t->bitstream_version = (int)(bv > 1000 ? 1000 : bv);
    t->sequence_counter = (int)ac4_u(b, 10);
    if (ac4_bit(b)) {                   /* b_wait_frames */
        if (ac4_u(b, 3) > 0)
            ac4_u(b, 2);                /* br_code */
    }
    t->fs_index = (int)ac4_bit(b);
    fri = (int)ac4_u(b, 4);
    t->frame_rate_index = fri;
    t->b_iframe_global = (int)ac4_bit(b);
    if (ac4_bit(b))                     /* b_single_presentation */
        n_pres = 1;
    else
        n_pres = ac4_bit(b) ? ac4_vb(b, 2) + 2 : 0;
    t->n_presentations = (long)n_pres;
    if (ac4_bit(b)) {                   /* b_payload_base */
        uint64_t pb = ac4_u(b, 5) + 1;
        if (pb == 0x20)
            pb += ac4_vb(b, 3);
        t->payload_base = (long)pb;
    }
    if (bv <= 1) {
        /* TS 103 190-1 4.2.3: ac4_presentation_info, substream infos
           inline (toc.presentation_info_v0) */
        for (i = 0; i < n_pres; i++) {
            ac4_toc_pres p;
            presentation_info_v0(b, fri, &frf, &t->rate_multi, t, &p);
            if (t->n_pres < AC4_MAX_PRES)
                t->pres[t->n_pres++] = p;
        }
    } else {
        if (ac4_bit(b)) {               /* b_program_id */
            ac4_u(b, 16);
            if (ac4_bit(b))             /* b_program_uuid_present */
                ac4_skip(b, 128);
        }
        for (i = 0; i < n_pres; i++) {
            ac4_toc_pres p;
            presentation_v1_info(b, fri, &frf, &t->rate_multi, &p,
                                 &max_group);
            if (t->n_pres < AC4_MAX_PRES)
                t->pres[t->n_pres++] = p;
        }
        /* total_n_substream_groups = 1 + max_group_index (6.3.2.1.8) */
        t->n_groups_all = max_group + 1;
        for (i = 0; i < (uint64_t)(max_group + 1); i++) {
            ac4_toc_group g;
            substream_group_info(b, frf, t, &g);
            if (t->n_groups < AC4_MAX_GROUPS)
                t->groups[t->n_groups++] = g;
        }
    }
    n_sub = ac4_u(b, 2);                /* substream_index_table */
    if (n_sub == 0)
        n_sub = ac4_vb(b, 2) + 4;
    t->n_substreams = (long)n_sub;
    size_present = n_sub == 1 ? (int)ac4_bit(b) : 1;
    t->sizes_min = -1;
    if (size_present) {
        for (i = 0; i < n_sub; i++) {
            uint64_t s;
            int more = (int)ac4_bit(b);
            s = ac4_u(b, 10);
            if (more)
                s += ac4_vb(b, 2) << 10;
            if (i < AC4_MAX_SIZES)
                t->sizes[i] = (long)s;
            t->sizes_sum += (double)s;
            if (t->sizes_min < 0 || (long)s < t->sizes_min)
                t->sizes_min = (long)s;
        }
        t->n_sizes = (int)(n_sub > 1000000 ? 1000000 : n_sub);
    }
    ac4_align(b);
    t->toc_bytes = b->p / 8;
    if (!size_present) {
        /* one substream without a size: the rest of the frame */
        double r = (double)(b->nbits / 8) - (double)t->toc_bytes
                   - (double)t->payload_base;
        t->sizes[0] = r > 0 ? (long)r : 0;
        t->sizes_sum = (double)t->sizes[0];
        t->sizes_min = t->sizes[0];
        t->n_sizes = 1;
    }
}

/* ------------------------------------------------------------ choice */

int ac4_toc_audio_substreams(const ac4_toc *t, int pi,
                             const ac4_toc_sub **out, long *grp, int max)
{
    int k, j, n = 0;
    long de = -1;
    const ac4_toc_pres *p;
    if (pi < 0 || pi >= t->n_pres)
        return 0;
    p = &t->pres[pi];
    if ((p->config == 1 || p->config == 4) && p->n_groups > 1)
        de = p->groups[1];
    for (k = 0; k < p->n_groups; k++) {
        long gi = p->groups[k];
        const ac4_toc_group *g;
        if (gi < 0 || gi >= t->n_groups || gi == de)
            continue;
        g = &t->groups[gi];
        for (j = 0; j < g->n_subs; j++) {
            const ac4_toc_sub *s = &g->subs[j];
            if ((s->kind == 0 || s->kind == 2) && s->index >= 0
                && s->index < t->n_sizes
                && s->index < AC4_MAX_SIZES) {
                if (n < max) {
                    out[n] = s;
                    if (grp)
                        grp[n] = gi;
                }
                n++;
            }
        }
    }
    if (n == 0) {
        /* no channel-coded substream: the A-JOC ones, in core decoding
           mode (toc.AJOC_CORE) */
        for (k = 0; k < p->n_groups; k++) {
            long gi = p->groups[k];
            const ac4_toc_group *g;
            if (gi < 0 || gi >= t->n_groups || gi == de)
                continue;
            g = &t->groups[gi];
            for (j = 0; j < g->n_subs; j++) {
                const ac4_toc_sub *s = &g->subs[j];
                if (s->kind == 1 && s->index >= 0 && s->index < t->n_sizes
                    && s->index < AC4_MAX_SIZES) {
                    if (n < max) {
                        out[n] = s;
                        if (grp)
                            grp[n] = gi;
                    }
                    n++;
                }
            }
        }
    }
    return n < max ? n : max;
}

const ac4_toc_sub *ac4_toc_audio_substream(const ac4_toc *t, int pi)
{
    const ac4_toc_sub *s[1];
    return ac4_toc_audio_substreams(t, pi, s, NULL, 1) ? s[0] : NULL;
}

int ac4_toc_substream_type(const ac4_toc *t, int pi, long gi)
{
    /* Table 53 (0 M&E / main, 1 dialogue, 2 associated, 3 DE) and Table
       54 (by content_classifier) */
    static const int CFG[5][3] = { { 0, 1, 1 }, { 0, 3, 3 }, { 0, 2, 2 },
                                   { 0, 1, 2 }, { 0, 3, 2 } };
    static const int CLS[8] = { 0, 0, 2, 2, 1, 2, 0, 0 };
    const ac4_toc_pres *p = &t->pres[pi];
    int pos = 0, k;
    if (p->config < 0)
        return 0;
    for (k = 0; k < p->n_groups; k++)
        if (p->groups[k] == gi) {
            pos = k;
            break;
        }
    if (p->config <= 4)
        return CFG[p->config][pos < 2 ? pos : 2];
    if (gi >= 0 && gi < t->n_groups && t->groups[gi].content_classifier >= 0)
        return CLS[t->groups[gi].content_classifier & 7];
    return 0;
}

static const char *pres_language(const ac4_toc *t, int pi)
{
    int k;
    for (k = 0; k < t->pres[pi].n_groups; k++) {
        long gi = t->pres[pi].groups[k];
        if (gi >= 0 && gi < t->n_groups && t->groups[gi].language[0])
            return t->groups[gi].language;
    }
    return NULL;
}

/* no content_type, or "complete main" among its groups */
static int pres_main(const ac4_toc *t, int pi)
{
    int k, any = 0;
    for (k = 0; k < t->pres[pi].n_groups; k++) {
        long gi = t->pres[pi].groups[k];
        if (gi < 0 || gi >= t->n_groups)
            continue;
        if (t->groups[gi].content_classifier == 0)
            return 1;
        if (t->groups[gi].content_classifier >= 0)
            any = 1;
    }
    return !any;
}

static int lang_eq(const char *a, const char *b)
{
    /* the primary subtags, case-insensitive */
    for (;; a++, b++) {
        int ca = (*a == '-' || !*a) ? 0 : (*a | 0x20);
        int cb = (*b == '-' || !*b) ? 0 : (*b | 0x20);
        if (ca != cb)
            return 0;
        if (!ca)
            return 1;
    }
}

int ac4_toc_choose(const ac4_toc *t, int index, long id, const char *language)
{
    int pi, pass;
    if (index >= 0)
        return index < t->n_pres ? index : -1;
    if (id >= 0) {
        for (pi = 0; pi < t->n_pres; pi++)
            if (t->pres[pi].presentation_id == id)
                return pi;
        return -1;
    }
    /* pass 0: language and main; 1: language; 2: main; 3: any. The
       language passes only when some candidate matches it (as toc.py). */
    for (pass = 0; pass < 4; pass++) {
        int want_lang = pass < 2, want_main = pass == 0 || pass == 2;
        if (want_lang && !(language && language[0]))
            continue;
        if (want_lang) {
            int hit = 0;
            for (pi = 0; pi < t->n_pres; pi++) {
                const char *l = pres_language(t, pi);
                if (t->pres[pi].enabled && ac4_toc_audio_substream(t, pi)
                    && l && lang_eq(l, language))
                    hit = 1;
            }
            if (!hit)
                continue;
        }
        for (pi = 0; pi < t->n_pres; pi++) {
            const char *l = pres_language(t, pi);
            if (!t->pres[pi].enabled || !ac4_toc_audio_substream(t, pi))
                continue;
            if (want_lang && !(l && lang_eq(l, language)))
                continue;
            if (want_main && !pres_main(t, pi))
                continue;
            return pi;
        }
    }
    return -1;
}
