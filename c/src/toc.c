/* SPDX-License-Identifier: Apache-2.0 */
/* The AC-4 table of contents, bitstream_version >= 2 (TS 103 190-2 6.2.1).
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

static long sgi_specifier(ac4_bits *b)
{
    uint64_t g = ac4_u(b, 3);
    if (g == 7)
        g += ac4_vb(b, 2);
    return (long)g;
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
static void presentation_v1_info(ac4_bits *b, int fri, int *frf_out,
                                 int *multi, ac4_toc_pres *p,
                                 long *max_group)
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
        for (i = 0; i < n_spec; i++) {
            long g = sgi_specifier(b);
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
    if (t->n_cmodes < AC4_MAX_CMODES)
        t->cmodes[t->n_cmodes] = cm;
    if (t->n_cmodes < 1 << 30)
        t->n_cmodes++;
    /* the spec compares the channel_mode code with 0b11111100,
       0b11111101, 0b111111100 and 0b111111101: ch_mode 11 .. 14 */
    if (cm >= 11 && cm <= 14) {
        ac4_bit(b);                     /* b_4_back_channels_present */
        ac4_bit(b);                     /* b_centre_present */
        ac4_u(b, 2);                    /* top_channels_present */
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

static void isf_config(ac4_bits *b)
{
    if (ac4_u(b, 3) > 5)
        ac4_fail(b, b->eof_code);       /* reserved */
}

/* 6.2.1.10 (only read past) */
static void bed_dyn_obj_assignment(ac4_bits *b, long n_signals)
{
    if (ac4_bit(b))                     /* b_dyn_objects_only */
        return;
    if (ac4_bit(b)) {                   /* b_isf */
        isf_config(b);
        return;
    }
    if (ac4_bit(b)) {                   /* b_ch_assign_code */
        ac4_u(b, 3);
        return;
    }
    if (ac4_bit(b)) {                   /* b_channel_assignment_flags_present */
        if (ac4_bit(b))
            ac4_u(b, 17);
        else
            ac4_u(b, 10);
        return;
    }
    {
        long n_bed = 1, i;
        if (n_signals > 1) {
            int nb = 0;
            while ((1L << nb) < n_signals)
                nb++;
            n_bed = (long)ac4_u(b, nb) + 1;
        }
        for (i = 0; i < n_bed; i++)
            ac4_u(b, 4);
    }
    (void)BED_AJOC_N;
    (void)ISF_N;
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
    ac4_bit(b);                         /* b_lfe */
    if (!ac4_bit(b)) {                  /* b_static_dmx */
        n = ac4_u(b, 4) + 1;
        bed_dyn_obj_assignment(b, (long)n);
    }
    if (ac4_bit(b))                     /* b_oamd_common_data_present */
        oamd_common_data(b);
    n = ac4_u(b, 4) + 1;
    if (n == 16)
        n += ac4_vb(b, 3);
    bed_dyn_obj_assignment(b, (long)(n > 1000 ? 1000 : n));
    sub_tail(b, present, frf, t->fs_index, s);
}

static void info_obj(ac4_bits *b, int present, int frf, ac4_toc *t,
                     ac4_toc_sub *s)
{
    memset(s, 0, sizeof(*s));
    s->kind = 2;
    s->channel_mode = -1;
    if (ac4_u(b, 3) > 5)                /* n_objects_code */
        ac4_fail(b, b->eof_code);
    if (ac4_bit(b)) {                   /* b_dynamic_objects */
        ac4_bit(b);                     /* b_lfe */
    } else if (ac4_bit(b)) {            /* b_bed_objects */
        if (ac4_bit(b)) {               /* b_bed_start */
            if (ac4_bit(b))
                ac4_u(b, 3);
            else if (ac4_bit(b))
                ac4_u(b, 17);
            else
                ac4_u(b, 10);
        }
    } else if (ac4_bit(b)) {            /* b_isf */
        if (ac4_bit(b))
            isf_config(b);
    } else {
        ac4_skip(b, 8 * (int)ac4_u(b, 4));   /* res_bytes */
    }
    sub_tail(b, present, frf, t->fs_index, s);
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
        if (ac4_bit(b)) {               /* b_oamd_substream */
            ac4_bit(b);                 /* b_oamd_ndot */
            if (present)
                substream_index(b);
        }
    }
    for (i = 0; i < n_lf; i++) {
        ac4_toc_sub s;
        if (g->channel_coded)
            info_chan(b, present, frf, t, &s);
        else if (ac4_bit(b))            /* b_ajoc */
            info_ajoc(b, present, frf, t, &s);
        else
            info_obj(b, present, frf, t, &s);
        if (hsf && present)
            substream_index(b);         /* ac4_hsf_ext_substream_info */
        if (g->n_subs < AC4_MAX_GSUBS)
            g->subs[g->n_subs++] = s;
    }
    if (ac4_bit(b))                     /* b_content_type */
        content_type(b, g);
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
    if (bv <= 1)
        ac4_fail(b, b->eof_code);
    if (ac4_bit(b)) {                   /* b_program_id */
        ac4_u(b, 16);
        if (ac4_bit(b))                 /* b_program_uuid_present */
            ac4_skip(b, 128);
    }
    for (i = 0; i < n_pres; i++) {
        ac4_toc_pres p;
        presentation_v1_info(b, fri, &frf, &t->rate_multi, &p, &max_group);
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
}

/* ------------------------------------------------------------ choice */

const ac4_toc_sub *ac4_toc_audio_substream(const ac4_toc *t, int pi)
{
    int k, j;
    if (pi < 0 || pi >= t->n_pres)
        return NULL;
    for (k = 0; k < t->pres[pi].n_groups; k++) {
        long gi = t->pres[pi].groups[k];
        const ac4_toc_group *g;
        if (gi < 0 || gi >= t->n_groups)
            continue;
        g = &t->groups[gi];
        for (j = 0; j < g->n_subs; j++) {
            const ac4_toc_sub *s = &g->subs[j];
            if (s->kind == 0 && s->index >= 0 && s->index < t->n_sizes
                && s->index < AC4_MAX_SIZES)
                return s;
        }
    }
    return NULL;
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
