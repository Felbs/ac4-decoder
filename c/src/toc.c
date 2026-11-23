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

static long channel_mode(ac4_bits *b)
{
    uint32_t two, tail;
    if (ac4_bit(b) == 0)
        return 0;
    if (ac4_bit(b) == 0)
        return 1;
    two = ac4_u(b, 2);
    if (two != 3)
        return two == 0 ? 2 : two == 1 ? 3 : 4;
    tail = ac4_u(b, 3);
    if (tail != 7)
        return 5 + (long)tail;
    if (ac4_bit(b) == 0)
        return 12;
    if (ac4_bit(b) == 0)
        return 13;
    return 14 + (long)ac4_vb(b, 2);
}

static void content_type(ac4_bits *b)
{
    ac4_u(b, 3);                        /* content_classifier */
    if (ac4_bit(b)) {                   /* b_language_indicator */
        if (ac4_bit(b)) {               /* b_serialized_language_tag */
            ac4_bit(b);
            ac4_u(b, 16);
        } else {
            uint32_t n = ac4_u(b, 6), i;
            for (i = 0; i < n; i++)
                ac4_u(b, 8);
        }
    }
}

static void emdf_info(ac4_bits *b)
{
    uint32_t v = ac4_u(b, 2), k, prim, sec;
    if (v == 3)
        ac4_vb(b, 2);
    k = ac4_u(b, 3);
    if (k == 7)
        ac4_vb(b, 3);
    if (ac4_bit(b)) {                   /* b_emdf_payloads_substream_info */
        if (ac4_u(b, 2) == 3)
            ac4_vb(b, 2);
    }
    prim = ac4_u(b, 2);
    sec = ac4_u(b, 2);
    if (PROT_PRIMARY[prim] < 0)
        ac4_fail(b, b->eof_code);       /* Reserved */
    ac4_skip(b, (size_t)PROT_PRIMARY[prim]);
    if (PROT_SECOND[sec])
        ac4_skip(b, (size_t)PROT_SECOND[sec]);
}

static int frame_rate_multiply_info(ac4_bits *b, int fri)
{
    int frf = 1;
    if (fri == 2 || fri == 3 || fri == 4) {
        if (ac4_bit(b))
            frf = ac4_bit(b) ? 4 : 2;
    } else if (fri == 0 || fri == 1 || fri == 7 || fri == 8 || fri == 9) {
        if (ac4_bit(b))
            frf = 2;
    }
    return frf;
}

static void frame_rate_fractions_info(ac4_bits *b, int fri, int frf)
{
    if (fri >= 5 && fri <= 9 && frf == 1) {
        ac4_bit(b);
    } else if (fri >= 10 && fri <= 12) {
        if (ac4_bit(b))
            ac4_bit(b);
    }
}

static void sgi_specifier(ac4_bits *b)
{
    if (ac4_u(b, 3) == 7)
        ac4_vb(b, 2);
}

static void presentation_substream_info(ac4_bits *b)
{
    ac4_bit(b);                         /* b_alternative */
    ac4_bit(b);                         /* b_pres_ndot */
    if (ac4_u(b, 2) == 3)
        ac4_vb(b, 2);
}

/* -> number of substream groups the presentation references */
static uint64_t presentation_v1_info(ac4_bits *b, int fri, int *frf_out)
{
    int single = (int)ac4_bit(b);
    uint64_t cfg = 0, n_sg = 0;
    int add_emdf;
    if (!single) {
        cfg = ac4_u(b, 3);
        if (cfg == 7)
            cfg += ac4_vb(b, 2);
    }
    while (ac4_bit(b))                  /* presentation_version, unary */
        ;
    if (!single && cfg == 6) {
        add_emdf = 1;
    } else {
        int frf;
        uint64_t i;
        ac4_u(b, 3);                    /* mdcompat */
        if (ac4_bit(b))                 /* b_presentation_group_index */
            ac4_vb(b, 2);
        frf = frame_rate_multiply_info(b, fri);
        *frf_out = frf;
        frame_rate_fractions_info(b, fri, frf);
        emdf_info(b);
        if (ac4_bit(b))                 /* b_presentation_filter */
            ac4_bit(b);                 /* b_enable_presentation */
        if (single) {
            sgi_specifier(b);
            n_sg = 1;
        } else {
            uint64_t n_spec;
            ac4_bit(b);                 /* b_multi_pid */
            switch (cfg) {
            case 0: n_spec = 2; n_sg = 2; break;
            case 1: n_spec = 2; n_sg = 1; break;
            case 2: n_spec = 2; n_sg = 2; break;
            case 3: n_spec = 3; n_sg = 3; break;
            case 4: n_spec = 3; n_sg = 2; break;
            case 5:
                n_sg = ac4_u(b, 2) + 2;
                if (n_sg == 5)
                    n_sg += ac4_vb(b, 2);
                n_spec = n_sg;
                break;
            default:                    /* presentation_config_ext_info */
                ac4_fail(b, b->eof_code);
                return 0;
            }
            for (i = 0; i < n_spec; i++)
                sgi_specifier(b);
        }
        ac4_bit(b);                     /* b_pre_virtualized */
        add_emdf = (int)ac4_bit(b);
        presentation_substream_info(b);
    }
    if (add_emdf) {
        uint64_t n = ac4_u(b, 2), i;
        if (n == 0)
            n = ac4_vb(b, 2) + 4;
        for (i = 0; i < n; i++)
            emdf_info(b);
    }
    return n_sg;
}

static void substream_info_chan(ac4_bits *b, int present, int frf,
                                ac4_toc *t)
{
    long cm = channel_mode(b);
    int i;
    if (t->n_cmodes < AC4_MAX_CMODES)
        t->cmodes[t->n_cmodes] = cm;
    if (t->n_cmodes < 1 << 30)
        t->n_cmodes++;
    if (cm == 252 || cm == 253 || cm == 508 || cm == 509) {
        ac4_bit(b);
        ac4_bit(b);
        ac4_u(b, 2);
    }
    if (t->fs_index == 1) {
        if (ac4_bit(b))                 /* b_sf_multiplier */
            ac4_bit(b);
    }
    if (ac4_bit(b)) {                   /* b_bitrate_info */
        if (ac4_u(b, 3) == 7)
            ac4_u(b, 2);
    }
    if (cm >= 7 && cm <= 10)
        ac4_bit(b);                     /* add_ch_base */
    for (i = 0; i < frf; i++)
        ac4_bit(b);                     /* b_audio_ndot */
    if (present) {
        if (ac4_u(b, 2) == 3)
            ac4_vb(b, 2);
    }
}

static void substream_group_info(ac4_bits *b, int frf, ac4_toc *t)
{
    int present = (int)ac4_bit(b);      /* b_substreams_present */
    int hsf = (int)ac4_bit(b);          /* b_hsf_ext */
    int single = (int)ac4_bit(b);       /* b_single_substream */
    uint64_t n_lf, i;
    if (single) {
        n_lf = 1;
    } else {
        n_lf = ac4_u(b, 2) + 2;
        if (n_lf == 5)
            n_lf += ac4_vb(b, 2);
    }
    if (ac4_bit(b)) {                   /* b_channel_coded */
        for (i = 0; i < n_lf; i++) {
            substream_info_chan(b, present, frf, t);
            if (hsf && present) {
                if (ac4_u(b, 2) == 3)
                    ac4_vb(b, 2);
            }
        }
    } else {
        ac4_fail(b, b->eof_code);       /* object / A-JOC: not implemented */
    }
    if (ac4_bit(b))                     /* b_content_type */
        content_type(b);
}

void ac4_toc_parse(ac4_bits *b, ac4_toc *t)
{
    uint64_t bv, n_pres, total_sg = 0, i, n_sub;
    int fri, frf = 0, size_present;
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
    for (i = 0; i < n_pres; i++)
        total_sg += presentation_v1_info(b, fri, &frf);
    for (i = 0; i < total_sg; i++)
        substream_group_info(b, frf, t);
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
            if (i < 8)
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
