/* SPDX-License-Identifier: Apache-2.0 */
/* Loudness normalisation and DRC (TS 103 190-1 5.7.9, 4.3.13): the
 * presentation substream's dialnorm and drc_frame (TS 103 190-2 6.2.2.3,
 * ac4/metadata.py), and the output stage of ac4/loudness.py, which this
 * follows step by step (the readings are in its docstring and the
 * README). */
#include <math.h>
#include <string.h>

#include "ac4_internal.h"

#define DRC_CB_OFF 127

static const double MODE_RANGE[4][2] = {
    { -31, -27 }, { -26, -17 }, { -16, 0 }, { -16, 0 } };

/* ------------------------------------------------------------ parse */
static void further_loudness_info(ac4_bits *b, ac4_meta *m)
{
    /* b_presentation_ldn = 1, sus_ver = 1 */
    uint32_t v = ac4_u(b, 2), t, n;
    int i;
    if (v == 3)
        ac4_u(b, 4);
    t = ac4_u(b, 4);
    if (t != 0) {
        if (ac4_bit(b))
            ac4_u(b, 3);
        ac4_bit(b);
    }
    if (ac4_bit(b))
        ac4_u(b, 11);
    if (ac4_bit(b)) {
        ac4_u(b, 11);
        ac4_u(b, 3);
    }
    /* loudstrm3s, max_loudstrm3s, truepk, max_truepk */
    for (i = 0; i < 4; i++)
        if (ac4_bit(b)) {
            uint32_t x = ac4_u(b, 11);
            if (i == 2) {
                m->has_truepk = 1;
                m->truepk = ((double)x - 1024.0) / 10.0;
            }
        }
    if (ac4_bit(b)) {                   /* b_prgmbndy */
        n = 0;
        while (ac4_bit(b) == 0)
            if (++n > 32)
                ac4_fail(b, AC4E_ELEMENT_PARSE);
        ac4_bit(b);
        if (ac4_bit(b))
            ac4_u(b, 11);
    }
    if (ac4_bit(b)) {
        ac4_u(b, 10);
        ac4_u(b, 3);
    }
    for (i = 0; i < 2; i++)
        if (ac4_bit(b))
            ac4_u(b, 11);
    if (ac4_bit(b))
        ac4_u(b, 8);
    if (ac4_bit(b)) {
        uint64_t e = ac4_u(b, 5);
        if (e == 31)
            e += ac4_vb(b, 4);
        if (b->p + e > b->nbits)
            ac4_fail(b, b->eof_code);
        b->p += (size_t)e;
    }
}

/* the substream group gains and the associated audio mixing metadata
   after drc_frame (TS 103 190-2 6.2.2.3; metadata.presentation_substream):
   each part is kept only when it is read whole */
/* Tables 128, 129 and TS 103 190-1 4.3.12.2.8 (index 7: -inf) */
static const double GAIN_F1[8] = { 3.0, 1.5, 0.0, -1.5, -3.0, -4.5, -6.0,
                                   -INFINITY };
static const double GAIN_CODE[8] = { 0.0, -1.5, -3.0, -4.5, -6.0, -9.0,
                                     -12.0, -INFINITY };

static void cdmx_set(ac4_cdmx *c, int occ, int k, double v)
{
    c->g[occ][k] = v;
    c->has_g[occ][k] = 1;
}

/* tool_t4_to_f_s_b (back) / tool_t4_to_f_s: the top front, then the top
   back pair */
static void cdmx_t4(ac4_bits *b, ac4_cdmx *c, int occ, int back)
{
    int h;
    for (h = 0; h < 2; h++) {
        int a = h ? AC4_G_T2D : AC4_G_T2A, s = h ? AC4_G_T2E : AC4_G_T2B,
            cc = h ? AC4_G_T2F : AC4_G_T2C;
        if (ac4_bit(b)) {               /* b_top_*_to_front */
            cdmx_set(c, occ, a, GAIN_CODE[ac4_u(b, 3)]);
            cdmx_set(c, occ, s, GAIN_CODE[7]);
        } else if (back) {
            if (ac4_bit(b)) {           /* b_top_*_to_side */
                cdmx_set(c, occ, s, GAIN_CODE[ac4_u(b, 3)]);
            } else {
                cdmx_set(c, occ, cc, GAIN_CODE[ac4_u(b, 3)]);
                cdmx_set(c, occ, s, GAIN_CODE[7]);
            }
        } else {
            cdmx_set(c, occ, s, GAIN_CODE[ac4_u(b, 3)]);
        }
    }
}

/* tool_t2_to_f_s_b (back) / tool_t2_to_f_s */
static void cdmx_t2(ac4_bits *b, ac4_cdmx *c, int occ, int back)
{
    if (ac4_bit(b)) {                   /* b_top_to_front */
        cdmx_set(c, occ, AC4_G_T2A, GAIN_CODE[ac4_u(b, 3)]);
        cdmx_set(c, occ, AC4_G_T2B, GAIN_CODE[7]);
    } else if (back) {
        if (ac4_bit(b)) {               /* b_top_to_side */
            cdmx_set(c, occ, AC4_G_T2B, GAIN_CODE[ac4_u(b, 3)]);
        } else {
            cdmx_set(c, occ, AC4_G_T2C, GAIN_CODE[ac4_u(b, 3)]);
            cdmx_set(c, occ, AC4_G_T2B, GAIN_CODE[7]);
        }
    } else {
        cdmx_set(c, occ, AC4_G_T2B, GAIN_CODE[ac4_u(b, 3)]);
    }
}

/* cdmx_parameters (6.2.9.3) */
static void cdmx_params(ac4_bits *b, ac4_cdmx *c, int bs, int occ)
{
    if (bs == 0 || bs == 3) {           /* tool_scr_to_c_l */
        if (ac4_bit(b))
            cdmx_set(c, occ, AC4_G_F1, GAIN_F1[ac4_u(b, 3)]);
        else
            cdmx_set(c, occ, AC4_G_F2, GAIN_CODE[ac4_u(b, 3)]);
    }
    if (bs < 2) {
        if (occ == 0)
            cdmx_t4(b, c, occ, 0);
        else if (occ == 1 || occ == 4)
            cdmx_set(c, occ, AC4_G_T1, GAIN_CODE[ac4_u(b, 3)]);
        else if (occ == 3)
            cdmx_t4(b, c, occ, 1);
        if (occ <= 2)                   /* tool_b4_to_b2 */
            cdmx_set(c, occ, AC4_G_B, GAIN_CODE[ac4_u(b, 3)]);
    }
    if (bs == 2) {
        if (occ == 0)
            cdmx_t4(b, c, occ, 0);
        else if (occ == 1)
            cdmx_set(c, occ, AC4_G_T1, GAIN_CODE[ac4_u(b, 3)]);
    }
    if (bs >= 3 && bs <= 4) {
        if (occ == 0)
            cdmx_t2(b, c, occ, 0);
        else if (occ == 3)
            cdmx_t2(b, c, occ, 1);
        if (occ <= 2)
            cdmx_set(c, occ, AC4_G_B, GAIN_CODE[ac4_u(b, 3)]);
    }
    if (bs == 5 && occ == 0)
        cdmx_t2(b, c, occ, 0);
}

/* custom_dmx_data (6.2.9.2); pm: pres_ch_mode, pres_ch_mode_core,
   b_pres_4_back_channels_present, pres_top_channel_pairs,
   b_pres_has_lfe (metadata.pres_modes_of) */
static void custom_dmx_data(ac4_bits *b, const int *pm, ac4_cdmx *c)
{
    int cm = pm[0], back = pm[2], pairs = pm[3], bs = -1, i;
    memset(c, 0, sizeof(*c));
    c->lfe_mixgain = -1;
    if (cm >= 11 && cm <= 14) {
        if (pairs == 2) {
            if (cm >= 13 && back)
                bs = 0;
            if (cm <= 12)
                bs = back ? 1 : 2;
        }
        if (pairs == 1) {
            if (cm >= 13 && back)
                bs = 3;
            if (cm <= 12)
                bs = back ? 4 : 5;
        }
    }
    c->bs_ch_config = bs;
    if (bs >= 0 && ac4_bit(b)) {        /* b_cdmx_data_present */
        int n = (int)ac4_u(b, 2) + 1;
        for (i = 0; i < n; i++) {
            int occ = (int)ac4_u(b, (bs == 2 || bs == 5) ? 1 : 3);
            /* a configuration sent twice: the last one counts */
            memset(c->g[occ], 0, sizeof(c->g[occ]));
            memset(c->has_g[occ], 0, sizeof(c->has_g[occ]));
            c->has_cfg[occ] = 1;
            cdmx_params(b, c, bs, occ);
        }
    }
    if (cm >= 3 || pm[1] >= 3) {
        if (ac4_bit(b)) {               /* b_stereo_dmx_coeff */
            c->has_stereo = 1;
            c->loro_c = GAIN_F1[ac4_u(b, 3)];
            c->loro_s = GAIN_F1[ac4_u(b, 3)];
            if (ac4_bit(b)) {           /* b_ltrt_mixinfo */
                ac4_u(b, 3);
                ac4_u(b, 3);
            }
            if (pm[4] && ac4_bit(b))    /* b_lfe_mixinfo */
                c->lfe_mixgain = (int)ac4_u(b, 5);
            ac4_u(b, 2);                /* preferred_dmx_method */
        }
    }
    c->present = 1;
}

static void mix_metadata(ac4_bits *b, long n_groups, const int *pm,
                         ac4_meta *m)
{
    int k;
    if (n_groups > 1 && ac4_bit(b)) {   /* b_substream_group_gains_present */
        if (ac4_bit(b)) {               /* b_keep */
            m->sg_state = 1;
        } else {
            double g[AC4_MAX_SG];
            long i;
            for (i = 0; i < n_groups; i++) {
                uint32_t v = ac4_u(b, 6);
                if (i < AC4_MAX_SG)
                    g[i] = v == 63 ? -INFINITY : -0.25 * (double)v;
            }
            m->n_sg = n_groups < AC4_MAX_SG ? (int)n_groups : AC4_MAX_SG;
            memcpy(m->sg, g, sizeof(double) * (size_t)m->n_sg);
            m->sg_state = 2;
        }
    }
    if (ac4_bit(b)) {                   /* b_associated */
        double v[4];
        int has[4];
        for (k = 0; k < 4; k++) {
            has[k] = (int)ac4_bit(b);
            if (has[k]) {
                uint32_t x = ac4_u(b, 8);
                if (k < 3)              /* 0.3 dB steps, 0xff mute */
                    v[k] = x == 0xFF ? -INFINITY : -0.3 * (double)x;
                else                    /* 1.5 degree steps */
                    v[k] = x < 0xF0 ? 1.5 * (double)x : 0.0;
            } else {
                v[k] = 0.0;
            }
        }
        memcpy(m->sc, v, sizeof(v));
        memcpy(m->has_sc, has, sizeof(has));
        m->has_assoc = 1;
    }
    if (pm) {
        /* kept only when it parses to its end (metadata.py) */
        ac4_cdmx c;
        custom_dmx_data(b, pm, &c);
        m->cdmx = c;
    }
}

int ac4_v0_metadata(const uint8_t *sub, size_t len, size_t bit,
                    ac4_meta *out)
{
    /* metadata.v0_loudness: dialnorm and truepk at the start of a sus_ver
       0 basic_metadata */
    jmp_buf jb;
    ac4_bits b;
    ac4_meta m;
    memset(out, 0, sizeof(*out));
    memset(&m, 0, sizeof(m));
    b.d = sub;
    b.nbits = len * 8;
    b.p = bit;
    b.jb = &jb;
    b.eof_code = AC4E_ELEMENT_PARSE;
    if (bit > b.nbits || setjmp(jb) != 0)
        return -1;
    m.dialnorm = -(double)ac4_u(&b, 7) / 4.0;
    if (ac4_bit(&b) && ac4_bit(&b)) {
        int i;
        if (ac4_u(&b, 2) == 3)
            ac4_u(&b, 4);
        if (ac4_u(&b, 4) != 0) {
            if (ac4_bit(&b))
                ac4_u(&b, 3);
            ac4_bit(&b);
        }
        if (ac4_bit(&b))
            ac4_u(&b, 11);
        if (ac4_bit(&b)) {
            ac4_u(&b, 11);
            ac4_u(&b, 3);
        }
        for (i = 0; i < 2; i++)
            if (ac4_bit(&b))
                ac4_u(&b, 11);
        if (ac4_bit(&b)) {
            m.has_truepk = 1;
            m.truepk = ((double)ac4_u(&b, 11) - 1024.0) / 10.0;
        }
    }
    m.valid = 1;
    *out = m;
    return 0;
}

static void compression_curve(ac4_bits *b, ac4_drc_mode *m)
{
    m->nb_low = (int)ac4_u(b, 4);
    m->nb_high = (int)ac4_u(b, 4);
    m->g_max_boost = (int)ac4_u(b, 4);
    m->n_boost = 0;
    if (m->g_max_boost > 0) {
        m->l_max_boost = (int)ac4_u(b, 5);
        m->n_boost = (int)ac4_bit(b);
        if (m->n_boost) {
            m->g_sec_boost = (int)ac4_u(b, 4);
            m->l_sec_boost = (int)ac4_u(b, 5);
        }
    }
    m->g_max_cut = (int)ac4_u(b, 5);
    m->n_cut = 0;
    if (m->g_max_cut > 0) {
        m->l_max_cut = (int)ac4_u(b, 6);
        m->n_cut = (int)ac4_bit(b);
        if (m->n_cut) {
            m->g_sec_cut = (int)ac4_u(b, 5);
            m->l_sec_cut = (int)ac4_u(b, 5);
        }
    }
    m->tc_default = (int)ac4_bit(b);
    m->adaptive = 0;
    if (!m->tc_default) {
        int i;
        for (i = 0; i < 4; i++)
            m->tc[i] = (int)ac4_u(b, 8);
        m->adaptive = (int)ac4_bit(b);
        if (m->adaptive) {
            m->th[0] = (int)ac4_u(b, 5);
            m->th[1] = (int)ac4_u(b, 5);
        }
    }
}

static void drc_config(ac4_bits *b, ac4_drc_config *c)
{
    int n = (int)ac4_u(b, 3) + 1, i, j;
    memset(c, 0, sizeof(*c));
    for (i = 0; i < n; i++) {
        ac4_drc_mode *m = &c->m[i];
        m->id = (int)ac4_u(b, 3);
        if (m->id > 3) {
            m->lo = -(double)ac4_u(b, 5);
            m->hi = -(double)ac4_u(b, 5);
        } else {
            m->lo = MODE_RANGE[m->id][0];
            m->hi = MODE_RANGE[m->id][1];
        }
        if (ac4_bit(b)) {               /* drc_repeat_profile_flag */
            int rid = (int)ac4_u(b, 3);
            const ac4_drc_mode *src = NULL;
            for (j = 0; j < i; j++)     /* the last mode with that id */
                if (c->m[j].id == rid)
                    src = &c->m[j];
            if (src) {
                double lo = m->lo, hi = m->hi;
                int id = m->id;
                *m = *src;
                m->id = id;
                m->lo = lo;
                m->hi = hi;
            } else {
                m->curve_flag = 1;
                m->has_curve = 0;
                m->default_profile = 0;
            }
        } else if (!ac4_bit(b)) {       /* !drc_default_profile_flag */
            m->curve_flag = (int)ac4_bit(b);
            m->default_profile = 0;
            if (m->curve_flag) {
                compression_curve(b, m);
                m->has_curve = 1;
            } else {
                m->gains_config = (int)ac4_u(b, 2);
            }
        } else {
            m->curve_flag = 1;
            m->default_profile = 1;
        }
    }
    c->n_modes = n;
    c->eac3_profile = (int)ac4_u(b, 3);
    c->valid = 1;
}

static int drc_subframes(int fl)
{
    switch (fl) {
    case 384: return 1;
    case 512: return 2;
    case 768: case 960: return 3;
    case 1024: return 4;
    case 1536: case 1920: return 6;
    case 2048: return 8;
    default: return 1;
    }
}

static void drc_gains(ac4_bits *b, const ac4_huff *hcb, int cfg, int n_ch,
                      int n_sf, ac4_drc_gains *g)
{
    static const int NB[4] = { 1, 1, 2, 4 };
    int nb = NB[cfg & 3], ch, sf, band, ref;
    int g0 = (int)ac4_u(b, 7) - 64;
    memset(g, 0, sizeof(*g));
    g->present = 1;
    if (cfg == 0) {
        g->n_ch = g->n_sf = g->n_band = 1;
        g->g[0][0][0] = g0;
        return;
    }
    g->n_ch = n_ch;
    g->n_sf = n_sf;
    g->n_band = nb;
    g->g[0][0][0] = g0;
    ref = g0;
    for (ch = 0; ch < n_ch; ch++) {
        for (band = 0; band < nb; band++) {
            for (sf = 0; sf < n_sf; sf++) {
                if (sf || band || ch)
                    g->g[ch][sf][band] = ref + ac4_huff_decode(hcb, b)
                                         - DRC_CB_OFF;
                ref = g->g[ch][sf][band];
            }
            ref = g->g[ch][0][band];
        }
        ref = g->g[ch][0][0];
    }
}

/* drc_frame (4.2.14.5): 0 = b_drc_present 0, 1 = parsed into m */
static int drc_frame(ac4_bits *b, const ac4_huff *hcb, int iframe,
                     ac4_drc_config *state, int ch_mode, int fl,
                     ac4_meta *m)
{
    int i, curve = 0;
    if (!ac4_bit(b))
        return 0;
    if (iframe) {
        ac4_drc_config c;
        drc_config(b, &c);
        *state = c;
    }
    if (!state->valid)
        ac4_fail(b, AC4E_ELEMENT_PARSE);  /* drc_data before drc_config */
    m->cfg = *state;
    m->reset = -1;
    for (i = 0; i < 8; i++)
        m->gains[i].present = 0;
    for (i = 0; i < state->n_modes; i++) {
        const ac4_drc_mode *md = &state->m[i];
        if (md->curve_flag == 0) {
            size_t size = ac4_u(b, 6), start;
            uint32_t ver;
            if (ac4_bit(b))
                size += (size_t)ac4_vb(b, 2) << 6;
            start = b->p;
            ver = ac4_u(b, 2);
            if (ver <= 1)
                drc_gains(b, hcb, md->gains_config,
                          ch_mode == 0 || ch_mode == 1 ? 1 : 3,
                          drc_subframes(fl), &m->gains[md->id]);
            if (ver >= 1)
                b->p = start + size;
        } else {
            curve = 1;
        }
    }
    if (curve) {
        m->reset = (int)ac4_bit(b);
        ac4_u(b, 2);
    }
    return 1;
}

int ac4_pres_metadata(const uint8_t *sub, size_t len, int b_alternative,
                      int b_pres_ndot, long n_substreams,
                      ac4_drc_config *state, const ac4_huff *hcb,
                      int ch_mode, int fl, long n_groups, const int *pm,
                      ac4_meta *out)
{
    jmp_buf jb;
    ac4_bits b;
    volatile size_t size = 0, start = 0;
    volatile int stage = 0;
    ac4_meta m;
    memset(out, 0, sizeof(*out));
    memset(&m, 0, sizeof(m));
    b.d = sub;
    b.nbits = len * 8;
    b.p = 0;
    b.jb = &jb;
    b.eof_code = AC4E_ELEMENT_PARSE;
    if (setjmp(jb) != 0) {
        if (stage == 0)
            return -1;
        if (stage == 1) {
            /* the drc_frame did not parse: dialnorm kept, no DRC; the
               mixing metadata after it is still read */
            m.has_drc = 0;
            stage = 2;
            b.p = start + size;
            mix_metadata(&b, n_groups, pm, &m);
        }
        *out = m;
        return 0;
    }
    if (b_alternative) {
        long i, j, nt;
        if (ac4_bit(&b)) {               /* b_name_present */
            int n = ac4_bit(&b) ? (int)ac4_u(&b, 5) : 32;
            for (i = 0; i < n; i++)
                ac4_u(&b, 8);
        }
        nt = (long)ac4_u(&b, 2) + 1;
        if (nt == 4)
            nt += (long)ac4_vb(&b, 2);
        for (i = 0; i < nt; i++) {
            ac4_u(&b, 3);
            ac4_u(&b, 4);
            if (ac4_bit(&b))
                ac4_u(&b, 4);
            if (ac4_bit(&b))
                ac4_u(&b, 6);
            if (ac4_bit(&b))
                ac4_u(&b, 5);
            for (j = 0; j < n_substreams; j++)
                if (ac4_bit(&b) && ac4_bit(&b))
                    ac4_vb(&b, 2);
        }
    }
    if (ac4_bit(&b)) {                   /* b_additional_data */
        uint64_t n = ac4_u(&b, 4) + 1;
        if (n == 16)
            n += ac4_vb(&b, 2);
        ac4_align(&b);
        b.p += (size_t)(8 * n);
    }
    m.dialnorm = -(double)ac4_u(&b, 7) / 4.0;
    m.valid = 1;
    if (ac4_bit(&b))
        further_loudness_info(&b, &m);
    size = ac4_u(&b, 5);               /* drc_metadata_size */
    if (ac4_bit(&b))
        size += (size_t)ac4_vb(&b, 3) << 5;
    start = b.p;
    stage = 1;
    m.has_drc = drc_frame(&b, hcb, b_pres_ndot, state, ch_mode, fl, &m);
    stage = 2;
    b.p = start + size;
    mix_metadata(&b, n_groups, pm, &m);
    *out = m;
    return 0;
}

/* ------------------------------------------------------------ curves */
typedef struct {
    double l0l, l0h, gmb, lmb, gsb, lsb, gmc, lmc, gsc, lsc;
    int has_gsb, has_gsc;
} curve_t;

/* Table 162: L0low, L0high, Gmaxboost, Lmaxboost, Gmaxcut, Lmaxcut,
   Lsectioncut, Gsectioncut (has), time constants */
static const struct {
    double v[8];
    int has_sc;
    double tc[6];
} PROFILE[6] = {
    { { 0 }, 0, { 0 } },
    { { 0, 5, 6, -12, -24, 35, 15, -5 }, 1, { 100, 3000, 10, 1000, 15, 20 } },
    { { -10, 10, 6, -22, -24, 40, 20, -5 }, 1,
      { 100, 3000, 10, 1000, 15, 20 } },
    { { 0, 5, 12, -24, -24, 35, 15, -5 }, 1,
      { 100, 10000, 10, 1000, 15, 20 } },
    { { -10, 10, 12, -34, -15, 40, 0, 0 }, 0,
      { 100, 3000, 10, 1000, 15, 20 } },
    { { 0, 5, 15, -19, -24, 35, 15, -5 }, 1, { 100, 1000, 10, 200, 10, 10 } },
};

/* -> 1 with the curve and time constants (th[0] < 0: no thresholds) */
static int curve_points(const ac4_drc_mode *m, int profile, curve_t *p,
                        double tc[6], int *has_th)
{
    memset(p, 0, sizeof(*p));
    if (m->default_profile) {
        if (profile < 1 || profile > 5)
            return 0;
        p->l0l = PROFILE[profile].v[0];
        p->l0h = PROFILE[profile].v[1];
        p->gmb = PROFILE[profile].v[2];
        p->lmb = PROFILE[profile].v[3];
        p->gmc = PROFILE[profile].v[4];
        p->lmc = PROFILE[profile].v[5];
        p->has_gsb = 0;
        p->lsb = p->l0l;
        p->has_gsc = PROFILE[profile].has_sc;
        p->gsc = PROFILE[profile].v[7];
        p->lsc = p->has_gsc ? PROFILE[profile].v[6] : p->l0h;
        memcpy(tc, PROFILE[profile].tc, sizeof(double) * 6);
        *has_th = 1;
        return 1;
    }
    if (!m->has_curve)
        return 0;
    p->l0l = 0 - m->nb_low;
    p->l0h = 0 + m->nb_high;
    p->gmb = m->g_max_boost;
    if (m->n_boost) {
        p->has_gsb = 1;
        p->gsb = 1 + m->g_sec_boost;
        p->lsb = p->l0l - (1 + m->l_sec_boost);
    } else {
        p->lsb = p->l0l;
    }
    p->lmb = p->gmb > 0 ? p->lsb - (1 + m->l_max_boost) : p->l0l;
    p->gmc = 0 - m->g_max_cut;
    if (m->n_cut) {
        p->has_gsc = 1;
        p->gsc = 0 - (1 + m->g_sec_cut);
        p->lsc = p->l0h + (1 + m->l_sec_cut);
    } else {
        p->lsc = p->l0h;
    }
    p->lmc = p->gmc < 0 ? p->lsc + (1 + m->l_max_cut) : p->l0h;
    if (m->tc_default) {
        tc[0] = 100;
        tc[1] = 3000;
        tc[2] = 10;
        tc[3] = 1000;
        *has_th = 0;
    } else {
        tc[0] = m->tc[0] * 5;
        tc[1] = m->tc[1] * 40;
        tc[2] = m->tc[2] * 5;
        tc[3] = m->tc[3] * 20;
        *has_th = m->adaptive;
        tc[4] = m->th[0];
        tc[5] = m->th[1];
    }
    return 1;
}

double ac4_curve_gain(double x, const double *pv)
{
    /* pv: l0l l0h gmb lmb gsb lsb gmc lmc gsc lsc (gsb/gsc 0 if absent) */
    double l0l = pv[0], l0h = pv[1], gmb = pv[2], lmb = pv[3], gsb = pv[4],
           lsb = pv[5], gmc = pv[6], lmc = pv[7], gsc = pv[8], lsc = pv[9];
    double d;
    if (x < lmb)
        return gmb;
    if (x <= lsb) {
        d = lsb - lmb;
        return d != 0 ? gsb + (gmb - gsb) * ((lsb - x) / d) : gsb;
    }
    if (x <= l0l) {
        d = l0l - lsb;
        return d != 0 ? gsb * ((l0l - x) / d) : 0.0;
    }
    if (x <= l0h)
        return 0.0;
    if (x <= lsc) {
        d = lsc - l0h;
        return d != 0 ? gsc * ((x - l0h) / d) : 0.0;
    }
    if (x <= lmc) {
        d = lmc - lsc;
        return d != 0 ? gsc + (gmc - gsc) * ((x - lsc) / d) : gsc;
    }
    return gmc;
}

static const ac4_drc_mode *select_mode(const ac4_drc_config *c, double t)
{
    const ac4_drc_mode *best = NULL;
    int i;
    for (i = 0; i < c->n_modes; i++) {
        const ac4_drc_mode *m = &c->m[i];
        double lo = m->lo < m->hi ? m->lo : m->hi;
        double hi = m->lo < m->hi ? m->hi : m->lo;
        if (lo < t && t < hi && (!best || m->id > best->id))
            best = m;
    }
    return best;
}

/* ------------------------------------------------------------ stage */
#define SLOT 64
static const double KB1[3] = { 1.53512485958697, -2.69169618940638,
                               1.19839281085285 };
static const double KA1[3] = { 1.0, -1.69065929318241, 0.73248077421585 };
static const double KB2[3] = { 1.0, -2.0, 1.0 };
static const double KA2[3] = { 1.0, -1.99004745483398, 0.99007225036621 };

void ac4_leveler_init(ac4_leveler *lv, int n_ch, const int *slot,
                      int backs, double target, int drc, double rate)
{
    /* per slot (decoder.c CH_*: L R C LFE Ls Rs X1 X2, then the immersive
       element's Lb Rb Tbl Tbr): BS.1770 weights (LFE 0, surrounds and
       back pair 1.41; the 7_X pair 1.41 when it is Lb / Rb; the top
       channels 1.0) and the transmitted-gain channel groups (Table 168: L
       R LFE, C, rest) */
    static const double W[12] = { 1.0, 1.0, 1.0, 0.0, 1.41, 1.41, 1.0, 1.0,
                                   1.41, 1.41, 1.0, 1.0 };
    static const int GRP[12] = { 0, 0, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2 };
    int k;
    memset(lv, 0, sizeof(*lv));
    lv->n_ch = n_ch;
    lv->target = target;
    lv->drc = drc;
    lv->rate = rate;
    lv->mode_id = -1;
    for (k = 0; k < n_ch; k++) {
        int sl = slot[k];
        if (sl < 0 || sl >= 12)
            sl = 6;
        lv->w[k] = (backs && (sl == 6 || sl == 7)) ? 1.41 : W[sl];
        lv->grp[k] = GRP[sl];
    }
}

static void transmitted(const ac4_drc_gains *g, const ac4_leveler *lv,
                        int n_slot, double out[][AC4_LEV_CH])
{
    int s, k, j;
    for (s = 0; s < n_slot; s++) {
        int sf = s * g->n_sf / n_slot;
        if (sf > g->n_sf - 1)
            sf = g->n_sf - 1;
        for (k = 0; k < lv->n_ch; k++) {
            int gi = g->n_ch == 1 ? 0 : lv->grp[k];
            double db = 0.0;
            if (gi > g->n_ch - 1)
                gi = g->n_ch - 1;
            for (j = 0; j < g->n_band; j++)
                db += g->g[gi][sf][j];
            db /= g->n_band;
            out[s][k] = pow(2.0, db / 6.0);
        }
    }
}

void ac4_leveler_frame(ac4_leveler *lv, const ac4_meta *md, double **pcm,
                       int n)
{
    double gains[AC4_MAX_FRAME / SLOT + 1][AC4_LEV_CH];
    int n_slot = (n + SLOT - 1) / SLOT, s, k, i, fresh = 0;
    double lin, norm;
    for (k = 0; k < lv->n_ch; k++)
        for (i = 0; i < n; i++)
            pcm[k][i] /= AC4_FULL_SCALE;
    if (md && md->valid) {
        lv->lin = md->dialnorm;
        lv->has_lin = 1;
    }
    lin = lv->has_lin ? lv->lin : lv->target;
    norm = pow(2.0, (lv->target - lin) / 6.0);
    if (md && md->valid && md->has_drc) {
        lv->dfr = *md;
        lv->has_dfr = 1;
        fresh = 1;
    }
    for (s = 0; s < n_slot; s++)
        for (k = 0; k < lv->n_ch; k++)
            gains[s][k] = 1.0;
    if (lv->drc != AC4_DRC_OFF) {
        /* K-weighting (DF2T biquads), state carried */
        for (k = 0; k < lv->n_ch; k++) {
            double z0 = lv->z1[0][k], z1 = lv->z1[1][k];
            double y0 = lv->z2[0][k], y1 = lv->z2[1][k];
            for (i = 0; i < n; i++) {
                double x = pcm[k][i], v, y;
                v = KB1[0] * x + z0;
                z0 = KB1[1] * x + z1 - KA1[1] * v;
                z1 = KB1[2] * x - KA1[2] * v;
                y = KB2[0] * v + y0;
                y0 = KB2[1] * v + y1 - KA2[1] * y;
                y1 = KB2[2] * v - KA2[2] * y;
                lv->kw[k][i] = y;
            }
            lv->z1[0][k] = z0;
            lv->z1[1][k] = z1;
            lv->z2[0][k] = y0;
            lv->z2[1][k] = y1;
        }
    }
    if (lv->drc != AC4_DRC_OFF && lv->has_dfr) {
        const ac4_drc_config *cfg = &lv->dfr.cfg;
        const ac4_drc_mode *m = NULL;
        if (lv->drc == AC4_DRC_AUTO) {
            m = select_mode(cfg, lv->target);
        } else {
            for (i = 0; i < cfg->n_modes && !m; i++)
                if (cfg->m[i].id == lv->drc)
                    m = &cfg->m[i];
        }
        if (m) {
            lv->mode_id = m->id;
            if (m->curve_flag == 0) {
                if (lv->dfr.gains[m->id].present)
                    transmitted(&lv->dfr.gains[m->id], lv, n_slot, gains);
            } else {
                curve_t p;
                double tc[6], pv[10], T;
                int has_th;
                if (curve_points(m, cfg->eac3_profile, &p, tc, &has_th)) {
                    pv[0] = p.l0l; pv[1] = p.l0h; pv[2] = p.gmb;
                    pv[3] = p.lmb; pv[4] = p.has_gsb ? p.gsb : 0.0;
                    pv[5] = p.lsb; pv[6] = p.gmc; pv[7] = p.lmc;
                    pv[8] = p.has_gsc ? p.gsc : 0.0; pv[9] = p.lsc;
                    if (fresh && lv->dfr.reset > 0)
                        lv->has_s = 0;
                    T = SLOT * 1000.0 / lv->rate;
                    for (s = 0; s < n_slot; s++) {
                        double ms = 0.0, lvl, x, gl;
                        int e = (s + 1) * SLOT < n ? (s + 1) * SLOT : n;
                        for (k = 0; k < lv->n_ch; k++) {
                            double acc = 0.0;
                            for (i = s * SLOT; i < e; i++)
                                acc += lv->kw[k][i] * lv->kw[k][i];
                            ms += lv->w[k] * (acc / (e - s * SLOT));
                        }
                        lvl = ms;
                        x = -0.691 + 10.0 * log10(lvl + 1e-30) - lin;
                        gl = pow(10.0, ac4_curve_gain(x, pv) / 20.0);
                        if (!lv->has_s) {
                            lv->g_s = gl;
                            lv->l_s = lvl;
                            lv->has_s = 1;
                        } else {
                            double tau = lvl > lv->l_s ? tc[0] : tc[1], al;
                            if (has_th && lvl > 0 && lv->l_s > 0) {
                                double r = 10.0 * log10(lvl / lv->l_s);
                                if (r > tc[4])
                                    tau = tc[2];
                                else if (-r > tc[5])
                                    tau = tc[3];
                            }
                            al = tau > 0 ? pow(2.0, -T / tau) : 0.0;
                            lv->g_s = al * lv->g_s + (1.0 - al) * gl;
                            lv->l_s = al * lv->l_s + (1.0 - al) * lvl;
                        }
                        for (k = 0; k < lv->n_ch; k++)
                            gains[s][k] = lv->g_s;
                    }
                }
            }
        }
    }
    for (s = 0; s < n_slot; s++)
        for (k = 0; k < lv->n_ch; k++) {
            double g = norm * gains[s][k];
            for (i = s * SLOT; i < (s + 1) * SLOT && i < n; i++)
                pcm[k][i] *= g;
        }
}
