/* SPDX-License-Identifier: Apache-2.0 */
/* OAMD and A-JOC side information: a port of ac4/oamd.py, ac4/ajoc.py
   (parse) and elements.ajoc_tail (TS 103 190-2 6.2.3.4, 6.2.5, 6.2.8). */
#include <math.h>
#include <string.h>

#include "ac4_internal.h"

static int clip3i(int lo, int hi, int v)
{
    return v < lo ? lo : v > hi ? hi : v;
}

static double clip3d(double lo, double hi, double v)
{
    return v < lo ? lo : v > hi ? hi : v;
}

static void obj_default_render(ac4_obj *o)
{
    o->px = o->py = 31;
    o->pz = 0;
    o->x = o->y = 0.5;
    o->z = 0.0;
}

void ac4_obj_init(ac4_obj *o)
{
    memset(o, 0, sizeof(*o));
    o->gain = -INFINITY;
    o->priority = 0.0;
    o->active = 0;
    obj_default_render(o);
}

/* oamd.ObjState.set_pos */
static void set_pos(ac4_obj *o, int px, int py, int pz, const int *ext)
{
    int ex = clip3i(-2, 2, ext[0]), ey = clip3i(-2, 2, ext[1]),
        ez = clip3i(-2, 2, ext[2]);
    o->px = px;
    o->py = py;
    o->pz = pz;
    o->x = clip3d(0.0, 1.0, (double)clip3i(0, 62, px) / 62
                  + (double)ex / (62 * 5));
    o->y = clip3d(0.0, 1.0, (double)clip3i(0, 62, py) / 62
                  + (double)ey / (62 * 5));
    o->z = clip3d(-1.0, 1.0, (double)clip3i(-15, 15, pz) / 15
                  + (double)ez / (15 * 5));
}

static int s3(uint32_t v)
{
    return v >= 4 ? (int)v - 8 : (int)v;
}

static void trim(ac4_bits *b)
{
    int c, i;
    if (!ac4_bit(b))
        return;
    ac4_u(b, 2);
    ac4_u(b, 2);
    if (ac4_u(b, 2) != 2)
        return;
    for (c = 0; c < 9; c++) {
        uint32_t pres;
        if (ac4_bit(b) || ac4_bit(b))   /* b_default_trim, b_disable_trim */
            continue;
        pres = ac4_u(b, 5);
        for (i = 4; i >= 2; i--)
            if ((pres >> i) & 1)
                ac4_u(b, 4);
        for (i = 1; i >= 0; i--)
            if ((pres >> i) & 1) {
                ac4_bit(b);
                ac4_u(b, 4);
            }
    }
}

static void tool_f_s(ac4_bits *b, int back)
{
    if (ac4_bit(b))
        ac4_u(b, 3);
    else if (back) {
        ac4_bit(b);
        ac4_u(b, 3);
    } else {
        ac4_u(b, 3);
    }
}

static void bed_render_info(ac4_bits *b)
{
    int tb, tf;
    if (!ac4_bit(b))
        return;
    if (ac4_bit(b)) {                   /* stereo_dmx_coeff */
        ac4_u(b, 3);
        ac4_u(b, 3);
        if (ac4_bit(b)) {
            ac4_u(b, 3);
            ac4_u(b, 3);
        }
        if (ac4_bit(b))
            ac4_u(b, 5);
        ac4_u(b, 2);
    }
    if (!ac4_bit(b))                    /* b_cdmx_data_present */
        return;
    if (ac4_bit(b))
        ac4_u(b, 3);
    if (ac4_bit(b))
        ac4_u(b, 3);
    if (ac4_bit(b)) {                   /* b_tm_ch_present */
        if (ac4_bit(b))
            tool_f_s(b, 1);
        if (ac4_bit(b))
            tool_f_s(b, 0);
    }
    tb = (int)ac4_bit(b);
    if (tb) {
        if (ac4_bit(b))
            tool_f_s(b, 1);
        if (ac4_bit(b))
            tool_f_s(b, 0);
    }
    tf = (int)ac4_bit(b);
    if (tf) {
        if (ac4_bit(b))
            tool_f_s(b, 1);
        if (ac4_bit(b))
            tool_f_s(b, 0);
    }
    if ((tb || tf) && ac4_bit(b))
        ac4_u(b, 3);
}

static void headphone(ac4_bits *b)
{
    if (ac4_bit(b)) {
        uint32_t m = ac4_u(b, 3);
        if (m == 1 || m == 2)
            ac4_bit(b);
    }
}

void ac4_oamd_common_data(ac4_bits *b)
{
    if (ac4_bit(b) == 0)
        ac4_u(b, 5);
    ac4_bit(b);
    if (ac4_bit(b)) {
        uint64_t n = ac4_u(b, 1) + 1;
        size_t p0 = b->p, bits;
        if (n == 2)
            n += ac4_vb(b, 2);
        bits = (size_t)(8 * n);
        trim(b);
        if (b->p - p0 < bits)
            bed_render_info(b);
        if (b->p - p0 < bits)
            headphone(b);
        if (b->p - p0 > bits)
            ac4_fail(b, AC4E_ELEMENT_PARSE);
        b->p = p0 + bits;
    }
}

/* oamd_timing_data (oamd.oamd_timing_data; ramps in samples, Tables 94,
   95) -> num_obj_info_blocks */
static int oamd_timing_data(ac4_bits *b, ac4_oamd_timing *t)
{
    static const int TAB[16] = { 32, 64, 128, 256, 320, 480, 1000, 1001,
                                 1024, 1600, 1601, 1602, 1920, 2000, 2002,
                                 2048 };
    int nb, i, so = 0;
    if (ac4_bit(b)) {                   /* oa_sample_offset_type 0b1x */
        if (ac4_bit(b) == 0) {
            if (ac4_bit(b) == 0)
                so = 16;
            else
                so = ac4_bit(b) == 0 ? 8 : 24;
        } else {
            so = (int)ac4_u(b, 5);
        }
    }
    nb = (int)ac4_u(b, 3);
    t->has = 1;
    t->sample_offset = so;
    t->n = nb;
    for (i = 0; i < nb; i++) {
        int rc;
        t->off[i] = (int)ac4_u(b, 6);
        rc = (int)ac4_u(b, 2);
        if (rc == 3) {
            if (ac4_bit(b))
                t->ramp[i] = TAB[ac4_u(b, 4)];
            else
                t->ramp[i] = (int)ac4_u(b, 11);
        } else {
            t->ramp[i] = rc == 0 ? 0 : rc == 1 ? 512 : 1536;
        }
    }
    return nb;
}

/* oamd.block_updates -> ev[k] = (update sample, ramp) for n blocks */
static void block_updates(const ac4_oamd_timing *t, int n, int (*ev)[2])
{
    int k;
    for (k = 0; k < n; k++) {
        if (!t->has || k >= t->n) {
            ev[k][0] = t->has ? t->sample_offset : 0;
            ev[k][1] = 0;
        } else {
            ev[k][0] = t->sample_offset + 32 * t->off[k];
            ev[k][1] = t->ramp[k];
        }
    }
}

static void object_basic_info(ac4_bits *b, ac4_obj *o)
{
    int md;
    if (ac4_bit(b)) {                   /* b_default_basic_info_md */
        o->gain = 0.0;
        o->priority = 1.0;
        return;
    }
    md = ac4_bit(b) == 0 ? 0 : (ac4_bit(b) == 0 ? 2 : 3);
    if (md == 0 || md == 2) {
        if (ac4_bit(b) == 0) {
            int v = (int)ac4_u(b, 6);
            o->gain = (double)(v <= 14 ? 15 - v : 14 - v);
        } else if (ac4_bit(b) == 0) {
            o->gain = -INFINITY;
        }
    } else {
        o->gain = 0.0;
    }
    o->priority = md == 2 || md == 3 ? (double)ac4_u(b, 5) / 31.0 : 1.0;
}

static void object_render_info(ac4_bits *b, ac4_obj *o, int all_new,
                               int no_delta)
{
    int other, zone, pos;
    if (all_new) {
        other = zone = pos = 1;
    } else {
        other = (int)ac4_bit(b);
        zone = (int)ac4_bit(b);
        pos = (int)ac4_bit(b);
    }
    if (pos) {
        int diff = no_delta ? 0 : (int)ac4_bit(b);
        static const int Z[3] = { 0, 0, 0 };
        if (diff) {
            int dx = s3(ac4_u(b, 3)), dy = s3(ac4_u(b, 3)),
                dz = s3(ac4_u(b, 3));
            set_pos(o, clip3i(0, 62, o->px + dx), clip3i(0, 62, o->py + dy),
                    clip3i(-15, 15, o->pz + dz), Z);
        } else {
            int px = (int)ac4_u(b, 6), py = (int)ac4_u(b, 6);
            int sign = ac4_bit(b) ? 1 : -1, pz = (int)ac4_u(b, 4);
            set_pos(o, px, py, sign * pz, Z);
        }
    }
    if (zone && ac4_bit(b) == 0) {      /* b_grouped_zone_defaults */
        if (ac4_u(b, 3) & 4)
            o->zone_mask = (int)ac4_u(b, 3);
    }
    if (other && ac4_bit(b) == 0) {     /* b_grouped_other_defaults */
        uint32_t m = ac4_u(b, 4);
        if (m & 1) {
            if (ac4_bit(b) == 0) {
                ac4_u(b, 5);
            } else {
                ac4_u(b, 5);
                ac4_u(b, 5);
                ac4_u(b, 5);
            }
        }
        if (m & 2) {
            ac4_u(b, 3);
            ac4_u(b, 2);
        }
        if ((m & 4) && ac4_bit(b) == 0)
            ac4_u(b, 4);
        if (m & 8) {
            uint32_t dm = ac4_u(b, 2);
            if (dm == 0)
                ac4_u(b, 2);
            else if (dm & 2)
                ac4_u(b, 6);
        }
    }
}

static void ext_prec_pos(ac4_bits *b, int *e)
{
    static const int EP[4] = { 1, 2, -1, -2 };
    uint32_t pres = ac4_u(b, 3);
    e[0] = e[1] = e[2] = 0;
    if (pres & 4)
        e[0] = EP[ac4_u(b, 2)];
    if (pres & 2)
        e[1] = EP[ac4_u(b, 2)];
    if (pres & 1)
        e[2] = EP[ac4_u(b, 2)];
}

static void object_info_block(ac4_bits *b, ac4_obj *o, int no_delta,
                              int dyn)
{
    int not_active = (int)ac4_bit(b), basic, render;
    o->active = !not_active;
    /* 0 DEFAULT, 1 ALL_NEW, 2 REUSE, 3 PART_REUSE */
    basic = not_active ? 0 : no_delta ? 1 : (ac4_bit(b) ? 2 : 1);
    if (basic == 1)
        object_basic_info(b, o);
    else if (basic == 0) {
        o->gain = -INFINITY;
        o->priority = 0.0;
    }
    if (not_active)
        render = 0;
    else if (dyn)
        render = no_delta ? 1 : (ac4_bit(b) ? 2 : (ac4_bit(b) ? 3 : 1));
    else
        render = 0;
    if (render == 1 || render == 3)
        object_render_info(b, o, render == 1, no_delta);
    else if (render == 0)
        obj_default_render(o);
    if (ac4_bit(b)) {                   /* b_add_table_data */
        size_t size = ac4_u(b, 4) + 1, p0 = b->p;
        ac4_bit(b);                     /* b_obj_trim_disable */
        if (!not_active && dyn && ac4_bit(b)) {
            int e[3];
            ext_prec_pos(b, e);
            set_pos(o, o->px, o->py, o->pz, e);
        }
        if (ac4_bit(b)) {               /* b_headphone */
            ac4_u(b, 2);
            ac4_bit(b);
        }
        if (b->p - p0 > 8 * size)
            ac4_fail(b, AC4E_ELEMENT_PARSE);
        b->p = p0 + 8 * size;
    }
}

/* oamd_dyndata_single without alternative metadata (b_alternative 0);
   blk[k][i]: object i after block k (blk_n objects a row) */
static void dyndata_single(ac4_bits *b, ac4_obj *objs, int n,
                           const signed char *type, int n_blocks, int ifr,
                           ac4_obj *blk, int blk_n)
{
    int i, k;
    for (i = 0; i < n; i++) {
        int dyn = type[i] == AC4_OBJ_DYN;
        for (k = 0; k < n_blocks; k++) {
            object_info_block(b, &objs[i], ifr && k == 0, dyn);
            if (k < AC4_OAMD_NB)
                blk[k * blk_n + i] = objs[i];
        }
    }
}

/* ajoc_huff_data -> diff_type; the values (the symbol minus cb_off) in
   v[0 .. nb) */
static int ajoc_huff(ac4_bits *b, const ac4_huff *bk, int wet, int fine,
                     int nb, int dfonly, int *v)
{
    int diff = dfonly ? 0 : (int)ac4_bit(b), i;
    const ac4_huff *h = &bk[6 * wet + 3 * fine];
    if (diff == 0) {
        v[0] = ac4_huff_decode(&h[0], b);
        for (i = 1; i < nb; i++)
            v[i] = ac4_huff_decode(&h[1], b);
    } else {
        /* cb_off (len - 1) / 2 of the DT books (Annex A.1.1) */
        int off = wet ? (fine ? 40 : 20) : (fine ? 100 : 50);
        for (i = 0; i < nb; i++)
            v[i] = ac4_huff_decode(&h[2], b) - off;
    }
    return diff;
}

/* ajoc.quantised / _diff for one matrix element: present (sent), its
   values, prev (the held values, *pn of them; updated) -> q */
static void q_diff(int sent, int diff, const int *v, int nb, int nq,
                   unsigned char *prev, unsigned char *pn,
                   unsigned char *q)
{
    int mid = (nq - 1) / 2, i;
    if (!sent) {
        for (i = 0; i < nb; i++)
            q[i] = (unsigned char)mid;
    } else if (diff == 0) {
        int x = v[0] < 0 ? 0 : v[0] > nq - 1 ? nq - 1 : v[0];
        q[0] = (unsigned char)x;
        for (i = 1; i < nb; i++) {
            x = (x + v[i]) % nq;
            if (x < 0)
                x += nq;
            q[i] = (unsigned char)x;
        }
    } else {
        for (i = 0; i < nb; i++) {
            int base = *pn == nb ? prev[i] : mid, x = base + v[i];
            q[i] = (unsigned char)(x < 0 ? 0 : x > nq - 1 ? nq - 1 : x);
        }
    }
    memcpy(prev, q, (size_t)nb);
    *pn = (unsigned char)nb;
}

static void ajoc_side(ac4_bits *b, const ac4_huff *bk, int n_dmx,
                      long n_umx, ac4_obj_state *st, ac4_ajoc_frame *fr)
{
    static const int NB[8] = { 23, 15, 12, 9, 7, 5, 3, 1 };
    int nd = (int)ac4_u(b, 3), ndp, d, o, ch, dp, nodt;
    unsigned char ss[AC4_AJOC_NU], dryp[AC4_AJOC_NU][AC4_AJOC_ND];
    unsigned char wetp[AC4_AJOC_NU][AC4_AJOC_ND];
    fr->nd = nd;
    for (d = 0; d < nd; d++)
        fr->en[d] = (int)ac4_bit(b);
    for (o = 0; o < n_umx; o++)
        fr->present[o] = (unsigned char)ac4_bit(b);
    ndp = (int)ac4_u(b, 2);
    fr->ndp = ndp;
    for (dp = 0; dp < ndp; dp++) {
        fr->start[dp] = (int)ac4_u(b, 5);
        fr->ramp[dp] = (int)ac4_u(b, 6) + 1;
    }
    for (o = 0; o < n_umx; o++) {
        fr->nb[o] = 0;
        fr->qs[o] = ss[o] = 0;
        for (ch = 0; ch < n_dmx; ch++)
            dryp[o][ch] = 1;
        for (d = 0; d < nd; d++)
            wetp[o][d] = 1;
        if (ndp && fr->present[o]) {
            fr->nb[o] = (unsigned char)NB[ac4_u(b, 3)];
            fr->qs[o] = (unsigned char)ac4_bit(b);
            ss[o] = (unsigned char)ac4_bit(b);
            if (ss[o]) {
                for (ch = 0; ch < n_dmx; ch++)
                    dryp[o][ch] = (unsigned char)ac4_bit(b);
                for (d = 0; d < nd; d++)
                    wetp[o][d] = fr->en[d] ? (unsigned char)ac4_bit(b) : 0;
            }
        }
    }
    nodt = (int)ac4_bit(b);
    for (o = 0; o < n_umx; o++) {
        int nb = fr->nb[o], qs = fr->qs[o];
        int nqd = qs ? 51 : 101, nqw = qs ? 21 : 41;
        if (!fr->present[o])
            continue;
        for (dp = 0; dp < ndp; dp++) {
            int dfonly = dp == 0 && nodt, v[23], df, sent;
            for (ch = 0; ch < n_dmx; ch++) {
                sent = !ss[o] || dryp[o][ch];
                df = sent ? ajoc_huff(b, bk, 0, !qs, nb, dfonly, v) : 0;
                q_diff(sent, df, v, nb, nqd, st->qd[o][ch], &st->nd_[o][ch],
                       fr->q[dp][o][ch]);
            }
            for (d = 0; d < nd; d++) {
                sent = !ss[o] || wetp[o][d];
                df = sent ? ajoc_huff(b, bk, 1, !qs, nb, dfonly, v) : 0;
                q_diff(sent, df, v, nb, nqw, st->qw[o][d], &st->nw_[o][d],
                       fr->q[dp][o][AC4_AJOC_ND + d]);
            }
        }
    }
}

static void dmx_de_data(ac4_bits *b, int n_dmx, long n_umx,
                        ac4_obj_state *st)
{
    int cfg = (int)ac4_bit(b), keep = (int)ac4_bit(b), o, k, j;
    if (cfg) {
        ac4_u(b, 2);
        st->n_dlg = 0;
        for (o = 0; o < n_umx; o++)
            if (ac4_bit(b))
                st->n_dlg++;
    }
    if (!keep)
        for (k = 0; k < st->n_dlg; k++)
            for (j = 0; j < n_dmx; j++)
                if (ac4_bit(b) && ac4_u(b, 3) != 7)
                    ac4_bit(b);
}

void ac4_ajoc_input_order(int n_fb, int *pos)
{
    int i;
    if (n_fb > 3) {
        int n_off = 2 + n_fb % 2;
        for (i = 0; i < n_off; i++)
            pos[i] = n_fb - n_off + i;
        for (i = 0; i < n_fb - n_off; i++)
            pos[n_off + i] = i;
    } else {
        for (i = 0; i < n_fb; i++)
            pos[i] = i;
    }
}

void ac4_ajoc_tail(ac4_bits *b, const ac4_huff *bk, const ac4_toc_sub *info,
                   int ifr, ac4_obj_state *st, ac4_ajoc_frame *fr)
{
    int lfe = info->lfe, n_dmx = info->n_dmx + (lfe ? 1 : 0), i, nb;
    long n_umx = info->n_umx + (lfe ? 1 : 0);
    signed char dt[17], ut[65];
    for (i = 0; i < n_dmx; i++)
        dt[i] = lfe && i == 0 ? RN_LFE : info->dmx_type[i - (lfe ? 1 : 0)];
    for (i = 0; i < n_umx; i++)
        ut[i] = lfe && i == 0 ? RN_LFE : info->umx_type[i - (lfe ? 1 : 0)];
    if (!st->init) {
        for (i = 0; i < 17; i++)
            ac4_obj_init(&st->dmx[i]);
        for (i = 0; i < 65; i++)
            ac4_obj_init(&st->umx[i]);
        st->blocks = 1;
        st->init = 1;
    }
    fr->n_dev = 0;
    if (!info->static_dmx) {
        if (ac4_bit(b))                 /* b_dmx_timing */
            st->blocks = oamd_timing_data(b, &st->dt);
        nb = st->blocks;
        dyndata_single(b, st->dmx, n_dmx, dt, nb, ifr, &fr->dblk[0][0], 17);
        if (nb > AC4_OAMD_NB)
            nb = AC4_OAMD_NB;
        block_updates(&st->dt, nb, fr->dev);
        fr->n_dev = nb;
        if (nb == 0) {
            memcpy(fr->dblk[0], st->dmx, sizeof(fr->dblk[0]));
            fr->dev[0][0] = fr->dev[0][1] = 0;
            fr->n_dev = 1;
        }
        if (ac4_bit(b)) {               /* b_oamd_extension_present */
            size_t bits = (size_t)((ac4_vb(b, 3) + 1) * 8), p0 = b->p;
            if (ac4_bit(b))             /* ajoc_bed_info */
                ac4_u(b, 3);
            if (b->p - p0 > bits)
                ac4_fail(b, AC4E_ELEMENT_PARSE);
            b->p = p0 + bits;
        }
    }
    ajoc_side(b, bk, info->n_dmx, info->n_umx, st, fr);
    dmx_de_data(b, info->n_dmx, info->n_umx, st);
    if (ac4_bit(b)) {                   /* b_umx_timing */
        st->blocks = oamd_timing_data(b, &st->ut);
    } else if (ac4_bit(b)) {            /* b_derive_timing_from_dmx */
        st->ut = st->dt;
    }
    nb = st->blocks;
    dyndata_single(b, st->umx, (int)n_umx, ut, nb, ifr, &fr->ublk[0][0], 65);
    if (nb > AC4_OAMD_NB)
        nb = AC4_OAMD_NB;
    block_updates(&st->ut, nb, fr->uev);
    fr->n_uev = nb;
    if (nb == 0) {
        memcpy(fr->ublk[0], st->umx, sizeof(fr->ublk[0]));
        fr->uev[0][0] = fr->uev[0][1] = 0;
        fr->n_uev = 1;
    }
}

void ac4_objs_tail(ac4_bits *b, const ac4_toc_group *g,
                   const ac4_toc_sub *s, int ndot, int alternative,
                   ac4_obj_state *st, ac4_ajoc_frame *fr)
{
    int n = g->n_obj, i, k, nb, k0 = s->oamd_offset, nk = s->n_types;
    if (n > 64 || k0 + nk > n || nk > 17)
        ac4_fail(b, AC4E_ELEMENT_PARSE);
    if (!st->init) {
        for (i = 0; i < 65; i++)
            ac4_obj_init(&st->umx[i]);
        st->blocks = 1;
        st->n_grp = n;
        st->init = 1;
    }
    if (st->n_grp != n)                 /* object count changed */
        ac4_fail(b, AC4E_ELEMENT_PARSE);
    if (ac4_bit(b))                     /* b_oamd_common_data_present */
        ac4_oamd_common_data(b);
    if (ac4_bit(b))                     /* b_oamd_timing_present */
        st->blocks = oamd_timing_data(b, &st->dt);
    nb = st->blocks;
    if (!alternative) {
        /* oamd_dyndata_multi (6.2.8.4) */
        for (i = 0; i < n; i++) {
            int dyn = g->obj_type[i] == AC4_OBJ_DYN && !g->obj_lfe[i];
            for (k = 0; k < nb; k++) {
                if (!g->obj_ajoc[i])
                    object_info_block(b, &st->umx[i], ndot && k == 0, dyn);
                if (k < AC4_OAMD_NB && i >= k0 && i < k0 + nk)
                    fr->dblk[k][i - k0] = st->umx[i];
            }
        }
    }
    if (nb > AC4_OAMD_NB)
        nb = AC4_OAMD_NB;
    if (alternative || nb == 0) {
        memcpy(fr->dblk[0], &st->umx[k0], sizeof(ac4_obj) * (size_t)nk);
        fr->dev[0][0] = fr->dev[0][1] = 0;
        fr->n_dev = 1;
        return;
    }
    block_updates(&st->dt, nb, fr->dev);
    fr->n_dev = nb;
}
