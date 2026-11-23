/* SPDX-License-Identifier: Apache-2.0 */
/* Dialogue enhancement: the audio substream's metadata() up to its DE
 * data (port of ac4/metadata.py basic_metadata, extended_metadata, the
 * tools block's de_config / de_data; TS 103 190-2 6.2.7) and de_method 0
 * in the QMF domain (port of ac4/de.py; TS 103 190-1 5.7.8). */
#include <math.h>
#include <string.h>

#include "ac4_internal.h"

/* Table 171 */
static const int DE_NR[8] = { 0, 1, 1, 2, 1, 2, 2, 3 };
/* cb_off of DE_HCB_ABS_0, DIFF_0, ABS_1, DIFF_1 */
static const int DE_OFF[4] = { 0, 31, 30, 60 };
/* Table 209 */
static const double PAR_CI[32] = {
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
    1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
    7.5, 8.0, 8.5, 9.0
};
/* Table 173: first QMF band of each parameter band (the last runs to 63) */
static const int BAND_FIRST[8] = { 0, 1, 2, 4, 7, 11, 17, 27 };

/* further_loudness_info (6.2.7.3) of a substream: sus_ver 1, not the
   presentation's */
static void sub_loudness(ac4_bits *b)
{
    int i;
    ac4_bit(b);                           /* b_loudcorr_dialgate */
    if (ac4_bit(b))
        ac4_u(b, 11);
    if (ac4_bit(b)) {
        ac4_u(b, 11);
        ac4_u(b, 3);
    }
    for (i = 0; i < 4; i++)
        if (ac4_bit(b))
            ac4_u(b, 11);
    if (ac4_bit(b)) {
        ac4_u(b, 10);
        ac4_u(b, 3);
    }
    for (i = 0; i < 2; i++)
        if (ac4_bit(b))
            ac4_u(b, 11);
    if (ac4_bit(b))                       /* rtll_comp */
        ac4_u(b, 8);
    if (ac4_bit(b)) {
        uint64_t n = ac4_u(b, 5);
        if (n == 31)
            n += ac4_vb(b, 4);
        if (b->p + n > b->nbits)
            ac4_fail(b, b->eof_code);
        b->p += (size_t)n;
    }
}

static void basic_metadata(ac4_bits *b, int cm)
{
    if (ac4_bit(b)) {                     /* b_more_basic_metadata */
        if (ac4_bit(b)) {                 /* b_substream_loudness_info */
            ac4_u(b, 8);
            if (ac4_bit(b))
                sub_loudness(b);
        }
        if (cm == 1 && ac4_bit(b)) {
            ac4_u(b, 3);
            ac4_u(b, 2);
        }
        if (cm > 1) {
            if (cm == 3 || cm == 4) {
                if (ac4_bit(b))
                    ac4_u(b, 3);
                if (ac4_bit(b))
                    ac4_u(b, 4);
            }
            if (cm >= 5 && ac4_bit(b)) {
                if (cm == 5 || cm == 6 || cm >= 11)
                    ac4_u(b, 2);
                else if (cm == 9 || cm == 10)
                    ac4_bit(b);
            }
            ac4_u(b, 2);
            ac4_bit(b);
            ac4_bit(b);
        }
        if (ac4_bit(b))                   /* b_dc_blocking */
            ac4_bit(b);
    }
}

static void two_level(ac4_bits *b)
{
    if (ac4_bit(b))
        ac4_bit(b);
}

static void extended_metadata(ac4_bits *b, int cm)
{
    if (ac4_bit(b)) {                     /* b_dialog */
        if (ac4_bit(b))
            ac4_u(b, 2);
        if (ac4_bit(b)) {
            ac4_u(b, 8);
            if (cm != 0) {
                ac4_u(b, 8);
                ac4_u(b, 2);
            }
        }
    }
    if (ac4_bit(b)) {                     /* b_channels_classifier */
        if (cm == 0 || (cm >= 2 && cm <= 15))
            two_level(b);
        if (cm >= 1 && cm <= 15) {
            two_level(b);
            two_level(b);
        }
        if (cm >= 3 && cm <= 15)
            ac4_u(b, 2);
        if (cm == 5 || cm == 6 || (cm >= 11 && cm <= 15))
            ac4_u(b, 2);
        if (cm == 7 || cm == 8 || cm == 15)
            ac4_u(b, 2);
        if (cm == 9 || cm == 10)
            ac4_u(b, 2);
        if (cm == 4 || cm == 6 || cm == 8 || cm == 10 || cm == 12
            || cm == 14 || cm == 15)
            ac4_bit(b);
    }
    if (ac4_bit(b))
        ac4_u(b, 4);
}

static int diff(ac4_bits *b, const ac4_huff *const books[4], int k)
{
    return ac4_huff_decode(books[k], b) - DE_OFF[k];
}

/* de_data (6.2.7.6; metadata.de_data) */
static void de_data(ac4_bits *b, int iframe, ac4_de_state *st,
                    const ac4_huff *const books[4], ac4_de_entry *out)
{
    int method = st->method, n_ch = DE_NR[st->channel_config];
    int prev[3][8], par[3][8], ms = 0, t = method % 2, ch, band, ref = 0;
    if (n_ch == 0)
        return;                           /* no parameters: none */
    if ((method == 1 || method == 3) && n_ch > 1) {
        int keep_pos = iframe ? 0 : (int)ac4_bit(b);
        if (!keep_pos) {
            ac4_u(b, 5);                  /* de_mix_coef1_idx */
            if (n_ch == 3)
                ac4_u(b, 5);
        }
    }
    if (st->has_prev)
        memcpy(prev, st->prev, sizeof(prev));
    else
        memset(prev, 0, sizeof(prev));
    if (!iframe && ac4_bit(b)) {          /* de_keep_data_flag */
        memcpy(out->par, prev, sizeof(prev));
        out->ms = 0;
        out->kind = AC4_DE_DATA;
        return;
    }
    if ((method == 0 || method == 2) && n_ch == 2)
        ms = (int)ac4_bit(b);             /* de_ms_proc_flag */
    memset(par, 0, sizeof(par));
    for (ch = 0; ch < n_ch - ms; ch++) {
        if (iframe && ch == 0) {
            par[0][0] = diff(b, books, 2 * t);
            ref = par[0][0];
            for (band = 1; band < 8; band++) {
                par[0][band] = ref + diff(b, books, 2 * t + 1);
                ref = par[0][band];
            }
        } else {
            for (band = 0; band < 8; band++) {
                if (iframe) {
                    par[ch][band] = ref + diff(b, books, 2 * t + 1);
                    ref = par[ch][band];
                } else {
                    par[ch][band] = prev[ch][band]
                                    + diff(b, books, 2 * t + 1);
                }
            }
        }
        ref = par[ch][0];
    }
    memcpy(st->prev, par, sizeof(par));
    st->has_prev = 1;
    if (method >= 2)
        ac4_u(b, 5);                      /* de_signal_contribution */
    memcpy(out->par, par, sizeof(par));
    out->ms = ms;
    out->kind = AC4_DE_DATA;
}

void ac4_de_metadata(const uint8_t *sub, size_t len, size_t pos,
                     int ch_mode, int iframe, ac4_de_state *st,
                     const ac4_huff *const books[4], ac4_de_entry *out)
{
    jmp_buf jb;
    ac4_bits b;
    ac4_de_entry e;
    volatile size_t start = 0, size = 0;
    volatile int stage = 0;
    memset(out, 0, sizeof(*out));
    memset(&e, 0, sizeof(e));
    b.d = sub;
    b.nbits = len * 8;
    b.p = pos;
    b.jb = &jb;
    b.eof_code = AC4E_ELEMENT_PARSE;
    if (setjmp(jb) != 0) {
        /* basic / extended metadata or the tools size did not parse: the
           stream's except (de_par_prev reset); the tools block: an error
           (the state as read so far) */
        if (stage == 0)
            st->has_prev = 0;
        return;
    }
    if (pos > b.nbits)
        ac4_fail(&b, AC4E_ELEMENT_PARSE);
    basic_metadata(&b, ch_mode);
    extended_metadata(&b, ch_mode);
    size = ac4_u(&b, 7);
    if (ac4_bit(&b))
        size += ac4_vb(&b, 3) << 7;
    start = b.p;
    stage = 1;
    if (ac4_bit(&b)) {                    /* b_de_data_present */
        if (iframe || ac4_bit(&b)) {      /* de_config */
            st->method = (int)ac4_u(&b, 2);
            st->max_gain = (int)ac4_u(&b, 2);
            st->channel_config = (int)ac4_u(&b, 3);
            st->has_cfg = 1;
        }
        if (!st->has_cfg)
            return;                       /* de_data before de_config */
        de_data(&b, iframe, st, books, &e);
        if (ch_mode == 13 || ch_mode == 14)
            return;                       /* not decoded here */
    } else {
        st->has_prev = 0;
    }
    if (b.p != start + size)
        return;                           /* "tools read x of y bits" */
    if (e.kind == AC4_DE_DATA) {
        e.method = st->method;
        e.max_gain = st->max_gain;
        e.channel_config = st->channel_config;
        *out = e;
    }
}

void ac4_de_frame(const ac4_de_entry *e, double level, ac4_de_mat *m)
{
    static const int BITS[3] = { 4, 2, 1 };    /* L, R, C (Table 171) */
    int chans[3], n = 0, i, sb;
    double gmax, g;
    memset(m, 0, sizeof(*m));
    if (!e || e->kind != AC4_DE_DATA || level == 0.0 || e->method != 0)
        return;
    gmax = (e->max_gain + 1) * 3.0;
    g = pow(10.0, (level < gmax ? level : gmax) / 20.0) - 1.0;
    for (i = 0; i < 3; i++)
        if (e->channel_config & BITS[i])
            chans[n++] = i;
    for (i = 0; i < n; i++) {
        const int *row = e->par[(e->ms && n == 2) ? 0 : i];
        double *dst = (e->ms && n == 2) ? m->msg : m->g[chans[i]];
        int band = 0;
        for (sb = 0; sb < 64; sb++) {
            int idx;
            while (band < 7 && sb >= BAND_FIRST[band + 1])
                band++;
            idx = row[band];
            if (idx < 0)
                idx = 0;
            if (idx > 31)
                idx = 31;
            dst[sb] = 1.0 + g * PAR_CI[idx];
        }
        if (e->ms && n == 2)
            break;
        m->has[chans[i]] = 1;
    }
    if (e->ms && n == 2) {
        m->ms = 1;
        m->msa = chans[0];
        m->msb = chans[1];
    }
    m->on = n > 0;
}

/* own and partner gain of channel k (0 L, 1 R, 2 C) at subband sb
   (de._full) */
static void full(const ac4_de_mat *m, int k, int sb, double *own,
                 double *cross)
{
    if (m->ms && (k == m->msa || k == m->msb)) {
        *own = (m->msg[sb] + 1.0) / 2.0;
        *cross = (m->msg[sb] - 1.0) / 2.0;
    } else {
        *own = m->has[k] ? m->g[k][sb] : 1.0;
        *cross = 0.0;
    }
}

/* de.apply for one frame: the channels ch[0..n-1] (0 L, 1 R, 2 C; their
   QMF slots re[i][s], im[i][s] for s < ns, slot s at frame position
   s + off) from prev to cur; partner[i]: the index in ch of channel i's
   M/S partner, or -1 */
void ac4_de_apply(const ac4_de_mat *prev, const ac4_de_mat *cur, int n,
                  const int *ch, double (*const re[3])[64],
                  double (*const im[3])[64], int ns, int off, int nts)
{
    const ac4_de_mat *pm = cur->ms ? cur : prev;
    int s, i, sb, partner[3];
    double tr[3][64], ti[3][64];
    if (!cur->on && !prev->on)
        return;
    for (i = 0; i < n; i++) {
        int j, other = -1;
        partner[i] = -1;
        if (pm->ms && (ch[i] == pm->msa || ch[i] == pm->msb))
            other = ch[i] == pm->msa ? pm->msb : pm->msa;
        for (j = 0; j < n; j++)
            if (other >= 0 && ch[j] == other)
                partner[i] = j;
    }
    for (s = 0; s < ns; s++) {
        double w = ((double)(s + off) + 0.5) / nts;
        for (i = 0; i < n; i++) {
            memcpy(tr[i], re[i][s], sizeof(tr[i]));
            memcpy(ti[i], im[i][s], sizeof(ti[i]));
        }
        for (i = 0; i < n; i++)
            for (sb = 0; sb < 64; sb++) {
                double o0, c0, o1, c1, own, yr, yi;
                full(prev, ch[i], sb, &o0, &c0);
                full(cur, ch[i], sb, &o1, &c1);
                own = (1.0 - w) * o0 + w * o1;
                yr = own * tr[i][sb];
                yi = own * ti[i][sb];
                if (partner[i] >= 0) {
                    double cr = (1.0 - w) * c0 + w * c1;
                    yr = yr + cr * tr[partner[i]][sb];
                    yi = yi + cr * ti[partner[i]][sb];
                }
                re[i][s][sb] = yr;
                im[i][s][sb] = yi;
            }
    }
}
