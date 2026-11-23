/* SPDX-License-Identifier: Apache-2.0 */
/* A-SPX subband tables, A-SPX side information and A-CPL side information
 * (parsing only). Port of ac4/aspx_bands.py (Pseudocodes 67-71),
 * ac4/aspx_envadj.py sbg_lim (Pseudocode 72, main form), ac4/aspx_env.py
 * sbg_maps, ac4/aspx_parse.py (Tables 51-58, Pseudocodes 76, 77, 79) and
 * ac4/acpl_parse.py (Tables 60, 62-65); TS 103 190-1 4.2.10, 4.2.13,
 * 5.7.6.3. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ac4_internal.h"

static const int TPL_HI[23] = { 18, 19, 20, 21, 22, 23, 24, 26, 28, 30, 32,
                                34, 36, 38, 40, 42, 44, 47, 50, 53, 56, 59,
                                62 };
static const int TPL_LO[21] = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                22, 24, 26, 28, 30, 32, 35, 38, 42, 46 };

enum { FIXFIX = 0, FIXVAR = 1, VARFIX = 2, VARVAR = 3 };
static const int MAX_NUM_ENV[4] = { 4, 5, 5, 5 };   /* Table 127 */

/* ------------------------------------------------------------ tables */
/* Pseudocode 71 (aspx_bands.patches); -> 0 or -1 (the Python raises) */
static int patches(ac4_aspx_tab *t, const ac4_aspx_cfg *c)
{
    const int *master = t->master;
    int n_master = t->n_master, len = n_master + 1;
    int sba = t->sba, sbx = t->sbx, span = t->num_sb;
    int msb = sba, usb = sbx, goal_sb = 43;
    int src_low = c->master_freq_scale == 1 ? 4 : 2;
    int sbg, i, guard = 0, n = 0, k;

    if (goal_sb < sbx + span) {
        sbg = 0;
        for (i = 0; i < len; i++) {
            if (master[i] >= goal_sb)
                break;
            sbg = i + 1;
        }
    } else {
        sbg = n_master;
    }
    for (;;) {
        int j = sbg, sb, odd, nn;
        if (++guard > 64)
            return -1;
        if (j < 0 || j >= len)
            return -1;
        sb = master[j];
        odd = ((sb - 2 + sba) % 2 + 2) % 2;
        while (sb > sba - src_low + msb - odd) {
            j -= 1;
            /* Python indexing: a negative index counts from the end */
            if (j < -len)
                return -1;
            sb = master[j < 0 ? j + len : j];
            odd = ((sb - 2 + sba) % 2 + 2) % 2;
        }
        nn = sb - usb > 0 ? sb - usb : 0;
        if (nn > 0) {
            if (n >= AC4_ASPX_MAX_PATCH)
                return -1;
            t->patch_num_sb[n] = nn;
            t->patch_start_sb[n] = sba - odd - nn;
            n++;
            usb = msb = sb;
        } else {
            msb = sbx;
        }
        if (sbg < 0 || sbg >= len)
            return -1;
        if (master[sbg] - sb < 3)
            sbg = n_master;
        if (sb == sbx + span)
            break;
    }
    if (n > 1 && t->patch_num_sb[n - 1] < 3)
        n--;
    t->n_patches = n;
    t->patches[0] = sbx;
    for (k = 0; k < n; k++)
        t->patches[k + 1] = t->patches[k] + t->patch_num_sb[k];
    return 0;
}

static int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static int in_list(const int *v, int n, int x)
{
    int i;
    for (i = 0; i < n; i++)
        if (v[i] == x)
            return 1;
    return 0;
}



int ac4_aspx_tables(const ac4_aspx_cfg *c, int xover, ac4_aspx_tab *t)
{
    const int *tpl = c->master_freq_scale == 1 ? TPL_HI : TPL_LO;
    int tpl_len = c->master_freq_scale == 1 ? 23 : 21;
    int n = (c->master_freq_scale == 1 ? 22 : 20) - 2 * c->start_freq
            - 2 * c->stop_freq;
    int base = 2 * c->start_freq, i, s, n_noise;
    int cand[2 * AC4_ASPX_MAX_SBG + AC4_ASPX_MAX_PATCH + 2], nc = 0;
    memset(t, 0, sizeof(*t));
    /* Pseudocode 67 */
    if (n < 0)
        return -1;                     /* empty master table: IndexError */
    if (base + n >= tpl_len || n + 1 > AC4_ASPX_MAX_SBG)
        return -1;
    for (i = 0; i <= n; i++)
        t->master[i] = tpl[base + i];
    t->n_master = n;
    /* Pseudocode 68: aspx_xover_subband_offset groups into the master
       table (fewer than one group left: no usable table, the Python's
       aspx.config) */
    t->n_hi = n - xover;
    if (xover < 0 || t->n_hi < 1)
        return -1;
    for (s = 0; s <= t->n_hi; s++)
        t->hi[s] = t->master[s + xover];
    t->sbx = t->hi[0];
    t->num_sb = t->hi[t->n_hi] - t->sbx;
    /* Pseudocode 69 */
    t->n_lo = t->n_hi - t->n_hi / 2;
    t->lo[0] = t->hi[0];
    for (s = 1; s <= t->n_lo; s++)
        t->lo[s] = t->n_hi % 2 == 0 ? t->hi[2 * s] : t->hi[2 * s - 1];
    /* Pseudocode 70: log2(sbz / sbx), from the crossover */
    t->sba = t->master[0];
    t->sbz = t->master[n];
    n_noise = (int)floor((double)c->noise_sbg
                         * log2((double)t->sbz / (double)t->sbx) + 0.5);
    if (n_noise < 1)
        n_noise = 1;
    if (n_noise > AC4_ASPX_MAX_NOISE)
        return -1;
    t->n_noise = n_noise;
    {
        int idx[AC4_ASPX_MAX_NOISE + 1];
        idx[0] = 0;
        t->noise[0] = t->lo[0];
        for (s = 1; s <= n_noise; s++) {
            int num = t->n_lo - idx[s - 1], den = n_noise + 1 - s;
            /* Python floor division (num >= 0, den >= 1) */
            idx[s] = idx[s - 1] + num / den;
            if (idx[s] < 0 || idx[s] > t->n_lo)
                return -1;
            t->noise[s] = t->lo[idx[s]];
        }
    }
    /* the Python's aspx.config rule */
    if (t->n_hi < 1 || t->n_lo < 1 || t->num_sb < 1)
        return -1;
    if (patches(t, c) != 0)
        return -1;
    /* patches must stay inside the 64 subbands (the Python would raise
       in hf_generate and render the whole pass core-only) */
    for (i = 0; i < t->n_patches; i++) {
        int st = t->patch_start_sb[i], nn = t->patch_num_sb[i];
        if (st < 0 || st + nn > 64 || t->patches[i] + nn > 64)
            return -1;
    }
    if (t->sbx + t->num_sb > 64)
        return -1;
    /* Pseudocode 72, main form (aspx_envadj.sbg_lim) */
    for (i = 0; i <= t->n_lo; i++)
        if (!in_list(cand, nc, t->lo[i]))
            cand[nc++] = t->lo[i];
    for (i = 0; i <= t->n_patches; i++)
        if (!in_list(cand, nc, t->patches[i]))
            cand[nc++] = t->patches[i];
    qsort(cand, (size_t)nc, sizeof(int), cmp_int);
    t->n_lim = 0;
    t->lim[t->n_lim++] = cand[0];
    for (i = 1; i < nc; i++) {
        int b = cand[i], last = t->lim[t->n_lim - 1];
        int b_patch = in_list(t->patches, t->n_patches + 1, b);
        int l_patch = in_list(t->patches, t->n_patches + 1, last);
        if (log2((double)b / (double)last) < 0.245) {
            if (b_patch && !l_patch)
                t->lim[t->n_lim - 1] = b;
            continue;
        }
        if (t->n_lim >= AC4_ASPX_MAX_LIM)
            return -1;
        t->lim[t->n_lim++] = b;
    }
    /* Pseudocode 80's maps (aspx_env.sbg_maps) */
    {
        int sbg, sbg_low = 0;
        for (i = 0; i <= t->n_lo; i++)
            t->low2high[i] = 0;
        for (sbg = 0; sbg < t->n_hi; sbg++) {
            if (sbg_low + 1 < t->n_lo + 1
                && t->lo[sbg_low + 1] == t->hi[sbg]) {
                sbg_low++;
                t->low2high[sbg_low] = sbg;
            }
            t->high2low[sbg] = sbg_low;
        }
    }
    return 0;
}

/* ------------------------------------------------------------ books */
int ac4_aspx_books_init(ac4_aspx_books *bk)
{
    int i;
    memset(bk, 0, sizeof(*bk));
    for (i = 0; i < AC4T_N_ASPX_BOOKS; i++) {
        if (ac4_huff_build(&bk->h[i], &AC4T_ASPX_BOOKS[i]) != 0)
            return -1;
        /* aspx_parse.CENTRE: F0 books raw, DF/DT centred */
        bk->centre[i] = i % 3 == 0 ? 0 : (AC4T_ASPX_BOOKS[i].n - 1) / 2;
    }
    for (i = 0; i < AC4T_N_ACPL_BOOKS; i++)
        if (ac4_huff_build(&bk->acpl[i], &AC4T_ACPL_BOOKS[i]) != 0)
            return -1;
    return 0;
}

void ac4_aspx_books_free(ac4_aspx_books *bk)
{
    int i;
    for (i = 0; i < AC4T_N_ASPX_BOOKS; i++)
        ac4_huff_free(&bk->h[i]);
    for (i = 0; i < AC4T_N_ACPL_BOOKS; i++)
        ac4_huff_free(&bk->acpl[i]);
}

/* ------------------------------------------------------------ parse */
static int int_class(ac4_bits *b)
{
    if (!ac4_bit(b))
        return FIXFIX;
    if (!ac4_bit(b))
        return FIXVAR;
    return ac4_bit(b) ? VARVAR : VARFIX;
}

static int freq_res(const int *borders, int atsg, int tsg_ptr, int mode,
                    int nats)
{
    if (mode == 1)
        return 0;
    if (mode == 3)
        return 1;
    if (atsg < tsg_ptr && nats > 8)
        return 1;
    return (double)(borders[atsg + 1] - borders[atsg])
           > ((double)nats / 6.0 + 3.25) ? 1 : 0;
}

/* Pseudocode 92's p_sine_at_end (aspx_parse.sine_at_end): 0 when the
   channel's previous interval had aspx_tsg_ptr == num_atsg_sig, else -1;
   this interval's pair is kept for the next */
static int sine_at_end(ac4_aspx_state *st, int grp, int ch, int tsg_ptr,
                       int num_env)
{
    int r = (st->has_ptr[grp][ch]
             && st->ptr[grp][ch] == st->nenv[grp][ch]) ? 0 : -1;
    st->ptr[grp][ch] = tsg_ptr;
    st->nenv[grp][ch] = num_env;
    st->has_ptr[grp][ch] = 1;
    return r;
}

/* Table 194: tab_border for num_aspx_timeslots 6, 8, 12, 15, 16 and
   1, 2, 4 envelopes -> 0, or -1 when the table has no entry */
static int tab_border(int nats, int n, int *a)
{
    static const int T6[3][5] = { { 0, 6 }, { 0, 3, 6 }, { 0, 2, 3, 4, 6 } };
    static const int T8[3][5] = { { 0, 8 }, { 0, 4, 8 }, { 0, 2, 4, 6, 8 } };
    static const int T12[3][5] = { { 0, 12 }, { 0, 6, 12 },
                                   { 0, 3, 6, 9, 12 } };
    static const int T15[3][5] = { { 0, 15 }, { 0, 8, 15 },
                                   { 0, 4, 8, 12, 15 } };
    static const int T16[3][5] = { { 0, 16 }, { 0, 8, 16 },
                                   { 0, 4, 8, 12, 16 } };
    const int (*tb)[5];
    int i, k = n == 1 ? 0 : n == 2 ? 1 : n == 4 ? 2 : -1;
    switch (nats) {
    case 6: tb = T6; break;
    case 8: tb = T8; break;
    case 12: tb = T12; break;
    case 15: tb = T15; break;
    case 16: tb = T16; break;
    default: return -1;
    }
    if (k < 0)
        return -1;
    for (i = 0; i <= n; i++)
        a[i] = tb[k][i];
    return 0;
}

static void aspx_framing(ac4_bits *b, const ac4_aspx_cfg *cfg, int nats,
                         int b_iframe, ac4_aspx_state *st, int grp, int ch,
                         ac4_aspx_framing *f)
{
    const int n_rel_bits = nats > 8 ? 2 : 1;
    int ic = int_class(b), num_env, tsg_ptr, e;
    int rel_l[8], rel_r[8], n_l = 0, n_r = 0, var_l = 0, var_r = 0, i;
    int *a = f->borders;
    memset(f, 0, sizeof(*f));
    if (ic == FIXFIX) {
        int envbits = cfg->num_env_bits_fixfix + 1;
        num_env = 1 << ac4_u(b, envbits);
        tsg_ptr = -1;                     /* FIXFIX: no aspx_tsg_ptr */
        if (tab_border(nats, num_env, a) != 0)   /* Tables 194, 127 */
            ac4_fail(b, AC4E_ASPX_NUM_ENV);
    } else {
        int k;
        if (ic == FIXVAR) {
            var_r = (int)ac4_u(b, 2);
            n_r = (int)ac4_u(b, n_rel_bits);
            for (k = 0; k < n_r; k++)
                rel_r[k] = 2 * (int)ac4_u(b, n_rel_bits) + 2;
        } else if (ic == VARFIX) {
            if (b_iframe)
                var_l = (int)ac4_u(b, 2);
            n_l = (int)ac4_u(b, n_rel_bits);
            for (k = 0; k < n_l; k++)
                rel_l[k] = 2 * (int)ac4_u(b, n_rel_bits) + 2;
        } else {
            if (b_iframe)
                var_l = (int)ac4_u(b, 2);
            n_l = (int)ac4_u(b, n_rel_bits);
            for (k = 0; k < n_l; k++)
                rel_l[k] = 2 * (int)ac4_u(b, n_rel_bits) + 2;
            var_r = (int)ac4_u(b, 2);
            n_r = (int)ac4_u(b, n_rel_bits);
            for (k = 0; k < n_r; k++)
                rel_r[k] = 2 * (int)ac4_u(b, n_rel_bits) + 2;
        }
        num_env = n_l + n_r + 1;
        if (num_env > MAX_NUM_ENV[ic])
            ac4_fail(b, AC4E_ASPX_NUM_ENV);
        {
            int ptr_bits = (int)ceil(log((double)(num_env + 2)) / log(2.0));
            tsg_ptr = (int)ac4_u(b, ptr_bits) - 1;
        }
        /* Pseudocode 76 */
        for (i = 0; i <= num_env; i++)
            a[i] = 0;
        if (ic == FIXVAR) {
            a[0] = 0;
            a[num_env] = var_r + nats;
            for (k = 0; k < n_r; k++)
                a[num_env - k - 1] = a[num_env - k] - rel_r[k];
        } else {
            if (b_iframe) {
                a[0] = var_l;
            } else {
                int prev = nats;
                if (st->has[grp][ch])
                    prev = st->stop[grp][ch];
                else
                    st->missing = 1;
                a[0] = prev - nats;
            }
            a[num_env] = ic == VARFIX ? nats : var_r + nats;
            for (k = 0; k < n_l; k++)
                a[k + 1] = a[k] + rel_l[k];
            if (ic == VARVAR)
                for (k = 0; k < n_r; k++)
                    a[num_env - k - 1] = a[num_env - k] - rel_r[k];
        }
        if (a[0] < 0 || a[num_env] > nats + 3)
            ac4_fail(b, AC4E_ASPX_BORDERS);
        for (i = 0; i < num_env; i++)
            if (a[i] >= a[i + 1])
                ac4_fail(b, AC4E_ASPX_BORDERS);
    }
    if (cfg->freq_res_mode == 0) {
        /* Table 53: FIXFIX sends one aspx_freq_res for every envelope
           (Pseudocode 76) */
        if (ic == FIXFIX) {
            int v = (int)ac4_bit(b);
            for (e = 0; e < num_env; e++)
                f->freq_res[e] = v;
        } else {
            for (e = 0; e < num_env; e++)
                f->freq_res[e] = (int)ac4_bit(b);
        }
    } else {
        for (e = 0; e < num_env; e++)
            f->freq_res[e] = freq_res(a, e, tsg_ptr < 0 ? 0 : tsg_ptr,
                                      cfg->freq_res_mode, nats);
    }
    st->stop[grp][ch] = a[num_env];
    st->has[grp][ch] = 1;
    f->tsg_ptr = tsg_ptr;
    f->sine_at_end = sine_at_end(st, grp, ch, tsg_ptr, num_env);
    f->num_env = num_env;
    f->num_noise = num_env > 1 ? 2 : 1;
    f->int_class = ic;
}

static void delta_dir(ac4_bits *b, const ac4_aspx_framing *f, int *ds,
                      int *dn)
{
    int e;
    for (e = 0; e < f->num_env; e++)
        ds[e] = (int)ac4_bit(b);
    for (e = 0; e < f->num_noise; e++)
        dn[e] = (int)ac4_bit(b);
}


static void read_bits(ac4_bits *b, int n, uint8_t *dst)
{
    int i;
    for (i = 0; i < n; i++)
        dst[i] = (uint8_t)ac4_bit(b);
}

/* Tables 55 and 56: TNA modes, add-harmonic, FIC and TIC flags into
   c[0] (and c[1]); with aspx_balance 1 channel 1 takes channel 0's TNA
   modes */
static void hfgen_iwc(ac4_bits *b, const ac4_aspx_tab *t, int two_ch,
                      int balance, int nats, ac4_aspx_chd *c)
{
    int n_noise = t->n_noise, n_hi = t->n_hi, i, k;
    for (i = 0; i < n_noise; i++)
        c[0].tna[i] = (uint8_t)ac4_u(b, 2);  /* aspx_tna_mode ch0 */
    if (two_ch) {
        if (balance == 0)
            for (i = 0; i < n_noise; i++)
                c[1].tna[i] = (uint8_t)ac4_u(b, 2);
        else
            memcpy(c[1].tna, c[0].tna, sizeof(c[0].tna));
        for (k = 0; k < 2; k++)
            if (ac4_bit(b))                   /* aspx_ah_left / _right */
                read_bits(b, n_hi, c[k].ah);
        if (ac4_bit(b)) {                     /* aspx_fic_present */
            for (k = 0; k < 2; k++)
                if (ac4_bit(b)) {             /* aspx_fic_left / _right */
                    c[k].has_fic = 1;
                    read_bits(b, n_hi, c[k].fic);
                }
        }
        if (ac4_bit(b)) {                     /* aspx_tic_present */
            int copy = (int)ac4_bit(b), left = 0, right = 0;
            if (copy == 0) {
                left = (int)ac4_bit(b);
                right = (int)ac4_bit(b);
            }
            if (copy || left) {
                c[0].has_tic = 1;
                read_bits(b, nats, c[0].tic);
            }
            if (right) {
                c[1].has_tic = 1;
                read_bits(b, nats, c[1].tic);
            }
            if (copy) {
                c[1].has_tic = 1;
                memcpy(c[1].tic, c[0].tic, sizeof(c[0].tic));
            }
        }
    } else {
        if (ac4_bit(b))                       /* aspx_ah_present */
            read_bits(b, n_hi, c[0].ah);
        if (ac4_bit(b)) {                     /* aspx_fic_present */
            c[0].has_fic = 1;
            read_bits(b, n_hi, c[0].fic);
        }
        if (ac4_bit(b)) {                     /* aspx_tic_present */
            c[0].has_tic = 1;
            read_bits(b, nats, c[0].tic);
        }
    }
}

/* Pseudocode 79: book index for (SIGNAL|NOISE, qmode, stereo mode) */
static int book_kind(int signal, int qmode, int smode)
{
    if (signal)
        return smode ? (qmode ? 3 : 2) : (qmode ? 1 : 0);
    return smode ? 5 : 4;
}

/* Tables 57 and 58 */
static void ec_data(ac4_bits *b, const ac4_aspx_books *bk,
                    const ac4_aspx_tab *t, int signal, int num_env,
                    const int *fres, int qmode, int smode, const int *dirs,
                    int *n_out, int (*vals)[AC4_ASPX_MAX_SBG > AC4_ASPX_MAX_NOISE
                                            ? AC4_ASPX_MAX_SBG
                                            : AC4_ASPX_MAX_NOISE])
{
    int kind = book_kind(signal, qmode, smode), env;
    for (env = 0; env < num_env; env++) {
        int n_sbg = signal ? (fres[env] ? t->n_hi : t->n_lo) : t->n_noise;
        int k = 0, s;
        if (dirs[env] == 0) {
            const ac4_huff *df = &bk->h[3 * kind + 1];
            int c = bk->centre[3 * kind + 1];
            vals[env][k++] = ac4_huff_decode(&bk->h[3 * kind], b);
            for (s = 1; s < n_sbg; s++)
                vals[env][k++] = ac4_huff_decode(df, b) - c;
        } else {
            const ac4_huff *dt = &bk->h[3 * kind + 2];
            int c = bk->centre[3 * kind + 2];
            for (s = 0; s < n_sbg; s++)
                vals[env][k++] = ac4_huff_decode(dt, b) - c;
        }
        n_out[env] = k;
    }
}

typedef int vrow_t[AC4_ASPX_MAX_SBG > AC4_ASPX_MAX_NOISE ? AC4_ASPX_MAX_SBG
                                                        : AC4_ASPX_MAX_NOISE];

static void ec_sig(ac4_bits *b, const ac4_aspx_books *bk,
                   const ac4_aspx_tab *t, ac4_aspx_chd *c, int qmode,
                   int smode)
{
    vrow_t v[AC4_ASPX_MAX_ENV];
    int e, s;
    ec_data(b, bk, t, 1, c->fr.num_env, c->fr.freq_res, qmode, smode,
            c->dir_sig, c->n_sig, v);
    for (e = 0; e < c->fr.num_env; e++)
        for (s = 0; s < c->n_sig[e]; s++)
            c->sig[e][s] = v[e][s];
}

static void ec_noise(ac4_bits *b, const ac4_aspx_books *bk,
                     const ac4_aspx_tab *t, ac4_aspx_chd *c, int smode)
{
    vrow_t v[2];
    int e, s;
    ec_data(b, bk, t, 0, c->fr.num_noise, NULL, 0, smode, c->dir_noise,
            c->n_noi, v);
    for (e = 0; e < c->fr.num_noise; e++)
        for (s = 0; s < c->n_noi[e]; s++)
            c->noi[e][s] = v[e][s];
}

void ac4_aspx_data(ac4_bits *b, const ac4_aspx_books *bk,
                   const ac4_aspx_cfg *cfg, int nats, int b_iframe,
                   int two_ch, ac4_aspx_state *st, int grp,
                   ac4_aspx_grp *out, ac4_aspx_tab *tab_out)
{
    ac4_aspx_grp tmp;
    ac4_aspx_tab tab;
    const ac4_aspx_tab *t = &tab;
    ac4_aspx_chd *c0, *c1;
    int q0, q1, balance, sm1;
    if (!out)
        out = &tmp;
    memset(out, 0, sizeof(*out));
    c0 = &out->c[0];
    c1 = &out->c[1];
    /* aspx_xover_subband_offset (I-frames; held per block up to the next
       one) selects this block's tables (Pseudocode 68) */
    if (b_iframe)
        st->xover[grp] = (int)ac4_u(b, 3);
    out->xover = st->xover[grp];
    if (ac4_aspx_tables(cfg, out->xover, &tab) != 0)
        ac4_fail(b, AC4E_ASPX_CONFIG);
    if (tab_out)
        *tab_out = tab;
    aspx_framing(b, cfg, nats, b_iframe, st, grp, 0, &c0->fr);
    q0 = cfg->quant_mode_env;
    if (c0->fr.int_class == FIXFIX && c0->fr.num_env == 1)
        q0 = 0;
    if (!two_ch) {
        delta_dir(b, &c0->fr, c0->dir_sig, c0->dir_noise);
        hfgen_iwc(b, t, 0, 0, nats, out->c);
        ec_sig(b, bk, t, c0, q0, 0);
        ec_noise(b, bk, t, c0, 0);
        c0->balance = 0;
        c0->ch = 0;
        c0->qmode = q0;
        return;
    }
    balance = (int)ac4_bit(b);
    c1->fr = c0->fr;
    q1 = q0;
    if (balance == 1) {
        /* channel 1's p_sine_at_end comes from its own previous interval */
        c1->fr.sine_at_end = sine_at_end(st, grp, 1, c0->fr.tsg_ptr,
                                         c0->fr.num_env);
        /* channel 1 uses channel 0's framing, so its previous_stop_pos
           (Pseudocode 76, per channel) is channel 0's stop border too
           (aspx_parse.BALANCE_SHARES_STOP) */
        st->stop[grp][1] = st->stop[grp][0];
        st->has[grp][1] = st->has[grp][0];
    } else {
        aspx_framing(b, cfg, nats, b_iframe, st, grp, 1, &c1->fr);
        q1 = cfg->quant_mode_env;
        if (c1->fr.int_class == FIXFIX && c1->fr.num_env == 1)
            q1 = 0;
    }
    delta_dir(b, &c0->fr, c0->dir_sig, c0->dir_noise);
    delta_dir(b, &c1->fr, c1->dir_sig, c1->dir_noise);
    hfgen_iwc(b, t, 1, balance, nats, out->c);
    sm1 = balance ? 1 : 0;
    ec_sig(b, bk, t, c0, q0, 0);
    ec_sig(b, bk, t, c1, q1, sm1);
    ec_noise(b, bk, t, c0, 0);
    ec_noise(b, bk, t, c1, sm1);
    c0->balance = c1->balance = balance;
    c0->ch = 0;
    c1->ch = 1;
    c0->qmode = q0;
    c1->qmode = q1;
}

/* ------------------------------------------------------------ A-CPL */
/* Table 62, bitstream order: data type (0 ALPHA, 1 BETA, 2 BETA3,
   3 GAMMA) and which quant mode */
static const int ACPL_TYPE[11] = { 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3 };
static const int ACPL_WHICH[11] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 };

/* Annex A.3: cb_off of the F0 / DF / DT books per data type and quant
   mode (0 fine, 1 coarse) (acpl_parse.CB_OFF) */
static const int ACPL_CB_OFF[4][2][3] = {
    { { 0, 32, 32 }, { 0, 16, 16 } },         /* ALPHA */
    { { 0, 8, 8 }, { 0, 4, 4 } },             /* BETA */
    { { 0, 16, 16 }, { 0, 8, 8 } },           /* BETA3 */
    { { 20, 40, 40 }, { 10, 20, 20 } },       /* GAMMA */
};

void ac4_acpl_data_2ch(ac4_bits *b, const ac4_aspx_books *bk,
                       const ac4_acpl_cfg *cfg, ac4_acpl_qstate *qs,
                       ac4_acpl_frame *out)
{
    int interp, n_sets, i, p, s, k;
    int n = cfg->num_param_bands, st = cfg->param_band;
    int tsl[2] = { 0, 0 };
    /* Table 63 */
    interp = (int)ac4_bit(b);
    n_sets = (int)ac4_bit(b) + 1;
    if (interp == 1)
        for (i = 0; i < n_sets; i++)
            tsl[i] = (int)ac4_u(b, 5);
    if (out) {
        memset(out, 0, sizeof(*out));
        out->kind = AC4_GRP_DATA;
        out->cfg = *cfg;
        out->interp = interp;
        out->n_sets = n_sets;
        out->ts[0] = tsl[0];
        out->ts[1] = tsl[1];
    }
    for (p = 0; p < 11; p++) {
        int q = ACPL_WHICH[p] ? cfg->quant_mode_1 : cfg->quant_mode_0;
        int base = 6 * ACPL_TYPE[p] + 3 * (q ? 1 : 0);
        const int *off = ACPL_CB_OFF[ACPL_TYPE[p]][q ? 1 : 0];
        for (s = 0; s < n_sets; s++) {
            int cur[AC4_ACPL_NB];
            for (k = 0; k < AC4_ACPL_NB; k++)
                cur[k] = 0;
            /* Table 65, Pseudocode 8; Pseudocode 121 */
            if (ac4_bit(b) == 0) {            /* DIFF_FREQ */
                int v = ac4_huff_decode(&bk->acpl[base], b);
                cur[st] = v - off[0];
                for (k = st + 1; k < n; k++)
                    cur[k] = cur[k - 1]
                             + (ac4_huff_decode(&bk->acpl[base + 1], b)
                                - off[1]);
            } else {                          /* DIFF_TIME */
                for (k = st; k < n; k++) {
                    int v = ac4_huff_decode(&bk->acpl[base + 2], b);
                    int prev = (qs && k < qs->n[p]) ? qs->q[p][k] : 0;
                    cur[k] = prev + (v - off[2]);
                }
            }
            if (out) {
                for (k = 0; k < n; k++)
                    out->q[p][s][k] = cur[k];
            }
            if (qs) {
                for (k = 0; k < n; k++)
                    qs->q[p][k] = cur[k];
                qs->n[p] = n;
            }
        }
    }
}

/* Table 61 (Tables 63-65), Pseudocode 121: alpha1, beta1 of one module */
void ac4_acpl_data_1ch(ac4_bits *b, const ac4_aspx_books *bk,
                       const ac4_acpl1_cfg *cfg, int (*qs)[AC4_ACPL_NB],
                       int *qn, ac4_acpl1_frame *out)
{
    int interp, n_sets, i, p, s, k;
    int n = cfg->num_param_bands, st = cfg->param_band;
    int qm = cfg->quant_mode ? 1 : 0;
    int tsl[2] = { 0, 0 };
    interp = (int)ac4_bit(b);
    n_sets = (int)ac4_bit(b) + 1;
    if (interp == 1)
        for (i = 0; i < n_sets; i++)
            tsl[i] = (int)ac4_u(b, 5);
    if (out) {
        memset(out, 0, sizeof(*out));
        out->kind = AC4_GRP_DATA;
        out->cfg = *cfg;
        out->interp = interp;
        out->n_sets = n_sets;
        out->ts[0] = tsl[0];
        out->ts[1] = tsl[1];
    }
    for (p = 0; p < 2; p++) {             /* ALPHA, BETA */
        int base = 6 * p + 3 * qm;
        const int *off = ACPL_CB_OFF[p][qm];
        for (s = 0; s < n_sets; s++) {
            int cur[AC4_ACPL_NB];
            for (k = 0; k < AC4_ACPL_NB; k++)
                cur[k] = 0;
            if (ac4_bit(b) == 0) {            /* DIFF_FREQ */
                int v = ac4_huff_decode(&bk->acpl[base], b);
                cur[st] = v - off[0];
                for (k = st + 1; k < n; k++)
                    cur[k] = cur[k - 1]
                             + (ac4_huff_decode(&bk->acpl[base + 1], b)
                                - off[1]);
            } else {                          /* DIFF_TIME */
                for (k = st; k < n; k++) {
                    int v = ac4_huff_decode(&bk->acpl[base + 2], b);
                    int prev = (qs && k < qn[p]) ? qs[p][k] : 0;
                    cur[k] = prev + (v - off[2]);
                }
            }
            if (out)
                for (k = 0; k < n; k++)
                    out->q[p][s][k] = cur[k];
            if (qs) {
                for (k = 0; k < n; k++)
                    qs[p][k] = cur[k];
                qn[p] = n;
            }
        }
    }
}
