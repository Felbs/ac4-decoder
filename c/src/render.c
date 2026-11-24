/* SPDX-License-Identifier: Apache-2.0 */
/* Rendering to an output layout: a port of ac4/render.py (TS 103 190-2
   5.10.2, Tables 38 - 43 with custom_dmx_data or Table 130's defaults;
   TS 103 190-1 6.2.17.4 LoRo, 6.2.17.3, 6.2.17.6, Table 219). */
#include <math.h>
#include <string.h>

#include "ac4_internal.h"

static const int LAY_N[8] = { 0, 2, 6, 8, 10, 8, 10, 12 };
static const int LAY[8][12] = {
    { 0 },
    { RN_L, RN_R },
    { RN_L, RN_R, RN_C, RN_LFE, RN_LS, RN_RS },
    { RN_L, RN_R, RN_C, RN_LFE, RN_LS, RN_RS, RN_TSL, RN_TSR },
    { RN_L, RN_R, RN_C, RN_LFE, RN_LS, RN_RS, RN_TFL, RN_TFR, RN_TBL,
      RN_TBR },
    { RN_L, RN_R, RN_C, RN_LFE, RN_LS, RN_RS, RN_LB, RN_RB },
    { RN_L, RN_R, RN_C, RN_LFE, RN_LS, RN_RS, RN_LB, RN_RB, RN_TSL,
      RN_TSR },
    { RN_L, RN_R, RN_C, RN_LFE, RN_LS, RN_RS, RN_LB, RN_RB, RN_TFL, RN_TFR,
      RN_TBL, RN_TBR },
};
/* the layout family (render.FAMILY): 0 5.X.0, 1 5.X.2, 2 5.X.4, 3 7.X.0,
   4 7.X.2, 5 7.X.4 -- the first five are out_ch_config (Table 127) */
static const int FAM[8] = { 0, 0, 0, 1, 2, 3, 4, 5 };
enum { CI_74, CI_54, CI_70, CI_520, CI_50, CI_30, CI_20, CI_10 };

int ac4_layout_channels(int layout)
{
    return layout >= 1 && layout <= 7 ? LAY_N[layout] : 0;
}

const int *ac4_layout_ids(int layout)
{
    return layout >= 1 && layout <= 7 ? LAY[layout] : NULL;
}

static double dbg(double x)
{
    return x == -INFINITY ? 0.0 : pow(10.0, x / 20.0);
}

static int input_config(const int *ids, int n)
{
    int has[RN_N], i;
    memset(has, 0, sizeof(has));
    for (i = 0; i < n; i++)
        if (ids[i] >= 0 && ids[i] < RN_N)
            has[ids[i]] = 1;
    if (has[RN_TFL] || has[RN_TFR] || has[RN_TBL] || has[RN_TBR])
        return has[RN_LB] ? CI_74 : CI_54;
    if (has[RN_LW])
        return CI_520;
    if (has[RN_LB])
        return CI_70;
    if (has[RN_LS])
        return CI_50;
    if (has[RN_L] && has[RN_C])
        return CI_30;
    if (has[RN_L])
        return CI_20;
    return CI_10;
}

static void add(double m[RN_N][RN_N], int o, int i, double v)
{
    if (v != 0.0)
        m[o][i] = m[o][i] + v;
}

/* render._family_matrix */
static void family_matrix(int ci, int fam, const double *g,
                          double m[RN_N][RN_N])
{
    static const int B74[7] = { RN_L, RN_R, RN_C, RN_LS, RN_RS, RN_LB,
                                RN_RB };
    const double m3 = pow(10.0, -3.0 / 20.0);
    int nb, k, tops = ci == CI_74 || ci == CI_54;
    int backs = ci == CI_74 || ci == CI_70, out_backs = fam >= 3;
    switch (ci) {
    case CI_74: case CI_70: nb = 7; break;
    case CI_54: case CI_50: nb = 5; break;
    case CI_30: nb = 3; break;
    case CI_20: nb = 2; break;
    default: nb = 0; break;
    }
    memset(m, 0, sizeof(double) * RN_N * RN_N);
    add(m, RN_LFE, RN_LFE, 1.0);
    if (ci == CI_10)
        add(m, RN_C, RN_C, 1.0);
    for (k = 0; k < nb; k++) {
        int c = B74[k];
        if ((c == RN_LB || c == RN_RB) && !out_backs)
            continue;
        if ((c == RN_LS || c == RN_RS) && backs && !out_backs)
            continue;
        add(m, c, c, 1.0);
    }
    if (backs && !out_backs) {
        double gb = tops ? dbg(g[AC4_G_B]) : m3;
        add(m, RN_LS, RN_LS, gb);
        add(m, RN_LS, RN_LB, gb);
        add(m, RN_RS, RN_RS, gb);
        add(m, RN_RS, RN_RB, gb);
    }
    if (!tops)
        return;
    if (fam == 2 || fam == 5) {
        add(m, RN_TFL, RN_TFL, 1.0);
        add(m, RN_TFR, RN_TFR, 1.0);
        add(m, RN_TBL, RN_TBL, 1.0);
        add(m, RN_TBR, RN_TBR, 1.0);
    } else if (fam == 1 || fam == 4) {
        double gt = (ci == CI_54 && fam == 4) ? m3 : dbg(g[AC4_G_T1]);
        add(m, RN_TSL, RN_TFL, gt);
        add(m, RN_TSL, RN_TBL, gt);
        add(m, RN_TSR, RN_TFR, gt);
        add(m, RN_TSR, RN_TBR, gt);
    } else if (fam == 3 && ci == CI_54) {
        add(m, RN_LS, RN_TFL, m3);
        add(m, RN_LS, RN_TBL, m3);
        add(m, RN_RS, RN_TFR, m3);
        add(m, RN_RS, RN_TBR, m3);
    } else {
        int h;
        for (h = 0; h < 2; h++) {
            int sd = h ? RN_R : RN_L, ss = h ? RN_RS : RN_LS;
            int tf = h ? RN_TFR : RN_TFL, tb = h ? RN_TBR : RN_TBL;
            int bk = h ? RN_RB : RN_LB;
            add(m, sd, tf, dbg(g[AC4_G_T2A]));
            add(m, ss, tf, dbg(g[AC4_G_T2B]));
            add(m, sd, tb, dbg(g[AC4_G_T2D]));
            add(m, ss, tb, dbg(g[AC4_G_T2E]));
            if (fam == 3) {
                add(m, bk, tf, dbg(g[AC4_G_T2C]));
                add(m, bk, tb, dbg(g[AC4_G_T2F]));
            }
        }
    }
}

/* m = a x b (each [out][in]) */
static void compose(double a[RN_N][RN_N], double b[RN_N][RN_N],
                    double out[RN_N][RN_N])
{
    int o, k, i;
    memset(out, 0, sizeof(double) * RN_N * RN_N);
    for (o = 0; o < RN_N; o++)
        for (k = 0; k < RN_N; k++)
            if (a[o][k] != 0.0)
                for (i = 0; i < RN_N; i++)
                    if (b[k][i] != 0.0)
                        out[o][i] = out[o][i] + a[o][k] * b[k][i];
}

void ac4_render_matrix(const int *ids, int n_in, int layout,
                       const ac4_cdmx *c, double *R)
{
    static const double DEF[AC4_CDMX_NG] = {
        -INFINITY, 0.0, -3.0, -3.0, -INFINITY, -3.0, -INFINITY, -INFINITY,
        -3.0, -INFINITY };               /* Table 130 */
    double m[RN_N][RN_N], t[RN_N][RN_N], g[AC4_CDMX_NG];
    const double m3 = pow(10.0, -3.0 / 20.0);
    int ci = input_config(ids, n_in), fam = FAM[layout], k, o, i;
    const int *out = LAY[layout];
    memcpy(g, DEF, sizeof(g));
    if (ci == CI_520) {
        /* Table 219 (add_ch_base 0), then a 5.X input */
        double m0[RN_N][RN_N];
        memset(m0, 0, sizeof(m0));
        m0[RN_L][RN_L] = 1.0;
        m0[RN_L][RN_LW] = m3;
        m0[RN_R][RN_R] = 1.0;
        m0[RN_R][RN_RW] = m3;
        m0[RN_C][RN_C] = m0[RN_LS][RN_LS] = m0[RN_RS][RN_RS] = 1.0;
        m0[RN_LFE][RN_LFE] = 1.0;
        family_matrix(CI_50, fam, g, t);
        compose(t, m0, m);
    } else {
        if (c && c->present) {
            if (fam <= 4 && c->has_cfg[fam])
                for (k = 0; k < AC4_CDMX_NG; k++)
                    if (c->has_g[fam][k])
                        g[k] = c->g[fam][k];
            if (fam == 4 && c->bs_ch_config == 1 && c->has_cfg[1]
                && c->has_g[1][AC4_G_T1]
                && !(c->has_cfg[4] && c->has_g[4][AC4_G_T1]))
                g[AC4_G_T1] = c->g[1][AC4_G_T1];
        }
        family_matrix(ci, fam, g, m);
    }
    if (layout == 1) {                   /* 2.0: LoRo of the 5.X.0 stage */
        double m2[RN_N][RN_N];
        double cmg = dbg(c && c->has_stereo ? c->loro_c : -3.0);
        double smg = dbg(c && c->has_stereo ? c->loro_s : -3.0);
        double lmg = c && c->has_stereo && c->lfe_mixgain >= 0
                     ? dbg(5.5 - (double)c->lfe_mixgain) : 0.0;
        memset(m2, 0, sizeof(m2));
        if (ci == CI_10) {
            m2[RN_L][RN_C] = m3;
            m2[RN_R][RN_C] = m3;
        } else {
            m2[RN_L][RN_L] = m2[RN_R][RN_R] = 1.0;
            m2[RN_L][RN_C] = m2[RN_R][RN_C] = cmg;
            m2[RN_L][RN_LS] = m2[RN_R][RN_RS] = smg;
            m2[RN_L][RN_LFE] = m2[RN_R][RN_LFE] = lmg;
        }
        compose(m2, m, t);
        memcpy(m, t, sizeof(m));
    }
    for (o = 0; o < LAY_N[layout]; o++)
        for (i = 0; i < n_in; i++)
            R[o * n_in + i] = ids[i] >= 0 && ids[i] < RN_N
                              ? m[out[o]][ids[i]] : 0.0;
}

/* render.Renderer._update: hold what a frame sends */
void ac4_cdmx_hold(ac4_cdmx *held, const ac4_cdmx *c)
{
    int k;
    if (!c->present)
        return;
    if (!held->present) {
        memset(held, 0, sizeof(*held));
        held->lfe_mixgain = -1;
    }
    held->present = 1;
    held->bs_ch_config = c->bs_ch_config;
    for (k = 0; k < 8; k++)
        if (c->has_cfg[k]) {
            held->has_cfg[k] = 1;
            memcpy(held->g[k], c->g[k], sizeof(held->g[k]));
            memcpy(held->has_g[k], c->has_g[k], sizeof(held->has_g[k]));
        }
    if (c->has_stereo) {
        held->has_stereo = 1;
        held->loro_c = c->loro_c;
        held->loro_s = c->loro_s;
        held->lfe_mixgain = c->lfe_mixgain;
    }
}
