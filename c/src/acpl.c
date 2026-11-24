/* SPDX-License-Identifier: Apache-2.0 */
/* A-CPL, advanced coupling: the 5.1 upmix of 5_X_codec_mode ASPX_ACPL_3,
 * TS 103 190-1 5.7.7 (dequantisation 5.7.7.7, Tables 203-208; parameter
 * bands 5.7.7.2, Table 197; interpolation 5.7.7.3, Pseudocodes 109-110;
 * decorrelators 5.7.7.4.2, Pseudocode 111, Tables 198-201; transient
 * ducker 5.7.7.4.3, Pseudocodes 112-114; the upmix 5.7.7.6.2, Pseudocodes
 * 118-119). A port of ac4/acpl.py, with the same arithmetic in the same
 * order; see that module for the readings taken where the standard leaves
 * room (time alignment, the ducker's time step and partition, the cold
 * start, frames without data). */
#include <math.h>
#include <string.h>

#include "ac4_internal.h"

#define NSB 64
#define NTS AC4_MAX_NTS     /* array room; the frame's own count is nts */
#define HIST 14
#define MAXL 7

/* Table 197 */
static const int GROUPS[15][2] = {
    { 0, 0 }, { 1, 1 }, { 2, 2 }, { 3, 3 }, { 4, 4 }, { 5, 5 }, { 6, 6 },
    { 7, 7 }, { 8, 8 }, { 9, 10 }, { 11, 13 }, { 14, 17 }, { 18, 22 },
    { 23, 34 }, { 35, 63 } };
static const int PB15[15] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                              14 };
static const int PB12[15] = { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10,
                              11 };
static const int PB9[15] = { 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8 };
static const int PB7[15] = { 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6 };

static void sb_to_pb(int nb, int *map)
{
    const int *t = nb == 15 ? PB15 : nb == 12 ? PB12 : nb == 9 ? PB9 : PB7;
    int g, sb;
    for (g = 0; g < 15; g++)
        for (sb = GROUPS[g][0]; sb <= GROUPS[g][1]; sb++)
            map[sb] = t[g];
}

int ac4_acpl_sb_to_pb(int num_param_bands, int sb)
{
    int map[NSB];
    sb_to_pb(num_param_bands, map);
    return sb >= 0 && sb < NSB ? map[sb] : 0;
}

/* Tables 203-206 */
static const double ALPHA_FINE[33] = {
    -2.0, -1.809375, -1.6375, -1.484375, -1.35, -1.234375, -1.1375,
    -1.059375, -1.0, -0.940625, -0.8625, -0.765625, -0.65, -0.515625,
    -0.3625, -0.190625, 0.0, 0.190625, 0.3625, 0.515625, 0.65, 0.765625,
    0.8625, 0.940625, 1.0, 1.059375, 1.1375, 1.234375, 1.35, 1.484375,
    1.6375, 1.809375, 2.0 };
static const int IBETA_FINE[33] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7,
    8, 7, 6, 5, 4, 3, 2, 1, 0 };
static const double ALPHA_COARSE[17] = {
    -2.0, -1.6375, -1.35, -1.1375, -1.0, -0.8625, -0.65, -0.3625, 0.0,
    0.3625, 0.65, 0.8625, 1.0, 1.1375, 1.35, 1.6375, 2.0 };
static const int IBETA_COARSE[17] = {
    0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0 };
static const double BETA_FINE[9][9] = {
    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.2375, 0.2035449, 0.1729297, 0.1456543, 0.1217188, 0.101123,
      0.0838672, 0.0699512, 0.059375 },
    { 0.55, 0.4713672, 0.4004688, 0.3373047, 0.281875, 0.2341797,
      0.1942188, 0.1619922, 0.1375 },
    { 0.9375, 0.8034668, 0.6826172, 0.5749512, 0.4804688, 0.3991699,
      0.3310547, 0.276123, 0.234375 },
    { 1.4, 1.199844, 1.019375, 0.8585938, 0.7175, 0.5960938, 0.494375,
      0.4123438, 0.35 },
    { 1.9375, 1.660498, 1.410742, 1.1882319, 0.9929688, 0.8249512,
      0.6841797, 0.5706543, 0.484375 },
    { 2.55, 2.18543, 1.856719, 1.563867, 1.306875, 1.085742, 0.9004688,
      0.7510547, 0.6375 },
    { 3.2375, 2.7746389, 2.357305, 1.985498, 1.659219, 1.378467, 1.143242,
      0.9535449, 0.809375 },
    { 4.0, 3.4281249, 2.9124999, 2.453125, 2.05, 1.703125, 1.4125,
      1.178125, 1.0 } };
static const double BETA_COARSE[5][5] = {
    { 0.0, 0.0, 0.0, 0.0, 0.0 },
    { 0.55, 0.4004688, 0.281875, 0.1942188, 0.1375 },
    { 1.4, 1.019375, 0.7175, 0.494375, 0.35 },
    { 2.55, 1.856719, 1.306875, 0.9004688, 0.6375 },
    { 4.0, 2.9124999, 2.05, 1.4125, 1.0 } };

/* acpl_parse.Q_RANGE, [type][fine, coarse] */
static const int Q_LO[4][2] = { { 0, 0 }, { 0, 0 }, { 0, 0 }, { -20, -10 } };
static const int Q_HI[4][2] = { { 32, 16 }, { 8, 4 }, { 16, 8 },
                                { 20, 10 } };

/* Tables 198-201: region k0 (0-6) delay 7, L 7; k1 (7-22) 10, 4;
   k2 (23-63) 12, 2. COEF[decorrelator][region][i - 1] = a[i] */
static const int REG[3][4] = { { 0, 7, 7, 7 }, { 7, 23, 10, 4 },
                               { 23, 64, 12, 2 } };
static const double COEF[3][3][7] = {
    { { 0.5306, -0.4533, -0.6248, 0.0424, 0.4237, 0.4311, 0.1688 },
      { 0.5561, -0.3039, -0.5024, -0.1850 },
      { 0.5773, 0.3321 } },
    { { -0.4178, 0.1082, -0.2368, -0.1014, -0.1052, -0.3528, 0.4665 },
      { 0.0425, 0.3235, -0.1556, 0.4958 },
      { 0.2327, -0.3901 } },
    { { 0.4007, 0.4747, 0.2611, -0.1211, -0.4248, -0.2989, -0.1932 },
      { -0.4361, 0.0345, 0.5215, -0.4178 },
      { -0.6057, 0.3804 } } };

/* Pseudocode 112 */
#define DUCK_ALPHA 0.76592833836465
#define DUCK_ALPHA_SMOOTH 0.25
#define DUCK_GAMMA 1.5
#define DUCK_EPSILON 1.0e-9

/* acpl.QUANTITIES */
enum { G1, G2, G3, G4, G5, G6, A1, A2, B1, B2, B3, G1A1, G2A1, G3A2, G4A2,
       B3A1, B3A2, G135, G246 };

static int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

/* acpl.dequantise_2ch + acpl.quantities for one parameter set:
   -> qv[AC4_ACPL_NQ][band] */
static void band_values(const ac4_acpl_frame *f, int ps,
                        double qv[AC4_ACPL_NQ][AC4_ACPL_NB])
{
    int qm0 = f->cfg.quant_mode_0 ? 1 : 0, qm1 = f->cfg.quant_mode_1 ? 1 : 0;
    int nb = f->cfg.num_param_bands, b, i;
    double d3 = qm0 ? 0.25 : 0.125;
    double dg = qm1 ? 3276.0 / 16384.0 : 1638.0 / 16384.0;
    for (b = 0; b < nb; b++) {
        for (i = 0; i < 2; i++) {
            int qa = clampi(f->q[i][ps][b], Q_LO[0][qm0], Q_HI[0][qm0]);
            int qb = clampi(f->q[2 + i][ps][b], Q_LO[1][qm0], Q_HI[1][qm0]);
            double a, be;
            if (qm0) {
                a = ALPHA_COARSE[qa];
                be = BETA_COARSE[qb][IBETA_COARSE[qa]];
            } else {
                a = ALPHA_FINE[qa];
                be = BETA_FINE[qb][IBETA_FINE[qa]];
            }
            qv[A1 + i][b] = a;
            qv[B1 + i][b] = be;
        }
        qv[B3][b] = (double)clampi(f->q[4][ps][b], Q_LO[2][qm0],
                                   Q_HI[2][qm0]) * d3;
        for (i = 0; i < 6; i++)
            qv[G1 + i][b] = (double)clampi(f->q[5 + i][ps][b], Q_LO[3][qm1],
                                           Q_HI[3][qm1]) * dg;
        qv[G1A1][b] = qv[G1][b] * qv[A1][b];
        qv[G2A1][b] = qv[G2][b] * qv[A1][b];
        qv[G3A2][b] = qv[G3][b] * qv[A2][b];
        qv[G4A2][b] = qv[G4][b] * qv[A2][b];
        qv[B3A1][b] = qv[B3][b] * qv[A1][b];
        qv[B3A2][b] = qv[B3][b] * qv[A2][b];
        qv[G135][b] = (qv[G1][b] + qv[G3][b]) + qv[G5][b];
        qv[G246][b] = (qv[G2][b] + qv[G4][b]) + qv[G6][b];
    }
}

/* Pseudocode 109 for one quantity, every subband */
static void interpolate(const double *prev, double sets[2][NSB], int n_sets,
                        int interp, const int *tsl, int n, int nts,
                        double ip[][NSB])
{
    int t, sb;
    for (t = 0; t < n; t++)
        for (sb = 0; sb < NSB; sb++) {
            double v;
            if (interp == 0) {
                if (n_sets == 1) {
                    double delta = sets[0][sb] - prev[sb];
                    v = prev[sb] + (double)(t + 1) * delta / (double)nts;
                } else if (t < nts / 2) {
                    v = prev[sb] + (double)(t + 1) * (sets[0][sb] - prev[sb])
                        / (double)(nts / 2);
                } else {
                    v = sets[0][sb] + (double)(t - nts / 2 + 1)
                        * (sets[1][sb] - sets[0][sb])
                        / (double)(nts - nts / 2);
                }
            } else if (n_sets == 1) {
                v = t < tsl[0] ? prev[sb] : sets[0][sb];
            } else {
                v = t < tsl[0] ? prev[sb]
                    : (t < tsl[1] ? sets[0][sb] : sets[1][sb]);
            }
            ip[t][sb] = v;
        }
}

/* numpy's |z| ** 2 */
static double abs2(double re, double im)
{
    double h = hypot(re, im);
    return h * h;
}

void ac4_acpl_run(ac4_acpl_state *s, ac4_acpl_work *w,
                  const ac4_acpl_frame *f, int n, int nts,
                  const double (*xr[2])[64], const double (*xi[2])[64],
                  double (*zr[5])[64], double (*zi[5])[64])
{
    double (*ip)[NTS][NSB] = w->ip;
    double (*sets)[2][NSB] = w->sets;
    double (*vr)[HIST + NTS][NSB] = w->vr, (*vi)[HIST + NTS][NSB] = w->vi;
    double (*ur)[MAXL + NTS][NSB] = w->ur, (*ui)[MAXL + NTS][NSB] = w->ui;
    const double gain_in = 1.0 + 2.0 * sqrt(0.5), sqrt2 = sqrt(2.0);
    int n_sets, interp, tsl[2] = { 0, 0 }, k, t, sb, d, i, ps;
    int pb15[NSB];

    if (!f || f->kind == AC4_GRP_NONE
        || (f->kind == AC4_GRP_CONCEALED && !s->warm)) {
        /* no A-CPL data: the core passes through, the state is cleared */
        memset(s, 0, sizeof(*s));
        for (t = 0; t < n; t++)
            for (k = 0; k < 5; k++)
                for (sb = 0; sb < NSB; sb++) {
                    zr[k][t][sb] = k < 2 ? xr[k][t][sb] : 0.0;
                    zi[k][t][sb] = k < 2 ? xi[k][t][sb] : 0.0;
                }
        return;
    }
    if (n <= 0)
        return;
    if (f->kind == AC4_GRP_DATA) {
        int map[NSB];
        double qv[AC4_ACPL_NQ][AC4_ACPL_NB];
        sb_to_pb(f->cfg.num_param_bands, map);
        n_sets = f->n_sets;
        for (ps = 0; ps < n_sets; ps++) {
            band_values(f, ps, qv);
            for (k = 0; k < AC4_ACPL_NQ; k++)
                for (sb = 0; sb < NSB; sb++)
                    sets[k][ps][sb] = qv[k][map[sb]];
        }
        if (!s->warm)
            for (k = 0; k < AC4_ACPL_NQ; k++)
                memcpy(s->prev[k], sets[k][0], sizeof(s->prev[k]));
        interp = f->interp;
        tsl[0] = f->ts[0];
        tsl[1] = f->ts[1];
    } else {                                /* concealed: hold */
        n_sets = 1;
        for (k = 0; k < AC4_ACPL_NQ; k++)
            memcpy(sets[k][0], s->prev[k], sizeof(s->prev[k]));
        interp = 0;
    }
    s->warm = 1;
    for (k = 0; k < AC4_ACPL_NQ; k++)
        interpolate(s->prev[k], sets[k], n_sets, interp, tsl, n, nts, ip[k]);
    if (f->kind == AC4_GRP_DATA)            /* Pseudocode 110 */
        for (k = 0; k < AC4_ACPL_NQ; k++)
            memcpy(s->prev[k], sets[k][n_sets - 1], sizeof(s->prev[k]));

    /* Transform(): v1, v2, v3 after the decorrelators' input history */
    for (d = 0; d < 3; d++) {
        memcpy(vr[d], s->xr[d], sizeof(s->xr[d]));
        memcpy(vi[d], s->xi[d], sizeof(s->xi[d]));
        memcpy(ur[d], s->yr[d], sizeof(s->yr[d]));
        memcpy(ui[d], s->yi[d], sizeof(s->yi[d]));
    }
    for (t = 0; t < n; t++)
        for (sb = 0; sb < NSB; sb++) {
            static const int QA[3] = { G1, G3, G135 }, QB[3] = { G2, G4,
                                                                  G246 };
            double x0r = xr[0][t][sb] * gain_in, x0i = xi[0][t][sb] * gain_in;
            double x1r = xr[1][t][sb] * gain_in, x1i = xi[1][t][sb] * gain_in;
            for (d = 0; d < 3; d++) {
                double ga = ip[QA[d]][t][sb], gb = ip[QB[d]][t][sb];
                vr[d][HIST + t][sb] = x0r * ga + x1r * gb;
                vi[d][HIST + t][sb] = x0i * ga + x1i * gb;
            }
        }
    /* Pseudocode 111 */
    for (d = 0; d < 3; d++) {
        int r;
        for (r = 0; r < 3; r++) {
            int lo = REG[r][0], hi = REG[r][1], dl = REG[r][2], L = REG[r][3];
            double a[MAXL + 1], b[MAXL + 1];
            a[0] = 1.0;
            for (i = 1; i <= L; i++)
                a[i] = COEF[d][r][i - 1];
            for (i = 0; i <= L; i++)
                b[i] = a[L - i];
            for (t = 0; t < n; t++) {
                int p = HIST + t, q = MAXL + t;
                for (sb = lo; sb < hi; sb++) {
                    double yr_ = b[0] * vr[d][p - dl][sb];
                    double yi_ = b[0] * vi[d][p - dl][sb];
                    for (i = 1; i <= L; i++) {
                        yr_ = yr_ + (b[i] * vr[d][p - i - dl][sb]
                                     - a[i] * ur[d][q - i][sb]);
                        yi_ = yi_ + (b[i] * vi[d][p - i - dl][sb]
                                     - a[i] * ui[d][q - i][sb]);
                    }
                    ur[d][q][sb] = yr_;
                    ui[d][q][sb] = yi_;
                }
            }
        }
        memcpy(s->xr[d], vr[d][n], sizeof(s->xr[d]));
        memcpy(s->xi[d], vi[d][n], sizeof(s->xi[d]));
        memcpy(s->yr[d], ur[d][n], sizeof(s->yr[d]));
        memcpy(s->yi[d], ui[d][n], sizeof(s->yi[d]));
    }
    /* Pseudocodes 112-114, once per time slot, 15 bands */
    sb_to_pb(15, pb15);
    for (d = 0; d < 3; d++)
        for (t = 0; t < n; t++) {
            double g[AC4_ACPL_NB];
            int q = MAXL + t;
            for (k = 0; k < AC4_ACPL_NB; k++) {
                int lo = GROUPS[k][0], w = GROUPS[k][1] - lo + 1;
                double e2[NSB], e, pk;
                for (i = 0; i < w; i++)
                    e2[i] = abs2(ur[d][q][lo + i], ui[d][q][lo + i]);
                e = ac4_np_sum(e2, w);
                pk = DUCK_ALPHA * s->peak[d][k];
                s->peak[d][k] = pk < e ? e : pk;
                s->smooth[d][k] = (1.0 - DUCK_ALPHA_SMOOTH) * s->smooth[d][k];
                s->smooth[d][k] = s->smooth[d][k] + DUCK_ALPHA_SMOOTH * e;
                s->spd[d][k] = (1.0 - DUCK_ALPHA_SMOOTH) * s->spd[d][k];
                s->spd[d][k] = s->spd[d][k]
                               + DUCK_ALPHA_SMOOTH * (s->peak[d][k] - e);
                if (DUCK_GAMMA * s->spd[d][k] > s->smooth[d][k])
                    g[k] = s->smooth[d][k]
                           / (DUCK_GAMMA * (s->spd[d][k] + DUCK_EPSILON));
                else
                    g[k] = 1.0;
            }
            for (sb = 0; sb < NSB; sb++) {
                ur[d][q][sb] = ur[d][q][sb] * g[pb15[sb]];
                ui[d][q][sb] = ui[d][q][sb] * g[pb15[sb]];
            }
        }
    /* ACplModule2, ACplModule3 */
    for (t = 0; t < n; t++)
        for (sb = 0; sb < NSB; sb++) {
            int q = MAXL + t;
            double x0r = xr[0][t][sb] * gain_in, x0i = xi[0][t][sb] * gain_in;
            double x1r = xr[1][t][sb] * gain_in, x1i = xi[1][t][sb] * gain_in;
            double y0r = ur[0][q][sb], y0i = ui[0][q][sb];
            double y1r = ur[1][q][sb], y1i = ui[1][q][sb];
            double y2r = 0.25 * ur[2][q][sb], y2i = 0.25 * ui[2][q][sb];
            double g1 = ip[G1][t][sb], g2 = ip[G2][t][sb];
            double g3 = ip[G3][t][sb], g4 = ip[G4][t][sb];
            double g5 = ip[G5][t][sb], g6 = ip[G6][t][sb];
            double g1a = ip[G1A1][t][sb], g2a = ip[G2A1][t][sb];
            double g3a = ip[G3A2][t][sb], g4a = ip[G4A2][t][sb];
            double b1 = ip[B1][t][sb], b2 = ip[B2][t][sb], b3 = ip[B3][t][sb];
            double b3a1 = ip[B3A1][t][sb], b3a2 = ip[B3A2][t][sb];
            double nb3 = -b3;
            double c, z[5][2];
            /* z0 (L), z1 (Ls) */
            z[0][0] = 0.5 * ((x0r * (g1 + g1a) + x1r * (g2 + g2a)) + y0r * b1);
            z[0][1] = 0.5 * ((x0i * (g1 + g1a) + x1i * (g2 + g2a)) + y0i * b1);
            z[3][0] = 0.5 * ((x0r * (g1 - g1a) + x1r * (g2 - g2a)) - y0r * b1);
            z[3][1] = 0.5 * ((x0i * (g1 - g1a) + x1i * (g2 - g2a)) - y0i * b1);
            /* z2 (R), z3 (Rs) */
            z[1][0] = 0.5 * ((x0r * (g3 + g3a) + x1r * (g4 + g4a)) + y1r * b2);
            z[1][1] = 0.5 * ((x0i * (g3 + g3a) + x1i * (g4 + g4a)) + y1i * b2);
            z[4][0] = 0.5 * ((x0r * (g3 - g3a) + x1r * (g4 - g4a)) - y1r * b2);
            z[4][1] = 0.5 * ((x0i * (g3 - g3a) + x1i * (g4 - g4a)) - y1i * b2);
            /* z4 (C) */
            z[2][0] = 0.5 * (x0r * (g5 + g5) + x1r * (g6 + g6));
            z[2][1] = 0.5 * (x0i * (g5 + g5) + x1i * (g6 + g6));
            c = b3 + b3a1;
            z[0][0] = z[0][0] + y2r * c;
            z[0][1] = z[0][1] + y2i * c;
            c = b3 - b3a1;
            z[3][0] = z[3][0] + y2r * c;
            z[3][1] = z[3][1] + y2i * c;
            c = b3 + b3a2;
            z[1][0] = z[1][0] + y2r * c;
            z[1][1] = z[1][1] + y2i * c;
            c = b3 - b3a2;
            z[4][0] = z[4][0] + y2r * c;
            z[4][1] = z[4][1] + y2i * c;
            c = nb3 + nb3;
            z[2][0] = z[2][0] + y2r * c;
            z[2][1] = z[2][1] + y2i * c;
            /* out: L, R, C * sqrt2, Ls * sqrt2, Rs * sqrt2 */
            for (k = 0; k < 5; k++) {
                double m = k < 2 ? 1.0 : sqrt2;
                zr[k][t][sb] = k < 2 ? z[k][0] : z[k][0] * m;
                zi[k][t][sb] = k < 2 ? z[k][1] : z[k][1] * m;
            }
        }
}

/* ------------------------------------------------- ASPX_ACPL_1 / _2 */
/* Pseudocode 111 then 112-114 for decorrelator d on n slots of w->vr/vi[d]
   (after the HIST slots of history taken from s), into w->ur/ui[d]
   (after MAXL slots), the state carried on: the loops of ac4_acpl_run */
static void dd_core(double (*sxr)[NSB], double (*sxi)[NSB],
                    double (*syr)[NSB], double (*syi)[NSB], double *peak,
                    double *smooth, double *spd, int cd, int n,
                    double (*vr)[NSB], double (*vi)[NSB], double (*ur)[NSB],
                    double (*ui)[NSB])
{
    int r, i, t, sb, k, pb15[NSB];
    memcpy(vr, sxr, sizeof(double) * HIST * NSB);
    memcpy(vi, sxi, sizeof(double) * HIST * NSB);
    memcpy(ur, syr, sizeof(double) * MAXL * NSB);
    memcpy(ui, syi, sizeof(double) * MAXL * NSB);
    for (r = 0; r < 3; r++) {
        int lo = REG[r][0], hi = REG[r][1], dl = REG[r][2], L = REG[r][3];
        double a[MAXL + 1], b[MAXL + 1];
        a[0] = 1.0;
        for (i = 1; i <= L; i++)
            a[i] = COEF[cd][r][i - 1];
        for (i = 0; i <= L; i++)
            b[i] = a[L - i];
        for (t = 0; t < n; t++) {
            int p = HIST + t, q = MAXL + t;
            for (sb = lo; sb < hi; sb++) {
                double yr_ = b[0] * vr[p - dl][sb];
                double yi_ = b[0] * vi[p - dl][sb];
                for (i = 1; i <= L; i++) {
                    yr_ = yr_ + (b[i] * vr[p - i - dl][sb]
                                 - a[i] * ur[q - i][sb]);
                    yi_ = yi_ + (b[i] * vi[p - i - dl][sb]
                                 - a[i] * ui[q - i][sb]);
                }
                ur[q][sb] = yr_;
                ui[q][sb] = yi_;
            }
        }
    }
    memcpy(sxr, vr[n], sizeof(double) * HIST * NSB);
    memcpy(sxi, vi[n], sizeof(double) * HIST * NSB);
    memcpy(syr, ur[n], sizeof(double) * MAXL * NSB);
    memcpy(syi, ui[n], sizeof(double) * MAXL * NSB);
    sb_to_pb(15, pb15);
    for (t = 0; t < n; t++) {
        double g[AC4_ACPL_NB];
        int q = MAXL + t;
        for (k = 0; k < AC4_ACPL_NB; k++) {
            int lo = GROUPS[k][0], wd = GROUPS[k][1] - lo + 1;
            double e2[NSB], e, pk;
            for (i = 0; i < wd; i++)
                e2[i] = abs2(ur[q][lo + i], ui[q][lo + i]);
            e = ac4_np_sum(e2, wd);
            pk = DUCK_ALPHA * peak[k];
            peak[k] = pk < e ? e : pk;
            smooth[k] = (1.0 - DUCK_ALPHA_SMOOTH) * smooth[k];
            smooth[k] = smooth[k] + DUCK_ALPHA_SMOOTH * e;
            spd[k] = (1.0 - DUCK_ALPHA_SMOOTH) * spd[k];
            spd[k] = spd[k] + DUCK_ALPHA_SMOOTH * (peak[k] - e);
            if (DUCK_GAMMA * spd[k] > smooth[k])
                g[k] = smooth[k] / (DUCK_GAMMA * (spd[k] + DUCK_EPSILON));
            else
                g[k] = 1.0;
        }
        for (sb = 0; sb < NSB; sb++) {
            ur[q][sb] = ur[q][sb] * g[pb15[sb]];
            ui[q][sb] = ui[q][sb] * g[pb15[sb]];
        }
    }
}

static void decorr_duck(ac4_acpl_state *s, ac4_acpl_work *w, int d, int cd,
                        int n)
{
    dd_core(s->xr[d], s->xi[d], s->yr[d], s->yi[d], s->peak[d],
            s->smooth[d], s->spd[d], cd, n, w->vr[d], w->vi[d], w->ur[d],
            w->ui[d]);
}

void ac4_decorr_run(ac4_decorr *s, int cd, int n, double (*vr)[64],
                    double (*vi)[64], double (*ur)[64], double (*ui)[64])
{
    dd_core(s->xr, s->xi, s->yr, s->yi, s->peak, s->smooth, s->spd, cd, n,
            vr, vi, ur, ui);
}

/* acpl.dequantise_1ch for one parameter set: -> a[band], b[band] */
static void band_values_1ch(const ac4_acpl1_frame *f, int ps, double *a,
                            double *be)
{
    int qm = f->cfg.quant_mode ? 1 : 0, b;
    for (b = 0; b < f->cfg.num_param_bands; b++) {
        int qa = clampi(f->q[0][ps][b], Q_LO[0][qm], Q_HI[0][qm]);
        int qb = clampi(f->q[1][ps][b], Q_LO[1][qm], Q_HI[1][qm]);
        if (qm) {
            a[b] = ALPHA_COARSE[qa];
            be[b] = BETA_COARSE[qb][IBETA_COARSE[qa]];
        } else {
            a[b] = ALPHA_FINE[qa];
            be[b] = BETA_FINE[qb][IBETA_FINE[qa]];
        }
    }
}

void ac4_acpl1_run(ac4_acpl_state *s, ac4_acpl_work *w,
                   const ac4_acpl1_frame *frames, int kind, int nm, int n,
                   int nts, int split, int dec,
                   const double (*x0r[2])[64], const double (*x0i[2])[64],
                   const double (*x1r[2])[64], const double (*x1i[2])[64],
                   double (*z0r[2])[64], double (*z0i[2])[64],
                   double (*z1r[2])[64], double (*z1i[2])[64])
{
    double (*ip)[AC4_MAX_NTS][NSB] = w->ip;
    double (*sets)[2][NSB] = w->sets;
    int m, t, sb, ps, k, tsl[2];
    if (kind == AC4_GRP_NONE || (kind == AC4_GRP_CONCEALED && !s->warm)) {
        memset(s, 0, sizeof(*s));
        for (m = 0; m < nm; m++)
            for (t = 0; t < n; t++)
                for (sb = 0; sb < NSB; sb++) {
                    double ar = x0r[m][t][sb], ai = x0i[m][t][sb];
                    double br = x1r[m] ? x1r[m][t][sb] : 0.0;
                    double bi = x1i[m] ? x1i[m][t][sb] : 0.0;
                    if (split) {
                        z0r[m][t][sb] = ar + br;
                        z0i[m][t][sb] = ai + bi;
                        z1r[m][t][sb] = ar - br;
                        z1i[m][t][sb] = ai - bi;
                    } else {
                        z0r[m][t][sb] = ar;
                        z0i[m][t][sb] = ai;
                        z1r[m][t][sb] = 0.0;
                        z1i[m][t][sb] = 0.0;
                    }
                }
        return;
    }
    if (n <= 0)
        return;
    for (m = 0; m < nm; m++) {
        int interp = 0, n_sets = 1;
        tsl[0] = tsl[1] = 0;
        if (kind == AC4_GRP_DATA) {
            const ac4_acpl1_frame *f = &frames[m];
            int map[NSB];
            double av[AC4_ACPL_NB], bv[AC4_ACPL_NB];
            sb_to_pb(f->cfg.num_param_bands, map);
            n_sets = f->n_sets;
            for (ps = 0; ps < n_sets; ps++) {
                band_values_1ch(f, ps, av, bv);
                for (sb = 0; sb < NSB; sb++) {
                    sets[2 * m][ps][sb] = av[map[sb]];
                    sets[2 * m + 1][ps][sb] = bv[map[sb]];
                }
            }
            if (!s->warm)
                for (k = 0; k < 2; k++)
                    memcpy(s->prev[2 * m + k], sets[2 * m + k][0],
                           sizeof(s->prev[0]));
            interp = f->interp;
            tsl[0] = f->ts[0];
            tsl[1] = f->ts[1];
        } else {                            /* concealed: hold */
            for (k = 0; k < 2; k++)
                memcpy(sets[2 * m + k][0], s->prev[2 * m + k],
                       sizeof(s->prev[0]));
        }
        for (k = 0; k < 2; k++)
            interpolate(s->prev[2 * m + k], sets[2 * m + k], n_sets, interp,
                        tsl, n, nts, ip[2 * m + k]);
        if (kind == AC4_GRP_DATA) {         /* Pseudocode 110 */
            for (k = 0; k < 2; k++)
                memcpy(s->prev[2 * m + k], sets[2 * m + k][n_sets - 1],
                       sizeof(s->prev[0]));
            s->qb[m] = frames[m].cfg.qmf_band;
        }
    }
    s->warm = 1;
    for (m = 0; m < nm; m++) {
        for (t = 0; t < n; t++)
            for (sb = 0; sb < NSB; sb++) {
                w->vr[m][HIST + t][sb] = 2.0 * x0r[m][t][sb];
                w->vi[m][HIST + t][sb] = 2.0 * x0i[m][t][sb];
            }
        /* module m on decorrelator D_m, or every module on D_dec (the
           immersive element: two instances of one decorrelator) */
        decorr_duck(s, w, m, dec >= 0 ? dec : m, n);
    }
    for (m = 0; m < nm; m++) {
        int qb = s->qb[m];
        for (t = 0; t < n; t++)
            for (sb = 0; sb < NSB; sb++) {
                double xr2 = 2.0 * x0r[m][t][sb], xi2 = 2.0 * x0i[m][t][sb];
                if (sb < qb) {
                    double br = x1r[m] ? 2.0 * x1r[m][t][sb] : 0.0;
                    double bi = x1i[m] ? 2.0 * x1i[m][t][sb] : 0.0;
                    z0r[m][t][sb] = 0.5 * (xr2 + br);
                    z0i[m][t][sb] = 0.5 * (xi2 + bi);
                    z1r[m][t][sb] = 0.5 * (xr2 - br);
                    z1i[m][t][sb] = 0.5 * (xi2 - bi);
                } else {
                    double a = ip[2 * m][t][sb], b = ip[2 * m + 1][t][sb];
                    double yr = w->ur[m][MAXL + t][sb];
                    double yi = w->ui[m][MAXL + t][sb];
                    z0r[m][t][sb] = 0.5 * (xr2 * (1.0 + a) + yr * b);
                    z0i[m][t][sb] = 0.5 * (xi2 * (1.0 + a) + yi * b);
                    z1r[m][t][sb] = 0.5 * (xr2 * (1.0 - a) - yr * b);
                    z1i[m][t][sb] = 0.5 * (xi2 * (1.0 - a) - yi * b);
                }
            }
    }
}
