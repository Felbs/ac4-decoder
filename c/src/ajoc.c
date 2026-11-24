/* SPDX-License-Identifier: Apache-2.0 */
/* A-JOC full decoding (TS 103 190-2 5.7; ac4/ajoc.py Engine, Ramp,
   frame_targets; ac4/decoder.py AjocFull) and the object rendering
   matrix (ac4/objrender.py matrix). The readings are ajoc.py's. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ac4dec.h"
#include "ac4_internal.h"

#define NU AC4_AJOC_NU
#define ND AC4_AJOC_ND
#define NSB 64
#define MAXTS AC4_MAX_NTS
#define NGEV 64

static const int CYCLE[ND] = { 0, 2, 1, 0, 2, 1, 0 };     /* 5.7.3.5 */
static const int NBANDS[8] = { 23, 15, 12, 9, 7, 5, 3, 1 };
/* Table 28: the first subband of each row, and the band per row */
static const int T28_SB[24] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                14, 16, 18, 20, 23, 26, 30, 35, 41, 48,
                                64 };
static const unsigned char T28[8][23] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
      19, 20, 21, 22 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 13,
      14, 14, 14 },
    { 0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10, 11,
      11, 11 },
    { 0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8 },
    { 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6 },
    { 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4 },
    { 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
    { 0 } };

static void sb_to_pb(int nb, int *map)
{
    int code = 0, r, sb;
    while (code < 7 && NBANDS[code] != nb)
        code++;
    for (r = 0; r < 23; r++)
        for (sb = T28_SB[r]; sb < T28_SB[r + 1]; sb++)
            map[sb] = T28[code][r];
}

typedef struct {
    double dry[NU][ND][NSB], wet[NU][ND][NSB], pre[ND][ND][NSB];
} targets_t;

struct ac4_ajoc_full {
    int n_in, n_umx, lfe, lay, n_out;
    int in_pos[ND];
    signed char types[NU + 1];
    long long slot;
    /* the three parameter ramps (ajoc.Ramp; evented and stepped
       together) */
    targets_t p, dl;
    int curr, target, warm;
    int nd;
    double en[ND];
    int n_pend;
    long long pend_slot[3];
    int pend_ramp[3];
    targets_t tg[3];
    /* the rendering gains (a Ramp of layout x objects) and their pending
       block updates */
    double gp[AC4DEC_MAX_CHANNELS][NU + 1], gd[AC4DEC_MAX_CHANNELS][NU + 1];
    int g_curr, g_target, g_warm;
    int n_gev;
    long long gev_slot[NGEV];
    int gev_ramp[NGEV];
    double gev_m[NGEV][AC4DEC_MAX_CHANNELS * (NU + 1)];
    ac4_decorr dec[ND];
    double syn[AC4DEC_MAX_CHANNELS][1280];
    /* scratch */
    double vr[14 + 1][NSB], vi[14 + 1][NSB], ur[7 + 1][NSB], ui[7 + 1][NSB];
    double zr[NU][NSB], zi[NU][NSB];
    double yr[AC4DEC_MAX_CHANNELS][MAXTS][NSB];
    double yi[AC4DEC_MAX_CHANNELS][MAXTS][NSB];
    double lfe_g[AC4DEC_MAX_CHANNELS][MAXTS];
    double m[AC4DEC_MAX_CHANNELS * (NU + 1)];
};

/* ------------------------------------------------ objrender.matrix */
void ac4_obj_matrix(const signed char *types, const ac4_obj *st, int n,
                    int lay, double *M)
{
    int no = ac4_layout_channels(lay), j, c;
    const int *out = ac4_layout_ids(lay);
    for (c = 0; c < no * n; c++)
        M[c] = 0.0;
    for (j = 0; j < n; j++) {
        int t = types[j];
        const ac4_obj *s = st ? &st[j] : NULL;
        double gdb = s ? s->gain : 0.0, g;
        if (s && !s->active)
            gdb = -INFINITY;
        g = gdb == -INFINITY ? 0.0 : pow(10.0, gdb / 20.0);
        if (t == RN_LFE) {
            for (c = 0; c < no; c++)
                if (out[c] == RN_LFE)
                    M[c * n + j] = g;
        } else if (t == AC4_OBJ_DYN || t == AC4_OBJ_ISF) {
            double gg[AC4DEC_MAX_CHANNELS];
            if (s)
                ac4_obj_gains(s->x, s->y, s->z, out, no, gg);
            else
                ac4_obj_gains(0.5, 0.0, 0.0, out, no, gg);
            for (c = 0; c < no; c++)
                M[c * n + j] = g * gg[c];
        } else {
            double col[AC4DEC_MAX_CHANNELS];
            int id = t;
            ac4_render_matrix(&id, 1, lay, NULL, col);
            for (c = 0; c < no; c++)
                M[c * n + j] = g * col[c];
        }
    }
}

/* ------------------------------------------------ the engine */
ac4_ajoc_full *ac4_ajoc_full_new(const ac4_toc_sub *info, int lay)
{
    ac4_ajoc_full *a = (ac4_ajoc_full *)calloc(1, sizeof(*a));
    int i;
    if (!a)
        return NULL;
    a->lfe = info->lfe;
    a->lay = lay;
    a->n_out = ac4_layout_channels(lay);
    a->n_umx = info->n_umx > NU ? NU : (int)info->n_umx;
    if (info->static_dmx) {
        a->n_in = 5;
        for (i = 0; i < 5; i++)
            a->in_pos[i] = i;
    } else {
        a->n_in = info->n_dmx > ND ? ND : info->n_dmx;
        ac4_ajoc_input_order(a->n_in, a->in_pos);
    }
    i = 0;
    if (a->lfe)
        a->types[i++] = RN_LFE;
    for (; i < a->n_umx + a->lfe; i++)
        a->types[i] = info->umx_type[i - a->lfe];
    ac4_ajoc_full_reset(a);
    return a;
}

void ac4_ajoc_full_free(ac4_ajoc_full *a)
{
    free(a);
}

int ac4_ajoc_full_n_in(const ac4_ajoc_full *a)
{
    return a->n_in;
}

const int *ac4_ajoc_full_in_pos(const ac4_ajoc_full *a)
{
    return a->in_pos;
}

void ac4_ajoc_full_reset(ac4_ajoc_full *a)
{
    int no = a->n_out, nobj = a->n_umx + a->lfe, c, j;
    memset(&a->p, 0, sizeof(a->p));
    memset(&a->dl, 0, sizeof(a->dl));
    a->curr = a->target = a->warm = 0;
    a->nd = 0;
    memset(a->en, 0, sizeof(a->en));
    a->n_pend = 0;
    a->slot = 0;
    memset(a->dec, 0, sizeof(a->dec));
    memset(a->syn, 0, sizeof(a->syn));
    /* the gains start warm at the unknown objects' matrix */
    ac4_obj_matrix(a->types, NULL, nobj, a->lay, a->m);
    for (c = 0; c < no; c++)
        for (j = 0; j < nobj; j++) {
            a->gp[c][j] = a->m[c * nobj + j];
            a->gd[c][j] = 0.0;
        }
    a->g_curr = a->g_target = 0;
    a->g_warm = 1;
    a->n_gev = 0;
}

static double dq(int q, int nq, double mx)
{
    int mid = (nq - 1) / 2;
    return (double)(q - mid) * (mx / mid);
}

/* ajoc.frame_targets for data point dp of fr into t */
static void targets(const ac4_ajoc_full *a, const ac4_ajoc_frame *fr,
                    int dp, targets_t *t)
{
    int n_in = a->n_in, nd = fr->nd, o, ch, de, sb, pbm[NSB];
    memset(t, 0, sizeof(*t));
    for (o = 0; o < a->n_umx; o++) {
        int qs, nqd, nqw;
        if (!fr->present[o])
            continue;
        sb_to_pb(fr->nb[o], pbm);
        qs = fr->qs[o];
        nqd = qs ? 51 : 101;
        nqw = qs ? 21 : 41;
        for (ch = 0; ch < n_in; ch++)
            for (sb = 0; sb < NSB; sb++)
                t->dry[o][ch][sb] = dq(fr->q[dp][o][ch][pbm[sb]], nqd,
                                       5.0048828125);
        for (de = 0; de < nd; de++)
            for (sb = 0; sb < NSB; sb++)
                t->wet[o][de][sb] = dq(fr->q[dp][o][ND + de][pbm[sb]], nqw,
                                       2.001953125);
    }
    for (o = 0; o < a->n_umx; o++)
        for (de = 0; de < nd; de++)
            for (ch = 0; ch < n_in; ch++)
                for (sb = 0; sb < NSB; sb++)
                    t->pre[de][ch][sb] = t->pre[de][ch][sb]
                        + fabs(t->wet[o][de][sb]) * t->dry[o][ch][sb];
}

/* Ramp.event on the three parameter arrays */
static void p_event(ac4_ajoc_full *a, const targets_t *t, int n)
{
    int n_in = a->n_in, o, ch, de, sb;
    if (!a->warm || n <= 0) {
        a->p = *t;
        memset(&a->dl, 0, sizeof(a->dl));
        a->curr = a->target = 0;
        a->warm = 1;
        return;
    }
    for (o = 0; o < a->n_umx; o++) {
        for (ch = 0; ch < n_in; ch++)
            for (sb = 0; sb < NSB; sb++)
                a->dl.dry[o][ch][sb] = (t->dry[o][ch][sb]
                                        - a->p.dry[o][ch][sb]) / n;
        for (de = 0; de < ND; de++)
            for (sb = 0; sb < NSB; sb++)
                a->dl.wet[o][de][sb] = (t->wet[o][de][sb]
                                        - a->p.wet[o][de][sb]) / n;
    }
    for (de = 0; de < ND; de++)
        for (ch = 0; ch < n_in; ch++)
            for (sb = 0; sb < NSB; sb++)
                a->dl.pre[de][ch][sb] = (t->pre[de][ch][sb]
                                         - a->p.pre[de][ch][sb]) / n;
    a->curr = 0;
    a->target = n;
}

static void p_step(ac4_ajoc_full *a)
{
    int n_in = a->n_in, o, ch, de, sb;
    if (a->curr >= a->target)
        return;
    for (o = 0; o < a->n_umx; o++) {
        for (ch = 0; ch < n_in; ch++)
            for (sb = 0; sb < NSB; sb++)
                a->p.dry[o][ch][sb] = a->p.dry[o][ch][sb]
                                      + a->dl.dry[o][ch][sb];
        for (de = 0; de < ND; de++)
            for (sb = 0; sb < NSB; sb++)
                a->p.wet[o][de][sb] = a->p.wet[o][de][sb]
                                      + a->dl.wet[o][de][sb];
    }
    for (de = 0; de < ND; de++)
        for (ch = 0; ch < n_in; ch++)
            for (sb = 0; sb < NSB; sb++)
                a->p.pre[de][ch][sb] = a->p.pre[de][ch][sb]
                                       + a->dl.pre[de][ch][sb];
    a->curr++;
}

/* the gains' Ramp */
static void g_event(ac4_ajoc_full *a, const double *M, int n)
{
    int no = a->n_out, nobj = a->n_umx + a->lfe, c, j;
    if (!a->g_warm || n <= 0) {
        for (c = 0; c < no; c++)
            for (j = 0; j < nobj; j++) {
                a->gp[c][j] = M[c * nobj + j];
                a->gd[c][j] = 0.0;
            }
        a->g_curr = a->g_target = 0;
        a->g_warm = 1;
        return;
    }
    for (c = 0; c < no; c++)
        for (j = 0; j < nobj; j++)
            a->gd[c][j] = (M[c * nobj + j] - a->gp[c][j]) / n;
    a->g_curr = 0;
    a->g_target = n;
}

static void g_step(ac4_ajoc_full *a)
{
    int no = a->n_out, nobj = a->n_umx + a->lfe, c, j;
    if (a->g_curr >= a->g_target)
        return;
    for (c = 0; c < no; c++)
        for (j = 0; j < nobj; j++)
            a->gp[c][j] = a->gp[c][j] + a->gd[c][j];
    a->g_curr++;
}

/* AjocFull._install at absolute slot `slot` */
static void install(ac4_ajoc_full *a, long long slot,
                    const ac4_ajoc_frame *fr)
{
    int dp, d, k, nobj = a->n_umx + a->lfe;
    a->nd = fr->nd;
    for (d = 0; d < ND; d++)
        a->en[d] = (d < fr->nd && fr->en[d]) ? 1.0 : 0.0;
    a->n_pend = 0;
    for (dp = 0; dp < fr->ndp && dp < 3; dp++) {
        targets(a, fr, dp, &a->tg[dp]);
        a->pend_slot[dp] = slot + fr->start[dp];
        a->pend_ramp[dp] = fr->ramp[dp];
        a->n_pend++;
    }
    for (k = 0; k < fr->n_uev && a->n_gev < NGEV; k++) {
        int smp = fr->uev[k][0] < 0 ? 0 : fr->uev[k][0];
        ac4_obj_matrix(a->types, fr->ublk[k], nobj, a->lay,
                       a->gev_m[a->n_gev]);
        a->gev_slot[a->n_gev] = slot + smp / 64;
        a->gev_ramp[a->n_gev] = (fr->uev[k][1] + 63) / 64;
        a->n_gev++;
    }
}

/* Engine.run of slots [t0, t1) of the frame: objects into a->z* per
   slot, then the rendering of that slot */
static void run(ac4_ajoc_full *a, const double (*const xr[ND])[64],
                const double (*const xi[ND])[64], int t0, int t1,
                long long base)
{
    int n_in = a->n_in, nd = a->nd, ts, k, o, ch, de, sb, c;
    int no = a->n_out, lo = a->lfe;
    for (ts = t0; ts < t1; ts++) {
        long long at = base + ts;
        for (k = 0; k < a->n_pend; k++)
            if (a->pend_slot[k] == at)
                p_event(a, &a->tg[k], a->pend_ramp[k]);
        /* the decorrelators' input (u) and output (y), one slot */
        for (de = 0; de < nd; de++) {
            for (sb = 0; sb < NSB; sb++) {
                double r_ = 0.0, i_ = 0.0;
                for (ch = 0; ch < n_in; ch++) {
                    if (!xr[ch])
                        continue;
                    r_ = r_ + a->p.pre[de][ch][sb] * xr[ch][ts][sb];
                    i_ = i_ + a->p.pre[de][ch][sb] * xi[ch][ts][sb];
                }
                a->vr[14][sb] = r_;
                a->vi[14][sb] = i_;
            }
            ac4_decorr_run(&a->dec[de], CYCLE[de], 1, a->vr, a->vi, a->ur,
                           a->ui);
            for (sb = 0; sb < NSB; sb++) {
                a->ur[7][sb] = a->ur[7][sb] * a->en[de];
                a->ui[7][sb] = a->ui[7][sb] * a->en[de];
            }
            /* the objects: the dry part with the first decorrelator,
               then each decorrelator's wet part in turn (Pseudocode 18's
               order of summation) */
            for (o = 0; o < a->n_umx; o++) {
                if (de == 0) {
                    for (sb = 0; sb < NSB; sb++) {
                        double r_ = 0.0, i_ = 0.0;
                        for (ch = 0; ch < n_in; ch++) {
                            if (!xr[ch])
                                continue;
                            r_ = r_ + xr[ch][ts][sb] * a->p.dry[o][ch][sb];
                            i_ = i_ + xi[ch][ts][sb] * a->p.dry[o][ch][sb];
                        }
                        a->zr[o][sb] = r_;
                        a->zi[o][sb] = i_;
                    }
                }
                for (sb = 0; sb < NSB; sb++) {
                    a->zr[o][sb] = a->zr[o][sb]
                                   + a->ur[7][sb] * a->p.wet[o][de][sb];
                    a->zi[o][sb] = a->zi[o][sb]
                                   + a->ui[7][sb] * a->p.wet[o][de][sb];
                }
            }
        }
        if (nd == 0)
            for (o = 0; o < a->n_umx; o++)
                for (sb = 0; sb < NSB; sb++) {
                    double r_ = 0.0, i_ = 0.0;
                    for (ch = 0; ch < n_in; ch++) {
                        if (!xr[ch])
                            continue;
                        r_ = r_ + xr[ch][ts][sb] * a->p.dry[o][ch][sb];
                        i_ = i_ + xi[ch][ts][sb] * a->p.dry[o][ch][sb];
                    }
                    a->zr[o][sb] = r_;
                    a->zi[o][sb] = i_;
                }
        p_step(a);
        /* the rendering of this slot */
        for (k = 0; k < a->n_gev; k++)
            if (a->gev_slot[k] == at)
                g_event(a, a->gev_m[k], a->gev_ramp[k]);
        for (c = 0; c < no; c++)
            for (sb = 0; sb < NSB; sb++)
                a->yr[c][ts][sb] = a->yi[c][ts][sb] = 0.0;
        for (o = 0; o < a->n_umx; o++)
            for (c = 0; c < no; c++) {
                double g = a->gp[c][lo + o];
                if (g == 0.0)
                    continue;
                for (sb = 0; sb < NSB; sb++) {
                    a->yr[c][ts][sb] = a->yr[c][ts][sb] + g * a->zr[o][sb];
                    a->yi[c][ts][sb] = a->yi[c][ts][sb] + g * a->zi[o][sb];
                }
            }
        for (c = 0; c < no; c++)
            a->lfe_g[c][ts] = lo ? a->gp[c][0] : 0.0;
        g_step(a);
    }
}

void ac4_ajoc_full_frame(ac4_ajoc_full *a, const ac4_qmf_tables *q,
                         const double (*const xr[AC4_AJOC_ND])[64],
                         const double (*const xi[AC4_AJOC_ND])[64],
                         int nts, int tso, const ac4_ajoc_frame *fr,
                         const double *lfe, double (*out)[2048])
{
    long long base = a->slot;
    int c, ts, i, k, n;
    run(a, xr, xi, 0, tso, base);
    if (fr)
        install(a, base + tso, fr);
    run(a, xr, xi, tso, nts, base);
    /* pending updates past this frame stay */
    for (k = 0, n = 0; k < a->n_pend; k++)
        if (a->pend_slot[k] >= base + nts) {
            if (n != k) {
                a->pend_slot[n] = a->pend_slot[k];
                a->pend_ramp[n] = a->pend_ramp[k];
                a->tg[n] = a->tg[k];
            }
            n++;
        }
    a->n_pend = n;
    for (k = 0, n = 0; k < a->n_gev; k++)
        if (a->gev_slot[k] >= base + nts) {
            if (n != k) {
                a->gev_slot[n] = a->gev_slot[k];
                a->gev_ramp[n] = a->gev_ramp[k];
                memcpy(a->gev_m[n], a->gev_m[k], sizeof(a->gev_m[n]));
            }
            n++;
        }
    a->n_gev = n;
    for (c = 0; c < a->n_out; c++) {
        for (ts = 0; ts < nts; ts++)
            ac4_qmf_synth(q, a->syn[c], a->yr[c][ts], a->yi[c][ts],
                          out[c] + 64 * ts);
        if (a->lfe && lfe)
            for (ts = 0; ts < nts; ts++)
                for (i = 0; i < 64; i++)
                    out[c][64 * ts + i] = out[c][64 * ts + i]
                        + a->lfe_g[c][ts] * lfe[64 * ts + i];
    }
    a->slot = base + nts;
}
