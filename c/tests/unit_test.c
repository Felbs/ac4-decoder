/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Unit tests for libac4dec's building blocks (no captures needed):
 *   bit reader and variable_bits; every Huffman codebook builds as a
 *   complete prefix code and round-trips every symbol; the scale factor
 *   band tables; the filterbank is an isometry (lapped orthogonal
 *   transform: ||S x|| == ||x||) for steady and switching block lengths,
 *   with a negative control; the resampler gives the same samples however
 *   the input is split, the right count, and unit DC gain; every A-SPX
 *   and A-CPL codebook builds and round-trips; the QMF bank (through the
 *   A-SPX pair renderer with no A-SPX data) reconstructs its input 577
 *   samples late, with a negative control; the A-SPX tables obey the
 *   structural rules of clause 5.7.6.3.1.
 *
 *   tests/unit_test --dump-aspx   prints the A-SPX tables for every
 *   aspx_config (c/tests/check_aspx_tables.py compares them with the
 *   Python's).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac4dec.h"
#include "ac4_internal.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static unsigned long long rs = 0x2545F4914F6CDD1Dull;
static double urand(void)
{
    rs ^= rs << 13;
    rs ^= rs >> 7;
    rs ^= rs << 17;
    return (double)(rs >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

static void test_bits(void)
{
    static const uint8_t d[] = { 0xA5, 0x3C, 0xFF, 0x00, 0x81 };
    jmp_buf jb;
    ac4_bits b;
    volatile int stage = 0;
    b.d = d;
    b.nbits = sizeof(d) * 8;
    b.p = 0;
    b.jb = &jb;
    b.eof_code = 99;
    if (setjmp(jb)) {
        CHECK(stage == 3, "unexpected bit reader error at stage %d", stage);
        return;
    }
    CHECK(ac4_u(&b, 4) == 0xA, "u(4)");
    CHECK(ac4_u(&b, 8) == 0x53, "u(8) across bytes");
    CHECK(ac4_u(&b, 3) == 0x6, "u(3)");
    CHECK(ac4_bit(&b) == 0, "bit");
    CHECK(ac4_u(&b, 0) == 0, "u(0)");
    stage = 1;
    b.p = 0;
    CHECK(ac4_u(&b, 32) == 0xA53CFF00u, "u(32)");
    /* variable_bits(2) on 01 1 10 0: (1 + 1) << 2 ... */
    {
        static const uint8_t v[] = { 0x78 }; /* 0111 1000 */
        ac4_bits c = b;
        c.d = v;
        c.nbits = 8;
        c.p = 0;
        /* 01 -> 1, cont 1 -> (1 << 2) + 4 = 8; 11 -> 8 + 3 = 11, cont 0 */
        CHECK(ac4_vb(&c, 2) == 11, "variable_bits");
    }
    stage = 3;
    b.p = 38;
    ac4_u(&b, 3);                       /* past the end: must raise */
    CHECK(0, "reading past the end did not raise");
}

/* encode symbol s, decode it back: -> the symbol, -1 on error */
static int round_trip(const ac4_huff *h, const ac4t_book *bk, int s)
{
    uint8_t buf[8] = { 0 };
    int L = bk->len[s], k, got;
    jmp_buf jb;
    ac4_bits b;
    for (k = 0; k < L; k++)
        if ((bk->cw[s] >> (L - 1 - k)) & 1)
            buf[k >> 3] |= (uint8_t)(0x80 >> (k & 7));
    b.d = buf;
    b.nbits = 64;
    b.p = 0;
    b.jb = &jb;
    b.eof_code = 1;
    if (setjmp(jb))
        return -1;
    got = ac4_huff_decode(h, &b);
    return (int)b.p == L ? got : -1;
}

static void test_huffman(void)
{
    int i, s;
    for (i = 0; i < AC4T_N_BOOKS; i++) {
        ac4_huff h;
        const ac4t_book *bk = &AC4T_BOOKS[i];
        double kraft = 0.0;
        for (s = 0; s < bk->n; s++)
            if (bk->len[s])
                kraft += ldexp(1.0, -bk->len[s]);
        CHECK(fabs(kraft - 1.0) < 1e-12, "book %d Kraft sum %.15f", i, kraft);
        CHECK(ac4_huff_build(&h, bk) == 0, "book %d does not build", i);
        for (s = 0; s < bk->n; s++)
            if (bk->len[s])
                CHECK(round_trip(&h, bk, s) == s, "book %d symbol %d round trip",
                      i, s);
        ac4_huff_free(&h);
    }
}

static void test_aspx_books(void)
{
    int i, s, set;
    for (set = 0; set < 2; set++) {
        int n = set ? AC4T_N_ACPL_BOOKS : AC4T_N_ASPX_BOOKS;
        for (i = 0; i < n; i++) {
            ac4_huff h;
            const ac4t_book *bk = set ? &AC4T_ACPL_BOOKS[i]
                                      : &AC4T_ASPX_BOOKS[i];
            double kraft = 0.0;
            for (s = 0; s < bk->n; s++)
                if (bk->len[s])
                    kraft += ldexp(1.0, -bk->len[s]);
            CHECK(fabs(kraft - 1.0) < 1e-12, "%s book %d Kraft sum %.15f",
                  set ? "A-CPL" : "A-SPX", i, kraft);
            CHECK(ac4_huff_build(&h, bk) == 0, "%s book %d does not build",
                  set ? "A-CPL" : "A-SPX", i);
            for (s = 0; s < bk->n; s++)
                if (bk->len[s])
                    CHECK(round_trip(&h, bk, s) == s,
                          "%s book %d symbol %d round trip",
                          set ? "A-CPL" : "A-SPX", i, s);
            ac4_huff_free(&h);
        }
    }
}

/* QMF analysis + synthesis with no A-SPX data: the input comes back 577
   samples late. -> reconstruction SNR (dB) over the interior */
static double qmf_snr(int corrupt)
{
    ac4_qmf_tables *q = ac4_qmf_tables_new();
    ac4_aspx_pair *p = (ac4_aspx_pair *)calloc(1, sizeof(*p));
    ac4_aspx_frame f;
    int nf = 12, j, i;
    double *x = (double *)malloc(sizeof(double) * 1536 * (size_t)nf);
    double *y = (double *)malloc(sizeof(double) * 1536 * (size_t)nf);
    double es = 0.0, en = 0.0, junk[1536];
    memset(&f, 0, sizeof(f));
    for (i = 0; i < 1536 * nf; i++) {
        x[i] = urand();
        y[i] = 0.0;
    }
    ac4_aspx_pair_reset(p);
    for (j = 0; j < nf; j++) {
        const double *in[2];
        double *out[2];
        in[0] = in[1] = x + 1536 * j;
        out[0] = y + 1536 * (j > 0 ? j - 1 : 0);
        out[1] = junk;
        ac4_aspx_pair_push(p, q, in, &f, out, NULL);
        if (corrupt && j == 5)
            p->qre[0][(5 * 24 + 3) % AC4_QRING][7] += 50.0;
    }
    for (i = 2 * 1536; i < (nf - 2) * 1536; i++) {
        double d = y[i] - x[i - 577];
        es += x[i - 577] * x[i - 577];
        en += d * d;
    }
    free(x);
    free(y);
    free(p);
    ac4_qmf_tables_free(q);
    return 10.0 * log10(es / (en > 1e-300 ? en : 1e-300));
}

static void test_qmf(void)
{
    double good = qmf_snr(0), bad = qmf_snr(1);
    CHECK(good > 70.0, "QMF round trip SNR %.1f dB", good);
    CHECK(bad < 40.0, "QMF negative control SNR %.1f dB", bad);
}

static int strictly_up(const int *v, int n)
{
    int i;
    for (i = 0; i + 1 < n; i++)
        if (v[i] >= v[i + 1])
            return 0;
    return 1;
}

static void test_aspx_tables(void)
{
    ac4_aspx_cfg c;
    ac4_aspx_tab t;
    int st, sp, ms, ns, n_ok = 0;
    memset(&c, 0, sizeof(c));
    for (st = 0; st < 8; st++)
        for (sp = 0; sp < 4; sp++)
            for (ms = 0; ms < 2; ms++)
                for (ns = 0; ns < 4; ns++) {
                    int i, k;
                    c.start_freq = st;
                    c.stop_freq = sp;
                    c.master_freq_scale = ms;
                    c.noise_sbg = ns;
                    if (ac4_aspx_tables(&c, &t) != 0)
                        continue;
                    n_ok++;
                    CHECK(t.master[0] % 2 == 0 && t.master[t.n_master] % 2 == 0,
                          "master table ends on odd subbands");
                    CHECK(strictly_up(t.master, t.n_master + 1)
                          && strictly_up(t.hi, t.n_hi + 1)
                          && strictly_up(t.lo, t.n_lo + 1)
                          && strictly_up(t.noise, t.n_noise + 1),
                          "a table is not strictly increasing");
                    for (i = 0; i <= t.n_lo; i++) {
                        int found = 0;
                        for (k = 0; k <= t.n_hi; k++)
                            found |= t.hi[k] == t.lo[i];
                        CHECK(found, "lowres border not in highres");
                    }
                    CHECK(t.patches[t.n_patches] <= t.sbx + t.num_sb,
                          "patches beyond the A-SPX range");
                }
    CHECK(n_ok > 100, "only %d A-SPX configurations give tables", n_ok);
}

static void dump_list(const char *name, const int *v, int n)
{
    int i;
    printf(" %s", name);
    for (i = 0; i < n; i++)
        printf("%c%d", i ? ',' : '=', v[i]);
    if (!n)
        printf("=");
}

static void dump_aspx(void)
{
    ac4_aspx_cfg c;
    ac4_aspx_tab t;
    int st, sp, ms, ns;
    memset(&c, 0, sizeof(c));
    for (st = 0; st < 8; st++)
        for (sp = 0; sp < 4; sp++)
            for (ms = 0; ms < 2; ms++)
                for (ns = 0; ns < 4; ns++) {
                    c.start_freq = st;
                    c.stop_freq = sp;
                    c.master_freq_scale = ms;
                    c.noise_sbg = ns;
                    printf("%d %d %d %d", st, sp, ms, ns);
                    if (ac4_aspx_tables(&c, &t) != 0) {
                        printf(" fail\n");
                        continue;
                    }
                    dump_list("master", t.master, t.n_master + 1);
                    dump_list("hi", t.hi, t.n_hi + 1);
                    dump_list("lo", t.lo, t.n_lo + 1);
                    dump_list("noise", t.noise, t.n_noise + 1);
                    dump_list("pnum", t.patch_num_sb, t.n_patches);
                    dump_list("pstart", t.patch_start_sb, t.n_patches);
                    dump_list("patches", t.patches, t.n_patches + 1);
                    dump_list("lim", t.lim, t.n_lim);
                    dump_list("h2l", t.high2low, t.n_hi);
                    dump_list("l2h", t.low2high, t.n_lo + 1);
                    printf("\n");
                }
}

static void test_sfb(void)
{
    static const int nsfb[AC4T_N_SFB] = { 55, 43, 33, 18, 12 }; /* Table B.1 */
    int i, k;
    for (i = 0; i < AC4T_N_SFB; i++) {
        const ac4t_sfb *t = &AC4T_SFB[i];
        CHECK(t->num_sfb == nsfb[i], "num_sfb %d for %d", t->num_sfb,
              t->length);
        CHECK(t->off[0] == 0 && t->off[t->num_sfb] == t->length,
              "band table %d ends", t->length);
        for (k = 0; k < t->num_sfb; k++)
            CHECK(t->off[k] < t->off[k + 1] && (t->off[k + 1] - t->off[k]) % 4 == 0,
                  "band table %d band %d", t->length, k);
    }
}

/* push blocks through the filterbank; -> output energy / input energy for
   interior blocks (first and last frame zero, so everything is flushed) */
static double isometry(const ac4_fb_tables *t, const int *seq, int nb,
                       int corrupt)
{
    ac4_fb_state st;
    double *X = (double *)malloc(sizeof(double) * 1536);
    double *out = (double *)malloc(sizeof(double) * 1536);
    double tmp[4 * 1536], ein = 0.0, eout = 0.0;
    int b, k;
    memset(&st, 0, sizeof(st));
    for (b = 0; b < nb; b++) {
        int n = seq[b];
        int interior = b > 0 && b < nb - 1;
        for (k = 0; k < n; k++) {
            X[k] = interior ? urand() : 0.0;
            ein += X[k] * X[k];
        }
        ac4_fb_block(t, &st, X, n, out, tmp);
        if (corrupt && b == nb / 2)
            st.overlap[700] += 0.5;
        for (k = 0; k < n; k++)
            eout += out[k] * out[k];
    }
    free(X);
    free(out);
    return eout / ein;
}

static void test_filterbank(void)
{
    ac4_fb_tables *t = ac4_fb_tables_new();
    static const int steady[] = { 1536, 1536, 1536, 1536, 1536 };
    int sw[64], n = 0, i;
    double r;
    CHECK(t != NULL, "fb tables");
    if (!t)
        return;
    r = isometry(t, steady, 5, 0);
    CHECK(fabs(r - 1.0) < 1e-12, "steady 1536: energy ratio %.15f", r);
    /* switching: long, 8x192, 4x384 (half 768 + 2x384), 16x96, long ... */
    sw[n++] = 1536;
    sw[n++] = 1536;
    for (i = 0; i < 8; i++) sw[n++] = 192;
    for (i = 0; i < 4; i++) sw[n++] = 384;
    sw[n++] = 1536;
    for (i = 0; i < 16; i++) sw[n++] = 96;
    for (i = 0; i < 2; i++) sw[n++] = 768;
    for (i = 0; i < 8; i++) sw[n++] = 96;
    for (i = 0; i < 4; i++) sw[n++] = 192;
    sw[n++] = 1536;
    sw[n++] = 1536;
    sw[n++] = 1536;
    r = isometry(t, sw, n, 0);
    CHECK(fabs(r - 1.0) < 1e-12, "block switching: energy ratio %.15f", r);
    r = isometry(t, sw, n, 1);
    CHECK(fabs(r - 1.0) > 1e-6, "negative control did not fail (%.15f)", r);
    ac4_fb_tables_free(t);
}

static void test_resampler(void)
{
    ac4_rs_filter *f = ac4_rs_filter_new();
    ac4_rs_state a, b;
    int N = 60000, i, na = 0, nb = 0, pos = 0;
    double *x = (double *)malloc(sizeof(double) * (size_t)N);
    double *ya = (double *)malloc(sizeof(double) * (size_t)(N * 2));
    double *yb = (double *)malloc(sizeof(double) * (size_t)(N * 2));
    static const int cuts[] = { 1, 7, 959, 961, 5000, 13, 1536, 7777 };
    double md = 0.0;
    CHECK(f && x && ya && yb, "alloc");
    for (i = 0; i < N; i++)
        x[i] = urand();
    ac4_rs_init(&a);
    ac4_rs_init(&b);
    na = ac4_rs_push(f, &a, x, N, ya);
    for (i = 0; i < 8; i++) {
        nb += ac4_rs_push(f, &b, x + pos, cuts[i], yb + nb);
        pos += cuts[i];
    }
    nb += ac4_rs_push(f, &b, x + pos, N - pos, yb + nb);
    CHECK(na == (int)(((long long)N * 1001 + 959) / 960), "output count %d",
          na);
    CHECK(na == nb, "split output count %d vs %d", nb, na);
    for (i = 0; i < na && i < nb; i++)
        if (fabs(ya[i] - yb[i]) > md)
            md = fabs(ya[i] - yb[i]);
    CHECK(md == 0.0, "split differs by %g", md);
    /* DC gain */
    for (i = 0; i < N; i++)
        x[i] = 1.0;
    ac4_rs_init(&a);
    na = ac4_rs_push(f, &a, x, N, ya);
    CHECK(fabs(ya[na / 2] - 1.0) < 1e-3, "DC gain %.6f", ya[na / 2]);
    free(x);
    free(ya);
    free(yb);
    ac4_rs_filter_free(f);
}

int main(int argc, char **argv)
{
    if (argc > 1 && !strcmp(argv[1], "--dump-aspx")) {
        dump_aspx();
        return 0;
    }
    test_bits();
    test_huffman();
    test_aspx_books();
    test_sfb();
    test_filterbank();
    test_resampler();
    test_qmf();
    test_aspx_tables();
    if (fails) {
        printf("UNIT TESTS FAILED (%d)\n", fails);
        return 1;
    }
    printf("UNIT TESTS PASS\n");
    return 0;
}
