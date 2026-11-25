/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Unit tests for libac4dec's building blocks (no captures needed):
 *   bit reader and variable_bits; every Huffman codebook builds as a
 *   complete prefix code and round-trips every symbol; the scale factor
 *   band tables; the filterbank is a lapped orthogonal transform with the
 *   standard's per-block gain (output energy == sum of 1536 / N times
 *   the block's input energy) for steady and switching block lengths,
 *   with a negative control; the resampler gives the same samples however
 *   the input is split, the right count, and unit DC gain; every A-SPX
 *   and A-CPL codebook builds and round-trips; the QMF bank (through the
 *   A-SPX pair renderer with no A-SPX data) reconstructs its input 577
 *   samples late, with a negative control; the A-SPX tables obey the
 *   structural rules of clause 5.7.6.3.1; channel_mode decodes every code
 *   of TS 103 190-2 table 56 and ac4_substream_info_chan reads the
 *   top-channel fields of 7.0.4 .. 9.1.4; the spectral noise fill's
 *   generator (Pseudocodes 24, 57) and levels (Pseudocodes 22, 23).
 *
 *   tests/unit_test --dump-snf    prints random channels noise-filled
 *   (c/tests/check_noise_fill.py compares them with the Python).
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
#include "ac4dec_input.h"
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
                    if (ac4_aspx_tables(&c, 0, &t) != 0)
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
                    if (ac4_aspx_tables(&c, 0, &t) != 0) {
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
    /* Table B.1 */
    static const int nsfb[AC4T_N_SFB] = { 63, 61, 55, 49, 49, 43, 36, 36,
                                          33, 20, 20, 18, 14, 14, 12 };
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
static double isometry_fl(const ac4_fb_tables *t, const int *seq, int nb,
                          int corrupt, int fl)
{
    ac4_fb_state st;
    double *X = (double *)malloc(sizeof(double) * 2048);
    double *out = (double *)malloc(sizeof(double) * 2048);
    double tmp[4 * 2048], ein = 0.0, eout = 0.0;
    int b, k;
    memset(&st, 0, sizeof(st));
    st.n_full = fl;
    for (b = 0; b < nb; b++) {
        int n = seq[b];
        int interior = b > 0 && b < nb - 1;
        for (k = 0; k < n; k++) {
            X[k] = interior ? urand() : 0.0;
            /* the standard's gain: block b carries (1536 / N_b) of an
               orthonormal transform's energy (TS 103 190-1 5.5.2.2) */
            ein += X[k] * X[k] * (1536.0 / n);
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

static double isometry(const ac4_fb_tables *t, const int *seq, int nb,
                       int corrupt)
{
    return isometry_fl(t, seq, nb, corrupt, 1536);
}

/* one block-switching sequence in a frame of fl: long, short groups of
   every length of the frame's family (Tables 100, 103), long */
static int family_seq(int fl, int *sw)
{
    int n = 0, i, d;
    sw[n++] = fl;
    sw[n++] = fl;
    for (d = 2; d <= 16 && fl / d >= 96; d *= 2) {
        for (i = 0; i < d; i++)
            sw[n++] = fl / d;
        sw[n++] = fl;
    }
    if (fl >= 1536) {               /* different halves: 8 x fl/16 + fl/2 */
        for (i = 0; i < 8; i++)
            sw[n++] = fl / 16;
        sw[n++] = fl / 2;
        sw[n++] = fl;
    }
    sw[n++] = fl;
    sw[n++] = fl;
    return n;
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
    {
        /* every frame length of Table 83, with its block switching */
        static const int fls[] = { 2048, 1920, 1536, 1024, 960, 768, 512,
                                   384 };
        int j;
        for (j = 0; j < 8; j++) {
            int seq[128], m = family_seq(fls[j], seq);
            r = isometry_fl(t, seq, m, 0, fls[j]);
            CHECK(fabs(r - 1.0) < 1e-12, "frame %d: energy ratio %.15f",
                  fls[j], r);
        }
    }
    ac4_fb_tables_free(t);
}

static void test_resampler_ratio(int up, int down)
{
    ac4_rs_filter *f = ac4_rs_filter_new_ratio(up, down);
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
    CHECK(na == (int)(((long long)N * up + down - 1) / down),
          "output count %d (%d/%d)", na, up, down);
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

/* Table 83's three ratios, and 1 (index 13: the samples pass through) */
static void test_resampler(void)
{
    ac4_rs_filter *f = ac4_rs_filter_new_ratio(1, 1);
    ac4_rs_state st;
    double x[100], y[100];
    int i, n;
    test_resampler_ratio(1001, 960);
    test_resampler_ratio(25, 24);
    test_resampler_ratio(15, 16);
    for (i = 0; i < 100; i++)
        x[i] = urand();
    ac4_rs_init(&st);
    n = ac4_rs_push(f, &st, x, 100, y);
    CHECK(n == 100 && !memcmp(x, y, sizeof(x)), "ratio 1 is not a copy");
    ac4_rs_filter_free(f);
}

/* TS 103 190-2 table 56 (channel_mode -> ch_mode) and the four
   top-channel bits of ac4_substream_info_chan for 7.0.4 .. 9.1.4 */
static int bits_from(const char *s, uint8_t *buf, size_t cap)
{
    size_t i, n = strlen(s);
    memset(buf, 0, cap);
    for (i = 0; i < n && i / 8 < cap; i++)
        if (s[i] == '1')
            buf[i / 8] |= (uint8_t)(0x80 >> (i & 7));
    return (int)n;
}

static void test_channel_mode(void)
{
    static const struct { const char *code; long ch_mode; } T[] = {
        {"0", 0}, {"10", 1}, {"1100", 2}, {"1101", 3}, {"1110", 4},
        {"1111000", 5}, {"1111001", 6}, {"1111010", 7}, {"1111011", 8},
        {"1111100", 9}, {"1111101", 10}, {"11111100", 11},
        {"11111101", 12}, {"111111100", 13}, {"111111101", 14},
        {"111111110", 15}, {"111111111" "000", 16},
        {"111111111" "011" "100", 26},
    };
    uint8_t buf[8];
    jmp_buf jb;
    size_t i;
    for (i = 0; i < sizeof(T) / sizeof(T[0]); i++) {
        ac4_bits b;
        int n = bits_from(T[i].code, buf, sizeof(buf));
        long cm;
        b.d = buf;
        b.nbits = 64;
        b.p = 0;
        b.jb = &jb;
        b.eof_code = AC4E_TOC_PARSE;
        if (setjmp(jb)) {
            CHECK(0, "channel_mode %s raised", T[i].code);
            continue;
        }
        cm = ac4_toc_channel_mode(&b);
        CHECK(cm == T[i].ch_mode && (int)b.p == n,
              "channel_mode %s -> %ld after %d bits (want %ld, %d)",
              T[i].code, cm, (int)b.p, T[i].ch_mode, n);
        if (T[i].ch_mode >= 11 && T[i].ch_mode <= 14) {
            /* code, 4 top-channel bits, b_bitrate_info 1 + 010,
               b_audio_ndot; fs_index 0, frame_rate_factor 1 */
            char s[64];
            ac4_toc t;
            memset(&t, 0, sizeof(t));
            snprintf(s, sizeof(s), "%s" "1011" "1010" "1", T[i].code);
            n = bits_from(s, buf, sizeof(buf));
            b.p = 0;
            ac4_toc_substream_info_chan(&b, 0, 1, &t);
            CHECK((int)b.p == n && t.n_cmodes == 1
                  && t.cmodes[0] == T[i].ch_mode,
                  "substream_info_chan %s ended at bit %d (want %d)",
                  T[i].code, (int)b.p, n);
        }
    }
}

/* ------------------------------------------------------------ noise fill */
void ac4_spectra_init(void);            /* src/spectra.c */
/* GetRandomNoiseValue after ResetRandGenStateSnf: state and first values
   (the same numbers as tests/test_units.py SNF_FIRST) */
static void test_snf_rng(void)
{
    static const struct {
        int seq;
        unsigned st[4];
        double v[6];
    } T[] = {
        {0, {0, 0, 1, 0}, {1.6131210327148438, 0.3268809914588928,
                           -0.3108939826488495, 1.3197810649871826,
                           1.1259419918060303, -0.547692060470581}},
        {1, {0, 1, 130, 130}, {-0.04938599839806557, 1.502228021621704,
                               -0.13683800399303436, -0.9735279679298401,
                               1.7364200353622437, -0.6638969779014587}},
        {255, {0, 255, 255, 126}, {-0.05372801423072815, 0.9366300106048584,
                                   -1.3645939826965332, -1.0417840480804443,
                                   -0.7589679956436157, -1.3996400833129883}},
        {1020, {0, 252, 115, 248}, {2.1083831787109375, -0.534654974937439,
                                    -0.47258099913597107, 0.04947199672460556,
                                    -0.47846800088882446, 1.1031620502471924}},
    };
    size_t i;
    int j;
    for (i = 0; i < sizeof(T) / sizeof(T[0]); i++) {
        ac4_snf_rng r;
        double m = 0.0, v2 = 0.0;
        ac4_snf_reset(&r, T[i].seq);
        CHECK(r.a == T[i].st[0] && r.b == T[i].st[1]
              && r.state == T[i].st[2] && r.cur == T[i].st[3],
              "snf reset %d: state %u %u %u %u", T[i].seq, r.a, r.b,
              r.state, r.cur);
        for (j = 0; j < 6; j++) {
            double v = ac4_snf_next(&r);
            CHECK(v == T[i].v[j], "snf %d value %d: %.17g", T[i].seq, j, v);
        }
        ac4_snf_reset(&r, T[i].seq);
        for (j = 0; j < 65280; j++) {
            double v = ac4_snf_next(&r);
            m += v;
            v2 += v * v;
        }
        m /= 65280.0;
        v2 = v2 / 65280.0 - m * m;
        CHECK(fabs(m) < 0.02 && fabs(v2 - 1.0) < 0.05,
              "snf %d: mean %g variance %g", T[i].seq, m, v2);
    }
}

/* a random channel for the noise fill: framing bits (returned as a 0/1
   string), lines, scale factors and noise fill deltas */
static const int SHORTL[4] = { 96, 192, 384, 768 };
static const int NGRP[4][4] = { {15, 10, 8, 7}, {10, 7, 4, 3},
                                {8, 4, 3, 1}, {7, 3, 1, 1} };
static int msfb_bits(int len) { return len >= 384 ? 6 : len == 192 ? 5 : 4; }
static int nsfb(int len)
{
    int i;
    for (i = 0; i < AC4T_N_SFB; i++)
        if (AC4T_SFB[i].length == len)
            return AC4T_SFB[i].num_sfb;
    return 0;
}
static unsigned rnd_u(unsigned n) { return (unsigned)((urand() + 1.0) * 0.5 * n) % n; }
static void put(char *s, int *n, unsigned v, int bits)
{
    int i;
    for (i = bits - 1; i >= 0; i--)
        s[(*n)++] = (char)('0' + ((v >> i) & 1));
    s[*n] = 0;
}

static int framing_from(const uint8_t *buf, ac4_framing *f)
{
    ac4_bits b;
    jmp_buf jb;
    b.d = buf;
    b.nbits = 128;
    b.p = 0;
    b.jb = &jb;
    b.eof_code = AC4E_ELEMENT_PARSE;
    if (setjmp(jb))
        return -1;
    ac4_framing_parse(&b, f, 1536);
    return 0;
}

/* packed spectrum + noise fill (generator reset from seq); -> lines, or
   -1 when a level check raised */
static int spec_fill(const ac4_chan *c, double *spec, int seq,
                     ac4_snf_rng *r)
{
    jmp_buf jb;
    int n;
    if (setjmp(jb))
        return -1;
    n = ac4_packed_spectrum(c, spec, &jb);
    ac4_snf_reset(r, seq);
    ac4_noise_fill(c, spec, n, r, &jb);
    return n;
}

static void snf_case(char *bits, ac4_framing *f, ac4_chan *c, uint8_t *buf)
{
    int n = 0, tl0, tl1, i, g, sfb, k;
    if (rnd_u(3) == 0) {
        put(bits, &n, 1, 1);
        put(bits, &n, rnd_u((unsigned)nsfb(1536) + 1), 6);
    } else {
        tl0 = (int)rnd_u(4);
        tl1 = rnd_u(3) ? tl0 : (int)rnd_u(4);
        put(bits, &n, 0, 1);
        put(bits, &n, (unsigned)tl0, 2);
        put(bits, &n, (unsigned)tl1, 2);
        put(bits, &n, rnd_u((unsigned)nsfb(SHORTL[tl0]) + 1),
            msfb_bits(SHORTL[tl0]));
        if (tl0 != tl1)
            put(bits, &n, rnd_u((unsigned)nsfb(SHORTL[tl1]) + 1),
                msfb_bits(SHORTL[tl1]));
        for (i = 0; i < NGRP[tl0][tl1]; i++)
            put(bits, &n, rnd_u(2), 1);
    }
    memset(buf, 0, 16);
    for (i = 0; i < n; i++)
        if (bits[i] == '1')
            buf[i / 8] |= (uint8_t)(0x80 >> (i & 7));
    memset(c, 0, sizeof(*c));
    if (framing_from(buf, f) != 0)
        return;
    c->fr = f;
    c->has_snf = 1;
    c->n_lines = 0;
    for (g = 0; g < f->num_groups; g++) {
        const ac4t_sfb *t = f->gtab[g];
        int nw = f->nwin[g], base = f->gbase[g];
        int m = f->gmax[g] < t->num_sfb ? f->gmax[g] : t->num_sfb;
        for (sfb = 0; sfb < m; sfb++) {
            int lo = base + t->off[sfb] * nw, hi = base + t->off[sfb + 1] * nw;
            c->sfs[g][sfb] = AC4_SF_NONE;
            c->snf[g][sfb] = AC4_SF_NONE;
            if (rnd_u(5) < 2) {                      /* coded band */
                c->sfs[g][sfb] = (int16_t)(80 + rnd_u(40));
                for (k = lo; k < hi; k++)
                    c->lines[k] = (int32_t)rnd_u(9) - 4;
                c->lines[lo] = 1 + (int32_t)rnd_u(5);
            } else {                                 /* noise band */
                c->snf[g][sfb] = (int16_t)((int)rnd_u(22) - 17);
            }
            if (hi > c->n_lines)
                c->n_lines = hi;
        }
    }
}

/* the noise fill against its pseudocode, on fixed bands of a long block:
   band 0 coded, band 1 noise at +2, band 2 escaped, band 3 noise at -1,
   band 4 coded; and spec NULL advances the generator as filling does */
static void test_noise_fill(void)
{
    uint8_t buf[16];
    ac4_framing f;
    ac4_chan c;
    ac4_snf_rng r1, r2;
    double spec[1536], plain[1536], lev0, e = 0.0;
    const int16_t *off;
    int k, sfb, ok = 1;
    jmp_buf jb;
    ac4_spectra_init();
    memset(buf, 0, sizeof(buf));
    buf[0] = 0x8a;                       /* 1 000101: long, max_sfb 5 */
    if (framing_from(buf, &f) != 0) {
        CHECK(0, "noise fill: framing raised");
        return;
    }
    memset(&c, 0, sizeof(c));
    c.fr = &f;
    off = f.gtab[0]->off;
    c.n_lines = off[5];
    for (sfb = 0; sfb < 5; sfb++)
        c.sfs[0][sfb] = c.snf[0][sfb] = AC4_SF_NONE;
    c.lines[off[0]] = 1;
    c.lines[off[0] + 1] = -1;
    c.lines[off[0] + 2] = 2;
    c.lines[off[4]] = 3;
    c.sfs[0][0] = 100;
    c.sfs[0][4] = 104;
    c.has_snf = 0;
    CHECK(spec_fill(&c, plain, 5, &r1) == off[5], "noise fill: no snf");
    c.has_snf = 1;
    c.snf[0][1] = 2;
    c.snf[0][2] = -17;
    c.snf[0][3] = -1;
    CHECK(spec_fill(&c, spec, 5, &r1) == off[5], "noise fill raised");
    for (k = off[0]; k < off[1]; k++)
        e += plain[k] * plain[k];
    lev0 = 1.44269504 * log(e / (double)(off[1] - off[0]));
    ac4_snf_reset(&r2, 5);
    for (k = 0; k < off[5]; k++) {
        double want = plain[k];
        if (k >= off[1] && k < off[2])
            want = ac4_snf_next(&r2) * pow(2.0, 0.5 * (lev0 + 2));
        else if (k >= off[3] && k < off[4])
            want = ac4_snf_next(&r2) * pow(2.0, 0.5 * (lev0 + 1));
        ok &= fabs(spec[k] - want) <= 1e-12 * fabs(want);
    }
    CHECK(ok && plain[off[4]] == pow(3.0, 4.0 / 3.0) * 2.0,
          "noise fill levels / escape / order");
    /* spec NULL: the generator advances exactly as filling does */
    ac4_snf_reset(&r2, 5);
    if (!setjmp(jb))
        ac4_noise_fill(&c, NULL, 0, &r2, &jb);
    CHECK(r1.a == r2.a && r1.b == r2.b && r1.state == r2.state
          && r1.cur == r2.cur, "noise fill: spec NULL advances alike");
}

/* random channels, noise-filled, printed for c/tests/check_noise_fill.py */
static void dump_snf(void)
{
    int i, g, sfb, k;
    ac4_spectra_init();
    for (i = 0; i < 400; i++) {
        char bits[64];
        uint8_t buf[16];
        ac4_framing f;
        ac4_chan c;
        ac4_snf_rng r;
        double spec[1536];
        int seq = (int)rnd_u(1021), n;
        snf_case(bits, &f, &c, buf);
        printf("case %d %s %d %d\n", i, bits, seq, c.n_lines);
        printf("lines");
        for (k = 0; k < c.n_lines; k++)
            if (c.lines[k])
                printf(" %d:%d", k, c.lines[k]);
        printf("\n");
        for (g = 0; g < f.num_groups; g++) {
            int m = f.gmax[g] < f.gtab[g]->num_sfb ? f.gmax[g]
                                                    : f.gtab[g]->num_sfb;
            printf("group %d", g);
            for (sfb = 0; sfb < m; sfb++)
                printf(" %d,%d", c.sfs[g][sfb], c.snf[g][sfb]);
            printf("\n");
        }
        n = spec_fill(&c, spec, seq, &r);
        if (n < 0) {
            printf("raised\n");
            continue;
        }
        printf("spec");
        for (k = 0; k < n; k++)
            printf(" %.17g", spec[k]);
        printf("\n");
    }
}

/* ------------------------------------------------------------ phase A */
/* the same bitstreams as tests/test_phase_a.py (written there from the
   syntax with a bit writer) */
static const uint8_t TOC_TWO[] =
    "\x80\x54\xe8\x30\xb8\x01\x00\x00\x4c\x2f\x00\x40\x01\x12\xf8\xb1"
    "\x04\xca\xdd\x70\xb6\x09\x99\xcb\x00\x60\x14\x03\x80"
    "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
static const uint8_t TOC_SHARED[] =
    "\x80\x54\xe8\x30\x00\x40\x00\x13\x00\x04\x00\x01\x2e\x2a\x00\x40\x10";
static const uint8_t TOC_NDOT[] =
    "\x80\x54\xd3\x00\x04\x00\x01\x2e\x2a\x00\x40\x10";

static int toc_of(const uint8_t *d, size_t n, ac4_toc *t)
{
    jmp_buf jb;
    ac4_bits b;
    b.d = d;
    b.nbits = n * 8;
    b.p = 0;
    b.jb = &jb;
    b.eof_code = AC4E_TOC_PARSE;
    if (setjmp(jb))
        return -1;
    ac4_toc_parse(&b, t);
    return 0;
}

static void test_toc_presentations(void)
{
    ac4_toc t;
    const ac4_toc_sub *s;
    ac4dec_presentation p[4];
    int n;
    CHECK(toc_of(TOC_TWO, 29, &t) == 0, "two-presentation TOC parses");
    CHECK(t.n_pres == 2 && t.n_groups == 2 && t.n_groups_all == 2,
          "presentations %d groups %d", t.n_pres, t.n_groups);
    CHECK(t.pres[0].presentation_id == 10 && t.pres[1].presentation_id == 11,
          "presentation_id");
    CHECK(t.groups[0].subs[0].channel_mode == 4
          && t.groups[0].subs[0].index == 1
          && t.groups[1].subs[0].channel_mode == 1
          && t.groups[1].subs[0].index == 2
          && t.groups[1].subs[0].ndot == 0, "substreams");
    CHECK(!strcmp(t.groups[0].language, "en")
          && !strcmp(t.groups[1].language, "fr")
          && t.groups[1].content_classifier == 5, "content_type");
    CHECK(t.n_sizes == 3 && t.sizes[0] == 3 && t.sizes[1] == 5
          && t.sizes[2] == 7, "sizes");
    CHECK(ac4_toc_choose(&t, -1, -1, NULL) == 0, "default choice");
    CHECK(ac4_toc_choose(&t, -1, -1, "fr") == 1, "language fr");
    CHECK(ac4_toc_choose(&t, -1, -1, "FR-ca") == 1, "language FR-ca");
    CHECK(ac4_toc_choose(&t, -1, -1, "de") == 0, "language de");
    CHECK(ac4_toc_choose(&t, 1, -1, NULL) == 1, "by index");
    CHECK(ac4_toc_choose(&t, 2, -1, NULL) == -1, "index out of range");
    CHECK(ac4_toc_choose(&t, -1, 11, NULL) == 1, "by id");
    CHECK(ac4_toc_choose(&t, -1, 12, NULL) == -1, "unknown id");
    s = ac4_toc_audio_substream(&t, 1);
    CHECK(s && s->index == 2, "audio substream of presentation 1");
    n = ac4dec_presentations(TOC_TWO, 29, p, 4);
    CHECK(n == 2 && p[0].is_default && !p[1].is_default
          && !strcmp(p[1].language, "fr") && !strcmp(p[1].content, "commentary")
          && p[0].channel_mode == 4 && p[1].channel_mode == 1,
          "ac4dec_presentations");
    /* two presentations sharing group 0: one group info is read */
    CHECK(toc_of(TOC_SHARED, sizeof(TOC_SHARED) - 1, &t) == 0
          && t.n_groups_all == 1 && t.n_sizes == 2 && t.sizes[0] == 2
          && t.sizes[1] == 4, "shared group read once");
    /* b_iframe_global 0, b_audio_ndot 1 */
    CHECK(toc_of(TOC_NDOT, sizeof(TOC_NDOT) - 1, &t) == 0
          && t.b_iframe_global == 0 && t.groups[0].subs[0].ndot == 1,
          "b_audio_ndot");
}

static void test_stereo_data(void)
{
    static const struct { const char *d; int n, end; } P[] = {
        { "\x00\x00\x30\x06\x43\x23\x59\xc0", 8, 46 },          /* proc on */
        { "\x00\x00\x08\x08\x0c\x86\x46\xb3\x80", 9, 53 },      /* proc off */
        { "\x00\x00\x0c\x00\x00\x00\x00", 7, -1 },              /* SSF */
    };
    ac4_huff huff[AC4T_N_BOOKS];
    ac4_books bk;
    static ac4_element e;
    jmp_buf jb;
    volatile int i;
    memset(&bk, 0, sizeof(bk));
    for (i = 0; i < AC4T_N_BOOKS; i++)
        ac4_huff_build(&huff[i], &AC4T_BOOKS[i]);
    for (i = 1; i <= 11; i++)
        bk.cb[i] = &huff[i - 1];
    bk.sf = &huff[AC4T_BOOK_SF];
    bk.snf = &huff[AC4T_BOOK_SNF];
    bk.fl = 1536;
    for (i = 0; i < 3; i++) {
        ac4_bits b;
        int code;
        b.d = (const uint8_t *)P[i].d;
        b.nbits = (size_t)P[i].n * 8;
        b.p = 0;
        b.jb = &jb;
        b.eof_code = AC4E_ELEMENT_PARSE;
        code = setjmp(jb);
        if (code) {
            CHECK(P[i].end < 0, "stereo_data %d raised %d", i, code);
            continue;
        }
        ac4_parse_pair(&b, &bk, 0, &e);
        CHECK((int)e.bits_end == P[i].end && e.fr[0].is_long
              && e.fr[1].is_long,
              "stereo_data %d ended at %d (want %d)", i, (int)e.bits_end,
              P[i].end);
    }
    for (i = 0; i < AC4T_N_BOOKS; i++)
        ac4_huff_free(&huff[i]);
}

static size_t put_sync(uint8_t *o, const uint8_t *raw, size_t n, int crc,
                       int escape)
{
    size_t k = 0;
    o[k++] = 0xAC;
    o[k++] = crc ? 0x41 : 0x40;
    if (escape) {
        o[k++] = 0xFF; o[k++] = 0xFF;
        o[k++] = (uint8_t)(n >> 16); o[k++] = (uint8_t)(n >> 8);
        o[k++] = (uint8_t)n;
    } else {
        o[k++] = (uint8_t)(n >> 8); o[k++] = (uint8_t)n;
    }
    memcpy(o + k, raw, n);
    k += n;
    if (crc) {
        unsigned c = ac4in_crc16(o + 2, k - 2, 0);
        o[k++] = (uint8_t)(c >> 8);
        o[k++] = (uint8_t)c;
    }
    return k;
}

static void test_inputs(void)
{
    uint8_t raw[4][64], buf[512];
    size_t n = 0, i;
    ac4in_frame *fr;
    ac4in_info info;
    long m;
    CHECK(ac4in_crc16((const uint8_t *)"123456789", 9, 0) == 0xFEE8,
          "CRC-16 check value");
    for (i = 0; i < 4; i++)
        memset(raw[i], 0x80 + (int)i, sizeof(raw[i]));
    n += put_sync(buf + n, raw[0], 20, 0, 0);
    n += put_sync(buf + n, raw[1], 23, 1, 1);
    memcpy(buf + n, "\x12\xac\x41garbage", 10);   /* lost sync */
    n += 10;
    i = n;
    n += put_sync(buf + n, raw[2], 26, 1, 0);
    buf[i + 6] ^= 0x10;                             /* bad CRC */
    n += put_sync(buf + n, raw[3], 29, 1, 0);
    CHECK(ac4in_detect(buf, n) == AC4IN_SYNC, "detect sync");
    m = ac4in_read(buf, n, &fr, &info);
    CHECK(m == 4 && info.n_crc == 3 && info.n_crc_bad == 1
          && info.n_skipped == 10, "sync frames: %ld frames, %ld crc, "
          "%ld bad, %ld skipped", m, info.n_crc, info.n_crc_bad,
          info.n_skipped);
    if (m == 4) {
        CHECK(fr[0].len == 20 && !memcmp(fr[0].p, raw[0], 20)
              && fr[1].len == 23 && !memcmp(fr[1].p, raw[1], 23)
              && fr[2].len == 0 && fr[3].len == 29
              && !memcmp(fr[3].p, raw[3], 29), "sync frame contents");
    }
    free(fr);
    ac4in_free(&info);
}

/* --dump-input FILE: kind, frame count, then per frame its length and
   FNV-1a hash (c/tests/check_inputs.py compares with ac4/container.py) */
static int dump_input(const char *path)
{
    FILE *f = fopen(path, "rb");
    uint8_t *d = NULL;
    size_t n = 0, cap = 0, r;
    ac4in_frame *fr;
    ac4in_info info;
    long k, m;
    if (!f)
        return 1;
    for (;;) {
        if (n == cap) {
            cap = cap ? cap * 2 : 1 << 20;
            d = (uint8_t *)realloc(d, cap);
        }
        r = fread(d + n, 1, cap - n, f);
        if (!r)
            break;
        n += r;
    }
    fclose(f);
    m = ac4in_read(d, n, &fr, &info);
    printf("%d %ld %ld %ld %ld\n", (int)info.kind, m, info.n_crc_bad,
           info.n_skipped, info.ts_cc_errors);
    for (k = 0; k < m; k++) {
        uint32_t h = 2166136261u;
        size_t i;
        for (i = 0; i < fr[k].len; i++)
            h = (h ^ fr[k].p[i]) * 16777619u;
        printf("%zu %08x\n", fr[k].len, (unsigned)h);
    }
    free(fr);
    ac4in_free(&info);
    free(d);
    return 0;
}

/* loudness.c: the film standard curve of Table 162 and the stage's
   normalisation gain 2^((Lout - Lin) / 6) */
static void test_loudness(void)
{
    /* l0l l0h gmb lmb gsb lsb gmc lmc gsc lsc */
    static const double fs[10] = { 0, 5, 6, -12, 0, 0, -24, 35, -5, 15 };
    static ac4_leveler lv;
    ac4_meta md;
    double buf[2][1536], *pp[2] = { buf[0], buf[1] };
    int i;
    CHECK(ac4_curve_gain(2.0, fs) == 0.0, "null band");
    CHECK(ac4_curve_gain(-30.0, fs) == 6.0, "max boost");
    CHECK(fabs(ac4_curve_gain(-6.0, fs) - 3.0) < 1e-12, "boost slope");
    CHECK(fabs(ac4_curve_gain(10.0, fs) + 2.5) < 1e-12, "cut section");
    CHECK(fabs(ac4_curve_gain(25.0, fs) + 14.5) < 1e-12, "cut slope");
    CHECK(ac4_curve_gain(60.0, fs) == -24.0, "max cut");
    memset(&md, 0, sizeof(md));
    md.valid = 1;
    md.dialnorm = -18.0;
    {
        static const int LR[2] = { 0, 1 };
        ac4_leveler_init(&lv, 2, LR, 0, -24.0, AC4_DRC_OFF, 48000.0);
    }
    for (i = 0; i < 1536; i++)
        buf[0][i] = buf[1][i] = 0.25 * AC4_FULL_SCALE;
    ac4_leveler_frame(&lv, &md, pp, 1536);
    CHECK(fabs(buf[0][100] - 0.125) < 1e-12 && fabs(buf[1][1535] - 0.125)
          < 1e-12, "normalisation -18 -> -24 dB: %g", buf[0][100]);
}

/* ac4dec_open_options: the text settings reach the decoder, bad ones are
   refused with a reason */
static void test_options(void)
{
    char err[128];
    ac4dec *d;
    CHECK(!strcmp(ac4dec_version(), AC4DEC_VERSION), "version");
    d = ac4dec_open_options("lookahead=12, layout=5.1 loudness=1 "
                            "target_level=-24 drc=auto", err, sizeof(err));
    CHECK(d != NULL, "open with options: %s", err);
    if (d) {
        CHECK(ac4dec_delay(d) == 12, "lookahead reached the decoder: %d",
              ac4dec_delay(d));
        ac4dec_close(d);
    }
    d = ac4dec_open_options(NULL, err, sizeof(err));
    CHECK(d != NULL && ac4dec_delay(d) == AC4DEC_DEFAULT_LOOKAHEAD,
          "no options = defaults");
    if (d)
        ac4dec_close(d);
    d = ac4dec_open_options("layout=9.1", err, sizeof(err));
    CHECK(d == NULL && strstr(err, "layout=9.1"), "bad layout: %s", err);
    d = ac4dec_open_options("speed=2", err, sizeof(err));
    CHECK(d == NULL && strstr(err, "speed=2"), "unknown key: %s", err);
    d = ac4dec_open_options("lookahead", err, sizeof(err));
    CHECK(d == NULL && strstr(err, "key=value"), "no value: %s", err);
    d = ac4dec_open_options("lookahead=3", err, 0);
    CHECK(d == NULL, "lookahead below the minimum");
}

/* ac4dec_decode_interleaved / _flush_interleaved give the same samples as
   ac4dec_decode_frame / ac4dec_flush, frame for frame (a synthetic stream
   of tests/data; skipped when the file is not there) */
static void test_interleaved(void)
{
    static const char *path = "tests/data/v1_cfg7.syn";
    FILE *fp = fopen(path, "rb");
    static uint8_t data[1 << 16];
    static float il[AC4DEC_MAX_OUT * AC4DEC_MAX_CHANNELS];
    ac4dec_config cfg;
    ac4dec *a, *b;
    ac4in_frame *fr = NULL;
    ac4in_info info;
    ac4dec_frame f;
    char err[128];
    long m, k, outs = 0, diffs = 0, got_a = 0, got_b = 0;
    size_t n;
    int ch = 0, rate = 0, st = 0, r, c, i;
    if (!fp) {
        printf("skip interleaved: no %s\n", path);
        return;
    }
    n = fread(data, 1, sizeof(data), fp);
    fclose(fp);
    m = ac4in_read(data, n, &fr, &info);
    CHECK(m > 8, "frames in %s: %ld", path, m);
    memset(&cfg, 0, sizeof(cfg));
    cfg.lookahead = 8;
    cfg.target_level = -24.0;
    a = ac4dec_open(&cfg);
    b = ac4dec_open_options("lookahead=8", err, sizeof(err));
    CHECK(a && b, "open both: %s", err);
    if (!a || !b || m <= 0)
        goto done;
    for (k = 0; k <= m; k++) {
        int ga, gb;
        memset(&f, 0, sizeof(f));
        if (k < m) {
            ga = ac4dec_decode_frame(a, fr[k].p, fr[k].len, &f);
            gb = ac4dec_decode_interleaved(b, fr[k].p, fr[k].len, il,
                                           AC4DEC_MAX_OUT, &ch, &rate, &st);
        } else {
            ga = ac4dec_flush(a, &f);
            gb = ac4dec_flush_interleaved(b, il, AC4DEC_MAX_OUT, &ch, &rate,
                                          &st);
            if (ga)
                k--;   /* keep flushing */
        }
        r = ga && f.n_channels > 0 ? f.n_samples : 0;
        if (gb != r) {
            diffs++;
            continue;
        }
        if (!r)
            continue;
        outs++;
        got_a += r;
        got_b += gb;
        if (ch != f.n_channels || rate != f.sample_rate || st != (int)f.status)
            diffs++;
        for (i = 0; i < r; i++)
            for (c = 0; c < ch; c++)
                if (il[(size_t)i * ch + c] != (float)f.pcm[c][i])
                    diffs++;
    }
    CHECK(outs >= m - 8 && got_a == got_b && diffs == 0,
          "interleaved = frame API: %ld frames out of %ld, %ld differences",
          outs, m, diffs);
done:
    if (a)
        ac4dec_close(a);
    if (b)
        ac4dec_close(b);
    free(fr);
    ac4in_free(&info);
}

/* dialogue enhancement, de_method 0 (TS 103 190-1 5.7.8.7, Tables 171,
   173, 209): the gains of a frame and one interpolated slot */
static void test_de(void)
{
    ac4_de_entry e;
    ac4_de_mat m, id;
    int k, ch = 2;
    double g = pow(10.0, 6.0 / 20.0) - 1.0;   /* G_max = (1 + 1) 3 dB */
    double re[1][64], im[1][64];
    double (*pre[3])[64], (*pim[3])[64];
    memset(&e, 0, sizeof(e));
    e.kind = AC4_DE_DATA;
    e.max_gain = 1;
    e.channel_config = 5;                     /* Left, Centre */
    for (k = 0; k < 8; k++) {
        e.par[0][k] = 3;                      /* 0.3 */
        e.par[1][k] = k == 7 ? 16 : 10;       /* 1.0; band 7: 1.75 */
    }
    ac4_de_frame(&e, 12.0, &m);
    CHECK(m.on && m.has[0] && !m.has[1] && m.has[2], "de channels");
    CHECK(fabs(m.g[0][5] - (1 + g * 0.3)) < 1e-12, "de L gain");
    CHECK(fabs(m.g[2][26] - (1 + g * 1.0)) < 1e-12
          && fabs(m.g[2][27] - (1 + g * 1.75)) < 1e-12
          && fabs(m.g[2][63] - (1 + g * 1.75)) < 1e-12, "de bands");
    memset(&id, 0, sizeof(id));
    for (k = 0; k < 64; k++) {
        re[0][k] = 1.0;
        im[0][k] = 0.0;
    }
    pre[0] = re;
    pim[0] = im;
    /* slot 3 of 24: (1 - 3.5 / 24) I + 3.5 / 24 H */
    ac4_de_apply(&id, &m, 1, &ch, (double (*const *)[64])pre,
                 (double (*const *)[64])pim, 1, 3, 24);
    CHECK(fabs(re[0][0] - ((1 - 3.5 / 24) + 3.5 / 24 * (1 + g * 1.0)))
          < 1e-12, "de interpolation");
}

int main(int argc, char **argv)
{
    if (argc > 2 && !strcmp(argv[1], "--dump-input"))
        return dump_input(argv[2]);
    if (argc > 1 && !strcmp(argv[1], "--dump-aspx")) {
        dump_aspx();
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--dump-snf")) {
        dump_snf();
        return 0;
    }
    test_bits();
    test_channel_mode();
    test_snf_rng();
    test_noise_fill();
    test_huffman();
    test_aspx_books();
    test_sfb();
    test_filterbank();
    test_resampler();
    test_qmf();
    test_aspx_tables();
    test_toc_presentations();
    test_stereo_data();
    test_inputs();
    test_loudness();
    test_de();
    test_options();
    test_interleaved();
    if (fails) {
        printf("UNIT TESTS FAILED (%d)\n", fails);
        return 1;
    }
    printf("UNIT TESTS PASS\n");
    return 0;
}
