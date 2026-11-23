/* SPDX-License-Identifier: Apache-2.0 */
/* Numeric tables (generated into ac4_tables.c by c/tools/gen_tables.py). */
#ifndef AC4_TABLES_H
#define AC4_TABLES_H

#include <stdint.h>

typedef struct {
    const uint8_t *len;     /* codeword length per symbol (0 = unused) */
    const uint32_t *cw;     /* codeword, right-aligned */
    int n;                  /* number of symbols */
} ac4t_book;

/* Book indices: 0..10 = ASF_HCB_1..11, then scale factor and noise fill. */
#define AC4T_BOOK_SF   11
#define AC4T_BOOK_SNF  12
#define AC4T_N_BOOKS   13
extern const ac4t_book AC4T_BOOKS[AC4T_N_BOOKS];

typedef struct {
    int length;             /* transform length */
    const int16_t *off;     /* num_sfb + 1 offsets */
    int num_sfb;
} ac4t_sfb;

/* A-SPX books: 3 * kind + h, kind 0..5 = ENV_LEVEL_15, ENV_LEVEL_30,
   ENV_BALANCE_15, ENV_BALANCE_30, NOISE_LEVEL, NOISE_BALANCE; h 0..2 =
   F0, DF, DT. */
#define AC4T_N_ASPX_BOOKS 18
extern const ac4t_book AC4T_ASPX_BOOKS[AC4T_N_ASPX_BOOKS];
/* A-CPL books: 6 * kind + 3 * q + h, kind 0..3 = ALPHA, BETA, BETA3,
   GAMMA; q 0..1 = FINE, COARSE; h 0..2 = F0, DF, DT. */
#define AC4T_N_ACPL_BOOKS 24
extern const ac4t_book AC4T_ACPL_BOOKS[AC4T_N_ACPL_BOOKS];

/* Table D.3 QMF prototype (640 taps) and Table D.2 ASPX_NOISE (re, im). */
extern const double AC4T_QWIN[640];
extern const double AC4T_ASPX_NOISE[512][2];

/* Index 0..4 = transform length 1536, 768, 384, 192, 96. */
#define AC4T_N_SFB 5
extern const ac4t_sfb AC4T_SFB[AC4T_N_SFB];

#endif
