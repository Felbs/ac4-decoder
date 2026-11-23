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

/* Index 0..4 = transform length 1536, 768, 384, 192, 96. */
#define AC4T_N_SFB 5
extern const ac4t_sfb AC4T_SFB[AC4T_N_SFB];

#endif
