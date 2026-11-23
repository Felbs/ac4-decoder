/* SPDX-License-Identifier: Apache-2.0 */
/*
 * A small fragmented-MP4 (moof + mdat) sample reader for AC-4, part of
 * libac4dec for convenience (the decoder itself takes raw frames). Port of
 * ac4/mp4.py: init segment + media fragments, or bare fragments; tfhd base
 * offset / default sample size / default-base-is-moof; several truns per
 * traf. The AC-4 track is the moov's first 'ac-4' sample entry; without a
 * moov, the first traf whose truns carry per-sample sizes.
 *
 * A fragment whose mdat is incomplete, or whose boxes are damaged, is
 * skipped. A sample whose offset/size points outside its mdat is returned
 * EMPTY (len 0), so the timeline is kept.
 */
#ifndef AC4DEC_MP4_H
#define AC4DEC_MP4_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t off;            /* byte offset of the frame in the input */
    size_t len;            /* 0 = an empty (lost) frame */
} ac4mp4_sample;

/* All AC-4 samples of the file in data[0..len), in file order.
   -> count, with *out a malloc'd array the caller frees (NULL when the
   count is 0); -1 when out of memory. */
long ac4mp4_samples(const uint8_t *data, size_t len, ac4mp4_sample **out);

#ifdef __cplusplus
}
#endif
#endif
