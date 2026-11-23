/* SPDX-License-Identifier: Apache-2.0 */
/*
 * AC-4 inputs, part of libac4dec for convenience. Port of ac4/container.py:
 *
 *   * AC-4 sync frames (TS 103 190-1 Annex G): 0xAC40 / 0xAC41, frame_size
 *     (16 bits, or 0xFFFF + 24 bits), raw_ac4_frame, and with 0xAC41 a
 *     crc_word (CRC-16, generator x^16 + x^15 + x^2 + 1, initial 0, no
 *     reflection, over frame_size and raw_ac4_frame). A frame whose CRC
 *     fails is returned EMPTY (the decoder conceals it); when sync is lost
 *     the reader searches for a sync word followed by a consistent frame.
 *     ID3v2 tags (HLS packed audio) are skipped.
 *   * MPEG-2 transport stream (ETSI TS 101 154): the first elementary
 *     stream of stream_type 0x06 with the DVB AC-4 descriptor (0x7F,
 *     extension 0x15) or an 'AC-4' registration descriptor; its PES
 *     payloads are sync frames.
 *   * MP4 (fragmented or not): ac4mp4_samples().
 *
 * ac4in_read detects which from the content.
 */
#ifndef AC4DEC_INPUT_H
#define AC4DEC_INPUT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AC4IN_UNKNOWN = 0,
    AC4IN_MP4 = 1,
    AC4IN_SYNC = 2,
    AC4IN_TS = 3
} ac4in_kind;

typedef struct {
    const uint8_t *p;      /* the raw AC-4 frame (NULL when len is 0) */
    size_t len;            /* 0 = an empty (lost or CRC-failed) frame */
} ac4in_frame;

typedef struct {
    ac4in_kind kind;
    long n_frames;
    long n_crc;            /* sync frames carrying a crc_word */
    long n_crc_bad;        /* of those, failed (returned empty) */
    long n_skipped;        /* bytes skipped to regain sync */
    long n_truncated;      /* a last frame cut short */
    long ts_pid;           /* -1 when not a TS */
    long ts_packets, ts_cc_errors, ts_pes;
    int has_dsi;           /* an MP4 with a dac4 box */
    int dsi_frame_rate_index, dsi_fs_index, dsi_bitstream_version;
    int dsi_n_presentations;
    uint8_t *buf;          /* owned storage the frames point into (TS) */
} ac4in_info;

/* -> "mp4" / "sync" / "ts" kind of data, AC4IN_UNKNOWN if none fits. */
ac4in_kind ac4in_detect(const uint8_t *data, size_t len);

/* All AC-4 frames of the input, in order. -> count (>= 0), *frames a
   malloc'd array (free with free(); NULL when 0) pointing into data or
   info->buf; -1 when out of memory. Free info->buf with ac4in_free. */
long ac4in_read(const uint8_t *data, size_t len, ac4in_frame **frames,
                ac4in_info *info);
void ac4in_free(ac4in_info *info);

/* The CRC of TS 103 190-1 G.4.2 over n bytes, starting from crc. */
unsigned ac4in_crc16(const uint8_t *d, size_t n, unsigned crc);

#ifdef __cplusplus
}
#endif
#endif
