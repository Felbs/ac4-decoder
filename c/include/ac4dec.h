/* SPDX-License-Identifier: Apache-2.0 */
/*
 * libac4dec -- an AC-4 audio decoder (ETSI TS 103 190-1/-2) in C99.
 *
 * A port of the Python decoder in ../ac4 (the reference), milestone 1: the
 * ASF/MDCT core. Feed it one raw AC-4 frame (an access unit, e.g. one MP4
 * sample) at a time; it returns double-precision PCM per channel.
 *
 * Milestone 1 does NOT render A-SPX or A-CPL: streams that use them decode
 * to their core band (what the Python decoder gives with A-SPX rendering
 * disabled). What milestone 1 leaves out is listed in src/decoder.c.
 *
 * Damaged input never crashes the decoder and never changes the output
 * length: every frame gives exactly one frame of samples (1536 at the
 * internal rate, or the matching 1601/1602 at 48 kHz).
 *
 * Thread safety: a decoder instance must not be used from two threads at
 * once; separate instances are independent.
 */
#ifndef AC4DEC_H
#define AC4DEC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AC4DEC_VERSION "0.1.0-m1"

/* Samples per frame at the internal rate (29.97 fps, 48 kHz family). */
#define AC4DEC_FRAME 1536
/* Most samples one frame can give at 48 kHz (1536 * 1001 / 960 = 1601.6). */
#define AC4DEC_MAX_OUT 1602
#define AC4DEC_MAX_CHANNELS 6

typedef enum {
    AC4DEC_ELEMENT_AUTO = 0,   /* from the first frame whose TOC parses */
    AC4DEC_ELEMENT_5X = 1,     /* 5_X channel element (5.1) */
    AC4DEC_ELEMENT_PAIR = 2    /* channel_pair_element (stereo) */
} ac4dec_element;

typedef enum {
    AC4DEC_OK = 0,         /* decoded */
    AC4DEC_CONCEALED = 1,  /* broke a bitstream rule: spectrum zeroed, the
                              previous frame's overlap fades out */
    AC4DEC_BAD = 2         /* could not be decoded and nothing to conceal
                              from (element not known yet, bad argument):
                              silence */
} ac4dec_status;

typedef struct {
    ac4dec_element element;
    int channels;          /* 0 = default (6 for 5_X, 2 for a pair); 2 or 6.
                              A pair element always gives 2. */
    int internal_rate;     /* 0 (default) = 48 kHz output through the
                              1001/960 resampler (1601 or 1602 samples per
                              frame); 1 = the internal rate, 1536/frame */
} ac4dec_config;

typedef struct {
    ac4dec_status status;
    const char *reason;    /* short reason code when not OK, else NULL */
    int n_channels;        /* channels in pcm[] */
    int n_samples;         /* samples per channel in this frame */
    const double *pcm[AC4DEC_MAX_CHANNELS];  /* planar, decoder scale;
                              valid until the next call on this decoder */
    int sequence_counter;  /* -1 if the TOC did not parse */
    int iframe;            /* b_iframe_global */
    int seq_jump;          /* the counter broke sequence (lost frames) */
} ac4dec_frame;

typedef struct ac4dec ac4dec;

/* Channel order: 6 channels = L R C LFE Ls Rs (WAV 5.1 order); 2 = L R. */

/* -> a decoder, or NULL (out of memory or a bad config). cfg NULL = auto
   element, default channels, 48 kHz output. */
ac4dec *ac4dec_open(const ac4dec_config *cfg);
void ac4dec_close(ac4dec *d);

/* Decode one frame. Always fills *out (never fails on bad data). Returns
   the status. data may be NULL when len is 0 (a lost frame). */
int ac4dec_decode_frame(ac4dec *d, const uint8_t *data, size_t len,
                        ac4dec_frame *out);

/* At 48 kHz the resampler holds no samples back: nothing to flush. */

/* The element of a stream from one frame's TOC: AC4DEC_ELEMENT_5X,
   AC4DEC_ELEMENT_PAIR, or AC4DEC_ELEMENT_AUTO when unknown. */
ac4dec_element ac4dec_probe(const uint8_t *data, size_t len);

/* The decoder's fixed per-element output reference (decoder units; the
   Python CLI writes it at 0.9 of full scale): 9e5 for 5_X, 2e5 for a pair. */
double ac4dec_ref_peak(ac4dec_element e);

typedef struct {
    long n_frames, n_ok, n_concealed, n_bad, n_seq_jumps;
    int element;           /* ac4dec_element once known */
    int acpl3;             /* the stream is ASPX_ACPL_3 (2-ch core) */
} ac4dec_stats;

void ac4dec_get_stats(const ac4dec *d, ac4dec_stats *s);

/* Concealment reasons seen so far: call with i = 0, 1, ... until it
   returns NULL. *count receives how many frames had that reason. */
const char *ac4dec_reason(const ac4dec *d, int i, long *count);

/* The 1001/960 resampler on its own (TS 103 190-1 Table 83), for callers
   that take the internal rate (internal_rate = 1) and resample later; the
   same filter as the decoder's 48 kHz output. Causal: after n inputs in
   total, exactly ceil(n * 1001 / 960) outputs have been given. */
typedef struct ac4dec_resampler ac4dec_resampler;
ac4dec_resampler *ac4dec_resampler_open(int channels);
void ac4dec_resampler_close(ac4dec_resampler *r);
/* in[c]: n samples per channel; out[c]: room for n * 1001 / 960 + 2.
   -> samples written per channel. */
int ac4dec_resampler_push(ac4dec_resampler *r, const double *const *in,
                          int n, double *const *out);

#ifdef __cplusplus
}
#endif
#endif
