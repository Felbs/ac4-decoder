/* SPDX-License-Identifier: Apache-2.0 */
/*
 * libac4dec -- an AC-4 audio decoder (ETSI TS 103 190-1/-2) in C99.
 *
 * A port of the Python decoder in ../ac4 (the reference): the ASF/MDCT
 * core, A-SPX (rendered on the L/R pair of a stereo element in
 * stereo_codec_mode 1 and on the L/R and Ls/Rs pairs of a 5_X element in
 * codec mode 1, joint balance included), A-CPL (the 5.1 upmix of an
 * ASPX_ACPL_3 stream: L, R, C, Ls, Rs from the two-channel core in the QMF
 * domain, the LFE from its own data; with channels = 2 such a stream gives
 * its stereo core), the 1001/960
 * resampler, and every damaged-input rule of ac4/robust.py including the
 * output guard. What it leaves out is the Python's list (README).
 *
 * Feed it one raw AC-4 frame (an access unit, e.g. one MP4 sample) at a
 * time; it returns double-precision PCM per channel.
 *
 * DELAY. Several rules look ahead (the sequence-outlier rule and the
 * TOC-change rule at the next frame, an A-SPX frame's envelopes into the
 * next frame, the output guard 6 frames ahead and its seed at the first
 * 30, the late audio_size check back from the first I-frame), so the
 * decoder is a delay line: the frame that ac4dec_decode_frame gives out is
 * the one fed `lookahead` calls earlier (ac4dec_delay), and the first
 * `lookahead` calls give none. At the end of the stream, call
 * ac4dec_flush until it returns 0 to get the frames still inside. Every
 * frame fed comes out exactly once, in order, with its own index.
 *
 * Of the lookahead N (>= AC4DEC_MIN_LOOKAHEAD = 8): 2 frames are the
 * parse and A-SPX stages, G = min(N - 2, 30) frames the guard's, and the
 * rest, L = N - 2 - G, hold parsed frames before synthesis. With N >= 32
 * the guard seeds as the Python (from the first 30 frames); below that it
 * seeds from the first N - 1. A frame from before the stream's first
 * aspx_config can be concealed by the late audio_size check only while it
 * is held, i.e. when the I-frame that brings the configuration arrives
 * within L + 1 frames of it (the Python checks a whole decode pass).
 * Every channel comes out 577 samples after the MDCT core (TS 103 190-1
 * 6.2.8 / 6.2.14 put each channel through the QMF bank): the pairs that
 * can carry A-SPX go through the QMF bank from their first frame (HF
 * generation and adjustment only where a frame has A-SPX data), the
 * others (centre, LFE, a pair without A-SPX) through a delay line of the
 * same length. The default, AC4DEC_DEFAULT_LOOKAHEAD = 180
 * (6 s), covers the I-frame intervals measured on air (up to 144 frames);
 * a live player may take the minimum.
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

#define AC4DEC_VERSION "0.2.0-m2"

/* Samples per frame at the internal rate (29.97 fps, 48 kHz family). */
#define AC4DEC_FRAME 1536
/* Most samples one frame can give at 48 kHz (1536 * 1001 / 960 = 1601.6). */
#define AC4DEC_MAX_OUT 1602
#define AC4DEC_MAX_CHANNELS 6

#define AC4DEC_MIN_LOOKAHEAD 8
#define AC4DEC_DEFAULT_LOOKAHEAD 180

typedef enum {
    AC4DEC_ELEMENT_AUTO = 0,   /* from the first frame whose TOC parses */
    AC4DEC_ELEMENT_5X = 1,     /* 5_X channel element (5.1) */
    AC4DEC_ELEMENT_PAIR = 2    /* channel_pair_element (stereo) */
} ac4dec_element;

typedef enum {
    AC4DEC_OK = 0,         /* decoded */
    AC4DEC_CONCEALED = 1,  /* broke a bitstream rule (spectrum zeroed, the
                              previous frame's overlap fades out, A-SPX
                              high band muted), or a level check fired
                              (reason "guard.level": muted with 256-sample
                              ramps; "aspx.level": plays its core band) */
    AC4DEC_BAD = 2         /* could not be decoded and nothing to conceal
                              from (element not known yet): silence */
} ac4dec_status;

typedef struct {
    ac4dec_element element;
    int channels;          /* 0 = default (6 for 5_X, 2 for a pair); 2 or 6.
                              A pair element always gives 2. */
    int internal_rate;     /* 0 (default) = 48 kHz output through the
                              1001/960 resampler (1601 or 1602 samples per
                              frame); 1 = the internal rate, 1536/frame */
    int lookahead;         /* frames of delay, 0 = AC4DEC_DEFAULT_LOOKAHEAD;
                              at least AC4DEC_MIN_LOOKAHEAD (see above) */
    int no_guard;          /* 1 = output guard off (Python guard=False;
                              non-finite samples are then zeroed) */
} ac4dec_config;

typedef struct {
    ac4dec_status status;
    const char *reason;    /* short reason code when not OK, else NULL
                              (the codes of ac4/robust.py) */
    long long index;       /* which input frame this is (0, 1, ...) */
    int n_channels;        /* channels in pcm[] */
    int n_samples;         /* samples per channel in this frame */
    const double *pcm[AC4DEC_MAX_CHANNELS];  /* planar, decoder scale;
                              valid until the next call on this decoder */
    int sequence_counter;  /* -1 if the TOC did not parse */
    int iframe;            /* b_iframe_global */
    int seq_jump;          /* the counter broke sequence (lost frames) */
} ac4dec_frame;

typedef struct ac4dec ac4dec;

/* Channel order: 6 channels = L R C LFE Ls Rs (WAV 5.1 order); 2 = L R.
   As in the Python, the A-SPX pairs (L R, Ls Rs) are delayed 577 samples
   by the QMF bank and C and LFE are not. */

/* -> a decoder, or NULL (out of memory or a bad config). cfg NULL = auto
   element, default channels, 48 kHz output, default lookahead, guard on. */
ac4dec *ac4dec_open(const ac4dec_config *cfg);
void ac4dec_close(ac4dec *d);

/* The decoder's delay in frames (the lookahead). */
int ac4dec_delay(const ac4dec *d);

/* Feed one frame (data may be NULL when len is 0: a lost frame). Never
   fails on bad data. -> 1 when *out holds the output frame of the frame
   fed ac4dec_delay() calls earlier, 0 while the delay line fills. */
int ac4dec_decode_frame(ac4dec *d, const uint8_t *data, size_t len,
                        ac4dec_frame *out);

/* End of stream: -> 1 with the next frame still inside the decoder in
   *out, 0 when none is left. Call until it returns 0. A frame fed after
   that starts a new stream (all stream state reset; stats kept). */
int ac4dec_flush(ac4dec *d, ac4dec_frame *out);

/* The element of a stream from one frame's TOC: AC4DEC_ELEMENT_5X,
   AC4DEC_ELEMENT_PAIR, or AC4DEC_ELEMENT_AUTO when unknown. */
ac4dec_element ac4dec_probe(const uint8_t *data, size_t len);

/* The decoder's fixed per-element output reference (decoder units; the
   Python CLI writes it at 0.9 of full scale): 9e5 for 5_X, 2e5 for a pair. */
double ac4dec_ref_peak(ac4dec_element e);

typedef struct {
    long n_frames;         /* frames fed */
    long n_out;            /* frames given out */
    long n_ok;
    long n_concealed;      /* frames given out not OK (all reasons; the
                              Python's Decoder.n_concealed) */
    long n_bad;            /* of those, by a bitstream rule or element
                              unknown (the Python's Decoder.n_bad) */
    long n_guard;          /* of those, guard.level */
    long n_aspx_level;     /* of those, aspx.level */
    long n_late;           /* of those, audio_size.late */
    long n_burst;          /* A-SPX frames that played their core band
                              (burst guard, aspx.level, and the 2 frames
                              after each: Decoder.n_burst) */
    long n_seq_jumps;
    int element;           /* ac4dec_element once known */
    int acpl3;             /* the stream is ASPX_ACPL_3 (2-ch core + A-CPL) */
    int aspx;              /* A-SPX is being rendered (a pair is on the QMF
                              path and an aspx_config is known) */
} ac4dec_stats;

void ac4dec_get_stats(const ac4dec *d, ac4dec_stats *s);

/* Reasons of the frames given out so far that were not OK: call with
   i = 0, 1, ... until it returns NULL. *count receives how many. */
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
