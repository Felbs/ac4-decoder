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
 * length: every frame gives exactly one frame of samples: frame_len_base
 * at the internal rate (TS 103 190-1 Table 83: 2048 at 25 fps, 1536 at
 * 29.97 fps, ...; 1536 for frames before the stream's rate is known), or
 * the matching count at the external rate through the table's resampler
 * (1601/1602 at 29.97 fps, 1920 at 25 fps).
 *
 * RATES. Every frame_rate_index of Table 83 (48 kHz) and index 13 of
 * Table 84 (44,1 kHz) is decoded. The stream's rate is that of its first
 * decodable frame whose next frame agrees; a frame at another rate is
 * concealed (toc.rate), as are frames with frame_rate_factor or
 * frame_rate_fraction above 1 and 96/192 kHz substreams
 * (toc.unsupported_rate).
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

#define AC4DEC_VERSION "0.3.0-b"

/* Samples per frame at the internal rate at 29.97 fps, 48 kHz (the frame
   length of frames before a stream's rate is known). */
#define AC4DEC_FRAME 1536
/* The longest internal frame (frame_len_base 2048: 25 fps, index 13). */
#define AC4DEC_MAX_FRAME 2048
/* Most samples one frame can give at the external rate (2048 at index 13;
   1536 * 1001 / 960 = 1601.6 at 29.97 fps). */
#define AC4DEC_MAX_OUT 2050
#define AC4DEC_MAX_CHANNELS 8

#define AC4DEC_MIN_LOOKAHEAD 8
#define AC4DEC_DEFAULT_LOOKAHEAD 180

typedef enum {
    AC4DEC_ELEMENT_AUTO = 0,   /* from the first frame whose TOC parses */
    AC4DEC_ELEMENT_5X = 1,     /* 5_X channel element (5.1) */
    AC4DEC_ELEMENT_PAIR = 2,   /* channel_pair_element (stereo) */
    AC4DEC_ELEMENT_MONO = 3,   /* single_channel_element (C) */
    AC4DEC_ELEMENT_3_0 = 4,    /* 3_0_channel_element (L R C) */
    AC4DEC_ELEMENT_5_0 = 5,    /* 5_X without LFE (L R C Ls Rs) */
    AC4DEC_ELEMENT_7X = 6      /* 7_X_channel_element (config ch_mode) */
} ac4dec_element;

typedef enum {
    AC4DEC_OK = 0,         /* decoded */
    AC4DEC_CONCEALED = 1,  /* broke a bitstream rule (spectrum zeroed, the
                              previous frame's overlap fades out, A-SPX
                              high band muted), or a level check fired
                              (reason "guard.level" or "guard.truepk":
                              muted with 256-sample ramps; "aspx.level":
                              plays its core band) */
    AC4DEC_BAD = 2         /* could not be decoded and nothing to conceal
                              from (element not known yet): silence */
} ac4dec_status;

typedef struct {
    ac4dec_element element;
    int channels;          /* 0 = default: every channel of the element
                              (6 for 5_X, 2 for a pair, 1 mono, 3 for 3.0,
                              5 for 5.0, 7 / 8 for 7.0 / 7.1); 2 = L, R
                              only (elements with L and R) */
    int internal_rate;     /* 0 (default) = the external rate (48 kHz, or
                              44,1 kHz for fs_index 0) through the stream's
                              resampler (Table 83); 1 = the internal rate,
                              frame_len_base samples per frame */
    int lookahead;         /* frames of delay, 0 = AC4DEC_DEFAULT_LOOKAHEAD;
                              at least AC4DEC_MIN_LOOKAHEAD (see above) */
    int no_guard;          /* 1 = output guard off (Python guard=False;
                              non-finite samples are then zeroed) */
    int presentation;      /* 0 = automatic (see below); n > 0 = the n-th
                              presentation of the TOC (index n - 1) */
    long presentation_id;  /* n > 0: the presentation whose
                              presentation_id is n - 1 (if presentation
                              is 0) */
    char language[16];     /* the automatic choice prefers this language
                              (BCP 47, e.g. "fr"; "" = none) */
    int loudness;          /* 1 = loudness normalisation and DRC (TS 103
                              190-1 5.7.9; ac4/loudness.py): pcm[] is then
                              in units of full scale (1.0), dialnorm
                              normalised to target_level; 0 = decoder
                              scale, untouched */
    double target_level;   /* dB, e.g. -24 */
    int drc;               /* 0 = the stream's DRC mode for target_level,
                              -1 = off, n > 0 = DRC mode ID n - 1 */
    int ch_mode;           /* 7_X: the channel mode (Table 88 ch_mode 5 -
                              10); 0 = from the stream (AUTO) or 6 (7.1
                              3/4/0) */
    double de_level;       /* dialogue enhancement gain in dB (TS 103
                              190-1 5.7.8, de_method 0; capped by the
                              stream's maximum); 0 = off */
    int substream;         /* which channel-coded substream of the
                              presentation (0 = the main one); n > 0
                              decodes only that one, unmixed */
    int no_mix;            /* 1 = the main substream only (see MIXING) */
    double assoc_gain;     /* associated audio gain g_assoc, dB <= 0 */
} ac4dec_config;

/* MIXING (TS 103 190-2 4.8.3.17, 4.8.3.18, 4.8.4; ac4/mix.py). When the
   chosen presentation (decided at the first frame whose TOC parses) has
   several channel-coded substreams -- music and effects + dialogue, main
   + associated audio -- and substream is 0 and no_mix 0, the decoder
   decodes each with an internal decoder of its own and gives their mix
   in the main substream's channel layout: substream group gains, the
   associated audio's scale_main / _centre / _front and pan_associated,
   assoc_gain, and the sum divided by the number of substreams. A frame
   is concealed when any substream's frame is. Unverified on real
   streams. */
#define AC4_MIX_MAX 4

/* PRESENTATIONS (TS 103 190-2 4.8.2). The automatic choice is, per frame,
   the first enabled presentation that has a channel-coded audio substream,
   preferring the configured language, then "complete main" content (not
   audio description, commentary, ...). Of that presentation the first
   channel-coded substream is decoded (several substream groups are not
   mixed). */
typedef struct {
    int index;             /* position in the TOC (0-based) */
    long presentation_id;  /* -1 when not signalled */
    int config;            /* presentation_config, -1 = single group */
    int enabled;           /* b_enable_presentation (1 when not sent) */
    int md_compat, version;
    char language[16];     /* first language of its groups, "" if none */
    const char *content;   /* first content_classifier name, "" if none */
    int channel_mode;      /* ch_mode of the decoded substream, -1 if none */
    int decodable;         /* has a channel-coded audio substream */
    int object_coded;      /* references object / A-JOC substreams */
    int is_default;        /* the automatic choice */
} ac4dec_presentation;

/* The presentations of one frame (one TOC): fills out[0..max) and returns
   how many the TOC has (may exceed max), or -1 when the TOC does not
   parse. */
int ac4dec_presentations(const uint8_t *data, size_t len,
                         ac4dec_presentation *out, int max);

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
    int sample_rate;       /* of pcm[], rounded (the internal rate is the
                              external one times down / up, e.g. 46034 Hz
                              at 29.97 fps) */
} ac4dec_frame;

typedef struct ac4dec ac4dec;

/* Channel order: 6 channels = L R C LFE Ls Rs (WAV 5.1 order); 2 = L R;
   mono C; 3.0 L R C; 5.0 L R C Ls Rs; 7.x L R C [LFE] Ls Rs and the
   layout's pair (Lb Rb, Lw Rw or Tfl Tfr). Every channel is 577 samples
   late (the QMF bank, or a delay line of the same length). */

/* -> a decoder, or NULL (out of memory or a bad config). cfg NULL = auto
   element, default channels, 48 kHz output, default lookahead, guard on. */
ac4dec *ac4dec_open(const ac4dec_config *cfg);
void ac4dec_close(ac4dec *d);

/* The channels the decoder gives out (0 while the element is not known:
   AUTO before the first frame whose TOC parses). */
int ac4dec_channels(const ac4dec *d);

/* The 7_X channel mode (Table 88 ch_mode 5 - 10) of a frame's chosen
   audio substream, or 0. */
int ac4dec_probe_ch_mode(const ac4dec_config *cfg, const uint8_t *data,
                         size_t len);

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
/* The same for the presentation cfg chooses (cfg->presentation,
   presentation_id, language; NULL = automatic). */
ac4dec_element ac4dec_probe_config(const ac4dec_config *cfg,
                                   const uint8_t *data, size_t len);

/* The rate signalling of one frame's TOC: -> 0 and fills fs_index and
   frame_rate_index, or -1 when the TOC does not parse. */
int ac4dec_probe_rate(const uint8_t *data, size_t len, int *fs_index,
                      int *frame_rate_index);
/* TS 103 190-1 Tables 83, 84 for a rate: -> 0 and fills frame_len
   (frame_len_base), up / down (the decoder resampling ratio), out_rate
   (48000 or 44100) and fps (frames per second); -1 when the tables
   define none (reserved index, 44,1 kHz other than index 13). */
int ac4dec_rate_info(int fs_index, int frame_rate_index, int *frame_len,
                     int *up, int *down, int *out_rate, double *fps);

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
    long n_guard;          /* of those, guard.level and guard.truepk */
    long n_aspx_level;     /* of those, aspx.level */
    long n_late;           /* of those, audio_size.late */
    long n_burst;          /* A-SPX frames that played their core band
                              (burst guard, aspx.level, and the 2 frames
                              after each: Decoder.n_burst) */
    long n_seq_jumps;
    int element;           /* ac4dec_element once known */
    int acpl3;             /* the stream is ASPX_ACPL_3 (2-ch core + A-CPL) */
    int frame_len;         /* frame_len_base, 0 while the rate is unknown */
    int out_rate;          /* the external rate (0 while unknown) */
    int aspx;              /* A-SPX is being rendered (a pair is on the QMF
                              path and an aspx_config is known) */
} ac4dec_stats;

void ac4dec_get_stats(const ac4dec *d, ac4dec_stats *s);

/* Reasons of the frames given out so far that were not OK: call with
   i = 0, 1, ... until it returns NULL. *count receives how many. */
const char *ac4dec_reason(const ac4dec *d, int i, long *count);

/* The output resampler on its own (TS 103 190-1 Table 83), for callers
   that take the internal rate (internal_rate = 1) and resample later; the
   same filter as the decoder's external-rate output. Causal: after n
   inputs in total, exactly ceil(n * up / down) outputs have been given
   (ratio 1: the samples pass through). _open is 1001 / 960 (29.97 fps). */
typedef struct ac4dec_resampler ac4dec_resampler;
ac4dec_resampler *ac4dec_resampler_open(int channels);
ac4dec_resampler *ac4dec_resampler_open_ratio(int channels, int up,
                                              int down);
void ac4dec_resampler_close(ac4dec_resampler *r);
/* in[c]: n samples per channel; out[c]: room for n * up / down + 2.
   -> samples written per channel. */
int ac4dec_resampler_push(ac4dec_resampler *r, const double *const *in,
                          int n, double *const *out);

#ifdef __cplusplus
}
#endif
#endif
