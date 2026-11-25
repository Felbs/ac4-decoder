# Comparison with the FFmpeg AC-4 patch

Measured 2026-11-25. This page compares this decoder with the only other
public AC-4 decoder we know of: the FFmpeg patch by Paul B Mahol (2019), as
carried for FFmpeg 6.1 in
[funnymanva/ffmpeg-with-ac4](https://github.com/funnymanva/ffmpeg-with-ac4)
(`ffmpeg_ac4.patch`, commit 674c4ef). It is called "the patch" below.

The patch was the first public AC-4 decoder. Its decoder dates from 2019
and is not part of FFmpeg itself.

## Short answer

* **Stereo:** both decode it. After correcting for the patch's speed error
  (below), the two outputs agree closely (correlation 0.96 to 1.00 below
  5 kHz, same bandwidth).
* **5.1:** the patch does not deliver a correct 5.1 signal on any 5.1
  stream we have. Depending on how the stream is coded, the surrounds are
  silent, or the channels come out in the wrong order; in every case the
  upper frequencies are missing.
* **Immersive (5.1.4) and immersive stereo (IMS):** the patch fails or
  crashes; this decoder decodes them.
* **Speed:** every stream we have (25 fps and 29.97 fps, the latter being
  what US broadcasts use) comes out of the patch at the wrong speed and
  pitch.

## What was compared

39 streams, each decoded by both programs with default settings:

| Set | Streams |
|---|---|
| Dolby AC-4 Online Delivery Kit 1.4.0 and 1.5 (public test signals), each at 25 and 29.97 fps | 34: 10 stereo, 16 5.1 at 128 kbit/s, 2 5.1 at 192 kbit/s, 2 5.1.4, 4 IMS |
| Off-air ATSC 3.0 captures, 10 s each, three broadcasters (A, B, C) | 5: A 5.1, A stereo, B 5.1, B stereo, C stereo |

* This decoder: the C command-line decoder (`c/ac4dec-c`), commit ae35398.
* The patch: FFmpeg 6.1 with `ffmpeg_ac4.patch` applied, a minimal build
  (`--disable-everything` plus the AC-4, PCM and E-AC-3 decoders), output
  written as 32-bit float WAV.

The kit signals and the captures are not in this repository.

## Results by kind of stream

| Kind of stream (coding mode) | Streams | The patch | This decoder |
|---|---|---|---|
| Stereo (`channel_pair_element`, SIMPLE or A-SPX) | 13 | decodes; agrees with this decoder once the speed is corrected | decodes |
| 5.1, A-SPX + A-CPL (`ASPX_ACPL_2`): every 128 kbit/s 5.1 kit stream | 16 | Ls and Rs are silent and their sound stays in L and R; the LFE is written to the right-surround position; nothing above 13 kHz | six channels, each sound in its own channel; up to 20 kHz |
| 5.1, A-SPX + A-CPL (`ASPX_ACPL_3`): off-air B | 1 | two front channels only, one of them 17 dB below the other; centre and surrounds silent; nothing above 12 kHz | six channels, up to 18 kHz |
| 5.1, A-SPX (`ASPX`): 192 kbit/s kit streams, off-air A | 3 | all the sound is there, in the wrong positions: centre and LFE are written where the surrounds belong and the surrounds where centre and LFE belong; nothing above 12 kHz on the kit streams | six channels in the standard order; up to 20 kHz on the kit streams |
| 5.1.4 (`immersive_channel_element`) | 2 | no output: decoding stops with errors | ten channels (5.1.4) |
| IMS (immersive stereo) | 4 | three crash (segmentation fault); the fourth writes 4 s of 8-channel noise with sample values up to 2e10 | stereo |

## The five differences, with the measurements

### 1. Speed and pitch

AC-4 codes each frame at a base length (2048 or 1536 samples here) and the
decoder resamples it to the frame's real duration. The
patch leaves this step out: its `nb_samples` is the base length, with the
resampling commented out in the source.

| Stream frame rate | True duration | Patch output | Error |
|---|---|---|---|
| 25 fps (kit, 32.00 s item) | 32.00 s | 34.13 s | 6.7 % slow, pitch down about a semitone |
| 29.97 fps (kit, 32.03 s item) | 32.03 s | 30.72 s | 4.1 % fast, pitch up |
| 29.97 fps (off-air, 300 frames) | 10.01 s | 9.60 s | 4.1 % fast |

A player that takes its timing from the container will hide the length
error by dropping or stretching, but not the pitch.

### 2. Surround channels with A-CPL

Most 5.1 at broadcast bit rates is coded as two or three channels plus
A-CPL parameters, from which the decoder rebuilds the surrounds. The patch
reads the A-CPL parameters and does not use them.

Measured on the kit's channel-identification item (a voice names each
speaker in turn): with this decoder each of the six channels is the loudest
one in turn (L, R, C, LFE, Ls, Rs for 10, 13, 10, 2, 18 and 12 seconds).
With the patch, Ls and Rs are never active; their announcements come from L
and R, which are the loudest channels for 21 and 25 seconds.

### 3. Channel order

The patch stores channels in the order the stream codes them and labels the
result with a fixed layout. The coded order depends on the coding
configuration, so the mapping changes from stream to stream. The same kit
programme at 128 and at 192 kbit/s comes out of the patch with its loudest
channel (the centre) in position 3 in one and position 5 in the other; on
off-air stream A the dialogue channel is written to the left-surround
position. The LFE is in the last position in every 5.1 output, where the
declared layout has the right surround.

This decoder's order was checked against an independent reference: the kit
carries one 5.1 programme also as E-AC-3, and each channel of our output
matches the same channel of the E-AC-3 decode (correlation below 5 kHz:
L 0.92, R 0.91, C 0.98, LFE 0.99, Ls 0.89, Rs 0.86).

### 4. High frequencies in 5.1

A-SPX rebuilds the upper part of the spectrum. The patch applies it for
mono and stereo only; for 5.1 the A-SPX data is read and not used. Highest
frequency with content within 60 dB of the loudest band, front channels:

| Stream | The patch | This decoder |
|---|---|---|
| Kit 5.1 film item, 128 kbit/s | 13 kHz | 20 kHz |
| Kit 5.1 film item, 192 kbit/s | 12 kHz | 20 kHz |
| Off-air B 5.1 | 12 kHz (one channel 10 kHz) | 18 kHz |
| Off-air A 5.1 | 12 kHz | 13 kHz |
| Stereo (all) | same as this decoder, within 1 to 2 kHz | |

### 5. What is decoded at all

The patch decodes one substream of the selected presentation.
It has no immersive channel element, no object or A-JOC decoding, no IMS
handling, and no rendering to other speaker layouts. Loudness, dynamic
range control and dialogue enhancement data are parsed and not applied.
See [What it decodes](../README.md#what-it-decodes) for this decoder.

## Where the two agree

Where the patch decodes a stream, its core (the MDCT part below the A-SPX
crossover) is right. After resampling the patch's output to the true speed:

* stereo streams: correlation with this decoder below 5 kHz is 0.96 to
  1.00 per channel (kit and off-air);
* off-air 5.1 streams: the sum of all channels correlates 0.98 to 1.00 with
  ours in every half second.

The patch's output has inverted polarity relative to this decoder and to
the E-AC-3 reference. On its own that is inaudible.

## What was not compared

* **Speed.** The patch was built without assembly (no assembler was
  installed), so timing it would be unfair. Not measured.
* **Damaged streams.** Only clean streams were used.
* **Listening tests.** Everything above is measurement, not a blind test.
* **Waveform accuracy of the surrounds.** On the kit's E-AC-3 comparison
  the patch's L, R and C correlate with the reference slightly better than
  ours below 5 kHz (0.94, 0.94, 0.995 against 0.92, 0.91, 0.98). On that
  item the surrounds carry nearly the same signal as the fronts, so a
  decoder that leaves the surrounds mixed into the fronts scores well on
  this measure; A-CPL rebuilds the surrounds parametrically and is not
  waveform exact. The channel-identification item (difference 2) is the
  better test of whether 5.1 is delivered.
* At the start of a stream the patch gives nothing until the first frame
  that carries a configuration (4.0 s on off-air B 5.1). That is a
  reasonable choice and is not counted as a difference.

## Method

* Durations and channel counts from the WAV files.
* Bandwidth: the highest frequency at which the long-term spectrum (Welch,
  4096 points) is within 60 dB of that channel's loudest band.
* Agreement: the patch's output is first resampled by 15/16 (25 fps
  streams) or 1001/960 (29.97 fps streams). Then, for each pair of
  channels, the normalised cross-correlation of the two signals low-passed
  at 5 kHz, at the best lag within 1.5 s. Channel matches are the pairs
  with the highest correlation.
* Channel activity on the identification item: the loudest channel in each
  one-second window above -50 dBFS.
