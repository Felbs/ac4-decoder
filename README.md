# ac4-decoder

An AC-4 audio decoder written from the published standard (ETSI TS 103 190-1
and -2), in Python and NumPy, with two optional C kernels for speed, and a
port of the whole decoder to C (`c/`, see [The C decoder](#the-c-decoder)).

It decodes the AC-4 that real ATSC 3.0 broadcasts carry. It is **not** a
complete AC-4 decoder, it is not certified, and it is not affiliated with or
endorsed by Dolby Laboratories (see [NOTICE](NOTICE)).

```
pip install -e .                 # numpy + scipy
python -m ac4.build_kernels      # optional: build the C kernels (faster)
ac4dec audio.m4s out.wav         # or: python -m ac4 audio.m4s out.wav
```

```python
from ac4 import decode, mp4
pcm, dec = decode(mp4.read_frames("audio.m4s"))   # float64 (n, ch), 48 kHz
print(dec.chans, dec.n_ok, dec.n_bad)             # decoder scale, not +-1.0
print(dec.n_concealed, dict(dec.concealed.reasons))   # damaged frames
```

## What it decodes

Everything below was measured on off-air captures from two ATSC 3.0
broadcasters using different encoder configurations. No captures are in this
repository.

| | |
|---|---|
| Bitstream | version 2, 48 kHz, 29.97 fps, one presentation, channel-coded |
| Input | fragmented MP4 (init segment + `moof`/`mdat` fragments, or bare fragments) |
| 5.1 (`5_X` channel element, coding_config 0) | codec mode 1 (A-SPX): all six channels. Codec mode 4 (ASPX_ACPL_3): the two-channel core only (see below) |
| Stereo (`channel_pair_element`) | stereo_codec_mode 0 (SIMPLE) and 1 (A-SPX) |
| Core (ASF/MDCT) | Huffman spectral data, scale factors, long and short frames with block switching, KBD windows, stereo processing (sap_mode 1 and 2: M/S; sap_mode 3: full SAP with prediction coefficients, see below), spectral noise fill (clause 5.1.4, the standard's own noise generator; no capture here signals it), LFE |
| A-SPX | parsed and rendered on the L/R and Ls/Rs pairs: 64-band complex QMF, envelope decoding, HF generation, envelope adjustment with the limiter, noise, joint-balance rendering |
| Output | 48 kHz via the 1001/960 resampler (TS 103 190-1 Table 83); without it audio runs 4.27 % fast. Every channel is 577 samples (12 ms) after the MDCT core: the A-SPX pairs through the QMF bank, the others (centre, LFE, a pair without A-SPX) through a delay line of the same length, so all channels are time-aligned |
| I-frames | decoding can start mid-stream; A-SPX starts at the first I-frame |
| Damaged input | never raises, never changes the output length; damaged frames are concealed (see below) |

This package started as a byte-exact copy of the original receiver's
decoder (see Provenance). It now differs from it on purpose: full SAP
frames (which the receiver passed through undecoded), an A-SPX parsing fix
(see Damaged input), and the fixes listed under
[Changes against the standard](#changes-against-the-standard).

**Speed** (measured 2026-11-16 with `ac4dec`, C kernels built, one thread,
`nice 19`, on a desktop CPU shared with other jobs, so expect +-20 %
between runs; the figure is the better of two runs, decode time only):

| stream | length | real-time factor |
|---|---|---|
| 5.1, codec mode 1 (A-SPX), broadcaster A | 178 s | 1.8x |
| stereo, broadcaster A | 178 s | 4.3x |
| 5.1 sent as ASPX_ACPL_3 (stereo core decoded), broadcaster B | 42 s | 3.4x |
| stereo, broadcaster B | 42 s | 4.4x |

The damaged-input checks and full SAP cost 1 to 19 % against the version
before them on the same runs (1.95x / 4.87x / 4.22x / 5.01x then); the
spread between runs is of the same size, so the cost is not pinned down.

**Bandwidth** (same decodes; the highest frequency at which a channel's
long-term average spectrum, Welch with 8192 points over the whole file, is
within 50 dB of its maximum): broadcaster A 5.1: L 16.9, R 16.5, C 11.5,
Ls 16.9, Rs 11.5 kHz (LFE 0.2 kHz); broadcaster A stereo: 12.8 kHz;
broadcaster B 5.1 (its stereo core): 18.0 kHz; broadcaster B stereo:
16.5 kHz. The centre has no A-SPX rendering (see below), so it stops where
the core does. These figures describe what the broadcasters send as much
as the decoder.

### Full SAP (sap_mode 3)

Stereo audio processing mode 3 sends, per pair of scale factor bands, a
prediction coefficient alpha (DPCM-coded in frequency, or in time across
short-block groups) and decodes the two tracks as
`L = (1 + alpha) M + S`, `R = (1 - alpha) M - S` (TS 103 190-1 Table 48,
Pseudocode 59, clause 5.3.3.2). It is common on air: 26 % of one
broadcaster's 5.1 frames (42 s measured) and 34 % of the other's (18
minutes measured) use it on at least one channel pair; until this was
added, those frames were decoded wrongly on both broadcasters. How it was checked, on captures:

* **Bit accounting.** Every SAP3 frame walked to the end of `audio_data()`
  (A-SPX and A-CPL side data included) stops 0 to 7 bits short of
  `audio_size`, i.e. exactly at the byte alignment: 300 of 300 on one
  broadcaster, 10 984 of 10 986 in the 18 minutes of the other (the two
  others, and 10 non-SAP3 frames, look like reception damage). A parse that
  reads one coefficient per band instead of per band pair fails this.
* **Against an independent reference.** The A-SPX envelopes carry each
  channel's high-band energy as the encoder measured it on the original.
  On SAP3 frames, the decoded L/R level difference just below the A-SPX
  crossover follows the envelopes' L/R difference just above it
  (correlation 0.61 and 0.80 on the two broadcasters' front pairs). With
  alpha negated the correlation turns negative (-0.62, -0.79); with alpha
  ignored it vanishes (-0.04, 0.13); with the matrix transposed the sign is
  right but the image is about half as wide (scale 0.35 and 0.31 against
  0.74 and 0.61); passing the tracks through (the old behaviour) puts L/R
  steps at every switch, 9 to 22 times the ordinary frame-to-frame change.
* **Nothing else changes.** Streams without SAP3 frames decode
  byte-identically to before; on streams with them, only SAP3 channel
  pairs change.

Two things this does not settle. The decoded image is not continuous with
the neighbouring M/S frames in a simple before/after comparison, and is not
meant to be: those frames often send an all-zero side signal in the upper
core bands (in 96 % of one broadcaster's M/S frames at 9 to 12 kHz), so
they are exactly mono there, and a variant that weakens alpha looks
"smoother" against them. And when a coded band's DPCM reference is a band
whose coefficient is not sent, the standard does not say what value to
use; this decoder uses 0, the best of three candidate rules on the frame-to-
frame continuity of alpha (inferred; about 4 000 such references in the
18 minutes, none on the other broadcaster). Finally, the capture checks
bound the alpha scale from above only: a decoder using twice the standard's
0.1 per step fails them, one using half of it passes them, and only the
unit tests (which encode the standard's text) catch that.

## Changes against the standard

Fixes of 2026-11-23, in both decoders (Python and C), each with a test that
fails without it. Output changes are measured on the 14 captures at hand
(both broadcasters), at 48 kHz, against the output before each fix, in
units of full scale (the CLI's level rule: the element's reference level
at 0.9 of full scale):

* **Spectral noise fill** (TS 103 190-1 5.1.4, Pseudocodes 22-24, 57).
  `asf_snf_data` was parsed but never applied, so bands the encoder left
  for noise stayed silent. Now applied with the standard's generator
  (RANDOM_NOISE_TABLE, reset from `sequence_counter` every frame, run
  through the channels in bitstream order: the standard says "at the
  beginning of the decoding of an ASF frame", read here as once per
  frame). No capture signals it: output unchanged; checked by unit tests
  and by a sample-exact comparison of the two decoders on 400 random
  channels (`c/tests/check_noise_fill.py`).
* **Each channel's own framing.** With `b_enable_mdct_stereo_proc` = 0 the
  two channels of a pair carry separate `sf_info`; the Python used L's for
  R. No capture has such a frame with differing framings: unchanged.
* **`channel_mode`** (TS 103 190-2 table 56). Codes above 7.1 (7.0.4 and
  up) were misdecoded and the top-channel fields of
  `ac4_substream_info_chan` never read. Stereo and 5.1: unchanged.
* **No HF patch before the first `aspx_config`.** A decode pass that held
  the stream's first configuration copied the low band up, unadjusted,
  for the frames before it. Now those frames are core only. Changes only
  those frames: up to 0.88 of full scale on broadcaster B's stereo (its
  first 48 frames), 0.047 on its 5.1 (first 120), 1e-6 elsewhere.
* **Time alignment.** L/R and Ls/Rs went through the QMF bank (577
  samples) and the centre and LFE did not, so in 5.1 the centre led the
  front pair by 12 ms; a stream or a decode pass without an A-SPX
  configuration was not delayed at all. Clauses 6.2.8 and 6.2.14 put every
  channel through the QMF bank; now every channel is 577 samples late
  (see Output above). The channels that move are the old ones delayed by
  12 ms (to 65 dB, the limit of the check; RMS within 0.4 %); the front
  pair against the centre went from +576 to -1 samples (the programme's
  own offset), measured by cross-correlation (`c/tests/align_check.py`).
* **A-SPX noise index** (Pseudocode 103): the index restarted at every
  envelope of a frame, so envelopes reused one another's noise. Now it
  runs from the frame's first envelope. Worst change 3.6e-3 of full scale,
  SNR 67 dB (broadcaster B stereo); 79 dB or more elsewhere.
* **A-SPX limiter groups** (Pseudocodes 96-101). The sums stopped one
  subband short of each group's top, so that subband was neither limited
  nor boosted; V1.3.1 of the standard prints this, V1.4.1 (2025) corrects
  it. Worst change 0.12 of full scale, SNR 36 dB (broadcaster B stereo,
  in its high band); 62 dB on B's 5.1 core, 93 dB or more on A's streams.
  The largest frame-peak rise: +0.4 dB on one frame; burst-guard firings
  on B's stereo unchanged (27 and 31 frames on its two captures).
* **The late `audio_size` check** could judge a frame already concealed
  for another reason again: the Python counted it twice in `n_bad`, the C
  replaced its reason. Now it keeps its first reason and is counted once.

## The C decoder

`c/` holds a port of the whole decoder to C99 (libm only, no other
dependency): `libac4dec` (frame-at-a-time API, `c/include/ac4dec.h`; a
fragmented-MP4 reader, `c/include/ac4dec_mp4.h`) and the `ac4dec-c`
command line, written from the standard and this package.

```
make -C c                        # libac4dec.a, libac4dec.so, ac4dec-c
make -C c test                   # unit tests + damaged-input gate (no captures)
make -C c asan                   # the same gate under AddressSanitizer + UBSan
c/ac4dec-c audio.m4s out.wav     # --f64 RAW, --status FILE, --lookahead N, ...
```

**Status.** It decodes what this package decodes (above), with the same
damaged-input rules and output guard, as a delay line: frames come out
`lookahead` frames after they go in (default 180, 6 s; minimum 8 for live
use). Against the Python on all 14 captures at hand (both broadcasters;
stereo, 5.1 codec mode 1, ASPX_ACPL_3), everything on: largest difference
4.7e-15 of full scale, 16-bit WAVs byte-identical, the same per-frame
statuses (`c/tests/compare.py`). On damaged input it gives the Python's
figures exactly at the default lookahead (`c/tests/robust_compare.py`); at
the minimum lookahead the guard seeds from fewer frames: 8 gate failures,
126 undamaged frames concealed. `c/tests/check_noise_fill.py`,
`late_check.py` and `align_check.py` check the noise fill, the late
`audio_size` check and the channel alignment against the Python. The C
streams and has no decode passes; it equals a one-pass Python decode.

**Speed** (2026-11-23, `c/tests/speed.py`, one thread, best of 2, decode
time; Python with its C kernels built):

| stream | length | Python | C |
|---|---|---|---|
| 5.1, codec mode 1 (A-SPX), broadcaster A | 10 s | 4.2x | 26.3x |
| stereo, broadcaster A | 10 s | 9.1x | 55.6x |
| 5.1 sent as ASPX_ACPL_3 (stereo core decoded), broadcaster B | 42 s | 7.5x | 51.7x |
| stereo, broadcaster B | 42 s | 9.3x | 60.1x |

## What it does not do yet

* **A-CPL.** An ASPX_ACPL_3 5.1 stream decodes to its stereo core (the coded
  downmix the broadcaster sent); C, LFE, Ls and Rs are written silent.
  `acpl_data_2ch()` is parsed, for bit accounting only (no upmix).
  ASPX_ACPL_1/2 and stereo_codec_mode 2/3 are concealed as damaged frames
  (`element.parse`).
* **A-SPX on the centre channel.** Its block is parsed (to keep state) but
  not rendered, so the centre stops where its core does (11.5 kHz measured).
* Companding (read, not applied); A-SPX additional harmonics (under 1.5 % of
  channel-frames); the SSF speech frontend; coding_config other than 0.
* Mono, 3.0, 5.0 and 7.x elements; immersive and object audio (A-JOC, IFM,
  OAMD).
* Bitstream versions 0 and 1; frame rates other than 29.97; sample rates
  other than 48 kHz.
* Choosing a presentation or substream: substream 1 is always decoded.
* DRC, dialogue enhancement and loudness metadata.
* Raw AC-4 sync frames, MPEG-TS, non-fragmented MP4, the `dac4` box.

## Damaged input

AC-4 in MP4 carries no frame CRC, and every Huffman codebook is a complete
prefix code, so random bits always "decode". The decoder therefore checks
what the bitstream structure fixes, conceals what fails, and keeps a level
guard as a last resort (`ac4/robust.py` lists every rule):

* **Bitstream checks.** The table of contents (substreams fit the frame,
  sequence_counter <= 1020, layout and rates as in the stream so far); `audio_size` closure: the parse of `audio_data()` must end 0..7
  bits before `audio_size` says (byte alignment), which holds on every
  clean frame measured (uniform over 0..7). That covers the core, all A-SPX
  blocks (centre included) and, for ASPX_ACPL_3, `acpl_data_2ch()`, which
  is now read for bit accounting (not synthesised); section codebooks 12-15,
  sections that overrun `max_sfb`, `max_sfb` beyond the band table,
  quantised lines above 8191, scale factors outside 0..255; A-SPX
  envelope counts (Table 127) and borders; and a sanity bound on dequantised lines (1e9, 37 dB above the
  largest line measured on air, 1.4e7). Frames from before a stream's
  first `aspx_config` cannot be sized when they arrive; once an I-frame
  later in the same decode pass brings the configuration, they are
  re-checked with it, if at least 90 % of them close.
* **Sequence counter.** A frame whose counter breaks sequence while the
  NEXT frame continues from the one BEFORE it is an outlier (swapped,
  repeated or foreign) and is concealed. Other jumps (frames lost on air)
  decode normally and are counted (`n_seq_jumps`).
* **Source changes.** A frame whose TOC layout, rates or codec mode differ
  from the stream's is concealed, unless the change is real: it is
  accepted at an I-frame (for a TOC change, when the next frame's TOC
  agrees) or after 3 frames in a row that show the same change
  (TS 103 190-1 4.3.3.2.2 expects such "uncontrolled" source changes).
  One damaged TOC field in the first frame now costs 0 to 2 frames (it
  used to mute 119 of 120); a splice to a stream in another codec mode
  recovers at that stream's first I-frame (it used to mute everything
  after it).
* **Concealment.** A concealed frame's spectrum is zeroed: the previous
  frame's overlap fades out through its window and the next good frame
  fades in through its own. Its A-SPX high band is muted, the A-SPX
  framing and envelope state are held from the last good frame, and the
  A-SPX noise sequence runs on as if the frame had been decoded. Output
  length is always 1536 samples per input frame.
* **Level checks.** Some damage changes values without changing any bit
  count (a flipped bit in an escape value, in a fixed-width field), and
  A-SPX envelopes are delta coded in time, so one such frame can leave the
  high band wrong for several frames. Two checks on the decoded signal:
  1. `aspx.level`: a frame whose A-SPX output RMS is above 16x the RMS of
     the core it was regenerated from plays its core band (frames whose
     A-SPX output peaks below 0.03x the element's fixed level are not
     judged: over a near-silent core the ratio means nothing, up to 7.5e6
     measured on clean frames whose A-SPX peak was below 0.012x). Clean
     maximum 7.43 (margin 2.15x), measured before full SAP was added.
  2. `guard.level`, the output guard: a frame whose peak is above 3x a
     reference is muted with 256-sample ramps. The reference is the larger
     of the stream level (the largest accepted frame peak, released by
     0.9999 per frame, seeded from the 5th-largest of the first 30 frames,
     and raised by at most 2x per frame, so one loud frame that gets
     through cannot switch the guard off) and a start level (0.75x the
     element's fixed level, halving every 46 s, so a decoder that starts
     in a quiet passage does not mute the loud passage after it). A rise
     that lasts (4 of the next 6 frames above the reference) is programme,
     not damage, and is not muted. Frames within 3 of a concealed frame,
     concealed frames themselves, and the 3 frames after a frame the 3x
     rule muted (every channel is 577 samples late, so a damaged frame
     reaches into the second output frame after it) are judged at 2x the
     stream level alone and do not raise it: damage comes in runs, and
     clean material has no concealed or muted frames, so this rule cannot
     touch it.
  Simulated on the per-frame peaks of 136,461 clean frames (about 76 min,
  both broadcasters, 5.1 with full SAP decoded and stereo) from 549 start
  points: no frame fires; the largest clean frame peak is 2.51x the
  reference, a margin of only 1.20x (1.6 dB). End to end, nothing was
  concealed and no level check fired on 103,616 clean frames of the merged decoder (about 58 min, both
  broadcasters, including three 16-18 minute files).
  `Decoder(guard=False)` / `ac4dec --no-guard` turn the output guard off;
  `robust.CHECKS = False` turns the bitstream checks and `aspx.level` off
  (for negative controls).
* **MP4.** A `trun` whose sample count cannot fit its box is ignored; a
  sample whose size or offset points outside the `mdat` is passed on empty
  (and concealed), so the sample count, and the timeline, is kept. A
  fragment whose boxes are destroyed is still lost as a whole.

`Decoder.n_concealed` counts concealed frames, `Decoder.concealed.reasons`
gives them by reason, and `Decoder.last_concealed` marks them per frame of
the last pass. `ac4dec` prints the count and the reasons.

Measured 2026-11-23 with `c/tests/robust_compare.py` (the measurement of
tests/test_robust.py) on 7 cuts of both broadcasters (4 fragments each),
12 kinds of damage x 20 seeds (11,046 damaged frames: 6 per decode for 10
kinds, a swap damaging two; runs of 3 to 20 consecutive frames; a damaged
TOC in the first frame): no exception, exact length everywhere. 8,185
damaged frames were concealed, 983 decoded identical to the clean decode
around them and 1,876 differently but with no sample above 2x the clean
stream's peak; 2 were LOUD. 27 undamaged frames were concealed in all (at
most 4 per hit allowed). Worst peak 4.45x the clean stream's own peak
(byte bursts; 1.48x for runs, 1.21x or less for every other kind): two
isolated burst frames early in broadcaster A's 5.1 cut, under the
guard's start-level limit (see below), are the only failing decodes (2,
with 4 gate messages; 3 decodes, 5 messages before the 2026-11-23 fixes).
On the clean cuts nothing is concealed and the guard changes no sample.
The C decoder gives exactly the same figures at its default lookahead.
(The figure of 1.82x printed here before was measured on an earlier set
of cuts and did not reproduce: the decoder of the C port's milestone 2
gave 4.45x, 2 LOUD, 5 failures on these.) With the checks and level checks
off, the old gate failed 246 times, worst peak 1.8e11x. Replaying the
original survey's damage (measured before the 2026-11-23 fixes; 15
frames with 3 bytes inverted, 5 truncated, 1 empty, in 300 frames) on
broadcaster B's two 42 s captures: the decoder before this work peaked at
2.3e5x (stereo) and 5.6e6x (5.1) the clean peak; this one at 1.05x and
0.98x, flagging 16 and 19 of the 21 frames.

On about 10 hours of one broadcaster's off-air captures (1.1 million
frames, 5.1 and stereo), the bitstream checks alone (an early version:
before the spectral bound, the late closure check, full SAP and the level
checks; no synthesis) flag 36 frames, every one of them in a capture file
that also shows sequence-counter jumps (frames lost on air); the files
without jumps have none. All the flagged frames that were checked sit
inside clusters of lost frames, so they are inferred to be real reception
damage. This survey was not repeated on the final decoder.

What is still possible (measured with the attacks of the review, on 4 s
cuts):
* Damage that keeps every rule and stays under the level checks decodes as
  a click or a wrong frame within the programme's own level (worst in the
  gate: one frame about 42x its local clean level, while under 2x the
  stream's peak).
* An ISOLATED damaged frame (no concealed neighbour) is judged only by the
  3x rule, and early in a stream that is quieter than the element's fixed
  level the start level dominates: on broadcaster B's quiet 5.1 stream the
  guard's limit is about 7.7x the stream's own peak in the first 4 s
  (about 4.9x after 10 s on its 42 s capture). Single damaged frames there
  still reached 4.37x the clean peak (one constructed case), and 1 of 40
  decodes with six random single-bit flips reached 3.43x.
* A frame from before the stream's first I-frame, in a pass with no
  I-frame, gets only the upper-bound closure check. The outlier rule needs
  the next frame, so it cannot act on the last frame of a decode pass. The
  sustained-rise exception needs 6 frames of lookahead, so a loud passage
  that starts in the last 6 frames of a pass can lose up to 6 frames. A
  fragment whose MP4 boxes are destroyed is lost, so its frames drop out.
* A real source change costs the frames up to the new stream's first
  I-frame (they cannot be decoded without its configuration), plus up to
  3 frames after it when the new stream is louder (the near-damage rule).

**A-SPX parsing fix (changes output on some clean frames).** With
`aspx_balance = 1` the second channel uses the first channel's framing,
and so (Pseudocode 76, per channel) its `previous_stop_pos` too. The
original parser left it stale, which misread the next VARFIX/VARVAR frame
of that channel: about 1 in 1,300 frames of one broadcaster's 5.1 stream
(26 of 32,700 fail closure without it). With the fix every such frame
closes exactly on `audio_size`; without it they fail closure and would be
concealed. Before full SAP was added, on a 3-minute 5.1 file the output
differed from the original receiver only in the 5 frames whose framing the
fix changes (and the frames their overlap reaches); 2 of those 5 fail
closure without the fix, the other 3 rest on the reading of the standard
alone. `aspx_parse.BALANCE_SHARES_STOP = False` restores the old
reading.

## Known issues

* **Output level.** The decoder's true full scale is not established. The
  CLI keeps the original receiver's level rule: a fixed reference per
  element (9e5 for 5.1, 2e5 for stereo, decoder units) written at 0.9 of
  full scale, rising only when a louder sample appears, capped at twice the
  start. `decode()` returns the decoder's own scale.
* **Burst guard.** Separate from the damaged-input guard above: an A-SPX
  frame above twice the element's fixed reference plays its core band. It
  is a heuristic carried over from the receiver, and it fires on clean
  material too (31 frames of one broadcaster's 42 s stereo lane).
* **Pass boundaries.** The output is not fully independent of how the
  frames are split into decode passes (`decode_pass`; the CLI's default
  pass is 1800 frames, 60 s): the last frame of a pass can differ from a
  one-pass decode in its last 1279 samples (the A-SPX path carries a
  4-frame lead-in between passes, not the whole past). Worst frame,
  measured before the 2026-11-23 fixes with 40-frame passes, in dB of the
  stream's peak: broadcaster B stereo -37.5, B 5.1 -46.7, A 5.1 -42.5
  (median of differing frames -80.0), A stereo -123.9. About one frame per
  pass differs. (The 577-sample jump a pass made when the first A-SPX
  configuration fell in a later pass is gone: every channel is now
  delayed the same way from the first frame.) The C decoder streams and
  has no passes; it equals a one-pass Python decode.
* **A-SPX envelope dequantisation** (found while checking full SAP, effect
  not measured): `aspx_envadj.map_scf` and `map_scf_joint` always use
  a = 1, while TS 103 190-1 Pseudocodes 82 and 84 use a = 2 for 1.5 dB
  steps (`aspx_qmode_env` 0, which `aspx_data` forces for FIXFIX frames
  with one envelope). Inherited unchanged from the original receiver.
* `channels.asf_scalefac_data` and `channels.asf_snf_data` refer to an
  undefined name (`NUM_SFB_48`). Nothing calls them; they are kept as copied.
* **A-SPX transient envelope** (found 2026-11-23, not fixed): Pseudocode 95
  leaves out the noise factor of the envelope at `aspx_tsg_ptr` (and
  `p_sine_at_end`), and Pseudocode 99 the noise term of the boost for it;
  both decoders apply them to every envelope (no sinusoids are rendered,
  see above).

## Tests

```
pytest                                    # unit tests, no audio needed
AC4_TEST_DIR=/path/to/captures pytest     # + integration on your captures
AC4_REFERENCE_REPO=/path/to/receiver ...  # + byte comparison with the original
```

Unit tests cover the banked tables (Kraft sum and prefix-freeness of every
Huffman codebook, scale-factor band layouts), the bit reader and
`variable_bits`, Huffman round trips on real codebooks, MDCT perfect
reconstruction, KBD windows, block-switching orthogonality with its negative
control, the QMF round trip, the stream resampler, the MP4 reader on
synthetic files, the C kernels against the Python reference, and the
fixes of 2026-11-23 (`channel_mode` codes, each channel's framing, the
noise fill generator and levels, the A-SPX limiter against the
standard's pseudocode, the noise index across envelopes, no HF patch
without A-SPX data, all six channels time-aligned, the late check on an
already concealed frame).

Integration tests cut each capture to its first two fragments (about 4 s)
and check that every frame decodes and the WAV length is exact. (The
byte comparison with the original receiver's audio worker is retired:
the 2026-11-23 fixes change the output on purpose.) Cuts with full SAP frames skip that comparison (the
receiver did not decode them); `tests/test_sap.py` (no captures) and
`tests/test_sap_captures.py` (`AC4_SAP_TEST_FRAGMENTS`, default 10) check
full SAP: bit accounting to `audio_size`, the A-SPX envelope reference
with wrong variants that must fail, and that only SAP3 channel pairs
change.

`tests/test_robust.py` is the damaged-input gate. Without captures it
feeds garbage, empty and truncated frames and damaged MP4 boxes, and checks
the output guard on synthetic signals (a burst, a crescendo, a loud return
after six quiet minutes, one loud frame that must not lift the guard). With
captures it damages each cut deterministically (`tests/inject.py`: bit
flips, byte bursts, truncation, zeroed bytes, swapped, repeated, extended,
empty and garbage frames, frames from another stream, runs of consecutive
damaged frames, a damaged first-frame TOC; `AC4_ROBUST_SEEDS` seeds per
kind, default 3) and requires: no exception, exact length, no sample above
2x the clean decode's own peak, at most 4 undamaged frames concealed per
hit, and every damaged frame concealed or harmless (output identical to the
clean decode around it, or no sample there above 2x the clean peak). It
also splices every pair of same-element cuts and requires recovery at the
new stream's first I-frame, and requires that nothing is concealed, and the
guard changes nothing, on the clean cuts. Its negative control turns the checks and the guard off
(`robust.CHECKS = False`, `guard=False`) and requires the same gate to
fail.

## Layout

| module | role |
|---|---|
| `mp4.py` | fragmented-MP4 sample reader |
| `toc.py`, `toc_v1.py` | table of contents (`toc_v1` holds the bit reader and constants) |
| `hcb.py`, `spectral.py` | Huffman codebooks and decoding |
| `sfb.py` | scale-factor band offsets |
| `channels.py`, `pair.py` | `5_X` channel element; `channel_pair_element` |
| `audio.py` | dequantisation, stereo processing (M/S and full SAP), noise fill |
| `acpl_parse.py` | A-CPL side data, parsed only (no upmix) |
| `frameinfo.py` | walks each frame's audio_data() to its end and checks it against audio_size |
| `mdct.py`, `filterbank.py` | MDCT, KBD windows, block switching, overlap-add |
| `qmf.py` | 64-band complex QMF bank |
| `aspx_*.py` | A-SPX bands, parse, envelopes, HF generation, adjustment, render |
| `resample.py` | 1001/960 output resampler |
| `stream.py` | `Ac4Stream`: frames to spectra, with cross-frame state |
| `decoder.py` | `Decoder`: spectra to PCM; `decode()` |
| `robust.py` | damaged-frame rules, sequence rules, output guard |
| `cli.py` | `ac4dec` |
| `scalefac.py`, `pcm.py` | early LFE prototypes, kept for constants they define |
| `tables/` | numeric tables (no standards text) |
| `*_kernel.c`, `build_kernels.py` | optional C kernels |

Environment: `AC4_QMF_KERNEL=0` / `AC4_SPECTRAL_KERNEL=0` disable a kernel.
`AC4_ETSI_TABLES_C` / `AC4_ETSI_PDF` (maintainers only) point the table
loaders at the standard's own files instead of the banked JSON.

## Provenance

This code was copied from the author's ATSC 3.0 receiver at commit `c4b296e`,
where it was written and tested against live broadcasts. The copy was
mechanical: modules renamed, imports made relative, the research command-line
entry points and capture-reading helpers removed, broadcast-identifying
comments reworded, the table path and kernel environment variable names
changed. The decoding arithmetic was unchanged, which a reference
comparison test checked byte for byte. Since then, one parsing fix (A-SPX
`previous_stop_pos` under `aspx_balance`, see Damaged input), full SAP
decoding (the receiver passed those frames through) and the fixes under
[Changes against the standard](#changes-against-the-standard) change the
output on purpose.

The one function the cluster used from the rest of the receiver, a
fragment-run scanner, was replaced by `mp4.py`. The PCM loop in `decoder.py`
was lifted from the receiver's audio worker without its file-tailing,
resume, padding and WAV-rolling logic.

Old to new module names: m19 `toc_v1`, m20 `toc`, m22 `mdct`, m23 `hcb`,
m24 `spectral`, m25 `scalefac`, m26 `pcm`, m27 `sfb`, m28 `channels`,
m29 `audio`, m30 `filterbank`, m31 `aspx_bands`, m32 `aspx_parse`, m33
`qmf`, m34 `aspx_env`, m35 `aspx_hfgen`, m36 `aspx_envadj`, m37
`aspx_render`, m38 `resample`, m42 `stream`, m43 `pair`. Docstrings and
comments still use the old names (M29, m37, ...) and the receiver's
experiment numbers (E77, E112, ...); dates in them (8/06, 11/10, ...) are
2026.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE), which also carries
the patent and trademark statements.
