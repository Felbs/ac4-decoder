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
ac4dec audio.ac4 --list          # inputs: MP4, AC-4 sync frames, MPEG-TS
ac4dec in.ts out.wav --language fr      # or --presentation N / id:N
```

```python
from ac4 import decode, container
pcm, dec = decode(container.read_frames("audio.m4s"))  # float64 (n, ch), 48 kHz
print(dec.chans, dec.n_ok, dec.n_bad)             # decoder scale, not +-1.0
print(dec.n_concealed, dict(dec.concealed.reasons))   # damaged frames
```

## What it decodes

Everything below was measured on off-air captures from two ATSC 3.0
broadcasters using different encoder configurations, and on Dolby's public
AC-4 Online Delivery Kit test signals. No captures or test signals are in
this repository.

| | |
|---|---|
| Bitstream | version 2, channel-coded; every frame rate of TS 103 190-1 Table 83 at 48 kHz (frame lengths 2048 .. 384, internal rate resampled by the table's ratio) and 44,1 kHz (Table 84, index 13). frame_rate_factor / frame_rate_fraction above 1 and 96/192 kHz substreams are concealed (`toc.unsupported_rate`), not misdecoded. Checked on air at 29.97 fps and on public test streams at 25 and 29.97 fps; the other rates on synthetic streams only (unverified on real streams) |
| Input | detected from the content: MP4 (fragmented: init segment + `moof`/`mdat` fragments, or bare fragments; or non-fragmented, from the sample table), AC-4 sync frames (TS 103 190-1 Annex G, CRC checked, ID3 tags skipped), MPEG-2 TS (ETSI TS 101 154). The `dac4` box is parsed (`--list`) |
| Presentations | the table of contents is parsed in full (presentations, substream groups, substream indices, content type and language, object/A-JOC substream info); the decoded presentation is chosen per TS 103 190-2 4.8.2 (see Interpretations), or with `--presentation N`, `id:N`, `--language`; its audio substream is located by its `substream_index`, and is an I-frame when its own `b_audio_ndot` says so |
| 5.1 (`5_X` channel element) | codec mode 1 (A-SPX), coding_config 0: all six channels (both broadcasters). Codec mode 3 (ASPX_ACPL_2): all six channels, Ls/Rs from the one-channel A-CPL of L/R (the public 5.1 test streams). Codec mode 4 (ASPX_ACPL_3): all six channels, L/R/C/Ls/Rs from the A-CPL upmix of the two-channel core, the LFE from its own data (see A-CPL below). On synthetic streams only (unverified on real streams): codec mode 0 (SIMPLE); coding_config 0 with 2ch_mode 1 and coding_config 1 - 3 (three, four and five_channel_data, every chel_matsel; Tables 178 - 180); ASPX_ACPL_1 (codec mode 2: the two max_sfb_master side tracks, Tables B.8 - B.19, 5.3.4.3.2) and coding_config 1 of codec modes 2 and 3 |
| Stereo (`channel_pair_element`) | stereo_codec_mode 0 (SIMPLE), 1 (A-SPX), 2 (ASPX_ACPL_1) and 3 (ASPX_ACPL_2); modes 2 and 3 on synthetic streams only (unverified on real streams) |
| Mono, 3.0, 5.0, 7.x | single_channel_element (mono: C), 3_0_channel_element (L R C, both coding configs), 5_X without LFE (5.0: L R C Ls Rs), 7_X_channel_element in codec modes SIMPLE and ASPX for all six 7.0 / 7.1 channel modes of Table 88 (L R C [LFE] Ls Rs plus Lb Rb, Lw Rw or Tfl Tfr; Tables 182, 183 with b_use_sap_add_ch; A-SPX on the four blocks of Table 213): on synthetic streams only (unverified on real streams). 7_X ASPX_ACPL_1 / _2 are concealed (`element.parse`) |
| "IMS" (the kit's immersive-stereo streams) | presentation_version 2 with a 7.1 channel mode: the substream is a channel_pair_element (see Interpretations); its stereo core is decoded, both kit streams without a concealed frame. The rendering to 7.1 is not in the standard at hand |
| Core (ASF/MDCT) | Huffman spectral data, scale factors, long and short frames with block switching, KBD windows, stereo processing (sap_mode 1 and 2: M/S; sap_mode 3: full SAP with prediction coefficients, see below), spectral noise fill (clause 5.1.4, the standard's own noise generator; no capture here signals it), LFE |
| A-SPX | parsed and rendered on the L/R and Ls/Rs pairs and the centre: 64-band complex QMF, envelope decoding (1.5 and 3 dB steps), HF generation with TNA and pre-flattening, envelope adjustment with the limiter, noise, additional harmonics (sinusoids), FIC/TIC (interleaved waveform coding), joint-balance rendering; companding (5.7.5) of the core |
| A-CPL | ASPX_ACPL_3 (clause 5.7.7.6.2): parameter decoding and dequantisation, interpolation, the three decorrelators with the transient ducker, the 5-channel upmix in the QMF domain. ASPX_ACPL_1/2 (5.7.7.6.1, one-channel A-CPL): in a channel pair, and ASPX_ACPL_2 in 5_X |
| Output | 48 kHz (44,1 kHz) via the resampler of TS 103 190-1 Table 83 for the stream's frame rate (1001/960 at 29.97 fps; without it audio runs 4.27 % fast). Every channel is 577 samples (12 ms) after the MDCT core: the A-SPX channels through the QMF bank, the others (LFE, a pair without A-SPX) through a delay line of the same length, so all channels are time-aligned. Channel order checked on a public 2.0 channel-identification signal against its AAC version (envelope correlation L-L 0.947, R-R 0.949; crossed 0.90, 0.89) |
| Loudness, DRC | applied (`ac4/loudness.py`, C `src/loudness.c`): dialnorm normalised to a target level (default -24 dB, `--target-level`), the stream's DRC decoder mode for that level (5.7.9.2): E-AC-3 profiles, transmitted curves, transmitted gains (`--drc auto|off|N`). The CLIs write this; `--legacy-level` / `--ref-peak` keep the old fixed-reference rule |
| Metadata | parsed: further loudness information; dialogue enhancement config and parameters (DE is at its default user level 0: no change); EMDF payload ids (`tools/survey.py` reports them) |
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
| 5.1 sent as ASPX_ACPL_3, broadcaster B (stereo core only; with the A-CPL upmix, 2026-11-23, `c/tests/speed.py`: 3.5x) | 42 s | 3.4x |
| stereo, broadcaster B | 42 s | 4.4x |

The damaged-input checks and full SAP cost 1 to 19 % against the version
before them on the same runs (1.95x / 4.87x / 4.22x / 5.01x then); the
spread between runs is of the same size, so the cost is not pinned down.

**Bandwidth** (same decodes; the highest frequency at which a channel's
long-term average spectrum, Welch with 8192 points over the whole file, is
within 50 dB of its maximum): broadcaster A 5.1: L 16.9, R 16.5, C 11.5,
Ls 16.9, Rs 11.5 kHz (LFE 0.2 kHz); broadcaster A stereo: 12.8 kHz;
broadcaster B 5.1 (its stereo core): 18.0 kHz, and with the A-CPL upmix
(2026-11-23, 4096 points, from the first I-frame) L 16.8, R 15.6, C 14.3,
Ls 17.5, Rs 15.0 kHz (LFE 0.2 kHz); broadcaster B stereo: 16.5 kHz. These
figures describe what the broadcasters send as much as the decoder, and
were measured before the A-SPX fixes of the next section (the centre now
has A-SPX; the high bands are now on the core's scale). After them, the
better measure is the spectral step at the A-SPX crossover, which a
correct high band keeps at the programme's own tilt: -0.9 dB on
broadcaster B stereo, -3.2 / -2.9 / -1.9 dB on broadcaster A's L / R / C,
-3.9 dB on a public stereo test stream (it was -35 dB, or +11 dB,
depending on which of the two bugs was in force).

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

### A-CPL (ASPX_ACPL_3)

Broadcaster B sends its 5.1 as a stereo core plus advanced-coupling
parameters (5_X_codec_mode 4). Every 5_X frame on the captures is codec
mode 1 (broadcaster A, which by Table 214 has no A-CPL) or 4 (B); none is
ASPX_ACPL_1/2. B's stream uses 15 parameter bands, fine quantisation, one
parameter set per frame, smooth interpolation (1,123 frames) and steep
(13). The decoder parses and differentially decodes the eleven parameter
sets (Tables 62-65, Pseudocode 121, the codebooks' cb_off), dequantises
them (Tables 203-208), and builds L, R, C, Ls, Rs from the core pair's
A-SPX output in the QMF domain: interpolation (Pseudocode 109), three
decorrelators with the transient ducker (Pseudocodes 111-114), the upmix
(Pseudocodes 118, 119); the LFE is decoded from its own data. Readings
are under [Interpretations](#interpretations).

There is no reference output, so it was checked against what the
standard implies (`c/tests/acpl_check.py`, B's 42 s capture, from its
first I-frame; levels in dB of full scale):

* **It sums back to the core.** Pseudocode 118 makes L + Ls/sqrt2 +
  C/sqrt2 a mix of the core pair alone (the decorrelated terms cancel);
  with the transmitted gammas, (L + C/sqrt2 + Ls/sqrt2) / (1 + sqrt2)
  equals the core's left channel to -49.8 dB (right: -48.8 dB; worst
  frame -24 dB, median -72 dB). This also checks the gamma decoding.
* **Levels.** RMS / peak: L -25.8 / -8.8, R -25.8 / -8.4, C -20.9 / -3.0,
  Ls -32.8 / -11.2, Rs -32.3 / -10.8, LFE -37.2 / -20.0 (the core pair
  -26.3 / -8.9); nothing near clipping, all samples finite. C carries 2.6
  dB more than the core pair below 8 kHz; the surrounds sit 7 to 10 dB
  under it, the fronts 2 to 3 dB under it.
* **Boundaries.** Frame-to-frame level steps of the upmixed channels are
  those of the core (99.9th percentile 15-19 dB against 16-17 dB, the
  programme's own); at the I-frames (where the configuration is sent) at
  most 3.5 dB on L/R/C and 7 dB on the surrounds. High-frequency energy
  folded on the frame period: peak/mean 1.12-1.23 on the upmixed channels
  against 1.15 on the core (no click at frame boundaries). At the first
  I-frame the upmix starts without a gap (C from -35 to -18 dB, the
  surrounds from silence to -32 dB).
* **Decorrelators.** All-pass sections with poles inside the unit circle
  (largest radius 0.959); an impulse keeps its energy (unit tests).
* **Alignment.** L and R at lag 0 against C (correlation 0.87, 0.91);
  the LFE at +165 samples (correlation 0.55, the programme's low-pass).
* **Listening proxies.** C correlates 0.99 with the core's mid signal and
  holds 42 % of its energy in the speech band (250 Hz - 4 kHz), as L and
  R do; the surrounds are decorrelated but not silent (Ls~L 0.20, Ls~Rs
  0.15, Rs~R -0.28; their best lag against C is 500-900 samples, the
  decorrelators' delay).

### Against independent references (phase B2)

The public test kit carries the 5.1 programme also as E-AC-3 (256 kbit/s)
and the stereo one as AAC; FFmpeg decodes those (E-AC-3 with
`-drc_scale 0`), and the decode of the AC-4 version (`--drc off`) is
compared after alignment (1163 samples for the 5.1, by cross-correlation
of the centres).

* **5.1 (ASPX_ACPL_2), levels and correlation.** Per channel within
  +-0.9 dB (L +0.5, R +0.1, C +0.1, LFE +0.1, Ls -0.5, Rs -0.9);
  correlation C 0.998, LFE 0.998, L 0.92, R 0.91, Ls 0.88, Rs 0.85.
* **The surrounds lean towards the fronts, by design.** Our Rs correlates
  a little better with the reference R (0.857) than with the reference Rs
  (0.853). This is the coding, not a mapping error: in ASPX_ACPL_2 a
  surround is rebuilt from its side's downmix L + Ls/sqrt2 (here R +
  Rs/sqrt2) and decorrelated noise (Pseudocodes 116, 117), and the
  programme's own R and Rs correlate 0.74. A best-case parametric
  reconstruction computed from the reference itself (per 43 ms x 1/7
  octave tile, the least-squares share of the downmix plus decorrelated
  energy for the rest) prefers the front too above 1.6 kHz (Rs~R 0.37
  against Rs~Rs 0.29 at 1.6 - 3.2 kHz). What the parameters control
  matches: per tile, the share of the downmix in each surround (its
  projection on the downmix) averages 0.18 - 0.51 against the
  reference's 0.17 - 0.46 between 0.4 and 6.4 kHz (within 0.03 in every
  octave of both sides but Rs at 0.4 - 0.8 kHz, 0.05), and the surround energy against the downmix 0.24 -
  0.46 against 0.21 - 0.41; the per-tile L/Ls and R/Rs level ratios
  correlate +0.4 to +0.5 with the reference's (a swapped pair or a
  negated alpha makes them negative). Ls/Rs are not swapped (Ls~Ls 0.88
  against Ls~Rs 0.70).
* **12 - 16 kHz.** Phase B's "coupled channels about 5 dB low at 12 - 16
  kHz" is the reference's own coding: above 11.4 kHz its L, R, Ls, Rs are
  a flat spectral-extension plateau (to 13.6 kHz, then 20 dB down), its
  centre is not. Our coupled channels sit 3 - 7 dB under that plateau, the
  centre within 1 dB. On the AC-4 side the A-SPX crossover is 12.75 kHz;
  the A-SPX output meets the encoder's transmitted envelope energies
  within 0.25 dB up to 15 kHz on L/R (within 1 dB on C); above 15 kHz the
  L/R output falls 5 - 11 dB under them, which is the standard's limiter
  (Pseudocodes 96 - 100, two limiter groups by Pseudocode 72) capping the
  gain where the patched source jumps by 24 dB at the patch border
  (subband 44). Below the crossover the MDCT core falls 10 dB from 11.5
  to 12.6 kHz on the L/R and C tracks alike.
* **Stereo against AAC.** Level within -0.7 to -0.9 dB in every band to
  10 kHz; the AAC version is low-passed at 11 kHz, so it says nothing
  about the A-SPX range.

No decoder error was found; nothing changed.

## Changes against the standard

### Phase A (2026-11-23, later)

In both decoders, each with a test that fails without it; output changes
measured on the 12 distinct captures (both broadcasters), each against the
output before it, at 48 kHz, in units of full scale (the CLI's level rule).

* **`stereo_data()`** (TS 103 190-1 Table 23). The channel_pair_element and
  the ASPX_ACPL_3 core read `two_channel_data()` (Table 26), which lacks the
  `spec_frontend` bit before each channel's `sf_info` when MDCT stereo
  processing is off. Such frames misparsed (none on the captures; 2.5 to
  4 % of the frames of the public stereo test streams).
* **The substream's I-frame flag** is its own `b_audio_ndot` (TS 103 190-2
  6.3.2.7.6), not `b_iframe_global`. The public test streams send a
  substream I-frame, with its `aspx_config`, halfway between global
  I-frames; the decoder skipped that configuration and misparsed the frame.
  The captures never differ: unchanged.
* **The TOC** is parsed in full; `total_n_substream_groups` is 1 + the
  largest group index (6.3.2.1.8), not the sum over presentations (two
  presentations sharing a group misread the substream table); the audio
  substream is located by its `substream_index`, not assumed to be
  substream 1. Captures unchanged.
* **The IMDCT scale** (5.5.2.2, Pseudocodes 60-63) is (1/N) * sum for every
  transform length N. Both decoders scaled each block by sqrt(N/2) (an
  orthonormal lapped transform): against the standard, short blocks came out
  sqrt(1536 / N) too quiet relative to long blocks (3, 6, 9, 12 dB at 768,
  384, 192, 96) and the time-domain aliasing at long/short transitions did
  not cancel. Now every block is (1/N) * sum times one constant,
  `IMDCT_K` = sqrt(2 * 1536), so long blocks keep the decoder scale. On air,
  (768, 768) frames between long frames sat 0.5 to 1.3 dB below their
  neighbours (median, 6 streams, 361 frames); now within 0.5 dB. Change:
  SNR 11 to 18 dB, all in frames with short blocks.
* **A-SPX envelopes in 1.5 dB steps** (Pseudocodes 82, 84: a = 2 when the
  interval's `aspx_qmode_env` is 0, which `aspx_data` forces for a FIXFIX
  interval with one envelope, the commonest framing). Both decoders used
  a = 1 (the known issue "effect not measured"), so those envelopes came
  out 2^(q/2) too large. The envelopes are absolute energies on the
  standard's scale, so they also take `IMDCT_K`^2 (`aspx_envadj.ENV_SCALE`);
  the a = 1 error had stood in for that factor on average. The frames it
  overshot were the "A-SPX bursts" (31 frames of broadcaster B's 42 s
  stereo): the burst guard now fires on no capture. Change: broadcaster B
  SNR 13 to 36 dB per channel, broadcaster A 26 to 69 dB.
* **The QMF tools' time grid** (the old known issue): `TS_OFFSET` 18, see
  Interpretations. Change: broadcaster B SNR 21-26 dB on stereo and the
  fronts, 11-12 dB on the upmixed surrounds (A-CPL moves with it);
  broadcaster A 33-71 dB.
* **A-SPX on the centre** of a 5_X element (its `aspx_data_1ch`, Table 25),
  rendered with the pair renderer fed the centre twice (with
  `aspx_balance` 0 each half decodes alone). Broadcaster A's centre now
  continues across the crossover (-1.9 dB). Change: centre only, SNR 44-46
  dB.
* **Frames at other rates** (not 48 kHz / 29.97 fps) are concealed as
  `toc.unsupported_rate` instead of being decoded as 1536-sample frames.

### Earlier (2026-11-23)

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
* **A-SPX transient envelope** (Pseudocodes 92, 95, 99). The envelope that
  starts at a transient (`atsg == aspx_tsg_ptr`), and envelope 0 when the
  previous interval of the channel ended on its pointer (`p_sine_at_end`),
  take no noise factor in their gain and no noise term in their boost;
  both decoders applied them to every envelope. Readings: FIXFIX carries
  no `aspx_tsg_ptr`, taken as -1 ("none", the value the field's `tmp - 1`
  coding gives); Pseudocode 95 computes a `b_sine_at_end` from the
  current interval but tests `p_sine_at_end`, which Pseudocode 92 defines
  from the previous one; that one is used. Rare on air (8 of 2,363
  envelopes on broadcaster B's 5.1, 8 of 2,670 on its stereo, 4 of 1,218
  and 10 of 632 on broadcaster A's 10 s lanes): worst change 3.7e-4 of
  full scale, SNR 102 dB (broadcaster B stereo, 42 s; 99 dB on its 10 s
  lane); 109 dB or more elsewhere.
* **The late `audio_size` check** could judge a frame already concealed
  for another reason again: the Python counted it twice in `n_bad`, the C
  replaced its reason. Now it keeps its first reason and is counted once.

## Interpretations

Where TS 103 190-1 leaves room, these are the readings both decoders take
(the A-SPX readings are under
[Changes against the standard](#changes-against-the-standard)).

**Phase B2**

* **The kit's "IMS" streams** (presentation_version 2, channel_mode 7.1
  3/4/0 in ac4_substream_info_chan): read as a 7_X_channel_element no
  frame parses (the LFE's sf_data fails at once); read as a
  channel_pair_element (ASPX) every frame of both streams closes on
  audio_size. TS 103 190-2 V1.3.1 gives no other syntax for such a
  substream, so a presentation_version 2 presentation with a 7.x channel
  mode is decoded as a stereo pair (`decoder.IMS_STEREO`); the decoder
  does not render it to the signalled 7.1.
* **7_X additional channels** (Table 33): the two chparam_info() of
  b_use_sap_add_ch come before the additional channels' sf_info; each is
  read with the framing of the track it pairs F (G) with (D / E for
  3/4/0, A / B otherwise), and Table 183 is applied per transform window,
  which needs both tracks to have the same windows: a frame where they
  differ is concealed (`asf.add_ch`). The BS.1770 weight of Lb / Rb is
  1.41 (the surrounds'), of Lw / Rw and Tfl / Tfr 1.0.

* **max_sfb_master side tracks** (ASPX_ACPL_1 in 5_X, Table 25; 4.3.5.13
  names only the two sf_data elements): each side track takes the
  transform lengths and window grouping of the track it is paired with
  (A, B; with coding_config 1 the shared one of three_channel_data), and
  each of the two chparam_info() is read with that track's own max_sfb, as
  the channel_pair_element reads its chparam_info with max_sfb and not
  max_sfb_side (Pseudocode 5, b_side_channel 0). The largest signalled
  transform length (n_side_bits, Tables B.8 - B.19) is taken over the
  sf_info of A and B. The side processing of 5.3.4.3.2 is applied per
  transform window after ungrouping (the two tracks have different
  max_sfb, so different packed layouts); the result is the same per tile.
* **The matrices of Tables 178 and 179** are cascades of the 2 x 2 stereo
  operation (one per chparam_info); `ac4/multichannel.py` builds them so,
  and tests/test_phase_b2.py checks every chel_matsel of Table 178 and
  five of Table 179 (and 5.3.3.4) entry by entry against the tables.

**Phase A**

* **The QMF tools' time grid: `TS_OFFSET` = 24 - `ts_offset_hfgen` = 18.**
  Frame f's A-SPX interval slot ts (and its A-CPL slots) act on QMF
  analysis slot 24 f + 18 + ts of the decoded core. Pseudocode 75
  (Q_low[ts] = Q[ts - ts_offset_hfgen], 6 slots for frame lengths 1536 to
  2048, Table 192) and 5.7.6.5 (Qout(i) = Qin(i - 6) + Y(i)), read
  literally, give 24 f - 6 + ts; the decoder used 24 f + 4 + ts (the
  analysis filter's measured delay). On air the data fit the literal reading
  one frame later: the envelope adjuster's E_ref / E_est, which jumps from
  envelope to envelope wherever envelopes and core are misaligned, swings
  least at offsets 16-18 on broadcaster B's stereo and 5.1 and a public
  stereo test stream (mean |step| 1.76 / 2.44 / 1.48 dB at 16-18 against
  2.32 / 3.47 / 2.15 dB at -6; broadcaster A's 5.1 is nearly flat), and the
  A-SPX transient borders led the core's transients by 12-13 slots at the
  old offset 4 (i.e. 16-17). The decoder's IMDCT output timing is the
  standard's (block f's left half is output frame f, Pseudocode 64), so the
  one frame is not a filterbank delay.
* **Decoder scale.** Every block's IMDCT is the standard's (1/N) * sum
  times `IMDCT_K` = sqrt(2 * 1536); A-SPX envelopes, absolute energies on
  the standard's scale, take `IMDCT_K`^2. Against the loudness metadata of
  five streams (BS.1770 integrated loudness of the decode against
  `loudrelgat`), full scale is 122.1 dB re 1 decoder unit (spread
  +-1.8 dB), close to `IMDCT_K` * 2^14.5 (122.2 dB): the standard's own
  arithmetic (dequantisation, IMDCT) puts full scale near 2^14.5. The
  loudness stage (`loudness.FULL_SCALE`) uses `IMDCT_K` * 2^14.5.
* **Companding scale (5.7.5).** The gain L^((1 - alpha) / alpha) *
  2^(1 / alpha) depends on the absolute scale of the levels L, which the
  standard does not give. Levels are taken relative to `IMDCT_K` * 2^16.5
  decoder units (4 x the full scale above): against the public stereo
  test signal's AAC version (same programme, decoded by FFmpeg, aligned)
  this gives the best SNR (13.2 / 14.5 dB at 29.97 / 25 fps, against 10.9 /
  11.5 off) and per-frame level deviation (p95 0.37 / 0.31 dB, against
  3.0 off); scales x1 to x64 were swept. The tool runs frame by frame in
  the A-SPX loop (each frame's interval before its HF generation). The
  g_avg formula's exponent prints as "1alpha / alpha"; it is read as
  (1 - alpha) / alpha.
* **Loudness and DRC** (5.7.9). The level detector (left to the
  implementation by 5.7.9.3.1.1) is the BS.1770 K-weighted mean square of
  one QMF slot (64 samples at the internal rate; the 48 kHz filter at every
  rate) summed over channels with BS.1770 weights (LFE 0, surrounds 1.41),
  -0.691 + 10 log10, minus dialnorm. Curve gains are 10^(G/20) dB, the
  normalisation and transmitted gains 2^(G/6) (dB2). DRC acts as one
  wideband gain per QMF slot on the output samples (the standard: on QMF
  samples before synthesis); multi-band transmitted gains are applied as
  the mean of their bands in dB2. The time constants of Table 162 apply to
  the default profiles, with their fast-attack/release thresholds.
  Frames concealed keep the last dialnorm and DRC frame. Mode selection:
  the largest mode ID whose (Lout,min, Lout,max) holds the target.
* **Rates.** The stream's rate is learnt from the first decodable frame
  whose successor agrees; frames before that are taken as 1536 samples.
  A later change of rate is concealed (`toc.rate`). The A-SPX/A-CPL time
  grid generalises the 29.97 fps reading below: `TS_OFFSET` = nts -
  `ts_offset_hfgen` (18 at 1536, 26 at 2048).
* **A-SPX additional readings (phase B).** The QMF slots of an envelope
  estimate are averaged per QMF slot (the standard writes per A-SPX slot;
  the same at num_ts_in_ats 1, measured the better fit at 2). The sine
  index of a frame is read per envelope as Pseudocode 92 builds it, the
  areas of Pseudocodes 93-95 then follow it.
* **QMF synthesis phase.** 5.7.4.2 prints exp(j pi (k + 1/2)(2n - 257)
  / 128); both decoders use 2n - 255, which reconstructs the analysis bank
  to 78 dB (delay 577) where 257 gives 43 dB (delay 578).
* **`ac4_presentation_substream_info()`** closes the else-branch of
  `ac4_presentation_v1_info()` (TS 103 190-2 6.2.1.3; the printed syntax
  loses a line at the page break).
* **Presentation choice** (TS 103 190-2 4.8.2 leaves it to the
  application): the first enabled presentation with a channel-coded audio
  substream, preferring the `--language` given (primary subtag), then
  "complete main" content (Table 91) over audio description, commentary
  and the rest. `md_compat` is reported, not used. Of the chosen
  presentation the first channel-coded substream is decoded; several
  substream groups (M&E + dialogue, main + associated) are not mixed
  (6.2.16). The choice is made per frame, so it follows a TOC that changes.
* **Sync frames.** A frame whose CRC fails is passed on empty (concealed,
  timeline kept). After lost sync a frame is accepted when its CRC checks
  or, for 0xAC40 frames, when the next sync word (or an ID3 tag, or the end)
  is where its frame_size says; the skipped bytes' frames are lost (the
  decoder sees a sequence-counter jump).
* **Metadata of 7_X and above** (`basic_metadata`): ch_mode 5-6 and 11-15
  are read as 3/4/0 (`pre_upmixtyp_3_4`), 9-10 as 3/2/2, 7-8 as 5/2/0.
  On the public immersive-stereo stream (signalled 7.1, sus_ver 1) this
  reading gives tools_metadata_size 144 bits whose first bit
  (b_de_data_present) is 0: the block does not close, so its dialogue
  enhancement data is not read. Phase C (immersive) will look at it.

**A-CPL (clause 5.7.7, ASPX_ACPL_3)**

* **Time grid.** The standard gives A-CPL the A-SPX output matrix of the
  frame (6.2.11); here frame f's A-CPL acts on the same QMF slots as frame
  f's A-SPX: 24 f + 18 .. 24 f + 41 (`aspx_render.TS_OFFSET`, see below).
  Checked on air: the two steep-interpolation frames of broadcaster B's 5.1
  that fall in a frame with an A-SPX transient change their parameters at
  slot 14 and 13, where A-SPX puts its transient border (slots 14 and 12).
* **Transient ducker** (Pseudocodes 112-114). Run once per QMF time slot
  (Pseudocode 113 takes one subsample per band; "the previous frame" in
  the text is read as the previous slot: the decay ALPHA then has a time
  constant of 5 ms, against 120 ms per frame, too slow for transients),
  over the 15-band partition of Table 197
  (the loop runs to acpl_max_num_param_bands = 15, whatever the stream's
  band count), on the energy of the ducker's own input, the decorrelator
  output (Pseudocodes 113 and 114 use the same x; "the input channel").
* **Cold start.** acpl_param_prev is 0 "when decoding the first AC-4
  frame". This decoder plays the stereo core in L/R before the stream's
  first acpl_config_2ch (it cannot upmix without it), so at the first
  frame with A-CPL data (and after any frame without) the previous
  parameters are taken equal to that frame's first set: the upmix starts
  without a fade from silence. Decorrelator and ducker states start at 0
  as the standard says.
* **Frames without A-CPL data** (before the first I-frame): L/R are the
  core (through the same QMF path), C/Ls/Rs silent, the tool's state
  cleared. A **concealed frame** holds the previous parameters (its core
  is already faded out by the concealment) and runs the decorrelators on.
* **Out-of-range values.** After time-differential decoding (Pseudocode
  121) a value can leave what the F0 codebook can code and what Tables
  203-206 cover; on air this happens only after frames lost on air (33
  frames after a 4-frame gap on broadcaster B, until each parameter's
  next frequency-coded set). Such a value is clamped to the range at
  dequantisation; the q value itself stays the reference for the next
  frame. It is not treated as damage (it was not, there).
* **Level checks.** The burst guard and `aspx.level` judge the A-SPX output
  of the core pair before the upmix, as for every other stream; a frame
  they replace plays the core (the stereo downmix) on L/R and nothing on
  C/Ls/Rs.
* With `channels=2` (`--channels 2`) an ASPX_ACPL_3 stream gives its stereo
  core, without the upmix.

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

**Inputs and presentations.** `include/ac4dec_input.h` (`ac4in_read`)
reads what `ac4/container.py` reads (MP4 with or without fragments, AC-4
sync frames with their CRC, MPEG-TS), frame for frame
(`c/tests/check_inputs.py`, clean and damaged inputs); `ac4dec_config`
takes `presentation`, `presentation_id` and `language`, and
`ac4dec_presentations()` lists a frame's presentations. `ac4dec-c IN
--list`, `--presentation`, `--language` as the Python CLI.

**Phase A figures** (2026-11-23): against the Python on the 12 distinct
captures, largest difference 4.19e-15 of full scale, 16-bit WAVs
identical, same per-frame statuses; on the 7 containers of a public
29.97 fps stereo test signal (sync frames, TS, MP4 with and without
fragments, muxed with video) 4.45e-15, nothing concealed. Damaged input
(`robust_compare.py`, 7 cuts, 12 kinds x 20 seeds): the C at the default
lookahead gives the Python's figures exactly: 11,046 damaged frames, 8,191
concealed, 984 identical, 1,868 bounded, 3 LOUD, 127 undamaged frames
concealed, worst peak 4.42x, 3 failing decodes (6 gate messages); at
lookahead 8: 5 LOUD, 161 collateral, 5 failing decodes (10 messages). This
is somewhat worse than before phase A (2 LOUD, 75 collateral, 4 messages):
the high bands now sit at their true level and the centre is on the QMF
path, so damage there reaches further. On the clean cuts nothing is
concealed and the guard changes nothing. `make asan` passes (also on two
capture cuts).

**Status.** It decodes what this package decodes (above), with the same
damaged-input rules and output guard, as a delay line: frames come out
`lookahead` frames after they go in (default 180, 6 s; minimum 8 for live
use). Against the Python on all 14 captures at hand (both broadcasters;
stereo, 5.1 codec mode 1, ASPX_ACPL_3 with the A-CPL upmix), everything
on: largest difference 4.7e-15 of full scale (1.75e-15 on the upmixed
5.1), 16-bit WAVs byte-identical, the same per-frame statuses
(`c/tests/compare.py`). `c/tests/check_acpl.py` holds the C A-CPL to the
Python on the branches the captures do not reach (12/9/7 bands, coarse
quantisation, two parameter sets, clamped values, concealed frames):
5.2e-16 of peak on 480 random frames. `c/tests/acpl_check.py` gives the
sanity measurements of the upmix (see A-CPL below). On damaged input it gives the Python's
figures exactly at the default lookahead (`c/tests/robust_compare.py`); at
the minimum lookahead the guard seeds from fewer frames: 4 gate messages (2 decodes),
168 undamaged frames concealed. `c/tests/check_noise_fill.py`,
`late_check.py` and `align_check.py` check the noise fill, the late
`audio_size` check and the channel alignment against the Python. The C
streams and has no decode passes; it equals a one-pass Python decode.

**Speed** (2026-11-23, `c/tests/speed.py`, one thread, best of 2, decode
time; Python with its C kernels built):

| stream | length | Python | C |
|---|---|---|---|
| 5.1, codec mode 1 (A-SPX), broadcaster A | 10 s | 3.6x | 20.0x |
| stereo, broadcaster A | 10 s | 9.1x | 55.6x |
| 5.1 sent as ASPX_ACPL_3 (A-CPL upmix), broadcaster B | 42 s | 3.5x | 25.1x |
| stereo, broadcaster B | 42 s | 9.1x | 60.1x |

(Re-measured after phase A. Broadcaster A's 5.1 is 15 % slower: the
centre now goes through the A-SPX renderer, as a pair of the same
channel, i.e. two more QMF channels.)

The upmix halves the speed on broadcaster B's 5.1 (7.5x / 51.7x with its
stereo core only): half the C's time is now QMF synthesis, of seven
channels (the five upmixed ones, and the core pair before the upmix,
which the level checks judge); the A-CPL itself is 12 %. A and B stereo
are unchanged within the spread between runs.

## What it does not do yet

* **frame_rate_factor / frame_rate_fraction above 1** (high frame rates,
  efficient HFR) and 96/192 kHz substreams: concealed as
  `toc.unsupported_rate`.
* **Dialogue enhancement** is parsed, not applied (its default user level
  0 leaves the signal unchanged); DRC gains act on the output samples per
  QMF slot, not on QMF samples (see Interpretations).
* **Metadata of the public 5.1.4 and immersive-stereo streams**: their
  audio is an immersive_channel_element (phase C), so the substream
  metadata after it is not reached; on the immersive-stereo stream, which
  signals 7.1, `basic_metadata` reads but its tools block does not close
  (see Interpretations).
* Mixing the substream groups of one presentation (M&E + dialogue, main +
  associated): only its first channel-coded substream is decoded.
* The SSF speech frontend; 7_X in ASPX_ACPL_1 / _2; immersive and
  object audio (A-JOC, IFM, OAMD), and the IMS rendering to 7.1;
  bitstream versions 0 and 1.

## Survey

`tools/survey.py FILE ...` parses every frame of any input and reports the
TOC, presentations, substream groups, each audio substream's element,
codec mode, coding_config, companding, stereo processing, short blocks,
noise fill and A-SPX tools, and the metadata. On the 7 distinct capture
streams of both broadcasters and the 18 distinct public test signals
(2026-11-23):

| | broadcasters A and B | public test signals |
|---|---|---|
| TOC | v2, 48 kHz, 29.97 fps, 1 presentation (single group), 2 substreams (presentation substream + audio); B sets payload_base | the same, at 25 and 29.97 fps |
| channel modes, elements | 5.1: 5_X ASPX (A), 5_X ASPX_ACPL_3 (B); stereo: channel_pair ASPX (A, B), SIMPLE (A) | stereo: channel_pair ASPX; 5.1: 5_X ASPX_ACPL_2, coding_config 0; 7.1.4 (immersive_channel_element, 5.1.4 content); "IMS": presentation_version 2, 7.1 (3/4/0) 7_X ASPX |
| I-frames | global I-frames only; 144 / 72 / 30 frames apart | a substream I-frame (b_audio_ndot) halfway between global ones |
| companding | B stereo: on in 99 % of frames; A stereo: on; both 5.1: off | stereo: on or averaged in every frame |
| A-SPX tools | TNA everywhere; additional harmonics rare | additional harmonics in most frames; FIC/TIC in the 25 fps stereo |
| stereo processing | sap_mode 1-3, short blocks | the same |
| dialnorm | -24 dBFS | -23 (2.0, 5.1), -19 (7.1.4, IMS), -16 dBFS |
| DRC | 4 default modes (home theatre, flat panel TV, portable speakers, headphones), all "default profile", E-AC-3 profile "none" | default profiles and one transmitted compression curve; E-AC-3 profiles film standard / film light / music light; no transmitted gains anywhere |
| dialogue enhancement | method 0 (channel independent), max gain 9 dB, on the centre (5.1) or L/R (stereo) | method 0; IMS also method 2 (hybrid) |
| EMDF payloads | none | payload id 18 in the IMS streams |
| loudness info | loudrelgat, loudspchgat, truepk, loudstrm3s, rtll_comp | the same |

## Damaged input

AC-4 in MP4 carries no frame CRC, and every Huffman codebook is a complete
prefix code, so random bits always "decode". The decoder therefore checks
what the bitstream structure fixes, conceals what fails, and keeps a level
guard as a last resort (`ac4/robust.py` lists every rule):

* **Bitstream checks.** The table of contents (substreams fit the frame,
  sequence_counter <= 1020, layout and rates as in the stream so far); `audio_size` closure: the parse of `audio_data()` must end 0..7
  bits before `audio_size` says (byte alignment), which holds on every
  clean frame measured (uniform over 0..7). That covers the core, all A-SPX
  blocks (centre included) and, for ASPX_ACPL_3, `acpl_data_2ch()`; section codebooks 12-15,
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
stream's peak; 2 were LOUD. 75 undamaged frames were concealed in all (at
most 4 per hit allowed; 27 before the A-CPL upmix, measured the same day
on the same cuts: the 48 more are on broadcaster B's 5.1 cuts, whose
frames 120-239 are now upmixed, every one within 3 frames of a damaged
frame, muted by the guard's near-damage rule). Worst peak 4.45x the clean
stream's own peak (byte bursts; 1.96x for runs, on B's upmixed 5.1, a
damaged frame inside a run that passed the checks and stayed under the
near-damage limit of 2x; 1.48x before the upmix; 1.21x or less for every
other kind): two
isolated burst frames early in broadcaster A's 5.1 cut, under the
guard's start-level limit (see below), are the only failing decodes (2,
with 4 gate messages; 3 decodes, 5 messages before the 2026-11-23 fixes).
On the clean cuts nothing is concealed and the guard changes no sample.
The C decoder gives exactly the same figures at its default lookahead
(at the minimum lookahead, 8: 2 LOUD, 4 gate messages, 168 collateral;
before the upmix 4, 8 and 126).
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
  is a heuristic carried over from the receiver. It used to fire on clean
  material (31 frames of one broadcaster's 42 s stereo); those were the
  1.5 dB envelopes dequantised as 3 dB (fixed), and it now fires on no
  capture. It still fires on 7 frames of a public stereo test stream,
  which is louder than the fixed reference.
* **Broadcaster A's stereo** keeps a 34 dB step down at its A-SPX
  crossover after the envelope fixes (its 5.1 and broadcaster B do not):
  its high band is still far too quiet. Not found yet.
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
* `channels.asf_scalefac_data` and `channels.asf_snf_data` refer to an
  undefined name (`NUM_SFB_48`). Nothing calls them; they are kept as copied.

## Tests

Phase A added `tests/test_phase_a.py` (stereo_data, the substream I-frame
flag, the TOC's presentations, groups and indices written bit by bit from
TS 103 190-2, presentation choice, envelope dequantisation, the time grid,
the centre's A-SPX, other rates) and `tests/test_inputs.py` (the CRC's
check value, sync frames with escape, bad CRC, resync and ID3, an MPEG-TS
built packet by packet, a non-fragmented MP4 with a `dac4`, and the same
inputs through the C reader when it is built); `tests/test_units.py`
holds the IMDCT against Pseudocodes 60-63 written out. `tools/survey.py`
reports what a stream uses (see Survey below).

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
standard's pseudocode, the noise index across envelopes, the transient envelope, no HF patch
without A-SPX data, all six channels time-aligned, the late check on an
already concealed frame). `tests/test_acpl.py` checks A-CPL against the
standard's Pseudocodes 109-114, 118, 119 and 121 written out sample by
sample (several band counts, both quantisation modes, one and two
parameter sets, smooth and steep interpolation), the dequantisation
tables, the decorrelators as stable all-pass filters, that the upmix sums
back to the core under downmix gammas, frames without data and concealed
frames, and a decode of a stubbed ASPX_ACPL_3 stream to six channels.

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
| `container.py` | input detection; AC-4 sync frames (CRC), MPEG-TS |
| `mp4.py` | MP4 sample reader (fragments and sample tables), `dac4` |
| `metadata.py` | loudness, DRC, dialogue enhancement and EMDF metadata (parsed) |
| `toc.py`, `toc_v1.py` | table of contents and presentation choice (`toc_v1` holds the bit reader and constants) |
| `hcb.py`, `spectral.py` | Huffman codebooks and decoding |
| `sfb.py` | scale-factor band offsets |
| `channels.py`, `pair.py` | `5_X` channel element; `channel_pair_element` |
| `audio.py` | dequantisation, stereo processing (M/S and full SAP), noise fill |
| `acpl_parse.py` | A-CPL side data: parsing, cb_off, differential decoding |
| `acpl.py` | A-CPL: dequantisation, interpolation, decorrelators, ducker, the ASPX_ACPL_3 upmix |
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
