# ac4-decoder

An AC-4 audio decoder written from the published standard (ETSI TS 103 190-1
and -2), in Python and NumPy, with two optional C kernels for speed.

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

Everything below was measured on off-air captures from two US ATSC 3.0
broadcasters using different encoder configurations. No captures are in this
repository.

| | |
|---|---|
| Bitstream | version 2, 48 kHz, 29.97 fps, one presentation, channel-coded |
| Input | fragmented MP4 (init segment + `moof`/`mdat` fragments, or bare fragments) |
| 5.1 (`5_X` channel element, coding_config 0) | codec mode 1 (A-SPX): all six channels. Codec mode 4 (ASPX_ACPL_3): the two-channel core only (see below) |
| Stereo (`channel_pair_element`) | stereo_codec_mode 0 (SIMPLE) and 1 (A-SPX) |
| Core (ASF/MDCT) | Huffman spectral data, scale factors, long and short frames with block switching, KBD windows, stereo processing (sap_mode 1 and 2: M/S; sap_mode 3: full SAP with prediction coefficients), noise fill, LFE |
| A-SPX | parsed and rendered on the L/R and Ls/Rs pairs: 64-band complex QMF, envelope decoding, HF generation, envelope adjustment, joint-balance rendering |
| Output | 48 kHz via the 1001/960 resampler (TS 103 190-1 Table 83); without it audio runs 4.27 % fast |
| I-frames | decoding can start mid-stream; A-SPX starts at the first I-frame |
| Damaged input | never raises, never changes the output length; damaged frames are concealed (see below) |

Measured in the original receiver on 10 s cuts (single core, C kernels
built): 0 bad frames on every stream; 5.1 in six channels at about 3x real
time, stereo elements at about 6-9x. This package reproduces that receiver's
output byte for byte on the test cuts (see Tests), with one deliberate
exception, an A-SPX parsing fix (see Damaged input). Top of spectrum with A-SPX, on the two
broadcasters' streams: about 12.3 kHz on one 5.1 stream's front pair, 16.6 to
18 kHz on the other broadcaster's stereo streams; without A-SPX the core
stops near 12 kHz.

### Full SAP (sap_mode 3)

Stereo audio processing mode 3 sends, per pair of scale factor bands, a
prediction coefficient alpha (DPCM-coded in frequency, or in time across
short-block groups) and decodes the two tracks as
`L = (1 + alpha) M + S`, `R = (1 - alpha) M - S` (TS 103 190-1 Table 48,
Pseudocode 59, clause 5.3.3.2). It is common on air: 26 % of one
broadcaster's 5.1 frames and 34 % of the other's (18 minutes measured)
use it on at least one channel pair. How it was checked, on captures:

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
18 minutes, none on the other broadcaster).

## What it does not do yet

* **A-CPL.** An ASPX_ACPL_3 5.1 stream decodes to its stereo core (the coded
  downmix the broadcaster sent); C, LFE, Ls and Rs are written silent.
  ASPX_ACPL_1/2 and stereo_codec_mode 2/3 raise and count as bad frames.
* **A-SPX on the centre channel.** Its block is parsed (to keep state) but
  not rendered, so the centre stops near 12 kHz.
* Companding (read, not applied); A-SPX additional harmonics (under 1.5 % of
  channel-frames); the SSF speech frontend; coding_config other than 0.
* Mono, 3.0, 5.0 and 7.x elements; immersive and object audio (A-JOC, IFM,
  OAMD).
* Bitstream versions 0 and 1; frame rates other than 29.97; sample rates
  other than 48 kHz.
* Choosing a presentation or substream: substream 1 is always decoded.
* DRC, dialogue enhancement and loudness metadata.
* Raw AC-4 sync frames, MPEG-TS, non-fragmented MP4, the `dac4` box.
* A-CPL synthesis: `acpl_data_2ch()` is parsed, for bit accounting only.

## Damaged input

AC-4 in MP4 carries no frame CRC, and every Huffman codebook is a complete
prefix code, so random bits always "decode". The decoder therefore checks
what the bitstream structure fixes, conceals what fails, and keeps a level
guard as a last resort (`ac4/robust.py` lists every rule):

* **Bitstream checks.** The table of contents (substreams fit the frame,
  sequence_counter <= 1020, layout and rates as in the stream's first
  frame); `audio_size` closure: the parse of `audio_data()` must end 0..7
  bits before `audio_size` says (byte alignment), which holds on every
  clean frame measured (uniform over 0..7). That covers the core, all A-SPX
  blocks (centre included) and, for ASPX_ACPL_3, `acpl_data_2ch()`, which
  is now read for bit accounting (not synthesised); section codebooks 12-15,
  sections that overrun `max_sfb`, `max_sfb` beyond the band table,
  quantised lines above 8191, scale factors outside 0..255; A-SPX
  envelope counts (Table 127) and borders; a codec mode change outside an
  I-frame; and a sanity bound on dequantised lines (1e9, 37 dB above the
  largest line measured on air, 1.4e7). Frames from before a stream's
  first `aspx_config` cannot be sized when they arrive; once an I-frame
  later in the same decode pass brings the configuration, they are
  re-checked with it, if at least 90 % of them close.
* **Sequence counter.** A frame whose counter breaks sequence while the
  NEXT frame continues from the one BEFORE it is an outlier (swapped,
  repeated or foreign) and is concealed. Other jumps (frames lost on air)
  decode normally and are counted (`n_seq_jumps`).
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
  (1) a frame whose A-SPX output RMS is above 16x the RMS of the core it
  was regenerated from plays its core band (`aspx.level`; frames whose
  A-SPX output peaks below 0.03x the element's fixed level are not judged:
  over a near-silent core the ratio means nothing, up to 7.5e6 measured
  on clean frames whose A-SPX peak was below 0.012x); (2) a frame
  whose peak is above 2.5x a running reference (the largest accepted frame
  peak, released by 0.9999 per frame; it starts at the larger of 0.75x
  the element's fixed level and the 5th-largest of the first 30 frame
  peaks) is muted with 256-sample ramps (`guard.level`). Tuned on 136,000 clean
  frames (about 76 min, both broadcasters, 5.1 and stereo), decoded from
  549 different start points, no frame went above 1.76x that reference
  (margin 1.42x, 3.1 dB)
  and none above an A-SPX/core RMS ratio of 7.43 (margin 2.15x), so
  neither fired. Confirmed end to end on 309,000 clean frames (about
  172 min, including six 16-minute files not used for tuning): nothing
  concealed, no level check fired. `Decoder(guard=False)` or `ac4dec --no-guard` turns both
  off.
* **MP4.** A `trun` whose sample count cannot fit its box is ignored; a
  sample whose size or offset points outside the `mdat` is passed on empty
  (and concealed), so the sample count, and the timeline, is kept. A
  fragment whose boxes are destroyed is still lost as a whole.

`Decoder.n_concealed` counts concealed frames, `Decoder.concealed.reasons`
gives them by reason, and `Decoder.last_concealed` marks them per frame of
the last pass. `ac4dec` prints the count and the reasons.

Measured with tests/test_robust.py on 7 cuts of both broadcasters
(4 fragments each), 10 kinds of damage x 20 seeds x 6 damaged frames
(9,240 damaged frames, a swap damaging two): no exception, exact length
everywhere, worst peak 1.82x the clean stream's own peak (byte bursts;
1.2x or less for every other kind). 6,976 damaged frames were concealed,
959 decoded identical to the clean decode around them and 1,305
differently but within the clean level (840 of those are the second
frame of a swapped pair, played one frame late). With the checks and
level checks off the same gate fails 246 times, worst peak 1.8e11x. Replaying the original survey's
damage (15 frames with 3 bytes inverted, 5 truncated, 1 empty, in 300
frames) on five streams: the old decoder peaked at 2.3e5 to 9.9e9 times
the clean peak on four of them; this one at 1.05x at most, and flags 16
to 20 of the 21 frames (the rest decode within the clean level).

On about 10 hours of one broadcaster's off-air captures (1.1 million
frames, 5.1 and stereo), the bitstream checks (as they stood before the
spectral bound and the late closure check were added) flag 36 frames, every one of them in a
capture file that also shows sequence-counter jumps (frames lost on air);
the files without jumps have none. All the flagged frames that were
checked sit inside clusters of lost frames, so they are inferred to be
real reception damage.

What is still possible: damage that keeps every rule and stays under the
level checks decodes as a click or a wrong frame within the programme's
own level (worst seen: one frame about 40x its local clean level, while
under 2x the stream's peak). A frame from before the stream's first
I-frame, in a pass with no I-frame, gets only the upper-bound closure
check. Early in a stream the output guard works from the element's fixed
level, so on a quiet stream it is looser. The outlier rule needs the next
frame, so it cannot act on the last frame of a decode pass. A fragment
whose MP4 boxes are destroyed is lost, so its frames drop out.

**A-SPX parsing fix (changes output on some clean frames).** With
`aspx_balance = 1` the second channel uses the first channel's framing,
and so (Pseudocode 76, per channel) its `previous_stop_pos` too. The
original parser left it stale, which misread the next VARFIX/VARVAR frame
of that channel: about 1 in 1,300 frames of one broadcaster's 5.1 stream
(26 of 32,700). With the fix every such frame closes exactly on
`audio_size`; without it they fail closure and would be concealed. On a
3-minute 5.1 file the output differs from the original receiver only in
the 5 frames whose framing the fix changes (and the frames their overlap
reaches). `aspx_parse.BALANCE_SHARES_STOP = False` restores the old
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
* **Pass boundaries.** The output is not fully independent of how the frames
  are split into decode passes. Before the first A-SPX configuration
  arrives, frames are delayed by 577 samples only when a later frame of the
  same pass carries the configuration; and the last frame of a pass can
  differ from a one-pass decode in its last 1279 samples (measured up to
  about 1 % of peak on one stereo stream, about -100 dB on the others). The
  CLI's default pass is 1800 frames (60 s).
* `channels.asf_scalefac_data` and `channels.asf_snf_data` refer to an
  undefined name (`NUM_SFB_48`). Nothing calls them; they are kept as copied.

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
synthetic files, and the C kernels against the Python reference.

Integration tests cut each capture to its first two fragments (about 4 s)
and check that every frame decodes, the WAV length is exact, and, when the
original receiver is available, that its production audio worker and this
package's CLI write **byte-identical** PCM with one decode pass and with
40-frame passes.

`tests/test_robust.py` is the damaged-input gate. Without captures it
feeds garbage, empty and truncated frames and damaged MP4 boxes. With
captures it damages each cut deterministically (`tests/inject.py`: bit
flips, byte bursts, truncation, zeroed bytes, swapped, repeated, extended,
empty and garbage frames, frames from another stream;
`AC4_ROBUST_SEEDS` seeds per kind, default 3) and requires: no exception,
exact length, no sample above 2x the clean decode's own peak, and every
damaged frame concealed or harmless (output identical to the clean decode
around it, or no sample there above 2x the clean peak). It also requires
that nothing is concealed, and the guard changes nothing, on the clean
cuts. Its negative control turns the checks and the guard off
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
changed. The decoding arithmetic is unchanged, which the reference
comparison test checks byte for byte. Since then, one parsing fix (A-SPX
`previous_stop_pos` under `aspx_balance`, see Damaged input) changes the
output on the frames it affects; the test cuts contain none of them.

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
