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

Measured in the original receiver on 10 s cuts (single core, C kernels
built): 0 bad frames on every stream; 5.1 in six channels at about 3x real
time, stereo elements at about 6-9x. This package reproduces that receiver's
output byte for byte (see Tests). Top of spectrum with A-SPX, on the two
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

## Known issues

* **Output level.** The decoder's true full scale is not established. The
  CLI keeps the original receiver's level rule: a fixed reference per
  element (9e5 for 5.1, 2e5 for stereo, decoder units) written at 0.9 of
  full scale, rising only when a louder sample appears, capped at twice the
  start. `decode()` returns the decoder's own scale.
* **Damaged frames.** AC-4 in MP4 has no frame CRC. In a test that damaged
  21 of 300 frames, 14 were caught and replaced by silence of the right
  length; 7 decoded as if valid, with peaks up to 35x the clean peak (5.1)
  and far more on a stereo stream. The burst guard (an A-SPX frame above a
  limit plays its core band) is a heuristic carried over from the receiver.
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
comparison test checks byte for byte.

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
