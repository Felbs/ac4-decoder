# SPDX-License-Identifier: Apache-2.0
"""Integration tests on real AC-4 captures, which are never part of this
repository (see tests/capture_util.py).

    AC4_TEST_DIR=/path/to/captures pytest tests/test_integration.py
    AC4_REFERENCE_REPO=/path/to/receiver  ... also compares against the
                                              code this was copied from

Each capture is cut to its first AC4_TEST_FRAGMENTS fragments (default 2,
about 4 s of audio) so a run stays short.
"""
import os
import shutil

import numpy as np
import pytest

import capture_util as CU
from ac4 import cli, mp4
from ac4.decoder import Decoder, probe_element

CAPTURES = CU.capture_files()
N_FRAG = int(os.environ.get("AC4_TEST_FRAGMENTS", "2"))

pytestmark = pytest.mark.skipif(
    not CAPTURES, reason="set AC4_TEST_DIR to a folder of AC-4 fMP4 captures")


def _ids():
    return [f"capture{i}" for i in range(len(CAPTURES))]   # no file names


@pytest.fixture(params=CAPTURES or [None], ids=_ids() or ["none"])
def cut(request, tmp_path):
    dst = tmp_path / "in.m4s"
    if CU.cut_fragments(request.param, str(dst), N_FRAG) == 0:
        pytest.skip("capture has no complete fragment")
    return str(dst)


def test_decodes_every_frame(cut):
    frames = mp4.read_frames(cut)
    assert frames
    element = probe_element(frames)
    assert element in ("5_X", "pair")
    dec = Decoder(element)
    y = dec.decode_pass(frames)
    assert dec.n_bad == 0
    assert y.shape == (len(frames) * 1536, len(dec.chans))
    assert np.isfinite(y).all()
    assert float(np.sqrt((y[:, :2] ** 2).mean())) > 0.0      # not silence


@pytest.mark.xfail(strict=False, reason=(
    "KNOWN, inherited unchanged from the original: on some streams the "
    "output depends on where decode passes are cut -- audio before the "
    "first A-SPX config is delayed 577 samples only when a later frame of "
    "the SAME pass carries the config, and the last frame of a pass "
    "differs from a one-pass decode in its last 1279 samples. Measured on "
    "a 4 s cut of a stereo-pair stream; see README 'Known issues'."))
def test_output_is_independent_of_pass_size(cut):
    frames = mp4.read_frames(cut)
    element = probe_element(frames)
    one = Decoder(element).decode_pass(frames)
    d = Decoder(element)
    parts = [d.decode_pass(frames[i:i + 17]) for i in range(0, len(frames), 17)]
    assert np.array_equal(one, np.vstack(parts))


def test_cli_wav_length(cut, tmp_path):
    out = str(tmp_path / "out.wav")
    assert cli.main([cut, out, "-q"]) == 0
    n_ch, rate, raw = CU.wav_data(out)
    n = len(mp4.read_frames(cut))
    assert rate == 48000
    # the stream resampler returns ceil(n_in * 1001 / 960) samples
    assert len(raw) // (2 * n_ch) == -(-n * 1536 * 1001 // 960)


@pytest.mark.parametrize("max_pass", [1800, 40])
def test_identical_to_reference_worker(cut, tmp_path, max_pass):
    """The copy changed nothing: the receiver's production audio worker
    and this package's CLI write byte-identical PCM for the same input,
    with one decode pass and with many."""
    repo = CU.reference_repo()
    if not repo:
        pytest.skip("set AC4_REFERENCE_REPO to compare with the original")
    element = probe_element(mp4.read_frames(cut))
    channels = 6 if element == "5_X" else 2
    wdir = tmp_path / "lane"
    wdir.mkdir()
    lane = str(wdir / "in.m4s")
    shutil.copy(cut, lane)
    ref = str(tmp_path / "ref.wav")
    rc, log = CU.run_reference_worker(repo, lane, ref, element, channels,
                                      max_pass)
    assert rc == 0, log
    assert "pad 0," in log, log                # no receiver-side padding
    new = str(tmp_path / "new.wav")
    assert cli.main([cut, new, "--element", element, "--channels",
                     str(channels), "--max-pass", str(max_pass), "-q"]) == 0
    a, b = CU.wav_data(ref), CU.wav_data(new)
    assert a[:2] == b[:2]
    assert len(a[2]) == len(b[2])
    assert a[2] == b[2]
