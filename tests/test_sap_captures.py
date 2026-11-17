# SPDX-License-Identifier: Apache-2.0
"""Full SAP (sap_mode 3) on real captures (see tests/capture_util.py).

    AC4_TEST_DIR=/path/to/captures pytest tests/test_sap_captures.py

Each capture is cut to its first AC4_SAP_TEST_FRAGMENTS fragments (default
10, about 20 s): SAP3 frames are only checkable after the first i-frame,
and the gate against wrong variants needs a few dozen of them. Captures
without SAP3 frames skip.

Gate 1: every SAP3 frame's audio_data() ends where audio_size says.
Gate 2: on SAP3 frames the decoded image follows the A-SPX envelopes (an
        encoder-side measurement of the original) with the right sign and
        scale, and wrong variants score clearly worse (tests/sap_gate.py).
Gate 3: everything that is not a SAP3 channel pair decodes exactly as
        before; byte identity of whole SAP3-free streams against the
        original receiver is test_integration's reference test.
"""
import os

import numpy as np
import pytest

import capture_util as CU
import sap_gate as G
from ac4 import audio, channels, frameinfo, mp4
from ac4.decoder import probe_element
from ac4.stream import Ac4Stream

CAPTURES = CU.capture_files()
N_FRAG = int(os.environ.get("AC4_SAP_TEST_FRAGMENTS", "10"))

pytestmark = pytest.mark.skipif(
    not CAPTURES, reason="set AC4_TEST_DIR to a folder of AC-4 fMP4 captures")


@pytest.fixture(scope="module", params=CAPTURES or [None],
                ids=[f"capture{i}" for i in range(len(CAPTURES))] or ["none"])
def lane(request, tmp_path_factory):
    dst = tmp_path_factory.mktemp("sap") / "in.m4s"
    if CU.cut_fragments(request.param, str(dst), N_FRAG) == 0:
        pytest.skip("capture has no complete fragment")
    frames = mp4.read_frames(str(dst))
    element = probe_element(frames)
    info = frameinfo.walk(frames, element)
    if not any(3 in r["sap"] for r in info):
        pytest.skip("no sap_mode 3 frames in this cut")
    return frames, element, info


# ------------------------------------------------------------- gate 1
def test_sap3_audio_data_closes_to_audio_size(lane):
    frames, element, info = lane
    sap3 = [r for r in info if 3 in r["sap"]]
    walked = [r for r in sap3 if r["status"] == "closed"]
    if not walked:
        pytest.skip("no SAP3 frame after the first i-frame")
    assert not [r for r in sap3 if r["status"] == "error"]
    bad = [r["slack"] for r in walked if not 0 <= r["slack"] <= 7]
    assert not bad, f"{len(bad)} of {len(walked)} SAP3 frames: slack {bad[:8]}"


def test_sap3_closure_fails_for_a_wrong_parse(lane, monkeypatch):
    """Negative control for gate 1: reading one coefficient per band (not
    per band pair) must leave SAP3 frames that no longer close."""
    frames, element, info = lane
    good = channels.sap_data

    def wrong(b, fr, sf_table):
        sap = good(b, fr, sf_table)
        for g, m in enumerate(sap["max_sfb"]):
            for sfb in range(1, m, 2):
                if sap["coeff_used"][g][sfb]:
                    sf_table.decode(b)
        return sap
    monkeypatch.setattr(channels, "sap_data", wrong)
    again = frameinfo.walk(frames, element)
    pairs = [(a, b) for a, b in zip(info, again)
             if 3 in a["sap"] and a["status"] == "closed"]
    if not pairs:
        pytest.skip("no SAP3 frame after the first i-frame")
    broken = [b for _a, b in pairs
              if b["status"] != "closed" or not 0 <= b["slack"] <= 7]
    assert len(broken) >= max(1, len(pairs) // 2)


# ------------------------------------------------------------- gate 2
MIN_REF = 40          # SAP3 frames with an A-SPX reference, per pair


def _pairs(element, info):
    if element == "pair" or any(r.get("codec_mode") == 4 for r in info):
        return [0]
    return [0, 1]


def test_sap3_image_follows_aspx_and_wrong_variants_fail(lane):
    frames, element, info = lane
    checked = 0
    for p in _pairs(element, info):
        n3 = sum(len(r["sap"]) > p and r["sap"][p] == 3 for r in info)
        if n3 < MIN_REF:
            continue
        rep = G.report(frames, element, p)
        spec = rep["spec"]
        if (spec["aspx_ref_n"] or 0) < MIN_REF:
            continue
        checked += 1
        msg = {v: {k: rep[v][k] for k in ("continuity_ild", "aspx_ref_r",
                                          "aspx_ref_scale")} for v in rep}
        # spec: the core image just below the crossover moves with the
        # A-SPX image just above it (measured r 0.61 to 0.88)
        assert spec["aspx_ref_r"] >= 0.4, msg
        # sign flip: anti-correlated (measured -0.56 to -0.89)
        assert rep["sign_flip"]["aspx_ref_r"] <= -0.3, msg
        # alpha ignored: the image is gone (measured -0.04 to 0.20)
        assert rep["ms_only"]["aspx_ref_r"] <= spec["aspx_ref_r"] - 0.3, msg
        # transposed: right sign, half the image (measured scale ratio
        # 1.95 to 2.2)
        assert spec["aspx_ref_scale"] >= 1.6 * rep["transposed"][
            "aspx_ref_scale"] > 0, msg
        # passthrough: L/R steps at every switch (measured 3.6x to 12x spec)
        assert rep["passthrough"]["continuity_ild"] >= 3 * spec[
            "continuity_ild"], msg
        # spec itself: switches no worse than 2.5x ordinary boundaries
        # (measured 1.29 to 1.84; see sap_gate for why it is not 1.0)
        assert spec["continuity_ild"] <= 2.5, msg
    if not checked:
        pytest.skip(f"fewer than {MIN_REF} SAP3 frames with A-SPX per pair")


# ------------------------------------------------------------- gate 3
def test_only_sap3_pairs_change(lane, monkeypatch):
    """Spectra of every channel pair that is not sap_mode 3 are identical
    to the old pass-through decoding; SAP3 pairs differ."""
    frames, element, info = lane
    pairs = _pairs(element, info)
    keys = [("L", "R"), ("Ls", "Rs")]
    chans = tuple(k for p in pairs for k in keys[p])

    def spectra(fn):
        monkeypatch.setattr(audio, "sap_coeffs", fn)
        s = Ac4Stream(element=element)
        return [s.decode_frames([f], chans)[0] for f in frames]
    new = spectra(G.GOOD)
    old = spectra(G.VARIANTS["passthrough"])
    changed = 0
    for r, a, b in zip(info, new, old):
        for p in pairs:
            same = all(np.array_equal(x, y) for k in keys[p]
                       for x, y in zip(a[k], b[k]))
            if len(r["sap"]) > p and r["sap"][p] == 3:
                changed += not same
            else:
                assert same
    assert changed > 0
