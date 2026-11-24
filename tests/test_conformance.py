# SPDX-License-Identifier: Apache-2.0
"""Expected playback of a set of public AC-4 conformance test streams.

The streams are never part of this repository. Put them anywhere under
AC4_TEST_DIR (tests/capture_util.py); they are recognised by their
SHA-256, not by name, and every test skips when none is found:

    AC4_TEST_DIR=/path/to/test/data pytest tests/test_conformance.py

What is checked comes from the kit's playback test procedure (a listening
test: "channel identifications are reproduced in the correct speakers",
with tables of which voice goes to which output, for native 5.1 and for
the Lo/Ro and Lt/Rt stereo downmixes; language selection; durations),
turned into energy measurements on the decoded output:

  * channel identification: the output is cut into 100 ms windows; a
    window belongs to channel c when c is at least MARGIN dB above every
    other channel. Runs of such windows (>= 3 s for a voice, >= 1.5 s for
    the LFE tone) give the order in which the channels speak, which must
    be L, R for 2.0 and L, C, R, Rs, Ls, then the LFE tone, for 5.1 (the
    procedure does not state the order; this is the order every stream of
    the set uses, and with only one channel speaking at a time any routing
    error changes it);
  * the Lo/Ro and Lt/Rt downmixes route each voice as the procedure's
    table says (Lo/Ro: Ls only left, Rs only right; Lt/Rt: both surrounds
    in both; centre and LFE tone in both, at equal level);
  * the immersive-stereo version (a 7.1 identification rendered to two
    channels by the encoder) speaks left-side channels left, right-side
    right and the centre in both;
  * the two-language stream: the default presentation is English,
    --language fr selects the French one, and each is sample-identical to
    the single-language stream of the same content;
  * every presentation of every stream decodes without a concealed frame,
    for the duration the procedure gives; the multi-presentation streams'
    presentations are sample-identical to the single streams they repeat,
    and their default is the one the procedure names;
  * 25 and 29.97 fps versions of the same content line up in time.

The decoder is c/ac4dec-c when it is built (fast; compare.py holds it equal
to the Python decoder), else the Python package; AC4_CONFORMANCE_DECODER=py
forces the Python one.
"""
from __future__ import annotations

import functools
import hashlib
import os
import subprocess
import wave

import numpy as np
import pytest

import capture_util as CU

HERE = os.path.dirname(os.path.abspath(__file__))
C_CLI = os.environ.get("AC4_C_CLI",
                       os.path.join(HERE, "..", "c", "ac4dec-c"))
WIN = 0.1                       # analysis window (s)
MARGIN = 20.0                   # dB over every other channel: "solo"
NAMES = {2: ("L", "R"), 6: ("L", "R", "C", "LFE", "Ls", "Rs")}
ORDER_20 = ["L", "R"]
ORDER_51 = ["L", "C", "R", "Rs", "Ls", "LFE"]

# SHA-256 (first 16 hex digits) -> role. fps: 25 or 29.97; pres: channel
# count of each presentation, in table-of-contents order; seconds: the
# duration the test procedure gives (None: it gives none for this stream);
# lang: the language of each presentation.
KNOWN = {
    "615b2fea66548043": dict(role="id20", fps=25, pres=[2], seconds=72),
    "68fb352c6099c201": dict(role="id20", fps=29.97, pres=[2], seconds=72),
    "265b8de175065dc2": dict(role="id51_en", fps=25, pres=[6], seconds=72,
                             lang=["en"]),
    "47eada21f3b1fc68": dict(role="id51_en", fps=29.97, pres=[6],
                             seconds=72, lang=["en"]),
    "b238d235e7b8aef4": dict(role="id51_fr", fps=25, pres=[6], seconds=72,
                             lang=["fr"]),
    "fcfb0870fa351597": dict(role="id51_fr", fps=29.97, pres=[6],
                             seconds=72, lang=["fr"]),
    "cb0082c598c699e3": dict(role="id_ims", fps=25, pres=[2], seconds=None),
    "54b1bc8b974846b2": dict(role="id_ims", fps=29.97, pres=[2],
                             seconds=None),
    "851a7baf1a333290": dict(role="id_2lang", fps=25, pres=[6, 6],
                             seconds=72, lang=["en", "fr"],
                             same=["id51_en", "id51_fr"]),
    "0a647a96c52f71fd": dict(role="id_2lang", fps=29.97, pres=[6, 6],
                             seconds=72, lang=["en", "fr"],
                             same=["id51_en", "id51_fr"]),
    "ffe923486990ff54": dict(role="prog20", fps=25, pres=[2], seconds=159),
    "49dbf09f5c96d4f7": dict(role="prog20", fps=29.97, pres=[2],
                             seconds=159),
    "3add503ae4a07bdf": dict(role="prog51_128", fps=25, pres=[6],
                             seconds=159),
    "39bc7d38ffcc5734": dict(role="prog51_128", fps=29.97, pres=[6],
                             seconds=159),
    "f7801a880f925d72": dict(role="prog51_192", fps=25, pres=[6],
                             seconds=159),
    "bd03708402a9f979": dict(role="prog51_192", fps=29.97, pres=[6],
                             seconds=159),
    # two bit rates of one 5.1 programme
    "2ff89e36cc8084b6": dict(role="multi_bitrate", fps=25, pres=[6, 6],
                             seconds=159,
                             same=["prog51_128", "prog51_192"]),
    "11e1b65dbd288b9a": dict(role="multi_bitrate", fps=29.97, pres=[6, 6],
                             seconds=159,
                             same=["prog51_128", "prog51_192"]),
    # a 2.0 and a 5.1 version; the procedure names the stereo one the
    # default and gives 159 s, but this elementary stream is 72 s long
    "d8ff5d02e8072d9b": dict(role="multi_layout", fps=25, pres=[2, 6],
                             seconds=None),
    "d459a97cf7eda1a3": dict(role="multi_layout", fps=29.97, pres=[2, 6],
                             seconds=None),
    # 2.0 at 64 kbps, 5.1 at 128 and at 192 kbps
    "902d9e51f878cf66": dict(role="multi_all", fps=25, pres=[2, 6, 6],
                             seconds=159,
                             same=["prog20", "prog51_128", "prog51_192"]),
    "bb5701470d63aebb": dict(role="multi_all", fps=29.97, pres=[2, 6, 6],
                             seconds=159,
                             same=["prog20", "prog51_128", "prog51_192"]),
}


@functools.lru_cache(maxsize=None)
def found():
    """-> {sha16: path} of the known streams under AC4_TEST_DIR."""
    out = {}
    for p in CU.elementary_streams():
        if os.path.getsize(p) > 64 << 20:
            continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        if h in KNOWN and h not in out:
            out[h] = p
    return out


def by_role(role, fps=None):
    return sorted(((h, p) for h, p in found().items()
                   if KNOWN[h]["role"] == role
                   and (fps is None or KNOWN[h]["fps"] == fps)),
                  key=lambda hp: KNOWN[hp[0]]["fps"])


pytestmark = pytest.mark.skipif(
    not found(), reason="no conformance test stream under AC4_TEST_DIR")


def _ids(items):
    return [f"{KNOWN[h]['role']}-{KNOWN[h]['fps']}" for h, _ in items]


def use_c():
    want = os.environ.get("AC4_CONFORMANCE_DECODER", "")
    if want == "py":
        return False
    if os.path.exists(C_CLI):
        return True
    if want == "c":
        pytest.fail(f"AC4_CONFORMANCE_DECODER=c but {C_CLI} is not built")
    return False


class Decoded:
    """What the tests need of one decode, without keeping the PCM: the
    per-window mean squares of every channel (e, shape (k, ch)), the
    per-window mean of L * R (x), the LFE signal when there is one, a
    digest of the PCM, and how many frames were concealed."""

    def __init__(self, pcm, rate, n_bad):
        pcm = np.asarray(pcm, dtype=np.float64)
        self.rate = rate
        self.n, self.ch = pcm.shape
        self.seconds = self.n / rate
        self.n_bad = n_bad
        w = int(rate * WIN)
        k = self.n // w
        blk = pcm[:k * w].reshape(k, w, self.ch)
        self.e = (blk ** 2).mean(axis=1)
        self.x = (blk[:, :, 0] * blk[:, :, 1]).mean(axis=1) \
            if self.ch >= 2 else None
        self.lfe = pcm[:, 3].astype(np.float32) if self.ch == 6 else None
        self.digest = hashlib.sha256(
            np.ascontiguousarray(pcm).tobytes()).hexdigest()


@functools.lru_cache(maxsize=None)
def decode(path, presentation=None, language=None, layout=None,
           stereo_dmx="loro"):
    """-> Decoded, from the C CLI (16-bit WAV, scaled back to +-1) or the
    Python decoder (float, decoder scale)."""
    if use_c():
        work = os.environ.get("AC4_CONFORMANCE_WORK") or None
        import tempfile
        with tempfile.TemporaryDirectory(dir=work) as d:
            wav, st = os.path.join(d, "o.wav"), os.path.join(d, "o.st")
            args = [C_CLI, path, wav, "--status", st, "-q"]
            if presentation is not None:
                args += ["--presentation", str(presentation)]
            if language:
                args += ["--language", language]
            if layout:
                args += ["--layout", layout, "--stereo-dmx", stereo_dmx]
            subprocess.run(args, check=True, capture_output=True)
            with wave.open(wav, "rb") as w:
                ch, rate = w.getnchannels(), w.getframerate()
                pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2")
            n_bad = sum(1 for ln in open(st) if ln.split()[1] != "0")
        return Decoded(pcm.reshape(-1, ch) / 32768.0, rate, n_bad)
    from ac4 import container
    from ac4.decoder import decode as py_decode
    frames = container.read_frames(path)
    pcm, dec = py_decode(frames, presentation=presentation,
                         language=language, layout=layout,
                         stereo_dmx=stereo_dmx, max_pass=10 ** 9)
    return Decoded(pcm, dec.out_rate, dec.n_bad)


def runs(d, names=None, margin=MARGIN):
    """-> [(channel name, start s, end s, median margin dB)]: the runs of
    windows in which one channel is `margin` dB above all the others
    (>= 3 s; >= 1.5 s for the LFE), ignoring windows more than 50 dB
    below the loudest one."""
    names = names or NAMES[d.ch]
    db = 10 * np.log10(d.e + 1e-20)
    floor = db.max() - 50
    srt = np.sort(db, axis=1)
    top = db.argmax(axis=1)
    gap = srt[:, -1] - srt[:, -2]
    lab = np.where((srt[:, -1] > floor) & (gap >= margin), top, -1)
    out, i = [], 0
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]:
            j += 1
        if lab[i] >= 0:
            nm = names[lab[i]]
            if (j - i) * WIN >= (1.5 if nm == "LFE" else 3.0):
                out.append((nm, i * WIN, j * WIN,
                            float(np.median(gap[i:j]))))
        i = j
    return out


def window(d, a, b):
    return slice(int(round(a / WIN)), int(round(b / WIN)))


def seq(rr):
    return [r[0] for r in rr]


def show(rr):
    return ", ".join(f"{n} {a:.1f}-{b:.1f}s ({m:.0f} dB)"
                     for n, a, b, m in rr)


ID_STREAMS = (by_role("id20") + by_role("id51_en") + by_role("id51_fr"))
ALL = sorted(found().items(), key=lambda hp: (KNOWN[hp[0]]["role"],
                                              KNOWN[hp[0]]["fps"]))
ALL_PRES = [(h, p, i) for h, p in ALL for i in range(len(KNOWN[h]["pres"]))]


# ---- every presentation decodes cleanly, for the expected time -----------

@pytest.mark.parametrize(
    "h,path,pi", ALL_PRES or [(None, None, None)],
    ids=[f"{KNOWN[h]['role']}-{KNOWN[h]['fps']}-p{i}"
         for h, _, i in ALL_PRES] or ["none"])
def test_every_presentation_decodes_cleanly(h, path, pi):
    k = KNOWN[h]
    d = decode(path, pi if len(k["pres"]) > 1 else None)
    assert d.n_bad == 0
    assert d.ch == k["pres"][pi]
    assert d.rate == 48000
    from ac4 import container
    n_frames = len(container.read_frames(path))
    frame_s = 1 / 25 if k["fps"] == 25 else 1001 / 30000
    assert abs(d.seconds - n_frames * frame_s) < 0.002
    if k["seconds"] is not None:
        # the procedure gives whole seconds
        assert abs(d.seconds - k["seconds"]) < 0.5, d.seconds
    # not silent: every channel but the LFE carries the programme
    for c in range(d.ch):
        if NAMES[d.ch][c] != "LFE":
            assert 10 * np.log10(d.e[:, c].mean() + 1e-20) > -50


# ---- channel identification ----------------------------------------------

@pytest.mark.parametrize("h,path", ID_STREAMS or [(None, None)],
                         ids=_ids(ID_STREAMS) or ["none"])
def test_channel_identification_order(h, path):
    d = decode(path)
    rr = runs(d)
    want = ORDER_20 if d.ch == 2 else ORDER_51
    assert seq(rr) == want, show(rr)
    # each announcement alone in its channel: >= 40 dB over the others
    # (the 20 dB of the detector is a floor, not the measured margin)
    assert min(m for *_, m in rr) > 40, show(rr)


def test_two_language_presentations_identify_channels():
    items = by_role("id_2lang")
    if not items:
        pytest.skip("stream not found")
    for h, path in items:
        for pi in (0, 1):
            rr = runs(decode(path, pi))
            assert seq(rr) == ORDER_51, (KNOWN[h]["fps"], pi, show(rr))


@pytest.mark.parametrize("h,path", by_role("id51_en") or [(None, None)],
                         ids=_ids(by_role("id51_en")) or ["none"])
def test_lfe_is_a_low_tone(h, path):
    """The procedure calls it a 100 Hz tone. It is a steady tone (most of
    its energy within 5 Hz of one frequency) below the LFE band edge; it
    measures 61 Hz, at both frame rates. (An independent decoder of the
    E-AC-3 version of another identification signal of the same kit
    family puts the LFE content at the same frequency as this decoder
    does, so the LFE path does not shift frequencies.)"""
    d = decode(path)
    rr = runs(d)
    n, a, b, _ = rr[-1]
    assert n == "LFE"
    s = d.lfe[int((a + 0.2) * d.rate):int((b - 0.2) * d.rate)]
    spec = np.abs(np.fft.rfft(s * np.hanning(len(s)), 1 << 18)) ** 2
    f = np.fft.rfftfreq(1 << 18, 1 / d.rate)
    pk = f[spec.argmax()]
    near = spec[np.abs(f - pk) < 5].sum() / spec.sum()
    assert 40 < pk < 120 and near > 0.85, (pk, near)


@pytest.mark.parametrize("h,path", by_role("id51_en") or [(None, None)],
                         ids=_ids(by_role("id51_en")) or ["none"])
@pytest.mark.parametrize("dmx", ["loro", "ltrt"])
def test_stereo_downmix_routing(h, path, dmx):
    """The procedure's table for a 2.0 output of the 5.1 signal: Lo/Ro
    puts Ls only left and Rs only right, Lt/Rt both surrounds in both;
    centre and LFE tone in both; L and R stay where they are."""
    rr = runs(decode(path))
    assert seq(rr) == ORDER_51
    d2 = decode(path, layout="2.0", stereo_dmx=dmx)
    assert d2.ch == 2
    both = {"C", "LFE"} | ({"Ls", "Rs"} if dmx == "ltrt" else set())
    left = {"L"} | ({"Ls"} if dmx == "loro" else set())
    report = []
    for n, a, b, _ in rr:
        sl = window(d2, a, b)
        eL, eR = (10 * np.log10(d2.e[sl, c].mean() + 1e-20) for c in (0, 1))
        report.append(f"{n}: L {eL:.1f} R {eR:.1f}")
        if n in both:
            assert abs(eL - eR) < 1.0, report
        elif n in left:
            assert eL - eR > 30, report
        else:
            assert eR - eL > 30, report


def test_immersive_stereo_identification():
    """A 7.1 identification (the 5.1 order, then the back pair: L C R Rs
    Ls LFE Lb Rb) the encoder rendered to two channels: the channels
    speak, in order, left, centre (both, equal), right, right, left, (the
    LFE is not rendered), left, right."""
    items = by_role("id_ims")
    if not items:
        pytest.skip("stream not found")
    for h, path in items:
        d = decode(path)
        assert d.ch == 2
        rr = runs(d)
        assert seq(rr)[:6] == ["L", "R", "R", "L", "L", "R"], show(rr)
        assert min(m for *_, m in rr[:6]) > 40, show(rr)
        # the centre: between the first two runs (less 0.3 s at each end,
        # where the neighbours' tails are), both channels equal and well
        # above the floor
        sl = window(d, rr[0][2] + 0.3, rr[1][1] - 0.3)
        db = 10 * np.log10(d.e[sl] + 1e-20)
        loud = db.max(axis=1) > db.max() - 30
        assert loud.sum() >= 10
        assert np.abs(db[loud, 0] - db[loud, 1]).max() < 1.0


# ---- language and presentation selection --------------------------------

def test_language_selection():
    items = by_role("id_2lang")
    if not items:
        pytest.skip("stream not found")
    for h, path in items:
        fps = KNOWN[h]["fps"]
        p0, p1 = decode(path, 0), decode(path, 1)
        assert p0.digest != p1.digest
        assert decode(path).digest == p0.digest            # default: en
        assert decode(path, language="en").digest == p0.digest
        assert decode(path, language="fr").digest == p1.digest
        for pd, role in ((p0, "id51_en"), (p1, "id51_fr")):
            for _, single in by_role(role, fps):
                assert decode(single).digest == pd.digest, role


@pytest.mark.parametrize(
    "role", ["multi_bitrate", "multi_all", "multi_layout"])
def test_multi_presentation_streams(role):
    items = by_role(role)
    if not items:
        pytest.skip("stream not found")
    for h, path in items:
        k = KNOWN[h]
        fps = k["fps"]
        # the default is the first presentation (for the 2.0 / 5.1 stream
        # the procedure names the stereo one the default)
        assert decode(path).digest == decode(path, 0).digest
        digests = [decode(path, i).digest for i in range(len(k["pres"]))]
        assert len(set(digests)) == len(digests)
        for i, role_i in enumerate(k.get("same", [])):
            for _, single in by_role(role_i, fps):
                assert decode(single).digest == digests[i], (role, i)


# ---- frame rates ---------------------------------------------------------

@pytest.mark.parametrize("role", ["id20", "id51_en", "id51_fr", "id_ims"])
def test_frame_rates_line_up(role):
    items = by_role(role)
    if len(items) < 2:
        pytest.skip("need the 25 and 29.97 fps versions")
    (_, p25), (_, p30) = items[0], items[1]
    a, b = decode(p25), decode(p30)
    assert abs(a.seconds - b.seconds) < 0.05
    ra, rb = runs(a), runs(b)
    assert seq(ra) == seq(rb), (show(ra), show(rb))
    for x, y in zip(ra, rb):
        assert abs(x[1] - y[1]) < 0.6 and abs(x[2] - y[2]) < 0.6, \
            (show(ra), show(rb))
