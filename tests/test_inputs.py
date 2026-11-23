# SPDX-License-Identifier: Apache-2.0
"""Inputs: AC-4 sync frames (CRC, escape, resync, ID3), MPEG-TS, the
sample table of a non-fragmented MP4, the dac4 box, detection. Built here
from the specifications; when the C library is built, each input is also
read by it (c/tests/unit_test --dump-input) and must give the same frames.
"""
import os
import subprocess

import pytest

from ac4 import container as CT
from ac4 import mp4

from test_phase_a import BW

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = os.path.join(HERE, "..", "c", "tests", "unit_test")


def test_crc16_is_the_annex_g_crc():
    """x^16 + x^15 + x^2 + 1, init 0, no reflection, no final XOR: the
    catalogued CRC-16/UMTS, check value 0xFEE8 on "123456789"."""
    assert CT.crc16(b"123456789") == 0xFEE8
    assert CT.crc16(b"") == 0


def syncframe(raw, crc=True, escape=False):
    sw = b"\xac\x41" if crc else b"\xac\x40"
    if escape or len(raw) >= 0xFFFF:
        size = b"\xff\xff" + len(raw).to_bytes(3, "big")
    else:
        size = len(raw).to_bytes(2, "big")
    out = sw + size + raw
    if crc:
        out += CT.crc16(size + raw).to_bytes(2, "big")
    return out


RAWS = [bytes([0x80 + i]) * (20 + 3 * i) for i in range(6)]


def test_sync_frames_round_trip_with_crc_and_escape():
    d = b"".join(syncframe(r, crc=i % 2 == 0, escape=i == 3)
                 for i, r in enumerate(RAWS))
    info = CT.SyncInfo()
    assert CT.sync_frames(d, info) == RAWS
    assert info.n_crc == 3 and info.n_crc_bad == 0 and info.n_skipped == 0
    assert CT.detect(d) == "sync"


def test_sync_frame_with_bad_crc_is_passed_on_empty():
    fr = [syncframe(r) for r in RAWS]
    bad = bytearray(fr[2])
    bad[6] ^= 0x10
    d = b"".join(fr[:2]) + bytes(bad) + b"".join(fr[3:])
    info = CT.SyncInfo()
    out = CT.sync_frames(d, info)
    assert out == RAWS[:2] + [b""] + RAWS[3:]
    assert info.n_crc_bad == 1


def test_sync_resync_after_garbage_and_id3():
    id3 = b"ID3\x04\x00\x00\x00\x00\x00\x05" + b"hello"
    fr = [syncframe(r) for r in RAWS]
    d = id3 + fr[0] + b"\x12\xac\x41\x99garbage" + fr[1] + id3 + b"".join(
        fr[2:])
    info = CT.SyncInfo()
    assert CT.sync_frames(d, info) == RAWS
    assert info.n_skipped == len(b"\x12\xac\x41\x99garbage")
    assert CT.detect(d) == "sync"


# ---------------------------------------------------------------- TS
def _crc32_mpeg(b):
    c = 0xFFFFFFFF
    for x in b:
        c ^= x << 24
        for _ in range(8):
            c = ((c << 1) ^ 0x04C11DB7) if c & 0x80000000 else (c << 1)
            c &= 0xFFFFFFFF
    return c


def _psi(table_id, ext, body):
    sec = bytes([table_id]) + (0xB000 | (len(body) + 9)).to_bytes(2, "big")
    sec += ext.to_bytes(2, "big") + b"\xc1\x00\x00" + body
    return sec + _crc32_mpeg(sec).to_bytes(4, "big")


def _packets(pid, payload, pusi_first=True, cc0=0):
    out, cc, first = [], cc0, True
    while payload or first:
        chunk = payload[:184]
        payload = payload[184:]
        hdr = bytes([0x47, (0x40 if first and pusi_first else 0) | (pid >> 8),
                     pid & 0xFF])
        if len(chunk) < 184:                   # stuff with an adaptation field
            pad = 184 - len(chunk)
            af = bytes([pad - 1]) + (b"\x00" + b"\xff" * (pad - 2)
                                     if pad >= 2 else b"")
            out.append(hdr + bytes([0x30 | cc]) + af + chunk)
        else:
            out.append(hdr + bytes([0x10 | cc]) + chunk)
        cc = (cc + 1) & 15
        first = False
    return out


def build_ts(frames_per_pes, pid=0x101, descriptor=b"\x7f\x02\x15\x00",
             drop=None):
    pat = _psi(0x00, 1, b"\x00\x01\xe1\x00")            # program 1 -> 0x100
    es = bytes([0x06, 0xE0 | (pid >> 8), pid & 0xFF,
                0xF0, len(descriptor)]) + descriptor
    pmt = _psi(0x02, 1, b"\xe1\x01\xf0\x00" + es)
    pk = _packets(0, b"\x00" + pat) + _packets(0x100, b"\x00" + pmt)
    cc = 0
    for group in frames_per_pes:
        body = b"".join(group)
        pes = b"\x00\x00\x01\xbd" + (len(body) + 8).to_bytes(2, "big") + \
            b"\x84\x80\x05" + b"\x21\x00\x01\x00\x01" + body
        ps = _packets(pid, pes, cc0=cc)
        cc = (cc + len(ps)) & 15
        pk += ps
    if drop is not None:
        del pk[drop]
    return b"".join(pk)


def test_ts_demux_reads_the_ac4_stream():
    fr = [syncframe(r, crc=False) for r in RAWS]
    d = build_ts([fr[:2], fr[2:3], fr[3:]])
    assert CT.detect(d) == "ts"
    info = CT.TsInfo()
    assert CT.ts_frames(d, info) == RAWS
    assert info.pid == 0x101 and info.n_pes == 3 and info.n_cc_errors == 0


def test_ts_registration_descriptor_and_lost_packet():
    big = [bytes([0x90 + i]) * 300 for i in range(4)]
    fr = [syncframe(r) for r in big]
    # packets: PAT, PMT, then the first PES in two (2, 3): lose its second
    d = build_ts([fr[:1], fr[1:2], fr[2:]], descriptor=b"\x05\x04AC-4",
                 drop=3)
    info = CT.TsInfo()
    out = CT.ts_frames(d, info)
    assert info.n_cc_errors == 1
    # the lost packet cuts the first frame: the reader skips it, regains
    # sync and keeps the frames after it
    assert out == big[1:] and info.sync.n_skipped > 0


# ---------------------------------------------------------------- MP4
def _box(t, payload):
    return (8 + len(payload)).to_bytes(4, "big") + t + payload


def _full(t, ver, flags, payload):
    return _box(t, bytes([ver]) + flags.to_bytes(3, "big") + payload)


def build_dac4():
    """An ac4_dsi_v1 with one presentation (TS 103 190-2 E.6, E.10, E.11):
    bitstream_version 2, 48 kHz, frame_rate_index 2, a single channel-coded
    5.1 group in French, presentation_id 7."""
    w = BW()
    w.u(3, 1).u(7, 2).u(1, 1).u(4, 2).u(9, 1)
    w.u(1, 0)                                        # b_program_id
    w.u(2, 0).u(32, 128000).u(32, 0)                 # ac4_bitrate_dsi
    w.align()
    p = BW()
    p.u(5, 0x1F).u(3, 1).u(1, 1).u(5, 7).u(2, 0).u(2, 0).u(5, 0).u(10, 0)
    p.u(1, 1).u(5, 4).u(6, 0).u(18, 0)               # channel coded, 5.1
    p.u(1, 0).u(1, 0)                                # core differs, filter
    g = BW()
    g.u(1, 1).u(1, 0).u(1, 1).u(8, 1)                # one substream
    g.u(2, 0).u(1, 0).u(6, 0).u(18, 0)
    g.u(1, 1).u(3, 0).u(1, 1).u(6, 2).u(8, ord("f")).u(8, ord("r"))
    p.bits += g.bits
    p.u(1, 0).u(1, 0)                                # pre_virtualized, emdf
    p.u(1, 0).u(1, 0)                                # bitrate, alternative
    pb = p.bytes()
    w.u(8, 1).u(8, len(pb))
    return w.bytes() + pb


def build_plain_mp4(samples, chunks=(2, 3), with_fragment=None):
    """A non-fragmented file: moov with stsz / stsc / stco, mdat."""
    entry = _box(b"ac-4", bytes(6) + b"\x00\x01" + bytes(8) + b"\x00\x02"
                 + b"\x00\x10" + bytes(4) + (48000 << 16).to_bytes(4, "big")
                 + _box(b"dac4", build_dac4()))
    ftyp = _box(b"ftyp", b"isom" + bytes(4))

    def moov(offsets):
        stsd = _full(b"stsd", 0, 0, (1).to_bytes(4, "big") + entry)
        stsz = _full(b"stsz", 0, 0, (0).to_bytes(4, "big")
                     + len(samples).to_bytes(4, "big")
                     + b"".join(len(s).to_bytes(4, "big") for s in samples))
        stsc = _full(b"stsc", 0, 0, (2).to_bytes(4, "big")
                     + (1).to_bytes(4, "big") + chunks[0].to_bytes(4, "big")
                     + (1).to_bytes(4, "big")
                     + (2).to_bytes(4, "big") + chunks[1].to_bytes(4, "big")
                     + (1).to_bytes(4, "big"))
        stco = _full(b"stco", 0, 0, len(offsets).to_bytes(4, "big")
                     + b"".join(o.to_bytes(4, "big") for o in offsets))
        stbl = _box(b"stbl", stsd + stsz + stsc + stco)
        tkhd = _full(b"tkhd", 0, 0, bytes(8) + (1).to_bytes(4, "big")
                     + bytes(68))
        return _box(b"moov", _box(b"trak", tkhd + _box(b"mdia", _box(
            b"minf", stbl))))

    # chunk layout: chunks[0] samples in chunk 1, then chunks[1] per chunk
    groups, i = [], 0
    sizes = [chunks[0]] + [chunks[1]] * len(samples)
    for n in sizes:
        if i >= len(samples):
            break
        groups.append(samples[i:i + n])
        i += n
    head = len(ftyp) + len(moov([0] * len(groups))) + 8
    offs, pos = [], head
    for g in groups:
        offs.append(pos)
        pos += sum(map(len, g))
    out = ftyp + moov(offs) + _box(b"mdat", b"".join(b"".join(g)
                                                     for g in groups))
    return out


def test_plain_mp4_sample_table_and_dac4():
    smp = [bytes([0x40 + i]) * (11 + 2 * i) for i in range(9)]
    d = build_plain_mp4(smp)
    assert CT.detect(d) == "mp4"
    assert mp4.frames(d) == smp
    raw = mp4.dac4_payload(d)
    dsi = mp4.parse_dac4(raw)
    assert (dsi["bitstream_version"], dsi["fs_index"],
            dsi["frame_rate_index"]) == (2, 1, 2)
    p = dsi["presentations"][0]
    assert p["presentation_id"] == 7 and p["channel_mode"] == 4
    assert p["groups"][0]["language"] == "fr"
    s = CT.read_bytes(d)
    assert s.kind == "mp4" and s.dsi["presentations"][0]["md_compat"] == 1


def test_plain_mp4_sample_outside_the_file_is_empty():
    smp = [bytes([0x40 + i]) * 30 for i in range(5)]
    d = build_plain_mp4(smp)
    cut = d[:-40]                                  # the last sample is cut
    fr = mp4.frames(cut)
    assert len(fr) == 5 and fr[-1] == b"" and fr[:3] == smp[:3]


# ---------------------------------------------------------------- C side
@pytest.mark.skipif(not os.path.exists(UNIT), reason="C library not built")
def test_c_reads_the_same_frames(tmp_path):
    def fnv(b):
        h = 2166136261
        for x in b:
            h = ((h ^ x) * 16777619) & 0xFFFFFFFF
        return h

    def c_frames(path):
        out = subprocess.run([UNIT, "--dump-input", path],
                             capture_output=True, text=True,
                             check=True).stdout.split("\n")
        kind = int(out[0].split()[0])
        return kind, [(int(a), int(b, 16)) for a, b in
                      (ln.split() for ln in out[1:] if ln.strip())]
    fr = [syncframe(r) for r in RAWS]
    bad = bytearray(fr[2])
    bad[6] ^= 0x10
    cases = {
        "sync.ac4": b"".join(fr[:2]) + bytes(bad) + b"\x12garbage"
        + b"".join(fr[3:]),
        "plain.mp4": build_plain_mp4([bytes([0x40 + i]) * (11 + 2 * i)
                                      for i in range(9)]),
        "a.ts": build_ts([[syncframe(r, crc=False) for r in RAWS[:3]],
                          [syncframe(r, crc=False) for r in RAWS[3:]]]),
        "lost.ts": build_ts([[syncframe(bytes([0x90 + i]) * 300)]
                             for i in range(4)], drop=3),
    }
    for name, data in cases.items():
        p = tmp_path / name
        p.write_bytes(data)
        s = CT.read_bytes(data)
        kind, frames = c_frames(str(p))
        assert kind == {"mp4": 1, "sync": 2, "ts": 3}[s.kind], name
        assert frames == [(len(f), fnv(f)) for f in s.frames], name
