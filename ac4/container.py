# SPDX-License-Identifier: Apache-2.0
"""AC-4 inputs other than MP4, and input auto-detection.

  * AC-4 sync frames (TS 103 190-1 Annex G): ac4_syncframe() = sync_word
    0xAC40 or 0xAC41, frame_size (16 bits, or 0xFFFF and 24 bits), the
    raw_ac4_frame, and with 0xAC41 a crc_word. The CRC (G.4.2) is CRC-16
    with generator x^16 + x^15 + x^2 + 1, initial state 0, no reflection,
    no final XOR, over frame_size and raw_ac4_frame; the protected payload
    followed by crc_word gives 0.
  * MPEG-2 transport stream: the AC-4 elementary stream of ETSI TS 101 154
    (stream_type 0x06, PES private_stream_1, with the DVB AC-4 descriptor:
    extension descriptor tag 0x7F, descriptor_tag_extension 0x15), whose
    PES payload is a sequence of AC-4 sync frames.
  * An ID3v2 tag in front of any of these (HLS packed audio) is skipped.

DAMAGE. A sync frame whose CRC fails is passed on EMPTY (the decoder
conceals it), so the timeline keeps its frame. A frame is taken when its
CRC checks or, without a CRC, when the next sync word (or an ID3 tag, or
the end of the data) is where frame_size says; otherwise the reader
searches for the next sync word that starts such a frame and counts the
bytes it skipped (those frames are lost: the timeline shortens, and the
decoder sees a sequence_counter jump).
A transport packet whose continuity_counter skips marks the PES it belongs
to as damaged; the sync frames inside are then judged by their own sync
words and CRCs.
"""
from __future__ import annotations

from . import mp4

SYNC = (0xAC40, 0xAC41)


# ---------------------------------------------------------------- CRC
def _crc_table():
    t = []
    for i in range(256):
        c = i << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x8005) if c & 0x8000 else (c << 1)
        t.append(c & 0xFFFF)
    return t


CRC_TABLE = _crc_table()


def crc16(data, crc=0):
    """CRC of TS 103 190-1 G.4.2 (generator 0x8005, MSB first, init 0)."""
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ CRC_TABLE[((crc >> 8) ^ byte) & 0xFF]
    return crc


# ---------------------------------------------------------------- ID3
def skip_id3(d, o=0):
    """-> offset after any ID3v2 tags starting at o."""
    while len(d) - o >= 10 and d[o:o + 3] == b"ID3":
        sz = 0
        for b in d[o + 6:o + 10]:
            sz = (sz << 7) | (b & 0x7F)
        foot = 10 if d[o + 5] & 0x10 else 0
        o += 10 + sz + foot
    return o


# ---------------------------------------------------------------- sync frames
def _frame_at(d, o):
    """-> (header_len, frame_size, total_len, has_crc) of a sync frame
    whose sync word is at o, or None when it cannot be one."""
    if len(d) - o < 4:
        return None
    sw = (d[o] << 8) | d[o + 1]
    if sw not in SYNC:
        return None
    fs = (d[o + 2] << 8) | d[o + 3]
    hdr = 4
    if fs == 0xFFFF:
        if len(d) - o < 7:
            return None
        fs = (d[o + 4] << 16) | (d[o + 5] << 8) | d[o + 6]
        hdr = 7
    crc = sw == 0xAC41
    return hdr, fs, hdr + fs + (2 if crc else 0), crc


def _plausible(d, o):
    """A sync word at o that starts a frame: one whose crc_word checks, or
    one followed, frame_size later, by another sync word, an ID3 tag or
    the end of the data."""
    f = _frame_at(d, o)
    if f is None or f[1] == 0:
        return False
    nxt = o + f[2]
    if f[3] and nxt <= len(d) and crc16(d[o + 2:nxt]) == 0:
        return True
    return nxt == len(d) or (nxt + 2 <= len(d)
                             and ((d[nxt] << 8) | d[nxt + 1]) in SYNC) or \
        d[nxt:nxt + 3] == b"ID3"          # HLS packed audio: a tag per segment


class SyncInfo:
    def __init__(self):
        self.n_frames = 0
        self.n_crc = 0            # frames with a crc_word
        self.n_crc_bad = 0        # of those, CRC failed (passed on empty)
        self.n_skipped = 0        # bytes skipped to regain sync
        self.n_truncated = 0      # a last frame cut short by the data's end

    def as_dict(self):
        return dict(self.__dict__)


def sync_frames(data, info=None, check_crc=True):
    """-> [bytes]: the raw_ac4_frame of each ac4_syncframe in data."""
    d = bytes(data)
    info = info if info is not None else SyncInfo()
    out = []
    o = 0
    while o < len(d):
        o = skip_id3(d, o)
        if o >= len(d):
            break
        f = _frame_at(d, o)
        if f is None or (f[1] == 0) or (o + f[2] < len(d)
                                        and not _plausible(d, o)):
            # lost sync: find the next sync word that starts a frame
            # followed by another one (or by the end)
            p = o + 1
            while p + 1 < len(d):
                if ((d[p] << 8) | d[p + 1]) in SYNC and _plausible(d, p):
                    break
                p += 1
            else:
                p = len(d)
            info.n_skipped += p - o
            o = p
            continue
        hdr, fs, total, has_crc = f
        if o + total > len(d):
            info.n_truncated += 1
            break
        raw = d[o + hdr:o + hdr + fs]
        info.n_frames += 1
        if has_crc:
            info.n_crc += 1
            if check_crc and crc16(d[o + 2:o + total]) != 0:
                info.n_crc_bad += 1
                raw = b""
        out.append(raw)
        o += total
    return out


# ---------------------------------------------------------------- MPEG-TS
AC4_EXT_TAG = 0x15        # DVB extension descriptor: AC-4 descriptor


class TsInfo:
    def __init__(self):
        self.pid = None
        self.n_packets = 0
        self.n_cc_errors = 0
        self.n_pes = 0
        self.sync = SyncInfo()

    def as_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "sync"}
        d["sync"] = self.sync.as_dict()
        return d


def _ts_packets(d):
    """Yield (pid, pusi, cc, payload) of each 188-byte packet in sync."""
    o = 0
    n = len(d)
    while o + 188 <= n:
        if d[o] != 0x47:
            # regain packet sync: three sync bytes 188 apart
            p = o + 1
            while p + 188 * 2 < n and not (d[p] == 0x47 and d[p + 188] == 0x47
                                           and d[p + 376] == 0x47):
                p += 1
            if p + 188 * 2 >= n:
                return
            o = p
            continue
        pkt = d[o:o + 188]
        o += 188
        if pkt[1] & 0x80:                       # transport_error_indicator
            pid = ((pkt[1] & 0x1F) << 8) | pkt[2]
            yield pid, False, -1, None
            continue
        pusi = bool(pkt[1] & 0x40)
        pid = ((pkt[1] & 0x1F) << 8) | pkt[2]
        afc = (pkt[3] >> 4) & 3
        cc = pkt[3] & 0x0F
        i = 4
        if afc & 2:
            i += 1 + pkt[4]
        if not afc & 1 or i > 188:
            yield pid, pusi, None, None          # no payload: cc unchanged
            continue
        yield pid, pusi, cc, pkt[i:]


def _section(payload):
    """The section a PSI packet with PUSI starts (pointer_field skipped)."""
    p = payload[0]
    return payload[1 + p:]


def _pmt_ac4_pids(sec):
    """-> [pid] of the AC-4 elementary streams a PMT section lists."""
    if len(sec) < 12 or sec[0] != 0x02:
        return []
    n = ((sec[1] & 0x0F) << 8) | sec[2]
    end = min(len(sec), 3 + n - 4)
    pil = ((sec[10] & 0x0F) << 8) | sec[11]
    k = 12 + pil
    out = []
    while k + 5 <= end:
        st = sec[k]
        pid = ((sec[k + 1] & 0x1F) << 8) | sec[k + 2]
        il = ((sec[k + 3] & 0x0F) << 8) | sec[k + 4]
        desc = sec[k + 5:k + 5 + il]
        k += 5 + il
        if st != 0x06:
            continue
        j = 0
        while j + 2 <= len(desc):
            tag, ln = desc[j], desc[j + 1]
            body = desc[j + 2:j + 2 + ln]
            if (tag == 0x7F and ln >= 1 and body[0] == AC4_EXT_TAG) or \
                    (tag == 0x05 and body[:4] == b"AC-4"):
                out.append(pid)
                break
            j += 2 + ln
    return out


def ts_elementary_stream(data, info=None, pid=None):
    """-> bytes: the PES payloads of the first AC-4 elementary stream (or
    of `pid`), concatenated in order."""
    d = bytes(data)
    info = info if info is not None else TsInfo()
    pmt_pids = set()
    target = pid
    out = bytearray()
    cur = None                  # the PES being collected
    last_cc = {}
    for p, pusi, cc, pl in _ts_packets(d):
        info.n_packets += 1
        if p == 0 and pusi and pl is not None and target is None:
            sec = _section(pl)
            if len(sec) >= 8 and sec[0] == 0x00:
                n = ((sec[1] & 0x0F) << 8) | sec[2]
                for k in range(8, min(len(sec), 3 + n - 4) - 3, 4):
                    prog = (sec[k] << 8) | sec[k + 1]
                    if prog:
                        pmt_pids.add(((sec[k + 2] & 0x1F) << 8) | sec[k + 3])
            continue
        if p in pmt_pids and pusi and pl is not None and target is None:
            pids = _pmt_ac4_pids(_section(pl))
            if pids:
                target = pids[0]
            continue
        if target is None or p != target:
            continue
        info.pid = target
        if cc is None:                          # adaptation only
            continue
        if cc < 0:                              # transport error
            info.n_cc_errors += 1
            continue
        prev = last_cc.get(p)
        if prev is not None and cc != ((prev + 1) & 0x0F) and cc != prev:
            info.n_cc_errors += 1
        last_cc[p] = cc
        if prev is not None and cc == prev:
            continue                            # a duplicate packet
        if pusi:
            if cur is not None:
                out += _pes_payload(cur)
            cur = bytearray(pl)
            info.n_pes += 1
        elif cur is not None:
            cur += pl
    if cur is not None:
        out += _pes_payload(cur)
    return bytes(out)


def _pes_payload(pes):
    if len(pes) < 9 or pes[:3] != b"\x00\x00\x01":
        return b""
    hl = pes[8]
    plen = (pes[4] << 8) | pes[5]
    body = pes[9 + hl:]
    if plen:
        body = body[:max(0, plen - 3 - hl)]
    return bytes(body)


def ts_frames(data, info=None, check_crc=True):
    info = info if info is not None else TsInfo()
    es = ts_elementary_stream(data, info)
    return sync_frames(es, info.sync, check_crc=check_crc)


# ---------------------------------------------------------------- detection
MP4_TOP = {b"ftyp", b"styp", b"moov", b"moof", b"sidx", b"free", b"skip",
           b"mdat", b"emsg", b"prft", b"uuid", b"wide"}


def detect(data):
    """-> "mp4", "sync", "ts" or None."""
    d = data
    if len(d) >= 188 * 2 + 1 and d[0] == 0x47 and d[188] == 0x47:
        return "ts"
    o = skip_id3(d)
    if len(d) - o >= 4 and ((d[o] << 8) | d[o + 1]) in SYNC:
        return "sync"
    if len(d) >= 8 and bytes(d[4:8]) in MP4_TOP:
        return "mp4"
    # a sync stream that starts mid-frame
    for p in range(o, min(len(d) - 1, o + (1 << 16))):
        if ((d[p] << 8) | d[p + 1]) in SYNC and _plausible(d, p):
            return "sync"
    return None


class Source:
    """What `read` found: frames (raw AC-4 frames, b"" for a lost one),
    kind, the dac4 DSI (dict or None), and per-kind information."""

    def __init__(self, frames, kind, dsi=None, info=None):
        self.frames = frames
        self.kind = kind
        self.dsi = dsi
        self.info = info or {}


def read_bytes(data, check_crc=True):
    d = bytes(data)
    kind = detect(d)
    if kind == "ts":
        ti = TsInfo()
        fr = ts_frames(d, ti, check_crc)
        return Source(fr, kind, info=ti.as_dict())
    if kind == "sync":
        si = SyncInfo()
        fr = sync_frames(d, si, check_crc)
        return Source(fr, kind, info=si.as_dict())
    if kind == "mp4":
        fr = mp4.frames(d)
        dsi = None
        try:
            raw = mp4.dac4_payload(d)
            if raw is not None:
                dsi = mp4.parse_dac4(raw)
        except Exception as e:                                 # noqa: BLE001
            dsi = dict(error=str(e)[:80])
        return Source(fr, kind, dsi=dsi)
    return Source([], None)


def read(path, check_crc=True):
    """-> Source of the AC-4 in an MP4 (fragmented or not), a stream of
    AC-4 sync frames (.ac4), or an MPEG-2 transport stream."""
    with open(path, "rb") as fh:
        return read_bytes(fh.read(), check_crc)


def read_frames(path):
    return read(path).frames
