# SPDX-License-Identifier: Apache-2.0
"""A small fragmented-MP4 (ISO-BMFF moof + mdat) sample reader for AC-4.

This replaces the one function the decoder borrowed from the receiver it
was written in (a fragment-run scanner shaped for that receiver's own
rebuilt fragments). It reads what the ISO base media file format says a
track fragment can carry, so it also covers fragments that rely on the
track-fragment header's default sample size or an explicit base data
offset:

  * tfhd: base_data_offset (0x01), default_sample_size (0x10),
          default-base-is-moof (0x20000)
  * trun: data_offset (0x01), first_sample_flags (0x04), per-sample
          duration/size/flags/composition offset (0x100/0x200/0x400/0x800)
  * several truns per traf (a trun without data_offset continues where the
    previous one ended)

TRACK CHOICE. When the file has a moov, the track whose sample entry is
'ac-4' is used (the first one if there are several). Without a moov (a bare
media segment), the first track fragment whose trun carries PER-SAMPLE
sizes is used; that is the rule the original scanner applied, and it skips
a hint track that only declares a default size.

A fragment whose mdat is not complete in the input is skipped as a whole,
the same way a reader of a growing file leaves it for later.

Not handled (stage 1): the 'dac4' decoder-specific box is not parsed (the
decoder reads everything from the in-band table of contents), raw AC-4
sync frames, MPEG-TS, and non-fragmented MP4 (stsz/stco sample tables).
"""
from __future__ import annotations


def boxes(d, start=0, end=None):
    """Yield (offset, size, type) for the boxes in d[start:end]."""
    end = len(d) if end is None else end
    o = start
    while o + 8 <= end:
        sz = int.from_bytes(d[o:o + 4], "big")
        typ = bytes(d[o + 4:o + 8])
        hdr = 8
        if sz == 1:
            if o + 16 > end:
                return
            sz = int.from_bytes(d[o + 8:o + 16], "big")
            hdr = 16
        elif sz == 0:
            sz = end - o
        if sz < hdr:
            return
        yield o, sz, typ
        if o + sz > end:
            return
        o += sz


def _child(d, o, sz, typ, skip=8):
    for co, csz, ct in boxes(d, o + skip, o + sz):
        if ct == typ:
            return co, csz
    return None


def ac4_track_ids(d):
    """-> [track_id] of every track in the moov whose sample entry is 'ac-4'."""
    out = []
    for o, sz, t in boxes(d):
        if t != b"moov":
            continue
        for to, tsz, tt in boxes(d, o + 8, o + sz):
            if tt != b"trak":
                continue
            tkhd = _child(d, to, tsz, b"tkhd")
            if not tkhd:
                continue
            ver = d[tkhd[0] + 8]
            tid_at = tkhd[0] + 12 + (16 if ver == 1 else 8)
            tid = int.from_bytes(d[tid_at:tid_at + 4], "big")
            p = (to, tsz)
            for name in (b"mdia", b"minf", b"stbl", b"stsd"):
                p = _child(d, p[0], p[1], name)
                if not p:
                    break
            if not p:
                continue
            # stsd: full box header (4) + entry_count (4), then entries
            for _eo, _esz, et in boxes(d, p[0] + 16, p[0] + p[1]):
                if et == b"ac-4":
                    out.append(tid)
                break
    return out


def _tfhd(d, o):
    fl = int.from_bytes(d[o + 9:o + 12], "big")
    tid = int.from_bytes(d[o + 12:o + 16], "big")
    q = o + 16
    base = None
    if fl & 0x01:
        base = int.from_bytes(d[q:q + 8], "big")
        q += 8
    if fl & 0x02:
        q += 4
    if fl & 0x08:
        q += 4
    dss = None
    if fl & 0x10:
        dss = int.from_bytes(d[q:q + 4], "big")
    return dict(flags=fl, track_id=tid, base=base, default_size=dss)


def _trun(d, o):
    fl = int.from_bytes(d[o + 9:o + 12], "big")
    n = int.from_bytes(d[o + 12:o + 16], "big")
    p = o + 16
    doff = None
    if fl & 0x001:
        doff = int.from_bytes(d[p:p + 4], "big", signed=True)
        p += 4
    if fl & 0x004:
        p += 4
    per = sum(4 for b in (0x100, 0x200, 0x400, 0x800) if fl & b)
    soff = 4 if fl & 0x100 else 0
    sizes = None
    if fl & 0x200:
        sizes = [int.from_bytes(d[p + i * per + soff:p + i * per + soff + 4], "big")
                 for i in range(n)]
    return dict(flags=fl, n=n, data_offset=doff, sizes=sizes)


def fragment_samples(d, moof_o, moof_sz, track_id=None):
    """-> [(abs_offset, size)] for one moof, or None if no usable track.

    `track_id` None = the first traf whose trun has per-sample sizes."""
    prev_end = moof_o
    for po, psz, pt in boxes(d, moof_o + 8, moof_o + moof_sz):
        if pt != b"traf":
            continue
        tf = None
        truns = []
        for co, csz, ct in boxes(d, po + 8, po + psz):
            if ct == b"tfhd":
                tf = _tfhd(d, co)
            elif ct == b"trun":
                truns.append(_trun(d, co))
        if tf is None or not truns:
            continue
        if tf["base"] is not None:
            base = tf["base"]
        elif tf["flags"] & 0x20000:
            base = moof_o
        else:
            # ISO-BMFF 8.8.7.1: the first traf's default base is the moof;
            # a later one continues after the previous traf's data
            base = prev_end
        out = []
        pos = None
        for tr0 in truns:
            if tr0["data_offset"] is not None:
                pos = base + tr0["data_offset"]
            elif pos is None:
                pos = base
            sizes = tr0["sizes"]
            if sizes is None:
                if tf["default_size"] is None:
                    sizes = None
                else:
                    sizes = [tf["default_size"]] * tr0["n"]
            if sizes is None:
                out = None
                break
            for s in sizes:
                out.append((pos, s))
                pos += s
        if out is None:
            continue
        prev_end = pos
        if track_id is None:
            if all(t["sizes"] is not None for t in truns):
                return out
        elif tf["track_id"] == track_id:
            return out
    return None


def frames(data, track_id=None):
    """-> [bytes]: one raw AC-4 frame per MP4 sample, in file order.

    `data` is the bytes of an init segment followed by media segments (or
    media segments alone). Incomplete trailing fragments are skipped."""
    d = bytes(data)
    if track_id is None:
        ids = ac4_track_ids(d)
        track_id = ids[0] if ids else None
    out = []
    tops = list(boxes(d))
    for i, (o, sz, t) in enumerate(tops):
        if t != b"moof":
            continue
        # the fragment is complete only when its mdat has fully arrived
        nxt = tops[i + 1] if i + 1 < len(tops) else None
        if nxt is None or nxt[2] != b"mdat" or nxt[0] + nxt[1] > len(d):
            continue
        smp = fragment_samples(d, o, sz, track_id)
        if smp is None:
            continue
        md_end = nxt[0] + nxt[1]
        if any(p < nxt[0] + 8 or p + s > md_end for p, s in smp):
            continue
        out.extend(d[p:p + s] for p, s in smp)
    return out


def read_frames(path, track_id=None):
    """frames() on a file."""
    with open(path, "rb") as fh:
        return frames(fh.read(), track_id)
