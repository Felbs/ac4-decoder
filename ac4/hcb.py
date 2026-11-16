# SPDX-License-Identifier: Apache-2.0
"""M23 -- the AC-4 Huffman codebooks, parsed from the spec's own C file and
gated individually on the Kraft equality.

Annex A does not print the tables; it ships them as `ts_103190_tables.c` inside
`ts_10319001v010301p0.zip`.  That is a better source than a PDF in every way --
machine readable, explicit `_LEN` / `_CW` pairs, no OCR ambiguity -- but "better
source" is not "verified", so every table gets a test it can fail.

THE GATE: KRAFT EQUALITY
-------------------------
A binary prefix code with codeword lengths L satisfies

    sum(2^-L) == 1

exactly, when the code is COMPLETE -- which every Huffman code is.  Less than 1
means the table is missing codewords; more than 1 means it is over-subscribed
and cannot be a prefix code at all.  Either way the sum is a single number that
catches a truncated read, an off-by-one in the array length, or a mis-parsed
entry, and it does so per table rather than in aggregate.

A SECOND, INDEPENDENT CHECK
----------------------------
The codewords themselves must actually form a prefix code: no codeword may be a
prefix of another.  Kraft is necessary but not sufficient (a set of lengths can
satisfy it while the assigned codewords collide), so both are run.  And the
declared `codebook_length` from Annex A -- 289 for HCB_11, 81 for HCB_1, and so
on -- must match the array size, which ties the C file back to the PDF.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_C = os.environ.get("AC4_ETSI_TABLES_C", "")  # optional, never shipped

# codebook_length values printed in Annex A, for the cross-check
ANNEX_A = {
    "ASF_HCB_SCALEFAC": 121, "ASF_HCB_1": 81, "ASF_HCB_2": 81,
    "ASF_HCB_3": 81, "ASF_HCB_4": 81, "ASF_HCB_5": 81, "ASF_HCB_6": 81,
    "ASF_HCB_7": 64, "ASF_HCB_8": 64, "ASF_HCB_9": 169, "ASF_HCB_10": 169,
    "ASF_HCB_11": 289, "ASF_HCB_SNF": 22,
}

ARRAY = re.compile(
    r"const\s+\w+\s+([A-Za-z0-9_]+)\s*\[\s*(\d*)\s*\]\s*=\s*\{(.*?)\}\s*;",
    re.S)


BANK = os.path.join(HERE, "tables", "ac4_huffman.json")


def parse_c(path):
    """-> {name: [ints]} for every const array in the file.

    A fresh clone has no ETSI source file -- the document is copyrighted and
    not ours to redistribute. The codebooks themselves are integers, so they
    ship as a banked artifact and the AC-4 decoder works out of the box.
    Without this the receiver decoded video and no sound.
    """
    if not os.path.exists(path) and os.path.exists(BANK):
        import json
        with open(BANK, encoding="utf-8") as fh:
            arrays = json.load(fh)
        # Gated on load, not trusted: a complete prefix code has Kraft sum
        # 1.0, which a transposed digit breaks.
        for name, lengths in arrays.items():
            if name.endswith("_LEN") and abs(kraft(lengths) - 1.0) > 1e-9:
                raise SystemExit(f"banked codebook {name} failed its Kraft gate")
        return arrays
    src = open(path, encoding="utf-8", errors="replace").read()
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    out = {}
    for m in ARRAY.finditer(src):
        name, _decl, body = m.group(1), m.group(2), m.group(3)
        # base 0 chokes on the file's zero-padded decimals ("-004" reads as
        # octal and raises), so hex and decimal are separated explicitly
        vals = []
        for v in re.findall(r"-?(?:0[xX][0-9A-Fa-f]+|\d+)", body):
            vals.append(int(v, 16) if v.lower().lstrip("-").startswith("0x")
                        else int(v, 10))
        out[name] = vals
    return out


def kraft(lengths):
    """sum(2^-L).  Exactly 1.0 for a complete prefix code."""
    return sum(2.0 ** -L for L in lengths if L > 0)


def prefix_free(lengths, words):
    """No codeword may be a prefix of another.  -> (ok, first collision)."""
    seen = {}
    for i, (L, w) in enumerate(zip(lengths, words)):
        if L <= 0:
            continue
        bits = format(w & ((1 << L) - 1), f"0{L}b")
        for k in range(1, len(bits) + 1):
            p = bits[:k]
            if p in seen and (k == len(bits) or seen[p][1] == k):
                return False, (i, bits, seen[p][0])
        seen[bits] = (i, len(bits))
    # a real prefix test: walk every codeword against the set of prefixes
    words_set = set(seen)
    for bits in list(words_set):
        for k in range(1, len(bits)):
            if bits[:k] in words_set:
                return False, (seen[bits][0], bits, seen[bits[:k]][0])
    return True, None
