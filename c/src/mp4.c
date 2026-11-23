/* SPDX-License-Identifier: Apache-2.0 */
/* Fragmented-MP4 AC-4 sample reader. Port of ac4/mp4.py (ISO/IEC 14496-12
 * moof/traf/tfhd/trun). Every read is bounds-checked: a read past the
 * input marks the structure being read as damaged (the moov gives no
 * track id, a moof is skipped), where the Python catches IndexError. */
#include <stdlib.h>
#include <string.h>

#include "ac4dec_mp4.h"

#define MAX_SAMPLES (1L << 16)
#define MAX_BOXES 4096

typedef struct {
    const uint8_t *d;
    size_t n;
    int bad;
} rd_t;

static uint64_t be(rd_t *r, size_t o, int nbytes)
{
    uint64_t v = 0;
    int i;
    if (o > r->n || (size_t)nbytes > r->n - o) {
        r->bad = 1;
        return 0;
    }
    for (i = 0; i < nbytes; i++)
        v = (v << 8) | r->d[o + (size_t)i];
    return v;
}

typedef struct {
    size_t o, sz;
    char t[4];
} box_t;

/* boxes in [start, end): yields boxes even when one overruns end (then
   stops), as the Python does */
static int boxes(rd_t *r, size_t start, size_t end, box_t *out, int max)
{
    size_t o = start;
    int n = 0;
    if (end > r->n)
        end = r->n;
    while (o + 8 <= end && n < max) {
        uint64_t sz = be(r, o, 4);
        size_t hdr = 8;
        if (sz == 1) {
            if (o + 16 > end)
                return n;
            sz = be(r, o + 8, 8);
            hdr = 16;
        } else if (sz == 0) {
            sz = end - o;
        }
        if (sz < hdr)
            return n;
        out[n].o = o;
        out[n].sz = (size_t)(sz > (uint64_t)SIZE_MAX / 2 ? SIZE_MAX / 2 : sz);
        memcpy(out[n].t, r->d + o + 4, 4);
        n++;
        if (o + out[n - 1].sz > end || o + out[n - 1].sz < o)
            return n;
        o += out[n - 1].sz;
    }
    return n;
}

static int child(rd_t *r, size_t o, size_t sz, const char *typ, size_t skip,
                 size_t *co, size_t *csz)
{
    box_t b[256];
    int n = boxes(r, o + skip, o + sz, b, 256), i;
    for (i = 0; i < n; i++) {
        if (!memcmp(b[i].t, typ, 4)) {
            *co = b[i].o;
            *csz = b[i].sz;
            return 1;
        }
    }
    return 0;
}

typedef struct {
    long tid;
    size_t stbl_o, stbl_sz;     /* the trak's stbl */
    size_t ent_o, ent_sz;       /* its 'ac-4' sample entry */
} trak_t;

/* the first 'ac-4' trak: -> its track id, or -1 */
static long ac4_trak(rd_t *r, trak_t *out)
{
    box_t *top = (box_t *)malloc(sizeof(box_t) * MAX_BOXES);
    box_t tr[256];
    int nt, i, j;
    long found = -1;
    if (!top)
        return -1;
    nt = boxes(r, 0, r->n, top, MAX_BOXES);
    for (i = 0; i < nt && found < 0; i++) {
        int ntr;
        if (memcmp(top[i].t, "moov", 4))
            continue;
        ntr = boxes(r, top[i].o + 8, top[i].o + top[i].sz, tr, 256);
        for (j = 0; j < ntr && found < 0; j++) {
            size_t to = tr[j].o, tsz = tr[j].sz, po, psz, ko, ksz;
            size_t so = 0, ssz = 0;
            long tid;
            int ver, ok = 1, q;
            static const char *path[4] = { "mdia", "minf", "stbl", "stsd" };
            box_t e[4];
            if (memcmp(tr[j].t, "trak", 4))
                continue;
            if (!child(r, to, tsz, "tkhd", 8, &ko, &ksz))
                continue;
            ver = (int)be(r, ko + 8, 1);
            tid = (long)be(r, ko + 12 + (ver == 1 ? 16 : 8), 4);
            po = to;
            psz = tsz;
            for (q = 0; q < 4; q++) {
                size_t no, nsz;
                if (!child(r, po, psz, path[q], 8, &no, &nsz)) {
                    ok = 0;
                    break;
                }
                po = no;
                psz = nsz;
                if (q == 2) {
                    so = no;
                    ssz = nsz;
                }
            }
            if (!ok || r->bad)
                continue;
            /* stsd: full box header + entry_count, then entries */
            if (boxes(r, po + 16, po + psz, e, 1) >= 1
                && !memcmp(e[0].t, "ac-4", 4)) {
                found = tid;
                out->tid = tid;
                out->stbl_o = so;
                out->stbl_sz = ssz;
                out->ent_o = e[0].o;
                out->ent_sz = e[0].sz;
            }
        }
    }
    free(top);
    if (r->bad)
        return -1;
    return found;
}

typedef struct {
    int bad;
    uint32_t flags;
    long n;
    int has_doff;
    long long doff;
    size_t sizes_at;        /* where the per-sample table starts */
    int per, soff, has_sizes;
} trun_t;

static void parse_trun(rd_t *r, size_t o, trun_t *t)
{
    size_t box_end = o + (size_t)be(r, o, 4), p, lim;
    int per = 0;
    memset(t, 0, sizeof(*t));
    t->flags = (uint32_t)be(r, o + 9, 3);
    t->n = (long)be(r, o + 12, 4);
    p = o + 16;
    if (t->flags & 0x001) {
        t->has_doff = 1;
        t->doff = (long long)(int32_t)(uint32_t)be(r, p, 4);
        p += 4;
    }
    if (t->flags & 0x004)
        p += 4;
    if (t->flags & 0x100) per += 4;
    if (t->flags & 0x200) per += 4;
    if (t->flags & 0x400) per += 4;
    if (t->flags & 0x800) per += 4;
    t->per = per;
    t->soff = (t->flags & 0x100) ? 4 : 0;
    lim = box_end < r->n ? box_end : r->n;
    if (t->n > MAX_SAMPLES || p + (size_t)t->n * (size_t)per > lim) {
        t->bad = 1;
        t->n = 0;
        return;
    }
    t->sizes_at = p;
    t->has_sizes = (t->flags & 0x200) != 0;
}

/* append samples of one moof to *v; -> 1 when a track was found */
static int fragment_samples(rd_t *r, size_t moof_o, size_t moof_sz,
                            long track_id, ac4mp4_sample **v, long *nv,
                            long *cap, size_t md_lo, size_t md_end)
{
    box_t tf[256], ch[64];
    int ntf = boxes(r, moof_o + 8, moof_o + moof_sz, tf, 256), i;
    size_t prev_end = moof_o;
    for (i = 0; i < ntf; i++) {
        int nch, j, k, have_tfhd = 0, n_trun = 0, any_bad = 0, all_sizes = 1;
        uint32_t tfl = 0;
        long tid = -1;
        int has_base = 0, has_dss = 0;
        uint64_t base = 0, dss = 0;
        trun_t tr[16];
        size_t pos = 0, start_n;
        int have_pos = 0, fail = 0;
        if (memcmp(tf[i].t, "traf", 4))
            continue;
        nch = boxes(r, tf[i].o + 8, tf[i].o + tf[i].sz, ch, 64);
        for (j = 0; j < nch; j++) {
            if (!memcmp(ch[j].t, "tfhd", 4)) {
                size_t o = ch[j].o, q;
                have_tfhd = 1;
                tfl = (uint32_t)be(r, o + 9, 3);
                tid = (long)be(r, o + 12, 4);
                q = o + 16;
                has_base = has_dss = 0;
                if (tfl & 0x01) {
                    base = be(r, q, 8);
                    has_base = 1;
                    q += 8;
                }
                if (tfl & 0x02)
                    q += 4;
                if (tfl & 0x08)
                    q += 4;
                if (tfl & 0x10) {
                    dss = be(r, q, 4);
                    has_dss = 1;
                }
            } else if (!memcmp(ch[j].t, "trun", 4) && n_trun < 16) {
                parse_trun(r, ch[j].o, &tr[n_trun]);
                if (tr[n_trun].bad)
                    any_bad = 1;
                if (!tr[n_trun].has_sizes)
                    all_sizes = 0;
                n_trun++;
            }
        }
        if (r->bad)
            return 0;
        if (!have_tfhd || !n_trun || any_bad)
            continue;
        if (!has_base)
            base = (tfl & 0x20000) ? moof_o : prev_end;
        start_n = (size_t)*nv;
        for (k = 0; k < n_trun && !fail; k++) {
            long s;
            if (tr[k].has_doff)
                pos = (size_t)((long long)base + tr[k].doff), have_pos = 1;
            else if (!have_pos)
                pos = (size_t)base, have_pos = 1;
            if (!tr[k].has_sizes && !has_dss) {
                fail = 1;
                break;
            }
            for (s = 0; s < tr[k].n; s++) {
                size_t sz = tr[k].has_sizes
                    ? (size_t)be(r, tr[k].sizes_at + (size_t)s * tr[k].per
                                 + tr[k].soff, 4)
                    : (size_t)dss;
                if (*nv >= *cap) {
                    long nc = *cap ? *cap * 2 : 1024;
                    ac4mp4_sample *nvp = (ac4mp4_sample *)realloc(
                        *v, sizeof(ac4mp4_sample) * (size_t)nc);
                    if (!nvp)
                        return -1;
                    *v = nvp;
                    *cap = nc;
                }
                if (md_lo + 8 <= pos && pos + sz <= md_end && pos + sz >= pos) {
                    (*v)[*nv].off = pos;
                    (*v)[*nv].len = sz;
                } else {
                    (*v)[*nv].off = 0;
                    (*v)[*nv].len = 0;
                }
                (*nv)++;
                pos += sz;
            }
        }
        if (r->bad)
            return 0;
        if (fail) {
            *nv = (long)start_n;
            continue;
        }
        prev_end = pos;
        if (track_id < 0 ? all_sizes : tid == track_id)
            return 1;
        *nv = (long)start_n;            /* not our track: drop */
    }
    return 0;
}

static int push(ac4mp4_sample **v, long *nv, long *cap, size_t off,
                size_t len)
{
    if (*nv >= *cap) {
        long nc = *cap ? *cap * 2 : 1024;
        ac4mp4_sample *n = (ac4mp4_sample *)realloc(
            *v, sizeof(ac4mp4_sample) * (size_t)nc);
        if (!n)
            return -1;
        *v = n;
        *cap = nc;
    }
    (*v)[*nv].off = off;
    (*v)[*nv].len = len;
    (*nv)++;
    return 0;
}

/* the samples of a stbl's sample table (stsz/stz2, stsc, stco/co64;
   ISO/IEC 14496-12 8.7), as mp4.py table_samples: appended to *v (a
   sample outside the file empty); a malformed table gives none.
   -> 0, or -1 when out of memory */
static int table_samples(rd_t *r, size_t so, size_t ssz,
                         ac4mp4_sample **v, long *nv, long *cap)
{
    size_t zo, zsz, z2o, z2sz, co, csz, c6o, c6sz, sco, scsz;
    int has_z = child(r, so, ssz, "stsz", 8, &zo, &zsz);
    int has_z2 = child(r, so, ssz, "stz2", 8, &z2o, &z2sz);
    int has_c = child(r, so, ssz, "stco", 8, &co, &csz);
    int has_c6 = child(r, so, ssz, "co64", 8, &c6o, &c6sz);
    int has_sc = child(r, so, ssz, "stsc", 8, &sco, &scsz);
    uint64_t n, nc, ne, fixed = 0, ri, k = 0;
    int fs = 0, w = 8;
    size_t sz_at, ch_at, sc_at;
    long start = *nv;
    if (!has_sc || !(has_c || has_c6) || !(has_z || has_z2) || r->bad)
        return 0;
    if (has_z) {
        fixed = be(r, zo + 12, 4);
        n = be(r, zo + 16, 4);
        if (n > (uint64_t)MAX_SAMPLES * 64
            || (!fixed && zo + 20 + 4 * n > zo + zsz))
            return 0;
        sz_at = zo + 20;
    } else {
        fs = (int)be(r, z2o + 15, 1);
        n = be(r, z2o + 16, 4);
        if ((fs != 4 && fs != 8 && fs != 16)
            || z2o + 20 + (n * (uint64_t)fs + 7) / 8 > z2o + z2sz)
            return 0;
        sz_at = z2o + 20;
    }
    if (has_c) {
        nc = be(r, co + 12, 4);
        if (co + 16 + 4 * nc > co + csz)
            return 0;
        ch_at = co + 16;
        w = 4;
    } else {
        nc = be(r, c6o + 12, 4);
        if (c6o + 16 + 8 * nc > c6o + c6sz)
            return 0;
        ch_at = c6o + 16;
        w = 8;
    }
    ne = be(r, sco + 12, 4);
    if (sco + 16 + 12 * ne > sco + scsz)
        return 0;
    sc_at = sco + 16;
    if (r->bad)
        return 0;
    for (ri = 0; ri < ne && k < n; ri++) {
        uint64_t first = be(r, sc_at + 12 * ri, 4);
        uint64_t per = be(r, sc_at + 12 * ri + 4, 4);
        uint64_t last = ri + 1 < ne ? be(r, sc_at + 12 * (ri + 1), 4) - 1
                                    : nc;
        uint64_t c;
        if (last > nc)
            last = nc;
        for (c = first ? first - 1 : 0; c < last && k < n; c++) {
            uint64_t pos = be(r, ch_at + (size_t)(w * c), w), q;
            for (q = 0; q < per && k < n; q++) {
                uint64_t sz;
                if (has_z) {
                    sz = fixed ? fixed : be(r, sz_at + 4 * k, 4);
                } else if (fs == 4) {
                    sz = (be(r, sz_at + k / 2, 1) >> (k % 2 == 0 ? 4 : 0))
                         & 15;
                } else {
                    sz = be(r, sz_at + (size_t)(fs / 8) * k, fs / 8);
                }
                if (r->bad) {
                    *nv = start;
                    return 0;
                }
                if (pos + sz <= r->n && pos + sz >= pos) {
                    if (push(v, nv, cap, (size_t)pos, (size_t)sz) < 0)
                        return -1;
                } else if (push(v, nv, cap, 0, 0) < 0) {
                    return -1;
                }
                pos += sz;
                k++;
            }
        }
    }
    return 0;
}

int ac4mp4_dac4(const uint8_t *data, size_t len, const uint8_t **dsi,
                size_t *dsi_len)
{
    rd_t r;
    trak_t t;
    box_t b[64];
    int n, i;
    r.d = data;
    r.n = len;
    r.bad = 0;
    *dsi = NULL;
    *dsi_len = 0;
    if (ac4_trak(&r, &t) < 0)
        return 0;
    n = boxes(&r, t.ent_o + 36, t.ent_o + t.ent_sz, b, 64);
    for (i = 0; i < n; i++) {
        if (!memcmp(b[i].t, "dac4", 4) && b[i].o + b[i].sz <= len) {
            *dsi = data + b[i].o + 8;
            *dsi_len = b[i].sz - 8;
            return 1;
        }
    }
    return 0;
}

long ac4mp4_samples(const uint8_t *data, size_t len, ac4mp4_sample **out)
{
    rd_t r;
    box_t *top;
    long track_id, nv = 0, cap = 0;
    int nt, i;
    ac4mp4_sample *v = NULL;
    *out = NULL;
    r.d = data;
    r.n = len;
    r.bad = 0;
    {
        trak_t t;
        track_id = ac4_trak(&r, &t);
        r.bad = 0;
        /* a non-fragmented file (or the moov part of one) */
        if (track_id >= 0 && table_samples(&r, t.stbl_o, t.stbl_sz, &v, &nv,
                                           &cap) < 0) {
            free(v);
            return -1;
        }
        r.bad = 0;
    }
    {
        int max = MAX_BOXES;
        for (;;) {
            top = (box_t *)malloc(sizeof(box_t) * (size_t)max);
            if (!top)
                return -1;
            nt = boxes(&r, 0, len, top, max);
            if (nt < max || max >= (1 << 26))
                break;
            free(top);
            max *= 4;
        }
    }
    for (i = 0; i < nt; i++) {
        long before = nv;
        int rc;
        if (memcmp(top[i].t, "moof", 4))
            continue;
        if (i + 1 >= nt || memcmp(top[i + 1].t, "mdat", 4)
            || top[i + 1].o + top[i + 1].sz > len)
            continue;
        r.bad = 0;
        rc = fragment_samples(&r, top[i].o, top[i].sz, track_id, &v, &nv,
                              &cap, top[i + 1].o,
                              top[i + 1].o + top[i + 1].sz);
        if (rc < 0) {
            free(v);
            free(top);
            return -1;
        }
        if (rc == 0 || r.bad)
            nv = before;
    }
    free(top);
    if (nv == 0) {
        free(v);
        v = NULL;
    }
    *out = v;
    return nv;
}
