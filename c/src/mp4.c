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

/* -> track id of the first 'ac-4' track, or -1 */
static long ac4_track_id(rd_t *r)
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
            }
            if (!ok || r->bad)
                continue;
            /* stsd: full box header + entry_count, then entries */
            if (boxes(r, po + 16, po + psz, e, 1) >= 1
                && !memcmp(e[0].t, "ac-4", 4))
                found = tid;
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
    track_id = ac4_track_id(&r);
    r.bad = 0;
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
