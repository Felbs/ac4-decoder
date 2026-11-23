/* SPDX-License-Identifier: Apache-2.0 */
/* AC-4 inputs: sync frames, MPEG-TS, MP4, and detection. Port of
 * ac4/container.py (see include/ac4dec_input.h). */
#include <stdlib.h>
#include <string.h>

#include "ac4dec_input.h"
#include "ac4dec_mp4.h"

/* ------------------------------------------------------------ CRC */
unsigned ac4in_crc16(const uint8_t *d, size_t n, unsigned crc)
{
    size_t i;
    int k;
    for (i = 0; i < n; i++) {
        crc ^= (unsigned)d[i] << 8;
        for (k = 0; k < 8; k++)
            crc = (crc & 0x8000) ? ((crc << 1) ^ 0x8005) : (crc << 1);
        crc &= 0xFFFF;
    }
    return crc;
}

/* ------------------------------------------------------------ ID3 */
static size_t skip_id3(const uint8_t *d, size_t n, size_t o)
{
    while (n - o >= 10 && n >= o && !memcmp(d + o, "ID3", 3)) {
        size_t sz = 0;
        int i;
        for (i = 6; i < 10; i++)
            sz = (sz << 7) | (d[o + (size_t)i] & 0x7F);
        o += 10 + sz + ((d[o + 5] & 0x10) ? 10 : 0);
        if (o > n)
            return n;
    }
    return o;
}

/* ------------------------------------------------------------ sync */
static int is_sync(const uint8_t *d, size_t n, size_t o)
{
    return o + 2 <= n && d[o] == 0xAC && (d[o + 1] == 0x40 || d[o + 1] == 0x41);
}

typedef struct {
    size_t hdr, fs, total;
    int crc;
} sf_t;

static int frame_at(const uint8_t *d, size_t n, size_t o, sf_t *f)
{
    if (o > n || n - o < 4 || !is_sync(d, n, o))
        return 0;
    f->fs = ((size_t)d[o + 2] << 8) | d[o + 3];
    f->hdr = 4;
    if (f->fs == 0xFFFF) {
        if (n - o < 7)
            return 0;
        f->fs = ((size_t)d[o + 4] << 16) | ((size_t)d[o + 5] << 8) | d[o + 6];
        f->hdr = 7;
    }
    f->crc = d[o + 1] == 0x41;
    f->total = f->hdr + f->fs + (f->crc ? 2 : 0);
    return 1;
}

static int plausible(const uint8_t *d, size_t n, size_t o)
{
    sf_t f;
    size_t nx;
    if (!frame_at(d, n, o, &f) || f.fs == 0)
        return 0;
    nx = o + f.total;
    if (f.crc && f.total <= n - o && ac4in_crc16(d + o + 2, f.total - 2, 0) == 0)
        return 1;
    return nx == n || is_sync(d, n, nx)
           || (nx + 3 <= n && !memcmp(d + nx, "ID3", 3));
}

typedef struct {
    ac4in_frame *v;
    long n, cap;
} fvec;

static int fpush(fvec *fv, const uint8_t *p, size_t len)
{
    if (fv->n >= fv->cap) {
        long nc = fv->cap ? fv->cap * 2 : 1024;
        ac4in_frame *nv = (ac4in_frame *)realloc(fv->v,
                                                 sizeof(*nv) * (size_t)nc);
        if (!nv)
            return -1;
        fv->v = nv;
        fv->cap = nc;
    }
    fv->v[fv->n].p = len ? p : NULL;
    fv->v[fv->n].len = len;
    fv->n++;
    return 0;
}

static int sync_frames(const uint8_t *d, size_t n, fvec *fv,
                       ac4in_info *info)
{
    size_t o = 0;
    while (o < n) {
        sf_t f;
        o = skip_id3(d, n, o);
        if (o >= n)
            break;
        if (!frame_at(d, n, o, &f) || f.fs == 0
            || (o + f.total < n && !plausible(d, n, o))) {
            size_t p = o + 1;
            while (p + 1 < n) {
                if (is_sync(d, n, p) && plausible(d, n, p))
                    break;
                p++;
            }
            if (p + 1 >= n)
                p = n;
            info->n_skipped += (long)(p - o);
            o = p;
            continue;
        }
        if (f.total > n - o) {
            info->n_truncated++;
            break;
        }
        info->n_frames++;
        if (f.crc) {
            info->n_crc++;
            if (ac4in_crc16(d + o + 2, f.total - 2, 0) != 0) {
                info->n_crc_bad++;
                if (fpush(fv, NULL, 0) < 0)
                    return -1;
                o += f.total;
                continue;
            }
        }
        if (fpush(fv, d + o + f.hdr, f.fs) < 0)
            return -1;
        o += f.total;
    }
    return 0;
}

/* ------------------------------------------------------------ TS */
typedef struct {
    uint8_t *d;
    size_t n, cap;
} bvec;

static int bput(bvec *b, const uint8_t *p, size_t n)
{
    if (b->n + n > b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 65536;
        uint8_t *nd;
        while (nc < b->n + n)
            nc *= 2;
        nd = (uint8_t *)realloc(b->d, nc);
        if (!nd)
            return -1;
        b->d = nd;
        b->cap = nc;
    }
    if (n)
        memcpy(b->d + b->n, p, n);
    b->n += n;
    return 0;
}

/* -> the first AC-4 PID a PMT section lists, or -1 */
static long pmt_ac4_pid(const uint8_t *sec, size_t len)
{
    size_t n, end, pil, k;
    if (len < 12 || sec[0] != 0x02)
        return -1;
    n = ((size_t)(sec[1] & 0x0F) << 8) | sec[2];
    end = 3 + n >= 4 ? 3 + n - 4 : 0;
    if (end > len)
        end = len;
    pil = ((size_t)(sec[10] & 0x0F) << 8) | sec[11];
    k = 12 + pil;
    while (k + 5 <= end) {
        int st = sec[k];
        long pid = ((long)(sec[k + 1] & 0x1F) << 8) | sec[k + 2];
        size_t il = ((size_t)(sec[k + 3] & 0x0F) << 8) | sec[k + 4];
        size_t j = 0, dl = il;
        const uint8_t *desc = sec + k + 5;
        if (k + 5 + dl > len)
            dl = len - (k + 5);
        k += 5 + il;
        if (st != 0x06)
            continue;
        while (j + 2 <= dl) {
            int tag = desc[j];
            size_t ln = desc[j + 1];
            const uint8_t *body = desc + j + 2;
            size_t bl = j + 2 + ln <= dl ? ln : dl - (j + 2);
            if ((tag == 0x7F && bl >= 1 && body[0] == 0x15)
                || (tag == 0x05 && bl >= 4 && !memcmp(body, "AC-4", 4)))
                return pid;
            j += 2 + ln;
        }
    }
    return -1;
}

static int pes_payload(const uint8_t *pes, size_t n, bvec *out)
{
    size_t hl, plen, body;
    if (n < 9 || pes[0] || pes[1] || pes[2] != 1)
        return 0;
    hl = pes[8];
    plen = ((size_t)pes[4] << 8) | pes[5];
    body = 9 + hl;
    if (body > n)
        return 0;
    n -= body;
    if (plen) {
        size_t lim = plen >= 3 + hl ? plen - 3 - hl : 0;
        if (n > lim)
            n = lim;
    }
    return bput(out, pes + body, n);
}

static int ts_es(const uint8_t *d, size_t n, bvec *es, ac4in_info *info)
{
    size_t o = 0;
    long pmt[16];
    int n_pmt = 0, i, last_cc = -1;
    long target = -1;
    bvec cur = { NULL, 0, 0 };
    int have_cur = 0, rc = 0;
    while (o + 188 <= n) {
        const uint8_t *pk;
        int pusi, afc, cc;
        long pid;
        size_t i0;
        if (d[o] != 0x47) {
            size_t p = o + 1;
            while (p + 376 < n && !(d[p] == 0x47 && d[p + 188] == 0x47
                                    && d[p + 376] == 0x47))
                p++;
            if (p + 376 >= n)
                break;
            o = p;
            continue;
        }
        pk = d + o;
        o += 188;
        info->ts_packets++;
        pid = ((long)(pk[1] & 0x1F) << 8) | pk[2];
        if (pk[1] & 0x80) {                 /* transport_error_indicator */
            if (target >= 0 && pid == target)
                info->ts_cc_errors++;
            continue;
        }
        pusi = (pk[1] & 0x40) != 0;
        afc = (pk[3] >> 4) & 3;
        cc = pk[3] & 0x0F;
        i0 = 4;
        if (afc & 2)
            i0 += 1 + (size_t)pk[4];
        if (!(afc & 1) || i0 > 188)
            continue;
        if (pid == 0 && pusi && target < 0) {
            const uint8_t *s = pk + i0 + 1 + pk[i0];
            size_t sl = 188 > i0 + 1 + pk[i0] ? 188 - (i0 + 1 + pk[i0]) : 0;
            if (sl >= 8 && s[0] == 0x00) {
                size_t sn = ((size_t)(s[1] & 0x0F) << 8) | s[2], k;
                size_t end = 3 + sn >= 4 ? 3 + sn - 4 : 0;
                if (end > sl)
                    end = sl;
                for (k = 8; k + 4 <= end; k += 4) {
                    int prog = (s[k] << 8) | s[k + 1];
                    long pm = ((long)(s[k + 2] & 0x1F) << 8) | s[k + 3];
                    int known = 0;
                    for (i = 0; i < n_pmt; i++)
                        known |= pmt[i] == pm;
                    if (prog && !known && n_pmt < 16)
                        pmt[n_pmt++] = pm;
                }
            }
            continue;
        }
        if (pusi && target < 0) {
            int is_pmt = 0;
            for (i = 0; i < n_pmt; i++)
                is_pmt |= pmt[i] == pid;
            if (is_pmt) {
                size_t off = i0 + 1 + pk[i0];
                if (off < 188)
                    target = pmt_ac4_pid(pk + off, 188 - off);
                continue;
            }
        }
        if (target < 0 || pid != target)
            continue;
        info->ts_pid = target;
        if (last_cc >= 0 && cc != ((last_cc + 1) & 15) && cc != last_cc)
            info->ts_cc_errors++;
        if (last_cc >= 0 && cc == last_cc) {
            continue;                       /* a duplicate packet */
        }
        last_cc = cc;
        if (pusi) {
            if (have_cur && pes_payload(cur.d, cur.n, es) < 0) {
                rc = -1;
                break;
            }
            cur.n = 0;
            have_cur = 1;
            info->ts_pes++;
            if (bput(&cur, pk + i0, 188 - i0) < 0) {
                rc = -1;
                break;
            }
        } else if (have_cur && bput(&cur, pk + i0, 188 - i0) < 0) {
            rc = -1;
            break;
        }
    }
    if (rc == 0 && have_cur && pes_payload(cur.d, cur.n, es) < 0)
        rc = -1;
    free(cur.d);
    return rc;
}

/* ------------------------------------------------------------ detection */
static const char *MP4_TOP[] = {
    "ftyp", "styp", "moov", "moof", "sidx", "free", "skip", "mdat", "emsg",
    "prft", "uuid", "wide"
};

ac4in_kind ac4in_detect(const uint8_t *d, size_t n)
{
    size_t o, p, lim;
    unsigned i;
    if (n >= 188 * 2 + 1 && d[0] == 0x47 && d[188] == 0x47)
        return AC4IN_TS;
    o = skip_id3(d, n, 0);
    if (o < n && n - o >= 4 && is_sync(d, n, o))
        return AC4IN_SYNC;
    if (n >= 8)
        for (i = 0; i < sizeof(MP4_TOP) / sizeof(MP4_TOP[0]); i++)
            if (!memcmp(d + 4, MP4_TOP[i], 4))
                return AC4IN_MP4;
    lim = o + ((size_t)1 << 16);
    if (lim > n - (n ? 1 : 0))
        lim = n ? n - 1 : 0;
    for (p = o; p < lim; p++)
        if (is_sync(d, n, p) && plausible(d, n, p))
            return AC4IN_SYNC;
    return AC4IN_UNKNOWN;
}

/* the top fields of ac4_dsi_v1 (TS 103 190-2 E.6) */
static void dsi_info(const uint8_t *p, size_t n, ac4in_info *info)
{
    uint32_t v;
    if (n < 3)
        return;
    v = ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2];
    info->has_dsi = 1;
    info->dsi_bitstream_version = (int)((v >> 14) & 0x7F);
    info->dsi_fs_index = (int)((v >> 13) & 1);
    info->dsi_frame_rate_index = (int)((v >> 9) & 15);
    info->dsi_n_presentations = (int)(v & 0x1FF);
}

long ac4in_read(const uint8_t *data, size_t len, ac4in_frame **frames,
                ac4in_info *info)
{
    fvec fv = { NULL, 0, 0 };
    memset(info, 0, sizeof(*info));
    info->ts_pid = -1;
    *frames = NULL;
    info->kind = ac4in_detect(data, len);
    if (info->kind == AC4IN_SYNC) {
        if (sync_frames(data, len, &fv, info) < 0) {
            free(fv.v);
            return -1;
        }
    } else if (info->kind == AC4IN_TS) {
        bvec es = { NULL, 0, 0 };
        if (ts_es(data, len, &es, info) < 0) {
            free(es.d);
            return -1;
        }
        info->buf = es.d;
        if (es.n && sync_frames(es.d, es.n, &fv, info) < 0) {
            free(fv.v);
            return -1;
        }
    } else if (info->kind == AC4IN_MP4) {
        ac4mp4_sample *s = NULL;
        long n = ac4mp4_samples(data, len, &s), i;
        const uint8_t *dp;
        size_t dl;
        if (n < 0)
            return -1;
        for (i = 0; i < n; i++) {
            if (fpush(&fv, s[i].len ? data + s[i].off : NULL, s[i].len) < 0) {
                free(s);
                free(fv.v);
                return -1;
            }
        }
        free(s);
        if (ac4mp4_dac4(data, len, &dp, &dl))
            dsi_info(dp, dl, info);
        info->n_frames = fv.n;
    }
    *frames = fv.v;
    return fv.n;
}

void ac4in_free(ac4in_info *info)
{
    if (info) {
        free(info->buf);
        info->buf = NULL;
    }
}
