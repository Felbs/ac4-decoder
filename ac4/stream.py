# SPDX-License-Identifier: Apache-2.0
"""M42 -- a resumable AC-4 decoder front half, for live use.

`m37.decode_all` reads a whole file and synthesises once. A live decoder has
to emit as it goes, and the only way to do that without re-deriving the past
is to hold the state that actually crosses a frame boundary:

    T, tabs   the Annex B tables and Huffman books -- constant, but building
              them per pass was pure waste
    cfg       the A-SPX configuration; it arrives on I-FRAMES ONLY, so a
              decoder that forgets it goes silent above the crossover until
              the next one
    fstate    A-SPX envelopes are DELTA CODED against the previous frame

This class holds all three and turns frames into spectra. The MDCT overlap
is the caller's to carry (m30.synthesise takes and returns it), because the
caller is the one who knows where a block boundary is.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from . import toc as M                                         # noqa: E402
from . import channels as C                                         # noqa: E402
from . import audio as R                                            # noqa: E402
from . import aspx_parse as P                                       # noqa: E402
from . import hcb as H                                              # noqa: E402
from . import spectral as S                                         # noqa: E402
from . import aspx_bands as A                                       # noqa: E402
from . import aspx_render as R37                                      # noqa: E402
from . import robust as RB                                          # noqa: E402
from . import acpl_parse as AP                                      # noqa: E402

# Marks a CONCEALED frame in the A-SPX group lists (where None means "no
# A-SPX data for this frame"). The renderer mutes the high band for it and
# leaves its envelope state untouched (hold).
CONCEALED = RB.CONCEALED


class Ac4Stream:
    def __init__(self, element="5_X"):
        # "5_X" = the 5.1 programme element. "pair" = the stereo
        # channel_pair_element (Table 22).
        self.element = element
        self.acpl3 = False      # 13.25: set once the stream shows a 2-ch core + A-CPL
        self.T = C.load_tables()
        arrays = H.parse_c(H.DEFAULT_C)
        self.tabs = {n[:-4]: S.Huff(arrays[n], arrays[n[:-4] + "_CW"])
                     for n in arrays
                     if n.endswith("_LEN") and n.startswith("ASPX_")}
        # A-CPL books, for bit accounting of ASPX_ACPL_3 frames only
        self.acpl_books = AP.load_books(arrays)
        self.acpl_cfg = None        # acpl_config_2ch, from the last I-frame
        for nm in self.tabs:
            P.CENTRE[nm] = 0 if nm.endswith("_F0") else \
                (len(arrays[nm + "_LEN"]) - 1) // 2
        self.nats = A.num_aspx_timeslots()
        self.cfg = None
        self.fstate = {}
        self.n_ok = 0
        self.n_bad = 0
        # robustness state (see robust.py)
        self.damage = RB.DamageLog()
        self.toc_shape = None       # the stream's TOC layout, from frame 1
        self.mode = None            # the element's codec mode, from frame 1
        self.seq_prev = None        # sequence_counter of the last frame
        self.relearn = RB.Relearn()  # a change of TOC layout / codec mode
        self.log = []
        self.n_seq_jumps = 0
        # per frame of the LAST decode_frames call: None = decoded, else the
        # reason it was concealed; and whether its counter broke sequence
        self.status = []
        self.seq_break = []

    # 5.1 channel order everywhere downstream: matches m29's proven batch
    # render AND the standard WAV 6-channel layout (FL FR FC LFE BL BR), so a
    # plain 6-channel wav is a correct 5.1 file with no channel map needed.
    CHANS = ("L", "R", "C", "lfe", "Ls", "Rs")

    def decode_frames(self, frames, chans=None):
        """-> (wins per channel, lengths per channel, groups {"lr","sr"}).

        ALL SIX CHANNELS. `decode_element` always decoded the whole 5.1
        element -- this method used to keep only the first stereo pair and
        throw away C, LFE, Ls, Rs after their bits were already spent
        (found 8/07 via the spec index: TS 103 190-1 Table 25).

        EACH CHANNEL CARRIES ITS OWN FRAMING (m29's lesson, kept): the two
        pairs, the centre and the LFE are signalled independently and can
        differ in window count and lengths, so nothing here assumes one
        window sequence. Every frame still contributes exactly one frame of
        samples per channel, which is what keeps the six lanes equal-length
        and the timeline frame-accurate through bad frames.

        A-SPX: Table 25 puts THREE blocks in sequence -- aspx_data_2ch (L/R),
        aspx_data_2ch (Ls/Rs), aspx_data_1ch (C) -- and the parser's state is
        keyed by group index, so the pairs get groups 0 and 1. The centre's
        1-ch block is parsed to keep its delta-state warm, but not yet
        rendered: only apply_hf_pair exists, and a rushed mono HF port would
        put its mistakes on DIALOG. Core-band centre until that is built and
        gated.
        """
        # `chans` limits which channels are PACKED and SYNTHESISED -- the
        # element decode itself always walks all six (the bits are one
        # stream). ("L","R") restores the old stereo cost: measured 1.27x
        # real time against 0.62x for six channels even threaded, because
        # the filterbank is many small windows and the GIL never lets the
        # threads overlap. Live keeps up in stereo; recordings render 5.1.
        chans = tuple(chans) if chans else self.CHANS
        if self.element == "pair":
            chans = ("L", "R")           # a pair element HAS only a pair
        wins = {k: [] for k in chans}
        lengths = {k: [] for k in chans}
        counts = {k: [] for k in chans}   # windows per FRAME, per channel --
        # the grouped filterbank needs frame boundaries because channels'
        # window lists do not align positionally when framings diverge
        groups = {"lr": [], "sr": []}
        self.status = []
        self.seq_break = []
        pending = []            # frames decoded before any aspx_config
        for fi, f in enumerate(frames):
            wpos = {k: len(v) for k, v in wins.items()}
            fstate0 = dict(self.fstate)
            seq = None
            brk = False
            resume = None       # the counter to expect after an outlier
            on = RB.CHECKS
            try:
                if on:
                    RB.check_frame_size(f)
                try:
                    st = M.parse(f)
                except Exception as e:                         # noqa: BLE001
                    raise RB.Damaged("toc.parse", str(e)[:60])
                # a TOC or codec mode that differs from the stream's is
                # judged only after the frame has passed every other check
                # (RB.Relearn): a real change is accepted at an I-frame, or
                # after RB.RELEARN frames in a row that agree on it
                changed = None
                if on:
                    try:
                        RB.check_toc(st, f, self.toc_shape)
                    except RB.Damaged as e:
                        if e.reason not in RB.RELEARNABLE:
                            raise
                        changed = e
                seq = st["sequence_counter"]
                brk = not RB.seq_continues(seq, self.seq_prev)
                if on and brk and seq != 0 and fi + 1 < len(frames):
                    # one frame of lookahead: when the NEXT frame carries on
                    # from the frame before this one (and not from this
                    # one), this frame is an outlier -- swapped, repeated
                    # or from another stream -- and is concealed
                    nxt = RB.peek_counter(frames[fi + 1])
                    prev = self.seq_prev
                    if (nxt is not None
                            and not RB.seq_continues(nxt, seq)
                            and (RB.seq_continues(nxt, prev)
                                 or nxt == RB.seq_next(RB.seq_next(prev)))):
                        resume = RB.seq_prev_of(nxt)
                        raise RB.Damaged("seq.outlier",
                                         f"{prev} -> {seq} -> {nxt}")
                # PAYLOAD BASE (TS 103 190-1 4.3.3.2.10/.11 + Pseudocode 1):
                # substream offsets are measured from the end of the
                # byte-aligned ac4_toc and START at payload_base --
                #     substream_n_offset = payload_base
                #     for (s = 0; s < n; s++) offset += substream_size[s]
                # -- and payload_base is 0 ONLY when b_payload_base is false
                # ("the first ac4_substream_data shall immediately follow the
                # ac4_toc element").  Omitting it was invisible for two years
                # because every capture of the first broadcaster signals
                # b_payload_base = false; the second signals payload_base
                # = 1 on every frame, so the element
                # was read one byte early and decoded as noise (E77).
                o = (st["toc_bytes"] + st.get("payload_base", 0)
                     + st["substream_sizes"][0])
                sub = f[o:o + st["substream_sizes"][1]]
                ifr = bool(st["b_iframe_global"])
                if self.element == "pair":
                    from . import pair as PAIR
                    r = PAIR.decode_pair(sub, self.T, b_iframe=ifr)
                else:
                    r = C.decode_element(sub, self.T, b_iframe=ifr)
                if r["bits"] > r["nbits"]:
                    raise RB.Damaged("element.parse",
                                     "element overran its substream")
                # a different codec mode: a foreign or damaged frame, or a
                # real change of source (4.3.3.2.2); judged like the TOC
                if (on and changed is None and self.mode is not None
                        and r["mode"] != self.mode):
                    changed = RB.Damaged("element.mode", f"{r['mode']} in "
                                         f"a {self.mode} stream")
                cfg = r.get("aspx") or self.cfg
                if on and r.get("aspx"):
                    try:
                        t_new = R37.t_for(r["aspx"])
                        ok_cfg = (t_new["n_hi"] >= 1 and t_new["n_lo"] >= 1
                                  and t_new["num_sb_aspx"] >= 1)
                    except Exception:                          # noqa: BLE001
                        ok_cfg = False
                    if not ok_cfg:
                        raise RB.Damaged("aspx.config", str(r["aspx"]))
                # A-SPX: Table 25 / Table 22 put the A-SPX blocks right after
                # the channel data; parse them all (the 5_X element: L/R,
                # Ls/Rs and the centre) so the frame closes on audio_size
                aspx_lr = aspx_sr = None
                end = r["bits"]
                acpl3 = bool(r.get("acpl3"))
                five = self.element != "pair" and not acpl3
                has_aspx = not (self.element == "pair" and r["mode"][1] == 0)
                if cfg and has_aspx:
                    from .toc import Bits
                    b = Bits(sub)
                    b.p = r["bits"]
                    t = R37.t_for(cfg)
                    try:
                        aspx_lr = P.aspx_data(b, self.tabs, t, cfg,
                                              self.nats, ifr, True,
                                              self.fstate, 0)
                        if five:
                            aspx_sr = P.aspx_data(b, self.tabs, t, cfg,
                                                  self.nats, ifr, True,
                                                  self.fstate, 1)
                            P.aspx_data(b, self.tabs, t, cfg,
                                        self.nats, ifr, False,
                                        self.fstate, 2)
                    except Exception as e:                     # noqa: BLE001
                        if on:
                            raise RB.Damaged("aspx.parse", str(e)[:60]) \
                                if not isinstance(e, RB.Damaged) else e
                        # (checks off: a later block mis-parsing does not
                        # cost an earlier one, as before)
                    end = b.p
                    # ASPX_ACPL_3: acpl_data_2ch() closes the frame
                    acfg = r.get("acpl") or self.acpl_cfg
                    if on and acpl3 and acfg:
                        try:
                            AP.acpl_data_2ch(b, self.acpl_books, acfg)
                        except Exception as e:                 # noqa: BLE001
                            raise RB.Damaged("acpl.parse", str(e)[:60])
                        end = b.p
                r["end"] = end
                # every block of audio_data() parsed?  Not before the first
                # aspx_config (the A-SPX blocks cannot be sized) nor, for
                # ASPX_ACPL_3, before the first acpl_config_2ch
                complete = ((bool(cfg) or not has_aspx)
                            and (not acpl3 or bool(r.get("acpl")
                                                   or self.acpl_cfg)))
                if on:
                    RB.check_closure(r, len(sub), complete)
                    if not cfg and has_aspx:
                        pending.append(dict(
                            fi=fi, sub=sub, bits=r["bits"], ifr=ifr,
                            start=r["audio_start"], size=r["audio_size"],
                            acpl3=acpl3, five=five, wpos=wpos))
                # noise fill seeded by the stream's own sequence counter,
                # exactly as the batch render does
                rng = np.random.default_rng(st["sequence_counter"])
                # each channel's OWN framing: with b_enable_mdct_stereo_proc
                # = 0 the two channels of a pair carry separate sf_info and
                # can differ (with it on, both are the shared one)
                fm = {"L": r["L"]["framing"], "R": r["R"]["framing"]}
                # ASPX_ACPL_3 (E77) carries the 5.1 programme as a TWO-channel
                # core plus A-CPL; Ls/Rs/C do not exist as coded channels, so
                # this frame has only a pair to give no matter what `chans`
                # asked for. Narrow the request instead of KeyError-ing.
                if r.get("acpl3"):
                    chans = ("L", "R")
                    self.acpl3 = True
                    # 13.25: the output lists were built for the REQUESTED
                    # channels; from here every frame gives only a pair, so
                    # C / Ls / Rs / lfe would keep a shorter history than
                    # L / R and the grouped filterbank would index past
                    # their frame counts (IndexError in synthesise_frames,
                    # the second broadcaster's 5.1, live and banked).  Drop them.
                    for k in list(wins):
                        if k not in chans:
                            wins.pop(k); lengths.pop(k); counts.pop(k)
                elif self.element != "pair":
                    fm.update({"Ls": r["Ls"]["framing"],
                               "Rs": r["Rs"]["framing"],
                               "C": r["C"]["framing"],
                               "lfe": r["lfe"]["framing"]})
                pk = {k: R.packed_spectrum(r[k], fm[k], rng)
                      for k in chans}
                if on:
                    for k, v in pk.items():
                        if len(v) and not (np.abs(v).max()
                                           <= RB.MAX_SPECTRAL_LINE):
                            raise RB.Damaged("asf.level", k)
                if changed is not None:
                    key = (changed.reason, RB.toc_shape(st), r["mode"])
                    at_i = ifr
                    if ifr and changed.reason != "element.mode":
                        # a TOC change at an I-frame: the next frame's TOC
                        # must agree (one damaged I-frame is not a change)
                        at_i = RB.next_toc_agrees(frames, fi, key[1])
                    if not self.relearn.accept(key, at_i):
                        raise changed
                    self.log.append(f"stream changed ({changed}) -- "
                                    "accepted " + ("at an I-frame" if ifr
                                                   else f"after {RB.RELEARN}"
                                                   " frames"))
                    self.toc_shape = RB.toc_shape(st)
                else:
                    self.relearn.reset()
                # ---- the frame is accepted: commit its state
                if r.get("aspx"):
                    self.cfg = r["aspx"]
                if r.get("acpl"):
                    self.acpl_cfg = r["acpl"]
                if self.toc_shape is None:
                    self.toc_shape = RB.toc_shape(st)
                self.mode = r["mode"]
                if "L" in pk:
                    pk["L"], pk["R"] = R.unmix(pk["L"], pk["R"],
                                               r["st_lr"], fm["L"])
                if "Ls" in pk:
                    pk["Ls"], pk["Rs"] = R.unmix(pk["Ls"], pk["Rs"],
                                                 r["st_sr"], fm["Ls"])
                u, seq_l = {}, {}
                for k in chans:
                    u[k], seq_l[k] = R.ungroup(pk[k], fm[k])
                got_lr = aspx_lr
                got_sr = aspx_sr if "Ls" in pk else None
                self.n_ok += 1
                self.status.append(None)
                self.seq_break.append(brk)
                if brk:
                    self.n_seq_jumps += 1
                self.seq_prev = seq
            except Exception as e:                             # noqa: BLE001
                # A bad frame must not desynchronise the timeline: emit
                # silence of the right length so every downstream index
                # stays frame-accurate.  Nothing it parsed is kept: the
                # A-SPX framing state goes back to before it (hold).
                self.fstate = fstate0
                reason = RB.reason_of(e)
                u = {k: [np.zeros(1536)] for k in chans}
                seq_l = {k: [1536] for k in chans}
                got_lr = got_sr = CONCEALED
                self.n_bad += 1
                self.damage.add(reason)
                self.status.append(reason)
                self.seq_break.append(False)
                # the next frame is expected to carry the counter this one
                # should have had
                # (a frame whose own counter the next frame continues is
                # taken at its word: after a source change, every later
                # frame would otherwise look like a break too)
                nxt = (RB.peek_counter(frames[fi + 1])
                       if fi + 1 < len(frames) else None)
                if resume is not None:
                    self.seq_prev = resume
                elif (seq is not None and nxt is not None
                        and RB.seq_continues(nxt, seq)):
                    self.seq_prev = seq
                elif self.seq_prev is not None:
                    self.seq_prev = RB.seq_next(self.seq_prev)
            for k in chans:
                wins[k].extend(u[k])
                lengths[k].extend(seq_l[k])
                counts[k].append(len(seq_l[k]))
            groups["lr"].append(got_lr)
            groups["sr"].append(got_sr)
        if pending and self.cfg and RB.CHECKS:
            self._late_closure(pending, wins, lengths, counts, groups)
        return wins, lengths, groups, counts

    def _late_closure(self, pending, wins, lengths, counts, groups):
        """Frames decoded before the stream's first aspx_config could only
        be checked against an upper bound. When an I-frame later in the
        same pass brings the configuration, parse their A-SPX (and A-CPL)
        blocks with it -- for accounting only -- and conceal the ones that
        do not close. Only when at least LATE_MAJORITY of the judged frames
        close: otherwise the configuration evidently changed at the I-frame
        and nothing is concluded. Frames whose framing needs a
        previous_stop_pos from before the pass are not judged, nor are
        frames concealed for another reason (each frame is counted once,
        under its first reason)."""
        from .toc import Bits
        cfg, acfg = self.cfg, self.acpl_cfg
        t = R37.t_for(cfg)
        state = RB.TrackingState()
        verdicts = []
        for pd in pending:
            if self.status[pd["fi"]] is not None:
                # concealed already (a rule checked after the frame was
                # put on this list): it is not judged again, counted
                # again, nor does its A-SPX feed the tracking state (the
                # decode rolled its state back too)
                continue
            state.missing = False
            b = Bits(pd["sub"])
            b.p = pd["bits"]
            try:
                P.aspx_data(b, self.tabs, t, cfg, self.nats, pd["ifr"],
                            True, state, 0)
                if pd["five"]:
                    P.aspx_data(b, self.tabs, t, cfg, self.nats, pd["ifr"],
                                True, state, 1)
                    P.aspx_data(b, self.tabs, t, cfg, self.nats, pd["ifr"],
                                False, state, 2)
                if pd["acpl3"]:
                    if not acfg:
                        continue
                    AP.acpl_data_2ch(b, self.acpl_books, acfg)
                stop = pd["start"] + 8 * pd["size"]
                ok = stop - 7 <= b.p <= stop
            except Exception:                                  # noqa: BLE001
                ok = False
            if not state.missing:
                verdicts.append((pd, ok))
            if not ok:
                # a frame that did not parse leaves no state to trust
                state = RB.TrackingState()
        if not verdicts:
            return
        if sum(ok for _, ok in verdicts) < RB.LATE_MAJORITY * len(verdicts):
            return
        for pd, ok in sorted(verdicts, key=lambda v: -v[0]["fi"]):
            if ok:
                continue
            fi = pd["fi"]
            for k in wins:
                p0, c = pd["wpos"].get(k), counts[k][fi]
                if p0 is None:
                    continue
                wins[k][p0:p0 + c] = [np.zeros(1536)]
                lengths[k][p0:p0 + c] = [1536]
                counts[k][fi] = 1
            groups["lr"][fi] = groups["sr"][fi] = CONCEALED
            self.status[fi] = "audio_size.late"
            self.n_bad += 1
            self.n_ok -= 1
            self.damage.add("audio_size.late")
