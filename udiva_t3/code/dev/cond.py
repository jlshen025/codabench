"""DEVELOPMENT EXPERIMENT (not part of the submitted method).

Input-conditioned anticipation, family 1+2: event-history Markov models and
ongoing-activity conditioning.

Both are ORACLE ceilings: the conditioning context is read from the ground-truth
annotations of the observed past (start < t_b), i.e. as if a perfect recognizer of
the past were available. If the oracle does not beat the constant prior, then no
amount of work on recognizing the past from video/transcript can help either.

Model: a lookup table P(future-window sequence | context) estimated on the training
sessions; at test time the top-`kctx` sequences of the matching context row become
alternatives 2..4, alternative 1 is always the empty sequence, and any remaining
slots are padded with the global prior alternatives (so the conditioned model can
never be worse than the prior by construction of the K=5 hedge).

Contexts (per participant, lookback W seconds unless noted):
  last     last observed event tuple (exact)
  last2    the last two observed event tuples (ordered)
  bag      set of observed event tuples in the window
  hi       last event's high-level action / utterance type (target dropped)
  tgt      last event's target
  count    number of observed events, capped at 4 (activity rate)
  ongoing  the event(s) in progress at t_b (start <= t_b < end)   [no W]
  ongoinghi  high-level action / utterance type of those, target dropped [no W]

Usage: python dev/cond.py            (full table, ~10 min single-core)
"""
from collections import Counter, defaultdict

import _path  # noqa: F401
import udiva_data as U
import sdl
import cv as CV
import predictors as P

PA, PB = "participant_a", "participant_b"


_EV_CACHE = {}   # (sid, t_b, participant, W) -> event list; folds re-query the same cells


def past_events(sid, t_b, participant, W):
    ck = (sid, t_b, participant, W)
    if ck in _EV_CACHE:
        return _EV_CACHE[ck]
    raw = U.load_raw(sid)
    ev = [a for a in raw if a["subject"] == participant and t_b - W < a["start"] < t_b]
    ev.sort(key=lambda a: (a["start"], a["end"]))
    _EV_CACHE[ck] = ev
    return ev


def ongoing_events(sid, t_b, participant):
    """Events in progress at the reference timestamp (start <= t_b < end)."""
    ck = (sid, t_b, participant, "ongoing")
    if ck in _EV_CACHE:
        return _EV_CACHE[ck]
    raw = U.load_raw(sid)
    ev = [a for a in raw if a["subject"] == participant and a["start"] <= t_b < a["end"]]
    ev.sort(key=lambda a: (a["start"], a["end"]))
    _EV_CACHE[ck] = ev
    return ev


def seq_key(seq):
    return tuple(tuple(sdl.ev_key(e)) for e in seq)


def key2seq(sk):
    return [P.k2e(list(k)) for k in sk]


def _coarse(a):
    return ("V", a["utterance_type"]) if a["act"] == "V" else ("NV", a["high_level_action"])


def ctx_last(sid, t_b, participant, W):
    ev = past_events(sid, t_b, participant, W)
    return tuple(sdl.ev_key(U.event_tuple(ev[-1]))) if ev else ("NONE",)


def ctx_bag(sid, t_b, participant, W):
    ev = past_events(sid, t_b, participant, W)
    return frozenset(tuple(sdl.ev_key(U.event_tuple(a))) for a in ev) or frozenset([("NONE",)])


def ctx_lastN(sid, t_b, participant, W, N=2):
    ev = past_events(sid, t_b, participant, W)
    return tuple(tuple(sdl.ev_key(U.event_tuple(a))) for a in ev[-N:]) or (("NONE",),)


def ctx_hi(sid, t_b, participant, W):
    ev = past_events(sid, t_b, participant, W)
    return _coarse(ev[-1]) if ev else ("NONE",)


def ctx_tgt(sid, t_b, participant, W):
    ev = past_events(sid, t_b, participant, W)
    if not ev:
        return ("NONE",)
    tf = ev[-1].get("target_filtered", ["none"])
    return ("T", tf[0] if tf else "none")


def ctx_count(sid, t_b, participant, W):
    n = len(past_events(sid, t_b, participant, W))
    return ("C", min(n, 4))


def ctx_ongoing(sid, t_b, participant, W):
    ev = ongoing_events(sid, t_b, participant)
    return tuple(tuple(sdl.ev_key(U.event_tuple(a))) for a in ev) or (("NONE",),)


def ctx_ongoing_hi(sid, t_b, participant, W):
    ev = ongoing_events(sid, t_b, participant)
    return tuple(sorted(set(_coarse(a) for a in ev))) or (("NONE",),)


CTX = {"last": ctx_last, "bag": ctx_bag,
       "last2": lambda s, t, p, W: ctx_lastN(s, t, p, W, 2),
       "hi": ctx_hi, "tgt": ctx_tgt, "count": ctx_count,
       "ongoing": ctx_ongoing, "ongoinghi": ctx_ongoing_hi}

# contexts that read the state AT t_b rather than a lookback window
NO_W = ("ongoing", "ongoinghi")


class CondPredictor(P.Predictor):
    def __init__(self, table, ctx_fn, W, global_alts, kctx=3):
        self.table = table          # ctx -> Counter(seq_key -> n)
        self.ctx_fn = ctx_fn
        self.W = W
        self.global_alts = global_alts
        self.kctx = kctx

    def predict(self, sid, seg):
        t_b = seg["t_b"]
        out = {}
        for p in (PA, PB):
            c = self.ctx_fn(sid, t_b, p, self.W)
            alts = [[]]                                   # always include empty
            cnt = self.table.get(c)
            if cnt:
                for sk, _ in cnt.most_common(self.kctx):
                    alts.append(key2seq(sk))
            for g in self.global_alts:                    # pad with global priors up to 5
                if len(alts) >= 5:
                    break
                alts.append([list(e) for e in g])
            out[p] = alts[:5]
        return out


def fit_table(train, gt, ctx_fn, W):
    table = defaultdict(Counter)
    for sid in train:
        for seg_id, seg in gt[sid].items():
            t_b = seg["t_b"]
            g = sdl.ref_seg_to_gt(seg)
            for p in (PA, PB):
                c = ctx_fn(sid, t_b, p, W)
                table[c][seq_key(g[p])] += 1
    return table


def make_factory(ctx_name, W, kctx=3):
    ctx_fn = CTX[ctx_name]

    def fac(train, gt):
        # global fallback alts = the B2 prior set, rebuilt per fold
        cV, cNV, _ = P._freqs(train)
        nv = [P.k2e(k) for k, _ in cNV.most_common(2)]
        v = [P.k2e(k) for k, _ in cV.most_common(2)]
        global_alts = [[nv[0]], [nv[1]], [v[0]], [v[1]]]
        table = fit_table(train, gt, ctx_fn, W)
        return CondPredictor(table, ctx_fn, W, global_alts, kctx)
    return fac


def rows(gt, kctx=3):
    """[(label, res), ...] for the conditioning families."""
    out = []
    for name in ("last", "last2", "bag", "hi", "tgt", "count"):
        for W in (2.0, 4.0):
            res = CV.evaluate(make_factory(name, W, kctx), gt)
            out.append(("markov/%s W=%.0fs" % (name, W), res))
    for name in ("ongoing", "ongoinghi"):
        res = CV.evaluate(make_factory(name, 0.0, kctx), gt)
        out.append(("ongoing-activity/%s" % name, res))
    return out


if __name__ == "__main__":
    gt = CV.build_all_gt()
    print("ORACLE conditioning (context read from GT past) — LOSO, n=%d segments"
          % sum(len(v) for v in gt.values()))
    print("reference: constant prior B2 mean4=0.4127")
    for label, res in rows(gt):
        print("%-26s %s" % (label, CV.fmt(res)))
