"""Prior-based predictors (no video needed; fully submittable).

A predictor: .predict(sid, seg) -> {participant: [alt_seq, ...]} (up to K=5 alts).
A factory: factory(train_sids, gt_all) -> predictor, fit on train only.
"""
from collections import Counter
import udiva_data as U
import sdl

PA, PB = "participant_a", "participant_b"
K = 5


class Predictor:
    def predict(self, sid, seg):
        raise NotImplementedError


# --------------------------------------------------------------- empty
class EmptyPredictor(Predictor):
    def predict(self, sid, seg):
        return {PA: [[]], PB: [[]]}


def empty_factory(train, gt):
    return EmptyPredictor()


# --------------------------------------------------------------- frequency model
_SESS_FREQ = {}


def _sess_freq(sid):
    """Per-session (cV, cNV, call) Counters, cached."""
    if sid in _SESS_FREQ:
        return _SESS_FREQ[sid]
    cV, cNV, call = Counter(), Counter(), Counter()
    for a in U.load_raw(sid):
        if a["subject"] not in (PA, PB):
            continue
        e = U.event_tuple(a)
        k = tuple(sdl.ev_key(e))
        call[k] += 1
        (cV if len(e) == 2 else cNV)[k] += 1
    _SESS_FREQ[sid] = (cV, cNV, call)
    return _SESS_FREQ[sid]


def _freqs(train):
    cV, cNV, call = Counter(), Counter(), Counter()
    for sid in train:
        sV, sNV, sall = _sess_freq(sid)
        cV.update(sV)
        cNV.update(sNV)
        call.update(sall)
    return cV, cNV, call


def k2e(k):
    if k[0] == "V":
        o = list(k[2]) if isinstance(k[2], tuple) else k[2]
        return [k[1], o]
    o = list(k[3]) if isinstance(k[3], tuple) else k[3]
    return [k[1], k[2], o]


class ConstAlts(Predictor):
    def __init__(self, alts):
        self.alts = alts

    def predict(self, sid, seg):
        return {PA: [list(a) for a in self.alts], PB: [list(a) for a in self.alts]}


def const_hedge_factory(train, gt):
    cV, cNV, call = _freqs(train)
    topAll = k2e(call.most_common(1)[0][0])
    topNV = [k2e(k) for k, _ in cNV.most_common(3)]
    topV = [k2e(k) for k, _ in cV.most_common(3)]
    # 5 diverse alts: empty, top overall, top NV seq(1&2), top V
    alts = [[], [topAll], topNV[:1], topV[:1], topNV[:2]]
    return ConstAlts(alts)


# ---- LOCKED HELD-BEST priors (constant K=5; fit on `train`) ----
def b2_factory(train, gt):
    """THE SUBMITTED PREDICTOR (entry 825331): [empty, NV1, NV2, V1, V2].

    Balanced across the four subtask columns. LOSO (cv.py default protocol, 2835 segments):
    next=0.3902 verbal=0.5760 nonverbal=0.3806 full=0.3041 -> mean4=0.4127 (fold std 0.039).
    """
    cV, cNV, _ = _freqs(train)
    nv = [k2e(k) for k, _ in cNV.most_common(2)]
    v = [k2e(k) for k, _ in cV.most_common(2)]
    return ConstAlts([[], [nv[0]], [nv[1]], [v[0]], [v[1]]])


def b4_factory(train, gt):
    """A/B VARIANT (entry 837289), non-verbal-heavy: [empty, NV1..NV4].

    LOSO: next=0.4114 verbal=0.4975 nonverbal=0.4261 full=0.3185 -> mean4=0.4134 — inside the
    fold noise of b2_factory, but it gives up the verbal column, which the official
    per-column rank average rewards; B2 stayed the final entry.
    """
    cV, cNV, _ = _freqs(train)
    nv = [k2e(k) for k, _ in cNV.most_common(4)]
    return ConstAlts([[], [nv[0]], [nv[1]], [nv[2]], [nv[3]]])


def alts_for_all(factory):
    """The constant alt set this factory produces when fit on ALL 21 sessions (the real submission)."""
    import udiva_data as U
    pred = factory(U.list_sessions(), None)
    return pred.alts
