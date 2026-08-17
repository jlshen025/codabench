"""DEVELOPMENT EXPERIMENT (not part of the submitted method).

Input-conditioned anticipation, family 5: priors conditioned on something other than
the pooled training distribution — participant role, position in the session
(time bin), and the session itself.

Each variant keeps the B2 SHAPE (empty + top-2 non-verbal + top-2 verbal singletons)
and only changes which subset of training events the frequencies are counted over:

  global        pooled over all training sessions and both participants  == the submitted B2
  role          counted separately for participant_a / participant_b
  timebin<T>    counted separately per absolute time bin floor(t_b/T) (capped at 5 bins)
  session_oracle counted on the HELD-OUT session's own annotations.
                 This is an ORACLE and an upper bound for the whole per-session-prior
                 family (metadata-conditioned, difficulty-conditioned, adaptive, ...):
                 no legitimate session-level conditioning can beat knowing the held-out
                 session's own event marginals.

Usage: python dev/priors.py
"""
from collections import Counter, defaultdict

import _path  # noqa: F401
import udiva_data as U
import sdl
import cv as CV
import predictors as P

PA, PB = "participant_a", "participant_b"


def _b2_from(cV, cNV, fallback):
    """B2-shaped alt set from verbal/non-verbal counters, backfilled from `fallback`."""
    nv = [P.k2e(k) for k, _ in cNV.most_common(2)]
    v = [P.k2e(k) for k, _ in cV.most_common(2)]
    fnv, fv = fallback
    while len(nv) < 2:
        nv.append(fnv[len(nv)])
    while len(v) < 2:
        v.append(fv[len(v)])
    return [[], [nv[0]], [nv[1]], [v[0]], [v[1]]]


def _global_top(train):
    cV, cNV, _ = P._freqs(train)
    return ([P.k2e(k) for k, _ in cNV.most_common(2)], [P.k2e(k) for k, _ in cV.most_common(2)])


def _counts_by(train, keyfn):
    """{group_key: (Counter verbal, Counter nonverbal)} over raw annotated events."""
    tab = defaultdict(lambda: (Counter(), Counter()))
    for sid in train:
        for a in U.load_raw(sid):
            if a["subject"] not in (PA, PB):
                continue
            g = keyfn(sid, a)
            if g is None:
                continue
            e = U.event_tuple(a)
            k = tuple(sdl.ev_key(e))
            cV, cNV = tab[g]
            (cV if len(e) == 2 else cNV)[k] += 1
    return tab


class GroupedPrior(P.Predictor):
    """Per-group constant alt sets; `groupfn(sid, seg, participant) -> group key`."""

    def __init__(self, alts_by_group, default_alts, groupfn):
        self.alts_by_group = alts_by_group
        self.default_alts = default_alts
        self.groupfn = groupfn

    def predict(self, sid, seg):
        out = {}
        for p in (PA, PB):
            g = self.groupfn(sid, seg, p)
            alts = self.alts_by_group.get(g, self.default_alts)
            out[p] = [[list(e) for e in a] for a in alts]
        return out


def role_factory(train, gt):
    fb = _global_top(train)
    tab = _counts_by(train, lambda sid, a: a["subject"])
    alts = {g: _b2_from(cV, cNV, fb) for g, (cV, cNV) in tab.items()}
    return GroupedPrior(alts, _b2_from(*_global_top_counters(train), fb),
                        lambda sid, seg, p: p)


def _global_top_counters(train):
    cV, cNV, _ = P._freqs(train)
    return cV, cNV


def timebin_factory(T=60.0, nbin=5):
    def fac(train, gt):
        fb = _global_top(train)
        keyfn = lambda sid, a: min(int(a["start"] // T), nbin - 1)  # noqa: E731
        tab = _counts_by(train, keyfn)
        alts = {g: _b2_from(cV, cNV, fb) for g, (cV, cNV) in tab.items()}
        return GroupedPrior(alts, _b2_from(*_global_top_counters(train), fb),
                            lambda sid, seg, p: min(int(seg["t_b"] // T), nbin - 1))
    return fac


def session_oracle_factory(train, gt):
    """ORACLE: the alt set is counted on the session being predicted (held-out included)."""
    fb = _global_top(train)
    tab = _counts_by(U.list_sessions(), lambda sid, a: sid)
    alts = {g: _b2_from(cV, cNV, fb) for g, (cV, cNV) in tab.items()}
    return GroupedPrior(alts, _b2_from(*_global_top_counters(train), fb),
                        lambda sid, seg, p: sid)


def rows(gt):
    out = [("prior/global (= submitted B2)", CV.evaluate(P.b2_factory, gt)),
           ("prior/per-role (A vs B)", CV.evaluate(role_factory, gt))]
    for T in (30.0, 60.0):
        out.append(("prior/time-bin %.0fs" % T, CV.evaluate(timebin_factory(T), gt)))
    out.append(("prior/per-session ORACLE", CV.evaluate(session_oracle_factory, gt)))
    return out


def role_marginal_report():
    """Why per-role conditioning cannot help: A and B have near-identical marginals."""
    tab = _counts_by(U.list_sessions(), lambda sid, a: a["subject"])
    for role in (PA, PB):
        cV, cNV = tab[role]
        print("  %-14s top NV: %s" % (role, [(P.k2e(k), n) for k, n in cNV.most_common(2)]))
        print("  %-14s top  V: %s" % ("", [(P.k2e(k), n) for k, n in cV.most_common(2)]))


if __name__ == "__main__":
    gt = CV.build_all_gt()
    print("role / time-bin / session priors — LOSO, n=%d segments"
          % sum(len(v) for v in gt.values()))
    for label, res in rows(gt):
        print("%-30s %s" % (label, CV.fmt(res)))
    print("\nper-role marginals (all 21 sessions):")
    role_marginal_report()
