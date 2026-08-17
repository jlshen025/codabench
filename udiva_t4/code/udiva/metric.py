"""Structured Damerau-Levenshtein (SDL) metric for UDIVA-HHOI Track 4.

VERIFIED from the organizer competition page (16645):
- Reported score = normalized SDL SIMILARITY in [0,1], HIGHER IS BETTER, 1=perfect.
- Per (participant,segment): Score = 1 - D_SDL/max(m,n); if m=n=0 -> 1.
- D_SDL = Damerau-Levenshtein: c_ins=c_del=1; c_tr=0.5 (adjacent transpose ONLY if
  both tuples EXACTLY equal); c_sub = {diff type:1; verbal(u,o):0.8*[u!=]+0.2*[o!=];
  nonverbal(h,l,o):0.4*[h!=]+0.4*[l!=]+0.2*[o!=]}.
- K=5 alternatives: best-of-K kept PER SUBTASK. Segment=mean(A,B). Subtask=mean over segments.
- 4 subtasks: next-action (first event only), verbal-only, nonverbal-only, full mixed.
- Official ranking = mean rank across the 4 subtask columns.

An event is a list: len 2 = verbal [utterance_type, target]; len 3 = nonverbal
[high_level_action, low_level_action, target]. target may be a str or list.
A prediction for one (participant,segment) is a list of up to K alternative sequences,
where each alternative is a list of events. (A single sequence is wrapped to K=1.)
"""
from functools import lru_cache

SUBTASKS = ("next_action", "verbal", "nonverbal", "full")


def _key(o):
    """Hashable form of a target field (str or list)."""
    return tuple(o) if isinstance(o, list) else o


def sub_cost(a, b):
    """Structured substitution cost between GT event a and pred event b."""
    la, lb = len(a), len(b)
    if la != lb:
        return 1.0
    if la == 2:  # verbal (u, o)
        return 0.8 * (a[0] != b[0]) + 0.2 * (_key(a[1]) != _key(b[1]))
    # nonverbal (h, l, o)
    return 0.4 * (a[0] != b[0]) + 0.4 * (a[1] != b[1]) + 0.2 * (_key(a[2]) != _key(b[2]))


def _eq(a, b):
    """Exact structured tuple equality (for transposition)."""
    if len(a) != len(b):
        return False
    return all(_key(x) == _key(y) for x, y in zip(a, b))


def sdl_distance(gt, pred):
    """D_SDL(gt, pred) via the organizer DP recursion."""
    m, n = len(gt), len(pred)
    if m == 0:
        return float(n)
    if n == 0:
        return float(m)
    prev2 = None
    prev = list(range(n + 1))          # D[i-1][*], start i=0 row = [0,1,..,n]
    for i in range(1, m + 1):
        cur = [float(i)] + [0.0] * n     # D[i][0] = i
        gi = gt[i - 1]
        for j in range(1, n + 1):
            pj = pred[j - 1]
            best = prev[j] + 1.0                       # deletion
            ins = cur[j - 1] + 1.0                     # insertion
            if ins < best:
                best = ins
            subv = prev[j - 1] + sub_cost(gi, pj)      # substitution
            if subv < best:
                best = subv
            if i > 1 and j > 1 and _eq(gi, pred[j - 2]) and _eq(gt[i - 2], pj):
                tr = prev2[j - 2] + 0.5                # transposition
                if tr < best:
                    best = tr
            cur[j] = best
        prev2 = prev
        prev = cur
    return prev[n]


def score_seq(gt, pred):
    """Normalized SDL similarity for one GT vs one predicted sequence."""
    m, n = len(gt), len(pred)
    if m == 0 and n == 0:
        return 1.0
    return 1.0 - sdl_distance(gt, pred) / max(m, n)


def _extract(seq, subtask):
    if subtask == "full":
        return seq
    if subtask == "next_action":
        return seq[:1]
    if subtask == "verbal":
        return [e for e in seq if len(e) == 2]
    if subtask == "nonverbal":
        return [e for e in seq if len(e) == 3]
    raise ValueError(subtask)


def _as_alts(pred):
    """Normalize a per-participant prediction to a list of <=5 alternative sequences.
    Accept either a single sequence (list of events) or a list of sequences."""
    if not pred:
        return [[]]
    # a list of events => each event is a list; a list of sequences => each item is a list of events (list of lists)
    first = pred[0]
    is_alts = isinstance(first, list) and (len(first) == 0 or isinstance(first[0], list))
    alts = pred if is_alts else [pred]
    return alts[:5] if alts else [[]]


def best_of_k(gt, pred, subtask):
    """Best-of-K subtask score for one participant."""
    gt_e = _extract(gt, subtask)
    alts = _as_alts(pred)
    return max(score_seq(gt_e, _extract(alt, subtask)) for alt in alts)


def score_dataset(ref, pred):
    """Compute the 4 subtask scores over a parsed dataset.

    ref:  {session: {seg: {'participants': {participant_a:{'events':[...]}, ...}}}}
    pred: {session: {seg: {'participants': {participant_a:{'events': <seq or list-of-seq>}, ...}}}}
          Missing sessions/segments/participants default to empty prediction.
    Returns dict subtask->mean score, plus 'mean' of the 4, and n_cells.
    """
    sums = {s: 0.0 for s in SUBTASKS}
    nseg = 0
    for sess, segs in ref.items():
        psess = pred.get(sess, {})
        for seg, sd in segs.items():
            pseg = psess.get(seg, {})
            for st in SUBTASKS:
                a = _participant_score(sd, pseg, "participant_a", st)
                b = _participant_score(sd, pseg, "participant_b", st)
                sums[st] += 0.5 * (a + b)
            nseg += 1
    out = {s: (sums[s] / nseg if nseg else 0.0) for s in SUBTASKS}
    out["mean"] = sum(out[s] for s in SUBTASKS) / len(SUBTASKS)
    out["n_segments"] = nseg
    return out


def _participant_score(seg_dict, pseg, participant, subtask):
    gt = seg_dict["participants"].get(participant, {}).get("events", [])
    pp = pseg.get("participants", {}).get(participant, {}).get("events", []) if pseg else []
    return best_of_k(gt, pp, subtask)
