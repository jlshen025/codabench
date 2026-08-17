"""UDIVA-HHOI Track 3 — official Structured Damerau-Levenshtein (SDL) scorer.

Recovered verbatim from the competition Description page (comp 16646):

Event tuple:  verbal e^nu=(u,o)  [len 2];  non-verbal e^eta=(h,l,o) [len 3].
Type is inferred from tuple LENGTH.

Substitution cost c_sub(e_i, ê_j):
  * different types                 -> 1.0
  * verbal vs verbal     -> 0.8*1[u≠û] + 0.2*1[o≠ô]
  * nonverbal vs nonverbal-> 0.4*1[h≠ĥ] + 0.4*1[l≠l̂] + 0.2*1[o≠ô]
Insertion = deletion = 1.0 ; transposition = 0.5 (OSA form, needs e_i=ê_{j-1} & e_{i-1}=ê_j).

Recurrence (D(0,0)=0, D(i,0)=i, D(0,j)=j):
  D(i,j)=min( D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1)+c_sub,
              D(i-2,j-2)+0.5 if i>1,j>1, e_i=ê_{j-1}, e_{i-1}=ê_j )
D_SDL = D(m,n).  Score_SDL = 1 - D_SDL/max(m,n)  in [0,1]; if m=n=0 -> 1.  HIGHER IS BETTER.

Per participant: best of K alternatives.  Per segment: mean over A,B.  Final: mean over segments.
FOUR subtasks (leaderboard columns; official rank = mean of per-column ranks):
  next      : first event only of each sequence (length<=1)
  verbal    : keep verbal (len2) events only
  nonverbal : keep nonverbal (len3) events only
  full      : the full typed mixed sequence
"""
SUBTASKS = ["next", "verbal", "nonverbal", "full"]
PA, PB = "participant_a", "participant_b"


def _nt(o):
    return tuple(o) if isinstance(o, list) else o


def ev_type(e):
    return "V" if len(e) == 2 else "NV"


def ev_key(e):
    return ("V", e[0], _nt(e[1])) if len(e) == 2 else ("NV", e[0], e[1], _nt(e[2]))


def ev_eq(a, b):
    return ev_key(a) == ev_key(b)


def c_sub(e, f):
    te, tf = ev_type(e), ev_type(f)
    if te != tf:
        return 1.0
    if te == "V":
        return 0.8 * (e[0] != f[0]) + 0.2 * (_nt(e[1]) != _nt(f[1]))
    return 0.4 * (e[0] != f[0]) + 0.4 * (e[1] != f[1]) + 0.2 * (_nt(e[2]) != _nt(f[2]))


def sdl_distance(Y, P):
    m, n = len(Y), len(P)
    if m == 0:
        return float(n)
    if n == 0:
        return float(m)
    D = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        D[i][0] = i
    for j in range(n + 1):
        D[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best = D[i - 1][j] + 1.0
            v = D[i][j - 1] + 1.0
            if v < best:
                best = v
            v = D[i - 1][j - 1] + c_sub(Y[i - 1], P[j - 1])
            if v < best:
                best = v
            if i > 1 and j > 1 and ev_eq(Y[i - 1], P[j - 2]) and ev_eq(Y[i - 2], P[j - 1]):
                v = D[i - 2][j - 2] + 0.5
                if v < best:
                    best = v
            D[i][j] = best
    return D[m][n]


def sdl_score(Y, P):
    m, n = len(Y), len(P)
    if m == 0 and n == 0:
        return 1.0
    return 1.0 - sdl_distance(Y, P) / max(m, n)


def _filt(seq, sub):
    if sub == "full":
        return list(seq)
    if sub == "verbal":
        return [e for e in seq if len(e) == 2]
    if sub == "nonverbal":
        return [e for e in seq if len(e) == 3]
    if sub == "next":
        return list(seq[:1])
    raise ValueError(sub)


def best_of_k(Y, alts, sub):
    Yf = _filt(Y, sub)
    if not alts:
        alts = [[]]
    return max(sdl_score(Yf, _filt(P, sub)) for P in alts)


def score_segment(gt, pred, subtasks=SUBTASKS):
    """gt={participant:[event,...]}, pred={participant:[alt_seq,...]}."""
    out = {}
    for sub in subtasks:
        s = 0.0
        for rho in (PA, PB):
            Y = gt.get(rho, [])
            alts = pred.get(rho) or [[]]
            s += best_of_k(Y, alts, sub)
        out[sub] = s / 2.0
    return out


def ref_seg_to_gt(seg):
    """A reference-style segment {participants:{p:{events:[...]}}} -> {p:[events]}."""
    pp = seg["participants"]
    return {p: pp.get(p, {}).get("events", []) for p in (PA, PB)}


def score_dataset(gt_segments, pred_segments, subtasks=SUBTASKS):
    """gt_segments / pred_segments: {seg_key: ...}.

    gt_segments values are reference-style segments (have 'participants').
    pred_segments values are {participant:[alt_seq,...]} OR reference-style w/ single seq.
    Returns {sub: mean_score, 'mean4': ...}.  seg_key may be (sid,seg_id) or seg_id.
    """
    agg = {s: [] for s in subtasks}
    for key, gseg in gt_segments.items():
        gt = ref_seg_to_gt(gseg) if "participants" in gseg else gseg
        praw = pred_segments.get(key, {})
        if "participants" in praw:  # reference-style single-seq prediction -> wrap as 1 alt
            pred = {p: [praw["participants"].get(p, {}).get("events", [])] for p in (PA, PB)}
        else:
            pred = praw
        sg = score_segment(gt, pred, subtasks)
        for s in subtasks:
            agg[s].append(sg[s])
    res = {s: (sum(v) / len(v) if v else 0.0) for s, v in agg.items()}
    res["mean4"] = sum(res[s] for s in subtasks) / len(subtasks)
    res["n_seg"] = len(next(iter(agg.values()))) if agg else 0
    return res


# ---------------------------------------------------------------- self test
if __name__ == "__main__":
    a = ["agree", "brick_3001"]            # verbal
    b = ["prepare", "pick_up", "leaflet"]  # nonverbal
    c = ["open", "flip", "leaflet"]        # nonverbal
    # perfect
    assert sdl_score([a, b], [a, b]) == 1.0
    # empty vs empty
    assert sdl_score([], []) == 1.0
    # empty pred vs 2 gt -> D=2, max=2 -> 0
    assert abs(sdl_score([a, b], [])) < 1e-9
    # transposition a,b -> b,a : distance 0.5, score 1-0.5/2=0.75
    assert abs(sdl_score([a, b], [b, a]) - 0.75) < 1e-9, sdl_score([a, b], [b, a])
    # verbal sub cost: same utt diff target -> 0.2
    assert abs(c_sub(["agree", "x"], ["agree", "y"]) - 0.2) < 1e-9
    assert abs(c_sub(["agree", "x"], ["doubt", "x"]) - 0.8) < 1e-9
    # nonverbal: diff hi only 0.4
    assert abs(c_sub(["open", "flip", "leaflet"], ["close", "flip", "leaflet"]) - 0.4) < 1e-9
    # cross type 1.0
    assert c_sub(a, b) == 1.0
    # substitution single: [a] vs [a2] where a2 diff target -> D=0.2, max=1 -> 0.8
    assert abs(sdl_score([a], [["agree", "z"]]) - 0.8) < 1e-9
    # best of K: alts include the truth
    assert best_of_k([a, b], [[], [a, b]], "full") == 1.0
    print("sdl.py self-tests PASS")
