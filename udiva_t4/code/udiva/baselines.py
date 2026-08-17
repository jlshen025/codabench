"""Static (input-free) prior baselines for UDIVA-HHOI Track 4.

All exploit the K=5 best-of-K rule: a prediction is a list of up to 5 alternative
sequences; the scorer keeps the best per subtask, so diverse alternatives never hurt.
GreedyK5 selects the 5 alternatives that maximize mean-over-4-subtasks best-of-K on
the training cells (a facility-location style greedy).
"""
from collections import Counter
from . import metric as M


def iter_cells(ds):
    for s, segs in ds.items():
        for sid, seg in segs.items():
            for p in ("participant_a", "participant_b"):
                yield seg["participants"][p]["events"]


def _tkey(t):
    return tuple(t) if isinstance(t, list) else (t,)


def ev_key(e):
    return (e[0], _tkey(e[1])) if len(e) == 2 else (e[0], e[1], _tkey(e[2]))


def seq_key(seq):
    return tuple(ev_key(e) for e in seq)


class EmptyPredictor:
    def fit(self, ds):
        pass

    def predict(self, s, seg, p):
        return []


class StaticSeqPredictor:
    """Predict one fixed K=1 sequence for all cells (default: most-frequent full GT seq)."""
    def __init__(self, seq=None):
        self.seq = seq

    def fit(self, ds):
        if self.seq is not None:
            return
        c = Counter()
        rep = {}
        for evs in iter_cells(ds):
            k = seq_key(evs)
            c[k] += 1
            rep[k] = evs
        self.seq = rep[c.most_common(1)[0][0]]

    def predict(self, s, seg, p):
        return self.seq


def candidate_pool(train_ds, tags=None):
    """Build a diverse candidate pool of sequences from training cells.

    `tags` is an optional out-list: one construction-family label is appended per
    candidate actually added, in pool order (reporting only; never read by the model).
    """
    full = Counter(); full_rep = {}
    evc = Counter(); ev_rep = {}
    bylen = {}
    for evs in iter_cells(train_ds):
        k = seq_key(evs); full[k] += 1; full_rep[k] = evs
        L = len(evs); bylen.setdefault(L, Counter()); bylen[L][k] += 1
        for e in evs:
            ek = ev_key(e); evc[ek] += 1; ev_rep[ek] = e
    pool = [[]]                                   # empty
    seen = {seq_key([])}
    if tags is not None and not tags:
        tags.append("empty")
    def add(seq, tag):
        k = seq_key(seq)
        if k not in seen:
            seen.add(k); pool.append(seq)
            if tags is not None:
                tags.append(tag)
    # top full sequences
    for k, _ in full.most_common(25):
        add(full_rep[k], "observed_top25")
    # modal sequence per length
    for L in sorted(bylen):
        if L == 0: continue
        k = bylen[L].most_common(1)[0][0]; add(full_rep[k], "observed_modal_per_length")
    # stacks of the top-N most frequent single events
    top_ev = [ev_rep[k] for k, _ in evc.most_common(8)]
    for n in range(1, 6):
        add([e for e in top_ev[:n]], "synthetic_stack")
    # top single verbal / nonverbal events
    for ek, _ in evc.most_common(40):
        e = ev_rep[ek]
        add([e], "observed_singleton")
    return pool


def rich_pool(train_ds, n_v=8, n_nv=8, mix=4, tags=None):
    """Hedge-packing candidate pool: top verbal types & nonverbal modes (modal target),
    plus mixed/repeated sequences. Exploits best-of-K's per-subtask independence.

    Construction, in pool order (each stage skips duplicates of an earlier one):
      empty · verbal singletons (n_v most frequent utterance types, each paired with the
      globally modal verbal target) · non-verbal singletons (n_nv most frequent
      (high,low) action modes, each paired with the globally modal non-verbal target) ·
      mixed verbal+non-verbal pairs, both orders, over the top `mix` of each ·
      ordered non-verbal pairs · repeated non-verbal events · then the observed-sequence
      pool of `candidate_pool` (real training sequences of any length + synthetic stacks
      of the 1..5 most frequent single events).

    `tags` is an optional out-list of per-candidate construction-family labels in pool
    order (reporting only; never read by the model).
    """
    vt = Counter(); nv = Counter(); vtgt = Counter(); ntgt = Counter()
    for evs in iter_cells(train_ds):
        for e in evs:
            if len(e) == 2:
                vt[e[0]] += 1; vtgt[_tkey(e[1])] += 1
            else:
                nv[(e[0], e[1])] += 1; ntgt[_tkey(e[2])] += 1

    def unkey(k):
        return list(k) if len(k) != 1 else k[0]
    mvt = unkey(vtgt.most_common(1)[0][0]) if vtgt else "none"
    mnt = unkey(ntgt.most_common(1)[0][0]) if ntgt else "none"
    V = [[t, mvt] for t, _ in vt.most_common(n_v)]
    NV = [[h, l, mnt] for (h, l), _ in nv.most_common(n_nv)]
    pool = [[]]; seen = {seq_key([])}
    if tags is not None and not tags:
        tags.append("empty")

    def add(s, tag):
        k = seq_key(s)
        if k not in seen:
            seen.add(k); pool.append(s)
            if tags is not None:
                tags.append(tag)
    for v in V:
        add([v], "verbal_singleton")
    for x in NV:
        add([x], "nonverbal_singleton")
    for v in V[:mix]:
        for x in NV[:mix]:
            add([v, x], "mixed_pair"); add([x, v], "mixed_pair")
    for j in range(min(mix, len(NV))):
        for k in range(min(mix, len(NV))):
            if j != k:
                add([NV[j], NV[k]], "nonverbal_pair")
    for j in range(min(3, len(NV))):
        add([NV[j], NV[j]], "nonverbal_repeat")
    # also blend in the observed-sequence pool (covers real multi-event patterns)
    obs_tags = []
    for s, t in zip(candidate_pool(train_ds, tags=obs_tags), obs_tags):
        add(s, t)
    return pool


def _cell_cand_scores(cells, pool):
    """scores[i][c][subtask] for cell i, candidate c."""
    out = []
    for evs in cells:
        row = []
        for cand in pool:
            row.append({st: M.best_of_k(evs, cand, st) for st in M.SUBTASKS})
        out.append(row)
    return out


class GreedyK5Predictor:
    """Select <=K alternatives maximizing the summed best-of-K score on train cells.

    Objective (all four subtasks EQUALLY weighted, i.e. plain unweighted sum):

        F(S) = sum_{i in C} sum_{t in T} max_{a in S} score_t(gt_i, a)

    C = the training cells, one per (segment, participant) — so the sum runs over
    participants, evaluation cells and subtasks; T = the four subtask reductions
    (next_action, verbal, nonverbal, full); score_t = the normalized SDL similarity of
    metric.score_seq applied to the subtask-t projection of both sequences. `objective`
    may repeat a subtask to up-weight it (not used by the submitted model).

    Greedy: start from S = {} and K times add the candidate with the largest F(S u {c}),
    maintaining the running per-cell per-subtask best `best[i][t]`. Because the test is
    strict (`tot > best_gain`) and candidates are scanned in pool order, TIES ARE BROKEN
    BY LOWEST POOL INDEX, i.e. in favour of the earlier-constructed candidate (empty
    sequence first, then the more frequent verbal/non-verbal modes).

    C is subsampled to at most `max_cells` by a deterministic uniform stride over the
    cell enumeration order (session, segment, participant_a then participant_b); no RNG
    is involved anywhere. `refine=True` additionally runs first-improvement hill-climbing
    over single-slot swaps until no swap improves F (used to certify local optimality).
    """
    def __init__(self, K=5, objective=M.SUBTASKS, max_cells=4000, pool_fn=None, refine=False):
        self.K = K; self.objective = objective; self.max_cells = max_cells
        self.pool_fn = pool_fn or candidate_pool; self.alts = [[]]; self.refine = refine

    def fit(self, ds):
        cells = list(iter_cells(ds))
        if len(cells) > self.max_cells:
            step = len(cells) / self.max_cells
            cells = [cells[int(i * step)] for i in range(self.max_cells)]
        pool = self.pool_fn(ds)
        S = _cell_cand_scores(cells, pool)
        nC = len(pool); ncell = len(cells)
        chosen = []
        # running best score per cell per subtask
        best = [{st: 0.0 for st in M.SUBTASKS} for _ in range(ncell)]
        for _ in range(self.K):
            best_gain = -1; best_c = None; best_newbest = None
            for c in range(nC):
                if c in chosen: continue
                tot = 0.0
                newbest = []
                for i in range(ncell):
                    nb = {}
                    for st in self.objective:
                        v = best[i][st]
                        sc = S[i][c][st]
                        nb[st] = sc if sc > v else v
                    newbest.append(nb)
                    tot += sum(nb[st] for st in self.objective)
                if tot > best_gain:
                    best_gain = tot; best_c = c; best_newbest = newbest
            chosen.append(best_c)
            best = best_newbest
        if self.refine and len(chosen) == self.K:
            import numpy as np
            sub_idx = [list(M.SUBTASKS).index(st) for st in self.objective]
            SA = np.empty((ncell, nC, len(M.SUBTASKS)), dtype=np.float32)
            for i in range(ncell):
                for c in range(nC):
                    d = S[i][c]
                    for j, st in enumerate(M.SUBTASKS):
                        SA[i, c, j] = d[st]
            SAo = SA[:, :, sub_idx]

            def obj(sel):
                return float(SAo[:, sel, :].max(axis=1).sum())
            cur = obj(chosen); improved = True
            while improved:
                improved = False
                for pos in range(len(chosen)):
                    for c in range(nC):
                        if c in chosen:
                            continue
                        trial = list(chosen); trial[pos] = c
                        o = obj(trial)
                        if o > cur + 1e-9:
                            chosen = trial; cur = o; improved = True; break
                    if improved:
                        break
        self.alts = [pool[c] for c in chosen]

    def predict(self, s, seg, p):
        return self.alts
