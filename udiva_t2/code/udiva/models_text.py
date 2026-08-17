"""Verbal channel: transcript cues -> (subject, utterance type, target, modifier) events.

The right-hand lane of Fig. 1 of the fact sheet. The classified unit is a transcript CUE, not a
2 s segment: cue-level events are mapped onto every segment the cue overlaps at the end.

  _part_cues / _align_cue_events   participant cues, and the ground-truth verbal events that
                                   overlap each cue in time (training only).
  VerbalCueModel._text             the input text of a cue: speaker tag + previous cue + cue.
  VerbalCueModel.fit               TF-IDF (word 1-2 grams + char_wb 2-5 grams) and TWO separate
                                   banks of one-vs-rest logistic heads:
                                     * 34 heads, one per utterance type u  -> P(u | t)
                                     * 108 heads, one per target string tau -> P(tau | t)
                                          ("the target text head" of the fact sheet)
                                   plus the priors P(tau | u) and P(modifier | u, tau).
  VerbalCueModel.predict           tuple composition and cue -> segment assignment.

The subject is NOT predicted: it is read from the speaker tag of the .srt.
"""
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from . import data as Data


def _part_cues(sid, trans_dir=None):
    """The transcript cues spoken by a participant (the supervisor is ignored)."""
    cues = Data.load_transcript(sid, trans_dir) if trans_dir else Data.load_transcript(sid)
    return [c for c in cues if c["speaker"] in ("participant_a", "participant_b")]


def _align_cue_events(sid):
    """[(cue, [ground-truth verbal events of the same subject overlapping that cue])]."""
    from .parse import load_raw
    raw = [e for e in load_raw(Data.ANN_DIR, sid) if e["act"] == "V"]
    out = []
    for c in _part_cues(sid):
        al = [e for e in raw
              if e["subject"] == c["speaker"] and e["start"] < c["end"] and e["end"] > c["start"]]
        out.append((c, al))
    return out


class VerbalCueModel:
    """Cue-level verbal model with separate text heads for the utterance type and the target.

    The defaults below are the configuration of the submitted entry: the top `ku` utterance
    types and top `kt` targets above their probability thresholds are paired, pairs that never
    co-occur in training are dropped, and the confidence is

        confidence = P(u | t) * P(tau | t) * (0.5 + P(tau | u))

    with the modifier taken as arg max P(m | u, tau).
    """

    def __init__(self, C=4.0, ku=8, kt=12, pu_thresh=0.02, pt_thresh=0.02, ctx=True,
                 word_ngram=(1, 2), char_ngram=(2, 5), max_features=60000, min_u=3, min_t=3):
        self.C = C; self.ku = ku; self.kt = kt
        self.pu_thresh = pu_thresh; self.pt_thresh = pt_thresh; self.ctx = ctx
        self.word_ngram = word_ngram; self.char_ngram = char_ngram
        self.max_features = max_features
        self.min_u = min_u; self.min_t = min_t

    def _text(self, cue, prev):
        """Input text of one cue: speaker tag, the previous cue (context), then the cue."""
        t = cue["text"]
        if self.ctx and prev is not None:
            t = prev["text"] + " </s> " + t
        return f"[{cue['speaker'][-1].upper()}] {t}"

    def _rows(self, sids, train, trans_dir=None):
        """[(text, aligned events or None, subject, cue, sid)] in cue order."""
        rows = []
        for sid in sids:
            if train:
                items = _align_cue_events(sid)
            else:      # at prediction time only the transcript is available
                items = [(c, None) for c in _part_cues(sid, trans_dir)]
            for i, (c, al) in enumerate(items):
                prev = items[i - 1][0] if i > 0 else None
                rows.append((self._text(c, prev), al, c["speaker"], c, sid))
        return rows

    def _fit_heads(self, X, rows, attr, min_c):
        """One one-vs-rest logistic head per value of `attr` seen >= min_c times in training."""
        if attr == "target":
            getval = lambda e: ",".join(e["target_filtered"])
        else:
            getval = lambda e: e[attr]
        cnt = Counter(getval(e) for _, al, _, _, _ in rows for e in al)
        labels = [v for v, n in cnt.items() if n >= min_c]
        idx = {v: i for i, v in enumerate(labels)}
        Y = np.zeros((len(rows), len(labels)), np.int8)
        for r, (_, al, _, _, _) in enumerate(rows):
            for e in al:
                j = idx.get(getval(e))
                if j is not None:
                    Y[r, j] = 1
        clf = []
        for j in range(len(labels)):
            if Y[:, j].sum() == 0:
                clf.append(None)
                continue
            m = LogisticRegression(C=self.C, max_iter=400)
            m.fit(X, Y[:, j])
            clf.append(m)
        return labels, clf

    def fit(self, train_sids):
        rows = self._rows(train_sids, True)
        texts = [r[0] for r in rows]
        self.wv = TfidfVectorizer(analyzer="word", ngram_range=self.word_ngram, min_df=2,
                                  max_features=self.max_features, sublinear_tf=True)
        self.cv = TfidfVectorizer(analyzer="char_wb", ngram_range=self.char_ngram, min_df=2,
                                  max_features=self.max_features, sublinear_tf=True)
        X = hstack([self.wv.fit_transform(texts), self.cv.fit_transform(texts)]).tocsr()
        self.us, self.uclf = self._fit_heads(X, rows, "utterance_type", self.min_u)
        self.ts, self.tclf = self._fit_heads(X, rows, "target", self.min_t)
        # priors: co-occurrence P(target | utterance type) and modifier P(m | u, target)
        self.cooc = defaultdict(Counter)
        self.pm = defaultdict(Counter)
        for _, al, _, _, _ in rows:
            for e in al:
                u = e["utterance_type"]
                tg = ",".join(e["target_filtered"])
                self.cooc[u][tg] += 1
                self.pm[(u, tg)][e["modifier"]] += 1
        return self

    def predict(self, test_sids, grid_by_sid=None, trans_dir=None):
        pred = {"verbal": {}}
        for sid in test_sids:
            grid = grid_by_sid[sid] if grid_by_sid else Data.gt_session(sid)["verbal"]
            seg_ev = {sk: {} for sk in grid}
            rows = self._rows([sid], False, trans_dir)
            texts = [r[0] for r in rows]
            X = hstack([self.wv.transform(texts), self.cv.transform(texts)]).tocsr()
            PU = np.zeros((len(rows), len(self.us)), np.float32)
            PT = np.zeros((len(rows), len(self.ts)), np.float32)
            for j, m in enumerate(self.uclf):
                if m is not None:
                    PU[:, j] = m.predict_proba(X)[:, 1]
            for j, m in enumerate(self.tclf):
                if m is not None:
                    PT[:, j] = m.predict_proba(X)[:, 1]
            for r, (_, _, subj, cue, _) in enumerate(rows):
                # every 2 s segment this cue overlaps receives the cue's events
                segs = [sk for sk, blk in grid.items()
                        if cue["start"] < blk["t_e"] and cue["end"] > blk["t_b"]]
                u_ord = [(self.us[j], PU[r, j]) for j in np.argsort(-PU[r])[:self.ku]
                         if PU[r, j] >= self.pu_thresh]
                t_ord = [(self.ts[j], PT[r, j]) for j in np.argsort(-PT[r])[:self.kt]
                         if PT[r, j] >= self.pt_thresh]
                for u, pu in u_ord:
                    utot = sum(self.cooc[u].values()) or 1
                    for tg, pt in t_ord:
                        cooc = self.cooc[u].get(tg, 0) / utot
                        if cooc == 0:
                            continue      # gate: (u, target) pairs never seen in training
                        md = (self.pm[(u, tg)].most_common(1) or [("none", 1)])[0][0]
                        sc = float(pu * pt * (0.5 + cooc))
                        tup = (subj, u, tg, md)     # subject = the .srt speaker tag
                        for sk in segs:
                            if seg_ev[sk].get(tup, 0) < sc:
                                seg_ev[sk][tup] = sc
            pred["verbal"][sid] = {sk: {"events": [
                {"subject": t[0], "utterance_type": t[1], "target": t[2], "modifier": t[3],
                 "score": s} for t, s in d.items()]} for sk, d in seg_ev.items()}
        return pred
