"""Feature-conditioned predictors for UDIVA-HHOI Track 4 (DINOv2 ego features).

FeatureKNN: retrieve the GT sequences of feature-similar TRAIN cells as the K=5
alternatives (exploits best-of-K). Optionally seed slots with the static GreedyK5
modal hedges (mix_static) so a bad-neighbor query still gets the global modes.
"""
import numpy as np
from . import feats as FT
from .baselines import GreedyK5Predictor, seq_key, rich_pool


class CondVerbalHedge:
    """Hedge-pack template with the VERBAL event types conditioned per-cell on the
    bge-m3 recent-transcript (logreg). Nonverbal modes stay global (video conditioning
    failed). base_alts = a hedge-pack 5-alt template; its verbal events' types are
    replaced by the top-n predicted utterance types for the cell."""
    def __init__(self, n_types=2, W=4.0, C=0.5, pool_fn=None, max_cells=2500, feats=("bge",)):
        self.n_types = n_types; self.W = W; self.C = C
        self.pool_fn = pool_fn or (lambda d: rich_pool(d, n_v=12, n_nv=10, mix=5))
        self.max_cells = max_cells; self.feats = feats

    def _feat(self, s, p, t_b):
        parts = []
        if "bge" in self.feats:
            parts.append(FT.text_window_feat(s, p, t_b, self.W))
        if "dialogue" in self.feats:
            parts.append(FT.dialogue_feat(s, p, t_b))
        return np.concatenate(parts)

    def fit(self, train_ds):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        g = GreedyK5Predictor(max_cells=self.max_cells, pool_fn=self.pool_fn)
        g.fit(train_ds)
        self.base_alts = g.alts
        X, y = [], []
        for s, segs in train_ds.items():
            for sid, seg in segs.items():
                for p in ("participant_a", "participant_b"):
                    verb = [e for e in seg["participants"][p]["events"] if len(e) == 2]
                    if verb:
                        X.append(self._feat(s, p, seg["t_b"]))
                        y.append(verb[0][0])
        self.scaler = StandardScaler().fit(np.stack(X))
        self.clf = LogisticRegression(max_iter=2000, C=self.C).fit(self.scaler.transform(np.stack(X)), y)
        self.classes = self.clf.classes_

    def predict(self, s, seg, p):
        f = self.scaler.transform(self._feat(s, p, seg["t_b"])[None])
        proba = self.clf.predict_proba(f)[0]
        top = [self.classes[i] for i in np.argsort(-proba)[:self.n_types]]
        out, vi = [], 0
        for alt in self.base_alts:
            na = []
            for e in alt:
                if len(e) == 2:
                    na.append([top[min(vi, len(top) - 1)], e[1]]); vi += 1
                else:
                    na.append(e)
            out.append(na)
        return out


class FeatureKNN:
    def __init__(self, K=5, M=120, W=1.5, pool="mean_last", mix_static=0):
        self.K = K; self.M = M; self.W = W; self.pool = pool; self.mix_static = mix_static
        self.static = None

    def fit(self, train_ds):
        X, seqs = [], []
        for s, segs in train_ds.items():
            for sid, seg in segs.items():
                for p in ("participant_a", "participant_b"):
                    X.append(FT.window_feat(s, p, seg["t_b"], self.W, self.pool))
                    seqs.append(seg["participants"][p]["events"])
        self.X = np.stack(X)
        self.seqs = seqs
        if self.mix_static:
            g = GreedyK5Predictor(max_cells=2500); g.fit(train_ds)
            self.static = g.alts

    def predict(self, sess, seg, p):
        q = FT.window_feat(sess, p, seg["t_b"], self.W, self.pool)
        order = np.argsort(-(self.X @ q))
        alts, seen = [], set()
        if self.static:
            for a in self.static[:self.mix_static]:
                k = seq_key(a)
                if k not in seen:
                    seen.add(k); alts.append(a)
        for i in order[:self.M]:
            a = self.seqs[i]; k = seq_key(a)
            if k not in seen:
                seen.add(k); alts.append(a)
            if len(alts) >= self.K:
                break
        return alts[:self.K] if alts else [[]]
