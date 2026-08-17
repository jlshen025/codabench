#!/usr/bin/env python3
"""Non-verbal base model -- candidate pool + per-attribute MLP ensemble (SHIPPED).

- build_candidates / vocabs_from_cands: non-verbal candidate tuples
  (high_level_action, low_level_action, target, modifier) observed >= min_count times,
  with their static base rate.
- MLP / encode / train_fold: a shared-trunk MLP with FOUR PER-ATTRIBUTE sigmoid heads
  (h, l, target, modifier) trained with class-balanced binary cross-entropy -- this is a
  per-attribute multi-label model, NOT a closed-set tuple classifier. At inference the four
  marginals are multiplied to score a candidate tuple; the rerankers average an 11-seed
  ensemble of these models (one train_fold per seed).

Consumed by rerank.py / rerank_v3.py / integ_tp.py. (The stand-alone LOSO selection grid
that lived here has been removed; the non-verbal base-model selection lives in integ_tp.py.)
"""
import numpy as np
import torch, torch.nn as nn
from collections import Counter


def build_candidates(gt, train, min_count=2):
    ct = Counter()
    ninst = 0
    for sid in train:
        for seg, (tb, te, s) in gt[sid]["nonverbal"].items():
            ninst += 2  # two subjects per segment
            for (subj, h, l, tg, mod) in s:
                ct[(h, l, tg, mod)] += 1
    cands = [(t, c) for t, c in ct.items() if c >= min_count]
    base = {t: c / max(ninst, 1) for t, c in cands}
    return [t for t, c in cands], base


def vocabs_from_cands(cands):
    return {"h": sorted({t[0] for t in cands}), "l": sorted({t[1] for t in cands}),
            "t": sorted({t[2] for t in cands}), "m": sorted({t[3] for t in cands})}


def encode(inst, vocab):
    idx = {k: {c: i for i, c in enumerate(v)} for k, v in vocab.items()}
    X = np.stack([x[2] for x in inst]).astype(np.float32)
    Y = {k: np.zeros((len(inst), len(vocab[k])), np.float32) for k in vocab}
    # nonverbal instance: (seg, subj, x, Hs, Ls, Ts, Ms)
    for r, (seg, subj, x, Hs, Ls, Ts, Ms) in enumerate(inst):
        for c in Hs:
            if c in idx["h"]: Y["h"][r, idx["h"][c]] = 1
        for c in Ls:
            if c in idx["l"]: Y["l"][r, idx["l"][c]] = 1
        for c in Ts:
            if c in idx["t"]: Y["t"][r, idx["t"][c]] = 1
        for c in Ms:
            if c in idx["m"]: Y["m"][r, idx["m"][c]] = 1
    return X, Y


class MLP(nn.Module):
    def __init__(self, din, dh, heads, p):
        super().__init__()
        self.bb = nn.Sequential(nn.Linear(din, dh), nn.GELU(), nn.Dropout(p),
                                nn.Linear(dh, dh), nn.GELU(), nn.Dropout(p))
        self.heads = nn.ModuleDict({k: nn.Linear(dh, n) for k, n in heads.items()})
    def forward(self, x):
        z = self.bb(x); return {k: h(z) for k, h in self.heads.items()}


def train_fold(Xtr, Ytr, vocab, dev, dh=512, p=0.3, epochs=80, lr=1e-3, wd=1e-4, bs=512, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xn = (Xtr - mu) / sd
    model = MLP(Xtr.shape[1], dh, {k: len(v) for k, v in vocab.items()}, p).to(dev)
    pw = {k: torch.tensor(np.clip((len(Ytr[k]) - Ytr[k].sum(0)) / np.maximum(Ytr[k].sum(0), 1), 1, 20),
                          dtype=torch.float32, device=dev) for k in vocab}
    crit = {k: nn.BCEWithLogitsLoss(pos_weight=pw[k]) for k in vocab}
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(Xn, device=dev); Yt = {k: torch.tensor(Ytr[k], device=dev) for k in vocab}
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev); model.train()
        for i in range(0, n, bs):
            b = perm[i:i + bs]; opt.zero_grad()
            out = model(Xt[b]); loss = sum(crit[k](out[k], Yt[k][b]) for k in vocab)
            loss.backward(); opt.step()
    return model, mu, sd
