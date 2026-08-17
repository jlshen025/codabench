"""Train the deployed linear head — the only trained component of the system (14 KB).

Trains on all 4 UPDRS-labeled CARE-PD cohorts (augmented joint-mean 512-d features):
z-score by train statistics -> Dropout(0) -> Linear(512, 4), FocalLoss(alpha=1, gamma=1),
AdamW lr 1e-3 wd 1e-3 bs 64, StepLR gamma 0.99. Epoch-tune on a 15% within-train split by
macro-F1, then REFIT on 100% of the data for that epoch count — the CARE-PD benchmark protocol.

Saves head_jm_zscore_focal.npz: W(4,512), b(4), feat_mean(512), feat_std(512), lam=0.8.

Environment:
  FEATS_NPZ  feature file name inside CACHE_DIR   (default feats_magfs_augE_pj.npz)
  CACHE_DIR  where that file lives                (default ./cache)
  HEAD_SEED  torch/numpy seed                     (default 0 — the shipped head)
  OUT_DIR    where to write the head + json       (default .)
"""
import os, sys, json, numpy as np, torch, torch.nn as nn
_DIR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _DIR)
from scorer import macro_f1_4fixed

CACHE = os.environ.get("CACHE_DIR", "./cache")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def focal_loss(logits, targets, alpha=1.0, gamma=1.0):
    ce = nn.functional.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()


def build():
    return nn.Sequential(nn.Dropout(0.0), nn.Linear(512, 4)).to(dev)


def fit(Xtr, ytr, max_ep, bs=64, lr=1e-3, wd=1e-3, Xva=None, yva=None, patience=15):
    m = build()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.99)
    xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev)
    n = len(xt); best_f1, best_ep, best_state = -1, max_ep, None
    for ep in range(1, max_ep + 1):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); focal_loss(m(xt[idx]), yt[idx]).backward(); opt.step()
        sch.step()
        if Xva is not None:
            m.eval()
            with torch.no_grad():
                vf1 = macro_f1_4fixed(yva, m(torch.tensor(Xva, device=dev)).argmax(1).cpu().numpy())
            if vf1 > best_f1:
                best_f1, best_ep = vf1, ep
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    if Xva is not None and best_state is not None:
        m.load_state_dict(best_state)
    return m, best_ep, best_f1


def main():
    FEATS = os.environ.get("FEATS_NPZ", "feats_magfs_augE_pj.npz")
    SCALE = os.environ.get("SCALE", "zscore")     # zscore (MAGF) or raw (MixSTE)
    OUT = os.environ.get("OUT_NAME", "head_jm_zscore_focal")
    # The scorer is deterministic and the test set fixed, so a different head initialisation is a
    # genuinely different candidate. The shipped head is seed 0; see the README on seed spread.
    SEED = int(os.environ.get("HEAD_SEED", "0"))
    torch.manual_seed(SEED); np.random.seed(SEED)
    z = np.load(f"{CACHE}/{FEATS}", allow_pickle=True)
    X = z["X_side_glob"].astype(np.float32).reshape(-1, 17, 512).mean(1)   # jointmean (Naug,512)
    y = z["y"].astype(np.int64); walkidx = z["walkidx"].astype(np.int64)
    yr = y[walkidx]
    print(f"[train-head] FEATS={FEATS} SCALE={SCALE} X{X.shape} class_dist={np.bincount(yr)}", flush=True)

    feat_mean = X.mean(0)
    if SCALE == "zscore":
        feat_std = X.std(0) + 1e-6; Xs = (X - feat_mean) / feat_std
    else:                                          # raw: head trains on raw X (uncentered); center at inference by feat_mean
        feat_std = np.ones_like(feat_mean); Xs = X.copy()
    # phase 1: epoch-tune on 15% within-train val
    rng = np.random.RandomState(0); idx = rng.permutation(len(Xs))
    nva = int(0.15 * len(Xs)); va_i, tr_i = idx[:nva], idx[nva:]
    _, best_ep, best_vf1 = fit(Xs[tr_i], yr[tr_i], max_ep=80, Xva=Xs[va_i], yva=yr[va_i])
    print(f"[train-head] tuned best_epoch={best_ep} (val macroF1={best_vf1:.4f}); refitting on 100%", flush=True)
    # phase 2: refit on 100% for best_ep epochs
    m, _, _ = fit(Xs, yr, max_ep=best_ep)
    m.eval()
    with torch.no_grad():
        train_f1 = macro_f1_4fixed(yr, m(torch.tensor(Xs, device=dev)).argmax(1).cpu().numpy())
    lin = m[1]
    W = lin.weight.detach().cpu().numpy().astype(np.float32)   # (4,512)
    b = lin.bias.detach().cpu().numpy().astype(np.float32)     # (4,)
    od = os.environ.get("OUT_DIR", ".")
    os.makedirs(od, exist_ok=True)
    payload = dict(W=W, b=b, feat_mean=feat_mean.astype(np.float32), feat_std=feat_std.astype(np.float32),
                   lam=np.float32(0.8), classes=np.arange(4, dtype=np.int64), best_epoch=np.int64(best_ep),
                   scale=np.str_(SCALE))
    np.savez(os.path.join(od, f"{OUT}.npz"), **payload)
    np.savez(f"{CACHE}/{OUT}.npz", **payload)      # cache copy for packaging
    json.dump(dict(best_epoch=int(best_ep), val_f1=round(float(best_vf1), 4), scale=SCALE, feats=FEATS,
                   train_f1=round(float(train_f1), 4), W_shape=list(W.shape)),
              open(os.path.join(od, "train_head.json"), "w"), indent=2)
    print(f"[train-head] SAVED {OUT}.npz (scale={SCALE}) W{W.shape} best_ep={best_ep} train_f1={train_f1:.4f}", flush=True)
    print("ALLDONE_TRAINHEAD", flush=True)


if __name__ == "__main__":
    main()
