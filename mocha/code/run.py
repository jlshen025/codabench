"""MoCha 2026 — inference entry point. Defines predict(data) -> {subject: {walk: label}}.

A frozen MotionAGFormer-S encoder with a linear head trained on the CARE-PD benchmark's winning
LODO recipe (configs/best_configs_augmented/LODO/motionagformer_*): FocalLoss(alpha=1, gamma=1) +
AdamW on z-scored joint-mean features. Only torch and numpy are needed — the SMPL forward
kinematics and the H36M joint regressor are vendored alongside this file.

Per-walk scoring:
  Xz = (X_raw - feat_mean) / feat_std     # z-score by TRAIN stats, stored in the head .npz
  Xz -= lam * Xz.mean(0)                  # transductive centering in z-score space
  P  = softmax(Xz @ W.T + b)              # the linear head

Those posteriors are then aggregated within each subject, pooled across neighbouring subjects, and
decided at a q-divisor operating point (tau > 0) or by plain argmax (tau = 0); a collapse guard
falls back to argmax. Every stage is driven by config.json — see the repository README.
"""
import os, sys, json, types, numpy as np, torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

if "timm" not in sys.modules:
    _t = types.ModuleType("timm"); _l = types.ModuleType("timm.layers")
    class _DropPath(nn.Module):
        def __init__(self, drop_prob=0.): super().__init__()
        def forward(self, x): return x
    _l.DropPath = _DropPath; _t.layers = _l
    sys.modules["timm"] = _t; sys.modules["timm.layers"] = _l

from smpl_fk_eval import SMPLH36M
from model.motionagformer.MotionAGFormer import MotionAGFormer

CLIP = 81
_LAM = 0.8            # transductive centering shrink (set from head.npz at load; this is the fallback)
_TAU = 0.0            # 0 = per-walk argmax (probe 1). >0 = v14-style posterior-overriding minority boost.
_MIN_FOR_ADJUST = 8
MAGF_CFG = dict(n_layers=26, dim_in=3, dim_feat=64, dim_rep=512, dim_out=3, mlp_ratio=4,
                attn_drop=0., drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5,
                use_adaptive_fusion=True, num_heads=8, qkv_bias=False, qkv_scale=None, hierarchical=False,
                num_joints=17, use_temporal_similarity=True, temporal_connection_len=1, use_tcn=False,
                graph_only=False, neighbour_num=2, n_frames=CLIP)

MIRROR_IDX = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]


def _aug_mirror(j):
    jm = j[:, MIRROR_IDX, :].copy(); jm[..., 0] *= -1.0; return jm


def _aug_rot(j, deg):
    th = np.radians(deg); c, s = np.cos(th), np.sin(th)
    jr = j.copy(); x = j[..., 0]; z = j[..., 2]
    jr[..., 0] = c * x + s * z; jr[..., 2] = -s * x + c * z
    return jr


def _aug_speed(j, factor):
    T = j.shape[0]; Tn = max(2, int(round(T / factor)))
    if Tn == T:
        return j.copy()
    src = np.linspace(0.0, T - 1.0, Tn)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, T - 1)
    w = (src - lo)[:, None, None]
    return (j[lo] * (1 - w) + j[hi] * w).astype(j.dtype)


def _apply_aug(j, name):
    if name == "orig":
        return j
    if name == "mirror":
        return _aug_mirror(j)
    if name.startswith("mir_spd"):
        return _aug_speed(_aug_mirror(j), float(name[len("mir_spd"):]))
    if name.startswith("rot"):
        return _aug_rot(j, float(name[len("rot"):]))
    if name.startswith("spd"):
        return _aug_speed(j, float(name[len("spd"):]))
    raise ValueError(name)


def _crop_scale(motion):
    res = motion.copy()
    valid = motion[motion[..., 2] != 0][:, :2]
    if len(valid) < 4:
        return np.zeros_like(motion)
    xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
    ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros_like(motion)
    xs = (xmin + xmax - scale) / 2; ys = (ymin + ymax - scale) / 2
    res[..., :2] = (motion[..., :2] - np.array([xs, ys])) / scale
    res[..., :2] = (res[..., :2] - 0.5) * 2
    return np.clip(res, -1, 1)


def _project(joints, view):
    j = joints.copy(); glob = view.endswith("_glob"); base = view[:-5] if glob else view
    if not glob:
        j[:, :, 0] -= j[:, 0:1, 0]; j[:, :, 2] -= j[:, 0:1, 2]
    if base == "side":
        x2, y2 = j[:, :, 2], -j[:, :, 1]
    elif base == "front":
        x2, y2 = j[:, :, 0], -j[:, :, 1]
    else:
        raise ValueError(view)
    return np.stack([x2, y2, np.ones_like(x2)], axis=-1).astype(np.float32)


def _clips(seq):
    T = seq.shape[0]
    if T < CLIP:
        pad = np.zeros((CLIP - T, 17, 3), np.float32)
        return np.concatenate([seq, pad], 0)[None], np.concatenate([np.ones(T), np.zeros(CLIP - T)])[None].astype(np.float32)
    cs, pms, s = [], [], 0
    while T - s >= CLIP:
        cs.append(seq[s:s + CLIP]); pms.append(np.ones(CLIP, np.float32)); s += CLIP
    return np.stack(cs), np.stack(pms)


def _softmax(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _logit_adjust_decide(P, tau):
    P = np.asarray(P, np.float64); N, K = P.shape
    z0 = P.argmax(1)
    if tau <= 0:
        return z0
    q = np.clip(P.mean(0), 1e-6, None) ** tau
    Q = P / q[None, :]
    z = (Q / Q.sum(1, keepdims=True)).argmax(1)
    nk = np.bincount(z, minlength=K)
    if (nk > 0).sum() < 2 or nk.max() > 0.85 * N:
        return z0
    return z


def _exp_macro_f1(z, Pw, W):
    """Plug-in expected macro-F1 for a subject assignment z. The metric counts WALKS while the
    prediction is constant per SUBJECT, so subject s carries weight W[s] = its walk count.
    E[true_k] is constant in z; only the predicted mass and the expected TP move."""
    Et = (W[:, None] * Pw).sum(0)
    pred = np.zeros(4); tp = np.zeros(4)
    for k in range(4):
        m = z == k
        if m.any():
            pred[k] = W[m].sum(); tp[k] = (W[m] * Pw[m, k]).sum()
    den = pred + Et
    return float(np.mean(np.where(den > 0, 2.0 * tp / np.maximum(den, 1e-12), 0.0)))


def _pool_beyond_subject(A, subj, Z, nbr_smooth, nbr_k, post_shrink):
    """Pool ONE LEVEL ABOVE the subject. A = the subject-aggregated posterior (one distinct row per
    subject, repeated across its walks); Z = the per-walk centered embedding.
    nbr_smooth blends a subject toward its k nearest subjects (cosine on subject-mean embeddings);
    post_shrink blends toward the global test-mean posterior. Both preserve the one-label-per-subject
    structure the server rewards -- they only change WHICH label, by borrowing strength across
    similar subjects rather than re-admitting per-walk noise."""
    out = A
    if nbr_smooth > 0:
        us = np.unique(subj)
        if len(us) >= 3:
            Ps = np.stack([A[subj == u][0] for u in us])
            Zs = np.stack([Z[subj == u].mean(0) for u in us])
            Zn = Zs / (np.linalg.norm(Zs, axis=1, keepdims=True) + 1e-9)
            S = Zn @ Zn.T
            k = int(min(max(1, nbr_k), len(us)))
            idx = np.argsort(-S, axis=1)[:, :k]                  # includes self (cosine 1.0)
            Ps2 = (1 - nbr_smooth) * Ps + nbr_smooth * Ps[idx].mean(1)
            out = out.copy()
            for u, p in zip(us, Ps2):
                out[subj == u] = p
    if post_shrink > 0:
        out = (1 - post_shrink) * out + post_shrink * out.mean(0, keepdims=True)
    return out


def _expf1_decode(A, subj, z0, max_pass=60):
    """Walk-count-weighted expected-macro-F1 coordinate ascent over subject labels. Deterministic and
    monotone in the objective; initialised from the op-point decode so it can only improve on it."""
    us = np.unique(subj)
    Pw = np.stack([A[subj == u].mean(0) for u in us])
    W = np.array([float((subj == u).sum()) for u in us])
    z = np.array([int(np.bincount(z0[subj == u], minlength=4).argmax()) for u in us])
    cur = _exp_macro_f1(z, Pw, W)
    for _ in range(max_pass):
        moved = False
        for s in range(len(z)):
            old = z[s]; bk, bv = old, cur
            for k in range(4):
                if k == old:
                    continue
                z[s] = k; v = _exp_macro_f1(z, Pw, W)
                if v > bv + 1e-12:
                    bk, bv = k, v
            z[s] = bk
            if bk != old:
                cur, moved = bv, True
        if not moved:
            break
    out = z0.copy()
    for u, k in zip(us, z):
        out[subj == u] = k
    return out


class _Model:
    def __init__(self):
        self.cfg = json.load(open(os.path.join(_ROOT, "config.json")))
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.smpl = SMPLH36M(device=self.dev)
        m = MotionAGFormer(act_layer=nn.GELU, **MAGF_CFG)
        _ckpt = os.path.join(_ROOT, "motionagformer-s-h36m.pth.tr")
        try:
            sd = torch.load(_ckpt, map_location="cpu", weights_only=False)["model"]
        except TypeError:
            sd = torch.load(_ckpt, map_location="cpu")["model"]
        m.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}, strict=True)
        self.enc = m.to(self.dev).eval()
        h = np.load(os.path.join(_ROOT, "head_jm_zscore_focal.npz"))
        self.W = h["W"].astype(np.float64); self.b = h["b"].astype(np.float64)
        self.feat_mean = h["feat_mean"].astype(np.float64); self.feat_std = h["feat_std"].astype(np.float64)
        self.lam = float(h["lam"]); self.classes = h["classes"]
        # v19b: allow the transductive-centering shrink to be overridden from config (default = head's 0.8,
        # tuned back at v12 PRE-aggregation; the optimum may shift once predictions are subject-level).
        self.lam = float(self.cfg.get("center_lam", self.lam))
        self.views = self.cfg["views"]; self.reduce = self.cfg["reduce"]
        self.tta = self.cfg.get("tta", ["orig"]); self.tau = float(self.cfg.get("tau", _TAU))
        self.subj_smooth = float(self.cfg.get("subj_smooth", 0.0))   # v19: within-subject posterior smoothing
        # v23a: the transductive centering offset estimates the DOMAIN mean. Walk-weighting lets a few
        # high-walk subjects dominate it (walk counts vary ~1-67), so "subject" weights each subject once.
        self.center_unit = str(self.cfg.get("center_unit", "walk"))
        # v23b: decision-level vote over NEIGHBOURING configs of the SAME model (not different models —
        # those all failed). Stabilises borderline subjects, where single-subject flips actually happen.
        self.vote_configs = self.cfg.get("vote_configs", [])
        # v24a: subject decode. "op" = q-div logit adjust (deployed). "expf1" = walk-count-weighted
        # expected-macro-F1 coordinate ascent over the subject label assignment (CV +0.0193).
        self.decode = str(self.cfg.get("decode", "op"))
        # v24b: confidence-gated per-walk override. Full aggregation forces one label per subject, but
        # 56.4% of labeled subjects carry MIXED walk labels, so ~30% of walks are unreachable. A gate
        # re-admits a walk's OWN posterior only where it is confident, keeping aggregation elsewhere.
        # 0 disables. gate_target: "raw" = the walk's plain argmax, "op" = its op-point-adjusted argmax.
        self.gate_thr = float(self.cfg.get("gate_thr", 0.0))
        self.gate_target = str(self.cfg.get("gate_target", "raw"))
        # v25: POOL BEYOND THE SUBJECT. The server has pushed both pooling knobs to their boundary
        # (subject aggregation lam->1.0 = +0.143, transductive centering c->0.9-1.0), and every lever
        # that sharpens or de-aggregates has lost. The untested direction is therefore MORE pooling,
        # one level above the subject. Two forms, both off by default:
        #   nbr_smooth: blend each subject's posterior toward its k nearest SUBJECTS in embedding
        #     space (cosine on subject-mean features). E5 measured the manifold severity-organized
        #     (kNN-agree 0.61); E5 itself was per-WALK and PRE-aggregation, so this is a new object.
        #   post_shrink: blend toward the global test-mean posterior — the posterior-space analogue
        #     of the transductive feature centering that is already my single biggest alignment win.
        self.nbr_smooth = float(self.cfg.get("nbr_smooth", 0.0))
        self.nbr_k = int(self.cfg.get("nbr_k", 5))
        self.post_shrink = float(self.cfg.get("post_shrink", 0.0))

    @torch.no_grad()
    def _encode(self, j):
        parts = []
        for v in self.views:
            seq = _project(j, v); clips_arr, pm = _clips(seq)
            clips = np.stack([_crop_scale(c) for c in clips_arr])
            xb = torch.as_tensor(clips, dtype=torch.float32, device=self.dev)
            rep = self.enc(xb, return_rep=True)
            mb = torch.as_tensor(pm, dtype=torch.float32, device=self.dev)
            pooled = ((rep * mb[:, :, None, None]).sum(1) / mb.sum(1).clamp(min=1e-6)[:, None, None])
            wfeat = pooled.mean(0).cpu().numpy()                # (17,512)
            parts.append(wfeat.mean(0) if self.reduce == "jointmean" else wfeat.reshape(-1))
        return np.concatenate(parts).astype(np.float64)         # (512,) for jointmean single view

    @torch.no_grad()
    def walk_feats(self, pose, trans, fps):
        stride = max(1, int(round(float(fps) / 30.0)))
        pose = np.ascontiguousarray(np.asarray(pose)[::stride])
        trans = np.ascontiguousarray(np.asarray(trans)[::stride])
        j = self.smpl.joints(pose, trans)
        return [self._encode(_apply_aug(j, a)) for a in self.tta]   # list over TTA views

    def zscored(self, Xraw):
        return (np.asarray(Xraw, np.float64) - self.feat_mean) / self.feat_std

    def center_offset(self, Xz, subj=None):
        """Domain-mean estimate. 'walk' = pooled mean (each WALK one vote, so many-walk subjects
        dominate). 'subject' = mean of per-subject means (each SUBJECT one vote) — the unbiased
        estimator of the site mean when walk counts are unequal."""
        if self.center_unit == "subject" and subj is not None:
            return np.stack([Xz[subj == u].mean(0) for u in np.unique(subj)], 0).mean(0)
        return Xz.mean(0)

    def posteriors(self, Xraw, subj=None, lam=None, tau_unused=None, return_feats=False):
        Xz = self.zscored(Xraw)
        c = self.lam if lam is None else float(lam)
        Xz = Xz - c * self.center_offset(Xz, subj)
        P = _softmax(Xz @ self.W.T + self.b)
        return (P, Xz) if return_feats else P


_M = None


def predict(data: dict) -> dict:
    global _M
    if _M is None:
        _M = _Model()
    # pass 1: per-walk per-TTA raw feats
    cache = {}
    for sid, walks in data.items():
        for wid, e in walks.items():
            try:
                cache[(sid, wid)] = _M.walk_feats(e["pose"], e["trans"], e.get("fps", 30))
            except Exception:
                cache[(sid, wid)] = None
    keys = [k for k in cache if cache[k] is not None]
    out = {sid: {} for sid in data}
    if len(keys) < 3:                                            # degenerate: majority fallback
        for sid, walks in data.items():
            for wid in walks:
                out[sid][wid] = 1
        return out
    n_tta = len(_M.tta)
    subj_arr = np.array([k[0] for k in keys])

    def _decide(lam_c, tau):
        """One full config: centering strength lam_c -> head -> subject aggregation -> op-point.
        Returns (labels, aggregated posterior, PRE-aggregation per-walk posterior). v24 needs the
        third: the gate's whole point is the per-walk evidence that aggregation averages away."""
        Ps, Zc = None, None
        for t in range(n_tta):
            Xraw = np.stack([cache[k][t] for k in keys], 0)
            Pt, Zt = _M.posteriors(Xraw, subj=subj_arr, lam=lam_c, return_feats=True)
            Ps = Pt if Ps is None else Ps + Pt
            Zc = Zt if Zc is None else Zc + Zt
        Praw = Ps / n_tta; Zc = Zc / n_tta
        Pc = Praw
        if _M.subj_smooth > 0:
            A = Praw.copy()
            for u in np.unique(subj_arr):
                m = subj_arr == u
                A[m] = (1 - _M.subj_smooth) * Praw[m] + _M.subj_smooth * Praw[m].mean(0, keepdims=True)
            Pc = A
        # v25: pool one level above the subject, BEFORE the op-point reads the marginal.
        if _M.nbr_smooth > 0 or _M.post_shrink > 0:
            Pc = _pool_beyond_subject(Pc, subj_arr, Zc, _M.nbr_smooth, _M.nbr_k, _M.post_shrink)
        z = _logit_adjust_decide(Pc, tau) if (len(keys) >= _MIN_FOR_ADJUST and tau > 0) else Pc.argmax(1)
        return z, Pc, Praw

    if _M.vote_configs:
        votes = [_decide(float(c.get("center_lam", _M.lam)), float(c.get("tau", _M.tau)))[0]
                 for c in _M.vote_configs]
        V = np.stack(votes, 0)                                   # (n_cfg, N)
        z = V[0].copy()                                          # tie-break -> first config (the best single)
        for i in range(V.shape[1]):
            cnt = np.bincount(V[:, i], minlength=4)
            if cnt.max() > 1:                                    # a real majority overrides the default
                z[i] = int(cnt.argmax())
        for k, zi in zip(keys, z):
            out[k[0]][k[1]] = int(_M.classes[int(zi)])
        for sid, walks in data.items():
            for wid in walks:
                if wid not in out[sid]:
                    out[sid][wid] = 1
        return out

    z, A, Praw = _decide(_M.lam, _M.tau)
    # v24a: replace the 1-parameter q-div op-point with the walk-count-weighted expected-macro-F1
    # decode over subject labels (initialised from z, so it can only improve that objective).
    if _M.decode == "expf1" and len(keys) >= _MIN_FOR_ADJUST:
        z = _expf1_decode(A, subj_arr, z)
    # v24b: confidence-gated per-walk override — the only stage that can escape one-label-per-subject.
    if _M.gate_thr > 0 and len(keys) >= _MIN_FOR_ADJUST:
        if _M.gate_target == "op" and _M.tau > 0:
            tgt = _logit_adjust_decide(Praw, _M.tau)
        else:
            tgt = Praw.argmax(1)
        sel = (Praw.max(1) >= _M.gate_thr) & (tgt != z)
        z = z.copy(); z[sel] = tgt[sel]
    for k, zi in zip(keys, z):
        out[k[0]][k[1]] = int(_M.classes[int(zi)])
    for sid, walks in data.items():                             # fill any missing walk
        for wid in walks:
            if wid not in out[sid]:
                out[sid][wid] = 1
    return out
