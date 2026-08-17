"""
train_mtl.py — Multi-task fine-tuning: stance + auxiliary (sentiment, sarcasm) heads.

The 2024 StanceEval winner's decisive trick: auxiliary heads regularize the shared
encoder. Only the STANCE head is used at inference; sentiment/sarcasm are auxiliary.
Loss = L_stance + lam_sent*L_sent + lam_sarc*L_sarc (aux CE with ignore_index=-100).

Encoder is obtained via AutoModelForSequenceClassification(...).base_model so the
converted bert.*-prefixed safetensors load correctly (a bare AutoModel would mismatch).
Same CV protocol / OOF saving / metrics.json as train.py.
"""
import os, sys, json, argparse, time
os.environ.setdefault("HF_HOME", "<cache>/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

MODELS = {
    "marbertv2": os.path.expanduser("~/scratch/projects/stanceeval_shared/models/marbertv2"),
    "camelbert-mix": os.path.expanduser("~/scratch/projects/stanceeval_shared/models/camelbert-mix"),
    "arabert-twitter": "aubmindlab/bert-base-arabertv02-twitter",
}


def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class MTLDS(Dataset):
    def __init__(self, df, tok, max_len, conf_weight="none", conf_mean=1.0):
        self.t = df[S.TARGET_COL].tolist(); self.x = df["text_proc"].tolist()
        self.y = df["label"].tolist()
        self.ys = df["sent_label"].tolist() if "sent_label" in df.columns else [-100] * len(df)
        self.yk = df["sarc_label"].tolist() if "sarc_label" in df.columns else [-100] * len(df)
        raw = df["conf"].tolist() if "conf" in df.columns else [1.0] * len(df)
        self.w = [(float(c) / conf_mean) if conf_weight == "linear" else 1.0 for c in raw]
        self.tok = tok; self.max_len = max_len

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        enc = self.tok(self.t[i], self.x[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.y[i]))
        item["sent"] = torch.tensor(int(self.ys[i]))
        item["sarc"] = torch.tensor(int(self.yk[i]))
        item["w"] = torch.tensor(float(self.w[i]), dtype=torch.float32)
        return item


def supcon_loss(feats, labels, tau=0.1):
    """Supervised contrastive loss (Khosla 2020 / Gunel 2021) over L2-normalized feats [B,D].
    Pull same-stance anchors together, push different-stance apart; anchors without an
    in-batch positive are skipped. Returns a scalar in float."""
    B = feats.size(0)
    self_mask = ~torch.eye(B, dtype=torch.bool, device=feats.device)
    sim = (feats @ feats.t()) / tau
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()           # numerical stability
    exp_sim = torch.exp(sim) * self_mask
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & self_mask
    pos_count = pos_mask.sum(1)
    valid = pos_count > 0
    if not valid.any():
        return feats.sum() * 0.0
    mlpp = (pos_mask.float() * log_prob).sum(1)[valid] / pos_count[valid].float()
    return -mlpp.mean()


class MTLModel(nn.Module):
    def __init__(self, path, p_drop=0.1, use_proj=False):
        super().__init__()
        seq = AutoModelForSequenceClassification.from_pretrained(path, num_labels=3)
        self.encoder = seq.base_model            # correctly-loaded BertModel
        H = self.encoder.config.hidden_size
        self.drop = nn.Dropout(p_drop)
        self.stance = nn.Linear(H, 3); self.sent = nn.Linear(H, 3); self.sarc = nn.Linear(H, 2)
        # SupCon projection head (Gunel 2021 joint CE+SupCon); created ONLY when enabled so the
        # CE-only path stays byte-identical (an extra head's init would shift the RNG stream).
        self.proj = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 128)) if use_proj else None

    def forward(self, **batch):
        keys = {k: batch[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in batch}
        h = self.encoder(**keys).last_hidden_state[:, 0]
        z = F.normalize(self.proj(h), dim=1) if self.proj is not None else None
        hd = self.drop(h)
        return self.stance(hd), self.sent(hd), self.sarc(hd), z


def train_one(train_df, val_df, model_path, args, device, return_model=False):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = MTLModel(model_path, use_proj=(getattr(args, "supcon", 0.0) > 0)).to(device)
    cw = getattr(args, "conf_weight", "none")
    conf_mean = float(train_df["conf"].mean()) if (cw == "linear" and "conf" in train_df.columns) else 1.0
    if cw == "linear":
        print(f"[conf] stance-loss weighting by stance:confidence (mean={conf_mean:.4f}) "
              f"— down-weights ambiguous boundary rows", flush=True)
    tr = DataLoader(MTLDS(train_df, tok, args.max_len, conf_weight=cw, conf_mean=conf_mean),
                    batch_size=args.batch_size, shuffle=True)
    va = DataLoader(MTLDS(val_df, tok, args.max_len), batch_size=64, shuffle=False) if val_df is not None else None
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    ce_none = nn.CrossEntropyLoss(reduction="none")   # per-example, for confidence weighting
    ce_aux = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = len(tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total), total)
    amp = device.type == "cuda"; scaler = torch.cuda.amp.GradScaler(enabled=amp)
    y_val = val_df["label"].to_numpy() if val_df is not None else None
    best = {"Favg2": -1}; best_proba = None
    for ep in range(args.epochs):
        model.train()
        for b in tr:
            ys = b.pop("sent").to(device); yk = b.pop("sarc").to(device); yl = b.pop("labels").to(device)
            wts = b.pop("w").to(device)
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                ls, lse, lsk, z = model(**b)
                if cw == "none":
                    l_stance = ce(ls, yl)
                else:
                    l_stance = (ce_none(ls, yl) * wts).sum() / wts.sum().clamp(min=1e-8)
                loss = l_stance + args.lam_sent * ce_aux(lse, ys) + args.lam_sarc * ce_aux(lsk, yk)
            if getattr(args, "supcon", 0.0) > 0 and z is not None:   # joint SupCon, fp32 outside autocast
                loss = loss + args.supcon * supcon_loss(z.float(), yl, args.supcon_tau)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        if va is None:
            continue
        model.eval(); P = []
        with torch.no_grad():
            for b in va:
                for k in ("sent", "sarc", "labels", "w"): b.pop(k, None)
                b = {k: v.to(device) for k, v in b.items()}
                with torch.autocast(device_type="cuda", enabled=amp):
                    ls, _, _, _ = model(**b)
                P.append(torch.softmax(ls.float(), 1).cpu().numpy())
        proba = np.concatenate(P, 0); m = S.compute_metrics(y_val, proba.argmax(1))
        if m["Favg2"] > best["Favg2"]:
            best = m; best["epoch"] = ep + 1; best_proba = proba
    if return_model:
        return model
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return best_proba, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="arabert-twitter")
    ap.add_argument("--cv", default="loto", choices=["loto", "kfold", "dev", "full"])
    ap.add_argument("--save_model", default="", help="dir to save full-data MTL model (cv=full)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--preprocess", default="baseline")
    ap.add_argument("--lam_sent", type=float, default=0.5)
    ap.add_argument("--lam_sarc", type=float, default=0.2)
    ap.add_argument("--label_smooth", type=float, default=0.0, help="stance-head label smoothing (boundary-calibration lever)")
    ap.add_argument("--conf_weight", default="none", choices=["none", "linear"],
                    help="per-instance stance-loss weighting by stance:confidence (down-weights ambiguous rows; distinct from uniform LS/E3)")
    ap.add_argument("--supcon", type=float, default=0.0,
                    help="weight for joint supervised contrastive loss on the stance representation (Gunel 2021; 0=off, ROBUSTNESS lever)")
    ap.add_argument("--supcon_tau", type=float, default=0.1, help="SupCon temperature")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("EXP_OUTPUT_DIR", "./_out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp = MODELS.get(args.model, args.model)
    print(f"[cfg] MTL model={args.model} cv={args.cv} ep={args.epochs} lam_sent={args.lam_sent} "
          f"lam_sarc={args.lam_sarc} dev={device}", flush=True)
    train_df = S.load_labeled(S.TRAIN_CSV, args.preprocess); dev_df = S.load_labeled(S.DEV_CSV, args.preprocess)
    if args.limit:
        train_df = train_df.groupby(S.TARGET_COL, group_keys=False).head(max(args.limit // 3, 8)).reset_index(drop=True)
        dev_df = dev_df.head(args.limit).reset_index(drop=True)
    if args.cv == "full":
        full_df = pd.concat([train_df, dev_df], ignore_index=True)
        model = train_one(full_df, None, mp, args, device, return_model=True)
        save_dir = args.save_model or os.path.join(args.out, "model")
        os.makedirs(save_dir, exist_ok=True)
        # drop the training-only SupCon projection head so the deploy loads into a
        # standard MTLModel(use_proj=False) via predict_test.py (stance head is all inference needs).
        sd = {k: v for k, v in model.state_dict().items() if not k.startswith("proj.")}
        torch.save(sd, os.path.join(save_dir, "mtl_state.pt"))
        AutoTokenizer.from_pretrained(mp).save_pretrained(save_dir)
        json.dump({"arch": "mtl", "base_model": mp, "max_len": args.max_len, "model": args.model,
                   "lam_sent": args.lam_sent, "lam_sarc": args.lam_sarc, "epochs": args.epochs,
                   "supcon": args.supcon, "n_train": len(full_df), "preprocess": args.preprocess},
                  open(os.path.join(save_dir, "mtl_config.json"), "w"), indent=2)
        json.dump({"model": args.model, "cv": "full", "mtl": True, "saved": save_dir, "n_train": len(full_df)},
                  open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
        print(f"[RESULT] saved full MTL model -> {save_dir} (n_train={len(full_df)})", flush=True)
        return
    t0 = time.time(); folds = []
    if args.cv == "dev":
        bp, bm = train_one(train_df, dev_df, mp, args, device); folds.append(("dev", dev_df, bp, bm))
    elif args.cv == "loto":
        for tri, vai, tn in S.leave_one_target_out(train_df):
            bp, bm = train_one(train_df.loc[tri].reset_index(drop=True), train_df.loc[vai].reset_index(drop=True), mp, args, device)
            print(f"[loto] held='{tn}' Favg2={bm['Favg2']*100:.2f}", flush=True)
            folds.append((tn, train_df.loc[vai].reset_index(drop=True), bp, bm))
    elif args.cv == "kfold":
        for fi, (tri, vai) in enumerate(S.stratified_kfold_indices(train_df, args.n_splits, args.seed)):
            bp, bm = train_one(train_df.loc[tri].reset_index(drop=True), train_df.loc[vai].reset_index(drop=True), mp, args, device)
            folds.append((f"fold{fi}", train_df.loc[vai].reset_index(drop=True), bp, bm))
    all_p = np.concatenate([f[2] for f in folds], 0); all_y = np.concatenate([f[1]["label"].to_numpy() for f in folds], 0)
    all_t = np.concatenate([f[1][S.TARGET_COL].to_numpy() for f in folds], 0)
    pooled = S.compute_metrics(all_y, all_p.argmax(1))
    np.savez_compressed(os.path.join(args.out, "oof_proba.npz"), proba=all_p, y_true=all_y, target=all_t.astype(str))
    res = {"model": args.model, "cv": args.cv, "mtl": True, "lam_sent": args.lam_sent, "lam_sarc": args.lam_sarc,
           "conf_weight": args.conf_weight, "supcon": args.supcon, "supcon_tau": args.supcon_tau,
           "seed": args.seed, "epochs": args.epochs, "max_len": args.max_len, "preprocess": args.preprocess,
           "class_weights": False, "pooled_Favg2": round(pooled["Favg2"] * 100, 2),
           "pooled_Favg3": round(pooled["Favg3"] * 100, 2), "pooled_Acc": round(pooled["Acc"] * 100, 2),
           "mean_fold_Favg2": round(float(np.mean([f[3]["Favg2"] for f in folds]) * 100), 2),
           "per_fold": {f[0]: {k: round(v * 100, 2) for k, v in f[3].items() if k != "epoch"} for f in folds},
           "n_eval": int(len(all_y)), "runtime_sec": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print("[RESULT] " + json.dumps({k: res[k] for k in ["model", "cv", "mtl", "lam_sent", "lam_sarc", "pooled_Favg2", "pooled_Favg3", "mean_fold_Favg2", "runtime_sec"]}), flush=True)


if __name__ == "__main__":
    main()
