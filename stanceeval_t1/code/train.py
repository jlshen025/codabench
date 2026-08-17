"""
train.py — Fine-tune an Arabic encoder for StanceEval-2026 and evaluate via CV.

CV modes (mirror the official metric, see stance_lib.compute_metrics):
  loto  : Leave-One-Target-Out on train.csv (Track-2 UNSEEN proxy). Trains one model
          per held-out target, pools out-of-fold (OOF) preds, reports pooled + per-target Favg2.
  kfold : Stratified k-fold on train(+dev) (Track-1 SEEN proxy). Pools OOF.
  dev   : Train on train.csv, evaluate on dev.csv (fast Track-1 signal).

Always saves OOF probabilities (oof_proba.npz) so thresholds/ensembles can be tuned
post-hoc without retraining, and metrics.json (the verifiable result artifact).

Offline-safe: HF cache + offline flags set before importing transformers.
"""
import os, sys, json, argparse, time
os.environ.setdefault("HF_HOME", "<cache>/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

MODELS = {
    "marbertv2": os.path.expanduser("~/scratch/projects/stanceeval_shared/models/marbertv2"),
    "camelbert-mix": os.path.expanduser("~/scratch/projects/stanceeval_shared/models/camelbert-mix"),
    "arabert-twitter": "aubmindlab/bert-base-arabertv02-twitter",
}


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class StanceDS(Dataset):
    def __init__(self, df, tok, max_len):
        self.t = df[S.TARGET_COL].tolist()
        self.x = df["text_proc"].tolist()
        self.y = df["label"].tolist() if "label" in df.columns else None
        self.tok = tok; self.max_len = max_len

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        enc = self.tok(self.t[i], self.x[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.y is not None:
            item["labels"] = torch.tensor(int(self.y[i]), dtype=torch.long)
        return item


def train_one(train_df, val_df, model_path, args, device, return_model=False):
    """Fine-tune once; return best-epoch (val proba, metrics, hist), or the trained model if return_model (val_df=None)."""
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=3, id2label=S.ID2LABEL, label2id=S.LABEL2ID).to(device)
    tr_loader = DataLoader(StanceDS(train_df, tok, args.max_len), batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(StanceDS(val_df, tok, args.max_len), batch_size=64, shuffle=False) if val_df is not None else None

    # class weights (inverse freq) to fight Favor-skew, optional
    if args.class_weights:
        counts = np.bincount(train_df["label"].to_numpy(), minlength=3).astype(float)
        w = counts.sum() / (3.0 * np.clip(counts, 1, None))
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float, device=device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(tr_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    y_val = val_df["label"].to_numpy() if val_df is not None else None
    best = {"Favg2": -1.0}; best_proba = None; hist = []
    for ep in range(args.epochs):
        model.train()
        for batch in tr_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        if va_loader is None:
            continue
        model.eval(); probs = []
        with torch.no_grad():
            for batch in va_loader:
                batch.pop("labels", None)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(**batch).logits
                probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        proba = np.concatenate(probs, 0)
        m = S.compute_metrics(y_val, proba.argmax(1))
        hist.append({"epoch": ep + 1, **{k: round(v * 100, 2) for k, v in m.items()}})
        if m["Favg2"] > best["Favg2"]:
            best = m; best_proba = proba; best["epoch"] = ep + 1
    if return_model:
        return model
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return best_proba, best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="marbertv2")
    ap.add_argument("--cv", default="loto", choices=["loto", "kfold", "dev", "full"])
    ap.add_argument("--save_model", default="", help="dir to save full-data model (cv=full)")
    ap.add_argument("--extra_csv", default="", help="extra labeled CSV to augment TRAINING (e.g. synthetic)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--preprocess", default="baseline")
    ap.add_argument("--class_weights", action="store_true")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="subsample for local smoke test")
    ap.add_argument("--out", default=os.environ.get("EXP_OUTPUT_DIR", "./_out"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = MODELS.get(args.model, args.model)
    print(f"[cfg] model={args.model} cv={args.cv} ep={args.epochs} lr={args.lr} "
          f"bs={args.batch_size} maxlen={args.max_len} cw={args.class_weights} dev={device}", flush=True)

    train_df = S.load_labeled(S.TRAIN_CSV, args.preprocess)
    dev_df = S.load_labeled(S.DEV_CSV, args.preprocess)
    if args.limit:  # smoke test
        train_df = train_df.groupby(S.TARGET_COL, group_keys=False).head(max(args.limit // 3, 8)).reset_index(drop=True)
        dev_df = dev_df.head(args.limit).reset_index(drop=True)

    extra = S.load_labeled(args.extra_csv, args.preprocess) if args.extra_csv else None
    def aug(tr):
        tr = tr.reset_index(drop=True)
        return pd.concat([tr, extra], ignore_index=True) if extra is not None else tr
    if extra is not None:
        print(f"[aug] +{len(extra)} synthetic rows from {args.extra_csv}", flush=True)

    if args.cv == "full":
        full_df = aug(pd.concat([train_df, dev_df], ignore_index=True))
        model = train_one(full_df, None, model_path, args, device, return_model=True)
        save_dir = args.save_model or os.path.join(args.out, "model")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        AutoTokenizer.from_pretrained(model_path).save_pretrained(save_dir)
        json.dump({"model": args.model, "cv": "full", "epochs": args.epochs, "n_train": len(full_df),
                   "preprocess": args.preprocess, "max_len": args.max_len, "saved": save_dir},
                  open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
        print(f"[RESULT] saved full model -> {save_dir} (n_train={len(full_df)})", flush=True)
        return

    t0 = time.time()
    folds = []        # list of (name, val_df, best_proba, best_metrics, hist)
    if args.cv == "dev":
        bp, bm, h = train_one(aug(train_df), dev_df, model_path, args, device)
        folds.append(("dev", dev_df, bp, bm, h))
    elif args.cv == "loto":
        for tr_idx, va_idx, tname in S.leave_one_target_out(train_df):
            bp, bm, h = train_one(aug(train_df.loc[tr_idx]),
                                  train_df.loc[va_idx].reset_index(drop=True), model_path, args, device)
            print(f"[loto] held='{tname}' Favg2={bm['Favg2']*100:.2f} (best ep {bm['epoch']})", flush=True)
            folds.append((tname, train_df.loc[va_idx].reset_index(drop=True), bp, bm, h))
    elif args.cv == "kfold":
        pool = train_df  # k-fold on train only (dev kept as external check)
        for fi, (tr_idx, va_idx) in enumerate(S.stratified_kfold_indices(pool, args.n_splits, args.seed)):
            bp, bm, h = train_one(aug(pool.loc[tr_idx]),
                                  pool.loc[va_idx].reset_index(drop=True), model_path, args, device)
            print(f"[kfold] fold{fi} Favg2={bm['Favg2']*100:.2f}", flush=True)
            folds.append((f"fold{fi}", pool.loc[va_idx].reset_index(drop=True), bp, bm, h))

    # pool OOF
    all_proba = np.concatenate([f[2] for f in folds], 0)
    all_y = np.concatenate([f[1]["label"].to_numpy() for f in folds], 0)
    all_tgt = np.concatenate([f[1][S.TARGET_COL].to_numpy() for f in folds], 0)
    pooled = S.compute_metrics(all_y, all_proba.argmax(1))
    per_fold = {f[0]: {k: round(v * 100, 2) for k, v in f[3].items() if k != "epoch"} for f in folds}
    mean_fold_favg2 = float(np.mean([f[3]["Favg2"] for f in folds]) * 100)

    np.savez_compressed(os.path.join(args.out, "oof_proba.npz"),
                        proba=all_proba, y_true=all_y, target=all_tgt.astype(str))
    result = {
        "model": args.model, "cv": args.cv, "seed": args.seed, "epochs": args.epochs,
        "lr": args.lr, "batch_size": args.batch_size, "max_len": args.max_len,
        "preprocess": args.preprocess, "class_weights": args.class_weights,
        "pooled_Favg2": round(pooled["Favg2"] * 100, 2),
        "pooled_Favg3": round(pooled["Favg3"] * 100, 2),
        "pooled_Acc": round(pooled["Acc"] * 100, 2),
        "mean_fold_Favg2": round(mean_fold_favg2, 2),
        "per_fold": per_fold,
        "n_eval": int(len(all_y)), "runtime_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as fp:
        json.dump(result, fp, indent=2)
    print("[RESULT] " + json.dumps({k: result[k] for k in
          ["model", "cv", "pooled_Favg2", "pooled_Favg3", "mean_fold_Favg2", "n_eval", "runtime_sec"]}), flush=True)


if __name__ == "__main__":
    main()
