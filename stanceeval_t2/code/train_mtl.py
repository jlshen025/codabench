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
os.environ["HF_HOME"] = "<cache>/huggingface"; os.environ["HF_HUB_CACHE"] = "<cache>/huggingface/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

MODELS = {
    "marbertv2": ("<scratch>/models/marbertv2"),
    "camelbert-mix": ("<scratch>/models/camelbert-mix"),
    "arabert-twitter": "aubmindlab/bert-base-arabertv02-twitter",
}


def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class MTLDS(Dataset):
    def __init__(self, df, tok, max_len):
        self.t = df[S.TARGET_COL].tolist(); self.x = df["text_proc"].tolist()
        self.y = df["label"].tolist()
        self.ys = df["sent_label"].tolist() if "sent_label" in df.columns else [-100] * len(df)
        self.yk = df["sarc_label"].tolist() if "sarc_label" in df.columns else [-100] * len(df)
        self.tok = tok; self.max_len = max_len

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        enc = self.tok(self.t[i], self.x[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.y[i]))
        item["sent"] = torch.tensor(int(self.ys[i]))
        item["sarc"] = torch.tensor(int(self.yk[i]))
        return item


class MTLModel(nn.Module):
    def __init__(self, path, p_drop=0.1):
        super().__init__()
        seq = AutoModelForSequenceClassification.from_pretrained(path, num_labels=3)
        self.encoder = seq.base_model            # correctly-loaded BertModel
        H = self.encoder.config.hidden_size
        self.drop = nn.Dropout(p_drop)
        self.stance = nn.Linear(H, 3); self.sent = nn.Linear(H, 3); self.sarc = nn.Linear(H, 2)

    def forward(self, **batch):
        keys = {k: batch[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in batch}
        h = self.encoder(**keys).last_hidden_state[:, 0]
        h = self.drop(h)
        return self.stance(h), self.sent(h), self.sarc(h)


def train_one(train_df, val_df, model_path, args, device, return_model=False):
    tok = AutoTokenizer.from_pretrained(model_path)
    model = MTLModel(model_path).to(device)
    tr = DataLoader(MTLDS(train_df, tok, args.max_len), batch_size=args.batch_size, shuffle=True)
    va = DataLoader(MTLDS(val_df, tok, args.max_len), batch_size=64, shuffle=False) if val_df is not None else None
    ce = nn.CrossEntropyLoss(); ce_aux = nn.CrossEntropyLoss(ignore_index=-100)
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
            b = {k: v.to(device) for k, v in b.items()}
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                ls, lse, lsk = model(**b)
                loss = ce(ls, yl) + args.lam_sent * ce_aux(lse, ys) + args.lam_sarc * ce_aux(lsk, yk)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
        if va is None:
            continue
        model.eval(); P = []
        with torch.no_grad():
            for b in va:
                for k in ("sent", "sarc", "labels"): b.pop(k, None)
                b = {k: v.to(device) for k, v in b.items()}
                with torch.autocast(device_type="cuda", enabled=amp):
                    ls, _, _ = model(**b)
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
        torch.save(model.state_dict(), os.path.join(save_dir, "mtl_state.pt"))
        AutoTokenizer.from_pretrained(mp).save_pretrained(save_dir)
        json.dump({"arch": "mtl", "base_model": mp, "max_len": args.max_len, "model": args.model,
                   "lam_sent": args.lam_sent, "lam_sarc": args.lam_sarc, "epochs": args.epochs,
                   "n_train": len(full_df), "preprocess": args.preprocess},
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
