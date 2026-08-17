#!/usr/bin/env python
"""train_none_det.py — a SUPERVISED binary None-vs-stance detector.

WHY SUPERVISED.  A frontier LLM asked "is this tweet's stance recoverable" reaches only
~0.31 precision on dev (3.2x lift), far under the 0.55/0.57 break-even that makes a
flip pay. Yet two rivals run None heads at 0.84 and 1.00 precision. The gap is almost
certainly convention: gold None here means "the ANNOTATORS could not determine the
stance" (none_reason = Not clear 313 / Not Related 11), which is an annotator habit, not
a semantic property an LLM can infer from the text. Only a model trained on those
annotators can learn it. That is the one None lever this project never built -- the
fine-tuned pipeline exists but enters the ensemble at weight 1e-3 as a pure tie-breaker,
so its None channel has never been given its own decision.

WHY BINARY.  Collapsing Favor/Against into one "stance" class puts all 3729 stance rows
behind one boundary instead of splitting capacity across a 3-way problem, and it removes
the polarity axis -- the part that does NOT transfer to an unseen target. What is left is
"is a stance expressed", which should transfer far better.

AUX SENTIMENT HEAD.  Measured on train+dev: P(None | sentiment=Neutral) = 0.229 vs
0.018 for Positive, and 74.7% of all gold None rows are Neutral. Sentiment is nearly
target-agnostic, so it is a legitimate transferring proxy; as an auxiliary head it
regularizes the encoder toward exactly the feature that carries the None signal.

VALIDATION = LEAVE-ONE-TARGET-OUT, not k-fold.  The blind test is a single HELD-OUT
target (Women Driving, absent from training), so in-distribution CV would overstate
transfer. LOTO trains on 2 targets and scores the 3rd, which is the deployment regime.
The number that decides everything is precision@k on the held-out target: it must clear
~0.56 for a flip to pay at all.

Usage
-----
    python train_none_det.py --mode loto           # 3 folds, report transfer precision
    python train_none_det.py --mode full --out_npz _eval/nd_sup_test.npz   # fit all, predict test
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S  # noqa: E402

MODELS = {
    "marbertv2": "<scratch>/models/marbertv2",
    "camelbert": "<scratch>/models/camelbert-mix",
    "araberttw": "<scratch>/models_final_v2/araberttw_base",
}


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class DS(Dataset):
    def __init__(self, df, tok, max_len):
        self.t = df[S.TARGET_COL].tolist()
        self.x = df["text_proc"].tolist()
        self.y = df["is_none"].tolist()
        self.ys = df["sent_label"].tolist()
        self.tok, self.max_len = tok, max_len

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        enc = self.tok(self.t[i], self.x[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.y[i]))
        item["sent"] = torch.tensor(int(self.ys[i]))
        return item


class NoneModel(nn.Module):
    def __init__(self, path, p_drop=0.1):
        super().__init__()
        seq = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
        self.encoder = seq.base_model
        H = self.encoder.config.hidden_size
        self.drop = nn.Dropout(p_drop)
        self.none = nn.Linear(H, 2)
        self.sent = nn.Linear(H, 3)

    def forward(self, **b):
        keys = {k: b[k] for k in ("input_ids", "attention_mask", "token_type_ids") if k in b}
        h = self.encoder(**keys).last_hidden_state[:, 0]
        hd = self.drop(h)
        return self.none(hd), self.sent(hd)


def load_all(preprocess="light"):
    tr = pd.read_csv(S.TRAIN_CSV, keep_default_na=False, dtype=str)
    dv = pd.read_csv(S.DEV_CSV, keep_default_na=False, dtype=str)
    df = pd.concat([tr, dv], ignore_index=True)
    fn = S.PREPROCESSORS[preprocess]
    df["text_proc"] = df[S.TEXT_COL].astype(str).map(fn)
    df[S.TARGET_COL] = df[S.TARGET_COL].astype(str).str.strip()
    df["is_none"] = (df[S.LABEL_COL].str.strip() == "None").astype(int)
    df["sent_label"] = df["sentiment"].str.strip().map(S.SENT2ID).fillna(-100).astype(int)
    return df


def run_epochs(model, tr_dl, args, device, n_pos, n_neg):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(tr_dl) * args.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1),
                                              pct_start=0.1, anneal_strategy="linear")
    # Class weighting: None is ~9.5% of rows. Without it the model collapses to
    # all-stance; capped at `--pos_weight` so it does not over-emit either.
    w = torch.tensor([1.0, min(args.pos_weight, n_neg / max(n_pos, 1))], device=device)
    ce = nn.CrossEntropyLoss(weight=w)
    ce_aux = nn.CrossEntropyLoss(ignore_index=-100)
    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for b in tr_dl:
            b = {k: v.to(device) for k, v in b.items()}
            ln, ls = model(**b)
            loss = ce(ln, b["labels"]) + args.lam_sent * ce_aux(ls, b["sent"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot += loss.item()
        print("    ep%d loss=%.4f" % (ep + 1, tot / max(len(tr_dl), 1)), flush=True)
    return model


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    out = []
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        ln, _ = model(**b)
        out.append(F.softmax(ln, -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def prec_report(score, y, label):
    """precision@k on the held-out target — the only number that matters."""
    order = np.argsort(-score)
    n_gold = int(y.sum())
    base = y.mean()
    print("  [%s] n=%d gold_none=%d base=%.3f" % (label, len(y), n_gold, base))
    res = {}
    for k in [10, 15, 20, 25, 30, n_gold]:
        if k > len(y) or k < 1:
            continue
        p = y[order[:k]].sum() / k
        res["p@%d" % k] = round(float(p), 4)
        print("    p@%-3d = %.3f  (lift %.2f)%s" % (k, p, p / base,
                                                    "   <== CLEARS break-even" if p >= 0.56 else ""))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loto", choices=["loto", "full"])
    ap.add_argument("--model", default="marbertv2", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--lam_sent", type=float, default=0.3)
    ap.add_argument("--pos_weight", type=float, default=4.0)
    ap.add_argument("--preprocess", default="light")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_csv", default="")
    ap.add_argument("--out_npz", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = MODELS[args.model]
    df = load_all(args.preprocess)
    print("[data] n=%d none=%d (%.3f)  device=%s model=%s"
          % (len(df), df["is_none"].sum(), df["is_none"].mean(), device, args.model), flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    out = {"args": vars(args)}

    if args.mode == "loto":
        for held in S.SEEN_TARGETS:
            print("\n=== LOTO fold: held-out target = %s ===" % held, flush=True)
            trd = df[df[S.TARGET_COL] != held].reset_index(drop=True)
            vad = df[df[S.TARGET_COL] == held].reset_index(drop=True)
            set_seed(args.seed)
            model = NoneModel(path).to(device)
            tr_dl = DataLoader(DS(trd, tok, args.max_len), batch_size=args.batch_size, shuffle=True)
            va_dl = DataLoader(DS(vad, tok, args.max_len), batch_size=64, shuffle=False)
            run_epochs(model, tr_dl, args, device, int(trd["is_none"].sum()),
                       int((1 - trd["is_none"]).sum()))
            sc = predict(model, va_dl, device)
            out[held] = prec_report(sc, vad["is_none"].to_numpy(), held)
            np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_eval",
                                             "ndsup_loto_%s.npz" % held.split()[0]),
                                score=sc, y=vad["is_none"].to_numpy())
            del model
            torch.cuda.empty_cache()
    else:
        set_seed(args.seed)
        model = NoneModel(path).to(device)
        tr_dl = DataLoader(DS(df, tok, args.max_len), batch_size=args.batch_size, shuffle=True)
        run_epochs(model, tr_dl, args, device, int(df["is_none"].sum()),
                   int((1 - df["is_none"]).sum()))
        te = pd.read_csv(args.test_csv, keep_default_na=False, dtype=str)
        tcol = S.TARGET_COL if S.TARGET_COL in te.columns else "Target"
        te["text_proc"] = te[S.TEXT_COL].astype(str).map(S.PREPROCESSORS[args.preprocess])
        te[S.TARGET_COL] = te[tcol].astype(str).str.strip()
        te["is_none"] = 0
        te["sent_label"] = -100
        dl = DataLoader(DS(te, tok, args.max_len), batch_size=64, shuffle=False)
        sc = predict(model, dl, device)
        np.savez_compressed(args.out_npz, score=sc)
        out["test_mean"] = float(sc.mean())
        out["test_top34_mean"] = float(np.sort(sc)[-34:].mean())
        print("[test] wrote %s  mean=%.4f  top34 mean=%.4f"
              % (args.out_npz, sc.mean(), np.sort(sc)[-34:].mean()))

    print("[RESULT] " + json.dumps(out, default=str))


if __name__ == "__main__":
    main()
