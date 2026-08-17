"""sarc_emit.py — emit the MTL encoders' SARCASM aux-head probabilities on the WD test.

Lever 5. Mechanism: the weak `ce` blend wins the disagreement-row tiebreak because
it catches implicit-Against sarcasm the frontier LLMs read as None/Favor. The sarcasm head is
TARGET-AGNOSTIC (it never learned seen-target stance cues), so unlike the stance head it carries
no unseen-target liability. Emit P(sarcastic) as an alternative tiebreak / None-flip ordering.

Saves _eval/sarc_test.npz: p_sarc [N] (mean over the given encoders), per-model probs.
CPU, 352 rows, 2 small encoders -> seconds.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from train_mtl import MTLModel


def emit(model_dir, texts, targets, maxlen=128, bs=32, device="cpu"):
    cfg = json.load(open(os.path.join(model_dir, "mtl_config.json")))
    base = cfg.get("base_model") or cfg.get("model")
    tok = AutoTokenizer.from_pretrained(model_dir)
    m = MTLModel(base)
    sd = torch.load(os.path.join(model_dir, "mtl_state.pt"), map_location="cpu")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"    [warn] missing keys: {list(missing)[:4]}", flush=True)
    m.eval().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            bt, bg = texts[i:i + bs], targets[i:i + bs]
            enc = tok(list(bg), list(bt), truncation=True, max_length=maxlen,
                      padding=True, return_tensors="pt").to(device)
            _, _, lsk, _ = m(**enc)
            out.append(torch.softmax(lsk, -1)[:, 1].cpu().numpy())   # P(sarcastic)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="results/deploy_enc_mb_s42/model,results/deploy_enc_atw_s42/model")
    ap.add_argument("--csv", default="_eval/test_seen_norm.csv")
    ap.add_argument("--out", default="_eval/sarc_test.npz")
    args = ap.parse_args()
    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    texts = df["text"].astype(str).str.strip().tolist()
    targets = df["target"].astype(str).str.strip().tolist()
    per = {}
    for md in args.models.split(","):
        md = md.strip()
        if not os.path.isdir(md):
            print(f"  SKIP missing {md}", flush=True); continue
        print(f"  emitting {md} ...", flush=True)
        per[os.path.basename(os.path.dirname(md))] = emit(md, texts, targets)
    if not per:
        raise SystemExit("no models emitted")
    stack = np.stack(list(per.values()), 0)
    p = stack.mean(0)
    np.savez_compressed(args.out, p_sarc=p, **{f"p_{k}": v for k, v in per.items()})
    print(f"[RESULT] {args.out}  n={len(p)}  mean P(sarc)={p.mean():.3f}  "
          f">0.5: {(p > 0.5).sum()}  models={list(per)}", flush=True)


if __name__ == "__main__":
    main()
