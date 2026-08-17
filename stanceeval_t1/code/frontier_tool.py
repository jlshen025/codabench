"""
frontier_tool.py — drive a frontier LLM zero-shot stance voter through an external
chat interface, for models whose relays 404 on direct /v1/chat/completions and so
cannot be reached by llm_predict.py's `requests` path.

Two modes:
  emit     — print the full prompt (SYS + numbered batch) for batch K of a CSV.
             Send it to the model; save the reply to <raw_dir>/b<K>.txt.
  assemble — parse all <raw_dir>/b*.txt back into a proba npz aligned to the CSV
             row order (batch-local number n in file bK -> global row K*bs+(n-1)).
             With --score (CSV has a `stance` col) prints pooled + per-target Favg2.

Proba is soft: a class gets 1.0 (one-hot on the parsed label); missing rows -> None.
Column order matches stance_lib.LABEL2ID (Against=0, Favor=1, None=2).
"""
import sys, os, argparse, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

SYS = ("You are an expert annotator for Arabic stance detection. For each numbered Arabic tweet and its "
       "TARGET topic, output the author's stance TOWARD THE TARGET as exactly one of: Favor, Against, None.\n"
       "- Favor: the author supports, endorses, or is glad about the target.\n"
       "- Against: the author opposes, criticizes, distrusts, mocks, or fears the target (or its mandate).\n"
       "- None: no clear stance, off-topic, purely factual/news, or ambiguous.\n"
       "Sarcasm and rhetorical questions usually signal Against. Judge the AUTHOR's own stance.\n"
       "Output EXACTLY one line per tweet formatted '<number>|<Label>' (Label = Favor, Against, or None). No other text.")

LBL_RE = re.compile(r'^\s*(\d+)\s*[|\.\):\-]\s*(favor|against|none)\b', re.I)


def load(csv):
    df = pd.read_csv(csv, keep_default_na=False, dtype=str)
    df.columns = [c.strip().lstrip('﻿') for c in df.columns]
    for c in ("text", "target"):
        df[c] = df[c].astype(str).str.strip()
    return df


def build_prompt(rows):
    user = "Classify each tweet's stance toward its TARGET:\n\n" + "\n".join(
        f"{i+1}| [TARGET: {tgt}] {txt}" for i, (tgt, txt) in enumerate(rows))
    return SYS + "\n\n" + user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["emit", "assemble"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--bs", type=int, default=44)
    ap.add_argument("--k", type=int, default=0, help="emit: which batch")
    ap.add_argument("--raw_dir", default="_eval/frontier_raw")
    ap.add_argument("--out", default="_eval/frontier_test.npz")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    df = load(args.csv)
    n = len(df)
    nb = (n + args.bs - 1) // args.bs
    rows = [(df.at[i, "target"], df.at[i, "text"]) for i in df.index]

    if args.mode == "emit":
        s, e = args.k * args.bs, min((args.k + 1) * args.bs, n)
        print(f"### BATCH {args.k}/{nb-1}  rows {s}..{e-1}  ({e-s} tweets)  — save reply to {args.raw_dir}/b{args.k}.txt", file=sys.stderr)
        print(build_prompt(rows[s:e]))
        return

    # assemble
    proba = np.zeros((n, 3), dtype=float)
    got_total = 0
    for k in range(nb):
        f = f"{args.raw_dir}/b{k}.txt"
        if not os.path.exists(f):
            print(f"[MISS] {f} absent — rows {k*args.bs}.. left as None", file=sys.stderr)
            continue
        s = k * args.bs
        e = min(s + args.bs, n)
        content = open(f, encoding="utf-8").read()
        got = {}
        for line in content.splitlines():
            m = LBL_RE.match(line)
            if m:
                got[int(m.group(1))] = m.group(2).capitalize()
        for j in range(e - s):
            lab = got.get(j + 1, "None")
            proba[s + j, S.LABEL2ID.get(lab, 2)] = 1.0
            if (j + 1) in got:
                got_total += 1
        miss = [j + 1 for j in range(e - s) if (j + 1) not in got]
        if miss:
            print(f"[b{k}] parsed {e-s-len(miss)}/{e-s}  missing local#: {miss}", file=sys.stderr)
    pred = proba.argmax(1)
    from collections import Counter
    c = Counter(pred.tolist())
    print(f"parsed {got_total}/{n} rows  dist={{ {', '.join(f'{S.ID2LABEL[k]}:{c.get(k,0)}' for k in range(3))} }}")
    np.savez_compressed(args.out, proba=proba, pred_id=pred)
    print(f"wrote {args.out}")
    if args.score and "stance" in df.columns and (df["stance"].str.strip() != "").all():
        y = np.array([S.LABEL2ID[df.at[i, "stance"].strip()] for i in df.index])
        m = S.compute_metrics(y, pred)
        print(f"[SCORE] pooled Favg2={m['Favg2']*100:.2f} Favg3={m['Favg3']*100:.2f} Acc={m.get('Accuracy', float('nan'))*100:.2f}")
        for t in dict.fromkeys(df["target"].tolist()):
            mask = (df["target"] == t).to_numpy()
            mm = S.compute_metrics(y[mask], pred[mask])
            print(f"   {t}: Favg2={mm['Favg2']*100:.2f} n={int(mask.sum())}")


if __name__ == "__main__":
    main()
