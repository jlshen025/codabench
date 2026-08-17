"""debate.py — 2-question decomposition ("debate-lite", ZSMD-style) stance voter.

Instead of one 3-way call, ask TWO focused binary passes per tweet:
  - Favor-advocate:  is there ANY favor/support evidence toward the target?  Y/N
  - Against-advocate: is there ANY against/criticism evidence toward the target?  Y/N
Aggregate: (Y,N)->Favor  (N,Y)->Against  (N,N)->None  (Y,Y)->conflict (0.5/0.5, resolve when blending).
Targets the AGREEMENT-row errors the None-sweep can't touch; decorrelated from 3-way voters.
Stores favor_bit/against_bit so conflicts can be resolved flexibly downstream.
"""
import os, sys, re, argparse
import numpy as np, pandas as pd, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_predict as LP  # reuse providers + call_any (module-level loads .env)
import stance_lib as S

SYS_FAVOR = ("You analyze Arabic tweets about a TARGET topic. For each numbered tweet decide: does the author express "
    "ANY support, favor, endorsement, defense, approval, or celebration toward the TARGET (explicit OR implicit)? "
    "Answer strictly '<number>|Y' (some favor present) or '<number>|N' (no favor). One line per tweet, nothing else.")
SYS_AGAINST = ("You analyze Arabic tweets about a TARGET topic. For each numbered tweet decide: does the author express "
    "ANY opposition, criticism, rejection, distrust, mockery, sarcasm-against, fear, or warning toward the TARGET (explicit OR implicit)? "
    "Answer strictly '<number>|Y' (some opposition present) or '<number>|N' (no opposition). One line per tweet, nothing else.")
YN_RE = re.compile(r'^\s*(\d+)\s*[|\.\):\-]\s*([YN])', re.I)


def ask_bits(rows, legs, sys_prompt, batch):
    bits = np.zeros(len(rows), dtype=int)
    for b0 in range(0, len(rows), batch):
        chunk = rows[b0:b0 + batch]
        user = "Analyze each tweet toward its TARGET:\n\n" + "\n".join(
            f"{i+1}| [TARGET: {t}] {x}" for i, (t, x) in enumerate(chunk))
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
        content = LP.call_any(legs, msgs, "none", 10 * len(chunk) + 100, 0.0)
        got = {}
        for line in content.splitlines():
            m = YN_RE.match(line)
            if m:
                got[int(m.group(1))] = 1 if m.group(2).upper() == "Y" else 0
        if sum(1 for k in range(1, len(chunk)+1) if k not in got) > max(2, len(chunk)//5):
            content = LP.call_any(legs, msgs, "none", 14 * len(chunk) + 200, 0.0)
            for line in content.splitlines():
                m = YN_RE.match(line)
                if m: got.setdefault(int(m.group(1)), 1 if m.group(2).upper()=="Y" else 0)
        for j in range(len(chunk)):
            bits[b0 + j] = got.get(j + 1, 0)
    return bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--csv", default="_eval/test_seen_norm.csv")
    ap.add_argument("--batch", type=int, default=44)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target_ctx", default="")
    args = ap.parse_args()
    if args.target_ctx:
        globals()["SYS_FAVOR"] += "\nTARGET CONTEXT: " + args.target_ctx
        globals()["SYS_AGAINST"] += "\nTARGET CONTEXT: " + args.target_ctx
    chain = LP.providers.chain_for(args.model)
    legs = [(c, c.get("model_map", {}).get(args.model, args.model))
            for _, c in reversed(chain) if c.get("api") == "chat" and c.get("base_url")]
    only = os.environ.get("AA_ONLY_URL")
    if only: legs = [(c, m) for c, m in legs if only in (c.get("base_url") or "")]
    if not legs: raise SystemExit(f"no chat endpoint for {args.model}")
    print(f"[debate] {args.model} legs={[c['base_url'] for c,_ in legs]}", flush=True)
    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    for c in ("text", "target"): df[c] = df[c].astype(str).str.strip()
    rows = [(df.at[i, "target"], df.at[i, "text"]) for i in df.index]
    fav = ask_bits(rows, legs, SYS_FAVOR, args.batch); print("favor pass done", flush=True)
    agn = ask_bits(rows, legs, SYS_AGAINST, args.batch); print("against pass done", flush=True)
    proba = np.zeros((len(rows), 3))
    for i in range(len(rows)):
        f, a = fav[i], agn[i]
        if f and not a: proba[i, 1] = 1.0
        elif a and not f: proba[i, 0] = 1.0
        elif not f and not a: proba[i, 2] = 1.0
        else: proba[i] = [0.5, 0.5, 0.0]  # both-Y conflict; resolve downstream
    out = {"proba": proba, "pred_id": proba.argmax(1), "favor_bit": fav, "against_bit": agn, "idx": df.index.to_numpy()}
    np.savez_compressed(args.out, **out)
    print("bits: favorY=%d againstY=%d bothY=%d bothN=%d" % (
        int(fav.sum()), int(agn.sum()), int(((fav == 1) & (agn == 1)).sum()), int(((fav == 0) & (agn == 0)).sum())), flush=True)
    print("dist(argmax, both-Y->Against):", dict(collections.Counter(proba.argmax(1).tolist())), flush=True)


if __name__ == "__main__":
    main()
