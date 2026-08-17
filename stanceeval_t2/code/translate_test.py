"""Translate the blind test tweets Arabic->English once, cached to _llm/test_en.csv.

Lever 3 (input-representation decorrelation): a stance read over an English rendering
is a genuinely different view of the same input than the Arabic read, so it is a candidate
equal-strength DECORRELATED partner (the piece E53 lacked). Translation is of the RELEASED test
INPUTS only — no labels, no leakage.
"""
import os, sys, json, time, argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_predict import _provider_for, call_api

MSA_SYS = ("You are an expert Arabic editor. Rewrite each numbered tweet into clear MODERN STANDARD ARABIC. "
           "Normalise Gulf/Saudi dialect words to their MSA equivalents, spell out abbreviations, and VERBALISE "
           "every emoji as an explicit Arabic phrase describing the attitude it conveys (e.g. sarcasm, anger, joy). "
           "PRESERVE the author's stance exactly: sarcasm, mockery, rhetorical questions and hedging must remain "
           "recognisable. Do not add commentary. Output EXACTLY one line per tweet formatted '<number>| <msa>'. "
           "No other text.")

SYS = ("You are a professional Arabic->English translator specialising in Gulf/Saudi social media. "
       "Translate each numbered tweet into natural English. PRESERVE stance-bearing signals exactly: "
       "sarcasm, mockery, rhetorical questions, hedging, emphasis, dialect attitude markers and emoji "
       "sentiment. Do not neutralise or soften opinions. Do not add commentary. "
       "Output EXACTLY one line per tweet formatted '<number>| <english>'. No other text.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--provider", default="sub2apigpt")
    ap.add_argument("--csv", default="test_norm.csv")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", default="_llm/test_en.csv")
    ap.add_argument("--mode", default="en", choices=["en", "msa"])
    args = ap.parse_args()
    _, cfg = _provider_for(args.model, args.provider); wire = cfg["wire_model"]
    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    texts = [" ".join(str(t).split()) for t in df["text"]]
    out, t0 = {}, time.time()
    for b0 in range(0, len(texts), args.batch):
        chunk = texts[b0:b0 + args.batch]
        user = "\n".join(f"{i+1}| {t}" for i, t in enumerate(chunk))
        _sys = MSA_SYS if args.mode == "msa" else SYS
        got = {}
        for attempt in range(2):
            content = call_api(cfg, wire, [{"role": "system", "content": _sys},
                                           {"role": "user", "content": user}], "none", 90 * len(chunk) + 800, 0.0)
            for line in content.splitlines():
                if "|" in line:
                    h, _, tail = line.partition("|")
                    h = h.strip()
                    if h.isdigit() and tail.strip():
                        got[int(h)] = tail.strip()
            if len(got) >= len(chunk) - 1:
                break
        for j in range(len(chunk)):
            out[b0 + j] = got.get(j + 1) or chunk[j]      # fall back to the Arabic on a miss
        print(f"  {b0+len(chunk)}/{len(texts)}", flush=True)
    df["text"] = [out[i] for i in range(len(texts))]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    miss = sum(1 for i in range(len(texts)) if out[i] == texts[i])
    print(f"[RESULT] {json.dumps({'n': len(texts), 'untranslated_fallback': miss, 'sec': round(time.time()-t0,1)})}", flush=True)


if __name__ == "__main__":
    main()
