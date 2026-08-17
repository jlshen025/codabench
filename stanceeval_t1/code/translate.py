"""translate.py — one-time Arabic→English translation of the WD test set.

Input-representation decorrelation family (lever 2): same models, different
input language → a genuinely different error surface (NOT the failed prompt-B reframe).
Also unblocks the distinct models whose documented failure was the ARABIC batched
label-format, not the task.

Writes _eval/test_seen_en.csv with the SAME id/target columns and row ORDER,
text replaced by the English translation (target gloss kept as-is).
"""
import os, sys, re, argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_predict as LP

SYS_T = ("You are a professional Arabic→English translator specializing in Saudi/Gulf social media. "
         "Translate each numbered Arabic tweet into natural English. PRESERVE the author's stance, tone, "
         "sarcasm, irony, insults, and hashtag meaning exactly — do NOT neutralize, soften, or editorialize. "
         "Keep it one line per tweet. Output EXACTLY '<number>| <english translation>' per line, nothing else.")
LINE_RE = re.compile(r'^\s*(\d+)\s*\|\s*(.+)$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--csv", default="_eval/test_seen_norm.csv")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", default="_eval/test_seen_en.csv")
    args = ap.parse_args()
    chain = LP.providers.chain_for(args.model)
    legs = [(c, c.get("model_map", {}).get(args.model, args.model))
            for _, c in reversed(chain) if c.get("api") == "chat" and c.get("base_url")]
    if not legs:
        raise SystemExit(f"no chat endpoint for {args.model}")
    print(f"[translate] {args.model} legs={[c['base_url'] for c,_ in legs]}", flush=True)
    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    for c in ("text", "target"):
        df[c] = df[c].astype(str).str.strip()
    texts = df["text"].tolist()
    out = [None] * len(texts)
    for b0 in range(0, len(texts), args.batch):
        chunk = texts[b0:b0 + args.batch]
        user = "Translate each tweet to English:\n\n" + "\n".join(f"{i+1}| {t}" for i, t in enumerate(chunk))
        msgs = [{"role": "system", "content": SYS_T}, {"role": "user", "content": user}]
        content = LP.call_any(legs, msgs, "none", 120 * len(chunk) + 300, 0.0)
        got = {}
        for line in content.splitlines():
            m = LINE_RE.match(line)
            if m:
                got[int(m.group(1))] = m.group(2).strip()
        miss = [k for k in range(1, len(chunk) + 1) if k not in got]
        if miss:  # one retry for the stragglers
            content = LP.call_any(legs, msgs, "none", 160 * len(chunk) + 400, 0.0)
            for line in content.splitlines():
                m = LINE_RE.match(line)
                if m:
                    got.setdefault(int(m.group(1)), m.group(2).strip())
        for j in range(len(chunk)):
            out[b0 + j] = got.get(j + 1) or texts[b0 + j]  # fall back to the Arabic original
        print(f"  {b0+len(chunk)}/{len(texts)}", flush=True)
    n_fallback = sum(1 for i, t in enumerate(out) if t == texts[i])
    df_en = df.copy()
    df_en["text"] = out
    df_en.to_csv(args.out, index=False)
    print(f"[RESULT] wrote {args.out} rows={len(df_en)} untranslated_fallbacks={n_fallback}", flush=True)


if __name__ == "__main__":
    main()
