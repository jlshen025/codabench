"""fewshot_conv.py — retrieval FEW-SHOT "annotator-convention" voters for the WD test.

Lever 1, the named mechanism: the decisive rows are ones where the ANNOTATOR
CONVENTION (not semantics) decides, and the weak `ce` model wins the tiebreak precisely
because it was trained on these annotators. Frontier LLMs are convention-blind. So SHOW the
convention: per test tweet retrieve k nearest LABELED train rows (Women-empowerment first —
same women's-rights domain) and put them in the prompt WITH gold stance + sarcasm labels.

Retrieval = TF-IDF char 3-5 grams (robust for Arabic dialect/orthography, no GPU).
Batches share the union of their members' neighbours, so cost stays ~N/batch calls.

The dev-era "few-shot ruled out" kill was a SEEN-regime result; no WD voter has ever used
labeled examples, so this is an untested class.
"""
import os, sys, re, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_predict as LP
import stance_lib as S
from sklearn.feature_extraction.text import TfidfVectorizer

SYS_FS = ("You are an expert annotator for Arabic stance detection, working to a FIXED annotation "
          "standard. Below are REAL labeled examples from the same annotation project — study how these "
          "annotators actually assign labels (especially for sarcasm, mockery, jokes and rhetorical "
          "questions), then label the new tweets THE SAME WAY.\n"
          "Labels: Favor (author supports the target), Against (author opposes it), None (no personal stance).\n"
          "Match the annotators' conventions, not your own intuition. "
          "Output EXACTLY one line per tweet formatted '<number>|<Label>'. No other text.")

# Convention-GROUNDED few-shot rubric: states the scheme's ACTUAL rules (measured from the
# annotation columns) instead of the wrong "None = takes no side" definition every earlier prompt used.
SYS_FS_G = ("You are applying a FIXED Arabic stance-annotation scheme. Below are REAL labeled examples "
            "from that same project, each tagged with the annotators' own sub-category.\n"
            "The scheme's actual rules (measured over its labels):\n"
            "- Favor / Against are EXPLICIT-FIRST: ~95% of Favor and ~85% of Against rest on an explicit "
            "statement (F_Explicit / A_Explicit). Implicit stance (F_Implicit / A_Implicit) is rare and marked.\n"
            "- None means 'Not clear' (the stance CANNOT BE DETERMINED from the text) or 'Not Related' "
            "(the tweet is not about the target). None does NOT mean the author is neutral or takes no side.\n"
            "- Sarcasm is NOT a reliable Against cue: sarcastic tweets split ~48% Against / ~30% Favor / "
            "~22% None, and are 2.5x more likely than average to be 'Not clear'. Do NOT invert sarcasm to "
            "its ironic reading; judge what is actually stated, else None.\n"
            "Decision rule: explicit statement present -> label it; only a guessing/inference chain -> None.\n"
            "Label the new tweets the SAME WAY. Output EXACTLY one line per tweet as '<number>|<Label>' "
            "where Label is Favor, Against, or None. No other text.")


def build_index(train_csv, prefer_target, prefer_boost=2.0):
    df = pd.read_csv(train_csv, keep_default_na=False, dtype=str)
    for c in ("text", "target", "stance"):
        df[c] = df[c].astype(str).str.strip()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200000)
    X = vec.fit_transform(df["text"].tolist())
    boost = np.where(df["target"].values == prefer_target, prefer_boost, 1.0)
    return df, vec, X, boost


def reason_code(df, i):
    """The annotators' own sub-category for this row (A_Explicit / F_Implicit / Not clear / ...)."""
    for col in ("against_reason", "favor_reason", "none_reason"):
        if col in df.columns:
            v = str(df.at[i, col]).strip()
            if v:
                return v
    return ""


def demos_for(idxs, df, codes=True):
    lines = []
    for i in idxs:
        sarc = df.at[i, "sarcasm"].strip() if "sarcasm" in df.columns else ""
        tag = " [sarcastic]" if sarc.lower().startswith("y") else ""
        rc = reason_code(df, i) if codes else ""
        lab = f"{df.at[i,'stance']}" + (f" ({rc})" if rc else "")
        lines.append(f"[TARGET: {df.at[i,'target']}]{tag} {df.at[i,'text']}\n=> {lab}")
    return "LABELED EXAMPLES FROM THE SAME ANNOTATION PROJECT:\n\n" + "\n\n".join(lines) + "\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--csv", default="_eval/test_seen_norm.csv")
    ap.add_argument("--train", default=S.TRAIN_CSV)
    ap.add_argument("--k", type=int, default=12, help="demos per batch (union of members' neighbours)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--prefer_target", default="Women empowerment")
    ap.add_argument("--balance", type=int, default=1, help="1 = force >=2 demos of each stance class")
    ap.add_argument("--grounded", type=int, default=0, help="1 = convention-grounded rubric (SYS_FS_G)")
    ap.add_argument("--codes", type=int, default=0, help="1 = show the annotators' reason codes in demos")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys_prompt = SYS_FS_G if args.grounded else SYS_FS

    chain = LP.providers.chain_for(args.model)
    legs = [(c, c.get("model_map", {}).get(args.model, args.model))
            for _, c in reversed(chain) if c.get("api") == "chat" and c.get("base_url")]
    only = os.environ.get("AA_ONLY_URL")
    if only: legs = [(c, m) for c, m in legs if only in (c.get("base_url") or "")]
    if not legs: raise SystemExit(f"no chat endpoint for {args.model}")
    print(f"[fewshot] {args.model} legs={[c['base_url'] for c,_ in legs]}", flush=True)

    tr, vec, Xtr, boost = build_index(args.train, args.prefer_target)
    te = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    for c in ("text", "target"): te[c] = te[c].astype(str).str.strip()
    Xte = vec.transform(te["text"].tolist())
    sim = (Xte @ Xtr.T).toarray() * boost[None, :]          # [Nte, Ntr] boosted cosine-ish
    rows = [(te.at[i, "target"], te.at[i, "text"]) for i in te.index]

    votes = np.zeros((len(rows), 3))
    for b0 in range(0, len(rows), args.batch):
        chunk = rows[b0:b0 + args.batch]
        s = sim[b0:b0 + len(chunk)].max(0)                  # best similarity to ANY member
        cand = np.argsort(-s)
        pick = list(cand[:args.k])
        if args.balance:                                     # ensure each stance class is represented
            have = tr.loc[pick, "stance"].value_counts().to_dict()
            for st in ("Favor", "Against", "None"):
                need = 2 - have.get(st, 0)
                if need > 0:
                    extra = [j for j in cand if tr.at[j, "stance"] == st and j not in pick][:need]
                    pick += extra
        pick = sorted(pick, key=lambda j: -s[j])
        user = demos_for(pick, tr, codes=bool(args.codes)) + "NOW LABEL THESE, USING THE SAME STANDARD:\n\n" + "\n".join(
            f"{i+1}| [TARGET: {t}] {x}" for i, (t, x) in enumerate(chunk))
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
        content = LP.call_any(legs, msgs, "none", 30 * len(chunk) + 200, 0.0)
        got = {}
        for line in content.splitlines():
            m = LP.LBL_RE.match(line)
            if m: got[int(m.group(1))] = m.group(2).capitalize()
        if len(chunk) - len(got) > max(2, len(chunk) // 5):
            content = LP.call_any(legs, msgs, "none", 40 * len(chunk) + 300, 0.0)
            for line in content.splitlines():
                m = LP.LBL_RE.match(line)
                if m: got.setdefault(int(m.group(1)), m.group(2).capitalize())
        for j in range(len(chunk)):
            votes[b0 + j, S.LABEL2ID.get(got.get(j + 1, "None"), 2)] += 1.0
        print(f"  {b0+len(chunk)}/{len(rows)}", flush=True)

    pred = votes.argmax(1)
    np.savez_compressed(args.out, proba=votes, pred_id=pred, idx=te.index.to_numpy())
    import collections
    c = collections.Counter(pred.tolist())
    print(f"[RESULT] {args.out} dist A{c.get(0,0)}/F{c.get(1,0)}/N{c.get(2,0)} (true prior ~A160/F158/N34)", flush=True)


if __name__ == "__main__":
    main()
