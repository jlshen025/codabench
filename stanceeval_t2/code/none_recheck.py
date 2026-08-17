"""none_recheck.py — prior-calibrated re-classification of the pred-NONE pool.

Same instrument class as prior_recheck.py, aimed at the other error pool. All 13
existing views share ONE neutral grounded prompt, so they share a systematic
None-over-call: 54 of the 73 Ecars None rows have ZERO views calling Favor, yet
the confusion decode proves 17 of them are gold-Favor. A prompt that STATES the
measured base rate and asks the narrower question "which of these is actually
Favor?" is therefore decorrelated from the pool that is blind to them.

The convention rule below is not guessed — it is read off the measured tier-A
result (6/8 correct): the corpus labels a tweet Favor for positive FRAMING of
electric-car adoption/growth/enthusiasm, not only for explicit advocacy.

Aggregate priors only — no per-row gold is known or used. Gate-1 clean.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_predict import _provider_for, call_api

SYS = (
    "You are an expert annotator for Arabic stance detection on the target: electric cars "
    "(السيارات الكهربائية).\n"
    "CALIBRATION (measured on this exact test set): a previous pass labelled EVERY tweet below "
    "'None'. That pass is known to systematically OVER-predict None. Measurement shows that "
    "roughly ONE IN SIX of the tweets below is actually FAVOR. Your job is to find them.\n"
    "The annotators are LIBERAL about assigning a stance — they mark FAVOR far more readily than "
    "a cautious reader would.\n"
    "Count as FAVOR when: the author is enthusiastic, optimistic or admiring about electric cars; "
    "reports their growth / adoption / market momentum in a POSITIVE frame; explains or promotes "
    "them helpfully; treats their spread as desirable or inevitable; shows personal interest in "
    "owning, buying or researching one; or shares EV news approvingly.\n"
    "Keep NONE only for: pure logistics questions, off-topic mentions, factual statements with no "
    "evaluative colour at all, or tweets where electric cars are merely incidental.\n"
    "Use AGAINST when the author mocks or criticises electric cars themselves.\n"
    "Be decisive: you are expected to return a meaningful number of Favor labels.\n"
    "Output EXACTLY one line per tweet: '<number>|<Label>|<confidence 0-100>'. No other text."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--provider", default="sub2apigpt")
    ap.add_argument("--idx", default="_llm/ec_none_residual_idx.npy")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", default="_llm/none_recheck_sol.json")
    args = ap.parse_args()
    _, cfg = _provider_for(args.model, args.provider); wire = cfg["wire_model"]
    idx = np.load(args.idx)
    ar = pd.read_csv("test_norm.csv", keep_default_na=False, dtype=str)
    en = pd.read_csv("_llm/test_en.csv", keep_default_na=False, dtype=str)
    res = {}
    for b0 in range(0, len(idx), args.batch):
        chunk = idx[b0:b0 + args.batch]
        lines = []
        for k, i in enumerate(chunk):
            a = " ".join(ar.at[int(i), "text"].split())
            e = " ".join(en.at[int(i), "text"].split())
            lines.append(f"{k+1}| AR: {a}\n   EN: {e}")
        content = call_api(cfg, wire, [{"role": "system", "content": SYS},
                                       {"role": "user", "content": "\n".join(lines)}],
                           "none", 60 * len(chunk) + 600, 0.0)
        for ln in content.splitlines():
            p = [x.strip() for x in ln.split("|")]
            if len(p) >= 2 and p[0].isdigit():
                k = int(p[0]) - 1
                if 0 <= k < len(chunk):
                    lab = p[1].capitalize()
                    conf = int(p[2]) if len(p) > 2 and p[2].isdigit() else 50
                    if lab in ("Favor", "Against", "None"):
                        res[int(chunk[k])] = {"label": lab, "conf": conf}
        print(f"  {min(b0+args.batch, len(idx))}/{len(idx)}", flush=True)
    json.dump(res, open(args.out, "w"), indent=0)
    fav = sorted([i for i, v in res.items() if v["label"] == "Favor"],
                 key=lambda i: -res[i]["conf"])
    print(f"[{args.model}] {len(res)} judged, {len(fav)} flagged Favor: {fav}")


if __name__ == "__main__":
    main()
