"""prior_recheck.py — prior-calibrated re-classification of a targeted row subset.

WHY this is a new instrument, not a re-run: all 9 existing views used the SAME neutral
grounded prompt, so they share a systematic Against-bias on Ecars and 126/132 of the
Against rows have ZERO views calling Favor. Here the measured gold base rate (from the
constant-probe aggregates) is stated IN the prompt, and the model is asked the narrower
question "is this actually Favor?" on rows already labelled Against. Different prompt,
different question, different prior => decorrelated from the pool that is blind.

Aggregate priors only — no per-row gold is known or used. Gate-1 clean.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_predict import _provider_for, call_api

SYS = (
    "You are an expert annotator for Arabic stance detection on the target: electric cars "
    "(السيارات الكهربائية).\n"
    "CALIBRATION (measured on this exact test set): 43.7% of these tweets are Favor, 35.2% are "
    "Against, 21.1% are None. A previous pass labelled EVERY tweet below as 'Against', and that "
    "pass is known to OVER-predict Against and UNDER-predict Favor.\n"
    "Your job: re-examine each tweet and decide whether it is genuinely Against, or whether it is "
    "actually FAVOR or NONE.\n"
    "Count as FAVOR when: the author supports/admires/wants electric cars; is optimistic about "
    "their future; defends them against criticism; treats them as a desirable achievement or "
    "aspiration; OR the complaint is aimed at a GOVERNMENT, COMPANY or PERSON for failing to "
    "deliver electric cars (the cars themselves being the thing wanted).\n"
    "Keep AGAINST only when the author criticises electric cars THEMSELVES (price, range, battery, "
    "charging, practicality, environmental hypocrisy, calling them a failure/bubble/illusion).\n"
    "Use NONE only for pure news/questions with no evaluation.\n"
    "Output EXACTLY one line per tweet: '<number>|<Label>|<confidence 0-100>'. No other text."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--provider", default="sub2apigpt")
    ap.add_argument("--idx", default="_llm/ec_against_idx.npy")
    ap.add_argument("--batch", type=int, default=22)
    ap.add_argument("--out", default="_llm/prior_recheck.json")
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
    flips = {i: v for i, v in res.items() if v["label"] == "Favor"}
    print(f"[RESULT] rechecked {len(res)}/{len(idx)} | says FAVOR on {len(flips)} rows")
    for i, v in sorted(flips.items(), key=lambda kv: -kv[1]["conf"]):
        print(f"   {i} conf={v['conf']}")


if __name__ == "__main__":
    main()
